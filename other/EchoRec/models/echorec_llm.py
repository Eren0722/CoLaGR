"""LLM adapter used by the SI stage."""

import math
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
import inspect


class SimplifiedOutput:
    """轻量级的输出容器，用于与Transformers接口保持一致。"""

    def __init__(self, hidden_state):
        # huggingface 输出是一个 tuple，我们只需要最后一层
        self.hidden_states = (hidden_state,)


class EchoRecLLM(nn.Module):
    """
    以大语言模型为核心的推荐表示学习模块。

    主要职责：
    - 加载指定的预训练LLM（如 Llama 系列）与对应分词器
    - 注册与扩展项目需要的特殊token（[UserRep]/[HistoryEmb]/[UserOut]/[ItemOut]）
    - 冻结大部分LLM参数，仅训练轻量级头部或部分嵌入（可选）
    - 提供若干工具函数：
      * replace_out_token_all / replace_out_token_all_infer：将文本中的占位符替换为向量
      * get_embeddings：定位并提取指定token位置
      * rec_loss / info_nce / uniformity：训练损失计算
    - 前向流程（train_mode0）：
      1) 用历史交互嵌入替换 [HistoryEmb]，从 LLM hidden states 抽取 [UserOut] 表示
      2) 用候选物品嵌入替换 [HistoryEmb]，从 LLM hidden states 抽取 [ItemOut] 表示
      3) 通过预测头映射到统一维度，计算推荐与对齐损失
    """
    def __init__(
        self,
        device,
        llm_model="",
        max_output_txt_len=256,
        args=None
    ):
        """
        初始化 InjectionLLM 模块

        Args:
            device (str): 计算设备，如 'cuda:0' 或 'cpu'
            llm_model (str): LLM模型名称，支持 'llama' 和 'llama-3b'
            max_output_txt_len (int): 最大输出文本长度，默认256
            args: 全局配置对象，包含训练策略和硬件配置

        初始化流程：
        1. 加载指定的预训练LLM模型和分词器
        2. 扩展词汇表以支持特殊token
        3. 配置参数冻结策略
        4. 初始化预测头和对齐模块
        """
        super().__init__()
        self.device = device
        self.bce_criterion = torch.nn.BCEWithLogitsLoss()  # 二元交叉熵损失（保留用于扩展）
        self.args = args
        self.is_rank0 = getattr(args, 'local_rank', 0) in (-1, 0) if args is not None else True
        self.hf_local_only = bool(getattr(args, 'hf_local_only', False)) if args is not None else False
        self.llm_path = (getattr(args, 'llm_path', None) if args is not None else None)
        if isinstance(self.llm_path, str):
            self.llm_path = self.llm_path.strip() or None
        raw_endpoint = getattr(args, 'hf_endpoint', '') if args is not None else ''
        raw_mirror = getattr(args, 'hf_mirror_endpoint', 'https://hf-mirror.com') if args is not None else 'https://hf-mirror.com'
        token_from_args = getattr(args, 'hf_access_token', None) if args is not None else None
        env_token = os.environ.get('HF_TOKEN') or os.environ.get('HUGGINGFACEHUB_API_TOKEN')
        self.hf_token = (token_from_args or env_token or '').strip() or None
        # 记录梯度检查点状态，方便验证/推理临时关闭提速
        self._ckpt_enabled = False
        self.hf_endpoint = self._normalize_endpoint(raw_endpoint)
        self.hf_mirror_endpoint = self._normalize_endpoint(raw_mirror)
        self.hf_use_mirror = bool(getattr(args, 'hf_use_mirror', False)) if args is not None else False
        if self.hf_use_mirror and not self.hf_endpoint:
            self.hf_endpoint = self.hf_mirror_endpoint
        self.hf_cache_dir = (getattr(args, 'hf_cache_dir', '') if args is not None else '').strip()
        self._configure_hf_runtime()
        self.chunk_split_threshold = getattr(args, 'candidate_chunk_threshold', 36) if args is not None else 36
        # 恢复 8×50 分块：默认 chunk=50，最小20
        self.candidate_chunk_size = max(20, getattr(args, 'candidate_chunk_size', 50) if args is not None else 50)
        self.min_candidate_chunk_size = max(10, getattr(args, 'min_candidate_chunk_size', 20) if args is not None else 20)
        # 禁用用户序列分块（阈值设为50，batch=20不会触发），避免破坏批处理效率
        self.sequence_chunk_threshold = getattr(args, 'sequence_chunk_threshold', 50) if args is not None else 50
        self.sequence_chunk_size = max(2, getattr(args, 'sequence_chunk_size', 5) if args is not None else 5)
        self.min_sequence_chunk_size = max(1, getattr(args, 'min_sequence_chunk_size', 3) if args is not None else 3)



        # 步骤1: 选择并配置LLM模型
        if llm_model == 'llama':
            model_id = "meta-llama/Meta-Llama-3-8B-Instruct"
        elif llm_model == 'llama-3b':
            model_id = "meta-llama/Llama-3.2-3B-Instruct"
        else:
            raise Exception(f'{llm_model} is not supported')
        if self.llm_path:
            self.llm_path = self._resolve_local_llm_path(self.llm_path)
            self._validate_local_llm_files(self.llm_path)
            model_id = self.llm_path
            self.hf_local_only = True
            self._hf_log(f"📂 本地LLM路径: {self.llm_path}")
        print()
        print("=========")

        # 步骤2: 加载LLM模型（对齐Start_old：纯FP16 + use_cache=False）
        model_kwargs = {
            'device_map': self.device,
            'torch_dtype': torch.float16,
            'trust_remote_code': True,
            'local_files_only': self.hf_local_only,
        }
        if self.hf_cache_dir:
            model_kwargs['cache_dir'] = self.hf_cache_dir
        if self.hf_token:
            model_kwargs['token'] = self.hf_token

        # ✅ 对齐Start_old: 禁用KV缓存，训练场景节省显存
        model_kwargs['use_cache'] = False
        self.llm_model = self._hf_download(
            lambda: AutoModelForCausalLM.from_pretrained(model_id, **model_kwargs),
            "LLM checkpoint"
        )

        # 🛡️ GPU保护：设置CUDA优化选项
        if torch.cuda.is_available():
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            torch.backends.cudnn.benchmark = True

        # 步骤3: 加载对应的分词器（强制离线模式避免网络问题）
        tokenizer_kwargs = {
            'use_fast': False,
            'trust_remote_code': True,
            'local_files_only': self.hf_local_only,
        }
        if self.hf_cache_dir:
            tokenizer_kwargs['cache_dir'] = self.hf_cache_dir
        if self.hf_token:
            tokenizer_kwargs['token'] = self.hf_token
        self.llm_tokenizer = self._hf_download(
            lambda: AutoTokenizer.from_pretrained(model_id, **tokenizer_kwargs),
            "LLM tokenizer"
        )

        # 🔧 运行时保护：禁用生成缓存 + 启用梯度检查点以节省显存
        try:
            self.llm_model.config.use_cache = False
            if hasattr(self.llm_model.config, 'max_length'):
                self.llm_model.config.max_length = min(getattr(self.llm_model.config, 'max_length', 2048), 1024)
        except Exception:
            pass
        if hasattr(self.llm_model, "gradient_checkpointing_enable"):
            try:
                self.llm_model.gradient_checkpointing_enable()
                self._hf_log("✅ LLM gradient checkpointing 已启用")
            except Exception as e:
                self._hf_log(f"⚠️ 无法启用 gradient checkpointing: {e}")

        # 控制Prompt长度，优先使用配置提供的上限以降低显存占用
        default_prompt_len = 896 if llm_model == 'llama-3b' else 1024
        requested_max_len = getattr(args, 'llm_max_length', default_prompt_len) if args is not None else default_prompt_len
        config_max_len = getattr(self.llm_model.config, 'max_position_embeddings', requested_max_len)
        self.max_prompt_length = max(256, min(requested_max_len, config_max_len))

        # 步骤4: 扩展分词器的特殊token
        # 添加标准的特殊token
        self.llm_tokenizer.add_special_tokens({'pad_token': '[PAD]'})    # 填充token
        self.llm_tokenizer.add_special_tokens({'bos_token': '</s>'})     # 序列开始token
        self.llm_tokenizer.add_special_tokens({'eos_token': '</s>'})     # 序列结束token
        self.llm_tokenizer.add_special_tokens({'unk_token': '</s>'})     # 未知token

        # 添加项目专用的特殊token
        # [UserRep]: 用户表示token（保留用于扩展）
        # [HistoryEmb]: 历史物品嵌入占位符，会被实际的物品嵌入替换
        # [UserOut]: 用户输出位置标记，LLM在此位置生成用户表示
        # [ItemOut]: 物品输出位置标记，LLM在此位置生成物品表示
        self.llm_tokenizer.add_special_tokens({
            'additional_special_tokens': ['[UserRep]', '[HistoryEmb]', '[UserOut]', '[ItemOut]']
        })
        self.llm_tokenizer.add_special_tokens({'cls_token': "[CLS]"})     # 分类token

        # 步骤5: 调整模型以适应扩展的词汇表
        self.llm_model.resize_token_embeddings(len(self.llm_tokenizer))
        # 🔧 移除量化训练准备，使用纯FP16训练避免CUDA内存访问错误
        # self.llm_model = prepare_model_for_kbit_training(self.llm_model)

        # 步骤6: 配置参数冻结策略
        for name, param in self.llm_model.named_parameters():
            if args.token:
                # 策略A: 仅训练词嵌入层（token embedding）
                if 'token' in name:
                    param.requires_grad = True
                else:
                    param.requires_grad = False
            else:
                # 策略B: 冻结所有LLM参数，使用可学习的CLS向量
                param.requires_grad = False

        # 步骤7: 初始化可学习的占位符向量（当不训练词嵌入时）
        if not args.token:
            if args.nn_parameter:
                # 神经网络处理器模式：使用Parameter直接存储
                self.CLS = nn.Parameter(torch.normal(0, 1, size=(1, self.llm_model.config.hidden_size))).to(device)
                self.CLS_item = nn.Parameter(torch.normal(0, 1, size=(1, self.llm_model.config.hidden_size))).to(device)
            else:
                # 标准模式：使用Embedding层存储
                self.CLS = nn.Embedding(1, self.llm_model.config.hidden_size).to(device)
                # 使用LLM词嵌入的统计特性初始化CLS向量
                nn.init.normal_(
                    self.CLS.weight,
                    mean=self.llm_model.model.embed_tokens.weight.mean(),
                    std=self.llm_model.model.embed_tokens.weight.std()
                )

                self.CLS_item = nn.Embedding(1, self.llm_model.config.hidden_size).to(device)
                nn.init.normal_(
                    self.CLS_item.weight,
                    mean=self.llm_model.model.embed_tokens.weight.mean(),
                    std=self.llm_model.model.embed_tokens.weight.std()
                )

        # 步骤8: 初始化预测头网络
        # 用户表示预测头：将LLM隐藏状态映射到统一的用户表示空间
        # 架构：LLM_hidden_size -> 2048 -> LayerNorm -> LeakyReLU -> 128
        self.pred_user = nn.Sequential(
                nn.Linear(self.llm_model.config.hidden_size, 2048),  # 第一层：维度扩展
                nn.LayerNorm(2048),                                  # 层归一化
                nn.LeakyReLU(),                                      # 激活函数
                nn.Linear(2048, 128)                                 # 第二层：映射到最终维度
            )
        # 使用Xavier初始化提高训练稳定性
        try:
            nn.init.xavier_normal_(self.pred_user[0].weight)
            nn.init.xavier_normal_(self.pred_user[3].weight)
        except Exception as e:
            print(f"⚠️ pred_user权重初始化跳过: {e}")

        # 物品表示预测头：将LLM隐藏状态映射到统一的物品表示空间
        # 架构与用户预测头相同，但参数独立
        self.pred_item = nn.Sequential(
                nn.Linear(self.llm_model.config.hidden_size, 2048),
                nn.LayerNorm(2048),
                nn.LeakyReLU(),
                nn.Linear(2048, 128)
            )
        try:
            nn.init.xavier_normal_(self.pred_item[0].weight)
            nn.init.xavier_normal_(self.pred_item[3].weight)
        except Exception as e:
            print(f"⚠️ pred_item权重初始化跳过: {e}")

        # CF用户表示对齐头：将CF-SRec的用户表示(64维)映射到统一空间(128维)
        # 用于知识蒸馏，使LLM学习CF模型的用户表示
        self.pred_user_CF2 = nn.Sequential(
                nn.Linear(64, 128),      # CF维度(64) -> 统一维度(128)
                nn.LayerNorm(128),       # 层归一化
                nn.GELU(),               # GELU激活函数
                nn.Linear(128, 128)      # 进一步变换
            )
        try:
            nn.init.xavier_normal_(self.pred_user_CF2[0].weight)
            nn.init.xavier_normal_(self.pred_user_CF2[3].weight)
        except Exception as e:
            print(f"⚠️ pred_user_CF2权重初始化跳过: {e}")

        # SA阶段对比学习温度（仅 semantic alignment 使用）
        self.contrastive_temperature = getattr(args, 'contrastive_temperature', 0.07)

        # 训练状态跟踪
        self.iteration_count = 0  # 迭代计数器，用于动态权重调整
        # 模块二彻底下线，仅保留占位符属性避免旧调用报错
        self.enable_heterogeneous_fusion = False
        self.llm_trainable = True

        # 步骤9: 初始化损失函数和配置参数
        self.mse = nn.MSELoss()  # 均方误差损失，用于表示对齐
        self.max_output_txt_len = max_output_txt_len  # 最大输出文本长度

        # 双向蒸馏配置已完成，移除未使用的参数
        self.fixed_item_embeddings = None
        self.fixed_item_embeddings_dim = None
        self._sa_index_warned = False
        self._sa_teacher_warned = False
        self.fixed_item_embeddings_path = None
        self._load_fixed_item_embeddings()

    def _normalize_endpoint(self, url: str) -> str:
        if not url:
            return ""
        return url.rstrip("/")

    def _hf_log(self, message: str):
        if self.is_rank0:
            print(message)

    def _resolve_local_llm_path(self, path: str) -> str:
        expanded = os.path.expandvars(os.path.expanduser(path))
        return os.path.abspath(expanded)

    def _validate_local_llm_files(self, base_dir: str):
        if not os.path.isdir(base_dir):
            raise FileNotFoundError(f"本地LLM目录不存在: {base_dir}")
        required_files = ["config.json", "tokenizer_config.json", "tokenizer.json"]
        missing = [fname for fname in required_files if not os.path.isfile(os.path.join(base_dir, fname))]
        if missing:
            raise FileNotFoundError(f"本地LLM缺少必要文件: {', '.join(missing)}")
        has_shard = any(
            fname.endswith(".safetensors") and fname.startswith("model-")
            for fname in os.listdir(base_dir)
        )
        if not has_shard:
            raise FileNotFoundError("本地LLM目录中未找到 *.safetensors 权重分片")

    def _configure_hf_runtime(self):
        if self.hf_cache_dir:
            os.makedirs(self.hf_cache_dir, exist_ok=True)
            os.environ.setdefault("HF_HOME", self.hf_cache_dir)
            transformers_cache = os.path.join(self.hf_cache_dir, "transformers")
            os.makedirs(transformers_cache, exist_ok=True)
            os.environ.setdefault("TRANSFORMERS_CACHE", transformers_cache)

        if self.hf_local_only:
            os.environ["HF_HUB_OFFLINE"] = "1"
            os.environ["TRANSFORMERS_OFFLINE"] = "1"
        else:
            os.environ.pop("HF_HUB_OFFLINE", None)
            os.environ.pop("TRANSFORMERS_OFFLINE", None)

        endpoint = self.hf_endpoint
        if not endpoint and self.hf_use_mirror:
            endpoint = self.hf_mirror_endpoint
        if endpoint:
            os.environ["HF_ENDPOINT"] = endpoint

    def _hf_download(self, factory, desc: str, allow_mirror: bool = True):
        if self.hf_local_only:
            return factory()

        endpoints = []

        def add_endpoint(url: str):
            normalized = self._normalize_endpoint(url)
            if normalized and normalized not in endpoints:
                endpoints.append(normalized)

        add_endpoint(self.hf_endpoint)
        add_endpoint(os.environ.get("HF_ENDPOINT"))
        add_endpoint("https://huggingface.co")
        if allow_mirror:
            add_endpoint(self.hf_mirror_endpoint)

        last_exc = None
        for endpoint in endpoints or ["https://huggingface.co"]:
            os.environ["HF_ENDPOINT"] = endpoint
            self._hf_log(f"🔄 {desc}: 尝试 {endpoint}")
            try:
                return factory()
            except OSError as exc:
                last_exc = exc
                self._hf_log(f"⚠️ {desc}: {endpoint} 失败，{exc}")
        raise last_exc

    def set_trainable(self, flag: bool):
        """
        设置LLM侧组件的可训练状态
        
        关键修复：仅训练预测头和映射层，LLM backbone始终冻结
        - sequence_injection阶段(flag=True)：训练pred_user/pred_item/pred_user_CF2/CLS/CLS_item
        - semantic alignment阶段(flag=False)：冻结LLM侧所有组件
        
        🔧 DDP安全：flag=True时先冻结所有子模块再精确解冻SI所需组件，
        确保find_unused_parameters=True下无undefined gradient。
        """
        self.llm_trainable = flag
        
        # LLM backbone始终冻结（Llama-3.2-3B太大，不能全部解冻）
        for param in self.llm_model.parameters():
            param.requires_grad = False
        
        if flag:
            # 仅解冻SI阶段train_mode0使用的轻量级组件
            # 注意：不要"先冻结全部再解冻"，因为set_llm_trainable已经解冻了
            # item_emb_proj等外层组件，这里再冻结会把它们意外冻住
            for param in self.pred_user.parameters():
                param.requires_grad = True
            for param in self.pred_item.parameters():
                param.requires_grad = True
            for param in self.pred_user_CF2.parameters():
                param.requires_grad = True
            
            # 如果使用可学习的CLS向量，也解冻
            if hasattr(self, 'CLS'):
                if isinstance(self.CLS, nn.Parameter):
                    self.CLS.requires_grad = True
                else:
                    for param in self.CLS.parameters():
                        param.requires_grad = True
            if hasattr(self, 'CLS_item'):
                if isinstance(self.CLS_item, nn.Parameter):
                    self.CLS_item.requires_grad = True
                else:
                    for param in self.CLS_item.parameters():
                        param.requires_grad = True
        else:
            # 反向蒸馏（semantic alignment）：冻结LLM侧所有组件
            for param in self.pred_user.parameters():
                param.requires_grad = False
            for param in self.pred_item.parameters():
                param.requires_grad = False
            if hasattr(self, 'CLS'):
                if isinstance(self.CLS, nn.Parameter):
                    self.CLS.requires_grad = False
                else:
                    for param in self.CLS.parameters():
                        param.requires_grad = False
            if hasattr(self, 'CLS_item'):
                if isinstance(self.CLS_item, nn.Parameter):
                    self.CLS_item.requires_grad = False
                else:
                    for param in self.CLS_item.parameters():
                        param.requires_grad = False

    def _set_gradient_checkpointing(self, enabled: bool):
        """设置梯度检查点开关：开启省显存，关闭可加速验证/推理。"""
        if not hasattr(self.llm_model, "gradient_checkpointing_enable"):
            self._ckpt_enabled = False
            return
        try:
            try:
                self.llm_model.config.use_cache = False
            except Exception:
                pass
            if enabled:
                sig = inspect.signature(self.llm_model.gradient_checkpointing_enable)
                if "use_reentrant" in sig.parameters:
                    self.llm_model.gradient_checkpointing_enable(use_reentrant=False)
                else:
                    self.llm_model.gradient_checkpointing_enable()
                self._ckpt_enabled = True
            else:
                if hasattr(self.llm_model, "gradient_checkpointing_disable"):
                    self.llm_model.gradient_checkpointing_disable()
                self._ckpt_enabled = False
        except Exception:
            self._ckpt_enabled = False

    def set_loss_weights(self, teacher_rec=None, student_rec=None, forward=None, backward=None):
        if teacher_rec is not None:
            self.teacher_rec_weight = teacher_rec
        if student_rec is not None:
            self.student_rec_weight = student_rec
        if forward is not None:
            self.forward_weight = forward
        if backward is not None:
            self.backward_weight = backward

    def user_item_contrastive_loss(self, user_repr, item_repr, negative_items):
        """
        用户-物品对比学习损失

        Args:
            user_repr: 用户表示 [batch_size, dim]
            item_repr: 正样本物品表示 [batch_size, dim]
            negative_items: 负样本物品表示 [batch_size, num_negatives, dim]

        Returns:
            loss: 用户-物品对比损失
        """
        # 动态适配用户表示维度
        actual_user_dim = user_repr.size(-1)
        if actual_user_dim != self.user_dim:
            if not hasattr(self, 'user_dim_adapter') or self.user_dim_adapter is None:
                self.user_dim_adapter = nn.Linear(actual_user_dim, self.user_dim).to(user_repr.device)
                print(f"🔧 [模块3] 动态创建用户维度适配器: {actual_user_dim} -> {self.user_dim}")
            user_repr = self.user_dim_adapter(user_repr)

        # 动态适配物品表示维度
        actual_item_dim = item_repr.size(-1)
        if actual_item_dim != self.item_dim:
            if not hasattr(self, 'item_dim_adapter') or self.item_dim_adapter is None:
                self.item_dim_adapter = nn.Linear(actual_item_dim, self.item_dim).to(item_repr.device)
                print(f"🔧 [模块3] 动态创建物品维度适配器: {actual_item_dim} -> {self.item_dim}")
            item_repr = self.item_dim_adapter(item_repr)

        # 动态适配负样本维度（负样本可能与正样本维度不同）
        actual_neg_dim = negative_items.size(-1)
        original_neg_shape = negative_items.shape  # 保存原始形状
        if actual_neg_dim != self.item_dim:
            if not hasattr(self, 'item_dim_adapter') or self.item_dim_adapter is None:
                self.item_dim_adapter = nn.Linear(actual_neg_dim, self.item_dim).to(negative_items.device)
                print(f"🔧 [模块3] 动态创建物品维度适配器(负样本): {actual_neg_dim} -> {self.item_dim}")
            # 适配负样本：先flatten，适配，再reshape回原来的batch和num_neg结构
            batch_size, num_negatives = original_neg_shape[0], original_neg_shape[1]
            negative_items = self.item_dim_adapter(negative_items.view(-1, actual_neg_dim))
            negative_items = negative_items.view(batch_size, num_negatives, self.item_dim)

        # 投影到对比学习空间
        user_proj = self.user_item_projector(user_repr)
        item_proj = self.item_projector(item_repr)
        neg_proj = self.item_projector(negative_items.view(-1, negative_items.size(-1)))
        neg_proj = neg_proj.view(negative_items.size(0), negative_items.size(1), -1)

        # 调试信息：检查维度匹配（只显示一次）
        if user_proj.size(0) != item_proj.size(0):
            if not hasattr(self, '_dim_mismatch_warned'):
                print(f"⚠️ [模块3] 维度不匹配: user_proj {user_proj.shape}, item_proj {item_proj.shape}")
                print(f"   自动修复：将使用较小的batch size进行对比学习")
                self._dim_mismatch_warned = True
            # 确保batch size一致，取较小的batch size
            min_batch_size = min(user_proj.size(0), item_proj.size(0))
            user_proj = user_proj[:min_batch_size]
            item_proj = item_proj[:min_batch_size]
            neg_proj = neg_proj[:min_batch_size]

        # 计算对比损失
        loss = self.compute_contrastive_loss(
            user_proj, item_proj, neg_proj, self.contrastive_temperature
        )

        return loss

    def sequence_contrastive_loss(self, sequence_repr, sequence_mask=None):
        """
        序列内对比学习损失

        Args:
            sequence_repr: 序列表示 [batch_size, seq_len, dim]
            sequence_mask: 序列掩码 [batch_size, seq_len]

        Returns:
            loss: 序列对比损失
        """
        batch_size, seq_len, dim = sequence_repr.shape

        if sequence_mask is None:
            sequence_mask = torch.ones(batch_size, seq_len, device=sequence_repr.device)

        # 投影到对比学习空间
        seq_proj = self.sequence_projector(sequence_repr.view(-1, dim))
        seq_proj = seq_proj.view(batch_size, seq_len, -1)

        total_loss = 0
        valid_pairs = 0

        # 在窗口内进行对比学习
        for i in range(seq_len - 1):
            for j in range(i + 1, min(i + self.sequence_contrastive_window + 1, seq_len)):
                # 检查掩码
                mask_i = sequence_mask[:, i]
                mask_j = sequence_mask[:, j]
                valid_mask = mask_i * mask_j  # 两个位置都有效

                if valid_mask.sum() == 0:
                    continue

                # 正样本：窗口内的相邻物品
                anchor = seq_proj[:, i][valid_mask.bool()]  # [valid_batch, dim]
                positive = seq_proj[:, j][valid_mask.bool()]  # [valid_batch, dim]

                # 负样本：随机采样其他序列中的物品
                neg_indices = torch.randint(0, batch_size * seq_len,
                                          (anchor.size(0), self.negative_sampling_ratio))
                negative = seq_proj.view(-1, seq_proj.size(-1))[neg_indices]  # [valid_batch, num_neg, dim]

                # 计算对比损失
                if anchor.size(0) > 0:
                    loss = self.compute_contrastive_loss(
                        anchor, positive, negative, self.contrastive_temperature
                    )
                    total_loss += loss
                    valid_pairs += 1

        return total_loss / max(valid_pairs, 1)

    def cross_modal_contrastive_loss(self, llm_repr, cf_repr):
        """
        跨模态对比学习损失

        Args:
            llm_repr: LLM表示 [batch_size, llm_dim]
            cf_repr: CF表示 [batch_size, cf_dim]

        Returns:
            loss: 跨模态对比损失
        """
        # 动态适配LLM表示维度
        actual_llm_dim = llm_repr.size(-1)
        if actual_llm_dim != 128:  # LLM投影器期望128维
            if not hasattr(self, 'llm_modal_adapter') or self.llm_modal_adapter is None:
                self.llm_modal_adapter = nn.Linear(actual_llm_dim, 128).to(llm_repr.device)
                print(f"🔧 [模块3] 动态创建LLM模态适配器: {actual_llm_dim} -> 128")
            llm_repr = self.llm_modal_adapter(llm_repr)

        # 动态适配CF表示维度
        actual_cf_dim = cf_repr.size(-1)
        if actual_cf_dim != 64:  # CF投影器期望64维
            if not hasattr(self, 'cf_modal_adapter') or self.cf_modal_adapter is None:
                self.cf_modal_adapter = nn.Linear(actual_cf_dim, 64).to(cf_repr.device)
                print(f"🔧 [模块3] 动态创建CF模态适配器: {actual_cf_dim} -> 64")
            cf_repr = self.cf_modal_adapter(cf_repr)

        # 投影到统一的对比学习空间
        llm_proj = self.llm_modal_projector(llm_repr)
        cf_proj = self.cf_modal_projector(cf_repr)

        # 正样本：同一样本的LLM和CF表示
        # 负样本：不同样本的表示
        batch_size = llm_proj.size(0)

        # 构造负样本：随机打乱CF表示
        neg_indices = torch.randperm(batch_size, device=cf_proj.device)
        while torch.equal(neg_indices, torch.arange(batch_size, device=cf_proj.device)):
            neg_indices = torch.randperm(batch_size, device=cf_proj.device)

        # 创建多个负样本
        negative_cf = []
        for _ in range(self.negative_sampling_ratio):
            neg_idx = torch.randperm(batch_size, device=cf_proj.device)
            negative_cf.append(cf_proj[neg_idx])
        negative_cf = torch.stack(negative_cf, dim=1)  # [batch_size, num_neg, dim]

        # 计算对比损失
        loss = self.compute_contrastive_loss(
            llm_proj, cf_proj, negative_cf, self.contrastive_temperature
        )

        return loss

    def multi_level_contrastive_loss(self, user_repr, item_repr, negative_items,
                                   sequence_repr=None, llm_repr=None, cf_repr=None):
        """
        多层次对比学习损失

        Args:
            user_repr: 用户表示
            item_repr: 物品表示
            negative_items: 负样本物品
            sequence_repr: 序列表示（可选）
            llm_repr: LLM表示（可选）
            cf_repr: CF表示（可选）

        Returns:
            total_loss: 总对比损失
            loss_info: 损失详情
        """
        total_loss = 0
        loss_info = {}

        # 安全的item()函数
        def safe_item(x):
            return x.item() if isinstance(x, torch.Tensor) else x

        # 1. 用户-物品对比
        if self.contrastive_strategy in ["user_item", "multi_level"]:
            ui_loss = self.user_item_contrastive_loss(user_repr, item_repr, negative_items)
            total_loss += ui_loss
            loss_info["user_item_loss"] = safe_item(ui_loss)

        # 2. 序列对比
        if self.contrastive_strategy in ["sequence", "multi_level"] and sequence_repr is not None:
            seq_loss = self.sequence_contrastive_loss(sequence_repr)
            total_loss += seq_loss
            loss_info["sequence_loss"] = safe_item(seq_loss)

        # 3. 跨模态对比
        if (self.contrastive_strategy in ["cross_modal", "multi_level"] and
            llm_repr is not None and cf_repr is not None):
            cm_loss = self.cross_modal_contrastive_loss(llm_repr, cf_repr)
            total_loss += self.cross_modal_weight * cm_loss
            loss_info["cross_modal_loss"] = safe_item(cm_loss)

        loss_info["total_contrastive_loss"] = safe_item(total_loss)

        return total_loss, loss_info

    def _sample_negative_items(self, batch_size, item_pool):
        """
        为对比学习采样负样本物品

        Args:
            batch_size: 批次大小
            item_pool: 物品池 [batch_size, dim] 或 [num_items, dim]

        Returns:
            negative_items: 负样本物品 [batch_size, num_negatives, dim]
        """
        if item_pool.dim() == 2 and item_pool.size(0) == batch_size:
            # 如果物品池大小等于批次大小，使用随机打乱策略
            negative_items = []
            for _ in range(self.negative_sampling_ratio):
                # 随机打乱物品顺序作为负样本
                neg_indices = torch.randperm(batch_size, device=item_pool.device)
                negative_items.append(item_pool[neg_indices])
            negative_items = torch.stack(negative_items, dim=1)  # [batch_size, num_neg, dim]
        else:
            # 如果有更大的物品池，随机采样
            num_items = item_pool.size(0)
            negative_items = []
            for i in range(batch_size):
                neg_indices = torch.randint(0, num_items, (self.negative_sampling_ratio,), device=item_pool.device)
                # 确保不采样到正样本（简化处理，假设正样本是第i个）
                if num_items > batch_size:
                    mask = neg_indices != i
                    while not mask.all():
                        new_indices = torch.randint(0, num_items, (self.negative_sampling_ratio,), device=item_pool.device)
                        neg_indices = torch.where(mask, neg_indices, new_indices)
                        mask = neg_indices != i
                negative_items.append(item_pool[neg_indices])
            negative_items = torch.stack(negative_items, dim=0)  # [batch_size, num_neg, dim]

        return negative_items

    def info_nce_loss_batch(self, anchor, log_emb, temperature=0.07):
        """
        批量InfoNCE对比损失计算

        InfoNCE是一种对比学习损失，通过最大化正样本相似度、最小化负样本相似度来学习表示。
        在这里用于对齐LLM生成的用户表示与CF-SRec的用户表示。

        Args:
            anchor (torch.Tensor): 锚点表示（LLM用户表示），形状 [batch_size, dim]
            log_emb (torch.Tensor): 目标表示（CF用户表示），形状 [batch_size, dim]
            temperature (float): 温度参数，控制分布的尖锐程度，默认0.07

        Returns:
            torch.Tensor: InfoNCE损失值

        计算原理：
        1. 对表示进行L2归一化
        2. 计算相似度矩阵（内积/温度）
        3. 对角线元素为正样本，其余为负样本
        4. 使用交叉熵损失优化
        """
        batch_size = anchor.shape[0]

        # L2归一化：将向量归一化到单位球面上
        anchor = F.normalize(anchor, p=2, dim=1)
        log_emb = F.normalize(log_emb, p=2, dim=1)

        # 计算相似度矩阵：anchor与log_emb之间的内积相似度
        similarity_matrix = torch.matmul(anchor, log_emb.T) / temperature

        # 创建对角线掩码：对角线位置为正样本，其余为负样本
        mask = torch.eye(batch_size, device=anchor.device).bool()

        # 提取正样本相似度（对角线元素）
        pos_sim = similarity_matrix[mask].view(batch_size, 1)
        # 提取负样本相似度（非对角线元素）
        neg_sim = similarity_matrix[~mask].view(batch_size, -1)

        # 拼接正负样本相似度：正样本在第一列
        logits = torch.cat([pos_sim, neg_sim], dim=1)

        # 标签全为0，表示正样本在第一个位置
        labels = torch.zeros(batch_size, dtype=torch.long, device=anchor.device)

        # 计算交叉熵损失
        loss = F.cross_entropy(logits, labels)

        return loss

    def rec_loss(self, anchor, items):
        """
        推荐损失：将正样本置于第一位，使用交叉熵进行排序学习

        ✅ 完全对齐Start_old: 原始内积logits + cross_entropy，无缩放/clamp/fp32转换
        """
        # 🔍 NaN 诊断：检查输入
        if torch.isnan(anchor).any() or torch.isinf(anchor).any():
            print(f"⚠️ rec_loss: anchor 包含 NaN/Inf! max={anchor.abs().max().item():.2f}")
        if torch.isnan(items).any() or torch.isinf(items).any():
            print(f"⚠️ rec_loss: items 包含 NaN/Inf! max={items.abs().max().item():.2f}")
        
        logits = torch.bmm(
            items.view(anchor.shape[0], -1, anchor.shape[1]),
            anchor.unsqueeze(2)
        ).squeeze(2)

        # 🔍 NaN 诊断：检查 logits
        if torch.isnan(logits).any() or torch.isinf(logits).any():
            print(f"⚠️ rec_loss: logits 包含 NaN/Inf! max={logits.abs().max().item():.2f}, anchor_max={anchor.abs().max().item():.2f}, items_max={items.abs().max().item():.2f}")

        labels = torch.zeros(logits.size(0), dtype=torch.long, device=logits.device)

        loss = F.cross_entropy(logits, labels)

        # 🔍 NaN 诊断：检查最终 loss
        if torch.isnan(loss) or torch.isinf(loss):
            print(f"❌ rec_loss 返回 NaN/Inf! logits_max={logits.abs().max().item():.2f}")

        return loss

    def uniformity(self, x, p=2):
        """
        Uniformity正则化损失：鼓励表征在空间中均匀分布

        这个正则化项防止表征塌缩到空间中的某个小区域，
        鼓励学到的表征能够充分利用整个表征空间。

        Args:
            x (torch.Tensor): 输入表征，形状 [batch_size, dim]
            p (int): 距离的p范数，默认为2（欧几里得距离）

        Returns:
            torch.Tensor: Uniformity损失值

        计算公式：
        uniformity = mean(exp(-p * ||xi - xj||_p^2))
        其中xi, xj是批次中的不同样本
        """
        return torch.pdist(x, p=p).pow(2).mul(-p).exp().mean()

    def replace_out_token_all(self, llm_tokens, inputs_embeds, token=[], embs=None):
        """
        训练场景的特殊token替换函数

        这是EchoRec的核心机制：将文本中的特殊token占位符替换为实际的向量表示。
        通过这种方式，可以将CF-SRec的物品嵌入无缝注入到LLM的输入中。

        Args:
            llm_tokens (dict): 分词后的token字典，包含 'input_ids' 等
            inputs_embeds (torch.Tensor): LLM的输入嵌入，形状 [batch_size, seq_len, hidden_size]
            token (list): 需要替换的特殊token列表，如 ['[UserOut]', '[HistoryEmb]']
            embs (dict): 替换用的嵌入字典，键为token名，值为对应的嵌入张量

        Returns:
            torch.Tensor: 替换后的输入嵌入

        替换策略：
        - [HistoryEmb]: 替换为历史物品的嵌入序列（支持多个物品）
        - [UserRep]: 替换为用户表示嵌入
        - [UserOut]/[ItemOut]: 替换为可学习的CLS向量（当不训练token时）

        工作原理：
        1. 遍历每个需要替换的token
        2. 找到该token在输入序列中的所有位置
        3. 根据token类型选择相应的替换策略
        4. 逐位置替换，保持序列长度不变
        """
        for t in token:
            # 获取当前token的ID
            token_id = self.llm_tokenizer(t, return_tensors="pt", add_special_tokens=False).input_ids.item()
            vectors = []

            # 遍历批次中的每个样本
            for inx in range(len(llm_tokens["input_ids"])):
                # 找到当前token在序列中的所有位置
                idx_tensor = (llm_tokens["input_ids"][inx] == token_id).nonzero().view(-1)
                user_vector = inputs_embeds[inx]  # 当前样本的嵌入序列

                if 'Emb' in t:
                    # 处理嵌入类token（如[HistoryEmb]）
                    # 用提供的嵌入序列替换对应位置
                    ee = embs[t][inx]  # 获取当前样本的嵌入序列
                    for idx, item_emb in zip(idx_tensor, ee):
                        # 逐个替换：保留前面部分 + 新嵌入 + 保留后面部分
                        user_vector = torch.cat((user_vector[:idx], item_emb.unsqueeze(0), user_vector[idx+1:]), dim=0)

                elif 'Rep' in t:
                    # 处理表示类token（如[UserRep]）
                    # 用单个表示向量替换对应位置
                    for idx in idx_tensor:
                        user_emb = embs[t][inx]
                        user_vector = torch.cat((user_vector[:idx], user_emb.unsqueeze(0), user_vector[idx+1:]), dim=0)

                else:
                    # 处理输出类token（如[UserOut], [ItemOut]）
                    # 当不训练token时，用可学习的CLS向量替换
                    if not self.args.token:
                        for idx in idx_tensor:
                            if 'UserOut' in t:
                                # 替换为用户CLS向量
                                if self.args.nn_parameter:
                                    user_vector = torch.cat((user_vector[:idx], self.CLS[torch.tensor([0]).to(self.device)], user_vector[idx+1:]), dim=0)
                                else:
                                    user_vector = torch.cat((user_vector[:idx], self.CLS(torch.tensor([0]).to(self.device)), user_vector[idx+1:]), dim=0)
                            elif 'ItemOut' in t:
                                # 替换为物品CLS向量
                                if self.args.nn_parameter:
                                    user_vector = torch.cat((user_vector[:idx], self.CLS_item[torch.tensor([0]).to(self.device)], user_vector[idx+1:]), dim=0)
                                else:
                                    user_vector = torch.cat((user_vector[:idx], self.CLS_item(torch.tensor([0]).to(self.device)), user_vector[idx+1:]), dim=0)

                vectors.append(user_vector.unsqueeze(0))

            # 将所有样本的替换结果拼接回批次
            inputs_embeds = torch.cat(vectors)
        return inputs_embeds
    
    def replace_out_token_all_infer(self, llm_tokens, inputs_embeds, token=[], embs=None, user_act=False, item_act=False):
        """
        推理场景的特殊token替换函数

        与 replace_out_token_all 的区别：
        - replace_out_token_all: 训练场景，支持序列化的嵌入替换
        - replace_out_token_all_infer: 推理场景，简化的嵌入替换

        主要差异：
        1. [HistoryEmb] 只替换一次（每条样本一个聚合向量）
        2. 优化了内存使用和计算效率
        3. 保留了未使用的参数以保持接口兼容性

        Args:
            llm_tokens (dict): 分词后的token字典
            inputs_embeds (torch.Tensor): LLM的输入嵌入
            token (list): 需要替换的特殊token列表
            embs (dict): 替换用的嵌入字典
            user_act (bool): 用户激活标志（保留用于扩展，当前未使用）
            item_act (bool): 物品激活标志（保留用于扩展，当前未使用）

        Returns:
            torch.Tensor: 替换后的输入嵌入
        """
        for t in token:
            # 获取当前token的ID
            token_id = self.llm_tokenizer(t, return_tensors="pt", add_special_tokens=False).input_ids.item()
            vectors = []

            # 遍历批次中的每个样本
            for inx in range(len(llm_tokens["input_ids"])):
                # 找到当前token在序列中的所有位置
                idx_tensor = (llm_tokens["input_ids"][inx] == token_id).nonzero().view(-1)
                user_vector = inputs_embeds[inx]

                if 'Emb' in t:
                    # 处理嵌入类token（推理模式的简化版本）
                    # 将单个嵌入包装成列表，保持与训练模式的接口一致
                    ee = [embs[t][inx]]
                    # 注释掉的代码：ee = embs[t][inx] （原始实现）
                    for idx, item_emb in zip(idx_tensor, ee):
                        user_vector = torch.cat((user_vector[:idx], item_emb.unsqueeze(0), user_vector[idx+1:]), dim=0)

                elif 'Rep' in t:
                    # 处理表示类token（与训练模式相同）
                    for idx in idx_tensor:
                        user_emb = embs[t][inx]
                        user_vector = torch.cat((user_vector[:idx], user_emb.unsqueeze(0), user_vector[idx+1:]), dim=0)

                else:
                    # 处理输出类token（与训练模式相同）
                    if not self.args.token:
                        for idx in idx_tensor:
                            if 'UserOut' in t:
                                # 替换为用户CLS向量
                                if self.args.nn_parameter:
                                    user_vector = torch.cat((user_vector[:idx], self.CLS[torch.tensor([0]).to(self.device)], user_vector[idx+1:]), dim=0)
                                else:
                                    user_vector = torch.cat((user_vector[:idx], self.CLS(torch.tensor([0]).to(self.device)), user_vector[idx+1:]), dim=0)
                            elif 'ItemOut' in t:
                                # 替换为物品CLS向量
                                if self.args.nn_parameter:
                                    user_vector = torch.cat((user_vector[:idx], self.CLS_item[torch.tensor([0]).to(self.device)], user_vector[idx+1:]), dim=0)
                                else:
                                    user_vector = torch.cat((user_vector[:idx], self.CLS_item(torch.tensor([0]).to(self.device)), user_vector[idx+1:]), dim=0)

                vectors.append(user_vector.unsqueeze(0))

            # 将所有样本的替换结果拼接回批次
            inputs_embeds = torch.cat(vectors)
        return inputs_embeds

    def _load_fixed_item_embeddings(self):
        """尝试加载 SA 阶段使用的固定 LLM 物品表示表。"""
        if self.args is None:
            return
        emb_path = getattr(self.args, 'llm_emb_path', '') or ''
        emb_path = emb_path.strip()
        if not emb_path:
            dataset = getattr(self.args, 'rec_pre_trained_data', '').strip()
            llm_name = getattr(self.args, 'llm', '').strip()
            save_dir = getattr(self.args, 'save_dir', '').strip()
            default_filename = f"{dataset}_{llm_name}_all_embeddings.pt" if dataset and llm_name else ''
            candidate_dirs = []
            if dataset and save_dir:
                candidate_dirs.append(os.path.join('./SA_assets', save_dir, dataset))
            if dataset:
                candidate_dirs.append(os.path.join('./SA_assets', dataset))
            if save_dir:
                candidate_dirs.append(os.path.join('./SA_assets', save_dir))
            candidate_dirs.append('./SA_assets')

            candidate_paths = []
            for base in candidate_dirs:
                if not base:
                    continue
                if default_filename:
                    candidate_paths.append(os.path.join(base, default_filename))
                candidate_paths.append(os.path.join(base, 'all_embeddings.pt'))

            for cand in candidate_paths:
                expanded_cand = os.path.abspath(os.path.expanduser(cand))
                if os.path.exists(expanded_cand):
                    emb_path = expanded_cand
                    if not getattr(self.args, 'local_rank', None) or self.args.local_rank in (-1, 0):
                        print(f"ℹ️ 自动匹配固定LLM物品嵌入: {expanded_cand}")
                    break

            if not emb_path:
                return
        expanded = os.path.abspath(os.path.expanduser(emb_path))
        if not os.path.exists(expanded):
            print(f"⚠️ 固定LLM物品表未找到: {expanded}")
            return
        try:
            emb_data = torch.load(expanded, map_location='cpu')
        except Exception as exc:
            print(f"⚠️ 无法加载固定LLM物品表 {expanded}: {exc}")
            return
        if isinstance(emb_data, dict):
            for key in ('embeddings', 'all_embeddings', 'item_embeddings', 'data', 'tensor'):
                if key in emb_data:
                    emb_data = emb_data[key]
                    break
        if not torch.is_tensor(emb_data):
            print(f"⚠️ 固定LLM物品表格式无效（需要Tensor）: {expanded}")
            return
        emb_data = emb_data.float().contiguous()
        if emb_data.ndim != 2 or emb_data.size(0) == 0:
            print(f"⚠️ 固定LLM物品表形状异常: {tuple(emb_data.shape)}")
            return
        first_row = emb_data[0]
        if not torch.allclose(first_row, torch.zeros_like(first_row), atol=1e-6):
            pad = torch.zeros(1, emb_data.size(1), dtype=emb_data.dtype)
            emb_data = torch.cat([pad, emb_data], dim=0)
        self.fixed_item_embeddings = emb_data
        self.fixed_item_embeddings_dim = emb_data.size(1)
        self.fixed_item_embeddings_path = expanded
        print(f"✅ 固定LLM物品嵌入加载完成: {emb_data.shape} <- {expanded}")

    def has_fixed_item_embeddings(self) -> bool:
        return isinstance(self.fixed_item_embeddings, torch.Tensor)

    def lookup_fixed_item_embeddings(self, item_ids, device=None):
        """根据物品 ID 批量检索固定的 LLM 物品嵌入。"""
        if not self.has_fixed_item_embeddings():
            raise RuntimeError("尚未加载固定的LLM物品嵌入。")
        if not torch.is_tensor(item_ids):
            ids = torch.as_tensor(item_ids, dtype=torch.long)
        else:
            ids = item_ids.detach().long()
        flat_ids = ids.view(-1).cpu()
        max_valid = self.fixed_item_embeddings.size(0) - 1
        if flat_ids.numel() > 0 and flat_ids.max().item() > max_valid:
            if not self._sa_index_warned:
                print(f"⚠️ 物品ID超出固定嵌入范围 ({flat_ids.max().item()} > {max_valid})，将自动截断")
                self._sa_index_warned = True
            flat_ids = torch.clamp(flat_ids, 0, max_valid)
        gathered = self.fixed_item_embeddings.index_select(0, flat_ids)
        gathered = gathered.view(*ids.shape, -1)
        target_device = device if device is not None else self.device
        return gathered.to(target_device)

    def encode_user_representations(self, text_input, history_embs, max_length=None, need_grad=False):
        """仅编码用户prompt，返回 pred_user 之后的表示。"""
        if text_input is None or len(text_input) == 0:
            return None
        max_input_length = max_length or getattr(self, 'max_prompt_length', 896)
        llm_tokens = self.llm_tokenizer(
            text_input,
            return_tensors="pt",
            padding="longest",
            truncation=True,
            max_length=max_input_length,
        ).to(self.device)

        inputs_embeds = self.llm_model.get_input_embeddings()(llm_tokens['input_ids'])
        inputs_embeds = self.replace_out_token_all(
            llm_tokens,
            inputs_embeds,
            token=['[UserOut]', '[HistoryEmb]'],
            embs={'[HistoryEmb]': history_embs}
        )

        batch_size = inputs_embeds.size(0)
        need_grad = bool(need_grad)
        chunk_threshold = getattr(self, 'sequence_chunk_threshold', batch_size + 1)

        if batch_size > chunk_threshold:
            seq_chunk_size = min(self.sequence_chunk_size, batch_size)
            min_seq_chunk = max(2, min(batch_size, self.min_sequence_chunk_size))
            hidden_states, _ = self._run_chunked_hidden_states(
                inputs_embeds,
                seq_chunk_size,
                need_grad,
                min_chunk_size=min_seq_chunk,
                log_prefix="SA用户序列",
                use_checkpoint=need_grad,
            )
            last_hidden = hidden_states
        else:
            grad_ctx = torch.enable_grad if need_grad else torch.no_grad
            with grad_ctx():
                with torch.amp.autocast('cuda'):
                    outputs = self.llm_model.model(
                        inputs_embeds=inputs_embeds,
                        output_hidden_states=True
                    )
            last_hidden = outputs.hidden_states[-1]

        indx = self.get_embeddings(llm_tokens, '[UserOut]')
        user_outputs = torch.cat([
            last_hidden[i, indx[i]].mean(axis=0).unsqueeze(0)
            for i in range(len(indx))
        ])

        if not need_grad:
            user_outputs = user_outputs.detach()
            with torch.no_grad():
                user_outputs = self.pred_user(user_outputs)
            return user_outputs.detach()

        user_outputs = self.pred_user(user_outputs)
        return user_outputs

    def _run_chunked_hidden_states(self, embeds, chunk_size, need_grad,
                                   min_chunk_size=4, log_prefix=None, use_checkpoint=True):
        batch_total = embeds.size(0)
        chunk_size = min(chunk_size, batch_total)
        min_chunk = max(1, min(batch_total, min_chunk_size))
        grad_ctx = torch.enable_grad if need_grad else torch.no_grad

        while True:
            try:
                # ✅ 预分配完整输出tensor,避免循环中反复torch.cat
                first_chunk_output = None
                result_outputs = None

                for i in range(0, batch_total, chunk_size):
                    end_idx = min(i + chunk_size, batch_total)
                    chunk_embeds = embeds[i:end_idx]
                    chunk_idx = i // chunk_size

                    with grad_ctx():
                        with torch.amp.autocast('cuda', dtype=torch.float16):
                            chunk_output = self.llm_model.model(
                                inputs_embeds=chunk_embeds,
                                output_hidden_states=True,
                                use_cache=True  # 训练时会触发警告并自动禁用
                            )

                    chunk_hidden = chunk_output.hidden_states[-1]
                    
                    # 第一个chunk:创建完整大小的tensor
                    if result_outputs is None:
                        hidden_dim = chunk_hidden.shape[-1]
                        result_outputs = torch.empty(
                            (batch_total, chunk_hidden.shape[1], hidden_dim),
                            dtype=chunk_hidden.dtype,
                            device=chunk_hidden.device
                        )
                    
                    # ✅ In-place填充,不创建新tensor
                    result_outputs[i:end_idx] = chunk_hidden
                    
                    del chunk_embeds, chunk_output, chunk_hidden

                return result_outputs, chunk_size
            except torch.cuda.OutOfMemoryError as oom:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                next_chunk = max(min_chunk, chunk_size // 2)
                if next_chunk == chunk_size:
                    raise oom
                chunk_size = next_chunk
                if log_prefix:
                    print(f"⚠️ {log_prefix}显存不足，自动将chunk_size降到 {chunk_size}")

    def get_embeddings(self, llm_tokens, token):
        """
        获取指定特殊token在输入序列中的位置索引

        这个函数用于定位特殊token（如[UserOut], [ItemOut]）在tokenized序列中的位置，
        以便后续从LLM的隐藏状态中提取对应位置的表示。

        Args:
            llm_tokens (dict): 分词后的token字典，包含 'input_ids'
            token (str): 要查找的特殊token，如 '[UserOut]'

        Returns:
            list: 每个样本中该token的位置索引列表
                 形如 [[pos1, pos2], [pos3], ...] 其中每个子列表对应一个样本

        使用场景：
        - 在LLM前向传播后，从hidden_states中提取特定位置的表示
        - 支持一个样本中有多个相同token的情况
        """
        token_idx = []
        # 获取token对应的ID
        token_id = self.llm_tokenizer(token, return_tensors="pt", add_special_tokens=False).input_ids.item()

        # 遍历批次中的每个样本
        for inx in range(len(llm_tokens['input_ids'])):
            # 找到该token在当前样本中的所有位置
            idx_tensor = (llm_tokens['input_ids'][inx] == token_id).nonzero().view(-1)
            token_idx.append(idx_tensor)
        return token_idx

    def forward(self, samples, mode=0):
        """
        模型前向传播的统一入口

        根据不同的模式调用相应的训练方法。当前主要支持mode=0的训练模式。

        Args:
            samples (dict): 训练样本字典，包含文本、嵌入等信息
            mode (int): 训练模式
                - 0: train_mode0，标准的EchoRec训练模式
                - 1: train_mode1，保留用于扩展（当前未实现）

        Returns:
            根据具体模式返回相应的损失值和指标
        """
        if mode == 0:
            return self.train_mode0(samples)
        elif mode == 1:
            return self.train_mode1(samples)

    def train_mode0(self, samples):
        """
        EchoRec的核心训练模式（完全对齐Start_old）

        这是模型的主要训练方法，实现以下关键流程：
        1. 处理用户历史文本，替换特殊token，生成用户表示
        2. 处理候选物品文本，替换特殊token，生成物品表示
        3. 计算推荐损失（用户-物品匹配）和对齐损失（与CF表示对齐）

        Args:
            samples (dict): 训练样本字典，包含：
                - 'text_input': 用户历史的文本prompt列表
                - 'log_emb': CF-SRec的用户表示（监督信号）
                - 'candidates_pos': 候选物品的文本prompt列表
                - 'interact': 历史物品嵌入列表（用于替换[HistoryEmb]）
                - 'candidate_embs': 候选物品嵌入张量

        Returns:
            Tuple[torch.Tensor, float, float]:
                - loss: 总损失（推荐损失 + 辅助损失）
                - rec_loss.item(): 推荐损失值
                - aux_summary.item(): 辅助损失值
        """
        self.iteration_count += 1

        max_input_length = getattr(self, 'max_prompt_length', 896)
        log_emb = samples['log_emb']
        student_repr = samples.get('student_repr', log_emb)
        student_items = samples.get('student_item_embs')
        teacher_user = samples.get('teacher_user_llm')
        teacher_items = samples.get('teacher_item_llm')

        llm_tokens = self.llm_tokenizer(
            samples['text_input'],
            return_tensors="pt",
            padding="longest",
            truncation=True,
            max_length=max_input_length,
        ).to(self.device)

        inputs_embeds = self.llm_model.get_input_embeddings()(llm_tokens['input_ids'])
        # 替换 [HistoryEmb] 为物品嵌入，[UserOut] 为CLS向量
        inputs_embeds = self.replace_out_token_all(
            llm_tokens,
            inputs_embeds,
            token=['[UserOut]', '[HistoryEmb]'],
            embs={'[HistoryEmb]': samples['interact']}
        )

        candi_tokens = self.llm_tokenizer(
            samples['candidates_pos'],
            return_tensors="pt",
            padding="longest",
            truncation=True,
            max_length=max_input_length,
        ).to(self.device)

        candi_embeds = self.llm_model.get_input_embeddings()(candi_tokens['input_ids'])
        candi_embeds = self.replace_out_token_all_infer(
            candi_tokens,
            candi_embeds,
            token=['[ItemOut]', '[HistoryEmb]'],
            embs={'[HistoryEmb]': samples['candidate_embs']}
        )

        need_llm_grad = True  # LLM 侧永远需要梯度
        total_candidates = candi_embeds.size(0)
        if total_candidates > self.chunk_split_threshold:
            chunk_size = min(self.candidate_chunk_size, total_candidates)
            min_chunk = max(4, min(total_candidates, self.min_candidate_chunk_size))

            if not hasattr(self, '_chunk_logged'):
                print(f"🔥 候选物品分块处理: {total_candidates} -> {math.ceil(total_candidates / chunk_size)}×{chunk_size}")
                self._chunk_logged = True

            candi_hidden_states, final_chunk = self._run_chunked_hidden_states(
                candi_embeds,
                chunk_size,
                need_llm_grad,
                min_chunk_size=min_chunk,
                log_prefix="LLM候选推理",
                use_checkpoint=need_llm_grad,
            )
            candi_outputs = SimplifiedOutput(candi_hidden_states)
            del candi_hidden_states
            self.candidate_chunk_size = final_chunk
        else:
            grad_ctx = torch.enable_grad() if need_llm_grad else torch.no_grad()
            with grad_ctx:
                with torch.amp.autocast('cuda'):
                    candi_outputs = self.llm_model.model(
                        inputs_embeds=candi_embeds,
                        output_hidden_states=True
                    )

        indx = self.get_embeddings(candi_tokens, '[ItemOut]')
        item_outputs = torch.cat([
            candi_outputs.hidden_states[-1][i, indx[i]].mean(axis=0).unsqueeze(0)
            for i in range(len(indx))
        ])
        del candi_outputs

        sequence_batch = inputs_embeds.size(0)
        sequence_chunk_threshold = getattr(self, 'sequence_chunk_threshold', sequence_batch + 1)
        if sequence_batch > sequence_chunk_threshold:
            if not hasattr(self, '_sequence_chunk_logged'):
                print(f"🔥 用户序列分块: {sequence_batch} -> {math.ceil(sequence_batch / self.sequence_chunk_size)}×{self.sequence_chunk_size}")
                self._sequence_chunk_logged = True

            seq_chunk_size = min(self.sequence_chunk_size, sequence_batch)
            min_seq_chunk = max(2, min(sequence_batch, self.min_sequence_chunk_size))
            user_hidden_states, final_seq_chunk = self._run_chunked_hidden_states(
                inputs_embeds,
                seq_chunk_size,
                need_llm_grad,
                min_chunk_size=min_seq_chunk,
                log_prefix="用户序列前向",
                use_checkpoint=need_llm_grad,
            )
            outputs = SimplifiedOutput(user_hidden_states)
        else:
            with torch.amp.autocast('cuda'):
                outputs = self.llm_model.model(
                    inputs_embeds=inputs_embeds,
                    output_hidden_states=True
                )

        indx = self.get_embeddings(llm_tokens, '[UserOut]')
        user_outputs = torch.cat([
            outputs.hidden_states[-1][i, indx[i]].mean(axis=0).unsqueeze(0)
            for i in range(len(indx))
        ])
        del outputs

        user_outputs = self.pred_user(user_outputs)
        item_outputs = self.pred_item(item_outputs)

        batch_size = student_repr.shape[0]
        num_candidates = samples.get('num_candidates', max(item_outputs.shape[0] // batch_size, 1))
        item_outputs_batched = item_outputs.view(batch_size, num_candidates, -1)

        llm_rec_loss = self.rec_loss(user_outputs, item_outputs)

        # match_loss: LLM → SASRec 对齐（MSE + uniformity，SASRec 冻结为教师）
        teacher_cf_mapped = self.pred_user_CF2(log_emb)
        user_outputs_norm = F.normalize(user_outputs, p=2, dim=1)
        teacher_norm = F.normalize(teacher_cf_mapped, p=2, dim=1)
        match_loss = self.mse(user_outputs_norm, teacher_norm)
        match_loss = match_loss + (self.uniformity(user_outputs_norm) + self.uniformity(teacher_norm))
        match_weight = float(getattr(self.args, 'match_weight', 1.0))
        total_loss = llm_rec_loss + match_weight * match_loss

        self._distillation_info = {
            "teacher_rec": llm_rec_loss.item(),
            "match_loss": match_loss.item(),
        }

        return total_loss, llm_rec_loss.item(), match_loss.item()


InjectionLLM = EchoRecLLM

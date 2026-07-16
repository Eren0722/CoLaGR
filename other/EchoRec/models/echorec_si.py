"""EchoRec SI model."""

import os
import pickle
import random
import time
from datetime import datetime
from functools import lru_cache

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from models.echorec_llm import EchoRecLLM
from models.echorec_teacher import EchoRecTeacher

try:
    import habana_frameworks.torch.core as htcore
except ImportError:
    htcore = None


class EchoRecSIModel(nn.Module):
    """Bridge a frozen sequential teacher and the trainable LLM side."""

    def __init__(self, args):
        super().__init__()
        self.args = args
        self.device = args.device
        self.inference_chunk_size = max(1, int(getattr(args, 'inference_chunk_size', 8)))

        with open(f'./SeqRec/data_{args.rec_pre_trained_data}/text_name_dict.json.gz', 'rb') as ft:
            self.text_name_dict = pickle.load(ft)

        self.recsys = EchoRecTeacher(
            args.recsys,
            args.rec_pre_trained_data,
            self.device,
            current_args=args,
        )

        self.student_trainable = getattr(args, 'train_student', False)
        for param in self.recsys.model.parameters():
            param.requires_grad = self.student_trainable

        if not hasattr(args, 'local_rank') or args.local_rank == 0:
            if self.student_trainable:
                trainable_params = sum(p.numel() for p in self.recsys.model.parameters() if p.requires_grad)
                total_params = sum(p.numel() for p in self.recsys.model.parameters())
                print(f"? SASRec ???: {trainable_params:,}/{total_params:,} ???????")
            else:
                print("?? SASRec ??????????")

        self.item_num = self.recsys.item_num
        self.rec_sys_dim = self.recsys.hidden_units
        self.llm_trainable = True
        self.optimizer_needs_reset = False
        self.mse = nn.MSELoss()

        self.llm = EchoRecLLM(device=self.device, llm_model=args.llm, args=self.args)
        hidden_size = self.llm.llm_model.config.hidden_size
        self.item_emb_proj = nn.Sequential(
            nn.Linear(self.rec_sys_dim, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.LeakyReLU(),
            nn.Linear(hidden_size, hidden_size),
        )
        nn.init.xavier_normal_(self.item_emb_proj[0].weight)
        nn.init.xavier_normal_(self.item_emb_proj[3].weight)

        self.current_phase = 'sequence_injection'
        self.all_embs = None
        self.history_window = max(3, min(int(getattr(self.args, 'train_history_window', 10)), 10))

        # === 状态变量 ===
        self.current_phase = 'sequence_injection'
        self.all_embs = None
        self.history_window = max(3, min(int(getattr(self.args, 'train_history_window', 10)), 10))

    def _resolve_positive_int(self, env_name: str = "", arg_name: str = "", default: int = 0) -> int:
        if env_name:
            raw_env = os.environ.get(env_name, "").strip()
            if raw_env:
                try:
                    parsed = int(raw_env)
                    if parsed > 0:
                        return parsed
                except ValueError:
                    pass

        if arg_name:
            raw_arg = getattr(self.args, arg_name, 0)
            try:
                parsed = int(raw_arg)
                if parsed > 0:
                    return parsed
            except (TypeError, ValueError):
                pass

        return default

    @staticmethod
    def _to_py_int(value):
        if isinstance(value, torch.Tensor):
            return int(value.item())
        if isinstance(value, np.generic):
            return int(value.item())
        return int(value)

    def _to_py_int_list(self, values):
        if torch.is_tensor(values):
            return [int(v) for v in values.detach().cpu().tolist()]
        return [self._to_py_int(v) for v in values]

    def _set_module_trainable(self, module, flag: bool):
        if module is None:
            return
        for param in module.parameters():
            param.requires_grad = flag

    def set_student_trainable(self, flag: bool, verbose: bool = False, force: bool = False):
        previous = getattr(self, 'student_trainable', None)
        if not force and previous == flag:
            return
        self.student_trainable = flag
        self._set_module_trainable(self.recsys.model, flag)
        try:
            self.recsys.model.train(flag)
        except Exception:
            pass
        if previous is not None and previous != flag:
            self.optimizer_needs_reset = True
        if verbose and (not hasattr(self.args, 'local_rank') or self.args.local_rank == 0):
            state = '解冻' if flag else '冻结'
            print(f"🔁 SASRec {state}，参与训练: {flag}")

    def set_llm_trainable(self, flag: bool, verbose: bool = False, force: bool = False):
        previous = getattr(self, 'llm_trainable', None)
        if not force and self.llm_trainable == flag:
            return
        self.llm_trainable = flag
        self._set_module_trainable(self.item_emb_proj, flag)
        self.llm.set_trainable(flag)
        if previous is not None and previous != flag:
            self.optimizer_needs_reset = True
        if verbose and (not hasattr(self.args, 'local_rank') or self.args.local_rank == 0):
            state = '解冻' if flag else '冻结'
            print(f"🔁 LLM 侧组件 {state}，参与训练: {flag}")
    
    def set_mapper_trainable(self, flag: bool, verbose: bool = False):
        """
        独立控制cf_to_llm_mapper及预测头的可训练状态
        语义对齐阶段若启用alignment_updates_mapper，可让这些轻量层继续接收小权重的前向蒸馏更新
        """
        previous = getattr(self, 'mapper_trainable', None)
        if previous is not None and previous == flag:
            return
        self.mapper_trainable = flag

        toggled_modules = []
        if hasattr(self.llm, 'cf_to_llm_mapper'):
            self._set_module_trainable(self.llm.cf_to_llm_mapper, flag)
            toggled_modules.append('cf_to_llm_mapper')
        if hasattr(self.llm, 'pred_user'):
            self._set_module_trainable(self.llm.pred_user, flag)
            toggled_modules.append('pred_user')
        if hasattr(self.llm, 'pred_item'):
            self._set_module_trainable(self.llm.pred_item, flag)
            toggled_modules.append('pred_item')

        if previous is not None and previous != flag:
            self.optimizer_needs_reset = True

        if verbose and toggled_modules and (not hasattr(self.args, 'local_rank') or self.args.local_rank == 0):
            state = '解冻' if flag else '冻结'
            module_names = ', '.join(toggled_modules)
            print(f"🔁 {module_names} {state}，参与训练: {flag}")

    def _ensure_item_embeddings_ready(self, force_recompute: bool = False, desc: str = "Building item embeddings"):
        """构建或刷新 LLM 物品嵌入缓存，可在阶段切换后强制重建"""
        if force_recompute:
            if hasattr(self, 'all_embs'):
                delattr(self, 'all_embs')

        if hasattr(self, 'all_embs') and self.all_embs is not None:
            return self.all_embs

        batch_ = 128
        if self.args.llm in ['llama', 'llama-3b']:
            batch_ = 32
        if self.args.rec_pre_trained_data in {'Electronics', 'Books'}:
            batch_ = 32
            if self.args.llm in ['llama', 'llama-3b']:
                batch_ = 16
        batch_ = self._resolve_positive_int(
            env_name="LLMSREC_EVAL_ITEM_BATCH",
            arg_name="eval_item_batch",
            default=batch_,
        )

        item_ids = list(range(1, self.item_num + 1))
        processed_items = 0
        adaptive_item_batch = batch_
        max_input_length = self._resolve_positive_int(
            env_name="LLMSREC_EVAL_MAX_LENGTH",
            arg_name="eval_max_length",
            default=512,
        )
        oom_notice_items = False
        self.all_embs = []

        with tqdm(total=self.item_num, desc=desc) as pbar:
            while processed_items < self.item_num:
                end_idx = min(processed_items + adaptive_item_batch, self.item_num)
                current_ids = item_ids[processed_items:end_idx]
                candidate_text = []
                candidate_ids = []
                for neg_candidate in current_ids:
                    candidate_text.append(self._candidate_prompt(neg_candidate))
                    candidate_ids.append(neg_candidate)

                try:
                    with torch.no_grad():
                        candi_tokens = self.llm.llm_tokenizer(
                            candidate_text,
                            return_tensors="pt",
                            padding="longest",
                            truncation=True,
                            max_length=max_input_length,
                        ).to(self.device)

                        candidate_embs = self.item_emb_proj(self.get_item_emb(candidate_ids))
                        candi_embeds = self.llm.llm_model.get_input_embeddings()(candi_tokens['input_ids'])
                        candi_embeds = self.llm.replace_out_token_all_infer(
                            candi_tokens,
                            candi_embeds,
                            token=['[ItemOut]', '[HistoryEmb]'],
                            embs={'[HistoryEmb]': candidate_embs}
                        )

                        with torch.amp.autocast('cuda'):
                            candi_outputs = self.llm.llm_model.model(
                                inputs_embeds=candi_embeds,
                                output_hidden_states=True
                            )

                            indx = self.llm.get_embeddings(candi_tokens, '[ItemOut]')
                            item_outputs = torch.cat([
                                candi_outputs.hidden_states[-1][i, indx[i]].mean(axis=0).unsqueeze(0)
                                for i in range(len(indx))
                            ])
                            item_outputs = self.llm.pred_item(item_outputs)

                        self.all_embs.append(item_outputs.detach().cpu())

                        del candi_outputs
                        del item_outputs
                        del candi_embeds
                        torch.cuda.empty_cache()

                    processed_items = end_idx
                    pbar.update(len(current_ids))
                    oom_notice_items = False

                except RuntimeError as e:
                    if 'CUDA out of memory' in str(e) and adaptive_item_batch > 1:
                        torch.cuda.empty_cache()
                        adaptive_item_batch = max(1, adaptive_item_batch // 2)
                        if not oom_notice_items:
                            print(f"⚠️ 物品编码阶段检测到OOM，自动将batch_size降至 {adaptive_item_batch}")
                            oom_notice_items = True
                        continue
                    else:
                        raise

        self.all_embs = torch.cat(self.all_embs, dim=0).contiguous()
        return self.all_embs

    def configure_training_phase(self, phase_name: str, weight_cfg: dict, verbose: bool = False, round_idx: int = 1):
        phase_name = phase_name.lower()
        if phase_name != 'sequence_injection':
            raise ValueError(f"未知的训练阶段: {phase_name}")
        self.set_student_trainable(False, verbose)
        self.set_llm_trainable(True, verbose)
        self.llm.set_loss_weights(
            teacher_rec=weight_cfg.get('teacher_rec', 1.0),
            student_rec=weight_cfg.get('student_rec', 0.0),
            forward=weight_cfg.get('forward', 1.0),
            backward=0.0,
        )
        self.current_phase = phase_name

        if verbose and (not hasattr(self.args, 'local_rank') or self.args.local_rank == 0):
            print(f"🌀 Round {round_idx} 进入阶段 [{phase_name}]：权重 {weight_cfg}")





    def save_model(self, args, epoch2=None, best=False, subdir=None):
        """
        保存当前训练好的模型组件到磁盘

        保存策略：
        - 仅保存可训练的组件（CF-SRec保持冻结，无需保存）
        - 根据训练模式保存不同的组件组合
        - 支持保存最佳模型到单独目录

        Args:
            args: 全局配置对象
            epoch2 (int, optional): 当前训练轮次，用于文件命名
            best (bool): 是否保存为最佳模型，默认False

        文件命名格式：
            ./models/{save_dir}/[best/]{rec_pre_trained_data}_{llm}_{epoch2}_{component}.pt
        """
        # 构建保存目录路径
        base_dir = os.path.join('./models', args.rec_pre_trained_data, args.save_dir)
        if subdir is not None:
            out_dir = os.path.join(base_dir, subdir)
        else:
            out_dir = os.path.join(base_dir, 'best') if best else base_dir

        # 确保目录存在
        os.makedirs(out_dir, exist_ok=True)

        # 构建文件名前缀：数据集_模型_轮次_
        out_dir = os.path.join(
            out_dir,
            f'{args.rec_pre_trained_data}_{args.llm}_{epoch2}_'
        )

        # 仅在训练模式下保存模型
        if args.train:
            compact_ckpt = {
                'item_emb_proj': self.item_emb_proj.state_dict(),
                'pred_user': self.llm.pred_user.state_dict(),
                'pred_item': self.llm.pred_item.state_dict(),
                'pred_user_CF2': self.llm.pred_user_CF2.state_dict(),
            }
            # 1. 保存物品嵌入投影层（连接CF-SRec和LLM的桥梁）
            torch.save(self.item_emb_proj.state_dict(), out_dir + 'item_proj.pt')

            # 2. 保存用户预测头（将LLM隐藏状态映射到用户表示空间）
            torch.save(self.llm.pred_user.state_dict(), out_dir + 'pred_user.pt')

            # 3. 保存物品预测头（将LLM隐藏状态映射到物品表示空间）
            torch.save(self.llm.pred_item.state_dict(), out_dir + 'pred_item.pt')
            torch.save(self.llm.pred_user_CF2.state_dict(), out_dir + 'pred_user_cf2.pt')

            # 4. 根据训练策略保存不同的特殊token表示
            if not args.token:
                # 策略A：使用可学习的CLS向量替换[UserOut]/[ItemOut]
                if args.nn_parameter:
                    # 神经网络处理器模式：直接保存Parameter的data
                    torch.save(self.llm.CLS.data, out_dir + 'CLS.pt')
                    torch.save(self.llm.CLS_item.data, out_dir + 'CLS_item.pt')
                else:
                    # 标准模式：保存Embedding层的state_dict
                    torch.save(self.llm.CLS.state_dict(), out_dir + 'CLS.pt')
                    torch.save(self.llm.CLS_item.state_dict(), out_dir + 'CLS_item.pt')
                compact_ckpt['CLS'] = self.llm.CLS.state_dict()
                compact_ckpt['CLS_item'] = self.llm.CLS_item.state_dict()
            if args.token:
                # 策略B：训练LLM的词嵌入层
                torch.save(self.llm.llm_model.model.embed_tokens.state_dict(), out_dir + 'token.pt')
                compact_ckpt['token'] = self.llm.llm_model.model.embed_tokens.state_dict()

            if getattr(args, 'train_student', False):
                torch.save(self.recsys.model.state_dict(), out_dir + 'student.pt')
                compact_ckpt['student'] = self.recsys.model.state_dict()

            compact_path = out_dir + 'model.pth'
            torch.save(compact_ckpt, compact_path)


    def load_model(self, args, phase1_epoch=None, phase2_epoch=None, subdir=None):
        """
        从磁盘加载预训练的模型组件

        加载策略：
        - 按照save_model的保存格式加载对应组件
        - 根据训练策略加载不同的特殊token表示
        - 使用map_location确保设备兼容性

        Args:
            args: 全局配置对象
            phase1_epoch (int, optional): 第一阶段轮次（当前未使用，保留兼容性）
            phase2_epoch (int, optional): 第二阶段轮次，用于确定加载的模型版本

        注意：
            - 必须与save_model的保存格式完全匹配
            - 加载后显式删除临时变量释放内存
        """
        # 构建加载路径，与save_model的命名格式一致
        base_dir = os.path.join('./models', args.rec_pre_trained_data, args.save_dir)
        if subdir is not None:
            base_dir = os.path.join(base_dir, subdir)
        out_dir = os.path.join(
            base_dir,
            f'{args.rec_pre_trained_data}_{args.llm}_{phase2_epoch}_'
        )

        compact_path = out_dir + 'model.pth'
        if os.path.exists(compact_path):
            checkpoint = torch.load(compact_path, map_location=self.device)
            self.item_emb_proj.load_state_dict(checkpoint['item_emb_proj'])
            self.llm.pred_user.load_state_dict(checkpoint['pred_user'])
            self.llm.pred_item.load_state_dict(checkpoint['pred_item'])
            if 'pred_user_CF2' in checkpoint:
                self.llm.pred_user_CF2.load_state_dict(checkpoint['pred_user_CF2'])
            if not args.token and 'CLS' in checkpoint:
                self.llm.CLS.load_state_dict(checkpoint['CLS'])
                self.llm.CLS_item.load_state_dict(checkpoint['CLS_item'])
            if args.token and 'token' in checkpoint:
                self.llm.llm_model.model.embed_tokens.load_state_dict(checkpoint['token'])
            if getattr(args, 'train_student', False) and 'student' in checkpoint:
                missing = self.recsys.model.load_state_dict(checkpoint['student'], strict=False)
                if not hasattr(self.args, 'local_rank') or self.args.local_rank == 0:
                    if missing.missing_keys or missing.unexpected_keys:
                        print(f"⚠️ 学生模型加载存在缺失/多余键: {missing}")
                    else:
                        print("✅ 学生模型权重加载成功")
            del checkpoint
            return

        # 兼容旧格式：逐文件加载
        item_emb_proj = torch.load(out_dir + 'item_proj.pt', map_location=self.device)
        self.item_emb_proj.load_state_dict(item_emb_proj)
        del item_emb_proj  # 释放内存

        pred_user = torch.load(out_dir + 'pred_user.pt', map_location=self.device)
        self.llm.pred_user.load_state_dict(pred_user)
        del pred_user  # 释放内存

        pred_item = torch.load(out_dir + 'pred_item.pt', map_location=self.device)
        self.llm.pred_item.load_state_dict(pred_item)
        del pred_item  # 释放内存

        pred_user_cf2_path = out_dir + 'pred_user_cf2.pt'
        if os.path.exists(pred_user_cf2_path):
            pred_user_cf2 = torch.load(pred_user_cf2_path, map_location=self.device)
            self.llm.pred_user_CF2.load_state_dict(pred_user_cf2)
            del pred_user_cf2  # 释放内存

        if not args.token:
            CLS = torch.load(out_dir + 'CLS.pt', map_location=self.device)
            self.llm.CLS.load_state_dict(CLS)
            del CLS

            CLS_item = torch.load(out_dir + 'CLS_item.pt', map_location=self.device)
            self.llm.CLS_item.load_state_dict(CLS_item)
            del CLS_item

        if args.token:
            token = torch.load(out_dir + 'token.pt', map_location=self.device)
            self.llm.llm_model.model.embed_tokens.load_state_dict(token)
            del token

        student_path = out_dir + 'student.pt'
        if getattr(args, 'train_student', False) and os.path.exists(student_path):
            student_state = torch.load(student_path, map_location=self.device)
            missing = self.recsys.model.load_state_dict(student_state, strict=False)
            if not hasattr(self.args, 'local_rank') or self.args.local_rank == 0:
                if missing.missing_keys or missing.unexpected_keys:
                    print(f"⚠️ 学生模型加载存在缺失/多余键: {missing}")
                else:
                    print("✅ 学生模型权重加载成功")
            del student_state
            

    def find_item_text(self, item, title_flag=True, description_flag=True):
        """
        批量获取物品的文本信息（标题和/或描述）

        Args:
            item (list): 物品ID列表
            title_flag (bool): 是否包含标题，默认True
            description_flag (bool): 是否包含描述，默认True

        Returns:
            list: 格式化的物品文本列表，每个元素为带引号的字符串
        """
        t = 'title'
        d = 'description'
        t_ = 'No Title'      # 缺失标题的默认值
        d_ = 'No Description'  # 缺失描述的默认值
        item_ids = self._to_py_int_list(item)

        if title_flag and description_flag:
            # 返回 "标题, 描述" 格式
            return [f'"{self.text_name_dict[t].get(i,t_)}, {self.text_name_dict[d].get(i,d_)}"' for i in item_ids]
        elif title_flag and not description_flag:
            # 仅返回标题
            return [f'"{self.text_name_dict[t].get(i,t_)}"' for i in item_ids]
        elif not title_flag and description_flag:
            # 仅返回描述
            return [f'"{self.text_name_dict[d].get(i,d_)}"' for i in item_ids]

    def find_item_time(self, item, user, title_flag=True, description_flag=True):
        """
        获取用户与物品交互的时间信息

        Args:
            item (list): 物品ID列表
            user (int): 用户ID
            title_flag (bool): 未使用，保留兼容性
            description_flag (bool): 未使用，保留兼容性

        Returns:
            list: 格式化的日期字符串列表，格式为 'YYYY-MM-DD'
        """
        # 从时间戳字典中获取交互时间（毫秒级时间戳）
        # 转换为UTC时间对象，然后格式化为日期字符串
        # 确保user是Python整数而不是tensor
        user_id = self._to_py_int(user)
        item_ids = self._to_py_int_list(item)
        l = []
        for i in item_ids:
            try:
                # 尝试获取时间戳
                timestamp = int(self.text_name_dict['time'][i][user_id])/1000
                l.append(datetime.utcfromtimestamp(timestamp))
            except (KeyError, ValueError):
                # 如果没有时间信息，使用默认时间（2020-01-01）
                l.append(datetime(2020, 1, 1))
        return [l_.strftime('%Y-%m-%d') for l_ in l]

    @lru_cache(maxsize=200000)
    def find_item_text_single(self, item, title_flag=True, description_flag=True):
        """
        获取单个物品的文本信息（标题和/或描述）

        Args:
            item (int): 单个物品ID
            title_flag (bool): 是否包含标题，默认True
            description_flag (bool): 是否包含描述，默认True

        Returns:
            str: 格式化的物品文本字符串，带引号
        """
        t = 'title'
        d = 'description'
        t_ = 'No Title'      # 缺失标题的默认值
        d_ = 'No Description'  # 缺失描述的默认值
        item = self._to_py_int(item)

        if title_flag and description_flag:
            # 返回 "标题, 描述" 格式
            return f'"{self.text_name_dict[t].get(item,t_)}, {self.text_name_dict[d].get(item,d_)}"'
        elif title_flag and not description_flag:
            # 仅返回标题
            return f'"{self.text_name_dict[t].get(item,t_)}"'
        elif not title_flag and description_flag:
            # 仅返回描述
            return f'"{self.text_name_dict[d].get(item,d_)}"'

    @lru_cache(maxsize=200000)
    def _candidate_prompt(self, item_id):
        item_id = self._to_py_int(item_id)
        item_title = self.find_item_text_single(item_id, title_flag=True, description_flag=False)
        return f'The item title and item embedding are as follows: {item_title}[HistoryEmb], then generate item representation token:[ItemOut]'

    def get_item_emb(self, item_ids):
        """
        从CF-SRec模型获取物品嵌入向量

        Args:
            item_ids (list): 物品ID列表

        Returns:
            torch.Tensor: 物品嵌入张量，形状为 [len(item_ids), embedding_dim]

        注意：
            - 根据 args.nn_parameter 决定使用不同的访问方式
            - nn_parameter=True: 直接索引访问（适用于某些硬件优化）
            - nn_parameter=False: 通过Embedding层调用（标准方式）
        """
        with torch.no_grad():
            if torch.is_tensor(item_ids):
                item_tensor = item_ids.to(self.device, dtype=torch.long, non_blocking=True)
            else:
                item_tensor = torch.as_tensor(item_ids, dtype=torch.long, device=self.device)
            if hasattr(self.recsys.model, 'get_item_embeddings'):
                item_embs = self.recsys.model.get_item_embeddings(item_tensor)
            elif self.args.nn_parameter:
                # 直接通过索引访问嵌入矩阵（硬件优化模式）
                item_embs = self.recsys.model.item_emb[item_tensor]
            else:
                # 通过Embedding层的forward方法（标准模式）
                item_embs = self.recsys.model.item_emb(item_tensor)

        return item_embs
    
    def forward(self, data, optimizer=None, batch_iter=None, mode='phase1'):
        """
        模型前向传播的统一入口，根据模式分发到不同的处理流程

        这是一个路由方法，根据不同的模式调用相应的核心方法：
        - phase2: 训练模式，执行核心训练循环
        - generate_batch: 推理评估模式，生成推荐并计算指标
        - extract: 表示提取模式，仅提取用户表示

        Args:
            data: 输入数据，格式根据模式而定
            optimizer: 优化器实例（仅训练模式需要）
            batch_iter: 训练进度信息（仅训练模式需要）
            mode (str): 运行模式，默认'phase1'
                - 'phase2': 第二阶段训练模式
                - 'generate_batch': 批量推理评估模式
                - 'extract': 用户表示提取模式

        注意：
            - phase1模式当前未实现，保留用于扩展
            - generate_batch模式会自动输出评估结果
        """
        if mode == 'phase2':
            # 训练模式：执行核心训练循环
            self.pre_train_phase2(data, optimizer, batch_iter)

        if mode == 'generate_batch':
            # 推理评估模式：生成推荐并计算评估指标
            self.generate_batch(data)

            # 输出评估结果：按累计用户数抽样打印
            eusers = getattr(self.args, 'eval_log_users', 0)
            if eusers and eusers > 0:
                if (self.users % eusers) < 1e-6:  # 例如每累计 1000 用户打印一次
                    print(self.args.save_dir, self.args.rec_pre_trained_data)
                    print('test (NDCG@10: %.4f, HR@10: %.4f), Num User: %.0f'
                          % (self.NDCG/self.users, self.HT/self.users, self.users))
                    print('test (NDCG@20: %.4f, HR@20: %.4f), Num User: %.0f'
                          % (self.NDCG_20/self.users, self.HIT_20/self.users, self.users))

        if mode == 'extract':
            # 表示提取模式：仅提取用户表示，不进行推荐评估
            self.extract_emb(data)

    def make_interact_text(self, interact_ids, interact_max_num, user):
        """
        构造用户历史交互的文本表示，用于LLM输入

        将用户的历史交互物品转换为可读的文本序列，每个物品包含：
        - 序号标识（Item No.1, Item No.2, ...）
        - 交互时间（格式化为日期）
        - 物品标题文本
        - 特殊占位符 [HistoryEmb] 用于后续的向量替换

        Args:
            interact_ids (list): 用户历史交互的物品ID列表，按时间顺序排列
            interact_max_num (int|str): 最大交互数量限制
                - 整数：取最近的N个交互
                - 'all'：使用全部历史交互
            user (int): 用户ID，用于获取交互时间信息

        Returns:
            Tuple[str, list]:
                - interact_text: 格式化的交互文本，用逗号连接
                  例如: "Item No.1, Time: 2023-01-15, iPhone 13[HistoryEmb],Item No.2, Time: 2023-02-20, MacBook Pro[HistoryEmb]"
                - interact_ids: 实际使用的物品ID列表（可能被截断）

        注意：
            - [HistoryEmb] 是特殊token，会在后续被对应物品的嵌入向量替换
            - 时间信息来自 text_name_dict 中的时间戳数据
        """
        # 获取所有交互物品的标题文本（仅标题，不包含描述）
        interact_item_titles_ = self.find_item_text(interact_ids, title_flag=True, description_flag=False)

        # 获取交互时间信息，用于构造时间上下文
        times = self.find_item_time(interact_ids, user)

        # 存储格式化后的交互文本片段
        interact_text = []
        count = 1  # 物品序号计数器

        # 根据 interact_max_num 决定使用全部历史还是截取最近的交互
        if interact_max_num == 'all':
            # 使用全部历史交互
            times = self.find_item_time(interact_ids, user)
        else:
            # 只使用最近的 interact_max_num 个交互
            # 取列表末尾的元素（最近的交互）
            times = self.find_item_time(interact_ids[-interact_max_num:], user)

        # 构造交互文本序列
        if interact_max_num == 'all':
            # 处理全部历史交互
            for title in interact_item_titles_:
                # 格式：Item No.序号, Time: 日期, 物品标题[HistoryEmb]
                interact_text.append(f'Item No.{count}, Time: {times[count-1]}, ' + title + '[HistoryEmb]')
                count += 1
        else:
            # 处理截取的最近交互
            for title in interact_item_titles_[-interact_max_num:]:
                interact_text.append(f'Item No.{count}, Time: {times[count-1]}, ' + title + '[HistoryEmb]')
                count += 1
            # 同步更新 interact_ids，保持与文本的一致性
            interact_ids = interact_ids[-interact_max_num:]

        # 将所有交互文本片段用逗号连接成完整的历史序列
        interact_text = ','.join(interact_text)
        return interact_text, interact_ids

    
    
    def make_candidate_text(self, interact_ids, candidate_num, target_item_id, target_item_title, candi_set=None, task='ItemTask'):
        """
        构造候选物品的文本表示，用于训练阶段的物品表示学习

        Args:
            interact_ids (list): 用户历史交互的物品ID列表，用于负采样时排除
            candidate_num (int): 候选物品总数（包含1个正样本 + N-1个负样本）
            target_item_id (int): 目标物品ID（正样本）
            target_item_title (str): 目标物品的标题文本
            candi_set (set, optional): 候选物品集合，如果提供则从中采样负样本
            task (str): 任务类型标识，默认为 'ItemTask'

        Returns:
            Tuple[list, list]:
                - candidate_text: 候选物品的文本表示列表
                - candidate_ids: 对应的候选物品ID列表（正样本在第一位）
        """
        need_neg = candidate_num - 1  # 需要的负样本数
        neg_item_id = []

        # 随机负样本采样
        interact_set = set(interact_ids) if not isinstance(interact_ids, set) else interact_ids
        exclude_set = set(interact_set)  # 只排除交互历史，不排除 target（与 EchoRec/A-LLMRec 一致）

        if candi_set is None:
            while len(neg_item_id) < max(need_neg, 99):
                t = np.random.randint(1, self.item_num + 1)
                if t not in exclude_set:
                    neg_item_id.append(t)
                    exclude_set.add(t)
        else:
            items = list(candi_set - exclude_set)
            if len(items) >= max(need_neg, 99) - len(neg_item_id):
                neg_item_id += random.sample(items, max(need_neg, 99) - len(neg_item_id))
            else:
                neg_item_id += items
                while len(neg_item_id) < max(need_neg, 49):
                    t = np.random.randint(1, self.item_num + 1)
                    if t not in exclude_set:
                        neg_item_id.append(t)
                        exclude_set.add(t)

        # 打乱负样本顺序，避免位置偏差
        random.shuffle(neg_item_id)

        # 构建候选物品ID列表：正样本在第一位
        candidate_ids = [target_item_id]

        # 构建候选物品文本表示列表
        # 正样本的文本表示
        candidate_text = [self._candidate_prompt(target_item_id)]

        # 负样本的文本表示（动态获取标题）
        for neg_candidate in neg_item_id[:candidate_num - 1]:
            # 获取负样本物品的标题
            candidate_text.append(self._candidate_prompt(neg_candidate))
            candidate_ids.append(neg_candidate)

        return candidate_text, candidate_ids
    
    
    def make_candidate(self, interact_ids, candidate_num, target_item_id, target_item_title, candi_set = None, task = 'ItemTask'):
        """
        生成候选物品ID列表，用于推理阶段的推荐评估

        与 make_candidate_text 的区别：
        - make_candidate_text: 生成文本表示 + ID列表（训练用）
        - make_candidate: 仅生成ID列表（推理评估用）

        Args:
            interact_ids (list): 用户历史交互的物品ID列表，用于负采样时排除
            candidate_num (int): 候选物品总数（包含1个正样本 + N-1个负样本）
            target_item_id (int): 目标物品ID（正样本）
            target_item_title (str): 目标物品标题（当前未使用，保留兼容性）
            candi_set (set, optional): 候选物品集合（当前未使用，保留兼容性）
            task (str): 任务类型标识（当前未使用，保留兼容性）

        Returns:
            list: 候选物品ID列表，正样本在第一位，后续为负样本
        """
        neg_item_id = []
        neg_item_id = []  # 重复初始化（可能是代码遗留，但保持不变）

        # 随机采样99个负样本（固定数量，用于评估）
        while len(neg_item_id) < 99:
            t = np.random.randint(1, self.item_num + 1)  # 随机选择物品ID（1到item_num）
            # 确保负样本不在用户历史交互中，且不重复
            if not (t in interact_ids or t in neg_item_id):
                neg_item_id.append(t)

        # 打乱负样本顺序，避免位置偏差
        random.shuffle(neg_item_id)

        # 构建候选物品ID列表：正样本在第一位
        candidate_ids = [target_item_id]

        # 添加指定数量的负样本
        candidate_ids = candidate_ids + neg_item_id[:candidate_num - 1]

        return candidate_ids
    
    
    def pre_train_phase2(self, data, optimizer, batch_iter):
        """
        EchoRec 核心训练循环 - 第二阶段预训练

        这是模型的核心训练方法，实现以下关键流程：
        1. 从CF-SRec获取用户历史表示（作为监督信号）
        2. 构造用户历史和候选物品的文本prompt
        3. 通过特殊token机制注入嵌入向量
        4. 计算推荐损失和对齐损失
        5. 反向传播更新模型参数

        Args:
            data (tuple): 批次训练数据 (u, seq, pos, neg)
                - u: 用户ID列表 [batch_size]
                - seq: 用户历史序列 [batch_size, max_len]
                - pos: 正样本物品ID [batch_size, max_len]
                - neg: 负样本物品ID [batch_size, max_len]
            optimizer: PyTorch优化器实例
            batch_iter (tuple): 训练进度信息 (epoch, total_epoch, step, total_step)

        核心思想：
            - 使用冻结的CF-SRec作为"教师"提供用户表示
            - 训练LLM学习从文本+嵌入中生成相同的用户表示
            - 同时学习物品表示用于推荐排序
        """
        # 解包训练进度信息，用于日志输出
        epoch, total_epoch, step, total_step = batch_iter

        if getattr(self.args, 'log_interval', 50) > 0 and (step % self.args.log_interval) == 0:
            print(self.args.save_dir, self.args.rec_pre_trained_data, self.args.llm)

        # 解包批次数据
        u, seq, pos, neg = data

        # 保存原始序列的副本（虽然当前未使用，但保留用于调试）
        original_seq = seq.clone() if torch.is_tensor(seq) else seq.copy()

        def _to_recsys_input(x):
            if torch.is_tensor(x):
                return x.detach().cpu().numpy().astype(np.int64, copy=False)
            return np.asarray(x, dtype=np.int64)

        recsys_u = _to_recsys_input(u)
        recsys_seq = _to_recsys_input(seq)
        recsys_pos = _to_recsys_input(pos)
        recsys_neg = _to_recsys_input(neg)

        # 初始化损失累积变量（当前未使用，但保留用于扩展）
        mean_loss = 0

        # ✅ 对齐Start_old: 变量初始化
        text_input = []           # 用户历史的文本prompt列表
        candidates_pos = []       # 候选物品的文本prompt列表
        candidates_neg = []       # 负样本文本（当前未使用）
        interact_embs = []        # 历史物品的嵌入向量列表
        candidate_embs_pos = []   # 候选物品的嵌入向量列表
        candidate_embs_neg = []   # 负样本嵌入（当前未使用）
        candidate_embs = []       # 最终的候选物品嵌入张量
        student_candidate_embs = []  # CF候选嵌入，用于学生损失
        teacher_candidate_embs = []  # RLMRec候选嵌入

        # 保留的损失变量（当前未使用，但保留用于多模式训练）
        loss_rm_mode2 = 0

        # 步骤1: 从CF-SRec获取用户表示（监督信号）
        def _student_params_have_grad():
            try:
                return any(param.requires_grad for param in self.recsys.model.parameters())
            except Exception:
                return True

        if self.student_trainable and not _student_params_have_grad():
            if primary_rank:
                print("⚠️ 检测到 SASRec 梯度已被关闭，Phase 2 自动重新解冻")
            self.set_student_trainable(True, verbose=False, force=True)

        if self.student_trainable:
            with torch.enable_grad():
                log_emb = self.recsys.model(recsys_u, recsys_seq, recsys_pos, recsys_neg, mode='log_only')
            if not isinstance(log_emb, torch.Tensor) or not log_emb.requires_grad:
                if primary_rank:
                    print("⚠️ SASRec 输出未连接梯度，尝试重新解冻后再计算一次")
                self.set_student_trainable(True, verbose=False, force=True)
                with torch.enable_grad():
                    log_emb = self.recsys.model(recsys_u, recsys_seq, recsys_pos, recsys_neg, mode='log_only')
                if primary_rank and (not isinstance(log_emb, torch.Tensor) or not log_emb.requires_grad):
                    print("❌ 警告：SASRec 在解冻状态下仍未产生梯度，Phase 2 将无法更新学生模型")
        else:
            with torch.no_grad():
                log_emb = self.recsys.model(recsys_u, recsys_seq, recsys_pos, recsys_neg, mode='log_only')

        # 步骤2: 逐样本构造训练数据
        # 🔧 Phase2显存优化：使用梯度累积支持更多候选
        # 默认20：batch20×20=400候选，配合梯度累积避免OOM
        history_window = getattr(self.args, 'train_history_window', 10)
        history_window = max(3, min(history_window, 10))
        candidate_num = getattr(self.args, 'train_candidate_num', 4)  # 默认4与EchoRec原版一致
        candidate_num = max(2, min(candidate_num, 20))

        batch_size = int(u.size(0)) if torch.is_tensor(u) else len(u)
        for i in range(batch_size):
            # 获取当前样本的目标物品（序列最后一个正样本）
            target_item_id = self._to_py_int(pos[i][-1])
            target_item_title = self.find_item_text_single(target_item_id, title_flag=True, description_flag=False)

            # 有效历史 ID（用于 neg sampling 排除和 candidate 构造）
            if torch.is_tensor(seq):
                raw_history_ids = self._to_py_int_list(seq[i][seq[i] > 0])
                user_id = self._to_py_int(u[i])
            else:
                raw_history_ids = seq[i][seq[i]>0]
                user_id = self._to_py_int(u[i])

            # 构造用户历史交互文本
            interact_text, interact_ids = self.make_interact_text(raw_history_ids, history_window, user_id)

            # 构造候选物品的文本表示（随机负采样）
            candidate_text, candidate_ids = self.make_candidate_text(
                raw_history_ids, candidate_num, target_item_id, target_item_title,
                task='RecTask'
            )

            # 构造用户历史的完整prompt
            input_text = ''
            input_text += 'This user has made a series of purchases in the following order: '
            input_text += interact_text  # 包含 [HistoryEmb] 占位符的历史文本
            input_text += ". Based on this sequence of purchases, generate user representation token:[UserOut]"

            # 添加到批次数据中
            text_input.append(input_text)
            candidates_pos += candidate_text  # 扩展候选物品文本列表

            # 步骤3: 准备嵌入向量用于替换特殊token
            # 历史物品嵌入：从CF-SRec获取并投影到LLM空间
            # 注意：使用 interact_ids（可能是增强后的 ID），嵌入也随之变化
            interact_embs.append(self.item_emb_proj((self.get_item_emb(interact_ids))))

            # 候选物品嵌入：用于学生和LLM两个通路
            cf_candidate_emb = self.get_item_emb(candidate_ids)
            student_candidate_embs.append(cf_candidate_emb)
            candidate_embs_pos.append(self.item_emb_proj(cf_candidate_emb))

        # 步骤4: 整合批次数据
        # 将所有候选物品嵌入拼接成一个张量
        candidate_embs = torch.cat(candidate_embs_pos)
        student_candidate_tensor = torch.stack(student_candidate_embs)

        # 构造LLM训练样本字典
        samples = {
            'text_input': text_input,           # 用户历史prompt列表
            'log_emb': log_emb,                 # SASRec用户表征（用于match_loss教师信号）
            'candidates_pos': candidates_pos,   # 候选物品prompt列表
            'interact': interact_embs,          # 历史物品嵌入列表（用于替换[HistoryEmb]）
            'candidate_embs': candidate_embs,   # 候选物品嵌入张量
        }

        samples['student_repr'] = log_emb
        samples['num_candidates'] = candidate_num

        # 步骤5: LLM前向传播和损失计算
        loss, llm_rec_loss, kd_loss = self.llm(samples, mode=0)

        # 输出训练日志
        if getattr(self.args, 'log_interval', 50) > 0 and (step % self.args.log_interval) == 0:
            match_weight = float(getattr(self.args, 'match_weight', 1.0))
            print("rec_loss epoch {}/{} iter {}/{}: {} | match_loss(raw): {} | match_weight: {} | match_loss(eff): {}".format(
                epoch, total_epoch, step, total_step, llm_rec_loss, kd_loss, match_weight, kd_loss * match_weight))
        
        # ✅ 恢复baseline的简单逻辑：直接backward + step（无梯度累积）
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        # 特殊硬件优化（如果使用神经网络处理器）
        if self.args.nn_parameter:
            htcore.mark_step()
    
    def split_into_batches(self, itemnum, m):
        """
        将物品ID范围分割成批次，用于批量处理

        Args:
            itemnum (int): 物品总数
            m (int): 每个批次的大小

        Returns:
            list: 批次列表，每个批次包含连续的物品ID
        """
        numbers = list(range(1, itemnum + 1))  # 物品ID从1开始（0是padding）
        batches = [numbers[i:i + m] for i in range(0, itemnum, m)]
        return batches

    def generate_batch(self, data):
        """
        推理阶段的批量生成方法 - 生成所有物品表示并进行推荐评估

        这个方法实现两个主要功能：
        1. 预计算所有物品的LLM表示（如果尚未计算）
        2. 为给定用户生成推荐并计算评估指标

        核心思想：
        - 首次调用时，批量生成所有物品的LLM表示并缓存
        - 后续调用直接使用缓存的表示进行快速推荐
        - 通过用户表示与物品表示的相似度进行排序推荐

        Args:
            data (tuple): 评估数据 (u, seq, pos, neg, rank, candi_set, files)
                - u: 用户ID列表
                - seq: 用户历史序列
                - pos: 正样本物品ID
                - neg: 负样本物品ID（当前未使用）
                - rank: 排序相关信息
                - candi_set: 候选物品集合
                - files: 文件相关信息（当前未使用）

        Returns:
            float: NDCG@10 评估指标
        """
        # 阶段1: 预计算所有物品的LLM表示（仅在首次调用时执行）
        self._ensure_item_embeddings_ready(desc="Building item embeddings")

        # 验证/推理临时关闭梯度检查点以提速
        prev_ckpt = getattr(self.llm, "_ckpt_enabled", False)
        if prev_ckpt:
            self.llm._set_gradient_checkpointing(False)
            
        try:
            # 阶段2: 为当前批次用户生成推荐
            u, seq, pos, neg, rank, candi_set, files = data
            original_seq = seq.copy()  # 保留原始序列（调试用）

            # 初始化批次数据容器
            text_input = []     # 用户历史的文本prompt
            interact_embs = []  # 用户历史物品的嵌入
            candidate = []      # 每个用户的候选物品ID列表

            # 无梯度模式进行推理
            with torch.no_grad():
                # 逐用户构造推荐数据
                for i in range(len(u)):
                    candidate_embs = []  # 当前用户的候选物品嵌入（未使用）

                    # 获取目标物品信息（用于评估）
                    target_item_id = pos[i]
                    target_item_title = self.find_item_text_single(target_item_id, title_flag=True, description_flag=False)

                    # 构造用户历史交互文本（最多10个最近交互）
                    items_i = seq[i][seq[i]>0]
                    interact_text, interact_ids = self.make_interact_text(items_i, 10, u[i])

                    # 生成候选物品集合（1个正样本 + 99个负样本）
                    candidate_num = 100
                    candidate_ids = self.make_candidate(seq[i][seq[i]>0], candidate_num, target_item_id, target_item_title, candi_set)
                    candidate.append(candidate_ids)

                    # 构造用户的完整prompt
                    input_text = ''
                    input_text += 'This user has made a series of purchases in the following order: '
                    input_text += interact_text  # 包含[HistoryEmb]占位符
                    input_text += ". Based on this sequence of purchases, generate user representation token:[UserOut]"

                    text_input.append(input_text)

                    # 准备历史物品嵌入用于替换[HistoryEmb]
                    interact_embs.append(self.item_emb_proj((self.get_item_emb(interact_ids))))
                    

                # 阶段3/4: 分块生成用户表示并评估，避免一次性占用全部显存
                max_input_length = self._resolve_positive_int(
                    env_name="LLMSREC_EVAL_MAX_LENGTH",
                    arg_name="eval_max_length",
                    default=getattr(self, 'eval_max_length', 512),
                )
                min_input_length = self._resolve_positive_int(
                    env_name="LLMSREC_EVAL_MIN_LENGTH",
                    arg_name="eval_min_length",
                    default=256,
                )
                min_input_length = min(min_input_length, max_input_length)
                base_chunk_size = max(1, getattr(self, 'inference_chunk_size', 8))
                total_users = len(text_input)
                chunk_start = 0
                adaptive_chunk_size = base_chunk_size
                oom_notice_printed = False

                while chunk_start < total_users:
                    current_chunk = min(adaptive_chunk_size, total_users - chunk_start)
                    current_max_len = max_input_length

                    while True:
                        end_idx = chunk_start + current_chunk
                        chunk_text = text_input[chunk_start:end_idx]
                        chunk_interact = interact_embs[chunk_start:end_idx]
                        chunk_candidates = candidate[chunk_start:end_idx]

                        try:
                            llm_tokens = self.llm.llm_tokenizer(
                                chunk_text,
                                return_tensors="pt",
                                padding="longest",
                                truncation=True,
                                max_length=current_max_len,
                            ).to(self.device)

                            inputs_embeds = self.llm.llm_model.get_input_embeddings()(llm_tokens['input_ids'])
                            inputs_embeds = self.llm.replace_out_token_all(
                                llm_tokens,
                                inputs_embeds,
                                token=['[UserOut]', '[HistoryEmb]'],
                                embs={'[HistoryEmb]': chunk_interact}
                            )

                            torch.cuda.empty_cache()

                            with torch.amp.autocast('cuda'):
                                with torch.no_grad():
                                    outputs = self.llm.llm_model.model(
                                        inputs_embeds=inputs_embeds,
                                        output_hidden_states=True
                                    )

                                    indx = self.llm.get_embeddings(llm_tokens, '[UserOut]')
                                    user_outputs = torch.cat([
                                        outputs.hidden_states[-1][i, indx[i]].mean(axis=0).unsqueeze(0)
                                        for i in range(len(indx))
                                    ])
                                    user_outputs = self.llm.pred_user(user_outputs)

                                    del outputs
                                    del inputs_embeds
                                    torch.cuda.empty_cache()

                                for local_idx, global_idx in enumerate(range(chunk_start, end_idx)):
                                    item_outputs = self.all_embs[np.array(chunk_candidates[local_idx]) - 1]
                                    if item_outputs.device != self.device:
                                        item_outputs = item_outputs.to(self.device, non_blocking=True)
                                    logits = torch.mm(item_outputs, user_outputs[local_idx].unsqueeze(0).T).squeeze(-1)
                                    logits = -1 * logits
                                    rank = logits.argsort().argsort()[0].item()

                                    if rank < 10:
                                        self.NDCG += 1 / np.log2(rank + 2)
                                        self.HT += 1
                                    if rank < 20:
                                        self.NDCG_20 += 1 / np.log2(rank + 2)
                                        self.HIT_20 += 1
                                    self.users += 1

                            chunk_start = end_idx
                            adaptive_chunk_size = current_chunk
                            self.inference_chunk_size = current_chunk
                            max_input_length = current_max_len
                            self.eval_max_length = current_max_len
                            break

                        except RuntimeError as e:
                            if 'CUDA out of memory' in str(e):
                                torch.cuda.empty_cache()
                                if current_chunk > 1:
                                    current_chunk = max(1, current_chunk // 2)
                                    if not oom_notice_printed:
                                        print(f"⚠️ 推理阶段检测到OOM，自动将chunk_size降至 {current_chunk}")
                                        oom_notice_printed = True
                                    continue
                                elif current_max_len > min_input_length:
                                    current_max_len = max(min_input_length, current_max_len - 64)
                                    if not oom_notice_printed:
                                        print(f"⚠️ 推理阶段检测到OOM，自动将max_length降至 {current_max_len}")
                                        oom_notice_printed = True
                                    continue
                            raise

            return self.NDCG
        finally:
            # 恢复原始梯度检查点状态
            if prev_ckpt:
                self.llm._set_gradient_checkpointing(True)

    def evaluate_student_batch(self, users, seq, pos, candidate_num=100, split='test'):
        """Evaluate the SASRec side with the legacy sampled-ranking protocol."""
        metrics = {'users': 0, 'HT10': 0.0, 'NDCG10': 0.0, 'HT20': 0.0, 'NDCG20': 0.0}

        if getattr(self, 'recsys', None) is None or getattr(self.recsys, 'model', None) is None:
            return metrics

        users = np.array(users, dtype=np.int64)
        seq = np.array(seq)
        pos = np.array(pos)
        for idx in range(len(users)):
            target_item = pos[idx]
            if isinstance(target_item, np.ndarray):
                target_item_id = int(target_item[-1])
            else:
                target_item_id = int(target_item)

            if target_item_id <= 0:
                continue

            seq_row = np.array(seq[idx], dtype=np.int64)
            history_ids = [int(item) for item in seq_row if int(item) > 0]
            item_idx = [target_item_id]

            if split == 'valid':
                rated = set(history_ids)
                rated.add(0)
                for _ in range(max(1, int(candidate_num))):
                    t = np.random.randint(1, self.item_num + 1)
                    while t in rated:
                        t = np.random.randint(1, self.item_num + 1)
                    item_idx.append(int(t))

                with torch.no_grad():
                    predictions = -self.recsys.model.predict(
                        np.array([users[idx]], dtype=np.int64),
                        np.array([seq_row], dtype=np.int64),
                        np.array(item_idx, dtype=np.int64),
                    )
                    if isinstance(predictions, torch.Tensor):
                        predictions = predictions.detach().cpu().numpy()

                predictions = predictions[0]
                rank = predictions.argsort().argsort()[0].item()
            else:
                num_candi = max(0, int(candidate_num) - 1)
                his = set(history_ids)
                his.add(target_item_id)
                his.add(0)

                items = list(set(range(1, self.item_num + 1)).difference(his))
                if len(items) > num_candi:
                    item_idx += random.sample(items, num_candi)
                else:
                    item_idx += items

                order = list(range(len(item_idx)))
                random.shuffle(order)

                with torch.no_grad():
                    predictions = -self.recsys.model.predict(
                        np.array([users[idx]], dtype=np.int64),
                        np.array([seq_row], dtype=np.int64),
                        np.array(item_idx, dtype=np.int64),
                    )
                    if isinstance(predictions, torch.Tensor):
                        predictions = predictions.detach().cpu().numpy()

                predictions = predictions[0]
                rank = predictions[order].argsort().argsort()[order.index(0)].item()

            metrics['users'] += 1
            if rank < 10:
                metrics['HT10'] += 1.0
                metrics['NDCG10'] += float(1.0 / np.log2(rank + 2))
            if rank < 20:
                metrics['HT20'] += 1.0
                metrics['NDCG20'] += float(1.0 / np.log2(rank + 2))

        return metrics
                
    def extract_emb(self, data):
        """
        用户表示提取方法 - 仅提取用户嵌入而不进行推荐评估

        这个方法专门用于提取用户的LLM表示，通常用于：
        1. 用户表示的可视化分析
        2. 用户聚类和相似度分析
        3. 下游任务的特征提取
        4. 模型表示质量的定性分析

        与 generate_batch 的区别：
        - generate_batch: 提取用户表示 + 进行推荐评估
        - extract_emb: 仅提取用户表示，保存到 self.extract_embs_list

        Args:
            data (tuple): 用户数据 (u, seq, pos, neg, original_seq, rank, files)
                - u: 用户ID列表
                - seq: 用户历史序列
                - pos: 正样本物品ID（当前未使用）
                - neg: 负样本物品ID（当前未使用）
                - original_seq: 原始序列（当前未使用）
                - rank: 排序信息（当前未使用）
                - files: 文件信息（当前未使用）

        Returns:
            int: 固定返回0（表示成功完成）

        副作用：
            - 将提取的用户表示添加到 self.extract_embs_list 中
            - 表示会被移动到CPU并分离梯度以节省内存
        """
        u, seq, pos, neg, original_seq, rank, files = data

        # 初始化数据容器
        text_input = []     # 用户历史的文本prompt
        interact_embs = []  # 用户历史物品的嵌入
        candidate = []      # 候选物品（当前未使用）

        # 无梯度模式进行推理
        with torch.no_grad():
            # 逐用户构造文本表示
            for i in range(len(u)):
                # 构造用户历史交互文本（最多10个最近交互）
                items_i = seq[i][seq[i]>0]
                interact_text, interact_ids = self.make_interact_text(items_i, 10, u[i])

                # 构造用户的完整prompt
                input_text = ''
                input_text += 'This user has made a series of purchases in the following order: '
                input_text += interact_text  # 包含[HistoryEmb]占位符
                input_text += ". Based on this sequence of purchases, generate user representation token:[UserOut]"

                text_input.append(input_text)

                # 准备历史物品嵌入用于替换[HistoryEmb]
                interact_embs.append(self.item_emb_proj((self.get_item_emb(interact_ids))))

            # 批量处理用户文本
            max_input_length = self._resolve_positive_int(
                env_name="LLMSREC_EVAL_MAX_LENGTH",
                arg_name="eval_max_length",
                default=512,
            )

            # 对用户历史文本进行tokenization
            llm_tokens = self.llm.llm_tokenizer(
                text_input,
                return_tensors="pt",
                padding="longest",
                truncation=True,
                max_length=max_input_length,
            ).to(self.device)

            # 获取初始词嵌入
            inputs_embeds = self.llm.llm_model.get_input_embeddings()(llm_tokens['input_ids'])

            # 用历史物品嵌入替换[HistoryEmb]，用CLS向量替换[UserOut]
            inputs_embeds = self.llm.replace_out_token_all(llm_tokens, inputs_embeds, token=['[UserOut]', '[HistoryEmb]'], embs={'[HistoryEmb]': interact_embs})

            # LLM前向传播生成用户表示
            with torch.amp.autocast('cuda'):
                outputs = self.llm.llm_model.model(
                    inputs_embeds=inputs_embeds,
                    output_hidden_states=True
                )

                # 提取[UserOut]位置的隐藏状态作为用户表示
                indx = self.llm.get_embeddings(llm_tokens, '[UserOut]')
                user_outputs = torch.cat([outputs.hidden_states[-1][i, indx[i]].mean(axis=0).unsqueeze(0) for i in range(len(indx))])

                # 通过预测头映射到最终用户表示空间
                user_outputs = self.llm.pred_user(user_outputs)

                # 保存用户表示到列表中（移动到CPU并分离梯度以节省GPU内存）
                self.extract_embs_list.append(user_outputs.detach().cpu())

        return 0


EchoRecModel = EchoRecSIModel

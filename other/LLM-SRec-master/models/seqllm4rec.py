import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, OPTForCausalLM, AutoModelForCausalLM
from peft import (
    prepare_model_for_kbit_training,
)


class llm4rec(nn.Module):
    """
    LLM 推荐模型基座类
    功能：加载 LLM，管理 Tokenizer，定义投影层(Projector)，执行向量注入，计算损失。
    """

    def __init__(
            self,
            device,
            llm_model="",
            max_output_txt_len=256,
            args=None
    ):
        super().__init__()
        self.device = device
        self.bce_criterion = torch.nn.BCEWithLogitsLoss()
        self.args = args
        local_model_path = ''
        if getattr(self.args, 'llm_path', None):
            local_model_path = self.args.llm_path.strip()
        if not local_model_path:
            local_model_path = os.environ.get('LLMREC_LLM_PATH', '').strip()

        # 1. 模型选择
        # 根据参数选择 Llama-3-8B 或 Llama-3.2-3B
        if local_model_path:
            if not os.path.exists(local_model_path):
                raise FileNotFoundError(f'Local LLM path does not exist: {local_model_path}')
            model_id = local_model_path
            local_files_only = True
        else:
            if llm_model == 'llama':
                model_id = "meta-llama/Meta-Llama-3-8B-Instruct"
            elif llm_model == 'llama-3b':
                model_id = "meta-llama/Llama-3.2-3B-Instruct"
            else:
                raise Exception(f'{llm_model} is not supported')
            local_files_only = False
        print()
        print("=========")
        print(f"Loading LLM from: {model_id}")
        # 2. 加载 LLM 模型
        # nn_parameter 控制是否量化加载 (8bit)
        if self.args.nn_parameter:
            self.llm_model = AutoModelForCausalLM.from_pretrained(
                model_id,
                device_map=self.device,
                torch_dtype=torch.float16,
                local_files_only=local_files_only,
            )
        else:
            # load_in_8bit=True: 节省显存，通常用于微调大模型
            self.llm_model = AutoModelForCausalLM.from_pretrained(
                model_id,
                device_map=self.device,
                torch_dtype=torch.float16,
                load_in_8bit=True,
                local_files_only=local_files_only,
            )
        self.llm_tokenizer = AutoTokenizer.from_pretrained(
            model_id,
            use_fast=False,
            local_files_only=local_files_only,
        )

        # 3. 加载 Tokenizer 并添加特殊 Token
        self.llm_tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        self.llm_tokenizer.add_special_tokens({'bos_token': '</s>'})
        self.llm_tokenizer.add_special_tokens({'eos_token': '</s>'})
        self.llm_tokenizer.add_special_tokens({'unk_token': '</s>'})
        # 【关键】添加业务相关的特殊 Token
        # [UserRep]: 用户表示占位符 (未使用)
        # [HistoryEmb]: 历史行为向量占位符 (会被 SASRec 向量替换)
        # [UserOut]: 用户输出占位符 (LLM 将在此位置输出最终用户向量)
        # [ItemOut]: 商品输出占位符 (LLM 将在此位置输出最终商品向量)
        self.llm_tokenizer.add_special_tokens(
            {'additional_special_tokens': ['[UserRep]', '[HistoryEmb]', '[UserOut]', '[ItemOut]']})
        self.llm_tokenizer.add_special_tokens({'cls_token': "[CLS]"})

        # 调整 Embedding 层大小以适应新 Token
        self.llm_model.resize_token_embeddings(len(self.llm_tokenizer))
        # 准备 k-bit 训练 (LoRA/QLoRA 的前置步骤)
        self.llm_model = prepare_model_for_kbit_training(self.llm_model)
        # 4. 冻结 LLM 参数
        # 除非 args.token 为 True (微调 Token Embedding)，否则冻结所有 LLM 参数
        for _, param in self.llm_model.named_parameters():
            if args.token:
                if 'token' in _:
                    param.requires_grad = True
                else:
                    param.requires_grad = False
            else:
                param.requires_grad = False
        # 5. 初始化可学习的 Prompt Token (CLS)
        # 如果不微调 Token Embedding，则创建一个可学习的向量作为 [UserOut]/[ItemOut] 的初始输入
        if not args.token:
            if args.nn_parameter:
                self.CLS = nn.Parameter(torch.normal(0, 1, size=(1, self.llm_model.config.hidden_size))).to(device)
                self.CLS_item = nn.Parameter(torch.normal(0, 1, size=(1, self.llm_model.config.hidden_size))).to(device)
            else:
                self.CLS = nn.Embedding(1, self.llm_model.config.hidden_size).to(device)
                # ... 初始化代码 ...
                nn.init.normal_(self.CLS.weight, mean=self.llm_model.model.embed_tokens.weight.mean(),
                                std=self.llm_model.model.embed_tokens.weight.std())
                self.CLS_item = nn.Embedding(1, self.llm_model.config.hidden_size).to(device)
                # ... 初始化代码 ...
                nn.init.normal_(self.CLS_item.weight, mean=self.llm_model.model.embed_tokens.weight.mean(),
                                std=self.llm_model.model.embed_tokens.weight.std())

        # 6. 定义输出投影层 (Prediction Heads)
        # pred_user: 将 LLM 的 Hidden State (e.g., 4096) 映射到 推荐空间 (128)
        self.pred_user = nn.Sequential(
            nn.Linear(self.llm_model.config.hidden_size, 2048),
            nn.LayerNorm(2048),
            nn.LeakyReLU(),
            nn.Linear(2048, 128)
        )
        # Xavier 初始化
        nn.init.xavier_normal_(self.pred_user[0].weight)
        nn.init.xavier_normal_(self.pred_user[3].weight)

        # pred_item: 同上，用于商品向量投影
        self.pred_item = nn.Sequential(
            nn.Linear(self.llm_model.config.hidden_size, 2048),
            nn.LayerNorm(2048),
            nn.LeakyReLU(),
            nn.Linear(2048, 128)
        )
        nn.init.xavier_normal_(self.pred_item[0].weight)
        nn.init.xavier_normal_(self.pred_item[3].weight)

        # 7. 蒸馏用的对齐层 (SASRec -> Latent Space)
        # pred_user_CF2: 将 SASRec 的 User Vector (64) 映射到 推荐空间 (128)
        # 只有映射到相同维度 (128)，才能计算 Distillation Loss
        self.pred_user_CF2 = nn.Sequential(
            nn.Linear(64, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Linear(128, 128)
        )
        nn.init.xavier_normal_(self.pred_user_CF2[0].weight)
        nn.init.xavier_normal_(self.pred_user_CF2[3].weight)
        # 似乎是冗余或备用的层
        self.cf_to_latent2 = nn.Sequential(
            nn.Linear(64, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Linear(128, 128)
        )
        nn.init.xavier_normal_(self.cf_to_latent2[0].weight)
        nn.init.xavier_normal_(self.cf_to_latent2[3].weight)

        self.mse = nn.MSELoss()

        self.max_output_txt_len = max_output_txt_len

    def forward_hidden(self, **kwargs):
        kwargs.setdefault("return_dict", True)
        kwargs.setdefault("output_hidden_states", True)
        backbone = getattr(self.llm_model, "model", None)
        if backbone is not None:
            return backbone(**kwargs)
        return self.llm_model(**kwargs)

    def info_nce_loss_batch(self, anchor, log_emb, temperature=0.07):
        """
        计算 InfoNCE 对比损失 (Batch 内负采样)
        输入:
            anchor: LLM 输出的用户向量 [Batch, Dim]
            log_emb: SASRec 输出的用户向量 [Batch, Dim]
        输出:
            loss: 标量
        """
        batch_size = anchor.shape[0]
        # L2 归一化
        anchor = F.normalize(anchor, p=2, dim=1)  # 1
        log_emb = F.normalize(log_emb, p=2, dim=1)  # 1
        # 计算相似度矩阵 (Batch x Batch)
        similarity_matrix = torch.matmul(anchor, log_emb.T) / temperature

        mask = torch.eye(batch_size, device=anchor.device).bool()

        # 正样本: 对角线上的元素 (同一个用户的 LLM 向量 vs SASRec 向量)
        pos_sim = similarity_matrix[mask].view(batch_size, 1)
        # 负样本: 非对角线元素 (Batch 内其他用户的 SASRec 向量)
        neg_sim = similarity_matrix[~mask].view(batch_size, -1)

        logits = torch.cat([pos_sim, neg_sim], dim=1)

        labels = torch.zeros(batch_size, dtype=torch.long, device=anchor.device)

        loss = F.cross_entropy(logits, labels)

        return loss

    def rec_loss(self, anchor, items):
        """
        计算推荐对比损失 (Recommendation Loss)
        逻辑: 用户向量应该和【正样本商品】相似，和【负样本商品】不相似。
        输入:
            anchor: LLM 用户向量 [Batch, Dim]
            items:  LLM 候选商品向量集 [Batch, Num_Candidates, Dim]
                    注意: items[:, 0, :] 是正样本，其他是负样本
        输出:
            loss: 标量 CrossEntropy
        """
        # 计算 anchor 和 items 的点积 (Similarity)
        # anchor.unsqueeze(2) -> [Batch, Dim, 1]
        # items -> [Batch, Num_Cand, Dim]
        # bmm 结果 -> [Batch, Num_Cand, 1] -> squeeze -> [Batch, Num_Cand]

        logits = torch.bmm(items.view(anchor.shape[0], -1, anchor.shape[1]), anchor.unsqueeze(2)).squeeze(2)
        # 标签全是 0，意味着我们希望第 0 个商品 (正样本) 的分数最高
        labels = torch.zeros(logits.size(0), dtype=torch.long).to(logits.device)

        loss = F.cross_entropy(logits, labels)

        return loss

    def uniformity(self, x, p=2):
        """
        均匀性损失 (Regularization)
        目的: 防止向量坍缩到同一个点，希望向量均匀分布在超球面上。
        """
        return torch.pdist(x, p=p).pow(2).mul(-p).exp().mean()

    def replace_out_token_all(self, llm_tokens, inputs_embeds, token=[], embs=None, ):
        """
        【核心魔法】向量注入 / Token 替换
        功能: 遍历文本序列，找到特殊 Token (如 [HistoryEmb])，把它的 Embedding 替换成外部传入的向量 (SASRec 向量)。

        输入:
            llm_tokens: Tokenizer 输出的结果 (包含 input_ids)
            inputs_embeds: LLM 原始的 Embedding 矩阵 [Batch, Seq_Len, Dim]
            token: 需要替换的特殊 Token 列表 (['[HistoryEmb]', '[UserOut]'])
            embs: 外部传入的向量字典 {'[HistoryEmb]': SASRec_Item_Embs}

        输出:
            inputs_embeds: 替换后的 Embedding 矩阵，准备送入 LLM
        """
        for t in token:
            # 1. 获取特殊 Token 的 ID
            token_id = self.llm_tokenizer(t, return_tensors="pt", add_special_tokens=False).input_ids.item()
            vectors = []
            # 2. 遍历 Batch 中的每一条数据
            for inx in range(len(llm_tokens["input_ids"])):
                # 找到特殊 Token 在序列中的位置索引
                idx_tensor = (llm_tokens["input_ids"][inx] == token_id).nonzero().view(-1)
                # 当前样本的 Embedding 序列
                user_vector = inputs_embeds[inx]
                # 情况 A: 替换历史行为向量 ([HistoryEmb])
                if 'Emb' in t:
                    ee = embs[t][inx]  # 拿到 SASRec 算出的对应向量序列
                    # 执行拼接: [前半段] + [SASRec向量] + [后半段]
                    for idx, item_emb in zip(idx_tensor, ee):
                        user_vector = torch.cat((user_vector[:idx], item_emb.unsqueeze(0), user_vector[idx + 1:]),
                                                dim=0)
                # 情况 B: 替换其他表示 ([UserRep])
                elif 'Rep' in t:
                    for idx in idx_tensor:
                        user_emb = embs[t][inx]
                        user_vector = torch.cat((user_vector[:idx], user_emb.unsqueeze(0), user_vector[idx + 1:]),
                                                dim=0)
                # 情况 C: 替换输出 Token ([UserOut] / [ItemOut])
                else:
                    if not self.args.token:
                        for idx in idx_tensor:
                            if 'UserOut' in t:
                                # 插入可学习的 CLS 向量作为初始状态
                                if self.args.nn_parameter:
                                    user_vector = torch.cat(
                                        (user_vector[:idx], self.CLS[torch.tensor([0]).to(self.device)],
                                         user_vector[idx + 1:]), dim=0)
                                else:
                                    user_vector = torch.cat(
                                        (user_vector[:idx], self.CLS(torch.tensor([0]).to(self.device)),
                                         user_vector[idx + 1:]), dim=0)
                            elif 'ItemOut' in t:
                                if self.args.nn_parameter:
                                    user_vector = torch.cat(
                                        (user_vector[:idx], self.CLS_item[torch.tensor([0]).to(self.device)],
                                         user_vector[idx + 1:]), dim=0)
                                else:
                                    user_vector = torch.cat(
                                        (user_vector[:idx], self.CLS_item(torch.tensor([0]).to(self.device)),
                                         user_vector[idx + 1:]), dim=0)

                vectors.append(user_vector.unsqueeze(0))
                # 重新堆叠回 Batch
            inputs_embeds = torch.cat(vectors)
        return inputs_embeds

    def replace_out_token_all_infer(self, llm_tokens, inputs_embeds, token=[], embs=None, user_act=False,
                                    item_act=False):
        """
        推理阶段的向量替换
        逻辑与 replace_out_token_all 类似，但针对推理时的输入格式做了微调。
        """
        # ... (代码逻辑与上面高度相似，省略重复注释) ...
        for t in token:
            token_id = self.llm_tokenizer(t, return_tensors="pt", add_special_tokens=False).input_ids.item()
            vectors = []
            for inx in range(len(llm_tokens["input_ids"])):
                idx_tensor = (llm_tokens["input_ids"][inx] == token_id).nonzero().view(-1)
                user_vector = inputs_embeds[inx]
                if 'Emb' in t:
                    ee = [embs[t][inx]]
                    # ee = embs[t][inx]
                    for idx, item_emb in zip(idx_tensor, ee):
                        user_vector = torch.cat((user_vector[:idx], item_emb.unsqueeze(0), user_vector[idx + 1:]),
                                                dim=0)

                elif 'Rep' in t:
                    for idx in idx_tensor:
                        user_emb = embs[t][inx]
                        user_vector = torch.cat((user_vector[:idx], user_emb.unsqueeze(0), user_vector[idx + 1:]),
                                                dim=0)
                else:
                    if not self.args.token:
                        for idx in idx_tensor:
                            if 'UserOut' in t:
                                if self.args.nn_parameter:
                                    user_vector = torch.cat(
                                        (user_vector[:idx], self.CLS[torch.tensor([0]).to(self.device)],
                                         user_vector[idx + 1:]), dim=0)
                                else:
                                    user_vector = torch.cat(
                                        (user_vector[:idx], self.CLS(torch.tensor([0]).to(self.device)),
                                         user_vector[idx + 1:]), dim=0)
                            elif 'ItemOut' in t:
                                if self.args.nn_parameter:
                                    user_vector = torch.cat(
                                        (user_vector[:idx], self.CLS_item[torch.tensor([0]).to(self.device)],
                                         user_vector[idx + 1:]), dim=0)
                                else:
                                    user_vector = torch.cat(
                                        (user_vector[:idx], self.CLS_item(torch.tensor([0]).to(self.device)),
                                         user_vector[idx + 1:]), dim=0)

                vectors.append(user_vector.unsqueeze(0))
            inputs_embeds = torch.cat(vectors)
        return inputs_embeds

    def get_embeddings(self, llm_tokens, token):
        """
        辅助函数：查找特定 Token 的索引
        输入: Token 序列
        输出: Token 在序列中的位置索引列表 (用于后续提取 Hidden State)
        """
        token_idx = []
        token_id = self.llm_tokenizer(token, return_tensors="pt", add_special_tokens=False).input_ids.item()
        for inx in range(len(llm_tokens['input_ids'])):
            idx_tensor = (llm_tokens['input_ids'][inx] == token_id).nonzero().view(-1)
            token_idx.append(idx_tensor)
        return token_idx

    def forward(self, samples, mode=0):
        if mode == 0:
            return self.train_mode0(samples)
        elif mode == 1:
            return self.train_mode1(samples)

    def train_mode0(self, samples):
        """
        核心训练逻辑 (Phase 2 Training Loop)

        输入: samples 字典
            - text_input: 用户历史 Prompt List
            - candidates_pos: 候选商品 Prompt List (包含正负样本)
            - interact: 用户的历史交互 SASRec 向量
            - candidate_embs: 候选商品的 SASRec 向量
            - log_emb: 用户的 SASRec 向量 (Teacher Label)

        输出:
            Total Loss, Rec Loss, Match Loss
        """
        max_input_length = 1024
        log_emb = samples['log_emb']  # Teacher 的用户向量
        # --- 1. 处理用户侧 (User Side) ---
        # 1.1 Tokenize 用户 Prompt
        llm_tokens = self.llm_tokenizer(
            samples['text_input'],
            return_tensors="pt",
            padding="longest",
            truncation=True,
            max_length=max_input_length,
        ).to(self.device)
        # 1.2 获取初始 Embeddings
        inputs_embeds = self.llm_model.get_input_embeddings()(llm_tokens['input_ids'])

        # 1.3 【注入】: 用 SASRec 的 interact 向量替换 [HistoryEmb]
        #              用 CLS 向量替换 [UserOut]
        # no user

        inputs_embeds = self.replace_out_token_all(llm_tokens, inputs_embeds, token=['[UserOut]', '[HistoryEmb]'],
                                                   embs={'[HistoryEmb]': samples['interact']})

        # --- 2. 处理商品侧 (Item Side) ---
        # 2.1 Tokenize 候选商品 Prompt
        candi_tokens = self.llm_tokenizer(
            samples['candidates_pos'],
            return_tensors="pt",
            padding="longest",
            truncation=True,
            max_length=max_input_length,
        ).to(self.device)
        # 2.2 获取初始 Embeddings
        candi_embeds = self.llm_model.get_input_embeddings()(candi_tokens['input_ids'])
        # 2.3 【注入】: 用 SASRec 的 candidate_embs 替换 [HistoryEmb]
        #              用 CLS_item 向量替换 [ItemOut]
        candi_embeds = self.replace_out_token_all_infer(candi_tokens, candi_embeds, token=['[ItemOut]', '[HistoryEmb]'],
                                                        embs={'[HistoryEmb]': samples['candidate_embs']})

        with torch.amp.autocast('cuda'):
            # --- 3. LLM 前向传播 (商品侧) ---
            # 让 LLM 读商品 Prompt，拿到 [ItemOut] 的向量
            candi_outputs = self.forward_hidden(
                inputs_embeds=candi_embeds,
                output_hidden_states=True
            )
            # 提取 [ItemOut] 位置的 Hidden State
            indx = self.get_embeddings(candi_tokens, '[ItemOut]')
            item_outputs = torch.cat(
                [candi_outputs.hidden_states[-1][i, indx[i]].mean(axis=0).unsqueeze(0) for i in range(len(indx))])
            # --- 4. LLM 前向传播 (用户侧) ---
            # 让 LLM 读用户 Prompt，拿到 [UserOut] 的向量
            outputs = self.forward_hidden(
                inputs_embeds=inputs_embeds,
                output_hidden_states=True
            )
            # 提取 [UserOut] 位置的 Hidden State
            indx = self.get_embeddings(llm_tokens, '[UserOut]')
            user_outputs = torch.cat(
                [outputs.hidden_states[-1][i, indx[i]].mean(axis=0).unsqueeze(0) for i in range(len(indx))])

        # --- 5. 投影 (Projection) ---
        # 将 LLM 的输出 (4096维) 映射到 推荐空间 (128维)
        user_outputs = self.pred_user(user_outputs)
        item_outputs = self.pred_item(item_outputs)
        # --- 6. 计算 Loss 1: Rec Loss (对比学习) ---
        # 比较 LLM(User) 和 LLM(Item) 的相似度
        rec_loss = self.rec_loss(user_outputs, item_outputs)
        # --- 7. 计算 Loss 2: Matching/Distillation Loss (蒸馏) ---
        # 目标: 让 LLM(User) 尽可能接近 Teacher SASRec(User)

        # 先把 Teacher 向量也映射到 128 维
        log_emb = self.pred_user_CF2(log_emb)

        # 归一化
        user_outputs = F.normalize(user_outputs, p=2, dim=1)  # 1
        log_emb = F.normalize(log_emb, p=2, dim=1)  # 1
        # MSE Loss
        match_loss = self.mse(user_outputs, log_emb)
        # 加上 Uniformity 正则化 (防止 Embedding 坍缩)
        match_loss += (self.uniformity(user_outputs) + self.uniformity(log_emb))

        # --- 8. 总 Loss ---
        loss = rec_loss + match_loss

        return loss, rec_loss.item(), match_loss.item()





import os
import random
import pickle

import torch
from torch.cuda.amp import autocast as autocast
import torch.nn as nn
import numpy as np

from models.recsys_model import *
from models.seqllm4rec import *
from sentence_transformers import SentenceTransformer
from datetime import datetime

from tqdm import trange, tqdm

try:
    import habana_frameworks.torch.core as htcore
except:
    0


# 这份代码是整个项目的**“心脏”**，定义了 LLM-SRec 的整体模型架构
# 它将 RecSys (SASRec) 和 LLM (Llama/Vicuna) 结合在一起，负责训练它们之间的“对齐”关系，并执行最终的推荐推理。
# 为了方便理解，我将这份代码拆解为四个核心模块：模型初始化、Prompt 构建、训练逻辑 (Phase 2) 和 推理逻辑 (Generate Batch)。


class llmrec_model(nn.Module):
    # 1. 模型初始化 (__init__)：搭建桥梁
    # 这一部分负责把 Teacher (SASRec) 和 Student/LLM 组装起来，并建立它们之间的连接通道。
    def __init__(self, args):
        super().__init__()
        rec_pre_trained_data = args.rec_pre_trained_data
        self.args = args
        self.device = args.device
        # 1. 加载文本字典 (ID -> Title/Description)
        with open(f'./SeqRec/data_{args.rec_pre_trained_data}/text_name_dict.json.gz', 'rb') as ft:
            self.text_name_dict = pickle.load(ft)
        # 2. 加载冻结的 Teacher 模型 (SASRec)
        # RecSys 类我们在上一段代码分析过，它只出特征，不更新参数
        self.recsys = RecSys(args.recsys, rec_pre_trained_data, self.device)

        self.item_num = self.recsys.item_num
        self.rec_sys_dim = self.recsys.hidden_units
        self.sbert_dim = 768

        self.mse = nn.MSELoss()
        self.l1 = nn.L1Loss()
        self.all_embs = None
        self.maxlen = args.maxlen
        self.NDCG = 0
        self.HIT = 0
        self.NDCG_20 = 0
        self.HIT_20 = 0

        self.rec_NDCG = 0
        self.rec_HIT = 0
        self.lan_NDCG = 0
        self.lan_HIT = 0
        self.num_user = 0
        self.yes = 0

        self.extract_embs_list = []

        self.bce_criterion = torch.nn.BCEWithLogitsLoss()
        # 3. 初始化 LLM (Student)
        self.llm = llm4rec(device=self.device, llm_model=args.llm, args=self.args)
        # 4.【核心组件】投影层 (Projector / Connector)
        # 这是一个 MLP (多层感知机)。
        # 它的任务是翻译：把 SASRec 的向量 (e.g., 64维) 翻译成 LLM 能懂的向量 (e.g., 4096维)
        # 关键点：item_emb_proj 是整个二阶段训练中主要更新的部分。它充当了 ID 特征空间到文本特征空间的转换器。
        self.item_emb_proj = nn.Sequential(
            nn.Linear(self.rec_sys_dim, self.llm.llm_model.config.hidden_size),
            nn.LayerNorm(self.llm.llm_model.config.hidden_size),
            nn.LeakyReLU(),
            nn.Linear(self.llm.llm_model.config.hidden_size, self.llm.llm_model.config.hidden_size)
        )
        nn.init.xavier_normal_(self.item_emb_proj[0].weight)
        nn.init.xavier_normal_(self.item_emb_proj[3].weight)

        self.users = 0.0
        self.NDCG = 0.0
        self.HT = 0.0

    def save_model(self, args, epoch2=None, best=False):
        out_dir = f'./models/{args.save_dir}/'
        if best:
            out_dir = out_dir[:-1] + 'best/'

        create_dir(out_dir)
        out_dir += f'{args.rec_pre_trained_data}_'

        out_dir += f'{args.llm}_{epoch2}_'
        if args.train:
            torch.save(self.item_emb_proj.state_dict(), out_dir + 'item_proj.pt')
            torch.save(self.llm.pred_user.state_dict(), out_dir + 'pred_user.pt')
            torch.save(self.llm.pred_item.state_dict(), out_dir + 'pred_item.pt')
            if not args.token:
                if args.nn_parameter:
                    torch.save(self.llm.CLS.data, out_dir + 'CLS.pt')
                    torch.save(self.llm.CLS_item.data, out_dir + 'CLS_item.pt')
                else:
                    torch.save(self.llm.CLS.state_dict(), out_dir + 'CLS.pt')
                    torch.save(self.llm.CLS_item.state_dict(), out_dir + 'CLS_item.pt')
            if args.token:
                torch.save(self.llm.llm_model.model.embed_tokens.state_dict(), out_dir + 'token.pt')

    def load_model(self, args, phase1_epoch=None, phase2_epoch=None):
        out_dir = f'./models/{args.save_dir}/{args.rec_pre_trained_data}_'

        out_dir += f'{args.llm}_{phase2_epoch}_'

        item_emb_proj = torch.load(out_dir + 'item_proj.pt', map_location=self.device)
        self.item_emb_proj.load_state_dict(item_emb_proj)
        del item_emb_proj

        pred_user = torch.load(out_dir + 'pred_user.pt', map_location=self.device)
        self.llm.pred_user.load_state_dict(pred_user)
        del pred_user

        pred_item = torch.load(out_dir + 'pred_item.pt', map_location=self.device)
        self.llm.pred_item.load_state_dict(pred_item)
        del pred_item

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

    def find_item_text(self, item, title_flag=True, description_flag=True):
        t = 'title'
        d = 'description'
        t_ = 'No Title'
        d_ = 'No Description'

        if title_flag and description_flag:
            return [f'"{self.text_name_dict[t].get(i, t_)}, {self.text_name_dict[d].get(i, d_)}"' for i in item]
        elif title_flag and not description_flag:
            return [f'"{self.text_name_dict[t].get(i, t_)}"' for i in item]
        elif not title_flag and description_flag:
            return [f'"{self.text_name_dict[d].get(i, d_)}"' for i in item]

    def find_item_time(self, item, user, title_flag=True, description_flag=True):
        t = 'title'
        d = 'description'
        t_ = 'No Title'
        d_ = 'No Description'

        l = [datetime.utcfromtimestamp(int(self.text_name_dict['time'][i][user]) / 1000) for i in item]
        return [l_.strftime('%Y-%m-%d') for l_ in l]

    def find_item_text_single(self, item, title_flag=True, description_flag=True):
        t = 'title'
        d = 'description'
        t_ = 'No Title'
        d_ = 'No Description'

        if title_flag and description_flag:
            return f'"{self.text_name_dict[t].get(item, t_)}, {self.text_name_dict[d].get(item, d_)}"'
        elif title_flag and not description_flag:
            return f'"{self.text_name_dict[t].get(item, t_)}"'
        elif not title_flag and description_flag:
            return f'"{self.text_name_dict[d].get(item, d_)}"'

    def get_item_emb(self, item_ids):
        with torch.no_grad():
            if self.args.nn_parameter:
                item_embs = self.recsys.model.item_emb[torch.LongTensor(item_ids).to(self.device)]
            else:
                item_embs = self.recsys.model.item_emb(torch.LongTensor(item_ids).to(self.device))

        return item_embs

    def forward(self, data, optimizer=None, batch_iter=None, mode='phase1'):
        if mode == 'phase2':
            self.pre_train_phase2(data, optimizer, batch_iter)
        if mode == 'generate_batch':
            self.generate_batch(data)
        if mode == 'extract':
            self.extract_emb(data)

    # 2. Prompt 构建 (Prompt Engineering)
    # 这几个函数 (make_interact_text, make_candidate_text) 负责把枯燥的 ID 序列变成 LLM 能读懂的“自然语言+特殊Token”的混合 Prompt。
    # 构建历史交互 Prompt (make_interact_text):
    '''输入: 用户历史交互序列 ID。
输出: 一段长文本。
逻辑:
遍历 ID，查出标题和时间。
拼接成句子：Item No.1, Time:..., Title: iPhone [HistoryEmb], Item No.2...
关键 Token [HistoryEmb]:
这是给 Projector 留的位置。
在后续步骤中，代码会用 item_emb_proj(sasrec_emb) 替换 掉这个文本 Token。
效果: LLM 读到这里时，既看到了文字 "iPhone"，也“感觉”到了来自 SASRec 的隐式特征。'''

    def make_interact_text(self, interact_ids, interact_max_num, user):
        interact_item_titles_ = self.find_item_text(interact_ids, title_flag=True, description_flag=False)
        times = self.find_item_time(interact_ids, user)
        interact_text = []
        count = 1

        if interact_max_num == 'all':
            times = self.find_item_time(interact_ids, user)
        else:
            times = self.find_item_time(interact_ids[-interact_max_num:], user)

        if interact_max_num == 'all':
            for title in interact_item_titles_:
                # 构建历史交互 Prompt (make_interact_text):
                interact_text.append(f'Item No.{count}, Time: {times[count - 1]}, ' + title + '[HistoryEmb]')
                # 生成结果示例：
                # "Item No.1, Time: 2023-01-01, iPhone 14 Pro [HistoryEmb], Item No.2, ..."
                # [HistoryEmb] 是什么？ 这是一个占位符。在送入 LLM 之前，代码会将 SASRec 计算出的物品向量（经过 item_emb_proj 投影后），替换掉这个 Token。
                # 效果：LLM 既看到了文本标题（语义信息），也看到了 SASRec 提供的 ID 向量（协同过滤信息）。
                count += 1
        else:
            for title in interact_item_titles_[-interact_max_num:]:
                interact_text.append(f'Item No.{count}, Time: {times[count - 1]}, ' + title + '[HistoryEmb]')

                count += 1
            interact_ids = interact_ids[-interact_max_num:]

        interact_text = ','.join(interact_text)
        return interact_text, interact_ids

    # 构建候选/目标 Prompt (make_candidate_text)
    '''函数: make_candidate_text(self, ...)
场景: 训练阶段 (Training)。
逻辑:
负采样: 随机挑 99 个用户没买过的商品 (Negative Items)。
正样本: 用户真实购买的商品 (Target Item)。
打包: 把它们都变成 Prompt。
The item title... [HistoryEmb], then generate item representation token:[ItemOut]
关键 Token [ItemOut]:
这是输出目标。
我们希望 LLM 在处理正样本时，[ItemOut] 输出的向量与用户的 [UserOut] 向量相似；处理负样本时则不相似。'''

    def make_candidate_text(self, interact_ids, candidate_num, target_item_id, target_item_title, candi_set=None,
                            task='ItemTask'):
        neg_item_id = []
        if candi_set == None:
            neg_item_id = []
            while len(neg_item_id) < 99:
                t = np.random.randint(1, self.item_num + 1)
                if not (t in interact_ids or t in neg_item_id):
                    neg_item_id.append(t)
        else:
            his = set(interact_ids)
            items = list(candi_set.difference(his))
            if len(items) > 99:
                neg_item_id = random.sample(items, 99)
            else:
                while len(neg_item_id) < 49:
                    t = np.random.randint(1, self.item_num + 1)
                    if not (t in interact_ids or t in neg_item_id):
                        neg_item_id.append(t)
        random.shuffle(neg_item_id)

        candidate_ids = [target_item_id]
        # 这里其实是序列item embedding
        candidate_text = [
            f'The item title and item embedding are as follows: ' + target_item_title + "[HistoryEmb], then generate item representation token:[ItemOut]"]

        for neg_candidate in neg_item_id[:candidate_num - 1]:
            # 构建候选/目标 Prompt (make_candidate_text):
            candidate_text.append(
                f'The item title and item embedding are as follows: ' + self.find_item_text_single(neg_candidate,
                                                                                                   title_flag=True,
                                                                                                   description_flag=False) + "[HistoryEmb], then generate item representation token:[ItemOut]")
            # [ItemOut]: 这是一个输出占位符。模型需要预测这个位置的 Embedding，使其接近真实的物品表示。
            candidate_ids.append(neg_candidate)

        return candidate_text, candidate_ids

    ''' 总结图示
假设 candidate_num = 2 (1 正 1 负)，返回的数据结构如下：
Python
(
  # 1. candidate_text
  [
    "The item title... iPhone ... [HistoryEmb] ... [ItemOut]",  # 正样本 Prompt
    "The item title... Nokia ... [HistoryEmb] ... [ItemOut]"    # 负样本 Prompt
  ],

  # 2. candidate_ids
  [
    1001,  # iPhone 的 ID
    205    # Nokia 的 ID
  ]
)
它们就像是“连体婴”，一个是给 LLM 看的（文本），一个是给 Projector 用的（ID），必须配合使用才能生成完整的输入 Embeddings'''

    # 函数: make_candidate(self, ...)
    # 场景: 推理阶段 (Inference)。
    # 区别: 它只返回 candidate_ids (一堆 ID 数字)，不返回文本 Prompt。因为推理阶段我们会用矩阵乘法加速，不需要一个个构建复杂的文本 Prompt（除非为了可视化）。
    def make_candidate(self, interact_ids, candidate_num, target_item_id, target_item_title, candi_set=None,
                       task='ItemTask'):
        neg_item_id = []
        neg_item_id = []
        while len(neg_item_id) < 99:
            t = np.random.randint(1, self.item_num + 1)
            if not (t in interact_ids or t in neg_item_id):
                neg_item_id.append(t)

        random.shuffle(neg_item_id)

        candidate_ids = [target_item_id]

        candidate_ids = candidate_ids + neg_item_id[:candidate_num - 1]

        return candidate_ids

    def pre_train_phase2(self, data, optimizer, batch_iter):
        epoch, total_epoch, step, total_step = batch_iter
        optimizer.zero_grad()
        u, seq, pos, neg = data

        original_seq = seq.copy()

        mean_loss = 0

        text_input = []
        candidates_pos = []
        candidates_neg = []
        interact_embs = []
        candidate_embs_pos = []
        candidate_embs_neg = []
        candidate_embs = []

        loss_rm_mode1 = 0
        loss_rm_mode2 = 0
        # 1. 拿到 SASRec 的“正确答案” (Teacher Output)
        # log_emb 是 SASRec 算出的用户向量
        with torch.no_grad():
            log_emb = self.recsys.model(u, seq, pos, neg, mode='log_only')
        # 2. 构建 Prompt 和 Embedding
        # 循环 Batch 中的每个用户
        for i in range(len(u)):
            target_item_id = pos[i][-1]
            target_item_title = self.find_item_text_single(target_item_id, title_flag=True, description_flag=False)

            interact_text, interact_ids = self.make_interact_text(seq[i][seq[i] > 0], 10, u[i])
            candidate_num = 4
            candidate_text, candidate_ids = self.make_candidate_text(seq[i][seq[i] > 0], candidate_num, target_item_id,
                                                                     target_item_title, task='RecTask')

            # no user
            input_text = ''

            # 造句： "This user has made ... purchases ... [UserOut]"
            input_text += 'This user has made a series of purchases in the following order: '

            input_text += interact_text

            input_text += ". Based on this sequence of purchases, generate user representation token:[UserOut]"

            text_input.append(input_text)

            candidates_pos += candidate_text

            # 准备要注入的向量：把 SASRec 的 Item 向量投影到 LLM 维度
            interact_embs.append(self.item_emb_proj((self.get_item_emb(interact_ids))))
            candidate_embs_pos.append(self.item_emb_proj((self.get_item_emb([candidate_ids]))).squeeze(0))

        candidate_embs = torch.cat(candidate_embs_pos)

        samples = {'text_input': text_input, 'log_emb': log_emb, 'candidates_pos': candidates_pos,
                   'interact': interact_embs, 'candidate_embs': candidate_embs, }
        # 3. 替换 Token 并计算 Loss
        # llm.replace_out_token_all 会把 Prompt 里的 [HistoryEmb] 替换成上面的 interact_embs
        loss, rec_loss, match_loss = self.llm(samples, mode=0)

        loss.backward()
        if self.args.nn_parameter:
            htcore.mark_step()
        optimizer.step()
        if self.args.nn_parameter:
            htcore.mark_step()
        should_log = (not self.args.multi_gpu) or os.environ.get("ID", "0") == "0"
        if should_log and (step % 20 == 0 or step == total_step - 1):
            print(self.args.save_dir, self.args.rec_pre_trained_data, self.args.llm)
            print("LLMRec model loss in epoch {}/{} iteration {}/{}: {}".format(epoch, total_epoch, step, total_step,
                                                                                rec_loss))
            print("LLMRec model Matching loss in epoch {}/{} iteration {}/{}: {}".format(epoch, total_epoch, step,
                                                                                         total_step, match_loss))
        # 目的：训练 item_emb_proj 和 [UserOut] 对应的输出头。
        # Loss 含义：
        # 让 LLM 输出的 [UserOut] 向量，去尽可能拟合/对齐 SASRec 输出的 log_emb。
        # 或者让 LLM 输出的 [UserOut] 能准确预测下一个物品。

    def split_into_batches(self, itemnum, m):
        numbers = list(range(1, itemnum + 1))

        batches = [numbers[i:i + m] for i in range(0, itemnum, m)]

        return batches

    def generate_batch(self, data):
        # 这是测试阶段的代码。因为 LLM 推理很慢，直接让 LLM 对每个用户的几千个候选商品打分是不现实的。
        # 代码采用了一种**“缓存+矩阵乘法”**的加速策略。

        # Step A: 预计算所有物品的向量 (Cache Item Embs)
        if self.all_embs == None:
            # 遍历所有商品 (batches)
            batch_ = 128
            if self.args.llm in ['llama', 'llama-3b']:
                batch_ = 64
            if self.args.rec_pre_trained_data in ['Electronics', 'Books', 'Industrial_and_Scientific']:
                batch_ = 64
                if self.args.llm in ['llama', 'llama-3b']:
                    batch_ = 32
            batch_ = int(os.environ.get("LLMSREC_EVAL_ITEM_BATCH", batch_))
            batches = self.split_into_batches(self.item_num, batch_)  # 128
            self.all_embs = []
            max_input_length = 1024
            for bat in tqdm(batches):
                # 构造 Item Prompt: "... [ItemOut]"
                # 跑一次 LLM
                # 拿到 [ItemOut] 位置的向量
                candidate_text = []
                candidate_ids = []
                candidate_embs = []
                for neg_candidate in bat:
                    candidate_text.append(
                        'The item title and item embedding are as follows: ' + self.find_item_text_single(neg_candidate,
                                                                                                          title_flag=True,
                                                                                                          description_flag=False) + "[HistoryEmb], then generate item representation token:[ItemOut]")

                    candidate_ids.append(neg_candidate)
                with torch.no_grad():
                    candi_tokens = self.llm.llm_tokenizer(
                        candidate_text,
                        return_tensors="pt",
                        padding="longest",
                        truncation=True,
                        max_length=max_input_length,
                    ).to(self.device)
                    candidate_embs.append(self.item_emb_proj((self.get_item_emb(candidate_ids))))

                    candi_embeds = self.llm.llm_model.get_input_embeddings()(candi_tokens['input_ids'])
                    candi_embeds = self.llm.replace_out_token_all_infer(candi_tokens, candi_embeds,
                                                                        token=['[ItemOut]', '[HistoryEmb]'],
                                                                        embs={'[HistoryEmb]': candidate_embs[0]})

                    with torch.amp.autocast('cuda'):
                        candi_outputs = self.llm.forward_hidden(
                            inputs_embeds=candi_embeds,
                            output_hidden_states=True
                        )

                        indx = self.llm.get_embeddings(candi_tokens, '[ItemOut]')
                        item_outputs = torch.cat(
                            [candi_outputs.hidden_states[-1][i, indx[i]].mean(axis=0).unsqueeze(0) for i in
                             range(len(indx))])

                        item_outputs = self.llm.pred_item(item_outputs)
                    # 存起来，变成一个大矩阵 [Item_Num, Dim]
                    self.all_embs.append(item_outputs)
                    del candi_outputs
                    del item_outputs
            self.all_embs = torch.cat(self.all_embs)

        u, seq, pos, neg, rank, candi_set, files = data
        original_seq = seq.copy()

        text_input = []
        interact_embs = []
        candidate = []
        with torch.no_grad():
            # Step B: 计算用户向量并打分
            for i in range(len(u)):
                # 1. 跑 LLM 计算用户向量
                # 输入用户的历史 Prompt
                # 拿到 [UserOut] 位置的向量 -> user_outputs
                candidate_embs = []
                target_item_id = pos[i]
                target_item_title = self.find_item_text_single(target_item_id, title_flag=True, description_flag=False)

                interact_text, interact_ids = self.make_interact_text(seq[i][seq[i] > 0], 10, u[i])

                candidate_num = 100
                candidate_ids = self.make_candidate(seq[i][seq[i] > 0], candidate_num, target_item_id,
                                                    target_item_title, candi_set)

                candidate.append(candidate_ids)

                # no user
                input_text = ''

                input_text += 'This user has made a series of purchases in the following order: '

                input_text += interact_text

                input_text += ". Based on this sequence of purchases, generate user representation token:[UserOut]"

                text_input.append(input_text)

                interact_embs.append(self.item_emb_proj((self.get_item_emb(interact_ids))))

            max_input_length = 1024

            llm_tokens = self.llm.llm_tokenizer(
                text_input,
                return_tensors="pt",
                padding="longest",
                truncation=True,
                max_length=max_input_length,
            ).to(self.device)

            inputs_embeds = self.llm.llm_model.get_input_embeddings()(llm_tokens['input_ids'])

            # no user
            inputs_embeds = self.llm.replace_out_token_all(llm_tokens, inputs_embeds,
                                                           token=['[UserOut]', '[HistoryEmb]'],
                                                           embs={'[HistoryEmb]': interact_embs})

            with torch.cuda.amp.autocast():
                outputs = self.llm.forward_hidden(
                    inputs_embeds=inputs_embeds,

                    output_hidden_states=True
                )

                indx = self.llm.get_embeddings(llm_tokens, '[UserOut]')
                user_outputs = torch.cat(
                    [outputs.hidden_states[-1][i, indx[i]].mean(axis=0).unsqueeze(0) for i in range(len(indx))])
                user_outputs = self.llm.pred_user(user_outputs)

                for i in range(len(candidate)):

                    item_outputs = self.all_embs[np.array(candidate[i]) - 1]
                    # 2. 极速打分 (矩阵乘法)
                    # 用 用户向量 (1, Dim) 乘以 所有物品向量矩阵 (Item_Num, Dim)
                    # 得到 logits (所有物品的得分)
                    logits = torch.mm(item_outputs, user_outputs[i].unsqueeze(0).T).squeeze(-1)

                    logits = -1 * logits
                    # 3. 排序算指标 (NDCG)
                    rank = logits.argsort().argsort()[0].item()

                    if rank < 10:
                        self.NDCG += 1 / np.log2(rank + 2)
                        self.HT += 1
                    if rank < 20:
                        self.NDCG_20 += 1 / np.log2(rank + 2)
                        self.HIT_20 += 1
                    self.users += 1
        return self.NDCG

    # 为了把经过 LLM “思考”后的用户向量（User Embeddings）提取出来并保存。
    def extract_emb(self, data):
        u, seq, pos, neg, original_seq, rank, files = data

        text_input = []
        interact_embs = []
        candidate = []
        with torch.no_grad():
            for i in range(len(u)):
                # 第一步：构建 Prompt (输入),
                # 它把用户的历史行为构造成一句完整的 Prompt。
                # 重点在于最后的 [UserOut] Token，这是我们希望 LLM 把对用户的理解“浓缩”进去的地方。
                interact_text, interact_ids = self.make_interact_text(seq[i][seq[i] > 0], 10, u[i])

                input_text = ''

                input_text += 'This user has made a series of purchases in the following order: '

                input_text += interact_text

                input_text += ". Based on this sequence of purchases, generate user representation token:[UserOut]"

                text_input.append(input_text)
                # 第二步：注入 SASRec 向量 (融合)
                interact_embs.append(self.item_emb_proj((self.get_item_emb(interact_ids))))

            max_input_length = 1024

            llm_tokens = self.llm.llm_tokenizer(
                text_input,
                return_tensors="pt",
                padding="longest",
                truncation=True,
                max_length=max_input_length,
            ).to(self.device)

            inputs_embeds = self.llm.llm_model.get_input_embeddings()(llm_tokens['input_ids'])
            # 第二步：注入 SASRec 向量 (融合)
            # 把 SASRec 算出的 Item ID 向量，通过 Projector 投影后，替换掉 Prompt 里的 [HistoryEmb] 占位符。
            # 这样 LLM 就同时拥有了文本语义信息和协同过滤信息。
            inputs_embeds = self.llm.replace_out_token_all(llm_tokens, inputs_embeds,
                                                           token=['[UserOut]', '[HistoryEmb]'],
                                                           embs={'[HistoryEmb]': interact_embs})

            with torch.cuda.amp.autocast():
                # 第三步：LLM 前向传播 (推理)
                # 让 LLM 读完这句话。
                # 注意 output_hidden_states=True，我们要拿中间层的向量，而不是生成的文本。
                outputs = self.llm.forward_hidden(
                    inputs_embeds=inputs_embeds,

                    output_hidden_states=True
                )
                # 第四步：提取关键向量 (抽取)
                # 找到 [UserOut] Token 在序列中的位置
                indx = self.llm.get_embeddings(llm_tokens, '[UserOut]')
                # 把那个位置对应的 Hidden State (向量) 拿出来
                user_outputs = torch.cat(
                    [outputs.hidden_states[-1][i, indx[i]].mean(axis=0).unsqueeze(0) for i in range(len(indx))])
                # 过一层线性层 (Pred_User) 得到最终的用户表示
                user_outputs = self.llm.pred_user(user_outputs)

                self.extract_embs_list.append(user_outputs.detach().cpu())

        return 0
# llmrec_model 是一个混合专家 (Mixture of Experts) 风格的实现：

# 它利用 SASRec 提取精准的协同过滤信号（User/Item ID 向量）。
# 它利用 LLM 处理文本语义和上下文推理。
# 它通过 Projector (MLP) 和 特殊 Token ([HistoryEmb]) 将两者融合。
# 在推理时，它巧妙地分别计算 User Embedding 和 Item Embedding，最后用点积完成推荐，避开了 LLM 逐个生成文本的高昂代价。

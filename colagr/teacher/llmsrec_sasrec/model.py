import numpy as np
import torch
import torch.nn as nn

#这是 Transformer 架构中的 Feed Forward Network (FFN) 层。
#作用：增加模型的非线性拟合能力。
#特点：这里使用了 Conv1d (卷积核大小为 1) 来代替全连接层 (Linear)。
# 在数学上，kernel_size=1 的一维卷积等价于对序列中每个位置独立做全连接层。这是 SASRec 原始论文代码的一种经典实现方式。
class PointWiseFeedForward(torch.nn.Module):
    def __init__(self, hidden_units, dropout_rate):

        super(PointWiseFeedForward, self).__init__()

        self.conv1 = torch.nn.Conv1d(hidden_units, hidden_units, kernel_size=1)
        self.dropout1 = torch.nn.Dropout(p=dropout_rate)
        self.relu = torch.nn.ReLU()
        self.conv2 = torch.nn.Conv1d(hidden_units, hidden_units, kernel_size=1)
        self.dropout2 = torch.nn.Dropout(p=dropout_rate)

    def forward(self, inputs):
        outputs = self.dropout2(self.conv2(self.relu(self.dropout1(self.conv1(inputs.transpose(-1, -2))))))
        outputs = outputs.transpose(-1, -2)
        outputs += inputs
        return outputs
    # SASRec 模型主体,这是整个序列推荐模型的主体。
class SASRec(torch.nn.Module):
    def __init__(self, user_num, item_num, args):
        super(SASRec, self).__init__()

        self.kwargs = {'user_num': user_num, 'item_num':item_num, 'args':args}
        self.user_num = user_num
        self.item_num = item_num
        self.dev = args.device
        self.embedding_dim = args.hidden_units
        self.nn_parameter = args.nn_parameter
        # 1. 定义 Embedding 层
        # item_emb: 商品 ID 对应的向量
        # pos_emb:  位置 ID (1, 2, ..., maxlen) 对应的向量
        if self.nn_parameter:
            self.item_emb = nn.Parameter(torch.normal(0,1, size = (self.item_num+1, args.hidden_units)))
            self.pos_emb = nn.Parameter(torch.normal(0,1, size=(args.maxlen, args.hidden_units)))
        else:
            self.item_emb = torch.nn.Embedding(self.item_num+1, args.hidden_units, padding_idx=0)
            self.item_emb.weight.data.normal_(0.0,1)
            self.pos_emb = torch.nn.Embedding(args.maxlen, args.hidden_units)

        self.emb_dropout = torch.nn.Dropout(p=args.dropout_rate)
        # 2. 定义 Transformer Block 组件
        # 包含 LayerNorm, MultiheadAttention (自注意力), FeedForward
        self.attention_layernorms = torch.nn.ModuleList()
        self.attention_layers = torch.nn.ModuleList()
        self.forward_layernorms = torch.nn.ModuleList()
        self.forward_layers = torch.nn.ModuleList()

        self.last_layernorm = torch.nn.LayerNorm(args.hidden_units, eps=1e-8)

        self.args =args
        
        
        for _ in range(args.num_blocks):
            new_attn_layernorm = torch.nn.LayerNorm(args.hidden_units, eps=1e-8)
            self.attention_layernorms.append(new_attn_layernorm)

            new_attn_layer =  torch.nn.MultiheadAttention(args.hidden_units,
                                                            args.num_heads,
                                                            args.dropout_rate)
            self.attention_layers.append(new_attn_layer)

            new_fwd_layernorm = torch.nn.LayerNorm(args.hidden_units, eps=1e-8)
            self.forward_layernorms.append(new_fwd_layernorm)

            new_fwd_layer = PointWiseFeedForward(args.hidden_units, args.dropout_rate)
            self.forward_layers.append(new_fwd_layer)
    # 核心组件：标准的 Transformer Encoder 结构（Attention + FFN + Norm）
    
    #B. log2feats (最核心的函数：序列编码)
    #这个函数负责把用户交互的历史 ID 序列，转换成高维的特征向量。
    def log2feats(self, log_seqs):
        # 1. 获取 Embedding
        # 商品 Embedding
        if self.nn_parameter:
            seqs = self.item_emb[torch.LongTensor(log_seqs).to(self.dev)]
            seqs *= self.embedding_dim **0.5
        else:
            seqs = self.item_emb(torch.LongTensor(log_seqs).to(self.dev))
            seqs *= self.item_emb.embedding_dim ** 0.5
        
        positions = np.tile(np.array(range(log_seqs.shape[1])), [log_seqs.shape[0], 1])
        
        #nn.Embedding
        # 加上 位置 Embedding (SASRec 知道顺序的关键)
        if self.nn_parameter:
            seqs += self.pos_emb[torch.LongTensor(positions).to(self.dev)]
        else:
            seqs += self.pos_emb(torch.LongTensor(positions).to(self.dev))
    # 2. Mask 处理 (两个 Mask)
    # timeline_mask: 把填充符 (Padding ID = 0) 的位置遮掉
    # attention_mask: 因果 Mask (Causal Mask)，确保预测 t 时刻只能看 t 及其之前的数据，不能偷看未来
        seqs = self.emb_dropout(seqs)

        timeline_mask = torch.BoolTensor(log_seqs == 0).to(self.dev)
        seqs *= ~timeline_mask.unsqueeze(-1)

        tl = seqs.shape[1]
        attention_mask = ~torch.tril(torch.ones((tl, tl), dtype=torch.bool, device=self.dev))
    # 3. Transformer 层堆叠
        for i in range(len(self.attention_layers)):
            # 注意：PyTorch 的 MultiheadAttention 默认输入是 (L, N, E)，所以这里做了 transpose
            seqs = torch.transpose(seqs, 0, 1)
            # Self-Attention + Residual
            Q = self.attention_layernorms[i](seqs)
            mha_outputs, _ = self.attention_layers[i](Q, seqs, seqs, 
                                            attn_mask=attention_mask)

            seqs = Q + mha_outputs
            seqs = torch.transpose(seqs, 0, 1)
# FFN + Residual# ...
            seqs = self.forward_layernorms[i](seqs)
            seqs = self.forward_layers[i](seqs)
            seqs *=  ~timeline_mask.unsqueeze(-1)

        log_feats = self.last_layernorm(seqs)
        return log_feats# 4. 输出特征
    #输入: log_seqs (Batch, Max_Len)，例如 [[10, 5, 3, 0...], ...]
    #输出: log_feats (Batch, Max_Len, Hidden_Dim)。这是序列中每一个位置经过处理后的深层特征。
    #C. forward (训练与特征提取)
    def forward(self, user_ids, log_seqs, pos_seqs, neg_seqs, mode='default'):
        # 先把序列编码成特征
        log_feats = self.log2feats(log_seqs)
        # --- LLM-SRec 的关键接口 ---
        if mode == 'log_only':
            # 如果模式是 'log_only'，只取序列的【最后一个时间步】的向量
            log_feats = log_feats[:, -1, :]
            return log_feats#核心产出
        
        #nn.Embedding
        if self.nn_parameter:
            pos_embs = self.item_emb[torch.LongTensor(pos_seqs).to(self.dev)]
            neg_embs = self.item_emb[torch.LongTensor(neg_seqs).to(self.dev)]
        else:
            pos_embs = self.item_emb(torch.LongTensor(pos_seqs).to(self.dev))
            neg_embs = self.item_emb(torch.LongTensor(neg_seqs).to(self.dev))
    #重点关注 mode == 'log_only'：
    #这是为了 LLM-SRec 专门设计的接口。
    #在第二阶段训练（蒸馏）时，LLM 需要读入用户的特征。程序会调用这个模式，拿到 SASRec 对用户历史的最终浓缩向量 (log_feats[:, -1, :])。
    #这个向量随后会被送入 Projector (MLP)，然后喂给 LLM。

    # --- 正常的 SASRec 训练逻辑 ---
    # 计算 Positive Item (真实购买) 和 Negative Item (负采样) 的分数    
        pos_logits = (log_feats * pos_embs).sum(dim=-1)
        neg_logits = (log_feats * neg_embs).sum(dim=-1)

        if mode == 'item':
            return log_feats.reshape(-1, log_feats.shape[2]), pos_embs.reshape(-1, log_feats.shape[2]), neg_embs.reshape(-1, log_feats.shape[2])
        else:
            return pos_logits, neg_logits

    def predict(self, user_ids, log_seqs, item_indices):
        # 1. 编码历史序列
        log_feats = self.log2feats(log_seqs)
        # 2. 取最后一个时间步的状态 (代表用户当前的兴趣)       LLM-SRec 到这边就停了
        final_feat = log_feats[:, -1, :]

        #nn.Embedding
        #这一步是为了准备矩阵乘法。 它把**“一堆 ID 数字”变成了“一堆向量”**，这样下一行代码 logits = item_embs.matmul(...) 才能拿着这些向量去和用户的兴趣向量做点积计算分数
        if self.nn_parameter:
            item_embs = self.item_emb[torch.LongTensor(item_indices).to(self.dev)]
        else:
            item_embs = self.item_emb(torch.LongTensor(item_indices).to(self.dev))
        # 3. 计算与候选商品 (item_indices) 的相似度 (点积)
        logits = item_embs.matmul(final_feat.unsqueeze(-1)).squeeze(-1)

        return logits
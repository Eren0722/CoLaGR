import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


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


class SASRec(torch.nn.Module):
    def __init__(self, user_num, item_num, args):
        super(SASRec, self).__init__()

        self.kwargs = {'user_num': user_num, 'item_num': item_num, 'args': args}
        self.user_num = user_num
        self.item_num = item_num
        self.dev = args.device
        self.args = args
        self.embedding_dim = args.hidden_units
        self.hidden_units = args.hidden_units
        self.maxlen = args.maxlen
        self.nn_parameter = args.nn_parameter
        self.rec_objective = str(getattr(args, 'rec_objective', 'bce')).lower()

        if self.nn_parameter:
            self.item_emb = nn.Parameter(torch.normal(0, 1, size=(self.item_num + 1, args.hidden_units)))
            self.pos_emb = nn.Parameter(torch.normal(0, 1, size=(args.maxlen, args.hidden_units)))
        else:
            self.item_emb = torch.nn.Embedding(self.item_num + 1, args.hidden_units, padding_idx=0)
            self.item_emb.weight.data.normal_(0.0, 1)
            self.pos_emb = torch.nn.Embedding(args.maxlen, args.hidden_units)
        self.emb_dropout = torch.nn.Dropout(p=args.dropout_rate)
        self.attention_layernorms = torch.nn.ModuleList()
        self.attention_layers = torch.nn.ModuleList()
        self.forward_layernorms = torch.nn.ModuleList()
        self.forward_layers = torch.nn.ModuleList()
        self.last_layernorm = torch.nn.LayerNorm(args.hidden_units, eps=1e-8)

        for _ in range(args.num_blocks):
            self.attention_layernorms.append(torch.nn.LayerNorm(args.hidden_units, eps=1e-8))
            self.attention_layers.append(
                torch.nn.MultiheadAttention(args.hidden_units, args.num_heads, args.dropout_rate)
            )
            self.forward_layernorms.append(torch.nn.LayerNorm(args.hidden_units, eps=1e-8))
            self.forward_layers.append(PointWiseFeedForward(args.hidden_units, args.dropout_rate))

    @property
    def item_embedding(self):
        return self.item_emb

    def _as_long_tensor(self, seqs):
        if torch.is_tensor(seqs):
            return seqs.to(self.dev, dtype=torch.long)
        return torch.as_tensor(seqs, dtype=torch.long, device=self.dev)

    def _lookup_item_emb(self, item_ids):
        if self.nn_parameter:
            return self.item_emb[item_ids]
        return self.item_emb(item_ids)

    def _real_item_embedding_matrix(self):
        if self.nn_parameter:
            return self.item_emb[1:self.item_num + 1]
        return self.item_emb.weight[1:self.item_num + 1]

    def _mean_pool(self, feats, item_seq):
        if torch.is_tensor(item_seq):
            seq_tensor = item_seq.to(feats.device, dtype=torch.long)
        else:
            seq_tensor = torch.as_tensor(item_seq, dtype=torch.long, device=feats.device)
        mask = seq_tensor.ne(0).unsqueeze(-1)
        denom = mask.sum(dim=1).clamp(min=1)
        return (feats * mask).sum(dim=1) / denom

    def _encode(self, log_seqs):
        item_seq = self._as_long_tensor(log_seqs)
        if self.nn_parameter:
            seqs = self.item_emb[item_seq]
            seqs *= self.embedding_dim ** 0.5
        else:
            seqs = self.item_emb(item_seq)
            seqs *= self.item_emb.embedding_dim ** 0.5

        positions = np.tile(np.array(range(item_seq.shape[1])), [item_seq.shape[0], 1])
        pos_tensor = torch.LongTensor(positions).to(self.dev)
        if self.nn_parameter:
            seqs += self.pos_emb[pos_tensor]
        else:
            seqs += self.pos_emb(pos_tensor)

        seqs = self.emb_dropout(seqs)
        timeline_mask = item_seq.eq(0)
        seqs *= ~timeline_mask.unsqueeze(-1)

        tl = seqs.shape[1]
        attention_mask = ~torch.tril(torch.ones((tl, tl), dtype=torch.bool, device=self.dev))

        for i in range(len(self.attention_layers)):
            seqs = torch.transpose(seqs, 0, 1)
            q = self.attention_layernorms[i](seqs)
            mha_outputs, _ = self.attention_layers[i](q, seqs, seqs, attn_mask=attention_mask)
            seqs = q + mha_outputs
            seqs = torch.transpose(seqs, 0, 1)
            seqs = self.forward_layernorms[i](seqs)
            seqs = self.forward_layers[i](seqs)
            seqs *= ~timeline_mask.unsqueeze(-1)

        return self.last_layernorm(seqs)

    def log2feats(self, log_seqs, llm_emb=None):
        del llm_emb
        return self._encode(log_seqs)

    def get_sa_repr(self, log_seqs, llm_emb=None):
        del llm_emb
        log_feats = self.log2feats(log_seqs)
        return self._mean_pool(log_feats, log_seqs)

    def calculate_loss(self, item_seq, pos_items):
        if pos_items is None:
            raise ValueError('SASRec calculate_loss requires pos_items.')

        seq_output = self.log2feats(item_seq)[:, -1, :]
        logits = torch.matmul(seq_output, self._real_item_embedding_matrix().transpose(0, 1))
        labels = self._as_long_tensor(pos_items) - 1
        valid_mask = labels >= 0
        if not torch.any(valid_mask):
            return torch.tensor(0.0, device=logits.device)
        return F.cross_entropy(logits[valid_mask], labels[valid_mask])

    def forward(self, user_ids, log_seqs, pos_seqs, neg_seqs, mode='default'):
        del user_ids
        log_feats = self.log2feats(log_seqs)
        if mode == 'log_only':
            return self._mean_pool(log_feats, log_seqs)

        pos_ids = self._as_long_tensor(pos_seqs)
        neg_ids = self._as_long_tensor(neg_seqs)
        pos_embs = self._lookup_item_emb(pos_ids)
        neg_embs = self._lookup_item_emb(neg_ids)

        pos_logits = (log_feats * pos_embs).sum(dim=-1)
        neg_logits = (log_feats * neg_embs).sum(dim=-1)

        if mode == 'item':
            return (
                log_feats.reshape(-1, log_feats.shape[2]),
                pos_embs.reshape(-1, log_feats.shape[2]),
                neg_embs.reshape(-1, log_feats.shape[2]),
            )
        return pos_logits, neg_logits

    def predict(self, user_ids, log_seqs, item_indices, llm_emb=None):
        del user_ids, llm_emb
        final_feat = self.log2feats(log_seqs)[:, -1, :]
        item_ids = self._as_long_tensor(item_indices)
        item_embs = self._lookup_item_emb(item_ids)
        logits = item_embs.matmul(final_feat.unsqueeze(-1)).squeeze(-1)
        return logits

    def full_sort_predict(self, log_seqs):
        final_feat = self.log2feats(log_seqs)[:, -1, :]
        return torch.matmul(final_feat, self._real_item_embedding_matrix().transpose(0, 1))

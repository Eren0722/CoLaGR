from __future__ import annotations

import torch
import torch.nn as nn


class GRU4Rec(nn.Module):
    """RecBole-style GRU4Rec backbone with EchoRec/SACP-compatible interfaces."""

    def __init__(self, user_num, item_num, args):
        super().__init__()
        self.kwargs = {"user_num": user_num, "item_num": item_num, "args": args}
        self.user_num = user_num
        self.item_num = item_num
        self.dev = args.device

        self.hidden_units = int(getattr(args, "hidden_units", 64))
        self.gru_hidden_units = int(getattr(args, "gru_hidden_units", 0) or self.hidden_units)
        self.hidden_size = self.hidden_units
        self.embedding_dim = self.hidden_units
        self.num_layers = int(getattr(args, "num_blocks", 1))
        self.dropout_input = float(getattr(args, "dropout_rate", 0.0))
        self.output_head = str(getattr(args, "gru_output_head", "linear")).lower()
        self.use_output_layer_norm = bool(getattr(args, "gru_output_layer_norm", False))

        self.item_emb = nn.Embedding(self.item_num + 1, self.hidden_units, padding_idx=0)
        self.item_embedding = self.item_emb
        self.emb_dropout = nn.Dropout(self.dropout_input)
        self.gru = nn.GRU(
            input_size=self.hidden_units,
            hidden_size=self.gru_hidden_units,
            num_layers=self.num_layers,
            bias=False,
            batch_first=True,
        )
        self.dense = nn.Linear(self.gru_hidden_units, self.hidden_units)
        self.output_norm = nn.LayerNorm(self.hidden_units) if self.use_output_layer_norm else nn.Identity()
        self.output_layer = None
        if self.output_head == "linear":
            self.output_layer = nn.Linear(self.hidden_units, self.item_num + 1)
            # Keep encoder input space, classifier space, and exported item space identical.
            self.output_layer.weight = self.item_emb.weight
        self.loss_fct = nn.CrossEntropyLoss()

        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.xavier_normal_(self.item_emb.weight)
        with torch.no_grad():
            self.item_emb.weight[0].fill_(0.0)

        for name, param in self.gru.named_parameters():
            if "weight_" in name:
                nn.init.xavier_uniform_(param)
        nn.init.xavier_uniform_(self.dense.weight)
        if self.dense.bias is not None:
            nn.init.zeros_(self.dense.bias)
        if self.output_layer is not None:
            if self.output_layer.bias is not None:
                nn.init.zeros_(self.output_layer.bias)

    def _as_long_tensor(self, seqs):
        if torch.is_tensor(seqs):
            return seqs.to(self.dev, dtype=torch.long)
        return torch.as_tensor(seqs, dtype=torch.long, device=self.dev)

    def _gather_last_valid(self, hidden_states: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        gather_index = lengths.clamp_min(1).sub(1).view(-1, 1, 1).expand(-1, 1, hidden_states.size(-1))
        return hidden_states.gather(1, gather_index).squeeze(1)

    def get_item_matrix(self, include_padding: bool = False) -> torch.Tensor:
        if self.output_layer is not None:
            return self.output_layer.weight if include_padding else self.output_layer.weight[1:]
        return self.item_emb.weight if include_padding else self.item_emb.weight[1:]

    def get_item_embeddings(self, item_ids) -> torch.Tensor:
        item_ids = self._as_long_tensor(item_ids)
        if self.output_layer is not None:
            flat_ids = item_ids.reshape(-1)
            gathered = self.output_layer.weight.index_select(0, flat_ids)
            return gathered.reshape(*item_ids.shape, -1)
        return self.item_emb(item_ids)

    def _left_align_sequences(self, seqs: torch.Tensor):
        batch_size, max_len = seqs.shape
        lengths = seqs.ne(0).sum(dim=1)
        positions = torch.arange(max_len, device=seqs.device).unsqueeze(0).expand(batch_size, -1)
        start = (max_len - lengths).unsqueeze(1)
        src_idx = (positions + start).clamp(max=max_len - 1)
        left_aligned = seqs.gather(1, src_idx)
        valid_mask = positions < lengths.unsqueeze(1)
        left_aligned = left_aligned.masked_fill(~valid_mask, 0)
        return left_aligned, lengths, valid_mask

    def _encode_left_hidden(self, log_seqs):
        seqs = self._as_long_tensor(log_seqs)
        left_aligned, lengths, valid_mask = self._left_align_sequences(seqs)

        seq_emb = self.item_emb(left_aligned)
        seq_emb = self.emb_dropout(seq_emb)

        left_feats, _ = self.gru(seq_emb)
        left_feats = self.dense(left_feats)
        left_feats = self.output_norm(left_feats)
        left_feats = left_feats * valid_mask.unsqueeze(-1).to(left_feats.dtype)
        return seqs, lengths, valid_mask, left_feats

    def log2feats(self, log_seqs, llm_emb=None):
        del llm_emb
        seqs, lengths, _, left_feats = self._encode_left_hidden(log_seqs)
        batch_size, max_len = seqs.shape

        positions = torch.arange(max_len, device=left_feats.device).unsqueeze(0).expand(batch_size, -1)
        offset = (max_len - lengths).unsqueeze(1)
        src_pos = (positions - offset).clamp(min=0)
        gather_idx = src_pos.unsqueeze(-1).expand(-1, -1, self.hidden_units)
        right_aligned = left_feats.gather(1, gather_idx)
        valid_mask = positions >= offset
        right_aligned = right_aligned.masked_fill(~valid_mask.unsqueeze(-1), 0.0)
        return right_aligned

    def _encode_last(self, log_seqs) -> torch.Tensor:
        _, lengths, _, left_feats = self._encode_left_hidden(log_seqs)
        return self._gather_last_valid(left_feats, lengths)

    def get_sa_repr(self, log_seqs, llm_emb=None) -> torch.Tensor:
        del llm_emb
        return self._encode_last(log_seqs)

    def calculate_loss(self, item_seq, pos_items):
        seq_output = self._encode_last(item_seq)
        if self.output_layer is None:
            logits = torch.matmul(seq_output, self.get_item_matrix().transpose(0, 1))
        else:
            logits = self.output_layer(seq_output)[:, 1:]
        labels = self._as_long_tensor(pos_items) - 1
        valid_mask = labels >= 0
        if not torch.any(valid_mask):
            return torch.tensor(0.0, device=seq_output.device)
        return self.loss_fct(logits[valid_mask], labels[valid_mask])

    def forward(self, user_ids, log_seqs, pos_seqs, neg_seqs, mode="default"):
        del user_ids
        log_feats = self.log2feats(log_seqs)

        if mode == "log_only":
            return log_feats[:, -1, :]

        pos_ids = self._as_long_tensor(pos_seqs)
        neg_ids = self._as_long_tensor(neg_seqs)
        pos_embs = self.get_item_embeddings(pos_ids)
        neg_embs = self.get_item_embeddings(neg_ids)
        if self.output_layer is None:
            pos_logits = (log_feats * pos_embs).sum(dim=-1)
            neg_logits = (log_feats * neg_embs).sum(dim=-1)
        else:
            weight = self.output_layer.weight
            bias = self.output_layer.bias
            pos_weight = weight[pos_ids]
            neg_weight = weight[neg_ids]
            pos_logits = (log_feats * pos_weight).sum(dim=-1)
            neg_logits = (log_feats * neg_weight).sum(dim=-1)
            if bias is not None:
                pos_logits = pos_logits + bias[pos_ids]
                neg_logits = neg_logits + bias[neg_ids]

        if mode == "item":
            return (
                log_feats.reshape(-1, log_feats.shape[-1]),
                pos_embs.reshape(-1, pos_embs.shape[-1]),
                neg_embs.reshape(-1, neg_embs.shape[-1]),
            )
        return pos_logits, neg_logits

    def predict(self, user_ids, log_seqs, item_indices, llm_emb=None):
        del user_ids
        final_feat = self.log2feats(log_seqs, llm_emb=llm_emb)[:, -1, :]
        item_ids = self._as_long_tensor(item_indices)
        if self.output_layer is not None:
            all_scores = self.output_layer(final_feat)
            if item_ids.dim() == 1:
                return all_scores.index_select(1, item_ids)
            return all_scores.gather(1, item_ids)

        item_embs = self.get_item_embeddings(item_ids)

        if item_embs.dim() == 2:
            return torch.matmul(final_feat, item_embs.transpose(0, 1))
        return torch.sum(final_feat.unsqueeze(1) * item_embs, dim=-1)

    def full_sort_predict(self, log_seqs):
        final_feat = self.log2feats(log_seqs)[:, -1, :]
        if self.output_layer is not None:
            return self.output_layer(final_feat)[:, 1:]
        item_matrix = self.get_item_matrix()
        return torch.matmul(final_feat, item_matrix.transpose(0, 1))

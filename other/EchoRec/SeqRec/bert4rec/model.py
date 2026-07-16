from __future__ import annotations

import math

import torch
import torch.nn as nn


class BERT4Rec(nn.Module):
    """BERT-style teacher supporting masked or next-item objectives."""

    def __init__(self, user_num, item_num, args):
        super().__init__()
        self.kwargs = {"user_num": user_num, "item_num": item_num, "args": args}
        self.user_num = user_num
        self.item_num = item_num
        self.dev = args.device
        self.args = args

        self.hidden_units = int(getattr(args, "hidden_units", 64))
        self.hidden_size = self.hidden_units
        self.embedding_dim = self.hidden_units
        self.maxlen = int(getattr(args, "maxlen", 128))
        self.num_layers = int(getattr(args, "num_blocks", 2))
        self.num_heads = int(getattr(args, "num_heads", 2))
        self.inner_size = int(getattr(args, "inner_size", self.hidden_units * 4))
        self.hidden_dropout_prob = float(getattr(args, "hidden_dropout_prob", getattr(args, "dropout_rate", 0.2)))
        self.attn_dropout_prob = float(getattr(args, "attn_dropout_prob", getattr(args, "dropout_rate", 0.2)))
        self.hidden_act = str(getattr(args, "hidden_act", "gelu")).lower()
        self.layer_norm_eps = float(getattr(args, "layer_norm_eps", 1e-12))
        self.initializer_range = float(getattr(args, "initializer_range", 0.02))
        self.nn_parameter = bool(getattr(args, "nn_parameter", False))
        self.bert_mask_prob = float(getattr(args, "bert_mask_prob", 0.15))
        self.bert_rec_objective = str(getattr(args, "bert_rec_objective", "masked")).lower()

        self.mask_token_id = self.item_num + 1
        self.real_item_start = 1
        self.real_item_end = self.item_num + 1

        vocab_size = self.item_num + 2
        if self.nn_parameter:
            self.item_emb = nn.Parameter(torch.empty(vocab_size, self.hidden_units))
            self.pos_emb = nn.Parameter(torch.empty(self.maxlen, self.hidden_units))
        else:
            self.item_emb = nn.Embedding(vocab_size, self.hidden_units, padding_idx=0)
            self.pos_emb = nn.Embedding(self.maxlen, self.hidden_units)
        self.item_embedding = self.item_emb

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.hidden_units,
            nhead=self.num_heads,
            dim_feedforward=self.inner_size,
            dropout=self.hidden_dropout_prob,
            activation=self.hidden_act if self.hidden_act in {"relu", "gelu"} else "gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=self.num_layers)
        self.input_norm = nn.LayerNorm(self.hidden_units, eps=self.layer_norm_eps)
        self.emb_dropout = nn.Dropout(self.hidden_dropout_prob)
        self.output_ffn = nn.Linear(self.hidden_units, self.hidden_units)
        self.output_act = nn.GELU() if self.hidden_act == "gelu" else nn.ReLU()
        self.output_norm = nn.LayerNorm(self.hidden_units, eps=self.layer_norm_eps)
        self.output_bias = nn.Parameter(torch.zeros(self.item_num + 1))
        self.loss_fct = nn.CrossEntropyLoss(ignore_index=0)

        self._reset_parameters()

    def _reset_parameters(self):
        if self.nn_parameter:
            nn.init.normal_(self.item_emb, mean=0.0, std=self.initializer_range)
            nn.init.normal_(self.pos_emb, mean=0.0, std=self.initializer_range)
            with torch.no_grad():
                self.item_emb[0].fill_(0.0)
        else:
            nn.init.normal_(self.item_emb.weight, mean=0.0, std=self.initializer_range)
            with torch.no_grad():
                self.item_emb.weight[0].fill_(0.0)
            nn.init.normal_(self.pos_emb.weight, mean=0.0, std=self.initializer_range)

        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=self.initializer_range)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

        nn.init.zeros_(self.output_bias)

    def _as_long_tensor(self, seqs):
        if torch.is_tensor(seqs):
            return seqs.to(self.dev, dtype=torch.long)
        return torch.as_tensor(seqs, dtype=torch.long, device=self.dev)

    def _lookup_item_emb(self, item_ids: torch.Tensor) -> torch.Tensor:
        if self.nn_parameter:
            return self.item_emb[item_ids]
        return self.item_emb(item_ids)

    def _lookup_pos_emb(self, position_ids: torch.Tensor) -> torch.Tensor:
        if self.nn_parameter:
            return self.pos_emb[position_ids]
        return self.pos_emb(position_ids)

    def _real_item_embedding_matrix(self) -> torch.Tensor:
        if self.nn_parameter:
            return self.item_emb[self.real_item_start : self.real_item_end]
        return self.item_emb.weight[self.real_item_start : self.real_item_end]

    def _training_item_embedding_matrix(self) -> torch.Tensor:
        if self.nn_parameter:
            return self.item_emb[: self.item_num + 1]
        return self.item_emb.weight[: self.item_num + 1]

    def _append_mask_token(self, item_seq: torch.Tensor) -> torch.Tensor:
        _, seq_len = item_seq.shape
        if seq_len != self.maxlen:
            raise ValueError(f"BERT4Rec expects seq_len == maxlen ({self.maxlen}), got {seq_len}")

        masked_seq = torch.zeros_like(item_seq)
        masked_seq[:, :-1] = item_seq[:, 1:]
        masked_seq[:, -1] = self.mask_token_id
        return masked_seq

    def _mask_training_tokens(self, item_seq: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        item_seq = self._as_long_tensor(item_seq).clone()
        labels = torch.zeros_like(item_seq)
        valid_mask = item_seq > 0
        if not torch.any(valid_mask):
            return item_seq, labels

        sample_probs = torch.rand(item_seq.shape, device=item_seq.device)
        masked_positions = (sample_probs < self.bert_mask_prob) & valid_mask

        # Match baseline behavior more closely by ensuring each non-empty sequence
        # contributes at least one masked position.
        nonempty_rows = valid_mask.any(dim=1)
        if torch.any(nonempty_rows & ~masked_positions.any(dim=1)):
            row_random = torch.rand(item_seq.shape, device=item_seq.device)
            row_random = row_random.masked_fill(~valid_mask, 2.0)
            chosen_cols = row_random.argmin(dim=1)
            missing_rows = (nonempty_rows & ~masked_positions.any(dim=1)).nonzero(as_tuple=False).squeeze(-1)
            masked_positions[missing_rows, chosen_cols[missing_rows]] = True

        labels[masked_positions] = item_seq[masked_positions]

        strategy_probs = torch.rand(item_seq.shape, device=item_seq.device)
        mask_token_positions = masked_positions & (strategy_probs < 0.8)
        random_item_positions = masked_positions & (strategy_probs >= 0.8) & (strategy_probs < 0.9)

        item_seq[mask_token_positions] = self.mask_token_id
        if torch.any(random_item_positions):
            random_items = torch.randint(
                1,
                self.item_num + 1,
                (int(random_item_positions.sum().item()),),
                device=item_seq.device,
            )
            item_seq[random_item_positions] = random_items

        return item_seq, labels

    def _encode(self, item_seq, append_mask: bool) -> torch.Tensor:
        item_seq = self._as_long_tensor(item_seq)
        if append_mask:
            item_seq = self._append_mask_token(item_seq)

        position_ids = torch.arange(item_seq.size(1), device=item_seq.device).unsqueeze(0).expand_as(item_seq)
        seq_emb = self._lookup_item_emb(item_seq) * math.sqrt(self.hidden_units)
        seq_emb = seq_emb + self._lookup_pos_emb(position_ids)
        seq_emb = self.input_norm(seq_emb)
        seq_emb = self.emb_dropout(seq_emb)

        padding_mask = item_seq.eq(0)
        feats = self.encoder(seq_emb, src_key_padding_mask=padding_mask)
        feats = self.output_norm(self.output_act(self.output_ffn(feats)))
        feats = feats.masked_fill(padding_mask.unsqueeze(-1), 0.0)
        return feats

    def _next_item_query(self, log_seqs) -> torch.Tensor:
        return self._encode(log_seqs, append_mask=True)[:, -1, :]

    def log2feats(self, log_seqs, llm_emb=None):
        del llm_emb
        return self._encode(log_seqs, append_mask=False)

    def get_sa_repr(self, log_seqs, llm_emb=None):
        del llm_emb
        return self.log2feats(log_seqs)[:, -1, :]

    def calculate_loss(self, item_seq, pos_items):
        if self.bert_rec_objective == "next_item":
            if pos_items is None:
                raise ValueError("BERT4Rec next_item objective requires pos_items.")
            final_feat = self._next_item_query(item_seq)
            item_matrix = self._real_item_embedding_matrix()
            logits = torch.matmul(final_feat, item_matrix.transpose(0, 1)) + self.output_bias[1:].unsqueeze(0)
            labels = self._as_long_tensor(pos_items) - 1
            valid_mask = labels >= 0
            if not torch.any(valid_mask):
                return torch.tensor(0.0, device=logits.device)
            return nn.functional.cross_entropy(logits[valid_mask], labels[valid_mask])

        masked_seq, labels = self._mask_training_tokens(item_seq)
        seq_output = self._encode(masked_seq, append_mask=False)
        item_matrix = self._training_item_embedding_matrix()
        logits = torch.matmul(seq_output, item_matrix.transpose(0, 1)) + self.output_bias.view(1, 1, -1)
        return self.loss_fct(logits.reshape(-1, logits.size(-1)), labels.reshape(-1))

    def forward(self, user_ids, log_seqs, pos_seqs, neg_seqs, mode="default"):
        del user_ids
        if mode == "log_only":
            return self.log2feats(log_seqs)[:, -1, :]

        log_feats = self._encode(log_seqs, append_mask=False)
        pos_ids = self._as_long_tensor(pos_seqs)
        neg_ids = self._as_long_tensor(neg_seqs)
        pos_embs = self._lookup_item_emb(pos_ids)
        neg_embs = self._lookup_item_emb(neg_ids)
        pos_logits = (log_feats * pos_embs).sum(dim=-1)
        neg_logits = (log_feats * neg_embs).sum(dim=-1)

        if mode == "item":
            return (
                log_feats.reshape(-1, log_feats.shape[-1]),
                pos_embs.reshape(-1, pos_embs.shape[-1]),
                neg_embs.reshape(-1, neg_embs.shape[-1]),
            )
        return pos_logits, neg_logits

    def predict(self, user_ids, log_seqs, item_indices, llm_emb=None):
        del user_ids, llm_emb
        final_feat = self._next_item_query(log_seqs)
        item_ids = self._as_long_tensor(item_indices)
        item_embs = self._lookup_item_emb(item_ids)
        if item_embs.dim() == 2:
            scores = torch.matmul(final_feat, item_embs.transpose(0, 1))
            bias = self.output_bias[item_ids]
            return scores + bias.unsqueeze(0)

        bias = self.output_bias[item_ids]
        return torch.sum(final_feat.unsqueeze(1) * item_embs, dim=-1) + bias

    def full_sort_predict(self, log_seqs):
        final_feat = self._next_item_query(log_seqs)
        item_matrix = self._real_item_embedding_matrix()
        return torch.matmul(final_feat, item_matrix.transpose(0, 1)) + self.output_bias[1:].unsqueeze(0)

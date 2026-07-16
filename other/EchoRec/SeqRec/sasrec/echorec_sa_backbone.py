import numpy as np
import torch
import torch.nn as nn


try:
    from recbole.model.layers import TransformerEncoder as RecboleTransformerEncoder
except ImportError:
    RecboleTransformerEncoder = None


class _FallbackTransformerEncoder(nn.Module):
    """Fallback transformer encoder when recbole is unavailable."""

    def __init__(
        self,
        n_layers,
        n_heads,
        hidden_size,
        inner_size,
        hidden_dropout_prob,
        attn_dropout_prob,
        hidden_act,
        layer_norm_eps,
    ):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=n_heads,
            dim_feedforward=inner_size,
            dropout=max(hidden_dropout_prob, attn_dropout_prob),
            activation=hidden_act,
            batch_first=True,
            layer_norm_eps=layer_norm_eps,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

    def forward(self, hidden_states, attention_mask, output_all_encoded_layers=True):
        del attention_mask
        output = self.encoder(hidden_states)
        if output_all_encoded_layers:
            return [output]
        return output


class _BaseEchoRecSABackbone(nn.Module):
    """Semantic-alignment recommender with a SASRec-compatible interface."""

    def __init__(self, user_num, item_num, args):
        super().__init__()
        self.kwargs = {"user_num": user_num, "item_num": item_num, "args": args}

        self.user_num = user_num
        self.item_num = item_num
        self.dev = args.device
        self.nn_parameter = getattr(args, "nn_parameter", False)

        self.hidden_units = getattr(args, "hidden_units", 64)
        self.embedding_dim = self.hidden_units
        self.maxlen = getattr(args, "maxlen", 128)

        self.n_layers = getattr(args, "num_blocks", getattr(args, "n_layers", 2))
        self.n_heads = getattr(args, "num_heads", getattr(args, "n_heads", 2))
        self.inner_size = getattr(args, "inner_size", 256)
        self.hidden_dropout_prob = getattr(args, "hidden_dropout_prob", 0.5)
        self.attn_dropout_prob = getattr(args, "attn_dropout_prob", 0.5)
        self.hidden_act = getattr(args, "hidden_act", "gelu")
        self.layer_norm_eps = getattr(args, "layer_norm_eps", 1e-12)
        self.initializer_range = getattr(args, "initializer_range", 0.02)

        if self.nn_parameter:
            self.item_emb = nn.Parameter(torch.empty(self.item_num + 1, self.hidden_units))
            self.pos_emb = nn.Parameter(torch.empty(self.maxlen, self.hidden_units))
            nn.init.normal_(self.item_emb, mean=0.0, std=self.initializer_range)
            nn.init.normal_(self.pos_emb, mean=0.0, std=self.initializer_range)
        else:
            self.item_emb = nn.Embedding(self.item_num + 1, self.hidden_units, padding_idx=0)
            self.pos_emb = nn.Embedding(self.maxlen, self.hidden_units)

        self.item_embedding = self.item_emb
        self.position_embedding = self.pos_emb

        encoder_cls = _FallbackTransformerEncoder if RecboleTransformerEncoder is None else RecboleTransformerEncoder
        self.trm_encoder = encoder_cls(
            n_layers=self.n_layers,
            n_heads=self.n_heads,
            hidden_size=self.hidden_units,
            inner_size=self.inner_size,
            hidden_dropout_prob=self.hidden_dropout_prob,
            attn_dropout_prob=self.attn_dropout_prob,
            hidden_act=self.hidden_act,
            layer_norm_eps=self.layer_norm_eps,
        )

        self.LayerNorm = nn.LayerNorm(self.hidden_units, eps=self.layer_norm_eps)
        self.dropout = nn.Dropout(self.hidden_dropout_prob)
        self.loss_fct = nn.CrossEntropyLoss()

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=self.initializer_range)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def _lookup_item_emb(self, item_ids):
        item_ids = torch.as_tensor(item_ids, dtype=torch.long, device=self.dev)
        if self.nn_parameter:
            return self.item_emb[item_ids]
        return self.item_emb(item_ids)

    def _lookup_pos_emb(self, position_ids):
        position_ids = torch.as_tensor(position_ids, dtype=torch.long, device=self.dev)
        if self.nn_parameter:
            return self.pos_emb[position_ids]
        return self.pos_emb(position_ids)

    def get_attention_mask(self, item_seq):
        attention_mask = (item_seq > 0).long()
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        max_len = attention_mask.size(-1)
        attn_shape = (1, max_len, max_len)
        subsequent_mask = torch.triu(torch.ones(attn_shape, device=item_seq.device), diagonal=1)
        subsequent_mask = (subsequent_mask == 0).unsqueeze(1).long()
        extended_attention_mask = extended_attention_mask * subsequent_mask
        extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype)
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        return extended_attention_mask

    def log2feats(self, log_seqs, llm_emb=None):
        del llm_emb
        if isinstance(log_seqs, np.ndarray):
            item_seq = torch.LongTensor(log_seqs).to(self.dev)
        else:
            item_seq = log_seqs.to(self.dev)

        position_ids = torch.arange(item_seq.size(1), dtype=torch.long, device=item_seq.device)
        position_ids = position_ids.unsqueeze(0).expand_as(item_seq)
        position_embedding = self._lookup_pos_emb(position_ids)

        item_emb = self._lookup_item_emb(item_seq)
        input_emb = item_emb + position_embedding
        input_emb = self.LayerNorm(input_emb)
        input_emb = self.dropout(input_emb)

        extended_attention_mask = self.get_attention_mask(item_seq)
        trm_output = self.trm_encoder(input_emb, extended_attention_mask, output_all_encoded_layers=True)
        if isinstance(trm_output, list):
            return trm_output[-1]
        return trm_output

    def forward(self, user_ids, log_seqs, pos_seqs, neg_seqs, mode="default"):
        del user_ids
        log_feats = self.log2feats(log_seqs)
        if mode == "log_only":
            return log_feats[:, -1, :]

        pos_embs = self._lookup_item_emb(pos_seqs)
        neg_embs = self._lookup_item_emb(neg_seqs)

        pos_logits = (log_feats * pos_embs).sum(dim=-1)
        neg_logits = (log_feats * neg_embs).sum(dim=-1)

        if mode == "item":
            return (
                log_feats.reshape(-1, log_feats.shape[2]),
                pos_embs.reshape(-1, log_feats.shape[2]),
                neg_embs.reshape(-1, log_feats.shape[2]),
            )
        return pos_logits, neg_logits

    def predict(self, user_ids, log_seqs, item_indices, llm_emb=None):
        del user_ids
        log_feats = self.log2feats(log_seqs, llm_emb=llm_emb)
        final_feat = log_feats[:, -1, :]
        item_embs = self._lookup_item_emb(item_indices)
        return item_embs.matmul(final_feat.unsqueeze(-1)).squeeze(-1)

    def calculate_loss(self, item_seq, pos_items):
        seq_output = self.log2feats(item_seq)[:, -1, :]
        if self.nn_parameter:
            test_item_emb = self.item_emb[1 : self.item_num + 1]
        else:
            test_item_emb = self.item_emb.weight[1 : self.item_num + 1]
        logits = torch.matmul(seq_output, test_item_emb.transpose(0, 1))
        pos_items = pos_items - 1
        return self.loss_fct(logits, pos_items)

    def full_sort_predict(self, item_seq):
        seq_output = self.log2feats(item_seq)[:, -1, :]
        if self.nn_parameter:
            test_items_emb = self.item_emb[1 : self.item_num + 1]
        else:
            test_items_emb = self.item_emb.weight[1 : self.item_num + 1]
        return torch.matmul(seq_output, test_items_emb.transpose(0, 1))


class EchoRecSABackbone(_BaseEchoRecSABackbone):
    """Transformer teacher backbone used by the semantic-alignment stage."""

    def __init__(self, user_num, item_num, args):
        super().__init__(user_num, item_num, args)


SemanticAlignmentTransformer = EchoRecSABackbone

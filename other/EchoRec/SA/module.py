import random
from functools import lru_cache
from typing import Dict, List, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def _mask_correlated_samples(batch_size: int):
    num_rows = 2 * batch_size
    mask = torch.ones((num_rows, num_rows), dtype=torch.bool)
    mask.fill_diagonal_(False)
    for i in range(batch_size):
        mask[i, batch_size + i] = False
        mask[batch_size + i, i] = False
    return mask


def info_nce(z_i, z_j, temp, batch_size, sim="dot"):
    num_rows = 2 * batch_size
    z = torch.cat((z_i, z_j), dim=0)
    if sim == "cos":
        sim_matrix = nn.functional.cosine_similarity(z.unsqueeze(1), z.unsqueeze(0), dim=2) / temp
    elif sim == "dot":
        sim_matrix = torch.mm(z, z.T) / temp
    else:
        raise ValueError(f"Unsupported similarity type: {sim}")

    sim_i_j = torch.diag(sim_matrix, batch_size)
    sim_j_i = torch.diag(sim_matrix, -batch_size)
    positive_samples = torch.cat((sim_i_j, sim_j_i), dim=0).reshape(num_rows, 1)
    mask = _mask_correlated_samples(batch_size)
    negative_samples = sim_matrix[mask].reshape(num_rows, -1)
    labels = torch.zeros(num_rows, dtype=torch.long, device=z.device)
    logits = torch.cat((positive_samples, negative_samples), dim=1)
    return logits, labels


class SemanticAlignmentModule(nn.Module):
    """Implements semantic alignment losses for teacher-space reshaping."""

    def __init__(self, args, recsys_model, feature_mapper: Optional[nn.Module] = None, detach_recsys_grad: bool = True):
        super().__init__()
        self.args = args
        self.device = args.device
        self.maxlen = getattr(args, "maxlen", 50)
        self.detach_recsys_grad = detach_recsys_grad
        self.temperature = getattr(args, "sa_temperature", 1.0)
        self.alpha = getattr(args, "sa_alpha", 0.1)
        self.beta = getattr(args, "sa_beta", 0.1)
        self.mlm_probability = getattr(args, "sa_mlm_probability", 0.2)
        self.similarity = getattr(args, "sa_similarity", "dot")
        self.contrast_norm = bool(getattr(args, "sa_contrast_norm", False))
        self.repr_mode = getattr(args, "sa_repr_mode", "mean")
        self.effective_repr_mode = self.repr_mode
        # Keep a plain reference to the backbone instead of registering it as a
        # child module. Otherwise semantic_module.parameters() recursively
        # includes all SASRec parameters, which breaks optimizer grouping.
        object.__setattr__(self, "recsys_model", recsys_model)

        # Assets prepared by prepare_semantic_assets
        self.item_neighbors = np.asarray(args.sorted_indices_numpy)
        self.user_neighbors = np.asarray(args.user_sorted_indices_numpy)
        self.user_neighbors_tensor = torch.from_numpy(args.user_sorted_indices_numpy).long().to(self.device)
        self.user_semantic_emb = args.user_semantic_emb.float().to(self.device)
        self.seq_keys_to_int: Dict[str, int] = args.seq_keys_to_int
        self.seq_int_to_keys: Dict[int, str] = args.seq_int_to_keys

        self.hidden_size = getattr(self.recsys_model, "embedding_dim", getattr(self.recsys_model, "hidden_units", 64))
        semantic_dim = self.user_semantic_emb.shape[-1]

        self.leakyrelu = nn.LeakyReLU(0.2)
        self.W = nn.Parameter(torch.empty(size=(semantic_dim, self.hidden_size)))
        self.a = nn.Parameter(torch.empty(size=(2 * self.hidden_size, 1)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

        self.nce_fct = nn.CrossEntropyLoss()
        self.projection_head = None
        if bool(getattr(args, "sa_use_projection_head", False)):
            proj_hidden_dim = int(getattr(args, "sa_proj_hidden_dim", 0) or self.hidden_size)
            proj_act = getattr(args, "sa_proj_act", "gelu")
            act_layer = nn.GELU() if proj_act == "gelu" else nn.ReLU()
            self.projection_head = nn.Sequential(
                nn.Linear(self.hidden_size, proj_hidden_dim),
                act_layer,
                nn.Linear(proj_hidden_dim, self.hidden_size),
            )

        # Contrastive objectives operate directly on backbone sequence features
        # unless an explicit external mapper is injected.
        self.feature_mapper = feature_mapper
        self._fusion_info = {
            "sa_status": "enabled",
            "projection_head": "enabled" if self.projection_head is not None else "disabled",
            "feature_mapper": "custom" if feature_mapper is not None else "disabled",
            "similarity": self.similarity,
            "contrast_norm": self.contrast_norm,
            "repr_mode": self.effective_repr_mode,
        }

    def forward(self, *args, **kwargs):
        raise NotImplementedError("SemanticAlignmentModule is used via compute_batch_losses().")

    def compute_batch_losses(
        self,
        seq_unique_ids: torch.Tensor,
        padded_seq: torch.Tensor,
        aug_seq1: torch.Tensor,
        aug_seq2: torch.Tensor,
        neighbor_seqs: Optional[torch.Tensor] = None,
        allow_recsys_grad: bool = False,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Compute semantic losses from already-prepared tensors (stage-one training)."""
        seq_ids = seq_unique_ids.detach().cpu().numpy().astype(np.int64)
        base_seq = padded_seq.detach().cpu().numpy().astype(np.int32)
        aug1 = aug_seq1.detach().cpu().numpy().astype(np.int32)
        aug2 = aug_seq2.detach().cpu().numpy().astype(np.int32)
        neighbor_array = None
        if neighbor_seqs is not None:
            neighbor_array = neighbor_seqs.detach().cpu().numpy().astype(np.int32)
        return self._compute_contrastive_losses(
            seq_ids=seq_ids,
            seq_array=base_seq,
            aug_seq1=aug1,
            aug_seq2=aug2,
            neighbor_seqs=neighbor_array,
            allow_gradient=allow_recsys_grad,
        )

    def _pool_repr(self, feats: torch.Tensor, seq_array: np.ndarray) -> torch.Tensor:
        mask = torch.as_tensor(seq_array > 0, dtype=feats.dtype, device=feats.device)
        denom = mask.sum(dim=1, keepdim=True).clamp_min(1.0)
        return (feats * mask.unsqueeze(-1)).sum(dim=1) / denom

    def _encode_sequences(self, seq_array: np.ndarray, detach: bool = True) -> torch.Tensor:
        feats = self.recsys_model.log2feats(seq_array)
        outputs = self._pool_repr(feats, seq_array)
        if detach:
            outputs = outputs.detach()
        return outputs.to(self.device)

    def _map_features(self, tensor: torch.Tensor) -> torch.Tensor:
        if self.projection_head is not None:
            tensor = self.projection_head(tensor)
        if isinstance(self.feature_mapper, nn.Module):
            tensor = self.feature_mapper(tensor)
        elif callable(self.feature_mapper):
            tensor = self.feature_mapper(tensor)
        if self.contrast_norm:
            tensor = F.normalize(tensor, dim=-1)
        return tensor

    def _pad_sequence(self, seq: List[int]):
        seq = seq[-self.maxlen:]
        padded = np.zeros(self.maxlen, dtype=np.int32)
        padded[-len(seq):] = seq
        return padded

    def _replace_input_ids(self, seq: np.ndarray):
        replaced = seq.copy()
        for idx, token in enumerate(seq):
            if token == 0:
                continue
            if random.random() < self.mlm_probability:
                if token >= len(self.item_neighbors):
                    continue
                neighbors = self.item_neighbors[token]
                if len(neighbors) == 0:
                    continue
                replaced[idx] = int(random.choice(neighbors))
        return replaced

    def _recall_similar_sequences(self, seq_ids: List[int]):
        neighbor_matrix = []
        for seq_id in seq_ids:
            neighbors = self.user_neighbors[seq_id]
            neighbor_seqs = []
            for neighbor_id in neighbors:
                neighbor_seqs.append(self._get_sequence_by_id(int(neighbor_id)))
            neighbor_matrix.append(neighbor_seqs)
        return np.array(neighbor_matrix, dtype=np.int32)

    @lru_cache(maxsize=100000)
    def _get_sequence_by_id(self, seq_id: int):
        key = self.seq_int_to_keys.get(seq_id)
        if not key:
            return np.zeros(self.maxlen, dtype=np.int32)
        parts = key.split(":")[1:]
        seq = list(map(int, parts))
        return self._pad_sequence(seq)

    def _select_similar_user_probs(self, seq_unique_id: torch.Tensor):
        neighbors_tensor = self.user_neighbors_tensor[seq_unique_id]
        neighbors_semantic_emb = self.user_semantic_emb[neighbors_tensor]
        neighbors_semantic_emb = neighbors_semantic_emb @ self.W

        seq_semantic_emb = self.user_semantic_emb[seq_unique_id]
        seq_semantic_emb = (seq_semantic_emb @ self.W).unsqueeze(1).expand_as(neighbors_semantic_emb)

        concat = torch.cat((seq_semantic_emb, neighbors_semantic_emb), dim=-1)
        attention = torch.matmul(concat, self.a).squeeze(-1)
        attention = self.leakyrelu(attention)
        attention = F.softmax(attention, dim=1)
        attention = F.dropout(attention, p=0.5, training=self.training)
        return attention

    def _compute_contrastive_losses(
        self,
        seq_ids: np.ndarray,
        seq_array: np.ndarray,
        aug_seq1: np.ndarray,
        aug_seq2: np.ndarray,
        neighbor_seqs: Optional[np.ndarray],
        allow_gradient: bool,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        batch_size = seq_array.shape[0]
        seq_outputs = self._map_features(self._encode_sequences(seq_array, detach=not allow_gradient))

        seq_output1 = self._map_features(self._encode_sequences(aug_seq1, detach=not allow_gradient))
        seq_output2 = self._map_features(self._encode_sequences(aug_seq2, detach=not allow_gradient))

        logits, labels = info_nce(
            seq_output1,
            seq_output2,
            self.temperature,
            seq_output1.shape[0],
            sim=self.similarity,
        )
        item_loss = self.nce_fct(logits, labels)

        if neighbor_seqs is None:
            neighbor_sequences = self._recall_similar_sequences(seq_ids)
        else:
            neighbor_sequences = neighbor_seqs

        neighbor_shape = neighbor_sequences.shape
        neighbor_outputs = self._map_features(
            self._encode_sequences(neighbor_sequences.reshape(-1, self.maxlen), detach=not allow_gradient)
        )
        neighbor_outputs = neighbor_outputs.reshape(neighbor_shape[0], neighbor_shape[1], -1)

        seq_unique_tensor = torch.tensor(seq_ids, dtype=torch.long, device=self.device)
        attention = self._select_similar_user_probs(seq_unique_tensor)
        neighbor_weighted = torch.sum(attention.unsqueeze(-1) * neighbor_outputs, dim=1)

        user_logits, user_labels = info_nce(
            seq_outputs,
            neighbor_weighted,
            self.temperature,
            batch_size,
            sim=self.similarity,
        )
        user_loss = self.nce_fct(user_logits, user_labels)

        total_loss = self.alpha * user_loss + self.beta * item_loss
        info = {
            "sa_status": "enabled",
            "user_cl_loss": user_loss.item(),
            "item_cl_loss": item_loss.item(),
            "alpha": self.alpha,
            "beta": self.beta,
            "similarity": self.similarity,
            "contrast_norm": float(self.contrast_norm),
            "repr_mode": self.effective_repr_mode,
            "valid_batch": batch_size,
        }
        self._fusion_info = info
        return total_loss, info

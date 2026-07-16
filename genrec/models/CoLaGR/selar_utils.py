import torch
import torch.nn.functional as F


def topk_normalized_entropy(logits, topk=5, eps=1e-10):
    """
    SeLaR-style top-k normalized entropy over the current SID level logits.
    Returns entropy in [0, 1], normalized top-k probabilities, and top-k indices.
    """
    k = min(int(topk), logits.shape[-1])
    probs = F.softmax(logits, dim=-1)
    topk_probs, topk_indices = torch.topk(probs, k=k, dim=-1)
    topk_probs_norm = topk_probs / topk_probs.sum(dim=-1, keepdim=True).clamp_min(eps)
    entropy = -(topk_probs_norm * topk_probs_norm.clamp_min(eps).log()).sum(dim=-1)
    max_entropy = torch.log(torch.tensor(float(k), dtype=logits.dtype, device=logits.device))
    normalized_entropy = torch.clamp(entropy / max_entropy.clamp_min(eps), 0.0, 1.0)
    return normalized_entropy, topk_probs_norm, topk_indices


def entropy_activation_mask(logits, topk=5, threshold=0.5):
    """
    Activate collaborative correction only when the base decoder is uncertain.
    """
    entropy, topk_probs, topk_indices = topk_normalized_entropy(logits, topk=topk)
    activate = entropy >= float(threshold)
    return activate, entropy, topk_probs, topk_indices

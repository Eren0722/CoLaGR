import torch


def grouped_advantages(rewards, group_size, eps=1e-6):
    """Compute per-original-sample relative advantages.

    Rollouts are laid out as ``[sample_0 variants..., sample_1 variants...]``.
    Keeping the reduction inside each local group is DDP-safe: no reward from
    another rank can become the baseline for a local sample.
    """
    if rewards.ndim != 1 or rewards.numel() % group_size:
        raise ValueError('rewards must be a flat tensor divisible by group_size')
    grouped = rewards.reshape(-1, group_size)
    mean = grouped.mean(dim=1, keepdim=True)
    std = grouped.std(dim=1, keepdim=True, unbiased=False)
    advantages = (grouped - mean) / std.clamp_min(eps)
    # Identical rewards contain no learning signal; avoid NaNs and preserve 0.
    advantages = torch.where(std > eps, advantages, torch.zeros_like(advantages))
    return advantages.reshape(-1)

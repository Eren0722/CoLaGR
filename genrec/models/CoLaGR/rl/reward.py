import torch


def latent_rewards(loss_gen, loss_pref, copref_weight):
    """Per-rollout reward used by latent policy optimization."""
    if loss_gen.shape != loss_pref.shape:
        raise ValueError('loss_gen and loss_pref must have the same shape')
    return -loss_gen.detach() - float(copref_weight) * loss_pref.detach()

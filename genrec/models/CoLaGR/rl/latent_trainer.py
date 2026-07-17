import copy
import math
from logging import getLogger

import torch
from torch.nn.utils import clip_grad_norm_
from torch.optim import AdamW
from tqdm import tqdm
from transformers.optimization import get_scheduler

from genrec.utils import config_for_log, get_file_name, log
from .advantage import grouped_advantages
from .reward import latent_rewards


class LatentRLTrainerMixin:
    """Stage-2 GRPO-style optimization over CoReason latent perturbations."""

    def fit_latent_rl(self, train_dataloader, val_dataloader):
        model = self.model
        base_model = self.accelerator.unwrap_model(model)
        if not base_model.use_coreason:
            raise ValueError('Latent RL requires use_coreason=True.')
        base_model.configure_latent_rl(True)
        trainable = [p for p in base_model.parameters() if p.requires_grad]
        if not trainable:
            raise RuntimeError('Latent RL has no trainable parameters.')

        reference = copy.deepcopy(base_model).to(self.accelerator.device)
        reference.configure_latent_rl(False)
        reference.eval()
        for parameter in reference.parameters():
            parameter.requires_grad = False

        optimizer = AdamW(
            trainable,
            lr=float(self.config.get('latent_rl_lr', 1e-4)),
            weight_decay=float(self.config.get('latent_rl_weight_decay', 0.0)),
        )
        epochs = int(self.config.get('latent_rl_epochs', 1))
        scheduler = get_scheduler(
            name='cosine',
            optimizer=optimizer,
            num_warmup_steps=int(self.config.get('latent_rl_warmup_steps', 0)),
            num_training_steps=max(1, epochs * len(train_dataloader)),
        )
        self.model, optimizer, train_dataloader, val_dataloader, scheduler = self.accelerator.prepare(
            model, optimizer, train_dataloader, val_dataloader, scheduler
        )
        tracker = self.config.get('tracker', 'tensorboard')
        project_name = get_file_name(self.config, suffix='_latent_rl')
        init_kwargs = {'tensorboard': {'flush_secs': 60}} if tracker == 'tensorboard' else {}
        self.accelerator.init_trackers(
            project_name=project_name,
            config=config_for_log(self.config),
            init_kwargs=init_kwargs,
        )

        rollout_count = max(2, int(self.config.get('latent_rl_rollouts', 4)))
        noise_std = float(self.config.get('latent_rl_noise_std', 0.10))
        copref_weight = float(self.config.get('latent_rl_reward_copref_weight', 0.10))
        clip_eps = float(self.config.get('latent_rl_clip_eps', 0.20))
        ref_beta = float(self.config.get('latent_rl_kl_beta', 0.01))
        update_epochs = max(1, int(self.config.get('latent_rl_update_epochs', 1)))
        best_score, best_epoch = float('-inf'), 0

        for epoch in range(epochs):
            self.model.train()
            total_loss = total_reward = total_kl = 0.0
            for batch in tqdm(train_dataloader, total=len(train_dataloader), desc=f'Latent RL [{epoch + 1}]'):
                batch = self._move_tensor_batch(batch)
                batch = self._attach_coprefs(batch, 'train')
                batch_size = batch['labels'].shape[0]
                d_model = base_model.t5.config.d_model
                noise = torch.randn(
                    batch_size, rollout_count, self.num_levels, d_model,
                    device=self.accelerator.device,
                ) * noise_std
                noise[:, 0].zero_()  # exact zero-noise anchor, as in R3
                flat_noise = noise.reshape(batch_size * rollout_count, self.num_levels, d_model)
                expanded = {
                    key: value.repeat_interleave(rollout_count, dim=0)
                    if torch.is_tensor(value) and value.shape[0] == batch_size else value
                    for key, value in batch.items()
                }
                if 'coprefs' in batch:
                    expanded['coprefs'] = [
                        value.repeat_interleave(rollout_count, dim=0)
                        for value in batch['coprefs']
                    ]
                expanded['_latent_rl_noise'] = flat_noise

                with torch.no_grad():
                    # Keep the behavior policy fixed during a rollout. This
                    # also prevents T5 dropout from changing old log-probs.
                    self.model.eval()
                    old_out = self.model(expanded)
                    ref_out = reference.forward_latent_rl(expanded, flat_noise)
                    rewards = latent_rewards(
                        old_out.loss_gen_per_sample,
                        old_out.loss_pref_per_sample,
                        copref_weight,
                    )
                    advantages = grouped_advantages(rewards, rollout_count)

                self.model.train()
                for _ in range(update_epochs):
                    current = self.model(expanded)
                    ratio = torch.exp(current.logprob - old_out.logprob.detach())
                    clipped = ratio.clamp(1.0 - clip_eps, 1.0 + clip_eps)
                    policy = -torch.minimum(ratio * advantages, clipped * advantages).mean()
                    log_delta = ref_out.logprob.detach() - current.logprob
                    kl = (torch.exp(log_delta) - log_delta - 1.0).mean()
                    loss = policy + ref_beta * kl
                    optimizer.zero_grad()
                    self.accelerator.backward(loss)
                    max_grad_norm = self.config.get('max_grad_norm')
                    if max_grad_norm is not None:
                        clip_grad_norm_(trainable, max_grad_norm)
                    optimizer.step()
                    scheduler.step()

                total_loss += float(loss.detach().item())
                total_reward += float(rewards.mean().item())
                total_kl += float(kl.detach().item())

            denom = max(1, len(train_dataloader))
            self.log(
                f'[Latent RL {epoch + 1}] loss={total_loss / denom}; '
                f'reward={total_reward / denom}; ref_kl={total_kl / denom}; '
                f'rollouts={rollout_count}; noise_std={noise_std}'
            )
            self.accelerator.log({
                'LatentRL/loss': total_loss / denom,
                'LatentRL/reward': total_reward / denom,
                'LatentRL/ref_kl': total_kl / denom,
            }, step=epoch + 1)

            interval = int(self.config.get('latent_rl_eval_interval', self.config['eval_interval']))
            if (epoch + 1) % max(1, interval) == 0:
                results = self.evaluate(val_dataloader, split='val')
                score = results[self.config['val_metric']]
                self.log(f'[Latent RL {epoch + 1}] Val Results: {results}')
                if score > best_score and self.accelerator.is_main_process:
                    best_score, best_epoch = score, epoch + 1
                    torch.save(self.accelerator.unwrap_model(self.model).state_dict(), self.saved_model_ckpt)

        self.log(f'Latent RL best epoch: {best_epoch}, best val score: {best_score}')
        return best_epoch, best_score

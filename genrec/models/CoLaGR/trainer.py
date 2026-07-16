import math
from collections import defaultdict, OrderedDict
from logging import getLogger

import torch
from torch.nn.utils import clip_grad_norm_
from torch.optim import AdamW
from tqdm import tqdm
from transformers.optimization import get_scheduler

from genrec.evaluator import Evaluator
from genrec.trainer import Trainer
from genrec.utils import config_for_log, get_file_name, get_total_steps, log


class CoLaGRTrainer(Trainer):
    def __init__(self, config, model, tokenizer):
        super().__init__(config, model, tokenizer)
        self.num_levels = tokenizer.n_digit
        self.tokenizer = tokenizer
        load_eval_copref_diagnostics = bool(config.get('load_eval_copref_diagnostics', False))
        self.copref_by_split = {
            'train': self._load_copref(config.get('copref_train_path')),
            'val': self._load_copref(config.get('copref_val_path')) if load_eval_copref_diagnostics else None,
            'test': self._load_copref(config.get('copref_test_path')) if load_eval_copref_diagnostics else None,
        }
        self.eval_copref_diagnostics_enabled = load_eval_copref_diagnostics
        self.evaluator = Evaluator(config, tokenizer)
        self.logger = getLogger()

    def _config_bool(self, key, default=False):
        value = self.config.get(key, default)
        if isinstance(value, str):
            return value.strip().lower() in {'1', 'true', 'yes', 'y', 'on'}
        return bool(value)

    def _level_float_config(self, list_key, scalar_key, level, default):
        values = self.config.get(list_key, None)
        if values is not None:
            if isinstance(values, str):
                values = values.strip()
                if values.lower() in {'', 'none', 'null'}:
                    values = None
                else:
                    raise ValueError(
                        f'{list_key} must be passed as a Python/YAML list, e.g. '
                        f'--{list_key}=[0.2,0.3,0.35]'
                    )
        if values is not None:
            if len(values) != self.num_levels:
                raise ValueError(f'{list_key} must contain {self.num_levels} values, got {len(values)}.')
            return float(values[level])
        return float(self.config.get(scalar_key, default))

    def _load_level_teacher(self, path, format_name, value_key):
        if path is None:
            return None
        data = torch.load(path, map_location='cpu')
        if isinstance(data, dict) and data.get('format') == format_name:
            sample_ids = data['sample_id'].long()
            direct = (
                sample_ids.numel() > 0
                and int(sample_ids.min()) == 0
                and int(sample_ids.max()) == sample_ids.numel() - 1
                and torch.equal(sample_ids, torch.arange(sample_ids.numel(), dtype=torch.long))
            )
            data['_direct_sample_id_index'] = direct
            if not direct:
                data['_sample_id_to_row'] = {
                    int(sample_id): row
                    for row, sample_id in enumerate(sample_ids.tolist())
                }
            return data
        if isinstance(data, dict) and 'format' in data:
            raise ValueError(
                f'Unexpected teacher format in {path}: {data.get("format")}. '
                f'Expected {format_name}.'
            )
        if isinstance(data, dict):
            return {int(k): v for k, v in data.items()}
        return {int(record['sample_id']): record for record in data}

    def _load_copref(self, path):
        return self._load_level_teacher(path, 'colagr_copref_tensor_v1', 'copref')

    def _move_tensor_batch(self, batch):
        return {
            key: value.to(self.accelerator.device) if torch.is_tensor(value) else value
            for key, value in batch.items()
        }

    def get_batch_coprefs(self, sample_ids, split):
        data = self.copref_by_split.get(split)
        return self._get_batch_level_teacher(data, sample_ids, 'copref')

    def _get_batch_level_teacher(self, data, sample_ids, value_key):
        if data is None:
            return None
        if isinstance(data, dict) and data.get('format') == 'colagr_copref_tensor_v1':
            cpu_sample_ids = sample_ids.detach().cpu().long()
            if data.get('_direct_sample_id_index', False):
                row_indices = cpu_sample_ids
            else:
                row_indices = torch.tensor([
                    data['_sample_id_to_row'][int(sample_id)]
                    for sample_id in cpu_sample_ids.tolist()
                ], dtype=torch.long)
            return [
                level_teacher.index_select(0, row_indices).to(self.accelerator.device)
                for level_teacher in data[value_key]
            ]

        sample_ids = sample_ids.detach().cpu().tolist()
        result = []
        for level in range(self.num_levels):
            result.append(torch.stack([
                torch.as_tensor(data[int(sample_id)][value_key][level], dtype=torch.float)
                for sample_id in sample_ids
            ]).to(self.accelerator.device))
        return result

    def _attach_coprefs(self, batch, split):
        if split != 'train' and not self.eval_copref_diagnostics_enabled:
            return batch
        coprefs = self.get_batch_coprefs(batch['sample_id'], split)
        if coprefs is not None:
            batch['coprefs'] = coprefs
        return batch

    def fit(self, train_dataloader, val_dataloader):
        optimizer = AdamW(
            [param for param in self.model.parameters() if param.requires_grad],
            lr=self.config['lr'],
            weight_decay=self.config['weight_decay'],
        )
        total_n_steps = get_total_steps(self.config, train_dataloader)
        if total_n_steps == 0:
            self.log('No training steps needed.')
            return None, None

        scheduler_total_n_steps = self.config.get('scheduler_total_steps')
        if scheduler_total_n_steps is None:
            scheduler_total_n_steps = total_n_steps
        scheduler = get_scheduler(
            name='cosine',
            optimizer=optimizer,
            num_warmup_steps=self.config['warmup_steps'],
            num_training_steps=scheduler_total_n_steps,
        )

        self.model, optimizer, train_dataloader, val_dataloader, scheduler = self.accelerator.prepare(
            self.model, optimizer, train_dataloader, val_dataloader, scheduler
        )
        tracker = self.config.get('tracker', 'tensorboard')
        if tracker == 'tensorboard':
            project_name = get_file_name(self.config, suffix='')
            init_kwargs = {'tensorboard': {'flush_secs': 60}}
        elif tracker == 'wandb':
            project_name = self.config['wandb_project']
            run_name = self.config.get('wandb_run_name', None) or get_file_name(self.config, suffix='')
            init_kwargs = {'wandb': {'name': run_name, 'group': f"{self.config['dataset']}-{self.config['model']}"}}
        else:
            raise ValueError(f'Invalid tracker type: {tracker}. Must be "tensorboard" or "wandb".')
        self.accelerator.init_trackers(project_name=project_name, config=config_for_log(self.config), init_kwargs=init_kwargs)

        n_epochs = math.ceil(total_n_steps / (len(train_dataloader) * self.accelerator.num_processes))
        best_val_score = float('-inf')
        best_epoch = 0
        for epoch in range(n_epochs):
            self.model.train()
            total_loss, total_loss_gen, total_loss_pref = 0.0, 0.0, 0.0
            total_loss_selective_fuse = 0.0
            total_route_balance, total_route_div = 0.0, 0.0
            entropy_sums = torch.zeros(self.num_levels, dtype=torch.float)
            activation_sums = torch.zeros(self.num_levels, dtype=torch.float)
            alpha_sums = torch.zeros(self.num_levels, dtype=torch.float)
            lm_ce_sums = torch.zeros(self.num_levels, dtype=torch.float)
            fused_ce_sums = torch.zeros(self.num_levels, dtype=torch.float)
            fused_gain_sums = torch.zeros(self.num_levels, dtype=torch.float)
            col_prior_norm_sums = torch.zeros(self.num_levels, dtype=torch.float)
            pref_sums = torch.zeros(self.num_levels, dtype=torch.float)
            pref_counts = torch.zeros(self.num_levels, dtype=torch.float)
            alpha_counts = torch.zeros(self.num_levels, dtype=torch.float)
            selective_fuse_counts = torch.zeros(self.num_levels, dtype=torch.float)
            route_entropy_sums = torch.zeros(self.num_levels, dtype=torch.float)
            entropy_counts = torch.zeros(self.num_levels, dtype=torch.float)
            route_entropy_counts = torch.zeros(self.num_levels, dtype=torch.float)
            progress_bar = tqdm(train_dataloader, total=len(train_dataloader), desc=f'Epoch {epoch + 1}')
            for batch in progress_bar:
                batch = self._move_tensor_batch(batch)
                batch = self._attach_coprefs(batch, 'train')
                optimizer.zero_grad()
                outputs = self.model(batch)
                loss = outputs.loss
                self.accelerator.backward(loss)
                if self.config['max_grad_norm'] is not None:
                    clip_grad_norm_(self.model.parameters(), self.config['max_grad_norm'])
                optimizer.step()
                scheduler.step()
                total_loss += loss.item()
                total_loss_gen += float(outputs.loss_gen.item())
                total_loss_pref += float(outputs.loss_pref.item())
                total_loss_selective_fuse += float(
                    getattr(outputs, 'loss_selective_fuse', torch.tensor(0.0)).item()
                )
                total_route_balance += float(getattr(outputs, 'route_balance_loss', torch.tensor(0.0)).item())
                total_route_div += float(getattr(outputs, 'route_div_loss', torch.tensor(0.0)).item())
                for level, value in enumerate(getattr(outputs, 'per_level_entropy', [])):
                    entropy_sums[level] += float(value.item())
                    entropy_counts[level] += 1
                for level, value in enumerate(getattr(outputs, 'per_level_activation', [])):
                    activation_sums[level] += float(value.item())
                for level, value in enumerate(getattr(outputs, 'per_level_alpha', [])):
                    alpha_sums[level] += float(value.item())
                    alpha_counts[level] += 1
                for level, value in enumerate(getattr(outputs, 'per_level_lm_ce', [])):
                    lm_ce_sums[level] += float(value.item())
                    selective_fuse_counts[level] += 1
                for level, value in enumerate(getattr(outputs, 'per_level_fused_ce', [])):
                    fused_ce_sums[level] += float(value.item())
                for level, value in enumerate(getattr(outputs, 'per_level_fused_gain', [])):
                    fused_gain_sums[level] += float(value.item())
                for level, value in enumerate(getattr(outputs, 'per_level_col_prior_norm', [])):
                    col_prior_norm_sums[level] += float(value.item())
                for level, value in enumerate(getattr(outputs, 'per_level_pref', [])):
                    pref_sums[level] += float(value.item())
                    pref_counts[level] += 1
                for level, value in enumerate(getattr(outputs, 'per_level_route_entropy', [])):
                    route_entropy_sums[level] += float(value.item())
                    route_entropy_counts[level] += 1

            denom = max(len(train_dataloader), 1)
            self.accelerator.log({
                'Loss/train_loss': total_loss / denom,
                'Loss/loss_gen': total_loss_gen / denom,
                'Loss/loss_pref': total_loss_pref / denom,
                'Loss/loss_selective_fuse': total_loss_selective_fuse / denom,
                'Loss/route_balance_loss': total_route_balance / denom,
                'Loss/route_div_loss': total_route_div / denom,
            }, step=epoch + 1)
            unwrapped = self.accelerator.unwrap_model(self.model)
            gate_maxs = [
                self._level_float_config('cofuse_gate_maxs', 'cofuse_gate_max', level, 1.0)
                for level in range(self.num_levels)
            ]
            if self._config_bool('fixed_cofuse_gate', False):
                raw_gates = torch.ones_like(unwrapped.fuse_gates).detach().cpu()
                effective_gates = gate_maxs
            else:
                raw_gates = torch.sigmoid(unwrapped.fuse_gates).detach().cpu()
                effective_gates = [
                    gate_maxs[level] * float(raw_gates[level].item())
                    for level in range(self.num_levels)
                ]
            entropy_means = (entropy_sums / entropy_counts.clamp_min(1)).tolist()
            activation_ratios = (activation_sums / entropy_counts.clamp_min(1)).tolist()
            alpha_means = (alpha_sums / alpha_counts.clamp_min(1)).tolist()
            lm_ce_means = (lm_ce_sums / selective_fuse_counts.clamp_min(1)).tolist()
            fused_ce_means = (fused_ce_sums / selective_fuse_counts.clamp_min(1)).tolist()
            fused_gain_means = (fused_gain_sums / selective_fuse_counts.clamp_min(1)).tolist()
            col_prior_norm_means = (col_prior_norm_sums / selective_fuse_counts.clamp_min(1)).tolist()
            pref_means = (pref_sums / pref_counts.clamp_min(1)).tolist()
            route_entropy_means = (route_entropy_sums / route_entropy_counts.clamp_min(1)).tolist()
            log_message = (
                f'[Epoch {epoch + 1}] Train Loss: {total_loss / denom}; '
                f'gen={total_loss_gen / denom}; pref={total_loss_pref / denom}; '
                f'selective_fuse={total_loss_selective_fuse / denom}; '
                f'pref_levels={pref_means}; '
                f'route_balance={total_route_balance / denom}; route_div={total_route_div / denom}; '
                f'gates={raw_gates.tolist()}; effective_gates={effective_gates}; '
                f'entropy={entropy_means}; activation={activation_ratios}; alpha={alpha_means}; '
                f'lm_ce={lm_ce_means}; fused_ce={fused_ce_means}; '
                f'fused_gain={fused_gain_means}; col_prior_norm={col_prior_norm_means}; '
                f'route_entropy={route_entropy_means}'
            )
            self.log(log_message)

            if (epoch + 1) % self.config['eval_interval'] == 0:
                all_results = self.evaluate(val_dataloader, split='val')
                if self.accelerator.is_main_process:
                    for key in all_results:
                        self.accelerator.log({f'Val_Metric/{key}': all_results[key]}, step=epoch + 1)
                    self.log(f'[Epoch {epoch + 1}] Val Results: {all_results}')
                val_score = all_results[self.config['val_metric']]
                if val_score > best_val_score:
                    best_val_score = val_score
                    best_epoch = epoch + 1
                    if self.accelerator.is_main_process:
                        unwrapped_model = self.accelerator.unwrap_model(self.model) if self.config['use_ddp'] else self.model
                        torch.save(unwrapped_model.state_dict(), self.saved_model_ckpt)
                        self.log(f'[Epoch {epoch + 1}] Saved model checkpoint to {self.saved_model_ckpt}')

                if self.config['patience'] is not None and epoch + 1 - best_epoch >= self.config['patience']:
                    self.log(f'Early stopping at epoch {epoch + 1}')
                    break

        self.log(f'Best epoch: {best_epoch}, Best val score: {best_val_score}')
        return best_epoch, best_val_score

    def evaluate(self, dataloader, split='test'):
        self.model.eval()
        all_results = defaultdict(list)
        progress_bar = tqdm(dataloader, total=len(dataloader), desc=f'Eval - {split}')
        for batch in progress_bar:
            with torch.no_grad():
                batch = self._move_tensor_batch(batch)
                batch.pop('coprefs', None)
                unwrapped_model = self.accelerator.unwrap_model(self.model)
                preds = unwrapped_model.generate(batch, n_return_sequences=self.evaluator.maxk)
                if self.config['use_ddp']:
                    all_preds, all_labels = self.accelerator.gather_for_metrics((preds, batch['labels']))
                    results = self.evaluator.calculate_metrics(all_preds, all_labels)
                else:
                    results = self.evaluator.calculate_metrics(preds, batch['labels'])
                for key, value in results.items():
                    all_results[key].append(value)

        output_results = OrderedDict()
        for metric in self.config['metrics']:
            for k in self.config['topk']:
                key = f'{metric}@{k}'
                output_results[key] = torch.cat(all_results[key]).mean().item()
        return output_results

    def log(self, message, level='info'):
        return log(message, self.config['accelerator'], self.logger, level=level)

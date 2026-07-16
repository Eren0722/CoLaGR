import json
from types import SimpleNamespace

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import T5Config, T5ForConditionalGeneration

from genrec.model import AbstractModel

class CoLaGR(AbstractModel):
    """
    CoLaGR uses clean PSID semantic IDs as labels and inserts CoReason states
    inside the decoder before each SID level.
    """

    def __init__(self, config, dataset, tokenizer):
        super(CoLaGR, self).__init__(config, dataset, tokenizer)

        t5config = T5Config(
            num_layers=config['num_layers'],
            num_decoder_layers=config['num_decoder_layers'],
            d_model=config['d_model'],
            d_ff=config['d_ff'],
            num_heads=config['num_heads'],
            d_kv=config['d_kv'],
            dropout_rate=config['dropout_rate'],
            activation_function=config['activation_function'],
            vocab_size=tokenizer.vocab_size,
            pad_token_id=tokenizer.padding_token,
            eos_token_id=tokenizer.eos_token,
            decoder_start_token_id=0,
            feed_forward_proj=config['feed_forward_proj'],
            n_positions=tokenizer.max_token_seq_len,
        )
        self.t5 = T5ForConditionalGeneration(config=t5config)

        self.num_levels = tokenizer.n_digit
        self.use_coreason = self._config_bool('use_coreason', True)
        self.coreason_token_ids = torch.tensor(tokenizer.coreason_token_ids, dtype=torch.long)
        requested_routes = max(int(config.get('num_coreason_routes', 1)), 1)
        self.num_coreason_routes = requested_routes if self._config_bool('use_coroute', False) else 1
        self.use_coroute = self.use_coreason and self.num_coreason_routes > 1
        route_token_ids = getattr(tokenizer, 'coreason_route_token_ids', None)
        if route_token_ids is None:
            route_token_ids = [
                [int(tokenizer.coreason_token_ids[level]) for _ in range(self.num_coreason_routes)]
                for level in range(self.num_levels)
            ]
        self.coreason_route_token_ids = torch.tensor(route_token_ids, dtype=torch.long)
        self.level_token_ids = self._load_level_token_ids(config.get('level_token_ids_path'))
        self.register_buffer(
            'level_token_id_tensor',
            torch.nn.utils.rnn.pad_sequence(self.level_token_ids, batch_first=True, padding_value=0),
            persistent=False,
        )

        self.copref_heads = nn.ModuleList([
            nn.Linear(config['d_model'], len(self.level_token_ids[level]))
            for level in range(self.num_levels)
        ])
        self.fuse_gates = nn.Parameter(torch.full(
            (self.num_levels,),
            float(config.get('fuse_gate_init', -2.0)),
        ))
        self.token_to_local_idx = [
            {int(token): idx for idx, token in enumerate(tokens.tolist())}
            for tokens in self.level_token_ids
        ]
        self.valid_prefix_trie = self._load_valid_prefix_trie(config.get('valid_prefix_trie_path'))
        self._init_prefix_trie_tensors()

    def _expand_embedding_checkpoint_tensor(self, key, value, current):
        expandable = (
            key == 't5.shared.weight'
            or key == 't5.lm_head.weight'
            or key.endswith('.embed_tokens.weight')
        )
        if not expandable:
            return value
        if value.ndim != 2 or current.ndim != 2:
            return value
        if value.shape[1] != current.shape[1]:
            return value
        expanded = current.clone()
        copy_rows = min(value.shape[0], current.shape[0])
        expanded[:copy_rows] = value[:copy_rows]
        if self.use_coroute:
            route_tokens = self.coreason_route_token_ids
            for level in range(self.num_levels):
                source = int(route_tokens[level, 0])
                if source >= value.shape[0]:
                    continue
                for route in range(1, self.num_coreason_routes):
                    target = int(route_tokens[level, route])
                    if target < expanded.shape[0]:
                        expanded[target] = value[source]
        return expanded

    def _prepare_checkpoint_state_dict(self, state_dict):
        current_state = super().state_dict()
        prepared = {}
        for key, value in state_dict.items():
            if key in current_state and value.shape != current_state[key].shape:
                prepared[key] = self._expand_embedding_checkpoint_tensor(
                    key,
                    value,
                    current_state[key],
                )
            else:
                prepared[key] = value
        return prepared

    def load_state_dict(self, state_dict, strict=True, assign=False):
        prepared = self._prepare_checkpoint_state_dict(state_dict)
        try:
            incompatible = super().load_state_dict(prepared, strict=False, assign=assign)
        except TypeError:
            incompatible = super().load_state_dict(prepared, strict=False)
        missing = list(incompatible.missing_keys)
        unexpected = list(incompatible.unexpected_keys)
        if strict and (missing or unexpected):
            messages = []
            if missing:
                messages.append('Missing key(s) in state_dict: {}.'.format(
                    ', '.join(f'"{key}"' for key in missing)
                ))
            if unexpected:
                messages.append('Unexpected key(s) in state_dict: {}.'.format(
                    ', '.join(f'"{key}"' for key in unexpected)
                ))
            raise RuntimeError(
                'Error(s) in loading state_dict for {}:\n\t{}'.format(
                    self.__class__.__name__,
                    '\n\t'.join(messages),
                )
            )
        return incompatible

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
                if values.lower() not in {'', 'none', 'null'}:
                    raise ValueError(
                        f'{list_key} must be passed as a Python/YAML list, e.g. '
                        f'--{list_key}=[0.2,0.3,0.35]'
                    )
                values = None
        if values is not None:
            if len(values) != self.num_levels:
                raise ValueError(f'{list_key} must contain {self.num_levels} values, got {len(values)}.')
            return float(values[level])
        return float(self.config.get(scalar_key, default))

    def _cofuse_gate(self, level):
        gate_max = self._level_float_config('cofuse_gate_maxs', 'cofuse_gate_max', level, 1.0)
        if self._config_bool('fixed_cofuse_gate', False):
            return self.fuse_gates.new_tensor(gate_max)
        return gate_max * torch.sigmoid(self.fuse_gates[level])

    def _normalized_entropy(self, logits, mask=None, eps=1e-12):
        if mask is not None:
            logits = logits.masked_fill(~mask, -1e9)
            denom = mask.to(logits.dtype).sum(dim=-1).clamp_min(2.0)
        else:
            denom = torch.full(
                logits.shape[:-1],
                float(logits.shape[-1]),
                dtype=logits.dtype,
                device=logits.device,
            )
        probs = F.softmax(logits, dim=-1)
        log_probs = torch.log(probs.clamp_min(eps))
        entropy = -(probs * log_probs).sum(dim=-1)
        return entropy / torch.log(denom)

    def _prior_for_fusion(self, prior_logits, detach=True, mask=None):
        prior_temperature = max(float(self.config.get('prior_temperature', 1.0)), 1e-6)
        prior = prior_logits.detach() if detach else prior_logits
        prior = prior / prior_temperature
        if mask is not None:
            prior = prior.masked_fill(~mask, -1e9)
        if self._config_bool('center_prior_in_fusion', True):
            prior = F.log_softmax(prior, dim=-1)
            if mask is not None:
                denom = mask.to(prior.dtype).sum(dim=-1, keepdim=True).clamp_min(1.0)
                mean = prior.masked_fill(~mask, 0.0).sum(dim=-1, keepdim=True) / denom
                prior = prior - mean
                prior = prior.masked_fill(~mask, -1e9)
            else:
                prior = prior - prior.mean(dim=-1, keepdim=True)
        return prior

    def _fuse_logits(self, lm_logits, prior_logits, level, detach_prior=True, entropy_detach=True):
        if not self._config_bool('use_cofuse', True):
            return lm_logits, None, None, None, None

        gate = self._cofuse_gate(level)
        prior_for_fusion = self._prior_for_fusion(
            prior_logits,
            detach=detach_prior,
        )
        if self._config_bool('use_selective_cofuse', True):
            entropy_logits = lm_logits.detach() if entropy_detach else lm_logits
            entropy = self._normalized_entropy(entropy_logits)
            gate_temp = max(float(self.config.get('entropy_gate_temp', 0.10)), 1e-6)
            threshold = self._level_float_config('entropy_thresholds', 'entropy_threshold', level, 0.5)
            activate = torch.sigmoid((entropy - threshold) / gate_temp)
            alpha = gate * activate.to(lm_logits.dtype)
            effective_gate = alpha.unsqueeze(-1)
        else:
            entropy = None
            activate = None
            alpha = gate.expand(lm_logits.shape[0]).to(lm_logits.dtype)
            effective_gate = gate

        return lm_logits + effective_gate * prior_for_fusion, entropy, activate, alpha, prior_for_fusion

    @property
    def n_parameters(self) -> str:
        total_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        emb_params = sum(
            p.numel() for p in self.t5.get_input_embeddings().parameters()
            if p.requires_grad
        )
        return f'#Embedding parameters: {emb_params}\n' \
               f'#Non-embedding parameters: {total_params - emb_params}\n' \
               f'#Total trainable parameters: {total_params}\n'

    def _load_level_token_ids(self, path):
        if path is not None:
            level_token_ids = torch.load(path, map_location='cpu')
            return [torch.as_tensor(tokens, dtype=torch.long) for tokens in level_token_ids]

        level_token_ids = []
        offset = 1
        for size in self.tokenizer.codebook_sizes:
            level_token_ids.append(torch.arange(offset, offset + size, dtype=torch.long))
            offset += size
        return level_token_ids

    def _load_valid_prefix_trie(self, path):
        if path is None:
            return None
        with open(path, 'r') as file:
            raw_trie = json.load(file)
        trie = {}
        for prefix, tokens in raw_trie.items():
            trie[prefix] = [int(token) for token in tokens]
        return trie

    def _init_prefix_trie_tensors(self):
        max_token_id = max(int(tokens.max().item()) for tokens in self.level_token_ids)
        for level, mapping in enumerate(self.token_to_local_idx):
            token_to_local = torch.full((max_token_id + 1,), -1, dtype=torch.long)
            for token, local_idx in mapping.items():
                token_to_local[token] = int(local_idx)
            self.register_buffer(f'level_{level}_token_to_local', token_to_local, persistent=False)

        if self.valid_prefix_trie is None:
            self.prefix_trie_level_masks = None
            self.prefix_trie_level_strides = None
            return

        level_widths = [len(tokens) for tokens in self.level_token_ids]
        self.prefix_trie_level_masks = []
        self.prefix_trie_level_strides = []
        for level, width in enumerate(level_widths):
            if level == 0:
                strides = torch.empty(0, dtype=torch.long)
                num_prefixes = 1
            else:
                stride_values = []
                for pos in range(level):
                    stride = 1
                    for next_width in level_widths[pos + 1:level]:
                        stride *= next_width
                    stride_values.append(stride)
                strides = torch.tensor(stride_values, dtype=torch.long)
                num_prefixes = 1
                for prev_width in level_widths[:level]:
                    num_prefixes *= prev_width

            mask = torch.zeros(num_prefixes, width, dtype=torch.bool)
            mapping = self.token_to_local_idx[level]
            for prefix_key, allowed_tokens in self.valid_prefix_trie.items():
                prefix = [] if prefix_key == '' else [int(token) for token in prefix_key.split(',')]
                if len(prefix) != level:
                    continue
                prefix_id = 0
                valid_prefix = True
                for pos, token in enumerate(prefix):
                    local_idx = self.token_to_local_idx[pos].get(token)
                    if local_idx is None:
                        valid_prefix = False
                        break
                    prefix_id += local_idx * int(strides[pos].item())
                if not valid_prefix:
                    continue
                allowed_local = [mapping[token] for token in allowed_tokens if token in mapping]
                if allowed_local:
                    mask[prefix_id, torch.tensor(allowed_local, dtype=torch.long)] = True

            self.register_buffer(f'prefix_trie_mask_level_{level}', mask, persistent=False)
            self.register_buffer(f'prefix_trie_strides_level_{level}', strides, persistent=False)
            self.prefix_trie_level_masks.append(f'prefix_trie_mask_level_{level}')
            self.prefix_trie_level_strides.append(f'prefix_trie_strides_level_{level}')

    def _prefix_trie_row_mask(self, prefixes, level, device):
        if (
            not bool(self.config.get('use_prefix_trie', False))
            or self.prefix_trie_level_masks is None
        ):
            return None
        if not torch.is_tensor(prefixes):
            prefixes = torch.as_tensor(prefixes, dtype=torch.long, device=device)
        prefixes = prefixes.to(device)
        mask_table = getattr(self, self.prefix_trie_level_masks[level]).to(device)
        if level == 0:
            row_mask = mask_table[0].unsqueeze(0).expand(prefixes.shape[0], -1)
        else:
            local_parts = []
            valid_rows = torch.ones(prefixes.shape[0], dtype=torch.bool, device=device)
            for pos in range(level):
                token_to_local = getattr(self, f'level_{pos}_token_to_local').to(device)
                token_ids = prefixes[:, pos]
                in_range = token_ids < token_to_local.numel()
                local = torch.full_like(token_ids, -1)
                local[in_range] = token_to_local.index_select(0, token_ids[in_range])
                valid_rows &= local >= 0
                local_parts.append(local)
            strides = getattr(self, self.prefix_trie_level_strides[level]).to(device)
            prefix_ids = torch.zeros(prefixes.shape[0], dtype=torch.long, device=device)
            for pos, local in enumerate(local_parts):
                prefix_ids += local.clamp_min(0) * strides[pos]
            row_mask = mask_table.index_select(0, prefix_ids.clamp_max(mask_table.shape[0] - 1))
            row_mask = row_mask & valid_rows.unsqueeze(1)

        return row_mask

    def _apply_prefix_trie_mask(self, log_probs, prefixes, level):
        row_mask = self._prefix_trie_row_mask(prefixes, level, log_probs.device)
        if row_mask is None:
            return log_probs
        has_allowed = row_mask.any(dim=-1)
        if not has_allowed.any():
            return log_probs
        masked = torch.full_like(log_probs, float('-inf'))
        masked[row_mask] = log_probs[row_mask]
        masked[~has_allowed] = log_probs[~has_allowed]
        return masked

    def build_coreason_decoder_inputs(self, labels_sid):
        batch_size = labels_sid.shape[0]
        device = labels_sid.device
        decoder_input_ids = torch.empty(
            batch_size,
            1 + 2 * self.num_levels,
            dtype=torch.long,
            device=device,
        )
        decoder_input_ids[:, 0] = self.t5.config.decoder_start_token_id
        coreason_ids = self.coreason_token_ids.to(device)
        for level in range(self.num_levels):
            decoder_input_ids[:, 1 + 2 * level] = coreason_ids[level]
            decoder_input_ids[:, 2 + 2 * level] = labels_sid[:, level]
        coreason_positions = torch.arange(
            1, 1 + 2 * self.num_levels, 2, dtype=torch.long, device=device
        )
        return decoder_input_ids, coreason_positions

    def build_sampled_coroute_decoder_inputs(self, labels_sid):
        batch_size = labels_sid.shape[0]
        device = labels_sid.device
        route_ids = torch.randint(
            0,
            self.num_coreason_routes,
            (batch_size,),
            dtype=torch.long,
            device=device,
        )
        decoder_input_ids = torch.empty(
            batch_size,
            1 + 2 * self.num_levels,
            dtype=torch.long,
            device=device,
        )
        decoder_input_ids[:, 0] = self.t5.config.decoder_start_token_id
        route_tokens = self.coreason_route_token_ids.to(device)
        for level in range(self.num_levels):
            decoder_input_ids[:, 1 + 2 * level] = route_tokens[level].index_select(0, route_ids)
            decoder_input_ids[:, 2 + 2 * level] = labels_sid[:, level]
        decision_positions = torch.arange(
            1, 1 + 2 * self.num_levels, 2, dtype=torch.long, device=device
        )
        return decoder_input_ids, decision_positions, route_ids

    def build_plain_decoder_inputs(self, labels_sid):
        batch_size = labels_sid.shape[0]
        device = labels_sid.device
        decoder_input_ids = torch.empty(
            batch_size,
            self.num_levels,
            dtype=torch.long,
            device=device,
        )
        decoder_input_ids[:, 0] = self.t5.config.decoder_start_token_id
        if self.num_levels > 1:
            decoder_input_ids[:, 1:] = labels_sid[:, :self.num_levels - 1]
        decision_positions = torch.arange(self.num_levels, dtype=torch.long, device=device)
        return decoder_input_ids, decision_positions

    def build_decoder_inputs(self, labels_sid):
        if self.use_coreason:
            return self.build_coreason_decoder_inputs(labels_sid)
        return self.build_plain_decoder_inputs(labels_sid)

    def _copref_level_weights(self, device, dtype):
        raw_weights = self.config.get('copref_level_weights', [1.0] * self.num_levels)
        if isinstance(raw_weights, str):
            raw_weights = [
                float(value.strip())
                for value in raw_weights.strip().strip('[]').split(',')
                if value.strip()
            ]
        if len(raw_weights) != self.num_levels:
            raise ValueError(
                f'copref_level_weights must contain {self.num_levels} values, '
                f'got {len(raw_weights)}.'
            )
        return torch.tensor(raw_weights, dtype=dtype, device=device)

    def _copref_hidden(self, hidden, decision_positions, level):
        anchor = str(self.config.get('copref_anchor', 'coreason')).lower()
        if anchor == 'coreason':
            return hidden[:, decision_positions[level], :]
        if anchor == 'pre_sid_hidden':
            if self.use_coreason:
                pos = int(decision_positions[level].item()) - 1
                if pos < 0:
                    raise ValueError('Invalid pre_sid_hidden anchor before decoder start.')
                return hidden[:, pos, :]
            return hidden[:, decision_positions[level], :]
        raise ValueError(
            f'Unknown copref_anchor={anchor}. Expected coreason or pre_sid_hidden.'
        )

    def _scaled_lm_logits(self, hidden):
        if self.t5.config.tie_word_embeddings:
            hidden = hidden * (self.t5.model_dim ** -0.5)
        return self.t5.lm_head(hidden)

    def _target_local_indices(self, labels_sid, level):
        mapping = self.token_to_local_idx[level]
        values = [mapping[int(token)] for token in labels_sid[:, level].detach().cpu().tolist()]
        return torch.tensor(values, dtype=torch.long, device=labels_sid.device)

    def _level_lm_logits(self, hidden, level):
        lm_logits = self._scaled_lm_logits(hidden)
        level_tokens = self.level_token_ids[level].to(hidden.device)
        return lm_logits.index_select(dim=-1, index=level_tokens)

    def _forward_coroute_sampled(self, batch, labels_sid):
        decoder_input_ids, decision_positions, route_ids = self.build_sampled_coroute_decoder_inputs(labels_sid)
        outputs = self.t5(
            input_ids=batch['input_ids'],
            attention_mask=batch['attention_mask'],
            decoder_input_ids=decoder_input_ids,
            output_hidden_states=True,
            return_dict=True,
        )
        hidden = outputs.decoder_hidden_states[-1]

        loss_gen = labels_sid.new_tensor(0.0, dtype=hidden.dtype)
        loss_pref = labels_sid.new_tensor(0.0, dtype=hidden.dtype)
        per_level_gen, per_level_pref = [], []
        per_level_entropy, per_level_activation = [], []
        use_copref_loss = self._config_bool('use_copref_loss', True) and 'coprefs' in batch
        use_cofuse = self._config_bool('use_cofuse', True)

        for level in range(self.num_levels):
            z_level = hidden[:, decision_positions[level], :]
            pref_hidden = self._copref_hidden(hidden, decision_positions, level)
            lm_logits = self._level_lm_logits(z_level, level)
            prior_logits = self.copref_heads[level](pref_hidden)
            target_local = self._target_local_indices(labels_sid, level)

            if use_cofuse:
                fused_logits, entropy, activate, _, _ = self._fuse_logits(
                    lm_logits,
                    prior_logits,
                    level,
                    detach_prior=bool(self.config.get('detach_prior_in_fusion', True)),
                    entropy_detach=True,
                )
            else:
                fused_logits, entropy, activate = lm_logits, None, None

            cur_gen = F.cross_entropy(fused_logits, target_local)
            loss_gen = loss_gen + cur_gen
            per_level_gen.append(cur_gen.detach())
            if entropy is not None:
                per_level_entropy.append(entropy.detach().mean())
            if activate is not None:
                per_level_activation.append(activate.detach().to(hidden.dtype).mean())

            if use_copref_loss:
                level_weight = self._copref_level_weights(z_level.device, z_level.dtype)[level]
                copref = batch['coprefs'][level].to(z_level.device, dtype=z_level.dtype)
                copref = copref.clamp_min(0)
                copref = copref / copref.sum(dim=-1, keepdim=True).clamp_min(1e-12)
                cur_pref = F.kl_div(
                    F.log_softmax(prior_logits, dim=-1),
                    copref,
                    reduction='batchmean',
                )
                loss_pref = loss_pref + level_weight * cur_pref
                per_level_pref.append(cur_pref.detach())

        loss_gen = loss_gen / self.num_levels
        if use_copref_loss:
            level_weights = self._copref_level_weights(hidden.device, hidden.dtype)
            loss_pref = loss_pref / level_weights.sum().clamp_min(1e-12)

        loss = loss_gen + float(self.config.get('lambda_c', 0.05)) * loss_pref
        gate_l2 = float(self.config.get('gate_l2', 0.0))
        if gate_l2 > 0 and not self._config_bool('fixed_cofuse_gate', False):
            effective_gates = torch.stack([self._cofuse_gate(level) for level in range(self.num_levels)])
            loss = loss + gate_l2 * (effective_gates ** 2).mean()

        route_entropy = hidden.new_tensor(float(self.num_coreason_routes)).log()
        return SimpleNamespace(
            loss=loss,
            loss_gen=loss_gen.detach(),
            loss_pref=loss_pref.detach(),
            route_balance_loss=hidden.new_tensor(0.0),
            route_div_loss=hidden.new_tensor(0.0),
            per_level_gen=per_level_gen,
            per_level_pref=per_level_pref,
            per_level_entropy=per_level_entropy,
            per_level_activation=per_level_activation,
            per_level_route_entropy=[route_entropy.detach() for _ in range(self.num_levels)],
            sampled_route_ids=route_ids.detach(),
        )

    def forward(self, batch):
        labels = batch['labels']
        labels_sid = labels[:, :self.num_levels]
        if self.use_coroute:
            return self._forward_coroute_sampled(batch, labels_sid)

        decoder_input_ids, decision_positions = self.build_decoder_inputs(labels_sid)
        outputs = self.t5(
            input_ids=batch['input_ids'],
            attention_mask=batch['attention_mask'],
            decoder_input_ids=decoder_input_ids,
            output_hidden_states=True,
            return_dict=True,
        )
        hidden = outputs.decoder_hidden_states[-1]

        loss_gen = labels_sid.new_tensor(0.0, dtype=hidden.dtype)
        loss_pref = labels_sid.new_tensor(0.0, dtype=hidden.dtype)
        loss_selective_fuse = labels_sid.new_tensor(0.0, dtype=hidden.dtype)
        per_level_gen, per_level_pref = [], []
        per_level_entropy, per_level_activation = [], []
        per_level_alpha, per_level_lm_ce, per_level_fused_ce = [], [], []
        per_level_fused_gain, per_level_col_prior_norm = [], []
        use_copref_loss = self._config_bool('use_copref_loss', True) and 'coprefs' in batch
        use_cofuse = self._config_bool('use_cofuse', True)
        use_selective_fuse_loss = self._config_bool('use_selective_fuse_loss', False)
        gen_loss_on_fused_logits = self._config_bool('gen_loss_on_fused_logits', False)

        for level in range(self.num_levels):
            z_level = hidden[:, decision_positions[level], :]
            pref_hidden = self._copref_hidden(hidden, decision_positions, level)
            lm_logits = self._level_lm_logits(z_level, level)
            prior_logits = self.copref_heads[level](pref_hidden)
            detach_prior = bool(self.config.get('detach_prior_in_fusion', True))
            target_local = self._target_local_indices(labels_sid, level)

            if use_cofuse:
                fused_logits, entropy, activate, alpha, prior_for_fusion = self._fuse_logits(
                    lm_logits,
                    prior_logits,
                    level,
                    detach_prior=detach_prior,
                    entropy_detach=True,
                )
            else:
                fused_logits, entropy, activate = lm_logits, None, None
                alpha, prior_for_fusion = None, None

            lm_ce_per_sample = F.cross_entropy(lm_logits, target_local, reduction='none')
            if use_selective_fuse_loss and not gen_loss_on_fused_logits:
                cur_gen = lm_ce_per_sample.mean()
            else:
                cur_gen = F.cross_entropy(fused_logits, target_local)

            loss_gen = loss_gen + cur_gen
            per_level_gen.append(cur_gen.detach())
            if entropy is not None:
                per_level_entropy.append(entropy.detach().mean())
            if activate is not None:
                per_level_activation.append(activate.detach().to(hidden.dtype).mean())
            if use_selective_fuse_loss and use_cofuse and alpha is not None:
                prior_for_loss = self._prior_for_fusion(prior_logits, detach=False)
                if self._config_bool('detach_activation_in_fuse_loss', True):
                    alpha_for_loss = alpha.detach()
                    weight_for_loss = activate.detach() if activate is not None else alpha.detach().new_ones(alpha.shape)
                else:
                    alpha_for_loss = alpha
                    weight_for_loss = activate if activate is not None else alpha.new_ones(alpha.shape)
                if self._config_bool('detach_lm_in_fuse_loss', True):
                    lm_base = lm_logits.detach()
                else:
                    lm_base = lm_logits
                fused_logits_for_loss = lm_base + alpha_for_loss.unsqueeze(-1) * prior_for_loss
                fused_ce_per_sample = F.cross_entropy(
                    fused_logits_for_loss,
                    target_local,
                    reduction='none',
                )
                cur_selective_fuse = (weight_for_loss.to(hidden.dtype) * fused_ce_per_sample).mean()
                loss_selective_fuse = loss_selective_fuse + cur_selective_fuse

                with torch.no_grad():
                    per_level_alpha.append(alpha.detach().mean())
                    per_level_lm_ce.append(lm_ce_per_sample.detach().mean())
                    per_level_fused_ce.append(fused_ce_per_sample.detach().mean())
                    per_level_fused_gain.append((lm_ce_per_sample - fused_ce_per_sample).detach().mean())
                    per_level_col_prior_norm.append(prior_for_loss.detach().norm(dim=-1).mean())
            elif alpha is not None:
                per_level_alpha.append(alpha.detach().mean())

            if use_copref_loss:
                level_weight = self._copref_level_weights(z_level.device, z_level.dtype)[level]
                copref = batch['coprefs'][level].to(z_level.device, dtype=z_level.dtype)
                copref = copref.clamp_min(0)
                copref = copref / copref.sum(dim=-1, keepdim=True).clamp_min(1e-12)
                cur_pref = F.kl_div(
                    F.log_softmax(prior_logits, dim=-1),
                    copref,
                    reduction='batchmean',
                )
                loss_pref = loss_pref + level_weight * cur_pref
                per_level_pref.append(cur_pref.detach())

        loss_gen = loss_gen / self.num_levels
        if use_selective_fuse_loss and use_cofuse:
            loss_selective_fuse = loss_selective_fuse / self.num_levels
        if use_copref_loss:
            level_weights = self._copref_level_weights(hidden.device, hidden.dtype)
            loss_pref = loss_pref / level_weights.sum().clamp_min(1e-12)

        loss = (
            loss_gen
            + float(self.config.get('lambda_c', 0.05)) * loss_pref
        )
        if use_selective_fuse_loss and use_cofuse:
            loss = loss + float(self.config.get('selective_fuse_weight', 0.05)) * loss_selective_fuse
        gate_l2 = float(self.config.get('gate_l2', 0.0))
        if gate_l2 > 0 and not self._config_bool('fixed_cofuse_gate', False):
            effective_gates = torch.stack([self._cofuse_gate(level) for level in range(self.num_levels)])
            loss = loss + gate_l2 * (effective_gates ** 2).mean()

        return SimpleNamespace(
            loss=loss,
            loss_gen=loss_gen.detach(),
            loss_pref=loss_pref.detach(),
            loss_selective_fuse=loss_selective_fuse.detach(),
            per_level_gen=per_level_gen,
            per_level_pref=per_level_pref,
            per_level_entropy=per_level_entropy,
            per_level_activation=per_level_activation,
            per_level_alpha=per_level_alpha,
            per_level_lm_ce=per_level_lm_ce,
            per_level_fused_ce=per_level_fused_ce,
            per_level_fused_gain=per_level_fused_gain,
            per_level_col_prior_norm=per_level_col_prior_norm,
        )

    def _encode(self, batch):
        return self.t5.encoder(
            input_ids=batch['input_ids'],
            attention_mask=batch['attention_mask'],
            return_dict=True,
        )

    def _assert_teacher_free_generation(self, batch):
        if 'coprefs' in batch:
            raise ValueError(
                'CoLaGR generation is teacher-free: do not pass CoPref/coprefs '
                'to generate(), generate_colagr_greedy(), or generate_colagr_beam().'
            )

    def _generation_copref_hidden(self, decoder_hidden):
        anchor = str(self.config.get('copref_anchor', 'coreason')).lower()
        if anchor == 'pre_sid_hidden' and self.use_coreason and decoder_hidden.shape[1] >= 2:
            return decoder_hidden[:, -2, :]
        return decoder_hidden[:, -1, :]

    def _next_level_log_probs(self, encoder_outputs, attention_mask, decoder_ids, level, prefixes=None):
        decoder_outputs = self.t5.decoder(
            input_ids=decoder_ids,
            encoder_hidden_states=encoder_outputs.last_hidden_state,
            encoder_attention_mask=attention_mask,
            return_dict=True,
            use_cache=False,
        )
        z_level = decoder_outputs.last_hidden_state[:, -1, :]
        lm_logits = self._level_lm_logits(z_level, level)
        pref_hidden = self._generation_copref_hidden(decoder_outputs.last_hidden_state)
        prior_logits = self.copref_heads[level](pref_hidden)
        if self._config_bool('use_cofuse', True):
            lm_logits, _, _, _, _ = self._fuse_logits(
                lm_logits,
                prior_logits,
                level,
                detach_prior=False,
                entropy_detach=False,
            )
        log_probs = F.log_softmax(lm_logits, dim=-1)
        return self._apply_prefix_trie_mask(log_probs, prefixes, level)

    def _next_level_coroute_log_probs(self, encoder_outputs, attention_mask, decoder_ids, level, prefixes=None):
        batch_beams = decoder_ids.shape[0]
        device = decoder_ids.device
        eval_routes = max(
            1,
            min(int(self.config.get('coroute_eval_num_routes', self.num_coreason_routes)), self.num_coreason_routes),
        )
        route_tokens = self.coreason_route_token_ids.to(device)[level, :eval_routes]
        route_cols = route_tokens.view(1, eval_routes, 1).expand(batch_beams, -1, -1)
        decoder_for_step = torch.cat([
            decoder_ids.unsqueeze(1).expand(-1, eval_routes, -1),
            route_cols,
        ], dim=2)
        flat_decoder_ids = decoder_for_step.reshape(batch_beams * eval_routes, -1)
        base_indices = torch.arange(batch_beams, dtype=torch.long, device=device).repeat_interleave(
            eval_routes
        )
        chunk_size = int(self.config.get('coroute_decode_chunk_size', 8192))
        log_prob_chunks = []
        for start in range(0, flat_decoder_ids.shape[0], chunk_size):
            end = min(start + chunk_size, flat_decoder_ids.shape[0])
            cur_indices = base_indices[start:end]
            decoder_outputs = self.t5.decoder(
                input_ids=flat_decoder_ids[start:end],
                encoder_hidden_states=encoder_outputs.last_hidden_state.index_select(0, cur_indices),
                encoder_attention_mask=attention_mask.index_select(0, cur_indices),
                return_dict=True,
                use_cache=False,
            )
            z_level = decoder_outputs.last_hidden_state[:, -1, :]
            lm_logits = self._level_lm_logits(z_level, level)
            pref_hidden = self._generation_copref_hidden(decoder_outputs.last_hidden_state)
            prior_logits = self.copref_heads[level](pref_hidden)
            cur_prefixes = None
            if prefixes is not None:
                route_prefixes = prefixes.unsqueeze(1).expand(-1, eval_routes, -1)
                cur_prefixes = route_prefixes.reshape(batch_beams * eval_routes, -1)[start:end]
            if self._config_bool('use_cofuse', True):
                lm_logits, _, _, _, _ = self._fuse_logits(
                    lm_logits,
                    prior_logits,
                    level,
                    detach_prior=False,
                    entropy_detach=False,
                )
            log_prob_chunks.append(F.log_softmax(lm_logits, dim=-1))
        route_log_probs = torch.cat(log_prob_chunks, dim=0)
        if bool(self.config.get('use_prefix_trie', False)) and self.valid_prefix_trie is not None:
            if prefixes is None:
                prefixes = torch.empty(batch_beams, 0, dtype=torch.long, device=device)
            route_prefixes = prefixes.unsqueeze(1).expand(-1, eval_routes, -1)
            route_log_probs = self._apply_prefix_trie_mask(
                route_log_probs,
                route_prefixes.reshape(batch_beams * eval_routes, -1),
                level,
            )
        route_log_alpha = route_log_probs.new_full(
            (batch_beams, eval_routes),
            -torch.log(route_log_probs.new_tensor(float(eval_routes))),
        )
        route_log_probs = route_log_probs.view(batch_beams, eval_routes, -1)
        return route_log_probs + route_log_alpha.unsqueeze(-1), decoder_for_step

    def generate_colagr_greedy(self, batch, n_return_sequences=1):
        self._assert_teacher_free_generation(batch)
        batch_size = batch['input_ids'].shape[0]
        device = batch['input_ids'].device
        encoder_outputs = self._encode(batch)
        decoder_ids = torch.full(
            (batch_size, 1),
            self.t5.config.decoder_start_token_id,
            dtype=torch.long,
            device=device,
        )
        generated = []
        prefix_tokens = torch.empty(batch_size, 0, dtype=torch.long, device=device)
        coreason_ids = self.coreason_token_ids.to(device)

        for level in range(self.num_levels):
            if self.use_coreason:
                coreason_col = torch.full(
                    (batch_size, 1), int(coreason_ids[level]), dtype=torch.long, device=device
                )
                decoder_for_step = torch.cat([decoder_ids, coreason_col], dim=1)
            else:
                decoder_for_step = decoder_ids
            log_probs = self._next_level_log_probs(
                encoder_outputs,
                batch['attention_mask'],
                decoder_for_step,
                level,
                prefixes=prefix_tokens,
            )
            log_probs = self._apply_prefix_trie_mask(log_probs, prefix_tokens, level)
            next_local = log_probs.argmax(dim=-1)
            next_global = self.level_token_ids[level].to(device).index_select(0, next_local)
            generated.append(next_global)
            prefix_tokens = torch.cat([prefix_tokens, next_global.unsqueeze(1)], dim=1)
            decoder_ids = torch.cat([decoder_for_step, next_global.unsqueeze(1)], dim=1)

        pred_tokens = torch.stack(generated, dim=1).unsqueeze(1)
        if n_return_sequences > 1:
            pred_tokens = pred_tokens.repeat(1, n_return_sequences, 1)
        return pred_tokens

    def generate_colagr_beam(self, batch, n_return_sequences=1):
        self._assert_teacher_free_generation(batch)
        batch_size = batch['input_ids'].shape[0]
        num_beams = max(int(self.config.get('num_beams', 1)), n_return_sequences)
        device = batch['input_ids'].device
        coreason_ids = self.coreason_token_ids.to(device)
        encoder_outputs = self._encode(batch)

        decoder_ids = torch.full(
            (batch_size, 1, 1),
            self.t5.config.decoder_start_token_id,
            dtype=torch.long,
            device=device,
        )
        sid_tokens = torch.empty(batch_size, 1, 0, dtype=torch.long, device=device)
        beam_scores = torch.zeros(batch_size, 1, dtype=torch.float, device=device)

        for level in range(self.num_levels):
            active_beams = decoder_ids.shape[1]
            seq_len = decoder_ids.shape[2]
            if self.use_coreason:
                coreason_col = torch.full(
                    (batch_size, active_beams, 1),
                    int(coreason_ids[level]),
                    dtype=torch.long,
                    device=device,
                )
                decoder_for_step = torch.cat([decoder_ids, coreason_col], dim=2)
            else:
                decoder_for_step = decoder_ids
            step_seq_len = decoder_for_step.shape[2]
            flat_decoder_ids = decoder_for_step.reshape(batch_size * active_beams, step_seq_len)
            flat_encoder_outputs = SimpleNamespace(
                last_hidden_state=encoder_outputs.last_hidden_state.repeat_interleave(active_beams, dim=0)
            )
            flat_attention_mask = batch['attention_mask'].repeat_interleave(active_beams, dim=0)
            log_probs = self._next_level_log_probs(
                flat_encoder_outputs,
                flat_attention_mask,
                flat_decoder_ids,
                level,
                prefixes=sid_tokens.reshape(batch_size * active_beams, -1),
            )
            if bool(self.config.get('use_prefix_trie', False)) and self.valid_prefix_trie is not None:
                prefixes = sid_tokens.reshape(batch_size * active_beams, -1)
                log_probs = self._apply_prefix_trie_mask(log_probs, prefixes, level)

            level_width = log_probs.shape[-1]
            log_probs = log_probs.view(batch_size, active_beams, level_width)
            candidate_scores = beam_scores.unsqueeze(-1) + log_probs
            next_beam_count = min(num_beams, active_beams * level_width)
            top_scores, top_indices = torch.topk(
                candidate_scores.view(batch_size, -1),
                k=next_beam_count,
                dim=-1,
            )
            selected_beams = torch.div(top_indices, level_width, rounding_mode='floor')
            selected_local = top_indices.remainder(level_width)

            selected_decoder = decoder_for_step.gather(
                1,
                selected_beams.unsqueeze(-1).expand(batch_size, next_beam_count, step_seq_len),
            )
            selected_sid = sid_tokens.gather(
                1,
                selected_beams.unsqueeze(-1).expand(batch_size, next_beam_count, sid_tokens.shape[2]),
            )
            level_tokens = self.level_token_ids[level].to(device)
            next_global = level_tokens.index_select(0, selected_local.reshape(-1)).view(batch_size, next_beam_count)
            decoder_ids = torch.cat([selected_decoder, next_global.unsqueeze(-1)], dim=2)
            sid_tokens = torch.cat([selected_sid, next_global.unsqueeze(-1)], dim=2)
            beam_scores = top_scores

        if sid_tokens.shape[1] < n_return_sequences:
            pad_count = n_return_sequences - sid_tokens.shape[1]
            sid_tokens = torch.cat([
                sid_tokens,
                sid_tokens[:, -1:, :].expand(batch_size, pad_count, self.num_levels),
            ], dim=1)
        return sid_tokens[:, :n_return_sequences, :]

    def _aggregate_coroute_beams(self, sid_tokens, beam_scores, top_k):
        method = self.config.get('coroute_aggregation_method', 'agg_max')
        batch_size, num_beams, num_levels = sid_tokens.shape
        result = torch.zeros(
            batch_size,
            top_k,
            num_levels,
            dtype=sid_tokens.dtype,
            device=sid_tokens.device,
        )
        for batch_idx in range(batch_size):
            scores_by_sid = {}
            for beam_idx in range(num_beams):
                sid = tuple(int(token) for token in sid_tokens[batch_idx, beam_idx].tolist())
                score = float(beam_scores[batch_idx, beam_idx].item())
                if method == 'agg_sum':
                    if sid in scores_by_sid:
                        old_score, old_sid = scores_by_sid[sid]
                        max_score = max(old_score, score)
                        merged = max_score + torch.log(
                            torch.exp(sid_tokens.new_tensor(old_score - max_score, dtype=torch.float))
                            + torch.exp(sid_tokens.new_tensor(score - max_score, dtype=torch.float))
                        ).item()
                        scores_by_sid[sid] = (merged, old_sid)
                    else:
                        scores_by_sid[sid] = (score, sid)
                else:
                    if sid not in scores_by_sid or score > scores_by_sid[sid][0]:
                        scores_by_sid[sid] = (score, sid)
            sorted_sids = sorted(scores_by_sid.values(), key=lambda item: -item[0])[:top_k]
            for rank, (_, sid) in enumerate(sorted_sids):
                result[batch_idx, rank] = torch.tensor(sid, dtype=sid_tokens.dtype, device=sid_tokens.device)
        return result

    def generate_coroute_beam(self, batch, n_return_sequences=1):
        self._assert_teacher_free_generation(batch)
        batch_size = batch['input_ids'].shape[0]
        num_beams = max(int(self.config.get('num_beams', 1)), n_return_sequences)
        device = batch['input_ids'].device
        encoder_outputs = self._encode(batch)

        decoder_ids = torch.full(
            (batch_size, 1, 1),
            self.t5.config.decoder_start_token_id,
            dtype=torch.long,
            device=device,
        )
        sid_tokens = torch.empty(batch_size, 1, 0, dtype=torch.long, device=device)
        beam_scores = torch.zeros(batch_size, 1, dtype=torch.float, device=device)

        for level in range(self.num_levels):
            active_beams = decoder_ids.shape[1]
            seq_len = decoder_ids.shape[2]
            flat_decoder_ids = decoder_ids.reshape(batch_size * active_beams, seq_len)
            flat_encoder_outputs = SimpleNamespace(
                last_hidden_state=encoder_outputs.last_hidden_state.repeat_interleave(active_beams, dim=0)
            )
            flat_attention_mask = batch['attention_mask'].repeat_interleave(active_beams, dim=0)
            flat_prefixes = sid_tokens.reshape(batch_size * active_beams, -1)
            route_log_probs, decoder_for_step = self._next_level_coroute_log_probs(
                flat_encoder_outputs,
                flat_attention_mask,
                flat_decoder_ids,
                level,
                prefixes=flat_prefixes,
            )
            level_width = route_log_probs.shape[-1]
            route_count = route_log_probs.shape[1]
            step_seq_len = decoder_for_step.shape[-1]
            route_log_probs = route_log_probs.view(
                batch_size,
                active_beams,
                route_count,
                level_width,
            )
            mixed_log_probs = torch.logsumexp(route_log_probs, dim=2)
            best_routes = route_log_probs.argmax(dim=2)
            candidate_scores = beam_scores.unsqueeze(-1) + mixed_log_probs
            next_beam_count = min(num_beams, active_beams * level_width)
            top_scores, top_indices = torch.topk(
                candidate_scores.view(batch_size, -1),
                k=next_beam_count,
                dim=-1,
            )
            selected_beams = torch.div(
                top_indices,
                level_width,
                rounding_mode='floor',
            )
            selected_local = top_indices.remainder(level_width)
            batch_indices = torch.arange(batch_size, dtype=torch.long, device=device).unsqueeze(1)
            selected_routes = best_routes[batch_indices, selected_beams, selected_local]

            decoder_for_step = decoder_for_step.view(
                batch_size,
                active_beams,
                route_count,
                step_seq_len,
            )
            selected_decoder = decoder_for_step.gather(
                1,
                selected_beams[:, :, None, None].expand(batch_size, next_beam_count, route_count, step_seq_len),
            ).gather(
                2,
                selected_routes[:, :, None, None].expand(batch_size, next_beam_count, 1, step_seq_len),
            ).squeeze(2)
            selected_sid = sid_tokens.gather(
                1,
                selected_beams.unsqueeze(-1).expand(batch_size, next_beam_count, sid_tokens.shape[2]),
            )
            level_tokens = self.level_token_ids[level].to(device)
            next_global = level_tokens.index_select(0, selected_local.reshape(-1)).view(batch_size, next_beam_count)
            decoder_ids = torch.cat([selected_decoder, next_global.unsqueeze(-1)], dim=2)
            sid_tokens = torch.cat([selected_sid, next_global.unsqueeze(-1)], dim=2)
            beam_scores = top_scores

        if sid_tokens.shape[1] < n_return_sequences:
            pad_count = n_return_sequences - sid_tokens.shape[1]
            sid_tokens = torch.cat([
                sid_tokens,
                sid_tokens[:, -1:, :].expand(batch_size, pad_count, self.num_levels),
            ], dim=1)
            beam_scores = torch.cat([
                beam_scores,
                beam_scores[:, -1:].expand(batch_size, pad_count),
            ], dim=1)
        return self._aggregate_coroute_beams(sid_tokens, beam_scores, n_return_sequences)

    def generate(self, batch, n_return_sequences=1):
        self._assert_teacher_free_generation(batch)
        with torch.no_grad():
            if self.use_coroute:
                eval_routes = max(
                    1,
                    min(int(self.config.get('coroute_eval_num_routes', self.num_coreason_routes)), self.num_coreason_routes),
                )
                if eval_routes == 1:
                    return self.generate_colagr_beam(batch, n_return_sequences)
                return self.generate_coroute_beam(batch, n_return_sequences)
            if int(self.config.get('num_beams', 1)) == 1:
                return self.generate_colagr_greedy(batch, n_return_sequences)
            return self.generate_colagr_beam(batch, n_return_sequences)

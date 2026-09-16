import json
import math
from types import SimpleNamespace

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import T5Config, T5ForConditionalGeneration

from genrec.model import AbstractModel


class CoLaGR(AbstractModel):
    """
    CoLaGR keeps PSID semantic IDs as labels. The active mainline uses CoPref
    states before SID actions; CoLeaf calibration is applied after generation.
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
        if self._config_bool('use_coreason', False):
            raise ValueError(
                'CoReason is not part of this release. Use the stepwise CoPref '
                'and CoLeaf ablations documented in colagr/README.md.'
            )
        self.use_coreason = False
        self.grounding_anchor = str(config.get('grounding_anchor', 'cp')).strip().lower()
        if self.grounding_anchor not in {'cp', 'action'}:
            raise ValueError("grounding_anchor must be either 'cp' or 'action'.")
        self.use_copref_module = (
            self._config_bool('use_copref_module', True)
            and self.grounding_anchor == 'cp'
        )
        self.use_copref_supervision = self._config_bool('use_copref_loss', True)
        self.copref_reasoning_mode = str(
            config.get('copref_reasoning_mode', 'per_level')
        ).strip().lower()
        if self.copref_reasoning_mode not in {'per_level', 'one_shot'}:
            raise ValueError("copref_reasoning_mode must be 'per_level' or 'one_shot'.")
        self.direct_copref_alignment = self._config_bool(
            'direct_copref_alignment', False
        )
        if self.direct_copref_alignment and self.use_copref_module:
            raise ValueError(
                'direct_copref_alignment requires use_copref_module=false.'
            )
        self.coreason_token_ids = torch.tensor(tokenizer.coreason_token_ids, dtype=torch.long)
        self.copref_token_ids = torch.tensor(
            getattr(tokenizer, 'copref_token_ids', tokenizer.coreason_token_ids), dtype=torch.long
        )
        self.level_token_ids = self._load_level_token_ids(config.get('level_token_ids_path'))
        self.register_buffer(
            'level_token_id_tensor',
            torch.nn.utils.rnn.pad_sequence(self.level_token_ids, batch_first=True, padding_value=0),
            persistent=False,
        )
        for level, tokens in enumerate(self.level_token_ids):
            self.register_buffer(
                f'level_{level}_token_ids',
                tokens.clone(),
                persistent=False,
            )

        self.copref_heads = nn.ModuleList([
            nn.Linear(config['d_model'], len(self.level_token_ids[level]))
            for level in range(self.num_levels)
        ])
        # R is a latent decision position supervised by future collaborative value.
        self.coreason_heads = nn.ModuleList([
            nn.Linear(config['d_model'], len(self.level_token_ids[level]))
            for level in range(self.num_levels)
        ])
        self.coreason_mode = str(config.get('coreason_mode', 'future_value')).strip().lower()
        if self.use_coreason and self.coreason_mode != 'future_value':
            raise ValueError("The only supported CoReason mode is 'future_value'.")
        self.use_future_value_reason = self.use_coreason
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

    def _refine_action(self, action_hidden, pref_hidden=None):
        """Return the state that directly makes the next SID decision."""
        return action_hidden

    def _collab_kl_loss(self, logits, target):
        """Dense collaborative supervision over the complete branch distribution."""
        target = target.clamp_min(0)
        target = target / target.sum(dim=-1, keepdim=True).clamp_min(1e-12)
        return F.kl_div(F.log_softmax(logits, dim=-1), target, reduction='batchmean')

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

    def build_plain_decoder_inputs(self, labels_sid, with_pref_positions=False):
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
        pref_positions = decision_positions if with_pref_positions else None
        return decoder_input_ids, decision_positions, pref_positions

    def build_decoder_inputs(self, labels_sid):
        batch_size = labels_sid.shape[0]
        device = labels_sid.device
        # The validated stable path uses one grounded action state per SID
        # level.  CoPref supervises that state directly when CoReason is on.
        # Keeping this layout avoids turning the effective action into a
        # separately supervised CP/R pair whose extra state can dilute the
        # original generation objective.
        if self.use_future_value_reason:
            stride, width = 3, 1 + 3 * self.num_levels
        elif self.use_coreason:
            stride, width = 2, 1 + 2 * self.num_levels
        elif self.use_copref_module:
            if self.copref_reasoning_mode == 'one_shot':
                # A single CP state precedes all SID actions.  It cannot see
                # later generated prefixes, making it a direct control for
                # per-level collaborative reasoning rather than extra compute.
                width = 2 + self.num_levels
                decoder_input_ids = torch.empty(
                    batch_size, width, dtype=torch.long, device=device
                )
                decoder_input_ids[:, 0] = self.t5.config.decoder_start_token_id
                decoder_input_ids[:, 1] = self.copref_token_ids.to(device)[0]
                decoder_input_ids[:, 2:] = labels_sid
                action_positions = torch.arange(
                    1, width - 1, dtype=torch.long, device=device
                )
                pref_positions = torch.ones(
                    self.num_levels, dtype=torch.long, device=device
                )
                return decoder_input_ids, action_positions, pref_positions
            stride, width = 2, 1 + 2 * self.num_levels
        else:
            return self.build_plain_decoder_inputs(
                labels_sid,
                with_pref_positions=(
                    self.use_copref_supervision
                    and self.grounding_anchor == 'action'
                ),
            )

        decoder_input_ids = torch.empty(batch_size, width, dtype=torch.long, device=device)
        decoder_input_ids[:, 0] = self.t5.config.decoder_start_token_id
        copref_ids = self.copref_token_ids.to(device)
        coreason_ids = self.coreason_token_ids.to(device)
        for level in range(self.num_levels):
            offset = 1 + stride * level
            if self.use_future_value_reason:
                decoder_input_ids[:, offset] = copref_ids[level]
                decoder_input_ids[:, offset + 1] = coreason_ids[level]
                decoder_input_ids[:, offset + 2] = labels_sid[:, level]
            elif self.use_copref_module:
                decoder_input_ids[:, offset] = copref_ids[level]
                decoder_input_ids[:, offset + 1] = labels_sid[:, level]

        action_start = 2 if (self.use_future_value_reason) else 1
        action_positions = torch.arange(
            action_start,
            width,
            stride,
            dtype=torch.long,
            device=device,
        )
        if not self.use_copref_module:
            pref_positions = None
        elif self.use_future_value_reason:
            pref_positions = action_positions - 1
        elif self.use_coreason:
            pref_positions = action_positions
        else:
            # CoPref-only uses the CP position both as its grounded state and
            # as the action representation; there is no synthetic anchor.
            pref_positions = action_positions
        return decoder_input_ids, action_positions, pref_positions

    def _scaled_lm_logits(self, hidden):
        if self.t5.config.tie_word_embeddings:
            hidden = hidden * (self.t5.model_dim ** -0.5)
        return self.t5.lm_head(hidden)

    def _target_local_indices(self, labels_sid, level):
        lookup = getattr(self, f'level_{level}_token_to_local')
        token_ids = labels_sid[:, level].long()
        local = lookup.index_select(0, token_ids)
        if (local < 0).any():
            raise ValueError(f'Found SID token outside level {level} vocabulary.')
        return local

    def _normalize_probs(self, probs):
        probs = probs.clamp_min(0)
        return probs / probs.sum(dim=-1, keepdim=True).clamp_min(1e-12)

    def _level_lm_logits(self, hidden, level):
        lm_logits = self._scaled_lm_logits(hidden)
        level_tokens = getattr(self, f'level_{level}_token_ids')
        return lm_logits.index_select(dim=-1, index=level_tokens)

    def forward(self, batch):
        if self.use_future_value_reason:
            return self.forward_future_value_reason(batch)

        labels_sid = batch['labels'][:, :self.num_levels]
        decoder_input_ids, action_positions, pref_positions = self.build_decoder_inputs(labels_sid)
        # Use the standard T5 wrapper for training.  It follows the reference
        # Latte execution path and avoids a slower Python-level split between
        # encoder and decoder while keeping generation separate below.
        outputs = self.t5(
            input_ids=batch['input_ids'],
            attention_mask=batch['attention_mask'],
            decoder_input_ids=decoder_input_ids,
            output_hidden_states=True,
            return_dict=True,
        )
        hidden = outputs.decoder_hidden_states[-1]
        loss_gen = hidden.new_zeros(())
        loss_pref = hidden.new_zeros(())
        loss_reason = hidden.new_zeros(())
        pref_losses = []
        reason_losses = []
        use_pref_teacher = (
            self.use_copref_supervision
            and 'coprefs' in batch
            and pref_positions is not None
        )
        # Avoid building an unused reason-head graph when the coefficient is
        # zero, which is the default for the clean CoReason ablation.
        refined_states = []
        pref_hiddens = []
        for level in range(self.num_levels):
            action_hidden = hidden[:, action_positions[level], :]
            pref_hidden = (
                hidden[:, pref_positions[level], :]
                if pref_positions is not None
                else None
            )
            refined = self._refine_action(action_hidden, pref_hidden)
            refined_states.append(refined)
            pref_hiddens.append(pref_hidden)

        # Project all SID decisions in one vocabulary GEMM, then select each
        # level's codebook. This preserves logits exactly while reducing small
        # per-level kernel launches.
        all_sid_logits = self._scaled_lm_logits(torch.stack(refined_states, dim=1))
        for level, pref_hidden in enumerate(pref_hiddens):
            sid_logits = all_sid_logits[:, level, :].index_select(
                dim=-1,
                index=getattr(self, f'level_{level}_token_ids'),
            )
            target_local = self._target_local_indices(labels_sid, level)
            loss_gen = loss_gen + F.cross_entropy(sid_logits, target_local)
            if use_pref_teacher:
                teacher = batch['coprefs'][level].to(sid_logits.device, sid_logits.dtype)
                teacher = teacher.clamp_min(0)
                teacher = teacher / teacher.sum(dim=-1, keepdim=True).clamp_min(1e-12)
                if self.direct_copref_alignment:
                    pref_logits = sid_logits
                else:
                    pref_logits = self.copref_heads[level](pref_hidden)
                pref_losses.append(self._collab_kl_loss(pref_logits, teacher))
        loss_gen = loss_gen / self.num_levels
        if pref_losses:
            loss_pref = torch.stack(pref_losses).mean()
        if reason_losses:
            loss_reason = torch.stack(reason_losses).mean()
        lambda_pref = float(self.config.get('lambda_pref', self.config.get('lambda_c', 0.10)))
        lambda_reason = float(self.config.get('lambda_reason', 1.0))
        loss = (
            loss_gen
            + lambda_pref * loss_pref
            + lambda_reason * loss_reason
        )
        return SimpleNamespace(
            loss=loss,
            loss_gen=loss_gen.detach(),
            loss_pref=loss_pref.detach(),
            loss_reason=loss_reason.detach(),
            loss_selective_fuse=hidden.new_zeros(()),
            per_level_gen=[],
            per_level_pref=pref_losses,
            per_level_reason=reason_losses,
            per_level_entropy=[],
            per_level_activation=[],
            per_level_alpha=[],
            per_level_lm_ce=[],
            per_level_fused_ce=[],
            per_level_fused_gain=[],
            per_level_col_prior_norm=[],
        )

    def forward_future_value_reason(self, batch):
        """Train CP on current preference and R on future branch utility.

        The decoder positions are CP -> R -> SID for every SID level.  The
        future-value target is produced offline from training-only
        collaborative transitions and is deliberately separate from the CP
        distribution.
        """
        labels_sid = batch['labels'][:, :self.num_levels]
        decoder_input_ids, action_positions, pref_positions = self.build_decoder_inputs(labels_sid)
        outputs = self.t5(
            input_ids=batch['input_ids'],
            attention_mask=batch['attention_mask'],
            decoder_input_ids=decoder_input_ids,
            output_hidden_states=True,
            return_dict=True,
        )
        hidden = outputs.decoder_hidden_states[-1]
        loss_gen = hidden.new_zeros(())
        loss_pref = hidden.new_zeros(())
        loss_reason = hidden.new_zeros(())
        pref_losses, reason_losses = [], []
        have_pref = self.use_copref_supervision and 'coprefs' in batch
        have_reason = 'coreason_targets' in batch

        for level in range(self.num_levels):
            pref_hidden = hidden[:, pref_positions[level], :]
            reason_hidden = hidden[:, action_positions[level], :]
            sid_logits = self._level_lm_logits(reason_hidden, level)
            target_local = self._target_local_indices(labels_sid, level)
            loss_gen = loss_gen + F.cross_entropy(sid_logits, target_local)

            if have_pref:
                teacher = self._normalize_probs(
                    batch['coprefs'][level].to(pref_hidden.device, pref_hidden.dtype)
                )
                pref_losses.append(self._collab_kl_loss(self.copref_heads[level](pref_hidden), teacher))
            if have_reason:
                reason_target = self._normalize_probs(
                    batch['coreason_targets'][level].to(reason_hidden.device, reason_hidden.dtype)
                )
                reason_losses.append(
                    self._collab_kl_loss(self.coreason_heads[level](reason_hidden), reason_target)
                )

        loss_gen = loss_gen / self.num_levels
        if pref_losses:
            loss_pref = torch.stack(pref_losses).mean()
        if reason_losses:
            loss_reason = torch.stack(reason_losses).mean()
        lambda_pref = float(self.config.get('lambda_pref', self.config.get('lambda_c', 0.10)))
        lambda_reason = float(self.config.get('lambda_reason', 1.0))
        loss = loss_gen + lambda_pref * loss_pref + lambda_reason * loss_reason
        return SimpleNamespace(
            loss=loss,
            loss_gen=loss_gen.detach(),
            loss_pref=loss_pref.detach(),
            loss_reason=loss_reason.detach(),
            loss_selective_fuse=hidden.new_zeros(()),
            per_level_gen=[],
            per_level_pref=pref_losses,
            per_level_reason=reason_losses,
            per_level_entropy=[],
            per_level_activation=[],
            per_level_alpha=[],
            per_level_lm_ce=[],
            per_level_fused_ce=[],
            per_level_fused_gain=[],
            per_level_col_prior_norm=[],
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

    def _next_level_log_probs(self, encoder_outputs, attention_mask, decoder_ids, level, prefixes=None):
        decoder_outputs = self.t5.decoder(
            input_ids=decoder_ids,
            encoder_hidden_states=encoder_outputs.last_hidden_state,
            encoder_attention_mask=attention_mask,
            return_dict=True,
            use_cache=False,
        )
        action_hidden = decoder_outputs.last_hidden_state[:, -1, :]
        pref_hidden = None
        if self.use_copref_module:
            pref_hidden = action_hidden if self.use_coreason else decoder_outputs.last_hidden_state[:, -1, :]
        z_level = self._refine_action(action_hidden, pref_hidden)
        lm_logits = self._level_lm_logits(z_level, level)
        log_probs = F.log_softmax(lm_logits, dim=-1)
        return self._apply_prefix_trie_mask(log_probs, prefixes, level)

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
        copref_ids = self.copref_token_ids.to(device)
        coreason_ids = self.coreason_token_ids.to(device)

        for level in range(self.num_levels):
            if self.use_future_value_reason:
                step_tokens = [int(copref_ids[level]), int(coreason_ids[level])]
            elif self.use_copref_module:
                if self.copref_reasoning_mode == 'one_shot':
                    step_tokens = [int(copref_ids[0])] if level == 0 else []
                else:
                    step_tokens = [int(copref_ids[level])]
            else:
                step_tokens = []
            decoder_for_step = decoder_ids
            for token_id in step_tokens:
                token_col = torch.full((batch_size, 1), token_id, dtype=torch.long, device=device)
                decoder_for_step = torch.cat([decoder_for_step, token_col], dim=1)
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
        copref_ids = self.copref_token_ids.to(device)
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
            if self.use_future_value_reason:
                step_tokens = [int(copref_ids[level]), int(coreason_ids[level])]
            elif self.use_copref_module:
                if self.copref_reasoning_mode == 'one_shot':
                    step_tokens = [int(copref_ids[0])] if level == 0 else []
                else:
                    step_tokens = [int(copref_ids[level])]
            else:
                step_tokens = []
            decoder_for_step = decoder_ids
            for token_id in step_tokens:
                token_col = torch.full(
                    (batch_size, active_beams, 1), token_id, dtype=torch.long, device=device
                )
                decoder_for_step = torch.cat([decoder_for_step, token_col], dim=2)
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

    def generate(self, batch, n_return_sequences=1):
        self._assert_teacher_free_generation(batch)
        with torch.no_grad():
            if int(self.config.get('num_beams', 1)) == 1:
                return self.generate_colagr_greedy(batch, n_return_sequences)
            return self.generate_colagr_beam(batch, n_return_sequences)

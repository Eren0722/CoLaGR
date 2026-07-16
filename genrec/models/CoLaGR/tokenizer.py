import torch

from genrec.models.PSID.tokenizer import PSIDTokenizer


class CoLaGRTokenizer(PSIDTokenizer):
    """
    CoLaGR tokenizer built on top of PSID semantic IDs.

    Labels remain [sid_1, ..., sid_m, eos]. CoReason tokens are appended after
    eos in the vocabulary and are only used by the CoLaGR decoder internals.
    """

    def __init__(self, config, dataset):
        super(CoLaGRTokenizer, self).__init__(config, dataset)
        self.psid_eos_token = self.eos_token
        self.use_coroute = self._config_bool('use_coroute', False)
        self.num_coreason_routes = (
            max(int(config.get('num_coreason_routes', 1)), 1)
            if self.use_coroute else 1
        )
        self.coreason_token_ids = [
            self.psid_eos_token + 1 + level for level in range(self.n_digit)
        ]
        self.coreason_route_token_ids = [
            [
                self.psid_eos_token
                + 1
                + level
                + route * self.n_digit
                for route in range(self.num_coreason_routes)
            ]
            for level in range(self.n_digit)
        ]
        self.collate_fn = {
            'train': self.collate_fn_common,
            'val': self.collate_fn_common,
            'test': self.collate_fn_common,
        }

    def _config_bool(self, key, default=False):
        value = self.config.get(key, default)
        if isinstance(value, str):
            return value.strip().lower() in {'1', 'true', 'yes', 'y', 'on'}
        return bool(value)

    @property
    def vocab_size(self) -> int:
        return self.psid_eos_token + 1 + self.n_digit * self.num_coreason_routes

    @property
    def label_len(self) -> int:
        return self.n_digit + 1

    def tokenize_function(self, example: dict, split: str, sample_start: int = 0) -> dict:
        if split == 'train':
            n_return_examples = len(example['item_seq'][0]) - 1
            all_input_ids, all_attention_mask, all_labels = [], [], []
            all_sample_ids, all_target_items = [], []
            for i in range(n_return_examples):
                cur_example = {
                    'user': example['user'][0],
                    'item_seq': example['item_seq'][0][:i + 2],
                }
                input_ids, attention_mask, labels = self._tokenize_once(cur_example)
                all_input_ids.append(input_ids)
                all_attention_mask.append(attention_mask)
                all_labels.append(labels)
                all_sample_ids.append(sample_start + i)
                all_target_items.append(cur_example['item_seq'][-1])
            return {
                'input_ids': all_input_ids,
                'attention_mask': all_attention_mask,
                'labels': all_labels,
                'sample_id': all_sample_ids,
                'target_item': all_target_items,
            }

        input_ids, attention_mask, labels = self._tokenize_once(
            {k: v[0] for k, v in example.items()}
        )
        return {
            'input_ids': [input_ids],
            'attention_mask': [attention_mask],
            'labels': [labels],
            'sample_id': [sample_start],
            'target_item': [example['item_seq'][0][-1]],
        }

    def tokenize(self, datasets: dict) -> dict:
        tokenized_datasets = {}
        for split in datasets:
            if split == 'train':
                offsets = []
                running = 0
                for item_seq in datasets[split]['item_seq']:
                    offsets.append(running)
                    running += len(item_seq) - 1
            else:
                offsets = list(range(len(datasets[split])))

            def tokenize_with_index(example, indices, cur_split=split, cur_offsets=offsets):
                return self.tokenize_function(example, cur_split, cur_offsets[indices[0]])

            tokenized_datasets[split] = datasets[split].map(
                tokenize_with_index,
                with_indices=True,
                batched=True,
                batch_size=1,
                remove_columns=datasets[split].column_names,
                num_proc=self.config['num_proc'],
                desc=f'Tokenizing {split} set: ',
            )

        for split in datasets:
            tensor_columns = ['input_ids', 'attention_mask', 'labels', 'sample_id']
            tokenized_datasets[split].set_format(
                type='torch',
                columns=tensor_columns,
                output_all_columns=True,
            )

        return tokenized_datasets

    def collate_fn_common(self, batch: list) -> dict:
        output = {
            'input_ids': torch.stack([data['input_ids'] for data in batch]),
            'attention_mask': torch.stack([data['attention_mask'] for data in batch]),
            'labels': torch.stack([data['labels'] for data in batch]),
            'sample_id': torch.stack([data['sample_id'] for data in batch]),
        }
        if 'target_item' in batch[0]:
            output['target_item'] = [data['target_item'] for data in batch]
        return output

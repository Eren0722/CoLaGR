import torch

def get_valid_tokens(trie, tokens):
    prefix = [0]
    valid_tokens_list = []
    for token in tokens:
        valid_tokens = trie.get(prefix)
        if valid_tokens:
            valid_tokens_list.append(valid_tokens)
        else:
            valid_tokens_list.append([])
        prefix.append(token)
    return valid_tokens_list

class Collator(object):
    def __init__(self, args, tokenizer, add_input_eos=True):
        self.args = args
        self.tokenizer = tokenizer
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = 0
        self.add_input_eos = add_input_eos

    def __call__(self, batch):

        input_texts = [d["input_ids"] for d in batch]
        label_texts = [d["labels"] for d in batch]

        inputs = self.tokenizer(
            input_texts,
            return_tensors="pt",
            padding="longest",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_attention_mask=True,
            add_special_tokens=self.add_input_eos
        )

        labels = self.tokenizer(
            label_texts,
            return_tensors="pt",
            padding="longest",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_attention_mask=True,
            add_special_tokens=True
        )

        inputs['labels'] = labels['input_ids']
        inputs['labels'][inputs['labels'] == self.tokenizer.pad_token_id] = -100
        return inputs

class TestCollator(object):
    def __init__(self, args, tokenizer, add_input_eos=True):
        self.args = args
        self.tokenizer = tokenizer
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = 0
        
        self.add_input_eos = add_input_eos

    def __call__(self, batch):

        input_texts = [d["input_ids"] for d in batch]
        targets = [d["labels"] for d in batch]

        inputs = self.tokenizer(
            text=input_texts,
            return_tensors="pt",
            padding="longest",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_attention_mask=True,
            add_special_tokens=self.add_input_eos
        )
        labels = self.tokenizer(
            targets,
            return_tensors="pt",
            padding="longest",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_attention_mask=True,
            add_special_tokens=True
        )
        inputs['labels'] = labels['input_ids']
        inputs['labels'][inputs['labels'] == self.tokenizer.pad_token_id] = -100

        return (inputs, targets)

# For MSL
class ConstrainedCollator(object):
    def __init__(self, args, tokenizer, trie, add_input_eos=True):
        self.args = args
        self.tokenizer = tokenizer
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = 0

        self.trie = trie
        self.add_input_eos = add_input_eos

    def __call__(self, batch):
        input_texts = [d["input_ids"] for d in batch]
        label_texts = [d["labels"] for d in batch]

        inputs = self.tokenizer(
            input_texts,
            return_tensors="pt",
            padding="longest",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_attention_mask=True,
            add_special_tokens=self.add_input_eos
        )

        labels = self.tokenizer(
            label_texts,
            return_tensors="pt",
            padding="longest",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_attention_mask=True,
            add_special_tokens=True
        )
        inputs['labels'] = labels['input_ids']
        inputs['labels'][inputs['labels'] == self.tokenizer.pad_token_id] = -100

        # constrain_mask
        vocab_size = len(self.tokenizer)
        constrain_mask = torch.ones(size=(inputs['labels'].shape[0], inputs['labels'].shape[1], vocab_size), dtype=torch.bool)

        # batch
        for b in range(inputs['labels'].shape[0]):
            allowed_tokens_list = get_valid_tokens(self.trie, inputs['labels'][b].tolist())
            assert constrain_mask.shape[1] == len(allowed_tokens_list)
            for i, allowed_tokens in enumerate(allowed_tokens_list):
                mask = torch.zeros(vocab_size, dtype=torch.bool)
                mask[allowed_tokens] = True
                constrain_mask[b][i] = mask

        inputs['constrain_mask'] = constrain_mask

        return inputs


class FastCollatorWithItemNeg(object):
    def __init__(self, args, tokenizer, sampling="uniform", add_input_eos=True, mask_type="prefix"):
        self.args = args
        self.tokenizer = tokenizer
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = 0
        self.sampling = sampling
        self.add_input_eos = add_input_eos
        self.mask_type = mask_type

    def __call__(self, batch):

        input_texts = [d["input_ids"] for d in batch]
        label_texts = [d["labels"] for d in batch]

        inputs = self.tokenizer(
            input_texts,
            return_tensors="pt",
            padding="longest",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_attention_mask=True,
            add_special_tokens=self.add_input_eos
        )

        labels = self.tokenizer(
            label_texts,
            return_tensors="pt",
            padding="longest",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_attention_mask=True,
            add_special_tokens=True
        )

        inputs['labels'] = labels['input_ids']
        inputs['labels'][inputs['labels'] == self.tokenizer.pad_token_id] = -100

        B = inputs['input_ids'].shape[0]  # batch size
        T = inputs['labels'].shape[1]


        if self.sampling in ["uniform"]:

            neg_input_ids = [d["neg_input_ids"] for d in batch]  # (B, neg_k, T)
            neg_input_ids = torch.stack(neg_input_ids, dim=0)  # (B, neg_k, T)
            neg_k = neg_input_ids.shape[1]
            all_input_ids = torch.cat([inputs['labels'].unsqueeze(1), neg_input_ids], dim=1)  # (B, neg_k + 1, T)

            if self.mask_type == "prefix":
                # Create prefix mask for negative labels
                labels_with_negs_prefix_mask = []  # (T, B, neg_k + 1)
                for t in range(1, T + 1):
                    t_negs_prefix_mask = torch.zeros((B, neg_k + 1), dtype=torch.bool)  # (B, neg_k + 1)
                    for ii in range(B):
                        seen_prefixes = set()
                        pos_prefix = inputs['labels'][ii, :t].cpu().tolist()
                        seen_prefixes.add(tuple(pos_prefix))
                        for j in range(1, neg_k + 1):
                            neg_prefix = tuple(all_input_ids[ii][j, :t].cpu().tolist())
                            if neg_prefix in seen_prefixes:
                                t_negs_prefix_mask[ii, j] = True
                            else:
                                seen_prefixes.add(tuple(neg_prefix))
                    labels_with_negs_prefix_mask.append(t_negs_prefix_mask)
                neg_labels = all_input_ids.view(B * (neg_k+1), -1)  # (B * neg_k, T)
            
                inputs['labels_with_negs'] = neg_labels # (B * N, T)
                inputs['labels_with_negs_prefix_mask'] = torch.stack(labels_with_negs_prefix_mask, dim=0) # (T, B, neg_k + 1)
            
            elif self.mask_type == "item":
                # Create item mask for negative labels
                labels_with_negs_mask = torch.zeros((B, neg_k + 1), dtype=torch.bool)  # (B, neg_k + 1)
                for ii in range(B):
                    seen_prefixes = set()
                    pos_prefix = inputs['labels'][ii, :T].cpu().tolist()  # item
                    seen_prefixes.add(tuple(pos_prefix))
                    for j in range(1, neg_k + 1):
                        neg_prefix = tuple(all_input_ids[ii][j, :T].cpu().tolist())
                        if neg_prefix in seen_prefixes:
                            labels_with_negs_mask[ii, j] = True
                        else:
                            seen_prefixes.add(tuple(neg_prefix))
                neg_labels = all_input_ids.view(B * (neg_k+1), -1)  # (B * neg_k, T)

                inputs['labels_with_negs'] = neg_labels # (B * N, T)
                inputs['labels_with_negs_mask'] = labels_with_negs_mask # (B, neg_k + 1)
            
            else:
                raise NotImplementedError
        else:
            raise NotImplementedError
        return inputs

class TestCollatorAllItem(object):
    def __init__(self, args, tokenizer, all_items=None, add_input_eos=False):
        self.args = args
        self.tokenizer = tokenizer
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = 0
        self.all_items = all_items
        self.n_items = len(all_items) if all_items is not None else None

        if self.all_items is None:
            raise ValueError(" all_items must be provided.")

        self.add_input_eos = add_input_eos

    def __call__(self, batch):

        input_texts = [d["input_ids"] for d in batch]
        label_texts = [d["labels"] for d in batch]

        inputs = self.tokenizer(
            input_texts,
            return_tensors="pt",
            padding="longest",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_attention_mask=True,
            add_special_tokens=self.add_input_eos
        )

        labels = self.tokenizer(
            label_texts,
            return_tensors="pt",
            padding="longest",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_attention_mask=True,
            add_special_tokens=True
        )

        inputs['labels'] = labels['input_ids']
        inputs['labels'][inputs['labels'] == self.tokenizer.pad_token_id] = -100

        B = inputs['input_ids'].shape[0]  # batch size
        T = inputs['labels'].shape[1]
        
        all_items_tensor = torch.tensor(self.all_items, dtype=torch.long).expand(B, self.n_items, -1)  # (B, N, T)
        all_labels = torch.cat([labels['input_ids'].unsqueeze(1), all_items_tensor], dim=1) # (B, N+1, T)

        # all_labels reshape (B * N, T)
        all_labels = all_labels.view(-1, T)  # (B * (N+1), T)
        inputs['labels_with_negs'] = all_labels

        return (inputs, label_texts)
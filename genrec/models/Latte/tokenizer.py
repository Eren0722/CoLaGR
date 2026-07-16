"""
LatteTokenizer: A tokenizer for the Latte (Latent Token) model.

Latte extends PSID by adding latent tokens before semantic IDs.
During training, a random latent token is prepended to the semantic ID labels.
During inference, the model first predicts a latent token, then the semantic IDs.

Key Features:
1. Token layout includes latent tokens at the beginning
2. Training: Labels are [latent_token, sem_id_1, sem_id_2, sem_id_3, eos]
3. Inference: Multiple predictions mapping to the same item can be aggregated

Token layout (example when vq_codebook_size=256, vq_n_codebooks=3, n_latent_tokens=20, n_user_tokens=1):
    0: padding
    1-20: latent tokens
    21-276: digit 1 (semantic)
    277-532: digit 2 (semantic)
    533-788: digit 3 (semantic)
    789-789: user tokens
    790: eos
"""

import os
import json
import math
import random
import numpy as np
import torch
from tqdm import tqdm
from collections import defaultdict
from itertools import product
from sentence_transformers import SentenceTransformer

from genrec.dataset import AbstractDataset
from genrec.tokenizer import AbstractTokenizer


class LatteTokenizer(AbstractTokenizer):
    """
    Tokenizer for the Latte (Latent Token) model.
    
    Latte adds latent tokens before semantic IDs. During training,
    a random latent token is prepended to labels. During inference,
    predictions can be aggregated by item.

    Args:
        config (dict): The configuration dictionary.
        dataset (AbstractDataset): The dataset object.

    Attributes:
        n_latent_tokens (int): Number of latent tokens.
        item2tokens (dict): Mapping from items to their semantic ID tokens.
        tokens_to_item (dict): Reverse mapping from semantic ID tuple to item.
        base_latent_token (int): The starting token ID for latent tokens.
        base_sem_token (int): The starting token ID for semantic tokens.
        base_user_token (int): The base token ID for user tokens.
        eos_token (int): The end-of-sequence token ID.
    """

    # Supported vq_methods for loading semantic IDs
    SUPPORTED_VQ_METHODS = ['rqkmeans', 'rqvae', 'opq']

    def __init__(self, config: dict, dataset: AbstractDataset):
        # Validate vq_method
        vq_method = config.get('vq_method', 'rqkmeans')
        if vq_method not in self.SUPPORTED_VQ_METHODS:
            raise ValueError(
                f"Latte supports vq_method in {self.SUPPORTED_VQ_METHODS}, but got '{vq_method}'."
            )
        
        super(LatteTokenizer, self).__init__(config, dataset)
        
        # Latte-specific attributes
        self.n_latent_tokens = config.get('n_latent_tokens', 16)
        
        self.n_codebook_bits = self._get_codebook_bits(config['vq_codebook_size'])
        self.index_factory = f'OPQ{config["vq_n_codebooks"]},IVF1,PQ{config["vq_n_codebooks"]}x{self.n_codebook_bits}'

        self.user2id = dataset.user2id
        self.id2item = dataset.id_mapping['id2item']
        
        # Initialize tokenizer (loads semantic IDs)
        self.item2tokens = self._init_tokenizer(dataset)
        
        # Build reverse mapping from tokens (with offsets) to items
        self.tokens_to_item = self._build_tokens_to_items()
        
        # Token layout:
        # 0: padding
        # 1 to n_latent_tokens: latent tokens
        # n_latent_tokens+1 onwards: semantic ID tokens
        self.base_latent_token = 1
        self.base_sem_token = self.base_latent_token + self.n_latent_tokens
        self.base_user_token = self.base_sem_token + sum(self.codebook_sizes)
        self.n_user_tokens = self.config['n_user_tokens']
        self.eos_token = self.base_user_token + self.n_user_tokens
        
        # Set up dynamic collate functions
        self.collate_fn = {
            'train': self.collate_fn_train,
            'val': None,
            'test': None,
        }
        
        self.ignored_label = -100

    def _build_tokens_to_items(self) -> dict:
        """Build reverse mapping from token tuple (with offsets) to item.
        Since semantic IDs are conflict-free, each token tuple maps to exactly one item.
        """
        tokens_to_item = {}
        for item, tokens in self.item2tokens.items():
            token_tuple = tuple(tokens)
            if token_tuple in tokens_to_item:
                raise ValueError(
                    f'Conflict detected: tokens {token_tuple} map to both '
                    f'{tokens_to_item[token_tuple]} and {item}. '
                    f'Semantic IDs should be conflict-free after resolution.'
                )
            tokens_to_item[token_tuple] = item
        return tokens_to_item

    def _encode_sent_emb(self, dataset: AbstractDataset, output_path: str):
        """
        Encodes the sentence embeddings for the given dataset and saves them to the specified output path.
        """
        assert self.config['metadata'] == 'sentence', \
            'This method only supports sentence metadata. Use _encode_multi_group_sent_emb for sentence_multi_group.'

        meta_sentences = []  # 1-base, meta_sentences[0] -> item_id = 1
        for i in range(1, dataset.n_items):
            meta_sentences.append(dataset.item2meta[dataset.id_mapping['id2item'][i]])

        if 'sentence-transformers' in self.config['sent_emb_model']:
            sent_emb_model = SentenceTransformer(
                self.config['sent_emb_model']
            ).to(self.config['device'])

            sent_embs = sent_emb_model.encode(
                meta_sentences,
                convert_to_numpy=True,
                batch_size=self.config['sent_emb_batch_size'],
                show_progress_bar=True,
                device=self.config['device']
            )
        elif 'text-embedding-3' in self.config['sent_emb_model']:
            from openai import OpenAI
            client = OpenAI(api_key=self.config['openai_api_key'])

            sent_embs = []
            for i in tqdm(range(0, len(meta_sentences), self.config['sent_emb_batch_size']), desc='Encoding'):
                try:
                    responses = client.embeddings.create(
                        input=meta_sentences[i: i + self.config['sent_emb_batch_size']],
                        model=self.config['sent_emb_model']
                    )
                except:
                    self.log(f'[TOKENIZER] Failed to encode sentence embeddings for {i} - {i + self.config["sent_emb_batch_size"]}')
                    batch = meta_sentences[i: i + self.config['sent_emb_batch_size']]

                    from genrec.utils import num_tokens_from_string
                    new_batch = []
                    for sent in batch:
                        n_tokens = num_tokens_from_string(sent, 'cl100k_base')
                        if n_tokens < 8192:
                            new_batch.append(sent)
                        else:
                            n_chars = 8192 / n_tokens * len(sent) - 100
                            new_batch.append(sent[:int(n_chars)])

                    self.log(f'[TOKENIZER] Retrying with {len(new_batch)} sentences')
                    responses = client.embeddings.create(
                        input=new_batch,
                        model=self.config['sent_emb_model']
                    )

                for response in responses.data:
                    sent_embs.append(response.embedding)
            sent_embs = np.array(sent_embs, dtype=np.float32)

        sent_embs.tofile(output_path)
        return sent_embs

    def _resolve_conflicts(self, item2sem_ids: dict, vq_centroids: np.ndarray = None) -> dict:
        """Resolve conflicts using rqkmeans method (the only supported method for Latte)."""
        if not item2sem_ids:
            self.log('[TOKENIZER] No semantic IDs to check for conflicts.')
            return {}

        # Log a random sample of semantic IDs for quick inspection
        total_items = len(item2sem_ids)
        sample_size = min(3, total_items)
        sample_pairs = random.sample(list(item2sem_ids.items()), sample_size)
        sample_msg = ', '.join([f'{item}: {sem_id}' for item, sem_id in sample_pairs])
        self.log(f'[TOKENIZER] Sample semantic IDs (showing {sample_size}/{total_items}): {sample_msg}')

        # Detect conflicts
        sem_id2items = defaultdict(list)
        for item, sem_id in item2sem_ids.items():
            key = tuple(sem_id)
            sem_id2items[key].append(item)

        conflict_groups = {key: items for key, items in sem_id2items.items() if len(items) > 1}
        conflict_count = sum(len(items) - 1 for items in conflict_groups.values())

        if conflict_count:
            self.log(
                f'[TOKENIZER] Semantic ID conflicts detected: {conflict_count} conflicts '
                f'across {len(conflict_groups)} semantic ID patterns.'
            )
            preview = list(conflict_groups.items())[:3]
            preview_msg = ', '.join([f'{sem_id} -> {items}' for sem_id, items in preview])
            self.log(f'[TOKENIZER] Conflict samples (up to 3): {preview_msg}')
        else:
            self.log('[TOKENIZER] Semantic ID conflicts detected: 0.')
            return dict(item2sem_ids)

        # Use rqkmeans conflict resolution
        if vq_centroids is None:
            raise ValueError('vq_centroids is required for rqkmeans conflict resolution')
        if vq_centroids.shape[0] != self.n_digit:
            raise ValueError(f'VQ centroid dimension mismatch: {vq_centroids.shape[0]} vs {self.n_digit}')
        return self._resolve_conflicts_rqkmeans(item2sem_ids, conflict_groups, vq_centroids)

    def _resolve_conflicts_rqkmeans(self, item2sem_ids: dict, conflict_groups: dict, vq_centroids: np.ndarray) -> dict:
        """
        Resolve conflicts for RQ K-means method by reassigning digits with minimal centroid drift.
        
        Note: RQ (Residual Quantization) differs from PQ in that:
        - RQ centroids have shape (M, K, D) where D is the full embedding dimension
        - RQ reconstructs vectors by summing centroids: x' = sum(centroids[i][code[i]])
        - We use L2 distance between reconstructed vectors to find the best alternative
        """
        self.log('[TOKENIZER] Resolving conflicts using RQ K-means method...')
        
        # Track which semantic IDs are used
        used_semantic_ids = set(tuple(sem_id) for sem_id in item2sem_ids.values())
        
        # Process each conflict group
        resolved_item2sem_ids = dict(item2sem_ids)
        n_resolved = 0
        
        for conflict_sem_id, conflict_items in tqdm(conflict_groups.items(), desc='Resolving conflicts'):
            # Compute the original reconstructed vector
            original_recon = np.zeros(vq_centroids.shape[-1], dtype=np.float32)
            for digit_idx in range(self.n_digit):
                original_recon += vq_centroids[digit_idx, conflict_sem_id[digit_idx]]
            
            # Keep the first item with its original semantic ID
            # Reassign the rest
            for item_idx, item in enumerate(conflict_items[1:], start=1):
                # Find top 5 closest tokens for each digit based on L2 distance to original centroid
                top_k_per_digit = []
                for digit_idx in range(self.n_digit):
                    original_token = conflict_sem_id[digit_idx]
                    original_centroid = vq_centroids[digit_idx, original_token]
                    
                    # Calculate L2 distances to all centroids in this digit
                    all_centroids = vq_centroids[digit_idx]  # shape: (K, D)
                    diff = all_centroids - original_centroid[None, :]  # shape: (K, D)
                    l2_distances = np.sum(diff * diff, axis=1)  # shape: (K,)
                    
                    # Get top 5 closest tokens (smallest L2 distance)
                    top_k_indices = np.argsort(l2_distances)[:5]
                    top_k_per_digit.append(top_k_indices.tolist())

                # Enumerate all combinations (5^n_digit possibilities)
                best_sem_id = None
                best_distance = float('inf')

                for candidate_tokens in product(*top_k_per_digit):
                    candidate_sem_id = tuple(candidate_tokens)
                    
                    # Skip if already used
                    if candidate_sem_id in used_semantic_ids:
                        continue
                    
                    # Compute reconstructed vector for candidate
                    candidate_recon = np.zeros(vq_centroids.shape[-1], dtype=np.float32)
                    for digit_idx in range(self.n_digit):
                        candidate_recon += vq_centroids[digit_idx, candidate_tokens[digit_idx]]
                    
                    # Calculate L2 distance between original and candidate reconstructions
                    total_distance = np.sum((original_recon - candidate_recon) ** 2)

                    if total_distance < best_distance:
                        best_distance = total_distance
                        best_sem_id = candidate_sem_id

                if best_sem_id is not None:
                    # Update the semantic ID for this item
                    resolved_item2sem_ids[item] = best_sem_id
                    used_semantic_ids.add(best_sem_id)
                    n_resolved += 1
                else:
                    self.log(f'[TOKENIZER] Warning: Could not find conflict-free semantic ID for item {item}')

        self.log(f'[TOKENIZER] Successfully resolved {n_resolved} conflicts.')
        return resolved_item2sem_ids

    def _get_items_for_training(self, dataset: AbstractDataset) -> np.ndarray:
        """
        Get a boolean mask indicating which items are used for training.
        """
        items_for_training = set()
        for item_seq in dataset.split_data['train']['item_seq']:
            for item in item_seq:
                items_for_training.add(item)
        self.log(f'[TOKENIZER] Items for training: {len(items_for_training)} of {dataset.n_items - 1}')
        mask = np.zeros(dataset.n_items - 1, dtype=bool)
        for item in items_for_training:
            mask[dataset.item2id[item] - 1] = True
        return mask

    def _generate_semantic_id_rqkmeans(
        self,
        sent_embs: np.ndarray,
        sem_ids_path: str,
        train_mask: np.ndarray
    ) -> None:
        """
        Generates semantic IDs using Faiss Residual Quantizer (RQ K-means) and saves them to a file.

        Args:
            sent_embs (np.ndarray): The sentence embeddings.
            sem_ids_path (str): The path to save the semantic IDs.
            train_mask (np.ndarray): A boolean mask indicating which items are used for training.

        Returns:
            None
        """
        n_bits = int(np.log2(self.config['vq_codebook_size']))
        K = self.config['vq_codebook_size']
        M = self.config['vq_n_codebooks']
        D = sent_embs.shape[-1]

        import faiss
        faiss.omp_set_num_threads(self.config['faiss_omp_num_threads'])
        index = faiss.IndexResidualQuantizer(
            D,
            M,
            n_bits,
            faiss.METRIC_INNER_PRODUCT
        )
        self.log(f'[TOKENIZER] Training RQ index...')
        index.train(sent_embs[train_mask])
        index.add(sent_embs)
        
        # Extract RQ centroids: shape (M, K, D)
        rq_centroids_flat = faiss.vector_to_array(index.rq.codebooks)
        rq_centroids = rq_centroids_flat.reshape(M, K, D)
        self.log(f'[TOKENIZER] Extracted RQ centroids with shape: {rq_centroids.shape}')
        
        # Generate semantic IDs
        faiss_sem_ids = []
        uint8_code = index.rq.compute_codes(sent_embs)
        n_bytes = uint8_code.shape[1]
        self.log(f'[TOKENIZER] Generating semantic IDs...')
        for u8_code in uint8_code:
            bs = faiss.BitstringReader(faiss.swig_ptr(u8_code), n_bytes)
            code = []
            for i in range(M):
                code.append(bs.read(n_bits))
            faiss_sem_ids.append(code)
        faiss_sem_ids = np.array(faiss_sem_ids)

        item2sem_ids = {}
        for i in range(faiss_sem_ids.shape[0]):
            item = self.id2item[i + 1]
            item2sem_ids[item] = tuple(faiss_sem_ids[i].tolist())

        # Resolve conflicts using RQ centroids
        self.log("[TOKENIZER] Resolving conflicts for rqkmeans...")
        conflict_free_item2sem_ids = self._resolve_conflicts(item2sem_ids, rq_centroids)

        self.log(f'[TOKENIZER] Saving semantic IDs to {sem_ids_path}...')
        with open(sem_ids_path, 'w') as f:
            json.dump(conflict_free_item2sem_ids, f)

    def _sem_ids_to_tokens_with_offset(self, item2sem_ids: dict) -> dict:
        """
        Converts semantic IDs to tokens with proper offsets.
        Accounts for latent tokens at the beginning.
        
        Token layout:
            0: padding
            1 to n_latent_tokens: latent tokens
            n_latent_tokens+1 onwards: semantic ID tokens
        """
        # Offset for each digit
        # First sem_id digit starts at n_latent_tokens + 1 (for padding)
        sem_id_offsets = [self.n_latent_tokens + 1]
        for digit in range(1, self.n_digit):
            sem_id_offsets.append(sem_id_offsets[-1] + self.codebook_sizes[digit - 1])
        
        item2tokens = {}
        for item in item2sem_ids:
            tokens = list(item2sem_ids[item])
            for digit in range(self.n_digit):
                tokens[digit] += sem_id_offsets[digit]
            item2tokens[item] = tuple(tokens)
        
        return item2tokens

    def _init_tokenizer(self, dataset: AbstractDataset):
        """
        Initialize the tokenizer.

        Args:
            dataset (AbstractDataset): The dataset object.

        Returns:
            dict: A dictionary mapping items to semantic IDs.
        """
        # Load semantic IDs - Latte can load semantic IDs generated by PSID with various vq_methods
        vq_method = self.config.get('vq_method', 'rqkmeans')
        vq_setting = f'{self.config["vq_n_codebooks"]}x{self.config["vq_codebook_size"]}'
        
        # Determine the sem_ids file path based on vq_method
        # Methods that use PSID-style conflict resolution
        if vq_method in ['opq', 'rqkmeans', 'rqvae']:
            deduplication_suffix = '_psid'
        else:
            deduplication_suffix = ''
        
        sem_ids_path = os.path.join(
            dataset.cache_dir, 'processed',
            f'{os.path.basename(self.config["sent_emb_model"])}_{vq_method}_{vq_setting}{deduplication_suffix}.sem_ids'
        )

        if not os.path.exists(sem_ids_path):
            # Only rqkmeans can be generated by Latte tokenizer
            # For other methods, the semantic IDs must be pre-generated by PSID
            if vq_method == 'rqkmeans':
                # Load or encode sentence embeddings
                sent_emb_path = os.path.join(
                    dataset.cache_dir, 'processed',
                    f'{os.path.basename(self.config["sent_emb_model"])}.sent_emb'
                )
                if os.path.exists(sent_emb_path):
                    self.log(f'[TOKENIZER] Loading sentence embeddings from {sent_emb_path}...')
                    sent_embs = np.fromfile(sent_emb_path, dtype=np.float32).reshape(-1, self.config['sent_emb_dim'])
                else:
                    self.log(f'[TOKENIZER] Encoding sentence embeddings...')
                    sent_embs = self._encode_sent_emb(dataset, sent_emb_path)
                
                # PCA
                if self.config['sent_emb_pca'] > 0:
                    self.log(f'[TOKENIZER] Applying PCA to sentence embeddings...')
                    from sklearn.decomposition import PCA
                    pca = PCA(n_components=self.config['sent_emb_pca'], whiten=True)
                    sent_embs = pca.fit_transform(sent_embs)
                self.log(f'[TOKENIZER] Sentence embeddings shape: {sent_embs.shape}')

                # Generate semantic IDs using rqkmeans
                training_item_mask = self._get_items_for_training(dataset)
                self.log(f'[TOKENIZER] Semantic IDs not found. Training index using RQ K-means (Faiss)...')
                self._generate_semantic_id_rqkmeans(sent_embs, sem_ids_path, training_item_mask)
            else:
                raise FileNotFoundError(
                    f"Semantic IDs file not found: {sem_ids_path}\n"
                    f"For vq_method='{vq_method}', please first generate semantic IDs using PSID model.\n"
                    f"Example: python main.py --model=PSID --vq_method={vq_method} ..."
                )

        self.log(f'[TOKENIZER] Loading semantic IDs from {sem_ids_path}...')
        item2sem_ids = json.load(open(sem_ids_path, 'r'))
        
        # Convert to tuples for consistency
        item2sem_ids = {k: tuple(v) for k, v in item2sem_ids.items()}
        
        # Apply token offsets (accounting for latent tokens)
        item2tokens = self._sem_ids_to_tokens_with_offset(item2sem_ids)

        return item2tokens

    @property
    def n_digit(self):
        """Returns the number of digits for the tokenizer."""
        return self.config['vq_n_codebooks']

    @property
    def codebook_sizes(self):
        """
        Returns the codebook size for the tokenizer.
        """
        if isinstance(self.config['vq_codebook_size'], list):
            return self.config['vq_codebook_size']
        else:
            return [self.config['vq_codebook_size']] * self.n_digit

    def _token_single_user(self, user: str) -> int:
        """Tokenizes a single user with updated offset."""
        user_id = self.user2id[user]
        return self.base_user_token + user_id % self.n_user_tokens

    def _token_single_item(self, item: str) -> tuple:
        """Tokenizes a single item."""
        return self.item2tokens[item]

    def _tokenize_once(self, example: dict) -> tuple:
        """
        Tokenizes a single example.

        Args:
            example (dict): A dictionary containing the example data.

        Returns:
            tuple: A tuple containing (input_ids, attention_mask, labels).
        """
        max_item_seq_len = self.config['max_item_seq_len']

        # input_ids
        user_token = self._token_single_user(example['user'])
        input_ids = [user_token]
        for item in example['item_seq'][:-1][-max_item_seq_len:]:
            input_ids.extend(self._token_single_item(item))
        input_ids.append(self.eos_token)
        input_ids.extend([self.padding_token] * (self.max_token_seq_len - len(input_ids)))

        # attention_mask
        item_seq_len = min(len(example['item_seq'][:-1]), max_item_seq_len)
        attention_mask = [1] * (self.n_digit * item_seq_len + 2)
        attention_mask.extend([0] * (self.max_token_seq_len - len(attention_mask)))

        # labels: semantic ID tokens + eos (no latent token for eval)
        labels = list(self._token_single_item(example['item_seq'][-1])) + [self.eos_token]

        return input_ids, attention_mask, labels

    def tokenize_function(self, example: dict, split: str) -> dict:
        """
        Tokenizes the input example.
        
        For training, input_ids and attention_mask are pre-computed here,
        while labels are generated dynamically in collate_fn with random latent token.
        
        For val/test, everything is pre-computed here.
        """
        if split == 'train':
            # For training, create multiple examples from one sequence
            n_return_examples = len(example['item_seq'][0]) - 1
            all_input_ids = []
            all_attention_mask = []
            all_target_items = []  # Store target item for dynamic label generation
            
            for i in range(n_return_examples):
                cur_example = {
                    'user': example['user'][0],
                    'item_seq': example['item_seq'][0][:i+2]
                }
                input_ids, attention_mask, _ = self._tokenize_once(cur_example)
                all_input_ids.append(input_ids)
                all_attention_mask.append(attention_mask)
                all_target_items.append(cur_example['item_seq'][-1])
            
            return {
                'input_ids': all_input_ids,
                'attention_mask': all_attention_mask,
                'target_item': all_target_items,
            }
        else:
            # For val/test, pre-compute everything including labels
            input_ids, attention_mask, labels = self._tokenize_once(
                {k: v[0] for k, v in example.items()}
            )
            return {
                'input_ids': [input_ids],
                'attention_mask': [attention_mask],
                'labels': [labels],
            }

    def tokenize(self, datasets: dict) -> dict:
        """
        Tokenizes the given datasets.
        
        For training: pre-computes input_ids/attention_mask, stores target_item for dynamic labels.
        For val/test: pre-computes everything including labels.
        """
        tokenized_datasets = {}
        for split in datasets:
            tokenized_datasets[split] = datasets[split].map(
                lambda t: self.tokenize_function(t, split),
                batched=True,
                batch_size=1,
                remove_columns=datasets[split].column_names,
                num_proc=self.config['num_proc'],
                desc=f'Tokenizing {split} set: '
            )

        for split in datasets:
            tokenized_datasets[split].set_format(type='torch')

        return tokenized_datasets

    @property
    def vocab_size(self) -> int:
        """Returns the vocabulary size including latent tokens."""
        return self.eos_token + 1

    @property
    def max_token_seq_len(self) -> int:
        """Returns the maximum token sequence length."""
        # +2 for user token and eos token
        return self.config['max_item_seq_len'] * self.n_digit + 2

    @property
    def label_len(self) -> int:
        """Returns the label length including latent token and eos."""
        # latent_token + n_digit semantic tokens + eos
        return self.n_digit + 2

    def collate_fn_train(self, batch: list) -> dict:
        """
        Dynamic collate function for training (vectorized).
        
        input_ids and attention_mask are already pre-computed.
        Only labels are generated dynamically with random latent token.
        """
        input_ids = torch.stack([data['input_ids'] for data in batch])
        attention_mask = torch.stack([data['attention_mask'] for data in batch])
        
        batch_size = len(batch)
        
        # Batch get all target items' semantic tokens
        sem_tokens = torch.tensor([
            list(self.item2tokens[data['target_item']]) for data in batch
        ], dtype=torch.long)  # (batch_size, n_digit)
        
        # Batch generate random latent tokens
        latent_tokens = self.base_latent_token + torch.randint(
            0, self.n_latent_tokens, (batch_size,), dtype=torch.long
        )  # (batch_size,)
        
        # Create eos tokens
        eos_tokens = torch.full((batch_size,), self.eos_token, dtype=torch.long)
        
        # Concatenate: [latent_token, sem_tokens..., eos]
        labels = torch.cat([
            latent_tokens.unsqueeze(1),  # (batch_size, 1)
            sem_tokens,                      # (batch_size, n_digit)
            eos_tokens.unsqueeze(1)          # (batch_size, 1)
        ], dim=1)  # (batch_size, n_digit + 2)
        
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels,
        }

    def _get_codebook_bits(self, n_codebook):
        """Get the number of bits needed to represent n_codebook values."""
        x = math.log2(n_codebook)
        assert x.is_integer() and x >= 0, "Invalid value for n_codebook"
        return int(x)

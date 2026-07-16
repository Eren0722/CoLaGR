import os
import math
import random
import numpy as np
from tqdm import tqdm
import json
from collections import defaultdict
from itertools import product, permutations
from sentence_transformers import SentenceTransformer
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from genrec.dataset import AbstractDataset
from genrec.tokenizer import AbstractTokenizer
from genrec.layers import RQVAEModel


class PSIDTokenizer(AbstractTokenizer):
    """
    Tokenizer for the PSID (Purely Semantic ID) model.
    
    PSID uses conflict resolution to ensure all semantic IDs are unique (non-conflicting)
    while maintaining purely semantic tokens. Unlike TIGER which adds an additional 
    non-semantic token to resolve conflicts, PSID reassigns conflicting items to nearby
    semantic IDs, trading some compression accuracy for conflict-free purely semantic tokens.

    An example when "vq_codebook_size == 256, vq_n_codebooks == 3, n_user_tokens == 2000":
        0: padding
        1-256: digit 1 (semantic)
        257-512: digit 2 (semantic)
        513-768: digit 3 (semantic)
        769-2768: user tokens
        2769: eos

    Args:
        config (dict): The configuration dictionary.
        dataset (AbstractDataset): The dataset object.

    Attributes:
        item2tokens (dict): A dictionary mapping items to their conflict-free semantic ID tokens.
        base_user_token (int): The base token ID for user tokens.
        n_user_tokens (int): The number of user tokens.
        eos_token (int): The end-of-sequence token ID.
    """
    def __init__(self, config: dict, dataset: AbstractDataset):
        super(PSIDTokenizer, self).__init__(config, dataset)
        self.n_codebook_bits = self._get_codebook_bits(config['vq_codebook_size'])
        self.index_factory = f'OPQ{config["vq_n_codebooks"]},IVF1,PQ{config["vq_n_codebooks"]}x{self.n_codebook_bits}'

        self.user2id = dataset.user2id
        self.id2item = dataset.id_mapping['id2item']
        self.item2tokens = self._init_tokenizer(dataset)
        self.base_user_token = sum(self.codebook_sizes) + 1
        self.n_user_tokens = self.config['n_user_tokens']
        self.eos_token = self.base_user_token + self.n_user_tokens

    def _encode_sent_emb(self, dataset: AbstractDataset, output_path: str):
        """
        Encodes the sentence embeddings for the given dataset and saves them to the specified output path.

        Args:
            dataset (AbstractDataset): The dataset containing the sentences to encode.
            output_path (str): The path to save the encoded sentence embeddings.

        Returns:
            numpy.ndarray: The encoded sentence embeddings.
        """
        assert self.config['metadata'] == 'sentence', \
            'This method only supports sentence metadata. Use _encode_multi_group_sent_emb for sentence_multi_group.'

        meta_sentences = [] # 1-base, meta_sentences[0] -> item_id = 1
        for i in range(1, dataset.n_items):
            meta_sentences.append(dataset.item2meta[dataset.id_mapping['id2item'][i]])

        sent_emb_model_name = self.config['sent_emb_model']
        is_local_st_model = os.path.exists(sent_emb_model_name)

        if is_local_st_model or 'sentence-transformers' in sent_emb_model_name:
            sent_emb_model = SentenceTransformer(
                sent_emb_model_name
            ).to(self.config['device'])

            sent_embs = sent_emb_model.encode(
                meta_sentences,
                convert_to_numpy=True,
                batch_size=self.config['sent_emb_batch_size'],
                show_progress_bar=True,
                device=self.config['device']
            )
        elif 'text-embedding-3' in sent_emb_model_name:
            from openai import OpenAI
            client = OpenAI(api_key=self.config['openai_api_key'])

            sent_embs = []
            for i in tqdm(range(0, len(meta_sentences), self.config['sent_emb_batch_size']), desc='Encoding'):
                try:
                    responses = client.embeddings.create(
                        input=meta_sentences[i: i + self.config['sent_emb_batch_size']],
                        model=sent_emb_model_name
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
                        model=sent_emb_model_name
                    )

                for response in responses.data:
                    sent_embs.append(response.embedding)
            sent_embs = np.array(sent_embs, dtype=np.float32)
        else:
            raise ValueError(
                f'Unsupported sent_emb_model: {sent_emb_model_name}. '
                'Expected a local SentenceTransformer directory, a '
                '"sentence-transformers/*" model name, or an OpenAI '
                '"text-embedding-3*" model name.'
            )

        sent_embs.tofile(output_path)
        return sent_embs

    def _resolve_conflicts(self, item2sem_ids: dict, vq_centroids: np.ndarray = None) -> dict:
        """Dispatcher for conflict resolution based on vq_method."""
        if not item2sem_ids:
            self.log('[TOKENIZER] No semantic IDs to check for conflicts.')
            return {}

        vq_method = self.config.get('vq_method', 'opq')
        
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

        # Dispatch to method-specific conflict resolution
        if vq_method == 'opq':
            if vq_centroids is None:
                raise ValueError('vq_centroids is required for opq conflict resolution')
            if vq_centroids.shape[0] != self.n_digit:
                raise ValueError(f'VQ centroid dimension mismatch: {vq_centroids.shape[0]} vs {self.n_digit}')
            return self._resolve_conflicts_opq(item2sem_ids, conflict_groups, vq_centroids)
        elif vq_method == 'rqkmeans':
            if vq_centroids is None:
                raise ValueError('vq_centroids is required for rqkmeans conflict resolution')
            if vq_centroids.shape[0] != self.n_digit:
                raise ValueError(f'VQ centroid dimension mismatch: {vq_centroids.shape[0]} vs {self.n_digit}')
            return self._resolve_conflicts_rqkmeans(item2sem_ids, conflict_groups, vq_centroids)
        elif vq_method == 'rqvae':
            if vq_centroids is None:
                raise ValueError('vq_centroids is required for rqvae conflict resolution')
            if vq_centroids.shape[0] != self.n_digit:
                raise ValueError(f'VQ centroid dimension mismatch: {vq_centroids.shape[0]} vs {self.n_digit}')
            return self._resolve_conflicts_rqvae(item2sem_ids, conflict_groups, vq_centroids)
        else:
            raise ValueError(f'Unknown vq_method: {vq_method}. Supported methods: opq, rqkmeans, rqvae')

    def _resolve_conflicts_opq(self, item2sem_ids: dict, conflict_groups: dict, vq_centroids: np.ndarray) -> dict:
        """Resolve conflicts for OPQ method by reassigning digits with minimal centroid drift."""
        self.log('[TOKENIZER] Resolving conflicts using OPQ method...')
        
        # Track which semantic IDs are used
        used_semantic_ids = set(tuple(sem_id) for sem_id in item2sem_ids.values())
        
        # Process each conflict group
        resolved_item2sem_ids = dict(item2sem_ids)
        n_resolved = 0
        
        for conflict_sem_id, conflict_items in tqdm(conflict_groups.items(), desc='Resolving conflicts'):
            # Keep the first item with its original semantic ID
            # Reassign the rest
            for item_idx, item in enumerate(conflict_items[1:], start=1):
                # Find top 5 closest tokens for each digit based on inner product
                top_k_per_digit = []
                for digit_idx in range(self.n_digit):
                    original_token = conflict_sem_id[digit_idx]
                    original_centroid = vq_centroids[digit_idx, original_token]
                    
                    # Calculate inner product distances to all centroids in this digit
                    all_centroids = vq_centroids[digit_idx]  # shape: (K, dsub)
                    inner_products = np.dot(all_centroids, original_centroid)  # shape: (K,)
                    
                    # Get top 5 closest tokens (highest inner product)
                    top_k_indices = np.argsort(-inner_products)[:5]  # descending order
                    top_k_per_digit.append(top_k_indices.tolist())

                # Enumerate all combinations (5^n_digit possibilities)
                best_sem_id = None
                best_distance = float('inf')

                for candidate_tokens in product(*top_k_per_digit):
                    candidate_sem_id = tuple(candidate_tokens)
                    
                    # Skip if already used
                    if candidate_sem_id in used_semantic_ids:
                        continue
                    
                    # Calculate total distance (sum of negative inner products)
                    total_distance = 0.0
                    for digit_idx in range(self.n_digit):
                        original_token = conflict_sem_id[digit_idx]
                        candidate_token = candidate_tokens[digit_idx]

                        original_centroid = vq_centroids[digit_idx, original_token]
                        candidate_centroid = vq_centroids[digit_idx, candidate_token]

                        # Distance is negative inner product (we want to maximize inner product)
                        distance = -np.dot(original_centroid, candidate_centroid)
                        total_distance += distance

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

    def _resolve_conflicts_rqvae(self, item2sem_ids: dict, conflict_groups: dict, vq_centroids: np.ndarray) -> dict:
        """
        Resolve conflicts for RQ-VAE method by reassigning digits with minimal reconstruction drift.
        
        RQ-VAE uses residual quantization similar to RQ K-means, where:
        - Centroids have shape (M, K, D) where D is the latent dimension from the encoder
        - Reconstructed vectors are computed by summing centroids: x' = sum(centroids[i][code[i]])
        - We use L2 distance between reconstructed vectors to find the best alternative
        
        This method is functionally equivalent to _resolve_conflicts_rqkmeans since both
        use residual quantization, but is separated for clarity and potential future customization.
        """
        self.log('[TOKENIZER] Resolving conflicts using RQ-VAE method...')
        
        # Track which semantic IDs are used
        used_semantic_ids = set(tuple(sem_id) for sem_id in item2sem_ids.values())
        
        # Process each conflict group
        resolved_item2sem_ids = dict(item2sem_ids)
        n_resolved = 0
        
        for conflict_sem_id, conflict_items in tqdm(conflict_groups.items(), desc='Resolving conflicts'):
            # Compute the original reconstructed vector (sum of centroids)
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

        Args:
            dataset (AbstractDataset): The dataset containing the item sequences.

        Returns:
            np.ndarray: A boolean mask indicating which items are used for training.
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

    def _generate_semantic_id_opq(
        self,
        sent_embs: np.ndarray,
        sem_ids_path: str,
        train_mask: np.ndarray
    ) -> None:
        """
        Generates semantic IDs using the Faiss library and saves them to a file.

        Args:
            sent_embs (np.ndarray): The sentence embeddings.
            sem_ids_path (str): The path to save the semantic IDs.
            train_mask (np.ndarray): A boolean mask indicating which items are used for training.

        Returns:
            None
        """
        import faiss
        faiss.omp_set_num_threads(self.config['faiss_omp_num_threads'])
        index = faiss.index_factory(
            sent_embs.shape[1],
            self.index_factory,
            faiss.METRIC_INNER_PRODUCT
        )
        self.log(f'[TOKENIZER] Training index...')
        index.train(sent_embs[train_mask])
        index.add(sent_embs)

        ivf_index = faiss.downcast_index(index.index)
        invlists = faiss.extract_index_ivf(ivf_index).invlists
        ls = invlists.list_size(0)
        pq_codes = faiss.rev_swig_ptr(invlists.get_codes(0), ls * invlists.code_size)
        pq_codes = pq_codes.reshape(-1, invlists.code_size)

        pq = ivf_index.pq
        pq_centroids = faiss.vector_to_array(pq.centroids).reshape(pq.M, pq.ksub, pq.dsub)

        faiss_sem_ids = []
        n_bytes = pq_codes.shape[1]
        for u8code in pq_codes:
            bs = faiss.BitstringReader(faiss.swig_ptr(u8code), n_bytes)
            code = []
            for i in range(self.n_digit):
                code.append(bs.read(self.n_codebook_bits))
            faiss_sem_ids.append(code)
        pq_codes = np.array(faiss_sem_ids)

        item2sem_ids = {}
        for i in range(pq_codes.shape[0]):
            item = self.id2item[i + 1]
            item2sem_ids[item] = tuple(pq_codes[i].tolist())
        conflict_free_item2sem_ids = self._resolve_conflicts(item2sem_ids, pq_centroids)
        self.log(f'[TOKENIZER] Saving semantic IDs to {sem_ids_path}...')
        with open(sem_ids_path, 'w') as f:
            json.dump(conflict_free_item2sem_ids, f)

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
        # In Faiss, RQ codebooks are stored as a flat array
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

    def _train_rqvae(self, sent_embs: torch.Tensor, model_path: str) -> RQVAEModel:
        """
        Trains the RQ-VAE model using the given sentence embeddings.

        Args:
            sent_embs (torch.Tensor): Array of sentence embeddings for training.
            model_path (str): Path to save the trained model.

        Returns:
            RQVAEModel: Trained RQ-VAE model.
        """
        device = self.config['device']

        # Initialize RQ-VAE model
        all_hidden_sizes = [sent_embs.shape[1]] + self.config['rqvae_hidden_sizes']
        rqvae_model = RQVAEModel(
            hidden_sizes=all_hidden_sizes,
            n_codebooks=self.config['vq_n_codebooks'],
            codebook_size=self.config['vq_codebook_size'],
            dropout=self.config['rqvae_dropout'],
            low_usage_threshold=self.config['rqvae_low_usage_threshold']
        ).to(device)
        self.log(rqvae_model)
        
        if os.path.exists(model_path):
            self.log(f"[TOKENIZER] Loading RQ-VAE model from {model_path}...")
            rqvae_model.load_state_dict(torch.load(model_path))
            return rqvae_model

        # Model training
        batch_size = self.config['rqvae_batch_size']
        num_epochs = self.config['rqvae_epoch']
        beta = self.config['rqvae_beta']
        verbose = self.config['rqvae_verbose']

        rqvae_model.generate_codebook(sent_embs, device)
        optimizer = torch.optim.Adagrad(rqvae_model.parameters(), lr=self.config['rqvae_lr'])
        train_dataset = TensorDataset(sent_embs)
        dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        self.log("[TOKENIZER] Training RQ-VAE model...")
        rqvae_model.train()
        for epoch in tqdm(range(num_epochs), desc='Training RQ-VAE'):
            total_loss = 0.0
            total_rec_loss = 0.0
            total_quant_loss = 0.0
            total_count = 0
            for batch in dataloader:
                x_batch = batch[0]
                optimizer.zero_grad()
                recon_x, quant_loss, count = rqvae_model(x_batch)
                reconstruction_mse_loss = F.mse_loss(recon_x, x_batch, reduction='mean')
                loss = reconstruction_mse_loss + beta * quant_loss
                loss.backward()
                optimizer.step()
                total_loss += loss.detach().cpu().item()
                total_rec_loss += reconstruction_mse_loss.detach().cpu().item()
                total_quant_loss += quant_loss.detach().cpu().item()
                total_count += count

            if (epoch + 1) % verbose == 0:
                self.log(
                    f"[TOKENIZER] RQ-VAE training\n"
                    f"\tEpoch [{epoch+1}/{num_epochs}]\n"
                    f"\t  Training loss: {total_loss / len(dataloader)}\n"
                    f"\t  Unused codebook: {total_count / len(dataloader)}\n"
                    f"\t  Reconstruction loss: {total_rec_loss / len(dataloader)}\n"
                    f"\t  Quantization loss: {total_quant_loss / len(dataloader)}\n")
        self.log("[TOKENIZER] RQ-VAE training complete.")

        # Save model
        torch.save(rqvae_model.state_dict(), model_path, pickle_protocol=4)
        return rqvae_model

    def _extract_rqvae_centroids(self, rqvae_model: RQVAEModel) -> np.ndarray:
        """
        Extract centroids from trained RQ-VAE model.

        Args:
            rqvae_model (RQVAEModel): Trained RQ-VAE model.

        Returns:
            np.ndarray: Centroids with shape (M, K, D) where M is number of codebooks,
                       K is codebook size, and D is latent dimension.
        """
        M = self.config['vq_n_codebooks']
        K = self.config['vq_codebook_size']
        
        centroids_list = []
        for layer in rqvae_model.quantization_layer.quantization_layers:
            # layer.embed has shape (D, K), need to transpose to (K, D)
            centroid = layer.embed.detach().cpu().numpy().T  # (K, D)
            centroids_list.append(centroid)
        
        # Stack to (M, K, D)
        centroids = np.stack(centroids_list, axis=0)
        self.log(f'[TOKENIZER] Extracted RQ-VAE centroids with shape: {centroids.shape}')
        return centroids

    def _generate_semantic_id_rqvae(
        self,
        sent_embs: np.ndarray,
        sem_ids_path: str,
        train_mask: np.ndarray
    ) -> None:
        """
        Generates semantic IDs using RQ-VAE (Residual Quantized VAE) with PSID-style
        conflict resolution and saves them to a file.

        Unlike TIGER which adds an extra digit to resolve conflicts, PSID reassigns
        conflicting items to nearby semantic IDs based on centroid distances.

        Args:
            sent_embs (np.ndarray): The sentence embeddings.
            sem_ids_path (str): The path to save the semantic IDs.
            train_mask (np.ndarray): A boolean mask indicating which items are used for training.

        Returns:
            None
        """
        device = self.config['device']
        
        # Prepare training embeddings
        embs_for_training = torch.FloatTensor(sent_embs[train_mask]).to(device)
        all_sent_embs = torch.FloatTensor(sent_embs).to(device)
        
        # Train or load RQ-VAE model
        # Use a different model path than TIGER to avoid conflicts
        model_dir = os.path.dirname(sem_ids_path)
        model_path = os.path.join(model_dir, 'rqvae_psid.pth')
        rqvae_model = self._train_rqvae(embs_for_training, model_path)
        
        # Extract centroids for conflict resolution
        rqvae_centroids = self._extract_rqvae_centroids(rqvae_model)
        
        # Generate semantic IDs using the trained model
        rqvae_model.eval()
        with torch.no_grad():
            rqvae_sem_ids = rqvae_model.encode(all_sent_embs)
        
        # Convert to item -> semantic ID dict (without extending with extra digit)
        item2sem_ids = {}
        for i in range(rqvae_sem_ids.shape[0]):
            item = self.id2item[i + 1]
            item2sem_ids[item] = tuple(rqvae_sem_ids[i].tolist())
        
        # Resolve conflicts using PSID strategy (reassign to nearby semantic IDs)
        self.log("[TOKENIZER] Resolving conflicts for rqvae using PSID strategy...")
        conflict_free_item2sem_ids = self._resolve_conflicts(item2sem_ids, rqvae_centroids)
        
        # Save result
        self.log(f'[TOKENIZER] Saving semantic IDs to {sem_ids_path}...')
        with open(sem_ids_path, 'w') as f:
            json.dump(conflict_free_item2sem_ids, f)

    def _permute_semantic_ids(self, item2sem_ids: dict) -> dict:
        """
        Permute semantic IDs according to the configured permutation index.
        For n_digit tokens, there are n_digit! possible permutations.
        
        Args:
            item2sem_ids (dict): A dictionary mapping items to their semantic IDs (tuples).
        
        Returns:
            dict: A dictionary with permuted semantic IDs.
        """
        perm_index = self.config.get('vq_permutation', 0)

        if perm_index == 0:
            self.log('[TOKENIZER] No permutation applied (vq_permutation=0).')
            return item2sem_ids

        # Generate all possible permutations for indices [0, 1, ..., n_digit-1]
        all_perms = list(permutations(range(self.n_digit)))

        if perm_index < 0 or perm_index >= len(all_perms):
            raise ValueError(
                f'Invalid vq_permutation={perm_index}. '
                f'For n_digit={self.n_digit}, valid range is 0-{len(all_perms)-1}.'
            )

        perm_order = all_perms[perm_index]
        self.log(f'[TOKENIZER] Applying permutation {perm_index}: {perm_order}')

        # Apply permutation to all semantic IDs
        permuted_item2sem_ids = {}
        for item, sem_id in item2sem_ids.items():
            sem_id_list = list(sem_id)
            permuted_sem_id = tuple(sem_id_list[i] for i in perm_order)
            permuted_item2sem_ids[item] = permuted_sem_id

        return permuted_item2sem_ids

    def _sem_ids_to_tokens(self, item2sem_ids: dict) -> dict:
        """
        Converts semantic IDs to tokens.

        Args:
            item2sem_ids (dict): A dictionary mapping items to their corresponding semantic IDs.

        Returns:
            dict: A dictionary mapping items to their corresponding tokens.
        """
        sem_id_offsets = [0]
        for digit in range(1, self.n_digit):
            sem_id_offsets.append(sem_id_offsets[-1] + self.codebook_sizes[digit - 1])
        for item in item2sem_ids:
            tokens = list(item2sem_ids[item])
            for digit in range(self.n_digit):
                # "+ 1" as 0 is reserved for padding
                tokens[digit] += sem_id_offsets[digit] + 1
            item2sem_ids[item] = tuple(tokens)
        return item2sem_ids

    def _init_tokenizer(self, dataset: AbstractDataset):
        """
        Initialize the tokenizer.

        Args:
            dataset (AbstractDataset): The dataset object.

        Returns:
            dict: A dictionary mapping items to semantic IDs.
        """
        # Load semantic IDs
        vq_method = self.config.get('vq_method', 'opq')
        vq_setting = f'{self.config["vq_n_codebooks"]}x{self.config["vq_codebook_size"]}'
        deduplication_method = '_psid' if vq_method in ["opq", "rqkmeans", "rqvae"] else ''
        sem_ids_path = os.path.join(
            dataset.cache_dir, 'processed',
            f'{os.path.basename(self.config["sent_emb_model"])}_{vq_method}_{vq_setting}{deduplication_method}.sem_ids'
        )

        if not os.path.exists(sem_ids_path):
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

            # Generate semantic IDs
            training_item_mask = self._get_items_for_training(dataset)
            if vq_method == 'opq':
                self.log(f'[TOKENIZER] Semantic IDs not found. Training index using Faiss OPQ...')
                self._generate_semantic_id_opq(sent_embs, sem_ids_path, training_item_mask)
            elif vq_method == 'rqkmeans':
                self.log(f'[TOKENIZER] Semantic IDs not found. Training index using RQ K-means (Faiss)...')
                self._generate_semantic_id_rqkmeans(sent_embs, sem_ids_path, training_item_mask)
            elif vq_method == 'rqvae':
                self.log(f'[TOKENIZER] Semantic IDs not found. Training RQ-VAE model...')
                self._generate_semantic_id_rqvae(sent_embs, sem_ids_path, training_item_mask)
            else:
                raise ValueError(f'Unknown vq_method: {vq_method}. Supported methods: opq, rqkmeans, rqvae')

        self.log(f'[TOKENIZER] Loading semantic IDs from {sem_ids_path}...')
        item2sem_ids = json.load(open(sem_ids_path, 'r'))
        item2sem_ids = self._permute_semantic_ids(item2sem_ids)
        item2tokens = self._sem_ids_to_tokens(item2sem_ids)

        return item2tokens

    @property
    def n_digit(self):
        """
        Returns the number of digits for the tokenizer.

        The number of digits is determined by the value of `vq_n_codebooks` in the configuration.
        """
        return self.config['vq_n_codebooks']

    @property
    def codebook_sizes(self):
        """
        Returns the codebook size for the PSID tokenizer.

        If `vq_codebook_size` is a list, it returns the list as is.
        If `vq_codebook_size` is an integer, it returns a list with `n_digit` elements,
        where each element is equal to `vq_codebook_size`.

        Returns:
            list: The codebook size for the PSID tokenizer.
        """
        if isinstance(self.config['vq_codebook_size'], list):
            return self.config['vq_codebook_size']
        else:
            return [self.config['vq_codebook_size']] * self.n_digit

    def _token_single_user(self, user: str) -> int:
        """
        Tokenizes a single user.

        Args:
            user (str): The user to tokenize.

        Returns:
            int: The tokenized user ID.

        """
        user_id = self.user2id[user]
        return self.base_user_token + user_id % self.n_user_tokens

    def _token_single_item(self, item: str) -> int:
        """
        Tokenizes a single item.

        Args:
            item (str): The item to be tokenized.

        Returns:
            list: The tokens corresponding to the item.
        """
        return self.item2tokens[item]

    def _tokenize_once(self, example: dict) -> tuple:
        """
        Tokenizes a single example.

        Args:
            example (dict): A dictionary containing the example data.

        Returns:
            tuple: A tuple containing the tokenized input_ids, attention_mask, and labels.
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

        # labels
        labels = list(self._token_single_item(example['item_seq'][-1])) + [self.eos_token]

        return input_ids, attention_mask, labels

    def tokenize_function(self, example: dict, split: str) -> dict:
        """
        Tokenizes the input example based on the specified split.

        Args:
            example (dict): The input example containing user and item sequence.
            split (str): The split type, either 'train' or any other value.

        Returns:
            dict: A dictionary containing the tokenized input, attention mask, and labels.
                - If split is 'train', returns:
                    {
                        'input_ids': List[List[int]],
                        'attention_mask': List[List[int]],
                        'labels': List[List[int]]
                    }
                - If split is not 'train', returns:
                    {
                        'input_ids': List[int],
                        'attention_mask': List[int],
                        'labels': List[int]
                    }
        """
        if split == 'train':
            n_return_examples = len(example['item_seq'][0]) - 1
            all_input_ids, all_attention_mask, all_labels = [], [], []
            for i in range(n_return_examples):
                cur_example = {
                    'user': example['user'][0],
                    'item_seq': example['item_seq'][0][:i+2]
                }
                input_ids, attention_mask, labels = self._tokenize_once(cur_example)
                all_input_ids.append(input_ids)
                all_attention_mask.append(attention_mask)
                all_labels.append(labels)
            return {
                'input_ids': all_input_ids,
                'attention_mask': all_attention_mask,
                'labels': all_labels
            }
        else:
            input_ids, attention_mask, labels = self._tokenize_once({k: v[0] for k, v in example.items()})
            return {'input_ids': [input_ids], 'attention_mask': [attention_mask], 'labels': [labels]}

    def tokenize(self, datasets: dict) -> dict:
        """
        Tokenizes the given datasets.

        Args:
            datasets (dict): A dictionary of datasets to tokenize.

        Returns:
            dict: A dictionary of tokenized datasets.
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
        """
        Returns the vocabulary size for the PSID tokenizer.
        """
        return self.eos_token + 1

    @property
    def max_token_seq_len(self) -> int:
        """
        Returns the maximum token sequence length for the PSID tokenizer.
        """
        # +2 for user token and eos token
        return self.config['max_item_seq_len'] * self.n_digit + 2

    def _get_codebook_bits(self, n_codebook):
        x = math.log2(n_codebook)
        assert x.is_integer() and x >= 0, "Invalid value for n_codebook"
        return int(x)

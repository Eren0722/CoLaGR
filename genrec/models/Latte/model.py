"""
Latte (Latent Token) Model

A model that extends PSID by adding latent tokens before semantic IDs.

Key Features:
1. Training: Standard T5 conditional generation with latent token + semantic IDs.
2. Inference: Beam search that generates [latent_token, sem_id_1, ..., sem_id_n],
   then aggregates probabilities for items with the same semantic ID.

The aggregation logic supports both 'agg_sum' and 'agg_max'.
"""

import torch
import numpy as np
from collections import defaultdict
from transformers import T5Config, T5ForConditionalGeneration

from genrec.model import AbstractModel
from genrec.dataset import AbstractDataset
from genrec.tokenizer import AbstractTokenizer


class Latte(AbstractModel):
    """
    Latte (Latent Token) model for generative recommendation.
    
    Latte extends PSID by adding latent tokens before the semantic IDs.
    During prediction, the model first predicts a latent token, then the semantic IDs.
    Multiple predictions mapping to the same item are aggregated using agg_sum or agg_max.

    Args:
        config (dict): Configuration parameters for the model.
        dataset (AbstractDataset): The dataset object.
        tokenizer (AbstractTokenizer): The tokenizer object.

    Attributes:
        t5 (T5ForConditionalGeneration): The T5 model for conditional generation.
        aggregation_method (str): Method to aggregate predictions ('agg_sum' or 'agg_max').
    """
    
    def __init__(
        self,
        config: dict,
        dataset: AbstractDataset,
        tokenizer: AbstractTokenizer,
    ):
        super(Latte, self).__init__(config, dataset, tokenizer)

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
        self.aggregation_method = config.get('aggregation_method', 'agg_max')
        self.log = tokenizer.log
        self.log("[MODEL] Latte model initialized with aggregation method: %s" % self.aggregation_method)

    @property
    def n_parameters(self) -> str:
        """
        Calculates the number of trainable parameters in the model.

        Returns:
            str: A string containing the number of embedding parameters, non-embedding parameters, and total trainable parameters.
        """
        total_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        emb_params = sum(p.numel() for p in self.t5.get_input_embeddings().parameters() if p.requires_grad)
        return f'#Embedding parameters: {emb_params}\n' \
                f'#Non-embedding parameters: {total_params - emb_params}\n' \
                f'#Total trainable parameters: {total_params}\n'

    def forward(self, batch: dict) -> torch.Tensor:
        """
        Forward pass of the model. Returns the output logits and the loss value.

        Args:
            batch (dict): A dictionary containing the input data for the model.

        Returns:
            outputs (ModelOutput): 
                The output of the model, which includes:
                - loss (torch.Tensor)
                - logits (torch.Tensor)
        """
        # Remove non-tensor fields for forward pass
        model_batch = {k: v for k, v in batch.items() if isinstance(v, torch.Tensor)}
        outputs = self.t5(**model_batch)
        return outputs

    def generate(self, batch: dict, n_return_sequences: int = 1) -> torch.Tensor:
        """
        Generates sequences using beam search and aggregates probabilities.

        Args:
            batch (dict): A dictionary containing input_ids and attention_mask.
            n_return_sequences (int): The number of sequences to return per example.

        Returns:
            torch.Tensor: Predicted semantic ID tokens, shape (batch_size, n_return_sequences, n_digit)
        """
        n_digit = self.tokenizer.n_digit
        num_beams = self.config['num_beams']
        batch_size = batch['input_ids'].shape[0]
        
        # Calculate max topk (topk can be a list)
        topk_config = self.config.get('topk', [10])
        max_topk = max(topk_config) if isinstance(topk_config, list) else topk_config
        
        # Try with current beam size first
        aggregated_preds = self._generate_and_aggregate(
            batch, num_beams, n_return_sequences, n_digit
        )
        
        # For 'agg_max' aggregation, retry with larger beam if insufficient results
        if self.aggregation_method == 'agg_max':
            # Check if any batch has insufficient predictions
            retry_needed = False
            for b in range(batch_size):
                # Count non-zero predictions (valid items)
                non_zero_count = (aggregated_preds[b].sum(dim=-1) != 0).sum().item()
                if non_zero_count < max_topk:
                    retry_needed = True
                    break
            
            if retry_needed:
                # Retry with doubled beam size
                num_beams_retry = num_beams * 2
                aggregated_preds = self._generate_and_aggregate(
                    batch, num_beams_retry, n_return_sequences, n_digit
                )
        
        return aggregated_preds

    def _generate_and_aggregate(
        self,
        batch: dict,
        num_beams: int,
        n_return_sequences: int,
        n_digit: int,
    ) -> torch.Tensor:
        """
        Helper method to generate sequences and aggregate predictions.
        
        Args:
            batch: Dictionary with 'input_ids', 'attention_mask', 'labels'
            num_beams: Number of beams for beam search
            n_return_sequences: Number of sequences to return per example
            n_digit: Number of semantic ID digits
            
        Returns:
            torch.Tensor: Aggregated predictions, shape (batch_size, n_return_sequences, n_digit)
        """
        batch_size = batch['input_ids'].shape[0]
        
        # Use HuggingFace's built-in generate with KV cache
        # max_new_tokens = latent_token + n_digit semantic tokens + eos = n_digit + 2
        with torch.no_grad():
            outputs = self.t5.generate(
                input_ids=batch['input_ids'],
                attention_mask=batch['attention_mask'],
                max_new_tokens=n_digit + 2,
                num_beams=num_beams,
                num_return_sequences=num_beams,
                return_dict_in_generate=True,
                output_scores=True,
                use_cache=True,  # Enable KV cache
            )
        
        # outputs.sequences: (batch_size * num_beams, seq_len)
        # outputs.sequences_scores: (batch_size * num_beams,) - normalized log probs
        sequences = outputs.sequences
        scores = outputs.sequences_scores
        
        # Reshape: (batch_size * num_beams, seq_len) -> (batch_size, num_beams, seq_len)
        seq_len = sequences.shape[1]
        sequences = sequences.reshape(batch_size, num_beams, seq_len)
        scores = scores.reshape(batch_size, num_beams)
        
        # Extract latent_token and semantic ID tokens
        # sequences[:, :, 0] is decoder_start_token (pad token)
        # sequences[:, :, 1] is latent_token
        # sequences[:, :, 2:2+n_digit] are semantic ID tokens
        pred_tokens = sequences[:, :, 1:2+n_digit]  # (batch_size, num_beams, n_digit+1)
        
        # Aggregate probabilities
        aggregated_preds = self._aggregate_predictions(
            pred_tokens, 
            scores, 
            n_return_sequences
        )
        
        return aggregated_preds

    def _aggregate_predictions(
        self,
        pred_tokens: torch.Tensor,
        pred_scores: torch.Tensor,
        top_k: int,
    ) -> torch.Tensor:
        """
        Aggregate predictions by item and return top-k items.
        
        Args:
            pred_tokens: (batch_size, num_beams, n_digit+1) containing [latent_token, sem_tokens...]
                         sem_tokens are already with offsets (as generated by beam search)
            pred_scores: (batch_size, num_beams) log probabilities
            top_k: Number of top items to return
            
        Returns:
            torch.Tensor: (batch_size, top_k, n_digit) semantic ID tokens (with offsets) for top items
        """
        if self.aggregation_method == 'agg_max':
            return self._aggregate_predictions_max(pred_tokens, pred_scores, top_k)
        else:  # 'agg_sum' or default
            return self._aggregate_predictions_sum(pred_tokens, pred_scores, top_k)

    def _aggregate_predictions_sum(
        self,
        pred_tokens: torch.Tensor,
        pred_scores: torch.Tensor,
        top_k: int,
    ) -> torch.Tensor:
        """
        Aggregate predictions by summing probabilities of same item (logsumexp).
        
        For each item, all predictions mapping to it have their log probabilities 
        aggregated using logsumexp.
        
        Args:
            pred_tokens: (batch_size, num_beams, n_digit+1) containing [latent_token, sem_tokens...]
            pred_scores: (batch_size, num_beams) log probabilities
            top_k: Number of top items to return
            
        Returns:
            torch.Tensor: (batch_size, top_k, n_digit) semantic ID tokens for top items
        """
        batch_size, num_beams, _ = pred_tokens.shape
        n_digit = self.tokenizer.n_digit
        device = pred_tokens.device
        
        # Output tensor
        result = torch.zeros(batch_size, top_k, n_digit, dtype=torch.long, device=device)
        
        for b in range(batch_size):
            item_scores = defaultdict(float)
            item_tokens = {}  # Store tokens (with offsets) for each item
            
            for beam_idx in range(num_beams):
                tokens = pred_tokens[b, beam_idx].tolist()
                latent_token = tokens[0]
                sem_tokens = tuple(tokens[1:])  # These are semantic tokens with offsets
                
                # Validate latent token
                if (latent_token < self.tokenizer.base_latent_token or 
                    latent_token >= self.tokenizer.base_latent_token + self.tokenizer.n_latent_tokens):
                    continue  # Invalid latent token
                
                # Look up item for these tokens (with offsets)
                # Each token tuple uniquely maps to one item (conflict-free)
                if sem_tokens in self.tokenizer.tokens_to_item:
                    item = self.tokenizer.tokens_to_item[sem_tokens]
                    # Aggregate log probabilities using logsumexp
                    score = pred_scores[b, beam_idx].item()
                    if item in item_scores:
                        item_scores[item] = np.logaddexp(item_scores[item], score)
                    else:
                        item_scores[item] = score
                        item_tokens[item] = sem_tokens
            
            # Sort items by aggregated score and take top-k
            sorted_items = sorted(item_scores.items(), key=lambda x: -x[1])[:top_k]
            
            for rank, (item, _) in enumerate(sorted_items):
                if item in item_tokens:
                    result[b, rank] = torch.tensor(
                        item_tokens[item], dtype=torch.long, device=device
                    )
        
        return result

    def _aggregate_predictions_max(
        self,
        pred_tokens: torch.Tensor,
        pred_scores: torch.Tensor,
        top_k: int,
    ) -> torch.Tensor:
        """
        Aggregate predictions by keeping only the highest probability semantic ID for each item.
        
        For each item, only the semantic ID with maximum probability is kept. If a lower 
        probability semantic ID for an already-seen item is encountered, it's discarded and 
        the next item fills the position.
        
        Args:
            pred_tokens: (batch_size, num_beams, n_digit+1) containing [latent_token, sem_tokens...]
            pred_scores: (batch_size, num_beams) log probabilities
            top_k: Number of top items to return
            
        Returns:
            torch.Tensor: (batch_size, top_k, n_digit) semantic ID tokens for top items
        """
        batch_size, num_beams, _ = pred_tokens.shape
        n_digit = self.tokenizer.n_digit
        device = pred_tokens.device
        
        # Output tensor
        result = torch.zeros(batch_size, top_k, n_digit, dtype=torch.long, device=device)
        
        for b in range(batch_size):
            # Parse all predictions first
            predictions = []  # List of (score, item, tokens)
            
            for beam_idx in range(num_beams):
                tokens = pred_tokens[b, beam_idx].tolist()
                latent_token = tokens[0]
                sem_tokens = tuple(tokens[1:])  # These are semantic tokens with offsets
                
                # Validate latent token
                if (latent_token < self.tokenizer.base_latent_token or 
                    latent_token >= self.tokenizer.base_latent_token + self.tokenizer.n_latent_tokens):
                    continue  # Invalid latent token
                
                # Look up item for these tokens
                if sem_tokens in self.tokenizer.tokens_to_item:
                    item = self.tokenizer.tokens_to_item[sem_tokens]
                    score = pred_scores[b, beam_idx].item()
                    predictions.append((score, item, sem_tokens))
            
            # Sort by score (highest first)
            predictions.sort(key=lambda x: -x[0])
            
            # Greedily select top-k unique items
            seen_items = set()
            selected_tokens = []
            
            for score, item, tokens in predictions:
                if item not in seen_items:
                    seen_items.add(item)
                    selected_tokens.append(tokens)
                    if len(selected_tokens) >= top_k:
                        break
            
            # Fill result tensor
            for rank, tokens in enumerate(selected_tokens):
                result[b, rank] = torch.tensor(tokens, dtype=torch.long, device=device)
        
        return result

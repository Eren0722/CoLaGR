"""LLM adapter used by the SI stage."""

import math
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
import inspect


class SimplifiedOutput:
    """Internal helper."""

    def __init__(self, hidden_state):
        self.hidden_states = (hidden_state,)


class EchoRecLLM(nn.Module):
    """

    -         LLM   Llama
    -             token [UserRep]/[HistoryEmb]/[UserOut]/[ItemOut]
    -      LLM
    -
      * replace_out_token_all / replace_out_token_all_infer
      * get_embeddings        token
      * rec_loss / uniformity
    -      train_mode0
      1)           [HistoryEmb]   LLM hidden states    [UserOut]
      2)           [HistoryEmb]   LLM hidden states    [ItemOut]
      3)
    """
    def __init__(
        self,
        device,
        llm_model="",
        max_output_txt_len=256,
        args=None
    ):
        """
            InjectionLLM

        Args:
            device (str):        'cuda:0'   'cpu'
            llm_model (str): LLM        'llama'   'llama-3b'
            max_output_txt_len (int):            256
            args:

        1.         LLM
        2.           token
        3.
        4.
        """
        super().__init__()
        self.device = device
        self.bce_criterion = torch.nn.BCEWithLogitsLoss()
        self.args = args
        self.is_rank0 = getattr(args, 'local_rank', 0) in (-1, 0) if args is not None else True
        self.hf_local_only = bool(getattr(args, 'hf_local_only', False)) if args is not None else False
        self.llm_path = (getattr(args, 'llm_path', None) if args is not None else None)
        if isinstance(self.llm_path, str):
            self.llm_path = self.llm_path.strip() or None
        raw_endpoint = getattr(args, 'hf_endpoint', '') if args is not None else ''
        raw_mirror = getattr(args, 'hf_mirror_endpoint', 'https://hf-mirror.com') if args is not None else 'https://hf-mirror.com'
        token_from_args = getattr(args, 'hf_access_token', None) if args is not None else None
        env_token = os.environ.get('HF_TOKEN') or os.environ.get('HUGGINGFACEHUB_API_TOKEN')
        self.hf_token = (token_from_args or env_token or '').strip() or None
        self._ckpt_enabled = False
        self.hf_endpoint = self._normalize_endpoint(raw_endpoint)
        self.hf_mirror_endpoint = self._normalize_endpoint(raw_mirror)
        self.hf_use_mirror = bool(getattr(args, 'hf_use_mirror', False)) if args is not None else False
        if self.hf_use_mirror and not self.hf_endpoint:
            self.hf_endpoint = self.hf_mirror_endpoint
        self.hf_cache_dir = (getattr(args, 'hf_cache_dir', '') if args is not None else '').strip()
        self._configure_hf_runtime()
        self.chunk_split_threshold = getattr(args, 'candidate_chunk_threshold', 36) if args is not None else 36
        self.candidate_chunk_size = max(20, getattr(args, 'candidate_chunk_size', 50) if args is not None else 50)
        self.min_candidate_chunk_size = max(10, getattr(args, 'min_candidate_chunk_size', 20) if args is not None else 20)
        self.sequence_chunk_threshold = getattr(args, 'sequence_chunk_threshold', 50) if args is not None else 50
        self.sequence_chunk_size = max(2, getattr(args, 'sequence_chunk_size', 5) if args is not None else 5)
        self.min_sequence_chunk_size = max(1, getattr(args, 'min_sequence_chunk_size', 3) if args is not None else 3)



        if llm_model == 'llama':
            model_id = "meta-llama/Meta-Llama-3-8B-Instruct"
        elif llm_model == 'llama-3b':
            model_id = "meta-llama/Llama-3.2-3B-Instruct"
        else:
            raise Exception(f'{llm_model} is not supported')
        if self.llm_path:
            self.llm_path = self._resolve_local_llm_path(self.llm_path)
            self._validate_local_llm_files(self.llm_path)
            model_id = self.llm_path
            self.hf_local_only = True
            self._hf_log(f"    LLM  : {self.llm_path}")
        print()
        print("=========")

        model_kwargs = {
            'device_map': self.device,
            'torch_dtype': torch.float16,
            'trust_remote_code': True,
            'local_files_only': self.hf_local_only,
        }
        if self.hf_cache_dir:
            model_kwargs['cache_dir'] = self.hf_cache_dir
        if self.hf_token:
            model_kwargs['token'] = self.hf_token

        model_kwargs['use_cache'] = False
        self.llm_model = self._hf_download(
            lambda: AutoModelForCausalLM.from_pretrained(model_id, **model_kwargs),
            "LLM checkpoint"
        )

        if torch.cuda.is_available():
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            torch.backends.cudnn.benchmark = True

        tokenizer_kwargs = {
            'use_fast': False,
            'trust_remote_code': True,
            'local_files_only': self.hf_local_only,
        }
        if self.hf_cache_dir:
            tokenizer_kwargs['cache_dir'] = self.hf_cache_dir
        if self.hf_token:
            tokenizer_kwargs['token'] = self.hf_token
        self.llm_tokenizer = self._hf_download(
            lambda: AutoTokenizer.from_pretrained(model_id, **tokenizer_kwargs),
            "LLM tokenizer"
        )

        try:
            self.llm_model.config.use_cache = False
            if hasattr(self.llm_model.config, 'max_length'):
                self.llm_model.config.max_length = min(getattr(self.llm_model.config, 'max_length', 2048), 1024)
        except Exception:
            pass
        if hasattr(self.llm_model, "gradient_checkpointing_enable"):
            try:
                self.llm_model.gradient_checkpointing_enable()
                self._hf_log("  LLM gradient checkpointing    ")
            except Exception as e:
                self._hf_log(f"        gradient checkpointing: {e}")

        default_prompt_len = 896 if llm_model == 'llama-3b' else 1024
        requested_max_len = getattr(args, 'llm_max_length', default_prompt_len) if args is not None else default_prompt_len
        config_max_len = getattr(self.llm_model.config, 'max_position_embeddings', requested_max_len)
        self.max_prompt_length = max(256, min(requested_max_len, config_max_len))

        self.llm_tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        self.llm_tokenizer.add_special_tokens({'bos_token': '</s>'})
        self.llm_tokenizer.add_special_tokens({'eos_token': '</s>'})
        self.llm_tokenizer.add_special_tokens({'unk_token': '</s>'})

        self.llm_tokenizer.add_special_tokens({
            'additional_special_tokens': ['[UserRep]', '[HistoryEmb]', '[UserOut]', '[ItemOut]']
        })
        self.llm_tokenizer.add_special_tokens({'cls_token': "[CLS]"})

        self.llm_model.resize_token_embeddings(len(self.llm_tokenizer))

        for name, param in self.llm_model.named_parameters():
            if args.token:
                if 'token' in name:
                    param.requires_grad = True
                else:
                    param.requires_grad = False
            else:
                param.requires_grad = False

        if not args.token:
            if args.nn_parameter:
                self.CLS = nn.Parameter(torch.normal(0, 1, size=(1, self.llm_model.config.hidden_size))).to(device)
                self.CLS_item = nn.Parameter(torch.normal(0, 1, size=(1, self.llm_model.config.hidden_size))).to(device)
            else:
                self.CLS = nn.Embedding(1, self.llm_model.config.hidden_size).to(device)
                nn.init.normal_(
                    self.CLS.weight,
                    mean=self.llm_model.model.embed_tokens.weight.mean(),
                    std=self.llm_model.model.embed_tokens.weight.std()
                )

                self.CLS_item = nn.Embedding(1, self.llm_model.config.hidden_size).to(device)
                nn.init.normal_(
                    self.CLS_item.weight,
                    mean=self.llm_model.model.embed_tokens.weight.mean(),
                    std=self.llm_model.model.embed_tokens.weight.std()
                )

        self.pred_user = nn.Sequential(
                nn.Linear(self.llm_model.config.hidden_size, 2048),
                nn.LayerNorm(2048),
                nn.LeakyReLU(),
                nn.Linear(2048, 128)
            )
        try:
            nn.init.xavier_normal_(self.pred_user[0].weight)
            nn.init.xavier_normal_(self.pred_user[3].weight)
        except Exception as e:
            pass

        self.pred_item = nn.Sequential(
                nn.Linear(self.llm_model.config.hidden_size, 2048),
                nn.LayerNorm(2048),
                nn.LeakyReLU(),
                nn.Linear(2048, 128)
            )
        try:
            nn.init.xavier_normal_(self.pred_item[0].weight)
            nn.init.xavier_normal_(self.pred_item[3].weight)
        except Exception as e:
            pass

        self.pred_user_CF2 = nn.Sequential(
                nn.Linear(64, 128),
                nn.LayerNorm(128),
                nn.GELU(),
                nn.Linear(128, 128)
            )
        try:
            nn.init.xavier_normal_(self.pred_user_CF2[0].weight)
            nn.init.xavier_normal_(self.pred_user_CF2[3].weight)
        except Exception as e:
            pass

        self.iteration_count = 0
        self.enable_heterogeneous_fusion = False
        self.llm_trainable = True

        self.mse = nn.MSELoss()
        self.max_output_txt_len = max_output_txt_len

        self.fixed_item_embeddings = None
        self.fixed_item_embeddings_dim = None
        self._sa_index_warned = False
        self._sa_teacher_warned = False
        self.fixed_item_embeddings_path = None
        self._load_fixed_item_embeddings()

    def _normalize_endpoint(self, url: str) -> str:
        if not url:
            return ""
        return url.rstrip("/")

    def _hf_log(self, message: str):
        if self.is_rank0:
            print(message)

    def _resolve_local_llm_path(self, path: str) -> str:
        expanded = os.path.expandvars(os.path.expanduser(path))
        return os.path.abspath(expanded)

    def _validate_local_llm_files(self, base_dir: str):
        if not os.path.isdir(base_dir):
            raise FileNotFoundError(f"Required path does not exist: {path}")
        required_files = ["config.json", "tokenizer_config.json", "tokenizer.json"]
        missing = [fname for fname in required_files if not os.path.isfile(os.path.join(base_dir, fname))]
        if missing:
            raise FileNotFoundError(f"Required path does not exist: {path}")
        has_shard = any(
            fname.endswith(".safetensors") and fname.startswith("model-")
            for fname in os.listdir(base_dir)
        )
        if not has_shard:
            raise FileNotFoundError(f"Required path does not exist: {path}")

    def _configure_hf_runtime(self):
        if self.hf_cache_dir:
            os.makedirs(self.hf_cache_dir, exist_ok=True)
            os.environ.setdefault("HF_HOME", self.hf_cache_dir)
            transformers_cache = os.path.join(self.hf_cache_dir, "transformers")
            os.makedirs(transformers_cache, exist_ok=True)
            os.environ.setdefault("TRANSFORMERS_CACHE", transformers_cache)

        if self.hf_local_only:
            os.environ["HF_HUB_OFFLINE"] = "1"
            os.environ["TRANSFORMERS_OFFLINE"] = "1"
        else:
            os.environ.pop("HF_HUB_OFFLINE", None)
            os.environ.pop("TRANSFORMERS_OFFLINE", None)

        endpoint = self.hf_endpoint
        if not endpoint and self.hf_use_mirror:
            endpoint = self.hf_mirror_endpoint
        if endpoint:
            os.environ["HF_ENDPOINT"] = endpoint

    def _hf_download(self, factory, desc: str, allow_mirror: bool = True):
        if self.hf_local_only:
            return factory()

        endpoints = []

        def add_endpoint(url: str):
            normalized = self._normalize_endpoint(url)
            if normalized and normalized not in endpoints:
                endpoints.append(normalized)

        add_endpoint(self.hf_endpoint)
        add_endpoint(os.environ.get("HF_ENDPOINT"))
        add_endpoint("https://huggingface.co")
        if allow_mirror:
            add_endpoint(self.hf_mirror_endpoint)

        last_exc = None
        for endpoint in endpoints or ["https://huggingface.co"]:
            os.environ["HF_ENDPOINT"] = endpoint
            self._hf_log(f"  {desc}:    {endpoint}")
            try:
                return factory()
            except OSError as exc:
                last_exc = exc
                self._hf_log(f"   {desc}: {endpoint}    {exc}")
        raise last_exc

    def set_trainable(self, flag: bool):
        """
          LLM
        
                        LLM backbone
        - sequence_injection  (flag=True)   pred_user/pred_item/pred_user_CF2/CLS/CLS_item
        - semantic alignment  (flag=False)   LLM
        
          DDP   flag=True              SI
          find_unused_parameters=True  undefined gradient
        """
        self.llm_trainable = flag
        
        for param in self.llm_model.parameters():
            param.requires_grad = False
        
        if flag:
            for param in self.pred_user.parameters():
                param.requires_grad = True
            for param in self.pred_item.parameters():
                param.requires_grad = True
            for param in self.pred_user_CF2.parameters():
                param.requires_grad = True
            
            if hasattr(self, 'CLS'):
                if isinstance(self.CLS, nn.Parameter):
                    self.CLS.requires_grad = True
                else:
                    for param in self.CLS.parameters():
                        param.requires_grad = True
            if hasattr(self, 'CLS_item'):
                if isinstance(self.CLS_item, nn.Parameter):
                    self.CLS_item.requires_grad = True
                else:
                    for param in self.CLS_item.parameters():
                        param.requires_grad = True
        else:
            for param in self.pred_user.parameters():
                param.requires_grad = False
            for param in self.pred_item.parameters():
                param.requires_grad = False
            if hasattr(self, 'CLS'):
                if isinstance(self.CLS, nn.Parameter):
                    self.CLS.requires_grad = False
                else:
                    for param in self.CLS.parameters():
                        param.requires_grad = False
            if hasattr(self, 'CLS_item'):
                if isinstance(self.CLS_item, nn.Parameter):
                    self.CLS_item.requires_grad = False
                else:
                    for param in self.CLS_item.parameters():
                        param.requires_grad = False

    def _set_gradient_checkpointing(self, enabled: bool):
        """Internal helper."""
        if not hasattr(self.llm_model, "gradient_checkpointing_enable"):
            self._ckpt_enabled = False
            return
        try:
            try:
                self.llm_model.config.use_cache = False
            except Exception:
                pass
            if enabled:
                sig = inspect.signature(self.llm_model.gradient_checkpointing_enable)
                if "use_reentrant" in sig.parameters:
                    self.llm_model.gradient_checkpointing_enable(use_reentrant=False)
                else:
                    self.llm_model.gradient_checkpointing_enable()
                self._ckpt_enabled = True
            else:
                if hasattr(self.llm_model, "gradient_checkpointing_disable"):
                    self.llm_model.gradient_checkpointing_disable()
                self._ckpt_enabled = False
        except Exception:
            self._ckpt_enabled = False

    def rec_loss(self, anchor, items):
        """

              Start_old:     logits + cross_entropy    /clamp/fp32
        """
        if torch.isnan(anchor).any() or torch.isinf(anchor).any():
            pass
        if torch.isnan(items).any() or torch.isinf(items).any():
            pass
        
        logits = torch.bmm(
            items.view(anchor.shape[0], -1, anchor.shape[1]),
            anchor.unsqueeze(2)
        ).squeeze(2)

        if torch.isnan(logits).any() or torch.isinf(logits).any():
            pass

        labels = torch.zeros(logits.size(0), dtype=torch.long, device=logits.device)

        loss = F.cross_entropy(logits, labels)

        if torch.isnan(loss) or torch.isinf(loss):
            pass

        return loss

    def uniformity(self, x, p=2):
        """
        Uniformity


        Args:
            x (torch.Tensor):         [batch_size, dim]
            p (int):    p      2

        Returns:
            torch.Tensor: Uniformity

        uniformity = mean(exp(-p * ||xi - xj||_p^2))
          xi, xj
        """
        return torch.pdist(x, p=p).pow(2).mul(-p).exp().mean()

    def replace_out_token_all(self, llm_tokens, inputs_embeds, token=[], embs=None):
        """
               token

          EchoRec             token
                  CF-SRec          LLM

        Args:
            llm_tokens (dict):     token      'input_ids'
            inputs_embeds (torch.Tensor): LLM         [batch_size, seq_len, hidden_size]
            token (list):        token     ['[UserOut]', '[HistoryEmb]']
            embs (dict):            token

        Returns:
            torch.Tensor:

        - [HistoryEmb]:
        - [UserRep]:
        - [UserOut]/[ItemOut]:        CLS       token

        1.          token
        2.    token
        3.   token
        4.
        """
        for t in token:
            token_id = self.llm_tokenizer(t, return_tensors="pt", add_special_tokens=False).input_ids.item()
            vectors = []

            for inx in range(len(llm_tokens["input_ids"])):
                idx_tensor = (llm_tokens["input_ids"][inx] == token_id).nonzero().view(-1)
                user_vector = inputs_embeds[inx]

                if 'Emb' in t:
                    ee = embs[t][inx]
                    for idx, item_emb in zip(idx_tensor, ee):
                        user_vector = torch.cat((user_vector[:idx], item_emb.unsqueeze(0), user_vector[idx+1:]), dim=0)

                elif 'Rep' in t:
                    for idx in idx_tensor:
                        user_emb = embs[t][inx]
                        user_vector = torch.cat((user_vector[:idx], user_emb.unsqueeze(0), user_vector[idx+1:]), dim=0)

                else:
                    if not self.args.token:
                        for idx in idx_tensor:
                            if 'UserOut' in t:
                                if self.args.nn_parameter:
                                    user_vector = torch.cat((user_vector[:idx], self.CLS[torch.tensor([0]).to(self.device)], user_vector[idx+1:]), dim=0)
                                else:
                                    user_vector = torch.cat((user_vector[:idx], self.CLS(torch.tensor([0]).to(self.device)), user_vector[idx+1:]), dim=0)
                            elif 'ItemOut' in t:
                                if self.args.nn_parameter:
                                    user_vector = torch.cat((user_vector[:idx], self.CLS_item[torch.tensor([0]).to(self.device)], user_vector[idx+1:]), dim=0)
                                else:
                                    user_vector = torch.cat((user_vector[:idx], self.CLS_item(torch.tensor([0]).to(self.device)), user_vector[idx+1:]), dim=0)

                vectors.append(user_vector.unsqueeze(0))

            inputs_embeds = torch.cat(vectors)
        return inputs_embeds
    
    def replace_out_token_all_infer(self, llm_tokens, inputs_embeds, token=[], embs=None, user_act=False, item_act=False):
        """
               token

          replace_out_token_all
        - replace_out_token_all:
        - replace_out_token_all_infer:

        1. [HistoryEmb]
        2.
        3.

        Args:
            llm_tokens (dict):     token
            inputs_embeds (torch.Tensor): LLM
            token (list):        token
            embs (dict):
            user_act (bool):
            item_act (bool):

        Returns:
            torch.Tensor:
        """
        for t in token:
            token_id = self.llm_tokenizer(t, return_tensors="pt", add_special_tokens=False).input_ids.item()
            vectors = []

            for inx in range(len(llm_tokens["input_ids"])):
                idx_tensor = (llm_tokens["input_ids"][inx] == token_id).nonzero().view(-1)
                user_vector = inputs_embeds[inx]

                if 'Emb' in t:
                    ee = [embs[t][inx]]
                    for idx, item_emb in zip(idx_tensor, ee):
                        user_vector = torch.cat((user_vector[:idx], item_emb.unsqueeze(0), user_vector[idx+1:]), dim=0)

                elif 'Rep' in t:
                    for idx in idx_tensor:
                        user_emb = embs[t][inx]
                        user_vector = torch.cat((user_vector[:idx], user_emb.unsqueeze(0), user_vector[idx+1:]), dim=0)

                else:
                    if not self.args.token:
                        for idx in idx_tensor:
                            if 'UserOut' in t:
                                if self.args.nn_parameter:
                                    user_vector = torch.cat((user_vector[:idx], self.CLS[torch.tensor([0]).to(self.device)], user_vector[idx+1:]), dim=0)
                                else:
                                    user_vector = torch.cat((user_vector[:idx], self.CLS(torch.tensor([0]).to(self.device)), user_vector[idx+1:]), dim=0)
                            elif 'ItemOut' in t:
                                if self.args.nn_parameter:
                                    user_vector = torch.cat((user_vector[:idx], self.CLS_item[torch.tensor([0]).to(self.device)], user_vector[idx+1:]), dim=0)
                                else:
                                    user_vector = torch.cat((user_vector[:idx], self.CLS_item(torch.tensor([0]).to(self.device)), user_vector[idx+1:]), dim=0)

                vectors.append(user_vector.unsqueeze(0))

            inputs_embeds = torch.cat(vectors)
        return inputs_embeds

    def _load_fixed_item_embeddings(self):
        """Internal helper."""
        if self.args is None:
            return
        emb_path = getattr(self.args, 'llm_emb_path', '') or ''
        emb_path = emb_path.strip()
        if not emb_path:
            dataset = getattr(self.args, 'rec_pre_trained_data', '').strip()
            llm_name = getattr(self.args, 'llm', '').strip()
            save_dir = getattr(self.args, 'save_dir', '').strip()
            default_filename = f"{dataset}_{llm_name}_all_embeddings.pt" if dataset and llm_name else ''
            candidate_dirs = []
            if dataset and save_dir:
                candidate_dirs.append(os.path.join('./SA_assets', save_dir, dataset))
            if dataset:
                candidate_dirs.append(os.path.join('./SA_assets', dataset))
            if save_dir:
                candidate_dirs.append(os.path.join('./SA_assets', save_dir))
            candidate_dirs.append('./SA_assets')

            candidate_paths = []
            for base in candidate_dirs:
                if not base:
                    continue
                if default_filename:
                    candidate_paths.append(os.path.join(base, default_filename))
                candidate_paths.append(os.path.join(base, 'all_embeddings.pt'))

            for cand in candidate_paths:
                expanded_cand = os.path.abspath(os.path.expanduser(cand))
                if os.path.exists(expanded_cand):
                    emb_path = expanded_cand
                    if not getattr(self.args, 'local_rank', None) or self.args.local_rank in (-1, 0):
                        pass
                    break

            if not emb_path:
                return
        expanded = os.path.abspath(os.path.expanduser(emb_path))
        if not os.path.exists(expanded):
            pass
            return
        try:
            emb_data = torch.load(expanded, map_location='cpu')
        except Exception as exc:
            pass
            return
        if isinstance(emb_data, dict):
            for key in ('embeddings', 'all_embeddings', 'item_embeddings', 'data', 'tensor'):
                if key in emb_data:
                    emb_data = emb_data[key]
                    break
        if not torch.is_tensor(emb_data):
            pass
            return
        emb_data = emb_data.float().contiguous()
        if emb_data.ndim != 2 or emb_data.size(0) == 0:
            pass
            return
        first_row = emb_data[0]
        if not torch.allclose(first_row, torch.zeros_like(first_row), atol=1e-6):
            pad = torch.zeros(1, emb_data.size(1), dtype=emb_data.dtype)
            emb_data = torch.cat([pad, emb_data], dim=0)
        self.fixed_item_embeddings = emb_data
        self.fixed_item_embeddings_dim = emb_data.size(1)
        self.fixed_item_embeddings_path = expanded
        pass

    def has_fixed_item_embeddings(self) -> bool:
        return isinstance(self.fixed_item_embeddings, torch.Tensor)

    def lookup_fixed_item_embeddings(self, item_ids, device=None):
        """Internal helper."""
        if not self.has_fixed_item_embeddings():
            raise RuntimeError("       LLM     ")
        if not torch.is_tensor(item_ids):
            ids = torch.as_tensor(item_ids, dtype=torch.long)
        else:
            ids = item_ids.detach().long()
        flat_ids = ids.view(-1).cpu()
        max_valid = self.fixed_item_embeddings.size(0) - 1
        if flat_ids.numel() > 0 and flat_ids.max().item() > max_valid:
            if not self._sa_index_warned:
                pass
                self._sa_index_warned = True
            flat_ids = torch.clamp(flat_ids, 0, max_valid)
        gathered = self.fixed_item_embeddings.index_select(0, flat_ids)
        gathered = gathered.view(*ids.shape, -1)
        target_device = device if device is not None else self.device
        return gathered.to(target_device)

    def encode_user_representations(self, text_input, history_embs, max_length=None, need_grad=False):
        """Internal helper."""
        if text_input is None or len(text_input) == 0:
            return None
        max_input_length = max_length or getattr(self, 'max_prompt_length', 896)
        llm_tokens = self.llm_tokenizer(
            text_input,
            return_tensors="pt",
            padding="longest",
            truncation=True,
            max_length=max_input_length,
        ).to(self.device)

        inputs_embeds = self.llm_model.get_input_embeddings()(llm_tokens['input_ids'])
        inputs_embeds = self.replace_out_token_all(
            llm_tokens,
            inputs_embeds,
            token=['[UserOut]', '[HistoryEmb]'],
            embs={'[HistoryEmb]': history_embs}
        )

        batch_size = inputs_embeds.size(0)
        need_grad = bool(need_grad)
        chunk_threshold = getattr(self, 'sequence_chunk_threshold', batch_size + 1)

        if batch_size > chunk_threshold:
            seq_chunk_size = min(self.sequence_chunk_size, batch_size)
            min_seq_chunk = max(2, min(batch_size, self.min_sequence_chunk_size))
            hidden_states, _ = self._run_chunked_hidden_states(
                inputs_embeds,
                seq_chunk_size,
                need_grad,
                min_chunk_size=min_seq_chunk,
                log_prefix="SA    ",
                use_checkpoint=need_grad,
            )
            last_hidden = hidden_states
        else:
            grad_ctx = torch.enable_grad if need_grad else torch.no_grad
            with grad_ctx():
                with torch.amp.autocast('cuda'):
                    outputs = self.llm_model.model(
                        inputs_embeds=inputs_embeds,
                        output_hidden_states=True
                    )
            last_hidden = outputs.hidden_states[-1]

        indx = self.get_embeddings(llm_tokens, '[UserOut]')
        user_outputs = torch.cat([
            last_hidden[i, indx[i]].mean(axis=0).unsqueeze(0)
            for i in range(len(indx))
        ])

        if not need_grad:
            user_outputs = user_outputs.detach()
            with torch.no_grad():
                user_outputs = self.pred_user(user_outputs)
            return user_outputs.detach()

        user_outputs = self.pred_user(user_outputs)
        return user_outputs

    def _run_chunked_hidden_states(self, embeds, chunk_size, need_grad,
                                   min_chunk_size=4, log_prefix=None, use_checkpoint=True):
        batch_total = embeds.size(0)
        chunk_size = min(chunk_size, batch_total)
        min_chunk = max(1, min(batch_total, min_chunk_size))
        grad_ctx = torch.enable_grad if need_grad else torch.no_grad

        while True:
            try:
                first_chunk_output = None
                result_outputs = None

                for i in range(0, batch_total, chunk_size):
                    end_idx = min(i + chunk_size, batch_total)
                    chunk_embeds = embeds[i:end_idx]
                    chunk_idx = i // chunk_size

                    with grad_ctx():
                        with torch.amp.autocast('cuda', dtype=torch.float16):
                            chunk_output = self.llm_model.model(
                                inputs_embeds=chunk_embeds,
                                output_hidden_states=True,
                                use_cache=True
                            )

                    chunk_hidden = chunk_output.hidden_states[-1]
                    
                    if result_outputs is None:
                        hidden_dim = chunk_hidden.shape[-1]
                        result_outputs = torch.empty(
                            (batch_total, chunk_hidden.shape[1], hidden_dim),
                            dtype=chunk_hidden.dtype,
                            device=chunk_hidden.device
                        )
                    
                    result_outputs[i:end_idx] = chunk_hidden
                    
                    del chunk_embeds, chunk_output, chunk_hidden

                return result_outputs, chunk_size
            except torch.cuda.OutOfMemoryError as oom:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                next_chunk = max(min_chunk, chunk_size // 2)
                if next_chunk == chunk_size:
                    raise oom
                chunk_size = next_chunk
                if log_prefix:
                    pass

    def get_embeddings(self, llm_tokens, token):
        """
              token

                  token  [UserOut], [ItemOut]  tokenized
             LLM

        Args:
            llm_tokens (dict):     token      'input_ids'
            token (str):       token   '[UserOut]'

        Returns:
            list:       token
                    [[pos1, pos2], [pos3], ...]

        -  LLM       hidden_states
        -             token
        """
        token_idx = []
        token_id = self.llm_tokenizer(token, return_tensors="pt", add_special_tokens=False).input_ids.item()

        for inx in range(len(llm_tokens['input_ids'])):
            idx_tensor = (llm_tokens['input_ids'][inx] == token_id).nonzero().view(-1)
            token_idx.append(idx_tensor)
        return token_idx

    def forward(self, samples, mode=0):
        """

                               mode=0

        Args:
            samples (dict):
            mode (int):
                - 0: train_mode0    EchoRec
                - 1: train_mode1

        Returns:
        """
        if mode == 0:
            return self.train_mode0(samples)
        elif mode == 1:
            return self.train_mode1(samples)

    def train_mode0(self, samples):
        """
        EchoRec            Start_old

        1.              token
        2.              token
        3.          -            CF

        Args:
            samples (dict):
                - 'text_input':        prompt
                - 'log_emb': CF-SRec
                - 'candidates_pos':        prompt
                - 'interact':              [HistoryEmb]
                - 'candidate_embs':

        Returns:
            Tuple[torch.Tensor, float, float]:
                - loss:          +
                - rec_loss.item():
                - aux_summary.item():
        """
        self.iteration_count += 1

        max_input_length = getattr(self, 'max_prompt_length', 896)
        log_emb = samples['log_emb']
        student_repr = samples.get('student_repr', log_emb)

        llm_tokens = self.llm_tokenizer(
            samples['text_input'],
            return_tensors="pt",
            padding="longest",
            truncation=True,
            max_length=max_input_length,
        ).to(self.device)

        inputs_embeds = self.llm_model.get_input_embeddings()(llm_tokens['input_ids'])
        inputs_embeds = self.replace_out_token_all(
            llm_tokens,
            inputs_embeds,
            token=['[UserOut]', '[HistoryEmb]'],
            embs={'[HistoryEmb]': samples['interact']}
        )

        candi_tokens = self.llm_tokenizer(
            samples['candidates_pos'],
            return_tensors="pt",
            padding="longest",
            truncation=True,
            max_length=max_input_length,
        ).to(self.device)

        candi_embeds = self.llm_model.get_input_embeddings()(candi_tokens['input_ids'])
        candi_embeds = self.replace_out_token_all_infer(
            candi_tokens,
            candi_embeds,
            token=['[ItemOut]', '[HistoryEmb]'],
            embs={'[HistoryEmb]': samples['candidate_embs']}
        )

        need_llm_grad = True
        total_candidates = candi_embeds.size(0)
        if total_candidates > self.chunk_split_threshold:
            chunk_size = min(self.candidate_chunk_size, total_candidates)
            min_chunk = max(4, min(total_candidates, self.min_candidate_chunk_size))

            if not hasattr(self, '_chunk_logged'):
                pass
                self._chunk_logged = True

            candi_hidden_states, final_chunk = self._run_chunked_hidden_states(
                candi_embeds,
                chunk_size,
                need_llm_grad,
                min_chunk_size=min_chunk,
                log_prefix="LLM    ",
                use_checkpoint=need_llm_grad,
            )
            candi_outputs = SimplifiedOutput(candi_hidden_states)
            del candi_hidden_states
            self.candidate_chunk_size = final_chunk
        else:
            grad_ctx = torch.enable_grad() if need_llm_grad else torch.no_grad()
            with grad_ctx:
                with torch.amp.autocast('cuda'):
                    candi_outputs = self.llm_model.model(
                        inputs_embeds=candi_embeds,
                        output_hidden_states=True
                    )

        indx = self.get_embeddings(candi_tokens, '[ItemOut]')
        item_outputs = torch.cat([
            candi_outputs.hidden_states[-1][i, indx[i]].mean(axis=0).unsqueeze(0)
            for i in range(len(indx))
        ])
        del candi_outputs

        sequence_batch = inputs_embeds.size(0)
        sequence_chunk_threshold = getattr(self, 'sequence_chunk_threshold', sequence_batch + 1)
        if sequence_batch > sequence_chunk_threshold:
            if not hasattr(self, '_sequence_chunk_logged'):
                pass
                self._sequence_chunk_logged = True

            seq_chunk_size = min(self.sequence_chunk_size, sequence_batch)
            min_seq_chunk = max(2, min(sequence_batch, self.min_sequence_chunk_size))
            user_hidden_states, final_seq_chunk = self._run_chunked_hidden_states(
                inputs_embeds,
                seq_chunk_size,
                need_llm_grad,
                min_chunk_size=min_seq_chunk,
                log_prefix="      ",
                use_checkpoint=need_llm_grad,
            )
            outputs = SimplifiedOutput(user_hidden_states)
        else:
            with torch.amp.autocast('cuda'):
                outputs = self.llm_model.model(
                    inputs_embeds=inputs_embeds,
                    output_hidden_states=True
                )

        indx = self.get_embeddings(llm_tokens, '[UserOut]')
        user_outputs = torch.cat([
            outputs.hidden_states[-1][i, indx[i]].mean(axis=0).unsqueeze(0)
            for i in range(len(indx))
        ])
        del outputs

        user_outputs = self.pred_user(user_outputs)
        item_outputs = self.pred_item(item_outputs)

        batch_size = student_repr.shape[0]
        num_candidates = samples.get('num_candidates', max(item_outputs.shape[0] // batch_size, 1))
        item_outputs_batched = item_outputs.view(batch_size, num_candidates, -1)

        llm_rec_loss = self.rec_loss(user_outputs, item_outputs)

        teacher_cf_mapped = self.pred_user_CF2(log_emb)
        user_outputs_norm = F.normalize(user_outputs, p=2, dim=1)
        teacher_norm = F.normalize(teacher_cf_mapped, p=2, dim=1)
        match_loss = self.mse(user_outputs_norm, teacher_norm)
        match_loss = match_loss + (self.uniformity(user_outputs_norm) + self.uniformity(teacher_norm))
        match_weight = float(getattr(self.args, 'match_weight', 1.0))
        total_loss = llm_rec_loss + match_weight * match_loss

        self._alignment_info = {
            "teacher_rec": llm_rec_loss.item(),
            "match_loss": match_loss.item(),
        }

        return total_loss, llm_rec_loss.item(), match_loss.item()


InjectionLLM = EchoRecLLM

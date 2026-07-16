"""EchoRec SI model."""

import os
import pickle
import random
import time
from datetime import datetime
from functools import lru_cache

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from models.echorec_llm import EchoRecLLM
from models.echorec_teacher import EchoRecTeacher

try:
    import habana_frameworks.torch.core as htcore
except ImportError:
    htcore = None


class EchoRecSIModel(nn.Module):
    """Bridge a frozen sequential teacher and the trainable LLM side."""

    def __init__(self, args):
        super().__init__()
        self.args = args
        self.device = args.device
        self.inference_chunk_size = max(1, int(getattr(args, 'inference_chunk_size', 8)))

        data_root = getattr(args, "data_root", "./datasets")
        text_candidates = [
            f'{data_root}/{args.rec_pre_trained_data}/text_name_dict.json.gz',
        ]
        text_path = next((path for path in text_candidates if os.path.exists(path)), text_candidates[0])
        with open(text_path, 'rb') as ft:
            self.text_name_dict = pickle.load(ft)

        self.recsys = EchoRecTeacher(
            args.recsys,
            args.rec_pre_trained_data,
            self.device,
            current_args=args,
        )

        self.student_trainable = getattr(args, 'train_student', False)
        for param in self.recsys.model.parameters():
            param.requires_grad = self.student_trainable

        self.item_num = self.recsys.item_num
        self.rec_sys_dim = self.recsys.hidden_units
        self.llm_trainable = True
        self.optimizer_needs_reset = False
        self.mse = nn.MSELoss()

        self.llm = EchoRecLLM(device=self.device, llm_model=args.llm, args=self.args)
        hidden_size = self.llm.llm_model.config.hidden_size
        self.item_emb_proj = nn.Sequential(
            nn.Linear(self.rec_sys_dim, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.LeakyReLU(),
            nn.Linear(hidden_size, hidden_size),
        )
        nn.init.xavier_normal_(self.item_emb_proj[0].weight)
        nn.init.xavier_normal_(self.item_emb_proj[3].weight)

        self.current_phase = 'sequence_injection'
        self.all_embs = None
        self.history_window = max(3, min(int(getattr(self.args, 'train_history_window', 10)), 10))

        self.current_phase = 'sequence_injection'
        self.all_embs = None
        self.history_window = max(3, min(int(getattr(self.args, 'train_history_window', 10)), 10))

    def _resolve_positive_int(self, env_name: str = "", arg_name: str = "", default: int = 0) -> int:
        if env_name:
            raw_env = os.environ.get(env_name, "").strip()
            if raw_env:
                try:
                    parsed = int(raw_env)
                    if parsed > 0:
                        return parsed
                except ValueError:
                    pass

        if arg_name:
            raw_arg = getattr(self.args, arg_name, 0)
            try:
                parsed = int(raw_arg)
                if parsed > 0:
                    return parsed
            except (TypeError, ValueError):
                pass

        return default

    @staticmethod
    def _to_py_int(value):
        if isinstance(value, torch.Tensor):
            return int(value.item())
        if isinstance(value, np.generic):
            return int(value.item())
        return int(value)

    def _to_py_int_list(self, values):
        if torch.is_tensor(values):
            return [int(v) for v in values.detach().cpu().tolist()]
        return [self._to_py_int(v) for v in values]

    def _set_module_trainable(self, module, flag: bool):
        if module is None:
            return
        for param in module.parameters():
            param.requires_grad = flag

    def set_student_trainable(self, flag: bool, verbose: bool = False, force: bool = False):
        previous = getattr(self, 'student_trainable', None)
        if not force and previous == flag:
            return
        self.student_trainable = flag
        self._set_module_trainable(self.recsys.model, flag)
        try:
            self.recsys.model.train(flag)
        except Exception:
            pass
        if previous is not None and previous != flag:
            self.optimizer_needs_reset = True

    def set_llm_trainable(self, flag: bool, verbose: bool = False, force: bool = False):
        previous = getattr(self, 'llm_trainable', None)
        if not force and self.llm_trainable == flag:
            return
        self.llm_trainable = flag
        self._set_module_trainable(self.item_emb_proj, flag)
        self.llm.set_trainable(flag)
        if previous is not None and previous != flag:
            self.optimizer_needs_reset = True
        if verbose and (not hasattr(self.args, 'local_rank') or self.args.local_rank == 0):
            state = '  ' if flag else '  '
            pass
    
    def set_mapper_trainable(self, flag: bool, verbose: bool = False):
        """
            cf_to_llm_mapper
                 alignment_updates_mapper
        """
        previous = getattr(self, 'mapper_trainable', None)
        if previous is not None and previous == flag:
            return
        self.mapper_trainable = flag

        toggled_modules = []
        if hasattr(self.llm, 'cf_to_llm_mapper'):
            self._set_module_trainable(self.llm.cf_to_llm_mapper, flag)
            toggled_modules.append('cf_to_llm_mapper')
        if hasattr(self.llm, 'pred_user'):
            self._set_module_trainable(self.llm.pred_user, flag)
            toggled_modules.append('pred_user')
        if hasattr(self.llm, 'pred_item'):
            self._set_module_trainable(self.llm.pred_item, flag)
            toggled_modules.append('pred_item')

        if previous is not None and previous != flag:
            self.optimizer_needs_reset = True

        if verbose and toggled_modules and (not hasattr(self.args, 'local_rank') or self.args.local_rank == 0):
            state = '  ' if flag else '  '
            module_names = ', '.join(toggled_modules)
            pass

    def configure_training_phase(self, phase_name: str, weight_cfg: dict, verbose: bool = False, round_idx: int = 1):
        phase_name = phase_name.lower()
        if phase_name != 'sequence_injection':
            raise ValueError(f"unknown training phase: {phase_name}")
        self.set_student_trainable(False, verbose)
        self.set_llm_trainable(True, verbose)
        if hasattr(self.llm, 'set_loss_weights'):
            self.llm.set_loss_weights(
                teacher_rec=weight_cfg.get('teacher_rec', 1.0),
                student_rec=weight_cfg.get('student_rec', 0.0),
                forward=weight_cfg.get('forward', 1.0),
                backward=0.0,
            )
        self.current_phase = phase_name
        self.current_round = round_idx

    def _ensure_item_embeddings_ready(self, force_recompute: bool = False, desc: str = "Building item embeddings"):
        """Internal helper."""
        if force_recompute:
            if hasattr(self, 'all_embs'):
                delattr(self, 'all_embs')

        if hasattr(self, 'all_embs') and self.all_embs is not None:
            return self.all_embs

        batch_ = 128
        if self.args.llm in ['llama', 'llama-3b']:
            batch_ = 32
        if self.args.rec_pre_trained_data in {'Electronics'}:
            batch_ = 32
            if self.args.llm in ['llama', 'llama-3b']:
                batch_ = 16
        batch_ = self._resolve_positive_int(
            env_name="LLMSREC_EVAL_ITEM_BATCH",
            arg_name="eval_item_batch",
            default=batch_,
        )

        item_ids = list(range(1, self.item_num + 1))
        processed_items = 0
        adaptive_item_batch = batch_
        max_input_length = self._resolve_positive_int(
            env_name="LLMSREC_EVAL_MAX_LENGTH",
            arg_name="eval_max_length",
            default=512,
        )
        oom_notice_items = False
        self.all_embs = []

        with tqdm(total=self.item_num, desc=desc) as pbar:
            while processed_items < self.item_num:
                end_idx = min(processed_items + adaptive_item_batch, self.item_num)
                current_ids = item_ids[processed_items:end_idx]
                candidate_text = []
                candidate_ids = []
                for neg_candidate in current_ids:
                    candidate_text.append(self._candidate_prompt(neg_candidate))
                    candidate_ids.append(neg_candidate)

                try:
                    with torch.no_grad():
                        candi_tokens = self.llm.llm_tokenizer(
                            candidate_text,
                            return_tensors="pt",
                            padding="longest",
                            truncation=True,
                            max_length=max_input_length,
                        ).to(self.device)

                        candidate_embs = self.item_emb_proj(self.get_item_emb(candidate_ids))
                        candi_embeds = self.llm.llm_model.get_input_embeddings()(candi_tokens['input_ids'])
                        candi_embeds = self.llm.replace_out_token_all_infer(
                            candi_tokens,
                            candi_embeds,
                            token=['[ItemOut]', '[HistoryEmb]'],
                            embs={'[HistoryEmb]': candidate_embs}
                        )

                        with torch.amp.autocast('cuda'):
                            candi_outputs = self.llm.llm_model.model(
                                inputs_embeds=candi_embeds,
                                output_hidden_states=True
                            )

                            indx = self.llm.get_embeddings(candi_tokens, '[ItemOut]')
                            item_outputs = torch.cat([
                                candi_outputs.hidden_states[-1][i, indx[i]].mean(axis=0).unsqueeze(0)
                                for i in range(len(indx))
                            ])
                            item_outputs = self.llm.pred_item(item_outputs)

                        self.all_embs.append(item_outputs.detach().cpu())

                        del candi_outputs
                        del item_outputs
                        del candi_embeds
                        torch.cuda.empty_cache()

                    processed_items = end_idx
                    pbar.update(len(current_ids))
                    oom_notice_items = False

                except RuntimeError as e:
                    if 'CUDA out of memory' in str(e) and adaptive_item_batch > 1:
                        torch.cuda.empty_cache()
                        adaptive_item_batch = max(1, adaptive_item_batch // 2)
                        if not oom_notice_items:
                            pass
                            oom_notice_items = True
                        continue
                    else:
                        raise

        self.all_embs = torch.cat(self.all_embs, dim=0).contiguous()
        return self.all_embs





    def save_model(self, args, epoch2=None, best=False, subdir=None):
        """

        -           CF-SRec
        -
        -

        Args:
            args:
            epoch2 (int, optional):
            best (bool):             False

            ./models/{save_dir}/[best/]{rec_pre_trained_data}_{llm}_{epoch2}_{component}.pt
        """
        base_dir = os.path.join('./models', args.rec_pre_trained_data, args.save_dir)
        if subdir is not None:
            out_dir = os.path.join(base_dir, subdir)
        else:
            out_dir = os.path.join(base_dir, 'best') if best else base_dir

        os.makedirs(out_dir, exist_ok=True)

        out_dir = os.path.join(
            out_dir,
            f'{args.rec_pre_trained_data}_{args.llm}_{epoch2}_'
        )

        if args.train:
            compact_ckpt = {
                'item_emb_proj': self.item_emb_proj.state_dict(),
                'pred_user': self.llm.pred_user.state_dict(),
                'pred_item': self.llm.pred_item.state_dict(),
                'pred_user_CF2': self.llm.pred_user_CF2.state_dict(),
            }
            torch.save(self.item_emb_proj.state_dict(), out_dir + 'item_proj.pt')

            torch.save(self.llm.pred_user.state_dict(), out_dir + 'pred_user.pt')

            torch.save(self.llm.pred_item.state_dict(), out_dir + 'pred_item.pt')
            torch.save(self.llm.pred_user_CF2.state_dict(), out_dir + 'pred_user_cf2.pt')

            if not args.token:
                if args.nn_parameter:
                    torch.save(self.llm.CLS.data, out_dir + 'CLS.pt')
                    torch.save(self.llm.CLS_item.data, out_dir + 'CLS_item.pt')
                else:
                    torch.save(self.llm.CLS.state_dict(), out_dir + 'CLS.pt')
                    torch.save(self.llm.CLS_item.state_dict(), out_dir + 'CLS_item.pt')
                compact_ckpt['CLS'] = self.llm.CLS.state_dict()
                compact_ckpt['CLS_item'] = self.llm.CLS_item.state_dict()
            if args.token:
                torch.save(self.llm.llm_model.model.embed_tokens.state_dict(), out_dir + 'token.pt')
                compact_ckpt['token'] = self.llm.llm_model.model.embed_tokens.state_dict()


            compact_path = out_dir + 'model.pth'
            torch.save(compact_ckpt, compact_path)

    def _lookup_item_text_value(self, field, item_id, default):
        values = self.text_name_dict.get(field, {})
        return values.get(item_id, values.get(str(item_id), default))

    def find_item_text(self, item, title_flag=True, description_flag=True):
        item_ids = self._to_py_int_list(item)
        return [
            self.find_item_text_single(
                item_id,
                title_flag=title_flag,
                description_flag=description_flag,
            )
            for item_id in item_ids
        ]

    def find_item_time(self, item, user, title_flag=True, description_flag=True):
        user_id = self._to_py_int(user)
        item_ids = self._to_py_int_list(item)
        time_dict = self.text_name_dict.get('time', {})
        times = []

        for item_id in item_ids:
            timestamp = None
            item_times = time_dict.get(item_id, time_dict.get(str(item_id), {}))
            if isinstance(item_times, dict):
                timestamp = item_times.get(user_id, item_times.get(str(user_id)))

            try:
                times.append(datetime.utcfromtimestamp(int(timestamp) / 1000))
            except (TypeError, ValueError, OSError, OverflowError):
                times.append(datetime(2020, 1, 1))

        return [value.strftime('%Y-%m-%d') for value in times]

    @lru_cache(maxsize=200000)
    def find_item_text_single(self, item, title_flag=True, description_flag=True):
        item_id = self._to_py_int(item)
        title = self._lookup_item_text_value('title', item_id, 'No Title')
        description = self._lookup_item_text_value('description', item_id, 'No Description')

        if title_flag and description_flag:
            return f'"{title}, {description}"'
        if title_flag:
            return f'"{title}"'
        if description_flag:
            return f'"{description}"'
        return '""'

    @lru_cache(maxsize=200000)
    def _candidate_prompt(self, item_id):
        item_id = self._to_py_int(item_id)
        item_title = self.find_item_text_single(item_id, title_flag=True, description_flag=False)
        return f'The item title and item embedding are as follows: {item_title}[HistoryEmb], then generate item representation token:[ItemOut]'


    def get_item_emb(self, item_ids):
        """
         CF-SRec

        Args:
            item_ids (list):   ID

        Returns:
            torch.Tensor:            [len(item_ids), embedding_dim]

            -    args.nn_parameter
            - nn_parameter=True:
            - nn_parameter=False:   Embedding
        """
        with torch.no_grad():
            if torch.is_tensor(item_ids):
                item_tensor = item_ids.to(self.device, dtype=torch.long, non_blocking=True)
            else:
                item_tensor = torch.as_tensor(item_ids, dtype=torch.long, device=self.device)
            if hasattr(self.recsys.model, 'get_item_embeddings'):
                item_embs = self.recsys.model.get_item_embeddings(item_tensor)
            elif self.args.nn_parameter:
                item_embs = self.recsys.model.item_emb[item_tensor]
            else:
                item_embs = self.recsys.model.item_emb(item_tensor)

        return item_embs
    
    def forward(self, data, optimizer=None, batch_iter=None, mode='phase2'):
        if mode == 'phase2':
            self.pre_train_phase2(data, optimizer, batch_iter)

        if mode == 'generate_batch':
            self.generate_batch(data)

            eusers = getattr(self.args, 'eval_log_users', 0)
            if eusers and eusers > 0:
                if (self.users % eusers) < 1e-6:
                    print(self.args.save_dir, self.args.rec_pre_trained_data)
                    print('test (NDCG@10: %.4f, HR@10: %.4f), Num User: %.0f'
                          % (self.NDCG/self.users, self.HT/self.users, self.users))
                    print('test (NDCG@20: %.4f, HR@20: %.4f), Num User: %.0f'
                          % (self.NDCG_20/self.users, self.HIT_20/self.users, self.users))


    def make_interact_text(self, interact_ids, interact_max_num, user):
        """
                        LLM

        -      Item No.1, Item No.2, ...
        -
        -
        -       [HistoryEmb]

        Args:
            interact_ids (list):          ID
            interact_max_num (int|str):
                -        N
                - 'all'
            user (int):   ID

        Returns:
            Tuple[str, list]:
                - interact_text:
                    : "Item No.1, Time: 2023-01-15, iPhone 13[HistoryEmb],Item No.2, Time: 2023-02-20, MacBook Pro[HistoryEmb]"
                - interact_ids:        ID

            - [HistoryEmb]    token
            -        text_name_dict
        """
        interact_item_titles_ = self.find_item_text(interact_ids, title_flag=True, description_flag=False)

        times = self.find_item_time(interact_ids, user)

        interact_text = []
        count = 1

        if interact_max_num == 'all':
            times = self.find_item_time(interact_ids, user)
        else:
            times = self.find_item_time(interact_ids[-interact_max_num:], user)

        if interact_max_num == 'all':
            for title in interact_item_titles_:
                interact_text.append(f'Item No.{count}, Time: {times[count-1]}, ' + title + '[HistoryEmb]')
                count += 1
        else:
            for title in interact_item_titles_[-interact_max_num:]:
                interact_text.append(f'Item No.{count}, Time: {times[count-1]}, ' + title + '[HistoryEmb]')
                count += 1
            interact_ids = interact_ids[-interact_max_num:]

        interact_text = ','.join(interact_text)
        return interact_text, interact_ids

    
    
    def make_candidate_text(self, interact_ids, candidate_num, target_item_id, target_item_title, candi_set=None, task='ItemTask'):
        """

        Args:
            interact_ids (list):          ID
            candidate_num (int):          1     + N-1
            target_item_id (int):     ID
            target_item_title (str):
            candi_set (set, optional):
            task (str):            'ItemTask'

        Returns:
            Tuple[list, list]:
                - candidate_text:
                - candidate_ids:        ID
        """
        need_neg = candidate_num - 1
        neg_item_id = []

        interact_set = set(interact_ids) if not isinstance(interact_ids, set) else interact_ids
        exclude_set = set(interact_set)

        if candi_set is None:
            while len(neg_item_id) < max(need_neg, 99):
                t = np.random.randint(1, self.item_num + 1)
                if t not in exclude_set:
                    neg_item_id.append(t)
                    exclude_set.add(t)
        else:
            items = list(candi_set - exclude_set)
            if len(items) >= max(need_neg, 99) - len(neg_item_id):
                neg_item_id += random.sample(items, max(need_neg, 99) - len(neg_item_id))
            else:
                neg_item_id += items
                while len(neg_item_id) < max(need_neg, 49):
                    t = np.random.randint(1, self.item_num + 1)
                    if t not in exclude_set:
                        neg_item_id.append(t)
                        exclude_set.add(t)

        random.shuffle(neg_item_id)

        candidate_ids = [target_item_id]

        candidate_text = [self._candidate_prompt(target_item_id)]

        for neg_candidate in neg_item_id[:candidate_num - 1]:
            candidate_text.append(self._candidate_prompt(neg_candidate))
            candidate_ids.append(neg_candidate)

        return candidate_text, candidate_ids
    
    
    def make_candidate(self, interact_ids, candidate_num, target_item_id, target_item_title, candi_set = None, task = 'ItemTask'):
        """
              ID

          make_candidate_text
        - make_candidate_text:        + ID
        - make_candidate:    ID

        Args:
            interact_ids (list):          ID
            candidate_num (int):          1     + N-1
            target_item_id (int):     ID
            target_item_title (str):
            candi_set (set, optional):
            task (str):

        Returns:
            list:     ID
        """
        neg_item_id = []
        neg_item_id = []

        while len(neg_item_id) < 99:
            t = np.random.randint(1, self.item_num + 1)
            if not (t in interact_ids or t in neg_item_id):
                neg_item_id.append(t)

        random.shuffle(neg_item_id)

        candidate_ids = [target_item_id]

        candidate_ids = candidate_ids + neg_item_id[:candidate_num - 1]

        return candidate_ids
    
    
    def pre_train_phase2(self, data, optimizer, batch_iter):
        """
        EchoRec        -

        1.  CF-SRec
        2.               prompt
        3.     token
        4.
        5.

        Args:
            data (tuple):        (u, seq, pos, neg)
                - u:   ID   [batch_size]
                - seq:        [batch_size, max_len]
                - pos:      ID [batch_size, max_len]
                - neg:      ID [batch_size, max_len]
            optimizer: PyTorch
            batch_iter (tuple):        (epoch, total_epoch, step, total_step)

            -      CF-SRec  "  "
            -   LLM     +
            -
        """
        epoch, total_epoch, step, total_step = batch_iter

        primary_rank = (not hasattr(self.args, 'local_rank')) or (self.args.local_rank == 0)
        if primary_rank and getattr(self.args, 'log_interval', 50) > 0 and (step % self.args.log_interval) == 0:
            print(self.args.save_dir, self.args.rec_pre_trained_data, self.args.llm)

        u, seq, pos, neg = data

        original_seq = seq.clone() if torch.is_tensor(seq) else seq.copy()

        def _to_recsys_input(x):
            if torch.is_tensor(x):
                return x.detach().cpu().numpy().astype(np.int64, copy=False)
            return np.asarray(x, dtype=np.int64)

        recsys_u = _to_recsys_input(u)
        recsys_seq = _to_recsys_input(seq)
        recsys_pos = _to_recsys_input(pos)
        recsys_neg = _to_recsys_input(neg)

        mean_loss = 0

        text_input = []
        candidates_pos = []
        candidates_neg = []
        interact_embs = []
        candidate_embs_pos = []
        candidate_embs_neg = []
        candidate_embs = []
        teacher_candidate_embs = []

        loss_rm_mode2 = 0

        with torch.no_grad():
            log_emb = self.recsys.model(recsys_u, recsys_seq, recsys_pos, recsys_neg, mode='log_only')

        history_window = getattr(self.args, 'train_history_window', 10)
        history_window = max(3, min(history_window, 10))
        candidate_num = getattr(self.args, 'train_candidate_num', 4)
        candidate_num = max(2, min(candidate_num, 20))

        batch_size = int(u.size(0)) if torch.is_tensor(u) else len(u)
        for i in range(batch_size):
            target_item_id = self._to_py_int(pos[i][-1])
            target_item_title = self.find_item_text_single(target_item_id, title_flag=True, description_flag=False)

            if torch.is_tensor(seq):
                raw_history_ids = self._to_py_int_list(seq[i][seq[i] > 0])
                user_id = self._to_py_int(u[i])
            else:
                raw_history_ids = seq[i][seq[i]>0]
                user_id = self._to_py_int(u[i])

            interact_text, interact_ids = self.make_interact_text(raw_history_ids, history_window, user_id)

            candidate_text, candidate_ids = self.make_candidate_text(
                raw_history_ids, candidate_num, target_item_id, target_item_title,
                task='RecTask'
            )

            input_text = ''
            input_text += 'This user has made a series of purchases in the following order: '
            input_text += interact_text
            input_text += ". Based on this sequence of purchases, generate user representation token:[UserOut]"

            text_input.append(input_text)
            candidates_pos += candidate_text

            interact_embs.append(self.item_emb_proj((self.get_item_emb(interact_ids))))

            cf_candidate_emb = self.get_item_emb(candidate_ids)
            candidate_embs_pos.append(self.item_emb_proj(cf_candidate_emb))

        candidate_embs = torch.cat(candidate_embs_pos)

        samples = {
            'text_input': text_input,
            'log_emb': log_emb,
            'candidates_pos': candidates_pos,
            'interact': interact_embs,
            'candidate_embs': candidate_embs,
        }

        samples['student_repr'] = log_emb
        samples['num_candidates'] = candidate_num

        loss, llm_rec_loss, match_loss = self.llm(samples, mode=0)

        if primary_rank and getattr(self.args, 'log_interval', 50) > 0 and (step % self.args.log_interval) == 0:
            match_weight = float(getattr(self.args, 'match_weight', 1.0))
            print("rec_loss epoch {}/{} iter {}/{}: {} | match_loss(raw): {} | match_weight: {} | match_loss(eff): {}".format(
                epoch, total_epoch, step, total_step, llm_rec_loss, match_loss, match_weight, match_loss * match_weight))
        
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if self.args.nn_parameter:
            htcore.mark_step()
    
    def split_into_batches(self, itemnum, m):
        """
           ID

        Args:
            itemnum (int):
            m (int):

        Returns:
            list:                 ID
        """
        numbers = list(range(1, itemnum + 1))
        batches = [numbers[i:i + m] for i in range(0, itemnum, m)]
        return batches

    def generate_batch(self, data):
        """
                    -

        1.         LLM
        2.

        -                LLM
        -
        -

        Args:
            data (tuple):      (u, seq, pos, neg, rank, candi_set, files)
                - u:   ID
                - seq:
                - pos:      ID
                - neg:      ID
                - rank:
                - candi_set:
                - files:

        Returns:
            float: NDCG@10
        """
        self._ensure_item_embeddings_ready(desc="Building item embeddings")

        prev_ckpt = getattr(self.llm, "_ckpt_enabled", False)
        if prev_ckpt:
            self.llm._set_gradient_checkpointing(False)
            
        try:
            u, seq, pos, neg, rank, candi_set, files = data
            original_seq = seq.copy()

            text_input = []
            interact_embs = []
            candidate = []

            with torch.no_grad():
                for i in range(len(u)):
                    candidate_embs = []

                    target_item_id = pos[i]
                    target_item_title = self.find_item_text_single(target_item_id, title_flag=True, description_flag=False)

                    items_i = seq[i][seq[i]>0]
                    interact_text, interact_ids = self.make_interact_text(items_i, 10, u[i])

                    candidate_num = 100
                    candidate_ids = self.make_candidate(seq[i][seq[i]>0], candidate_num, target_item_id, target_item_title, candi_set)
                    candidate.append(candidate_ids)

                    input_text = ''
                    input_text += 'This user has made a series of purchases in the following order: '
                    input_text += interact_text
                    input_text += ". Based on this sequence of purchases, generate user representation token:[UserOut]"

                    text_input.append(input_text)

                    interact_embs.append(self.item_emb_proj((self.get_item_emb(interact_ids))))
                    

                max_input_length = self._resolve_positive_int(
                    env_name="LLMSREC_EVAL_MAX_LENGTH",
                    arg_name="eval_max_length",
                    default=getattr(self, 'eval_max_length', 512),
                )
                min_input_length = self._resolve_positive_int(
                    env_name="LLMSREC_EVAL_MIN_LENGTH",
                    arg_name="eval_min_length",
                    default=256,
                )
                min_input_length = min(min_input_length, max_input_length)
                base_chunk_size = max(1, getattr(self, 'inference_chunk_size', 8))
                total_users = len(text_input)
                chunk_start = 0
                adaptive_chunk_size = base_chunk_size
                oom_notice_printed = False

                while chunk_start < total_users:
                    current_chunk = min(adaptive_chunk_size, total_users - chunk_start)
                    current_max_len = max_input_length

                    while True:
                        end_idx = chunk_start + current_chunk
                        chunk_text = text_input[chunk_start:end_idx]
                        chunk_interact = interact_embs[chunk_start:end_idx]
                        chunk_candidates = candidate[chunk_start:end_idx]

                        try:
                            llm_tokens = self.llm.llm_tokenizer(
                                chunk_text,
                                return_tensors="pt",
                                padding="longest",
                                truncation=True,
                                max_length=current_max_len,
                            ).to(self.device)

                            inputs_embeds = self.llm.llm_model.get_input_embeddings()(llm_tokens['input_ids'])
                            inputs_embeds = self.llm.replace_out_token_all(
                                llm_tokens,
                                inputs_embeds,
                                token=['[UserOut]', '[HistoryEmb]'],
                                embs={'[HistoryEmb]': chunk_interact}
                            )

                            torch.cuda.empty_cache()

                            with torch.amp.autocast('cuda'):
                                with torch.no_grad():
                                    outputs = self.llm.llm_model.model(
                                        inputs_embeds=inputs_embeds,
                                        output_hidden_states=True
                                    )

                                    indx = self.llm.get_embeddings(llm_tokens, '[UserOut]')
                                    user_outputs = torch.cat([
                                        outputs.hidden_states[-1][i, indx[i]].mean(axis=0).unsqueeze(0)
                                        for i in range(len(indx))
                                    ])
                                    user_outputs = self.llm.pred_user(user_outputs)

                                    del outputs
                                    del inputs_embeds
                                    torch.cuda.empty_cache()

                                for local_idx, global_idx in enumerate(range(chunk_start, end_idx)):
                                    item_outputs = self.all_embs[np.array(chunk_candidates[local_idx]) - 1]
                                    if item_outputs.device != self.device:
                                        item_outputs = item_outputs.to(self.device, non_blocking=True)
                                    logits = torch.mm(item_outputs, user_outputs[local_idx].unsqueeze(0).T).squeeze(-1)
                                    logits = -1 * logits
                                    rank = logits.argsort().argsort()[0].item()

                                    if rank < 10:
                                        self.NDCG += 1 / np.log2(rank + 2)
                                        self.HT += 1
                                    if rank < 20:
                                        self.NDCG_20 += 1 / np.log2(rank + 2)
                                        self.HIT_20 += 1
                                    self.users += 1

                            chunk_start = end_idx
                            adaptive_chunk_size = current_chunk
                            self.inference_chunk_size = current_chunk
                            max_input_length = current_max_len
                            self.eval_max_length = current_max_len
                            break

                        except RuntimeError as e:
                            if 'CUDA out of memory' in str(e):
                                torch.cuda.empty_cache()
                                if current_chunk > 1:
                                    current_chunk = max(1, current_chunk // 2)
                                    if not oom_notice_printed:
                                        pass
                                        oom_notice_printed = True
                                    continue
                                elif current_max_len > min_input_length:
                                    current_max_len = max(min_input_length, current_max_len - 64)
                                    if not oom_notice_printed:
                                        pass
                                        oom_notice_printed = True
                                    continue
                            raise

            return self.NDCG
        finally:
            if prev_ckpt:
                self.llm._set_gradient_checkpointing(True)

EchoRecModel = EchoRecSIModel

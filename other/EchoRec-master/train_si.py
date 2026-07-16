import os
import torch
import random
import time
import os
import sys
import tempfile
import shutil
import glob
import copy
import atexit
from datetime import timedelta
from contextlib import nullcontext

if 'TOKENIZERS_PARALLELISM' not in os.environ:
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'

from tqdm import tqdm

import torch.multiprocessing as mp
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
from torch.optim.lr_scheduler import LambdaLR
from torch.cuda.amp import GradScaler, autocast

try:
    import habana_frameworks.torch.core as htcore
except ImportError:
    htcore = None

from models.echorec_si import EchoRecSIModel
try:
    from SeqRec.bert4rec.utils import data_partition, SeqDataset, SeqDataset_Inference, SeqDataset_Validation
except ImportError:
    def data_partition(*args, **kwargs):
        pass
    class SeqDataset:
        def __init__(self, *args, **kwargs):
            pass
    class SeqDataset_Inference:
        def __init__(self, *args, **kwargs):
            pass
    class SeqDataset_Validation:
        def __init__(self, *args, **kwargs):
            pass
from utils import create_dir


def _dist_barrier(rank):
    if not torch.distributed.is_initialized():
        return
    if torch.cuda.is_available():
        torch.distributed.barrier(device_ids=[rank])
    else:
        torch.distributed.barrier()

def setup_ddp(rank, world_size, args):
    if torch.distributed.is_initialized():
        try:
            existing_rank = torch.distributed.get_rank()
            existing_ws = torch.distributed.get_world_size()
            pass
        except Exception:
            pass
        destroy_process_group()

    if 'MASTER_ADDR' not in os.environ:
        os.environ['MASTER_ADDR'] = 'localhost'
    if 'MASTER_PORT' not in os.environ:
        os.environ['MASTER_PORT'] = '12355'
    os.environ["ID"] = str(rank)

    os.environ['NCCL_DEBUG'] = 'WARN'
    os.environ['NCCL_TREE_THRESHOLD'] = '0'
    os.environ['NCCL_IB_DISABLE'] = '1'
    os.environ['NCCL_SOCKET_IFNAME'] = 'lo'
    os.environ['NCCL_TIMEOUT'] = '7200'
    os.environ['NCCL_BLOCKING_WAIT'] = '0'
    os.environ['NCCL_ASYNC_ERROR_HANDLING'] = '1'

    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:1024'
    os.environ['CUDA_LAUNCH_BLOCKING'] = '0'

    try:
        pass
        torch.distributed.init_process_group(
            backend='nccl' if torch.cuda.is_available() else 'gloo',
            rank=rank,
            world_size=world_size,
            timeout=timedelta(minutes=120)
        )
        pass
    except Exception as e:
        pass
        raise

def optimize_memory_usage(rank=0, verbose=True):
    """
                                  ENABLE_MEM_PREALLOC=1
    """
    if torch.cuda.is_available():
        prealloc = str(os.getenv('ENABLE_MEM_PREALLOC', '0')).lower() in ('1', 'true', 'yes')
        if prealloc:
            dummy = torch.zeros((1024, 1024, 800), device='cuda')
            del dummy
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

        if rank == 0 and verbose:
            memory_allocated = torch.cuda.memory_allocated() / 1024**3
            memory_reserved = torch.cuda.memory_reserved() / 1024**3
            tag = "preallocated" if prealloc else "not preallocated"
            pass

def cleanup_temp_files():
    """Internal helper."""
    try:
        temp_patterns = [
            'pymp-*',
            'tmp*',
            'pytorch-errorfile-*.pickle'
        ]

        for pattern in temp_patterns:
            for file_path in glob.glob(pattern):
                try:
                    if os.path.isfile(file_path):
                        os.remove(file_path)
                        pass
                    elif os.path.isdir(file_path):
                        shutil.rmtree(file_path)
                        pass
                except Exception as e:
                    pass

        temp_dir = tempfile.gettempdir()
        for pattern in ['pymp-*', 'pytorch-*']:
            for file_path in glob.glob(os.path.join(temp_dir, pattern)):
                try:
                    if os.path.isfile(file_path):
                        os.remove(file_path)
                    elif os.path.isdir(file_path):
                        shutil.rmtree(file_path)
                except:
                    pass

    except Exception as e:
        pass

def reset_best_model_dir(args):
    """Internal helper."""
    base_dir = os.path.join('./models', args.rec_pre_trained_data, args.save_dir.rstrip('/'), 'best')
    if not os.path.isdir(base_dir):
        os.makedirs(base_dir, exist_ok=True)
        return

    for entry in os.listdir(base_dir):
        if entry.endswith('_results.txt'):
            continue
        path = os.path.join(base_dir, entry)
        try:
            if os.path.isdir(path):
                shutil.rmtree(path, ignore_errors=True)
            else:
                os.remove(path)
        except OSError:
            pass


def train_si(args):
    print('EchoRec SI training\n')

    cleanup_temp_files()

    atexit.register(cleanup_temp_files)

    if args.multi_gpu and 'RANK' in os.environ:
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        pass
        _run_train_si(rank, world_size, args)
    elif args.multi_gpu:
        world_size = args.world_size
        try:
            mp.spawn(_run_train_si,
                 args=(world_size,args),
                 nprocs=world_size,
                 join=True)
        finally:
            cleanup_temp_files()
    else:
        _run_train_si(0, 0, args)

def _run_train_si(rank,world_size,args):
    args.rank = rank
    args.local_rank = rank
    if args.multi_gpu:
        setup_ddp(rank, world_size, args)
        if args.device == 'hpu':
            args.device = torch.device('hpu')
        else:
            args.device = 'cuda:' + str(rank)

    args.use_amp = getattr(args, 'use_amp', True)
    amp_enabled = bool(args.use_amp and torch.cuda.is_available())
    amp_dtype = torch.float16 if torch.cuda.is_available() else None
    if amp_enabled and rank == 0:
        pass
    shared_scaler = GradScaler(enabled=amp_enabled)
    _run_train_si._shared_scaler = shared_scaler

    if torch.cuda.is_available():
        gpu_total_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        if gpu_total_gb >= 30:
            mem_fraction = 0.95
        elif gpu_total_gb >= 22:
            mem_fraction = 0.92
        else:
            mem_fraction = 0.90
        mem_fraction = min(mem_fraction, 0.92)
        torch.cuda.set_per_process_memory_fraction(mem_fraction)

        allowed_gb = gpu_total_gb * mem_fraction
        if args.multi_gpu:
            if rank == 0:
                pass
        else:
            if rank == 0:
                pass

    if hasattr(args, 'seed') and args.seed is not None and not getattr(args, 'no_seed', False):
        import numpy as np
        torch.manual_seed(args.seed + rank)
        torch.cuda.manual_seed(args.seed + rank)
        np.random.seed(args.seed + rank)
        random.seed(args.seed + rank)
        if rank == 0:
            pass


    if args.multi_gpu and torch.cuda.is_available():
        torch.cuda.synchronize()
        if rank == 0:
            pass

    dataset = data_partition(
        args.rec_pre_trained_data,
        args,
        path=f'./SeqRec/data_{args.rec_pre_trained_data}/{args.rec_pre_trained_data}',
    )
    [user_train, user_valid, user_test, usernum, itemnum, eval_set] = dataset

    model = EchoRecSIModel(args).to(args.device)

    if args.multi_gpu:
        if not torch.distributed.is_initialized():
            pass
            raise RuntimeError("         ")

        try:
            torch.cuda.synchronize()

            model = DDP(
                model,
                device_ids=[rank],
                output_device=rank,
                static_graph=True,
                broadcast_buffers=False,
            )
            if rank == 0:
                pass

        except RuntimeError as e:
            if rank == 0:
                pass
            raise e

        scaler = GradScaler()
        del scaler

        if args.multi_gpu:
            _dist_barrier(rank)

            _dist_barrier(rank)
    core_model = model.module if isinstance(model, DDP) else model
    core_model._shared_scaler = shared_scaler

    def lr_lambda(epoch):
        if epoch <= 5:
            return 1.0
        elif epoch <= 10:
            return 0.98 ** (epoch - 5)
        else:
            return 0.95 ** (epoch - 10) * 0.98 ** 5

    optimizer_param_signature = None
    adam_optimizer = None
    scheduler = None

    def _collect_trainable_params():
        params = [p for p in model.parameters() if p.requires_grad]
        if not params:
            return list(model.parameters())
        return params

    def rebuild_optimizer(reason='initial', epoch_marker=0, force=False):
        nonlocal adam_optimizer, scheduler, optimizer_param_signature
        params = _collect_trainable_params()
        param_ids = tuple(id(p) for p in params)
        flag = getattr(core_model, 'optimizer_needs_reset', False)
        needs_reset = force or flag or optimizer_param_signature != param_ids or adam_optimizer is None
        if not needs_reset:
            return
        
        old_optimizer_existed = adam_optimizer is not None
        if old_optimizer_existed:
            del adam_optimizer
        if scheduler is not None:
            del scheduler
        
        adam_optimizer = torch.optim.Adam(params, lr=args.stage2_lr, betas=(0.9, 0.98))
        
        for group in adam_optimizer.param_groups:
            group.setdefault('initial_lr', args.stage2_lr)
        
        sched_last_epoch = -1 if epoch_marker <= 0 else epoch_marker - 1
        
        scheduler = LambdaLR(adam_optimizer, lr_lambda=lr_lambda, last_epoch=sched_last_epoch)
        
        optimizer_param_signature = param_ids
        if hasattr(core_model, 'optimizer_needs_reset'):
            core_model.optimizer_needs_reset = False
        if rank == 0:
            mem_after = torch.cuda.memory_allocated() / 1024**3 if torch.cuda.is_available() else 0
            current_lr = adam_optimizer.param_groups[0]['lr']
            pass
    epoch_start_idx = 1
    total_epoch_cap = args.num_epochs
    
    T = 0.0
    perform = 0

    best_perform = 0.0
    early_stop = 0
    early_thres = max(1, int(getattr(args, 'early_stop_patience', 4)))
    min_epochs_before_early_stop = max(1, int(getattr(args, 'min_epochs_before_early_stop', 12)))
    t0 = time.time()
    run_eval_on_rank = rank == 0
    train_data_set = None
    train_data_loader = None
    inference_data_loader = None
    current_num_batches = 0

    def ensure_phase_dataloader(announce=False):
        nonlocal train_data_set, train_data_loader, inference_data_loader, current_num_batches
        if train_data_loader is not None:
            return

        train_data_set = SeqDataset(user_train, usernum, itemnum, args.maxlen)
        if args.multi_gpu:
            train_sampler = DistributedSampler(
                train_data_set,
                shuffle=True,
                rank=rank,
                num_replicas=world_size,
                seed=getattr(args, "seed", 42),
                drop_last=False,
            )
            train_data_loader = DataLoader(
                train_data_set,
                batch_size=args.batch_size,
                sampler=train_sampler,
                pin_memory=True,
                num_workers=getattr(args, "train_num_workers", 0),
            )
        else:
            train_data_loader = DataLoader(
                train_data_set,
                batch_size=args.batch_size,
                shuffle=True,
                pin_memory=True,
                num_workers=getattr(args, "train_num_workers", 0),
            )
        current_num_batches = len(train_data_loader)

        eval_set_use = eval_set[1]
        if len(eval_set_use) > 10000:
            users = random.sample(list(eval_set_use), 10000)
        else:
            users = list(eval_set_use)

        user_list = []
        for u in users:
            if len(user_test[u]) < 1:
                continue
            user_list.append(u)

        inference_data_set = SeqDataset_Inference(user_train, user_valid, user_test, user_list, itemnum, args.maxlen)
        if args.multi_gpu:
            inference_data_loader = DataLoader(
                inference_data_set,
                batch_size=args.batch_size_infer,
                sampler=DistributedSampler(inference_data_set, shuffle=True, rank=rank, num_replicas=world_size),
                pin_memory=True,
                num_workers=0,
            )
        else:
            inference_data_loader = DataLoader(
                inference_data_set,
                batch_size=args.batch_size_infer,
                pin_memory=True,
                num_workers=0,
            )

        if announce and rank == 0:
            print(
                f"SI dataloader ready: users={len(train_data_set)}, "
                f"batches/rank={current_num_batches}, eval_on_rank0={run_eval_on_rank}"
            )

    base_phase_weights = {
        'teacher_rec': getattr(args, 'teacher_rec_weight', 1.0),
        'forward': getattr(args, 'forward_weight', getattr(args, 'forward_kd_weight', 1.0)),
    }
    if hasattr(core_model, 'configure_training_phase'):
        core_model.configure_training_phase(
            'sequence_injection',
            {
                'teacher_rec': base_phase_weights['teacher_rec'],
                'forward': base_phase_weights['forward'],
            },
            verbose=(rank == 0),
            round_idx=1,
        )

    ensure_phase_dataloader(announce=True)
    rebuild_optimizer(reason='initial', epoch_marker=0, force=True)

    def run_validation_and_maybe_test(epoch, trigger='epoch-end', step=None):
        nonlocal best_perform, early_stop

        if not run_eval_on_rank:
            return None

        model.eval()

        eval_set_use = eval_set[0]
        if len(eval_set_use) > 10000:
            users = random.sample(list(eval_set_use), 10000)
        else:
            users = list(eval_set_use)

        user_list_valid = [u for u in users if len(user_valid[u]) >= 1]

        if len(user_list_valid) == 0:
            pass
            return None

        valid_data_set = SeqDataset_Validation(user_train, user_valid, user_list_valid, itemnum, args.maxlen)
        if args.multi_gpu:
            valid_data_loader = DataLoader(
                valid_data_set,
                batch_size=args.batch_size_infer,
                sampler=DistributedSampler(valid_data_set, shuffle=True, rank=rank, num_replicas=world_size),
                pin_memory=True,
                num_workers=0,
            )
        else:
            valid_data_loader = DataLoader(
                valid_data_set,
                batch_size=args.batch_size_infer,
                pin_memory=True,
                shuffle=True,
                num_workers=0,
            )

        core_model_local = model.module if args.multi_gpu else model
        core_model_local.users = 0.0
        core_model_local.NDCG = 0.0
        core_model_local.HT = 0.0
        core_model_local.NDCG_20 = 0.0
        core_model_local.HIT_20 = 0.0
        core_model_local.all_embs = None

        with torch.no_grad():
            for batch_idx, data in enumerate(valid_data_loader):
                if batch_idx == 0 and rank == 0:
                    tag = f"Validation ({trigger}, epoch {epoch}"
                    if step is not None:
                        tag += f", step {step}"
                    tag += f") early stop: {early_stop}"
                    print(tag)
                u, seq, pos, neg = data
                u, seq, pos, neg = u.numpy(), seq.numpy(), pos.numpy(), neg.numpy()
                
                core_model_local([u, seq, pos, neg, rank, None, 'original'], mode='generate_batch')

        valid_users = core_model_local.users
        perform = core_model_local.HT / valid_users if valid_users > 0 else 0.0
        ndcg_perform = core_model_local.NDCG / valid_users if valid_users > 0 else 0.0
        llm_hr20 = core_model_local.HIT_20 / valid_users if valid_users > 0 else 0.0
        llm_ndcg20 = core_model_local.NDCG_20 / valid_users if valid_users > 0 else 0.0

        if rank == 0:
            print(f"[{trigger}] Valid Set - LLM HR@10 = {perform:.4f}, NDCG@10 = {ndcg_perform:.4f}")

        apply_early_stop = True
        is_new_best = perform >= best_perform

        den = 1.0
        test_llm_hr10 = test_llm_ndcg10 = test_llm_hr20 = test_llm_ndcg20 = 0.0
        if inference_data_loader is not None:
            core_model_local.users = 0.0
            core_model_local.NDCG = 0.0
            core_model_local.HT = 0.0
            core_model_local.NDCG_20 = 0.0
            core_model_local.HIT_20 = 0.0
            core_model_local.all_embs = None

            with torch.no_grad():
                print("Testing")
                for batch_idx, data in enumerate(inference_data_loader):
                    if batch_idx == 0 and rank == 0:
                        pass
                    u, seq, pos, neg = data
                    u, seq, pos, neg = u.numpy(), seq.numpy(), pos.numpy(), neg.numpy()
                    
                    core_model_local([u, seq, pos, neg, rank, None, 'original'], mode='generate_batch')

            den = core_model_local.users if core_model_local.users > 0 else 1.0
            test_llm_hr10 = core_model_local.HT / den
            test_llm_ndcg10 = core_model_local.NDCG / den
            test_llm_hr20 = core_model_local.HIT_20 / den
            test_llm_ndcg20 = core_model_local.NDCG_20 / den

            if rank == 0:

                out_dir = os.path.join('./models', args.rec_pre_trained_data, args.save_dir.rstrip('/'))
                create_dir(out_dir)
                all_metrics_file = os.path.join(out_dir, f"{args.rec_pre_trained_data}_{args.llm}_all_results.txt")
                status_label = "NEW BEST" if is_new_best else "EPOCH RESULT"
                with open(all_metrics_file, 'a') as f:
                    f.write(f"========== Epoch {epoch} ({trigger}) - {status_label} ==========\n")
                    f.write("Test Set:\n")
                    f.write(f"  LLM    - HR@10: {test_llm_hr10:.16f}, NDCG@10: {test_llm_ndcg10:.16f}, HR@20: {test_llm_hr20:.16f}, NDCG@20: {test_llm_ndcg20:.16f}\n")
                    f.write("\n")

                best_dir = os.path.join('./models', args.rec_pre_trained_data, args.save_dir.rstrip('/'), 'best')
                create_dir(best_dir)
                result_file = os.path.join(
                    best_dir,
                    f"{args.rec_pre_trained_data}_{args.llm}_{epoch}_results.txt"
                )
                with open(result_file, 'a') as f:
                    f.write(f"LLM NDCG@10: {test_llm_ndcg10:.16f}, LLM HR@10: {test_llm_hr10:.16f}\n")
                    f.write(f"LLM NDCG@20: {test_llm_ndcg20:.16f}, LLM HR@20: {test_llm_hr20:.16f}\n")

        metrics_bundle = {
            'llm': {
                'hr10': perform,
                'ndcg10': ndcg_perform,
                'hr20': llm_hr20,
                'ndcg20': llm_ndcg20,
            },
            'test_llm': {
                'hr10': test_llm_hr10,
                'ndcg10': test_llm_ndcg10,
                'hr20': test_llm_hr20,
                'ndcg20': test_llm_ndcg20,
            },
        }

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        if rank == 0 and inference_data_loader is not None:
            print(f"\n[{trigger}] Test Set Results:")
            print(f"  LLM:    HR@10={test_llm_hr10:.4f}, NDCG@10={test_llm_ndcg10:.4f}")

        if is_new_best:
            best_perform = perform
            if not getattr(args, 'disable_model_saving', False):
                if rank == 0:
                    pass
                    try:
                        save_target = model.module if args.multi_gpu else model
                        reset_best_model_dir(args)
                        save_target.save_model(args, epoch2=epoch, best=True)
                        pass
                    except Exception as e:
                        pass
                        import traceback
                        traceback.print_exc()

            if apply_early_stop:
                early_stop = 0
            if rank == 0:
                pass
        else:
            if apply_early_stop:
                early_stop += 1
                pass

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        
        return metrics_bundle

    def guarded_validation_call(epoch, trigger='epoch-end', step=None):
        """Internal helper."""
        metrics = run_validation_and_maybe_test(epoch, trigger=trigger, step=step)
        return metrics

    for epoch in tqdm(range(epoch_start_idx, total_epoch_cap + 1)):
        if torch.cuda.is_available():
            import gc
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            if args.multi_gpu:
                _dist_barrier(rank)
        
        rebuild_optimizer(reason='auto-check', epoch_marker=max(0, epoch - 1))
        model.train()
        if args.multi_gpu:
            train_data_loader.sampler.set_epoch(epoch)
            _dist_barrier(rank)

        if hasattr(core_model, 'update_adaptive_weights'):
            core_model.update_adaptive_weights(epoch)

        if hasattr(core_model, 'current_epoch'):
            core_model.current_epoch = epoch


        epoch_start_time = time.time()
        epoch_loss = 0.0
        step_count = 0

        stage_epoch_idx = epoch
        stage_epoch_total = args.num_epochs

        optimize_memory_usage(rank, verbose=True)
        if rank == 0 and epoch % 3 == 1:
            pass

        if rank == 0:
            pass

        train_iter = train_data_loader
        if rank == 0:
            train_iter = tqdm(
                train_data_loader,
                total=current_num_batches,
                desc=f"Epoch {epoch}/{total_epoch_cap}",
                leave=False,
                dynamic_ncols=True,
            )
        for step, data in enumerate(train_iter):
            u, seq, pos, neg = data
            batch_payload = [u.numpy(), seq.numpy(), pos.numpy(), neg.numpy()]

            loss_tensor = model(
                batch_payload,
                optimizer=adam_optimizer,
                batch_iter=[stage_epoch_idx, stage_epoch_total, step, current_num_batches],
                mode='phase2',
            )

            if getattr(args, 'nn_parameter', False) and 'htcore' in globals() and htcore is not None:
                htcore.mark_step()

            if loss_tensor is None:
                loss_scalar = 0.0
            elif isinstance(loss_tensor, torch.Tensor):
                loss_scalar = loss_tensor.detach().item()
            else:
                loss_scalar = float(loss_tensor)
            epoch_loss += loss_scalar

            step_count += 1


            if rank == 0:
                current_lr = adam_optimizer.param_groups[0]['lr']
                avg_loss = epoch_loss / step_count if step_count > 0 else 0
                train_iter.set_postfix(
                    loss=f"{avg_loss:.4f}",
                    lr=f"{current_lr:.6f}",
                    step=f"{step + 1}/{current_num_batches}",
                )



        if rank == 0:
            train_iter.close()
            epoch_time = time.time() - epoch_start_time
            samples_per_sec = len(train_data_set) / epoch_time
            gpu_info = f"{world_size}GPU" if args.multi_gpu else "1GPU"
            pass


        epoch_metrics = guarded_validation_call(epoch, trigger='epoch-end')
        model.train()

        stop_training = False
        if rank == 0:
            if early_stop >= early_thres and epoch >= min_epochs_before_early_stop:
                pass
                pass
                stop_training = True
            elif early_stop >= early_thres:
                pass

        if args.multi_gpu:
            stop_tensor = torch.zeros(1, device=args.device)
            if rank == 0:
                stop_tensor.fill_(1 if stop_training else 0)
            torch.distributed.broadcast(stop_tensor, src=0)
            stop_training = bool(stop_tensor.item())
            _dist_barrier(rank)

        if stop_training:
            break

        scheduler.step()

        model.train()


    if args.multi_gpu:
        _dist_barrier(rank)
        if rank == 0:
            pass

    if rank == 0:
        pass

    if args.multi_gpu:
        destroy_process_group()

    cleanup_temp_files()
    return

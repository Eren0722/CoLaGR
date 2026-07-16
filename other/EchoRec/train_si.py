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

# 禁用HuggingFace tokenizer的多进程并行警告（可通过环境变量覆写）
if 'TOKENIZERS_PARALLELISM' not in os.environ:
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'

from tqdm import tqdm

import torch.multiprocessing as mp
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
from torch.optim.lr_scheduler import LambdaLR
from torch.cuda.amp import GradScaler, autocast  # 混合精度训练

try:
    import habana_frameworks.torch.core as htcore
except ImportError:
    htcore = None

from models.echorec_si import EchoRecSIModel
try:
    from SeqRec.sasrec.utils import data_partition, SeqDataset, SeqDataset_Inference, SeqDataset_Validation
except ImportError:
    # 如果导入失败，使用占位符函数
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
    # 🔧 若外部已初始化，先销毁再按自定义超时重建，避免默认600s超时
    if torch.distributed.is_initialized():
        try:
            existing_rank = torch.distributed.get_rank()
            existing_ws = torch.distributed.get_world_size()
            print(f"⚙️ 重新初始化DDP (先销毁已有进程组) rank={existing_rank}, world_size={existing_ws}")
        except Exception:
            pass
        destroy_process_group()

    # 设置DDP环境变量（如果torchrun没有设置）
    if 'MASTER_ADDR' not in os.environ:
        os.environ['MASTER_ADDR'] = 'localhost'
    if 'MASTER_PORT' not in os.environ:
        os.environ['MASTER_PORT'] = '12355'
    os.environ["ID"] = str(rank)

    # 优化NCCL设置
    os.environ['NCCL_DEBUG'] = 'WARN'
    os.environ['NCCL_TREE_THRESHOLD'] = '0'
    os.environ['NCCL_IB_DISABLE'] = '1'  # 禁用InfiniBand，使用以太网
    os.environ['NCCL_SOCKET_IFNAME'] = 'lo'  # 使用本地回环接口
    # 🚦 增大NCCL超时，避免长时间资产生成/阶段切换时触发watchdog
    os.environ['NCCL_TIMEOUT'] = '7200'  # 7200秒=2小时
    os.environ['NCCL_BLOCKING_WAIT'] = '0'  # 🔧 禁用阻塞等待，避免watchdog超时
    os.environ['NCCL_ASYNC_ERROR_HANDLING'] = '1'  # 🔧 启用异步错误处理

    # 🚀 双卡优化：1GB大分块，减少碎片化
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:1024'  # 🔧 OOM修复：移除expandable_segments:True，恢复Start_old成功配置
    os.environ['CUDA_LAUNCH_BLOCKING'] = '0'  # 异步执行

    # ✅ 关键修复：实际初始化分布式进程组
    try:
        print(f"🚀 初始化DDP进程组: rank={rank}, world_size={world_size}")
        torch.distributed.init_process_group(
            backend='nccl' if torch.cuda.is_available() else 'gloo',
            rank=rank,
            world_size=world_size,
            timeout=timedelta(minutes=120)  # 🔧 增加到120分钟以支持长时间验证/资产生成
        )
        print(f"✅ DDP初始化成功: rank={rank}, 超时设置: 120分钟")
    except Exception as e:
        print(f"❌ DDP初始化失败: {e}")
        raise

def optimize_memory_usage(rank=0, verbose=True):
    """
    内存使用优化函数（可选预分配）。
    默认关闭预分配，避免额外占用；如需恢复旧行为，设置环境变量 ENABLE_MEM_PREALLOC=1。
    """
    if torch.cuda.is_available():
        prealloc = str(os.getenv('ENABLE_MEM_PREALLOC', '0')).lower() in ('1', 'true', 'yes')
        if prealloc:
            # ✅ 可选：预分配显存触发分配器预留更多空间（~3GB）
            dummy = torch.zeros((1024, 1024, 800), device='cuda')  # ~3GB临时分配
            del dummy
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

        if rank == 0 and verbose:
            memory_allocated = torch.cuda.memory_allocated() / 1024**3
            memory_reserved = torch.cuda.memory_reserved() / 1024**3
            tag = "（预分配已启用）" if prealloc else "(未预分配)"
            print(f"🔧 内存优化后: 已分配 {memory_allocated:.2f}GB, 已保留 {memory_reserved:.2f}GB {tag}")

def cleanup_temp_files():
    """清理PyTorch multiprocessing产生的临时文件"""
    try:
        # 清理当前目录下的临时文件
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
                        print(f"🧹 清理临时文件: {file_path}")
                    elif os.path.isdir(file_path):
                        shutil.rmtree(file_path)
                        print(f"🧹 清理临时目录: {file_path}")
                except Exception as e:
                    print(f"⚠️ 清理 {file_path} 失败: {e}")

        # 清理系统临时目录中的相关文件
        temp_dir = tempfile.gettempdir()
        for pattern in ['pymp-*', 'pytorch-*']:
            for file_path in glob.glob(os.path.join(temp_dir, pattern)):
                try:
                    if os.path.isfile(file_path):
                        os.remove(file_path)
                    elif os.path.isdir(file_path):
                        shutil.rmtree(file_path)
                except:
                    pass  # 系统临时文件可能被其他进程使用，忽略错误

    except Exception as e:
        print(f"⚠️ 临时文件清理过程中出现错误: {e}")

def reset_best_model_dir(args):
    """在刷新最佳模型前清理 best 目录，但保留 *_results.txt 历史记录。"""
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


class WeightingScheduler:
    """Adaptive scheduler for tuning the distillation weight (β) over epochs."""

    def __init__(self, start_beta: float = 0.1, max_beta: float = 0.5, warmup_epochs: int = 5):
        self.start_beta = float(start_beta)
        self.max_beta = float(max_beta)
        self.warmup_epochs = max(1, int(warmup_epochs))
        self.epoch = 0
        self.beta = self.start_beta

    def step(self):
        self.epoch += 1
        progress = min(1.0, self.epoch / self.warmup_epochs)
        self.beta = self.start_beta + (self.max_beta - self.start_beta) * progress

    def get_beta(self) -> float:
        return self.beta


def build_weighting_scheduler(args):
    if not getattr(args, 'enable_bidirectional_kd', False):
        return None
    start_beta = max(0.0, getattr(args, 'weighting_start_beta', 0.1))
    max_beta = max(start_beta, getattr(args, 'weighting_max_beta', 0.5))
    warmup_epochs = max(1, getattr(args, 'weighting_warmup_epochs', 5))
    return WeightingScheduler(start_beta=start_beta, max_beta=max_beta, warmup_epochs=warmup_epochs)


def train_si(args):
    print('EchoRec SI training\n')

    # 训练前清理临时文件
    cleanup_temp_files()

    # 注册退出时清理函数
    atexit.register(cleanup_temp_files)

    # 🔧 修复：检测是否已经在torchrun环境中
    if args.multi_gpu and 'RANK' in os.environ:
        # 已经在torchrun进程中，直接调用训练函数
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        print(f"🚀 使用torchrun进程: rank={rank}, world_size={world_size}")
        _run_train_si(rank, world_size, args)
    elif args.multi_gpu:
        # 传统mp.spawn方式（备用）
        world_size = args.world_size
        try:
            mp.spawn(_run_train_si,
                 args=(world_size,args),
                 nprocs=world_size,
                 join=True)
        finally:
            # 训练完成后立即清理
            cleanup_temp_files()
    else:
        _run_train_si(0, 0, args)

def inference(args):
    print('EchoRec inference\n')

    # 推理前清理临时文件
    cleanup_temp_files()

    if args.multi_gpu:
        world_size = args.world_size
        try:
            mp.spawn(inference_,
                 args=(world_size,args),
                 nprocs=world_size,
                 join=True)
        finally:
            # 推理完成后立即清理
            cleanup_temp_files()
    else:
        inference_(0,0,args)
  

def _run_train_si(rank,world_size,args):
    if args.multi_gpu:
        setup_ddp(rank, world_size, args)
        if args.device == 'hpu':
            args.device = torch.device('hpu')
        else:
            args.device = 'cuda:' + str(rank)

    # 默认开启混合精度；可通过 --use_amp False 关闭
    args.use_amp = getattr(args, 'use_amp', True)
    amp_enabled = bool(args.use_amp and torch.cuda.is_available())
    amp_dtype = torch.float16 if torch.cuda.is_available() else None
    if amp_enabled and rank == 0:
        print("✅ 启用混合精度训练 (autocast, dtype=float16)")
    shared_scaler = GradScaler(enabled=amp_enabled)
    _run_train_si._shared_scaler = shared_scaler

    #  多GPU CUDA内存管理优化 - 自动检测GPU显存大小
    if torch.cuda.is_available():
        gpu_total_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        if gpu_total_gb >= 30:
            # RTX 5090D / A100 等 ≥32GB 显卡：95% 限制，留 ~1.5GB 系统开销
            mem_fraction = 0.95
        elif gpu_total_gb >= 22:
            # RTX 3090 / RTX 4090 等 24GB 显卡：92% 限制，留 ~2GB 系统开销
            mem_fraction = 0.92
        else:
            # 更小的显卡：90% 限制
            mem_fraction = 0.90
        # 固定回 0.92（24GB 卡）或更低，确保 >2GB 余量
        mem_fraction = min(mem_fraction, 0.92)
        torch.cuda.set_per_process_memory_fraction(mem_fraction)

        allowed_gb = gpu_total_gb * mem_fraction
        if args.multi_gpu:
            if rank == 0:
                print(f"🚀 双卡batch_size={args.batch_size}模式: {world_size}×{gpu_total_gb:.1f}GB GPU，{mem_fraction*100:.0f}%内存限制({allowed_gb:.1f}GB)，1GB大分块")
        else:
            if rank == 0:
                print(f"🚀 单GPU模式: {gpu_total_gb:.1f}GB GPU，{mem_fraction*100:.0f}%内存限制({allowed_gb:.1f}GB)，1GB大分块")

    # 🔧 确保多GPU训练时每个进程使用相同的种子
    # 种子已在main.py中设置，但为了确保分布式训练的一致性，这里再次设置
    if hasattr(args, 'seed') and args.seed is not None and not getattr(args, 'no_seed', False):
        import numpy as np
        torch.manual_seed(args.seed + rank)  # 每个rank使用不同的种子，避免完全相同
        torch.cuda.manual_seed(args.seed + rank)
        np.random.seed(args.seed + rank)
        random.seed(args.seed + rank)
        if rank == 0:
            print(f"🎲 Rank {rank}: 随机种子设置为 {args.seed + rank}")

    # 🎯 高效的10%性能提升配置检查
    if hasattr(args, 'efficient_10percent_boost') and args.efficient_10percent_boost:
        print("🔧 高效的10%性能提升已在main.py中应用")
        print("   🎯 目标: NDCG@10从0.3661提升到0.4027+ (10%+)")
        print("   🎯 策略1: 精准训练优化（30轮，batch_size=20）")
        print("   🎯 策略2: 强化知识蒸馏（温度2.0，强传递）")
        print("   🎯 策略3: 精选损失优化（ranking+一致性）")
        print("   🎯 策略4: 轻量级架构增强（特征对齐）")

    # 🔧 DDP初始化前的内存清理 - 移除缓存清理避免CUDA错误
    if args.multi_gpu and torch.cuda.is_available():
        # torch.cuda.empty_cache()  # 移除以避免CUDA非法内存访问
        torch.cuda.synchronize()
        if rank == 0:
            print("🔧 DDP初始化前内存清理完成")

    dataset = data_partition(args.rec_pre_trained_data, args, path=f'./SeqRec/data_{args.rec_pre_trained_data}/{args.rec_pre_trained_data}')
    [user_train, user_valid, user_test, usernum, itemnum, eval_set] = dataset

    model = EchoRecSIModel(args).to(args.device)

    preload_epoch = getattr(args, 'preload_epoch', None)
    if preload_epoch:
        preload_subdir = getattr(args, 'preload_subdir', 'best')
        if rank == 0:
            print(f"🔄 预加载阶段权重: epoch={preload_epoch}, 子目录={preload_subdir}")
        try:
            model.load_model(args, phase2_epoch=preload_epoch, subdir=preload_subdir)
            if rank == 0:
                print("✅ 预加载完成，即将在此基础上继续训练")
        except Exception as e:
            if rank == 0:
                print(f"⚠️ 预加载失败（继续随机初始化）: {e}")
        if args.multi_gpu:
            _dist_barrier(rank)
    print('user num:', usernum, 'item num:', itemnum)
    cc = 0.0
    for u in user_train:
        cc += len(user_train[u])
    print('average sequence length: %.2f' % (cc / len(user_train)))
    # Init Dataloader, Model, Optimizer
    base_seq_dataset = SeqDataset(user_train, len(user_train.keys()), itemnum, args.maxlen)
    train_data_set = base_seq_dataset
    train_dataset_kind = 'sequence_injection'
    sa_dataset_meta = {'neighbor_path': None}

    train_data_loader = None
    current_batch_size = None
    current_num_batches = 0

    def build_train_dataloader(batch_size: int):
        if args.multi_gpu:
            if not torch.distributed.is_initialized():
                print(f"❌ 错误：分布式环境未初始化，无法创建DistributedSampler")
                raise RuntimeError("分布式环境未初始化")
            if rank == 0:
                print(f"🔧 创建DistributedSampler: rank={rank}, world_size={world_size}, batch_size={batch_size}, dataset={len(train_data_set)}")
            sampler = DistributedSampler(train_data_set, shuffle=True, rank=rank, num_replicas=world_size)
            loader = DataLoader(
                train_data_set,
                batch_size=batch_size,
                sampler=sampler,
                pin_memory=True,
                num_workers=4,
            )
        else:
            loader = DataLoader(
                train_data_set,
                batch_size=batch_size,
                pin_memory=True,
                shuffle=True,
                num_workers=4,
            )
        steps = max(1, len(loader))
        return loader, steps

    def build_sequence_injection_dataset():
        """构建 SI 阶段数据集：使用基础序列数据集"""
        return base_seq_dataset

    def ensure_phase_dataloader(announce: bool = False) -> bool:
        nonlocal train_data_set, train_dataset_kind, train_data_loader, current_batch_size, current_num_batches
        if train_data_set is None or train_dataset_kind != 'sequence_injection':
            train_data_set = build_sequence_injection_dataset()
            train_dataset_kind = 'sequence_injection'
        target_bs = args.batch_size
        if train_data_loader is not None and current_batch_size == target_bs:
            return False
        if rank == 0:
            print(f"📊 准备构建DataLoader: dataset_size={len(train_data_set)}, kind={train_dataset_kind}")
        train_data_loader, current_num_batches = build_train_dataloader(target_bs)
        current_batch_size = target_bs
        if rank == 0:
            print(f"🧰 序列注入阶段 batch_size = {target_bs}, batches = {current_num_batches}")
        return True

    if args.multi_gpu:
        # ✅ 确保分布式环境已完全初始化
        if not torch.distributed.is_initialized():
            print(f"❌ 错误：分布式环境未初始化，无法创建DistributedSampler")
            raise RuntimeError("分布式环境未初始化")

        # 🔧 DDP配置 - 根据训练模式选择
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
                print("✅ DDP初始化成功 (static_graph=True)")

        except RuntimeError as e:
            if rank == 0:
                print(f"⚠️ DDP初始化失败: {e}")
            raise e

        # 保留一次空初始化，兼容旧日志中的 FutureWarning 行为。
        scaler = GradScaler()
        del scaler

        # 🔥 DDP参数同步和梯度修复
        if args.multi_gpu:
            # 确保所有进程的参数同步
            _dist_barrier(rank)

            # 为双向知识蒸馏做特殊处理
            if hasattr(args, 'enable_bidirectional_kd') and args.enable_bidirectional_kd:
                # 确保双向蒸馏参数在所有GPU上同步
                for param in model.parameters():
                    if param.requires_grad:
                        torch.distributed.broadcast(param.data, src=0)

                if rank == 0:
                    print("🔥 DDP双向知识蒸馏参数同步完成")

            _dist_barrier(rank)
    # 取出真实模型（DDP 包装则使用 .module）以读取/写入统计指标
    core_model = model.module if isinstance(model, DDP) else model
    core_model._shared_scaler = shared_scaler

    # 🔧 动态优化器：仅跟踪当前解冻的参数，阶段切换时重建以释放显存
    def lr_lambda(epoch):
        if epoch <= 5:
            return 1.0  # 前5轮保持原始学习率
        elif epoch <= 10:
            return 0.98 ** (epoch - 5)  # 第6-10轮温和衰减
        else:
            return 0.95 ** (epoch - 10) * 0.98 ** 5  # 后续轮次加速衰减

    optimizer_param_signature = None
    adam_optimizer = None
    scheduler = None

    def _collect_trainable_params():
        params = [p for p in model.parameters() if p.requires_grad]
        if not params:  # 兜底：若全部被冻结，仍返回全部参数避免空列表
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
        
        # 🔥 显式释放旧优化器状态
        old_optimizer_existed = adam_optimizer is not None
        if old_optimizer_existed:
            del adam_optimizer
        if scheduler is not None:
            del scheduler
        
        # 创建新的优化器
        adam_optimizer = torch.optim.Adam(params, lr=args.stage2_lr, betas=(0.9, 0.98))
        
        # 🔥 关键修复：为每个参数组设置initial_lr，LambdaLR需要这个属性
        for group in adam_optimizer.param_groups:
            group.setdefault('initial_lr', args.stage2_lr)
        
        sched_last_epoch = -1 if epoch_marker <= 0 else epoch_marker - 1
        
        # 创建新的学习率调度器
        scheduler = LambdaLR(adam_optimizer, lr_lambda=lr_lambda, last_epoch=sched_last_epoch)
        
        optimizer_param_signature = param_ids
        if hasattr(core_model, 'optimizer_needs_reset'):
            core_model.optimizer_needs_reset = False
        if rank == 0:
            mem_after = torch.cuda.memory_allocated() / 1024**3 if torch.cuda.is_available() else 0
            current_lr = adam_optimizer.param_groups[0]['lr']
            print(f"🧽 重建优化器[{reason}]：可训练参数 {len(params)} 个，last_epoch={scheduler.last_epoch}, 当前LR={current_lr:.6f}, GPU显存={mem_after:.2f}GB")
    weighting_scheduler = build_weighting_scheduler(args)
    if weighting_scheduler is not None and rank == 0:
        print(f"🔥 自适应蒸馏权重调度器启动: start={weighting_scheduler.start_beta:.3f}, "
              f"max={weighting_scheduler.max_beta:.3f}, warmup={weighting_scheduler.warmup_epochs} epoch(s)")
    base_phase_weights = {
        'teacher_rec': getattr(args, 'teacher_rec_weight', 1.0),
        'student_rec': getattr(args, 'student_rec_weight', 0.0),
        'forward': getattr(args, 'forward_weight', getattr(args, 'forward_kd_weight', 1.0)),
        'backward': getattr(args, 'reverse_weight', getattr(args, 'backward_kd_weight', 1.0)),
    }

    current_phase = 'sequence_injection'
    current_round = 1

    # SI阶段权重配置
    si_weight_cfg = {
        'teacher_rec': base_phase_weights['teacher_rec'],
        'forward': base_phase_weights['forward'],
    }
    core_model.configure_training_phase(
        current_phase,
        si_weight_cfg,
        verbose=(rank == 0),
        round_idx=current_round,
    )

    ensure_phase_dataloader(announce=True)

    rebuild_optimizer(reason='initial', epoch_marker=0, force=True)
    
    # 🔥 如果使用 --preload_epoch，从该epoch+1开始训练
    preload_epoch_arg = getattr(args, 'preload_epoch', None)
    if preload_epoch_arg:
        epoch_start_idx = preload_epoch_arg + 1
        if rank == 0:
            print(f"🔄 从预加载的epoch {preload_epoch_arg} 继续，将从epoch {epoch_start_idx} 开始训练")
    else:
        epoch_start_idx = 1
    total_epoch_cap = args.num_epochs
    
    T = 0.0
    perform = 0

    # Early stop on validation HR@10, but keep a minimum number of epochs.
    best_perform = getattr(args, 'preload_best_metric', 0.0)
    best_student_metric = float('-inf')
    early_stop = 0
    early_thres = max(1, int(getattr(args, 'early_stop_patience', 4)))
    min_epochs_before_early_stop = max(1, int(getattr(args, 'min_epochs_before_early_stop', 12)))
    t0 = time.time()

    # 断点续训：加载checkpoint
    if hasattr(args, 'resume') and args.resume and os.path.exists(args.resume):
        if rank == 0:
            print(f"🔄 从checkpoint恢复训练: {args.resume}")
        checkpoint = torch.load(args.resume, map_location=args.device)

        # 加载模型状态
        if args.multi_gpu:
            model.module.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint['model_state_dict'])

        # 加载优化器和调度器状态
        adam_optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        # 恢复训练状态
        epoch_start_idx = checkpoint['epoch'] + 1
        best_perform = checkpoint.get('best_perform', 0)
        early_stop = checkpoint.get('early_stop', 0)

        # 恢复早停配置（如果checkpoint中有保存）
        if 'early_thres' in checkpoint:
            early_thres = checkpoint['early_thres']
        if 'min_epochs_before_early_stop' in checkpoint:
            min_epochs_before_early_stop = checkpoint['min_epochs_before_early_stop']

        if rank == 0:
            print(f"✅ 恢复成功！从第 {epoch_start_idx} 个epoch开始，最佳性能: {best_perform:.4f}")
    elif hasattr(args, 'resume') and args.resume:
        if rank == 0:
            print(f"⚠️ 警告：checkpoint文件不存在: {args.resume}")
            print("🔄 将从头开始训练...")
    
    eval_set_use = eval_set[1]
    if len(eval_set_use)>10000:
        users = random.sample(list(eval_set_use), 10000)
    else:
        users = list(eval_set_use)
    
    user_list = []
    for u in users:
        if len(user_test[u]) < 1: continue
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
            num_workers=0,  # 禁用多进程，避免与NCCL冲突
        )

    run_eval_on_rank = (not args.multi_gpu) or rank == 0
    eval_rounds = getattr(args, 'eval_rounds_per_epoch', 0)
    eval_interval_steps = None

    def recompute_eval_interval_steps(announce: bool = False):
        nonlocal eval_interval_steps
        if eval_rounds > 0 and current_num_batches > 0:
            eval_interval_steps = max(1, current_num_batches // eval_rounds)
            if announce and rank == 0:
                print(f"📊 每个epoch将进行 {eval_rounds} 轮验证（约每 {eval_interval_steps} step 一次）")
        else:
            eval_interval_steps = None
            if announce and rank == 0:
                print("📊 eval_rounds_per_epoch<=0，仅在每个epoch结束时执行验证")

    recompute_eval_interval_steps(announce=True)

    def run_validation_and_maybe_test(epoch, trigger='epoch-end', step=None):
        nonlocal best_perform, early_stop, best_student_metric

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
            print(f"⚠️ [{trigger}] 无可用验证用户，跳过验证")
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
        core_model_local.all_embs = None  # 清空缓存，强制重新构建 item embeddings

        # 计算“学生模型”指标
        student_stats = {'users': 0.0, 'HT10': 0.0, 'HT20': 0.0, 'NDCG10': 0.0, 'NDCG20': 0.0}

        with torch.no_grad():
            for batch_idx, data in enumerate(valid_data_loader):
                if batch_idx == 0 and rank == 0:
                    tag = f"Validation ({trigger}, epoch {epoch}"
                    if step is not None:
                        tag += f", step {step}"
                    tag += f")，early stop: {early_stop}"
                    print(tag)
                u, seq, pos, neg = data
                u, seq, pos, neg = u.numpy(), seq.numpy(), pos.numpy(), neg.numpy()
                
                core_model_local([u, seq, pos, neg, rank, None, 'original'], mode='generate_batch')

                student_batch = core_model_local.evaluate_student_batch(
                    u, seq, pos,
                    candidate_num=getattr(args, 'student_eval_candidates', 100),
                    split='valid',
                )
                for key in student_stats:
                    student_stats[key] += student_batch.get(key, 0.0)

        valid_users = core_model_local.users
        perform = core_model_local.HT / valid_users if valid_users > 0 else 0.0
        ndcg_perform = core_model_local.NDCG / valid_users if valid_users > 0 else 0.0
        llm_hr20 = core_model_local.HIT_20 / valid_users if valid_users > 0 else 0.0
        llm_ndcg20 = core_model_local.NDCG_20 / valid_users if valid_users > 0 else 0.0

        student_users = student_stats['users']
        if student_users > 0:
            student_hr = student_stats['HT10'] / student_stats['users']
            student_ndcg = student_stats['NDCG10'] / student_stats['users']
            student_hr20 = student_stats['HT20'] / student_stats['users']
            student_ndcg20 = student_stats['NDCG20'] / student_stats['users']
        else:
            student_hr = student_ndcg = student_hr20 = student_ndcg20 = 0.0

        core_model_local._last_student_metrics = {
            'hr10': student_hr,
            'ndcg10': student_ndcg,
            'hr20': student_hr20,
            'ndcg20': student_ndcg20,
            'users': student_users,
        }

        if rank == 0:
            print(f"[{trigger}] Valid Set - LLM HR@10 = {perform:.4f}, NDCG@10 = {ndcg_perform:.4f}")
            print(f"[{trigger}] Valid Set - Small HR@10 = {student_hr:.4f}, NDCG@10 = {student_ndcg:.4f} | HR@20 = {student_hr20:.4f}, NDCG@20 = {student_ndcg20:.4f}")

        # 🔥 删除save_intermediate和细粒度阶段跟踪，仅保留测试集评估和全局早停
        apply_early_stop = True
        is_new_best = perform >= best_perform

        # === 每个epoch都执行测试与记录 ===
        student_hr10 = student_ndcg10 = student_hr20 = student_ndcg20 = 0.0
        den = 1.0
        # 🔥 只有 rank0 执行测试
        test_llm_hr10 = test_llm_ndcg10 = test_llm_hr20 = test_llm_ndcg20 = 0.0
        if inference_data_loader is not None:
            core_model_local.users = 0.0
            core_model_local.NDCG = 0.0
            core_model_local.HT = 0.0
            core_model_local.NDCG_20 = 0.0
            core_model_local.HIT_20 = 0.0
            core_model_local.all_embs = None
            student_test_stats = {'users': 0.0, 'HT10': 0.0, 'HT20': 0.0, 'NDCG10': 0.0, 'NDCG20': 0.0}
            with torch.no_grad():
                print("Testing")
                for batch_idx, data in enumerate(inference_data_loader):
                    if batch_idx == 0 and rank == 0:
                        print(f"🧪 Testing on test set ({len(inference_data_loader)} batches)")
                    u, seq, pos, neg = data
                    u, seq, pos, neg = u.numpy(), seq.numpy(), pos.numpy(), neg.numpy()
                    
                    core_model_local([u, seq, pos, neg, rank, None, 'original'], mode='generate_batch')

                    student_batch = core_model_local.evaluate_student_batch(
                        u, seq, pos,
                        candidate_num=getattr(args, 'student_eval_candidates', 100),
                        split='test',
                    )
                    for key in student_test_stats:
                        student_test_stats[key] += student_batch.get(key, 0.0)

            student_users = student_test_stats['users']
            if student_users > 0:
                student_hr10 = student_test_stats['HT10'] / student_users
                student_ndcg10 = student_test_stats['NDCG10'] / student_users
                student_hr20 = student_test_stats['HT20'] / student_users
                student_ndcg20 = student_test_stats['NDCG20'] / student_users
            den = core_model_local.users if core_model_local.users > 0 else 1.0
            test_llm_hr10 = core_model_local.HT / den
            test_llm_ndcg10 = core_model_local.NDCG / den
            test_llm_hr20 = core_model_local.HIT_20 / den
            test_llm_ndcg20 = core_model_local.NDCG_20 / den

            if rank == 0:
                print(f"[{trigger}] 📊 Test Set - Small HR@10={student_hr10:.4f}, NDCG@10={student_ndcg10:.4f}, HR@20={student_hr20:.4f}, NDCG@20={student_ndcg20:.4f}")

                out_dir = os.path.join('./models', args.rec_pre_trained_data, args.save_dir.rstrip('/'))
                create_dir(out_dir)
                all_metrics_file = os.path.join(out_dir, f"{args.rec_pre_trained_data}_{args.llm}_all_results.txt")
                status_label = "NEW BEST" if is_new_best else "EPOCH RESULT"
                with open(all_metrics_file, 'a') as f:
                    f.write(f"========== Epoch {epoch} ({trigger}) - {status_label} ==========\n")
                    f.write("Test Set:\n")
                    f.write(f"  LLM    - HR@10: {test_llm_hr10:.16f}, NDCG@10: {test_llm_ndcg10:.16f}, HR@20: {test_llm_hr20:.16f}, NDCG@20: {test_llm_ndcg20:.16f}\n")
                    f.write(f"  SASRec - HR@10: {student_hr10:.16f}, NDCG@10: {student_ndcg10:.16f}, HR@20: {student_hr20:.16f}, NDCG@20: {student_ndcg20:.16f}\n")
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
                    f.write(f'Small NDCG@10: {student_ndcg10:.16f}, Small HR@10: {student_hr10:.16f}\n')
                    f.write(f'Small NDCG@20: {student_ndcg20:.16f}, Small HR@20: {student_hr20:.16f}\n')

        metrics_bundle = {
            'llm': {
                'hr10': perform,
                'ndcg10': ndcg_perform,
                'hr20': llm_hr20,
                'ndcg20': llm_ndcg20,
            },
            'student': {
                'hr10': student_hr,
                'ndcg10': student_ndcg,
                'hr20': student_hr20,
                'ndcg20': student_ndcg20,
            },
            'test_llm': {
                'hr10': test_llm_hr10,
                'ndcg10': test_llm_ndcg10,
                'hr20': test_llm_hr20,
                'ndcg20': test_llm_ndcg20,
            },
            'test_student': {
                'hr10': student_hr10,
                'ndcg10': student_ndcg10,
                'hr20': student_hr20,
                'ndcg20': student_ndcg20,
            }
        }

        # 🧹 验证/测试结束后清理一次显存
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # 🔥 优先显示测试集结果
        if rank == 0 and inference_data_loader is not None:
            print(f"\n📊 [{trigger}] Test Set Results:")
            print(f"  LLM:    HR@10={test_llm_hr10:.4f}, NDCG@10={test_llm_ndcg10:.4f}")
            print(f"  SASRec: HR@10={student_hr10:.4f}, NDCG@10={student_ndcg10:.4f}\n")

        # 🔥 使用验证集LLM HR@10判断全局最佳（与Start 1.0一致）
        if is_new_best:
            best_perform = perform
            if not getattr(args, 'disable_model_saving', False):
                if rank == 0:
                    print(f"💾 [{trigger}] 开始保存最佳模型...", flush=True)
                    try:
                        save_target = model.module if args.multi_gpu else model
                        reset_best_model_dir(args)
                        save_target.save_model(args, epoch2=epoch, best=True)
                        print(f"💾 [{trigger}] 最佳模型保存完成", flush=True)
                    except Exception as e:
                        print(f"⚠️ [{trigger}] 模型保存失败: {e}", flush=True)
                        import traceback
                        traceback.print_exc()

            if apply_early_stop:
                early_stop = 0
            if rank == 0:
                print(f"🎉 [{trigger}] 新的最佳性能 (Valid Set)! LLM HR@10 = {perform:.4f}, NDCG@10 = {ndcg_perform:.4f}", flush=True)
        else:
            # ❌ 性能未提升时不保存模型（节省磁盘空间）
            if apply_early_stop:
                early_stop += 1
                print(f"⚠️ [{trigger}] 验证集LLM HR@10未提升，早停计数: {early_stop}/{early_thres}")

        # 🔥 确保所有文件I/O完成后再返回
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        
        return metrics_bundle

    def guarded_validation_call(epoch, trigger='epoch-end', step=None):
        """简单封装，不添加额外 barrier（保持原始 EchoRec 行为）"""
        metrics = run_validation_and_maybe_test(epoch, trigger=trigger, step=step)
        return metrics

    for epoch in tqdm(range(epoch_start_idx, total_epoch_cap + 1)):
        # 🔥 每个epoch开始前强制清理显存，防止验证后残留导致OOM
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
            _dist_barrier(rank)  # 🔥 确保所有 rank 同时开始新 epoch，防止步数错位

        # 🔧 自适应权重调整（每个epoch开始时）
        if hasattr(core_model, 'update_adaptive_weights'):
            core_model.update_adaptive_weights(epoch)

        # 🎯 更新当前epoch（用于动态权重调整）
        if hasattr(core_model, 'current_epoch'):
            core_model.current_epoch = epoch

        # 🔥 梯度屏蔽：更新自适应权重调节器
        if weighting_scheduler is not None:
            weighting_scheduler.step()
            if rank == 0 and epoch % 3 == 1:
                beta = weighting_scheduler.get_beta()
                print(f"🔥 Epoch {epoch}: 自适应蒸馏权重 β = {beta:.4f}")

        # 🚀 训练速度监控 + 内存管理
        epoch_start_time = time.time()
        epoch_loss = 0.0
        step_count = 0

        stage_epoch_idx = epoch
        stage_epoch_total = args.num_epochs

        # 🔧 每个epoch开始前优化内存
        optimize_memory_usage(rank, verbose=True)
        if rank == 0 and epoch % 3 == 1:
            print(f"💾 Epoch {epoch} 开始前内存状态检查完成")

        # ✅ 恢复baseline：不使用混合精度，直接前向传播
        if rank == 0:
            print(f"🚀 训练: batches={current_num_batches}, loader_len={len(train_data_loader)}")

        # 完整训练一个epoch的所有数据
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

            # SI阶段：直接调用，内部train_mode0已有autocast
            loss_tensor = model(
                batch_payload,
                optimizer=adam_optimizer,
                batch_iter=[stage_epoch_idx, stage_epoch_total, step, current_num_batches],
                mode='phase2',
            )
            # SI阶段在模型内部已执行backward/step/zero_grad

            if getattr(args, 'nn_parameter', False) and 'htcore' in globals() and htcore is not None:
                htcore.mark_step()

            # ✅ 对齐Start_old: SI阶段返回None，跳过loss统计
            if loss_tensor is None:
                loss_scalar = 0.0  # SI阶段已在内部打印日志，这里只做占位
            elif isinstance(loss_tensor, torch.Tensor):
                loss_scalar = loss_tensor.detach().item()
            else:
                loss_scalar = float(loss_tensor)
            epoch_loss += loss_scalar

            step_count += 1

            # 🧹 梯度清理（可选，通常不需要）
            # if step % 50 == 0 and torch.cuda.is_available():  # 进一步降低清理频率
            #     torch.cuda.empty_cache()  # 移除以避免CUDA非法内存访问

            # 中间进度显示（不中断训练，仅记录进度）
            if rank == 0:
                current_lr = adam_optimizer.param_groups[0]['lr']
                avg_loss = epoch_loss / step_count if step_count > 0 else 0
                train_iter.set_postfix(
                    loss=f"{avg_loss:.4f}",
                    lr=f"{current_lr:.6f}",
                    step=f"{step + 1}/{current_num_batches}",
                )



        # 🚀 训练速度统计
        if rank == 0:
            train_iter.close()
            epoch_time = time.time() - epoch_start_time
            samples_per_sec = len(train_data_set) / epoch_time
            gpu_info = f"{world_size}GPU" if args.multi_gpu else "1GPU"
            print(f"⚡ Epoch {epoch} 训练速度: {epoch_time:.1f}s, {samples_per_sec:.1f} samples/s ({gpu_info})")

        # ✅ SI阶段现在走DDP forward，不再需要手动同步参数

        epoch_metrics = guarded_validation_call(epoch, trigger='epoch-end')
        model.train()

        stop_training = False
        if rank == 0:
            if early_stop >= early_thres and epoch >= min_epochs_before_early_stop:
                print(f"🛑 达到早停条件，训练结束")
                print(f"📊 训练轮数: {epoch}, 早停计数: {early_stop}, 最佳性能: {best_perform:.4f}")
                stop_training = True
            elif early_stop >= early_thres:
                print(f"⏳ 早停条件满足但未达到最小训练轮数 ({epoch}/{min_epochs_before_early_stop})")

        if args.multi_gpu:
            stop_tensor = torch.zeros(1, device=args.device)
            if rank == 0:
                stop_tensor.fill_(1 if stop_training else 0)
            torch.distributed.broadcast(stop_tensor, src=0)
            stop_training = bool(stop_tensor.item())
            _dist_barrier(rank)

        if stop_training:
            break

        # 🔥 关键修复：所有rank都要执行scheduler.step()，保持状态一致
        scheduler.step()

        # 确保模型回到训练模式
        model.train()


    # 训练正常结束，确保所有进程同步
    if args.multi_gpu:
        _dist_barrier(rank)
        if rank == 0:
            print('✅ 所有进程训练完成，开始清理...')

    if rank == 0:
        print(f'⏱️ 总训练时间: {time.time() - t0:.1f}s')

    if args.multi_gpu:
        destroy_process_group()

    # 训练结束时清理临时文件
    cleanup_temp_files()
    return


def extract_teacher_embeddings(args):
    """
    提取用户和物品表示向量
    """
    print(f"开始提取表示向量...")
    print(f"数据集: {args.rec_pre_trained_data}")
    print(f"模型: {args.llm}")
    print(f"保存目录: {args.save_dir}")

    # 设置设备
    device = args.device

    # 加载数据
    dataset = data_partition(args.rec_pre_trained_data, args, path=f'./SeqRec/data_{args.rec_pre_trained_data}/{args.rec_pre_trained_data}')
    [user_train, user_valid, user_test, usernum, itemnum, eval_set] = dataset
    print('用户数:', usernum, '物品数:', itemnum)

    # 创建模型
    model = EchoRecSIModel(args)
    model = model.to(device)

    # 加载训练好的模型权重
    try:
        # 从保存目录中查找模型文件来确定epoch
        import glob
        model_files = glob.glob(f'./models/{args.rec_pre_trained_data}/{args.save_dir}/{args.rec_pre_trained_data}_{args.llm}_*_item_proj.pt')
        if model_files:
            # 从文件名中提取epoch数字
            # 文件名格式: Movies_and_TV_llama-3b_1_item_proj.pt
            filename = os.path.basename(model_files[0])
            # 移除.pt后缀，然后按_分割
            name_parts = filename.replace('.pt', '').split('_')
            # 找到数字部分（应该在llama-3b之后）
            for i, part in enumerate(name_parts):
                if part.isdigit():
                    phase2_epoch = int(part)
                    break
            else:
                phase2_epoch = 1  # 如果没找到数字，使用默认值
            print(f"找到模型文件: {filename}, epoch: {phase2_epoch}")
        else:
            print("未找到模型文件，使用默认epoch=1")
            phase2_epoch = 1

        model.load_model(args, phase2_epoch=phase2_epoch)
        print("✅ 模型权重加载成功")
    except Exception as e:
        print(f"❌ 模型权重加载失败: {e}")
        return

    model.eval()

    # 准备提取数据 - 使用最简单的SeqDataset，只需要训练数据
    user_list_extract = list(user_train.keys())
    # 使用SeqDataset，它只需要user_train数据，不依赖验证或测试数据
    extract_data_set = SeqDataset(user_train, len(user_list_extract), itemnum, args.maxlen)
    extract_data_loader = DataLoader(
        extract_data_set,
        batch_size=args.batch_size_infer,
        pin_memory=True,
        shuffle=False,
        num_workers=0  # 禁用多进程以避免索引问题
    )

    print(f"开始提取 {len(user_list_extract)} 个用户的表示向量...")

    # 提取用户表示
    with torch.no_grad():
        for step, data in enumerate(tqdm(extract_data_loader, desc="提取用户表示")):
            # SeqDataset返回(user_id, seq, pos, neg)，需要扩展为extract_emb期望的格式
            u, seq, pos, neg = data
            # 扩展数据格式以匹配extract_emb的期望：(u, seq, pos, neg, original_seq, rank, files)
            extended_data = (u, seq, pos, neg, seq, None, None)  # original_seq=seq, rank=None, files=None
            model(extended_data, mode='extract')

    # 保存提取的表示
    if hasattr(model, 'extract_embs_list') and model.extract_embs_list:
        import pickle
        extract_embs = torch.cat(model.extract_embs_list, dim=0)

        # 创建保存目录
        save_path = f'./models/{args.rec_pre_trained_data}/{args.save_dir}/'
        os.makedirs(save_path, exist_ok=True)

        # 保存用户表示
        user_emb_path = f'{save_path}{args.rec_pre_trained_data}_{args.llm}_user_embeddings.pkl'
        with open(user_emb_path, 'wb') as f:
            pickle.dump(extract_embs.numpy(), f)

        print(f"✅ 用户表示已保存到: {user_emb_path}")
        print(f"表示形状: {extract_embs.shape}")
    else:
        print("❌ 没有提取到用户表示")

    print("🎉 表示提取完成！")

    # 优雅清理分布式进程
    if args.multi_gpu and dist.is_initialized():
        try:
            dist.destroy_process_group()
            print("🔧 分布式进程组已清理")
        except Exception as e:
            print(f"⚠️ 清理分布式进程组失败: {e}")

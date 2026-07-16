#!/usr/bin/env python
"""
Latte Model Quick Start Training Script

This script performs a single training run for Latte model with fixed hyperparameters,
followed by inference with num_beams=500.

Usage:
    python train_latte.py \
        --category=Industrial_and_Scientific \
        --gpu_id=0 \
        --n_latent_tokens=8 \
        --vq_method=rqkmeans \
        --lr=3e-3
"""

import os
import sys
import shutil
import argparse
from datetime import datetime
import hashlib

from genrec.pipeline import Pipeline
from genrec.utils import parse_command_line_args


# Model configuration for Latte
LATTE_CONFIG = {
    'model': 'Latte',
    'aggregation_method': 'agg_max',
}

# Model size configuration
MODEL_SIZE_CONFIG = {
    'd_model': 128,
    'd_ff': 1024
}


def get_wandb_run_name(category, lr, n_latent_tokens, vq_method, phase='train'):
    """Generate a meaningful wandb run name."""
    lr_str = f'lr{lr:.0e}'.replace('e-0', 'e-')
    if phase == 'train':
        return f'Latte_{category}_{lr_str}_nlt{n_latent_tokens}_{vq_method}'
    else:
        return f'Latte_{category}_{lr_str}_nlt{n_latent_tokens}_{vq_method}_beam500'


def run_training(category, lr, model_config, wandb_run_name, extra_args=None):
    """Run a single training experiment and return results."""
    kwargs = {
        'dataset': 'AmazonReviews2023',
        'category': category,
        'tracker': 'wandb',
        'wandb_run_name': wandb_run_name,
        'eval_interval': 3,
        'lr': lr,
    }
    kwargs.update(model_config)
    kwargs.update(MODEL_SIZE_CONFIG)
    if extra_args:
        kwargs.update(extra_args)
    
    pipeline = Pipeline(
        model_name='Latte',
        dataset_name='AmazonReviews2023',
        config_dict=kwargs
    )
    ret = pipeline.run(skip_end_training=True)
    
    # Get the checkpoint path
    ckpt_path = pipeline.trainer.saved_model_ckpt
    
    return {
        'best_epoch': ret['best_epoch'],
        'best_val_score': ret['best_val_score'],
        'test_results': ret['test_results'],
        'ckpt_path': ckpt_path,
    }


def run_inference(category, checkpoint_path, model_config, lr, wandb_run_name, extra_args=None):
    """Run inference with num_beams=500."""
    kwargs = {
        'dataset': 'AmazonReviews2023',
        'category': category,
        'tracker': 'wandb',
        'wandb_run_name': wandb_run_name,
        'epochs': 0,  # No training, just inference
        'num_beams': 500,
        'eval_batch_size': 32,
        'lr': lr,
    }
    kwargs.update(model_config)
    kwargs.update(MODEL_SIZE_CONFIG)
    if extra_args:
        kwargs.update(extra_args)
    
    pipeline = Pipeline(
        model_name='Latte',
        dataset_name='AmazonReviews2023',
        checkpoint_path=checkpoint_path,
        config_dict=kwargs
    )
    ret = pipeline.run(skip_end_training=False)
    
    return ret['test_results']


def main():
    parser = argparse.ArgumentParser(description='Latte Model Quick Start Training Script')
    parser.add_argument('--category', type=str, default='Industrial_and_Scientific',
                        help='Dataset category (e.g., Industrial_and_Scientific, Video_Games)')
    parser.add_argument('--gpu_id', type=int, default=0,
                        help='GPU ID to use (single GPU)')
    parser.add_argument('--n_latent_tokens', type=int, default=8,
                        help='Number of latent tokens')
    parser.add_argument('--vq_method', type=str, default='rqkmeans',
                        help='VQ method to use (rqkmeans, opq, rqvae)')
    parser.add_argument('--aggregation_method', type=str, default='agg_max',
                        help='Aggregation method (agg_max, agg_sum)')
    parser.add_argument('--lr', type=float, default=3e-3,
                        help='Learning rate (default: 3e-3)')
    args, unparsed_args = parser.parse_known_args()
    extra_args = parse_command_line_args(unparsed_args)

    # Set CUDA device
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu_id)

    # Get model configuration
    model_config_base = LATTE_CONFIG.copy()
    model_name = model_config_base.pop('model')
    model_config_base['vq_method'] = args.vq_method
    model_config_base['n_latent_tokens'] = args.n_latent_tokens
    model_config_base['aggregation_method'] = args.aggregation_method
    
    # Initialize logging
    os.makedirs('logs', exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    cmd_hash = hashlib.md5('-'.join(sys.argv).encode()).hexdigest()[:8]
    LOG_NAME = os.path.join('logs', f'train_latte_{args.category}_{timestamp}_{cmd_hash}.log')
    
    hlog = open(LOG_NAME, 'a+')
    hlog.write('=' * 80 + '\n')
    hlog.write(f'Latte Model Quick Start Training\n')
    hlog.write(f'Command line: {" ".join(sys.argv)}\n')
    hlog.write('=' * 80 + '\n')
    hlog.write('======= CONFIG =======\n')
    hlog.write(f'Model: {model_name}\n')
    hlog.write(f'Category: {args.category}\n')
    hlog.write(f'GPU ID: {args.gpu_id}\n')
    hlog.write(f'Learning Rate: {args.lr}\n')
    hlog.write(f'n_latent_tokens: {args.n_latent_tokens}\n')
    hlog.write(f'vq_method: {args.vq_method}\n')
    hlog.write(f'aggregation_method: {args.aggregation_method}\n')
    hlog.write(f'Model Config: {model_config_base}\n\n')
    hlog.flush()

    print(f'\n{"=" * 60}')
    print(f'Latte Model Quick Start Training')
    print(f'Category: {args.category}')
    print(f'GPU ID: {args.gpu_id}')
    print(f'Learning Rate: {args.lr}')
    print(f'n_latent_tokens: {args.n_latent_tokens}')
    print(f'vq_method: {args.vq_method}')
    print(f'aggregation_method: {args.aggregation_method}')
    print(f'{"=" * 60}\n')

    # Phase 1: Training
    print('\n' + '=' * 60)
    print('Phase 1: Training')
    print('=' * 60 + '\n')
    hlog.write('\n======= PHASE 1: TRAINING =======\n\n')
    hlog.flush()

    wandb_run_name = get_wandb_run_name(args.category, args.lr, args.n_latent_tokens, args.vq_method, phase='train')
    print(f'WandB Run Name: {wandb_run_name}')
    
    try:
        ret = run_training(
            category=args.category,
            lr=args.lr,
            model_config=model_config_base,
            wandb_run_name=wandb_run_name,
            extra_args=extra_args,
        )
        
        hlog.write(f'Best Epoch: {ret["best_epoch"]}\n')
        hlog.write(f'Best Valid Score: {ret["best_val_score"]}\n')
        hlog.write(f'Test Results: {ret["test_results"]}\n')
        hlog.write(f'Checkpoint: {ret["ckpt_path"]}\n\n')
        hlog.flush()

        print(f'Best Epoch: {ret["best_epoch"]}')
        print(f'Best Valid Score: {ret["best_val_score"]}')
        print(f'Test Results: {ret["test_results"]}')
        
        ckpt_path = ret['ckpt_path']
        
    except Exception as e:
        hlog.write(f'Training FAILED\n')
        hlog.write(f'Error: {str(e)}\n\n')
        hlog.flush()
        print(f'Training failed with error: {e}')
        hlog.close()
        return

    # Phase 2: Save checkpoint to saved/ folder
    print('\n' + '=' * 60)
    print('Phase 2: Saving checkpoint')
    print('=' * 60 + '\n')
    hlog.write('\n======= PHASE 2: SAVING CHECKPOINT =======\n\n')
    hlog.flush()

    os.makedirs('saved', exist_ok=True)
    saved_ckpt_name = f'{model_name}-{args.category}-{args.vq_method}-nlt{args.n_latent_tokens}.pth'
    saved_ckpt_path = os.path.join('saved', saved_ckpt_name)
    
    shutil.copy(ckpt_path, saved_ckpt_path)
    print(f'Saved checkpoint to: {saved_ckpt_path}')
    hlog.write(f'Source checkpoint: {ckpt_path}\n')
    hlog.write(f'Saved checkpoint: {saved_ckpt_path}\n\n')
    hlog.flush()

    # Phase 3: Inference with num_beams=500
    print('\n' + '=' * 60)
    print('Phase 3: Inference with num_beams=500')
    print('=' * 60 + '\n')
    hlog.write('\n======= PHASE 3: INFERENCE (num_beams=500) =======\n\n')
    hlog.flush()

    wandb_run_name = get_wandb_run_name(args.category, args.lr, args.n_latent_tokens, args.vq_method, phase='inference')
    print(f'WandB Run Name: {wandb_run_name}')
    
    inference_results = run_inference(
        category=args.category,
        checkpoint_path=saved_ckpt_path,
        model_config=model_config_base,
        lr=args.lr,
        wandb_run_name=wandb_run_name,
        extra_args=extra_args,
    )

    # Final summary
    print('\n' + '=' * 60)
    print('FINAL RESULTS')
    print('=' * 60)
    print(f'Learning Rate: {args.lr}')
    print(f'Best Epoch: {ret["best_epoch"]}')
    print(f'Best Validation Score: {ret["best_val_score"]}')
    print(f'Test Results (default beam): {ret["test_results"]}')
    print(f'Inference Results (beam=500): {inference_results}')
    print(f'Saved Checkpoint: {saved_ckpt_path}')
    print('=' * 60 + '\n')

    hlog.write('\n======= FINAL RESULTS =======\n')
    hlog.write(f'Learning Rate: {args.lr}\n')
    hlog.write(f'Best Epoch: {ret["best_epoch"]}\n')
    hlog.write(f'Best Validation Score: {ret["best_val_score"]}\n')
    hlog.write(f'Test Results (default beam): {ret["test_results"]}\n')
    hlog.write(f'Inference Results (beam=500): {inference_results}\n')
    hlog.write(f'Saved Checkpoint: {saved_ckpt_path}\n')
    hlog.close()

    print(f'\nLog saved to: {LOG_NAME}')


if __name__ == '__main__':
    main()

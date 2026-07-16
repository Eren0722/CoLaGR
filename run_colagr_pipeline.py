import argparse
import os
import subprocess
import sys


def run(cmd):
    print('+ ' + ' '.join(cmd))
    subprocess.run(cmd, check=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='AmazonReviews2023')
    parser.add_argument('--category', default='Industrial_and_Scientific')
    parser.add_argument('--vq_method', default='rqkmeans')
    parser.add_argument('--artifact_root', default='artifacts')
    parser.add_argument('--top_m', type=int, default=200)
    parser.add_argument('--limit_samples', type=int, default=None)
    parser.add_argument('--sasrec_checkpoint', default=None)
    parser.add_argument('--llmsrec_root', default='colagr/teacher/llmsrec_sasrec')
    parser.add_argument('--sasrec_device', default='cpu')
    parser.add_argument('--export_sasrec_data', action='store_true')
    parser.add_argument('--skip_train', action='store_true')
    parser.add_argument('--quiet_progress', action='store_true')
    args, unknown = parser.parse_known_args()

    base_dir = os.path.join(args.artifact_root, args.category)
    sid_dir = os.path.join(base_dir, args.vq_method)
    teacher_dir = os.path.join(base_dir, 'teacher')
    copref_dir = os.path.join(base_dir, 'copref')
    python = sys.executable

    common = [f'--dataset={args.dataset}', f'--category={args.category}']
    run([
        python, 'colagr/copref/export_sid_artifacts_latte.py',
        '--model=CoLaGR',
        *common,
        f'--vq_method={args.vq_method}',
        f'--output_dir={sid_dir}',
        *unknown,
    ])
    if args.export_sasrec_data:
        run([
            python, 'colagr/teacher/export_latte_sasrec_data.py',
            *common,
            f'--sasrec_dataset_name={args.category}',
            f'--output_dir={args.llmsrec_root}/..',
            *unknown,
        ])
    teacher_cmd = [
        python, 'colagr/teacher/export_topm_sasrec_latte.py',
        *common,
        f'--output_dir={teacher_dir}',
        f'--top_m={args.top_m}',
        f'--llmsrec_root={args.llmsrec_root}',
        f'--device={args.sasrec_device}',
        '--splits=train,val,test',
        *unknown,
    ]
    if args.sasrec_checkpoint is not None:
        teacher_cmd.append(f'--checkpoint={args.sasrec_checkpoint}')
    if args.limit_samples is not None:
        teacher_cmd.append(f'--limit_samples={args.limit_samples}')
    if args.quiet_progress:
        teacher_cmd.append('--quiet_progress')
    run(teacher_cmd)
    copref_cmd = [
        python, 'colagr/copref/build_copref_latte.py',
        f'--artifacts_dir={sid_dir}',
        f'--teacher_dir={teacher_dir}',
        f'--output_dir={copref_dir}',
    ]
    if args.quiet_progress:
        copref_cmd.append('--quiet_progress')
    run(copref_cmd)

    if not args.skip_train:
        run([
            python, 'main.py',
            '--model=CoLaGR',
            *common,
            f'--vq_method={args.vq_method}',
            f'--level_token_ids_path={sid_dir}/level_token_ids.pt',
            f'--copref_train_path={copref_dir}/copref_train.pt',
            f'--copref_val_path={copref_dir}/copref_val.pt',
            f'--copref_test_path={copref_dir}/copref_test.pt',
            *unknown,
        ])


if __name__ == '__main__':
    main()

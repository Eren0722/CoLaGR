import argparse
import os

from colagr.common import ensure_dir, load_dataset_and_tokenizer, parse_unknown_args


def write_lines(path, lines):
    with open(path, 'w') as file:
        for user_id, item_id in lines:
            file.write(f'{user_id} {item_id}\n')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='AmazonReviews2023')
    parser.add_argument('--category', required=True)
    parser.add_argument('--sasrec_dataset_name', default=None)
    parser.add_argument('--output_dir', default='colagr/teacher')
    args, unknown = parser.parse_known_args()

    overrides = parse_unknown_args(unknown)
    overrides.update({'category': args.category})
    _, dataset, split_datasets, _ = load_dataset_and_tokenizer('CoLaGR', args.dataset, overrides)

    sasrec_name = args.sasrec_dataset_name or args.category
    output_dir = os.path.abspath(os.path.join(args.output_dir, f'data_{sasrec_name}'))
    ensure_dir(output_dir)

    train_lines, valid_lines, test_lines = [], [], []
    for example in split_datasets['train']:
        user_id = dataset.user2id[example['user']]
        train_lines.extend((user_id, dataset.item2id[item]) for item in example['item_seq'])
    for example in split_datasets['val']:
        valid_lines.append((dataset.user2id[example['user']], dataset.item2id[example['item_seq'][-1]]))
    for example in split_datasets['test']:
        test_lines.append((dataset.user2id[example['user']], dataset.item2id[example['item_seq'][-1]]))

    write_lines(os.path.join(output_dir, f'{sasrec_name}_train.txt'), train_lines)
    write_lines(os.path.join(output_dir, f'{sasrec_name}_valid.txt'), valid_lines)
    write_lines(os.path.join(output_dir, f'{sasrec_name}_test.txt'), test_lines)
    print(f'Exported LLM-SRec SASRec files to {output_dir}')
    print(f'train={len(train_lines)} valid={len(valid_lines)} test={len(test_lines)}')


if __name__ == '__main__':
    main()

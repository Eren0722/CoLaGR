import inspect
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from genrec.models.CoLaGR.model import CoLaGR
from genrec.models.CoLaGR.trainer import CoLaGRTrainer
from colagr.common import split_examples
from colagr.copref import build_copref_latte
from colagr.teacher import export_topm_sasrec_latte


class DummyTokenizer:
    vocab_size = 18
    padding_token = 0
    eos_token = 14
    max_token_seq_len = 8
    n_digit = 3
    codebook_sizes = [4, 4, 4]
    coreason_token_ids = [15, 16, 17]


class DummyDataset:
    pass


def check(condition, message):
    if not condition:
        raise AssertionError(message)
    print(f'OK: {message}')


def check_copref_prefix_filter():
    source = inspect.getsource(build_copref_latte.build_record)
    check('target_prefix = target_sid[:level]' in source, 'CoPref filters by target_sid[:level]')
    check('target_sid[:level + 1]' not in source, 'CoPref does not filter by target_sid[:level+1]')


def check_topm_history_protocol():
    examples = list(split_examples([{'user': 'u1', 'item_seq': ['a', 'b', 'c']}], 'train'))
    check(examples[0]['history_items'] == ['a'], 'train sample 0 history is before target only')
    check(examples[0]['target_item'] == 'b', 'train sample 0 target follows history')
    check(examples[1]['history_items'] == ['a', 'b'], 'train sample 1 history is before target only')
    check(examples[1]['target_item'] == 'c', 'train sample 1 target follows history')

    source = inspect.getsource(export_topm_sasrec_latte.main)
    check('top_items.append' not in source, 'teacher exporter does not forcibly append target to top_items')
    check('insert' not in source, 'teacher exporter does not forcibly insert target into top-M')


def check_teacher_free_generation():
    config = {
        'num_layers': 1,
        'num_decoder_layers': 1,
        'd_model': 24,
        'd_ff': 32,
        'num_heads': 4,
        'd_kv': 8,
        'dropout_rate': 0.0,
        'activation_function': 'relu',
        'feed_forward_proj': 'relu',
        'level_token_ids_path': None,
        'valid_prefix_trie_path': None,
        'use_prefix_trie': False,
        'fuse_gate_init': -2.0,
        'lambda_c': 0.10,
        'num_beams': 1,
        'use_copref_loss': True,
        'use_cofuse': True,
    }
    model = CoLaGR(config, DummyDataset(), DummyTokenizer())
    batch = {
        'input_ids': torch.tensor([[13, 1, 5, 9, 14, 0, 0, 0]]),
        'attention_mask': torch.tensor([[1, 1, 1, 1, 1, 0, 0, 0]]),
        'coprefs': [torch.ones(1, 4) / 4 for _ in range(3)],
    }
    try:
        model.generate(batch)
    except ValueError:
        print('OK: generate rejects coprefs')
    else:
        raise AssertionError('generate must reject coprefs')


def check_trainer_eval_protocol():
    source = inspect.getsource(CoLaGRTrainer.evaluate)
    check("batch.pop('coprefs', None)" in source, 'evaluation strips coprefs before generate')
    check('export_topm_sasrec' not in source, 'evaluation does not call teacher exporter')
    check('SASRec' not in source, 'evaluation does not call CF teacher')

    init_source = inspect.getsource(CoLaGRTrainer.__init__)
    check('load_eval_copref_diagnostics' in init_source, 'eval CoPref loading is diagnostics-gated')
    check(
        "'test': self._load_copref(config.get('copref_test_path')) if load_eval_copref_diagnostics else None" in init_source,
        'test CoPref is not unconditionally loaded',
    )


def main():
    check((REPO_ROOT / 'colagr' / 'teacher' / 'llmsrec_sasrec' / 'model.py').exists(), 'vendored SASRec exists')
    check_copref_prefix_filter()
    check_topm_history_protocol()
    check_teacher_free_generation()
    check_trainer_eval_protocol()
    print('All CoLaGR leakage protocol checks passed.')


if __name__ == '__main__':
    main()

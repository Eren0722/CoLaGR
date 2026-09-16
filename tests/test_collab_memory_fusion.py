from collections import defaultdict

import pytest
from colagr.eval.collab_memory_fusion_diagnostic import residual_leaf_scores


def test_residual_uses_catalog_parent_mean_and_keeps_unseen_candidates():
    item2tokens = {
        'a': (1, 2, 3),
        'b': (1, 2, 4),
        'c': (1, 2, 5),
        'd': (9, 9, 9),
    }
    parent_to_items = defaultdict(list)
    for item, tokens in item2tokens.items():
        parent_to_items[tokens[:-1]].append(item)

    # Only a and d are in the sparse memory. b and c must still get residuals.
    scores = {'a': 1.0, 'd': 0.5}
    residual = residual_leaf_scores(
        scores,
        item2tokens,
        parent_to_items,
        candidate_items=['a', 'b', 'c', 'd'],
    )

    # Parent (1, 2) mean is (1 + 0 + 0) / 3, not 1 / 1.
    assert residual['a'] == pytest.approx(2.0 / 3.0)
    assert residual['b'] == pytest.approx(-1.0 / 3.0)
    assert residual['c'] == pytest.approx(-1.0 / 3.0)
    assert residual['d'] == pytest.approx(0.0)


def test_residual_can_cover_all_catalog_items_by_default():
    item2tokens = {'a': (1, 2), 'b': (1, 3)}
    parent_to_items = defaultdict(list, { (1,): ['a', 'b'] })
    residual = residual_leaf_scores(
        {'a': 1.0},
        item2tokens,
        parent_to_items,
    )
    assert set(residual) == {'a', 'b'}
    assert residual['a'] == pytest.approx(0.5)
    assert residual['b'] == pytest.approx(-0.5)

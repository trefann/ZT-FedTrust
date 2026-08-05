"""Tests for RealAttackTrafficAttack: dataset-level, trains on real attack traffic."""

import pytest
from torch.utils.data import TensorDataset
import torch

from attacks.real_traffic import RealAttackTrafficAttack


def _pool(size=20):
    features = torch.randn(size, 4)
    labels = torch.ones(size, dtype=torch.int64)  # all "attack"-labeled
    return TensorDataset(features, labels)


def _client_dataset(size=8):
    features = torch.zeros(size, 4)
    labels = torch.zeros(size, dtype=torch.int64)
    return TensorDataset(features, labels)


def test_empty_pool_raises_value_error():
    empty = TensorDataset(torch.empty(0, 4), torch.empty(0, dtype=torch.int64))
    with pytest.raises(ValueError):
        RealAttackTrafficAttack(attack_pool=empty)


def test_apply_to_dataset_returns_pool_sized_sample():
    attack = RealAttackTrafficAttack(attack_pool=_pool(20), seed=1)
    result = attack.apply_to_dataset(_client_dataset(8))
    assert len(result) == 8


def test_apply_to_dataset_never_exceeds_pool_size():
    attack = RealAttackTrafficAttack(attack_pool=_pool(5), seed=1)
    result = attack.apply_to_dataset(_client_dataset(8))
    assert len(result) == 5


def test_apply_to_dataset_draws_from_pool_not_original():
    attack = RealAttackTrafficAttack(attack_pool=_pool(20), seed=1)
    result = attack.apply_to_dataset(_client_dataset(8))
    _, label = result[0]
    assert label.item() == 1  # pool's label, not the client's original 0


def test_apply_to_update_is_default_noop():
    attack = RealAttackTrafficAttack(attack_pool=_pool(20), seed=1)
    state = {"w": torch.tensor([1.0, 2.0])}
    reference = {"w": torch.tensor([0.0, 0.0])}
    assert attack.apply_to_update(state, reference) is state

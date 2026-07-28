"""Unit tests for FedAvg aggregation correctness."""

import pytest
import torch

from server.fedavg import federated_average


def test_federated_average_weighted_correctly():
    state_a = {"w": torch.tensor([1.0, 2.0])}
    state_b = {"w": torch.tensor([3.0, 4.0])}
    result = federated_average([(state_a, 1), (state_b, 3)])
    expected = (state_a["w"] * 1 + state_b["w"] * 3) / 4
    assert torch.allclose(result["w"], expected)


def test_federated_average_equal_weights_is_plain_mean():
    state_a = {"w": torch.tensor([2.0])}
    state_b = {"w": torch.tensor([6.0])}
    result = federated_average([(state_a, 10), (state_b, 10)])
    assert torch.allclose(result["w"], torch.tensor([4.0]))


def test_federated_average_single_update_is_unchanged():
    state_a = {"w": torch.tensor([5.0, -1.0])}
    result = federated_average([(state_a, 7)])
    assert torch.allclose(result["w"], state_a["w"])


def test_federated_average_empty_list_raises():
    with pytest.raises(ValueError):
        federated_average([])


def test_federated_average_zero_total_samples_raises():
    state_a = {"w": torch.tensor([1.0])}
    with pytest.raises(ValueError):
        federated_average([(state_a, 0)])

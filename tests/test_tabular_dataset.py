"""Tests for TabularDataset: the torch.utils.data.Dataset wrapper for tabular IoT data."""

import numpy as np
import pytest
import torch

from data.tabular_dataset import TabularDataset


def _make_dataset(num_samples=10, num_features=4):
    features = np.arange(num_samples * num_features, dtype=np.float64).reshape(
        num_samples, num_features
    )
    labels = np.array([i % 3 for i in range(num_samples)])
    return TabularDataset(features, labels)


def test_len_matches_number_of_rows():
    dataset = _make_dataset(num_samples=7)
    assert len(dataset) == 7


def test_getitem_returns_float_tensor_and_int_label():
    dataset = _make_dataset(num_samples=5, num_features=3)
    features, label = dataset[0]
    assert isinstance(features, torch.Tensor)
    assert features.dtype == torch.float32
    assert features.shape == (3,)
    assert isinstance(label, torch.Tensor)
    assert label.dtype == torch.int64


def test_targets_attribute_matches_labels():
    dataset = _make_dataset(num_samples=6)
    labels = [i % 3 for i in range(6)]
    assert dataset.targets == labels


def test_mismatched_lengths_raise_value_error():
    features = np.zeros((5, 2))
    labels = np.zeros(4)
    with pytest.raises(ValueError):
        TabularDataset(features, labels)

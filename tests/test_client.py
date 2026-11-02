"""Tests for standalone Client local training and the get/set_weights API."""

import copy

import torch
from torch.utils.data import TensorDataset

from clients.client import Client
from helpers import make_synthetic_dataset
from models.cnn import build_model


def _make_client(num_samples: int = 64, batch_size: int = 16, local_epochs: int = 1) -> Client:
    dataset = make_synthetic_dataset(num_samples=num_samples, seed=7)
    model = build_model("cnn", in_channels=1, num_classes=10)
    return Client(
        client_id=0,
        model_template=model,
        dataset=dataset,
        device=torch.device("cpu"),
        batch_size=batch_size,
        learning_rate=0.05,
        momentum=0.9,
        local_epochs=local_epochs,
    )


def test_standalone_client_training_runs_and_returns_metrics():
    client = _make_client(num_samples=64)
    result = client.train()
    assert result.client_id == 0
    assert result.num_samples == 64
    assert result.average_loss > 0
    assert set(result.state_dict.keys()) == set(client.model.state_dict().keys())


def test_get_set_weights_roundtrip():
    client = _make_client()
    original = client.get_weights()

    zeroed = {key: torch.zeros_like(value) for key, value in original.items()}
    client.set_weights(zeroed)
    for key, value in client.get_weights().items():
        assert torch.equal(value, zeroed[key])

    client.set_weights(original)
    for key, value in client.get_weights().items():
        assert torch.equal(value, original[key])


def test_training_changes_weights():
    client = _make_client(local_epochs=2)
    before = copy.deepcopy(client.get_weights())
    client.train()
    after = client.get_weights()
    assert any(not torch.equal(before[key], after[key]) for key in before)


def _make_tabular_dataset(num_samples: int, num_features: int = 32, seed: int = 0) -> TensorDataset:
    """Build a small tabular (non-image) dataset for TabularMLP testing."""
    generator = torch.Generator().manual_seed(seed)
    labels = torch.randint(0, 10, (num_samples,), generator=generator)
    # Features: use label offsets so the model can learn the task
    features = torch.randn(num_samples, num_features, generator=generator) * 0.5
    offsets = (labels.float() / 10).view(-1, 1).expand(-1, num_features)
    features = features + offsets
    return TensorDataset(features, labels)


def test_drop_last_on_size_1_remainder_vision_model():
    """Verify that training with size-1 remainder uses drop_last correctly on vision models.

    When len(dataset) % batch_size == 1 and len(dataset) > batch_size,
    the final batch of size 1 should be dropped. This test verifies that
    SimpleCNN (which has no BatchNorm) trains correctly with this behavior.
    """
    # 17 samples, 16 batch_size → 17 % 16 = 1 remainder, so drop_last should be True
    dataset = make_synthetic_dataset(num_samples=17, seed=42)
    model = build_model("cnn", in_channels=1, num_classes=10)
    client = Client(
        client_id=0,
        model_template=model,
        dataset=dataset,
        device=torch.device("cpu"),
        batch_size=16,
        learning_rate=0.05,
        momentum=0.9,
        local_epochs=1,
    )

    # Count forward passes to verify only 1 batch was trained (not 2)
    forward_count = [0]

    def count_forward(module, input, output):
        forward_count[0] += 1

    # Register hook on first layer to count forward passes (one per batch)
    hook = client.model.conv1.register_forward_hook(count_forward)
    try:
        result = client.train()
        # With drop_last=True on 17 samples with batch_size=16,
        # exactly 1 batch (of 16 samples) should be trained per epoch
        assert forward_count[0] == 1, f"Expected 1 batch forward pass, got {forward_count[0]}"
        assert result.num_samples == 17
        assert result.average_loss > 0
    finally:
        hook.remove()


def test_drop_last_on_size_1_remainder_tabular_model():
    """Verify that TabularMLP (with BatchNorm1d) doesn't crash on size-1 remainder.

    TabularMLP uses BatchNorm1d which raises "Expected more than 1 value per
    channel when training" if a batch has size 1. The drop_last logic should
    prevent this crash by not training on the final size-1 batch when a full
    batch still exists in the dataset.
    """
    # 17 samples, 16 batch_size → drop_last should be True, preventing BatchNorm crash
    dataset = _make_tabular_dataset(num_samples=17, num_features=32, seed=42)
    model = build_model("mlp", in_channels=32, num_classes=10)
    client = Client(
        client_id=0,
        model_template=model,
        dataset=dataset,
        device=torch.device("cpu"),
        batch_size=16,
        learning_rate=0.05,
        momentum=0.9,
        local_epochs=1,
    )

    # Count forward passes to verify drop_last behavior
    forward_count = [0]

    def count_forward(module, input, output):
        forward_count[0] += 1

    # Register hook on first layer to count forward passes (one per batch)
    hook = client.model.fc1.register_forward_hook(count_forward)
    try:
        # This would have raised BatchNorm1d error without the drop_last fix
        result = client.train()
        # With drop_last=True on 17 samples with batch_size=16,
        # exactly 1 batch (of 16 samples) should be trained per epoch
        assert forward_count[0] == 1, f"Expected 1 batch forward pass, got {forward_count[0]}"
        assert result.num_samples == 17
        assert result.average_loss > 0
    finally:
        hook.remove()


def test_class_weights_change_the_loss_for_an_imbalanced_batch():
    torch.manual_seed(0)
    dataset = _make_tabular_dataset(num_samples=64, num_features=10, seed=1)
    model_unweighted = build_model("mlp", in_channels=10, num_classes=10)
    model_weighted = build_model("mlp", in_channels=10, num_classes=10)
    model_weighted.load_state_dict(model_unweighted.state_dict())

    class_weights = torch.ones(10)
    class_weights[0] = 5.0  # heavily up-weight class 0

    client_unweighted = Client(
        client_id=0, model_template=model_unweighted, dataset=dataset, device=torch.device("cpu"),
        batch_size=16, learning_rate=0.01, momentum=0.9, local_epochs=1,
    )
    client_weighted = Client(
        client_id=1, model_template=model_weighted, dataset=dataset, device=torch.device("cpu"),
        batch_size=16, learning_rate=0.01, momentum=0.9, local_epochs=1, class_weights=class_weights,
    )

    torch.manual_seed(1)
    result_unweighted = client_unweighted.train()
    torch.manual_seed(1)
    result_weighted = client_weighted.train()

    assert result_unweighted.average_loss != result_weighted.average_loss

"""Tests for standalone Client local training and the get/set_weights API."""

import copy

import torch

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

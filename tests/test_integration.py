"""End-to-end pipeline tests on synthetic data (no network download required).

Exercises the same partition -> Client -> Server -> FedAvg -> evaluate flow
that `simulation/run.py` drives against real datasets, but against a small
synthetic dataset so the suite stays fast and offline.
"""

import pytest
import torch

from clients.client import Client
from data.partition import make_client_subsets, partition_iid
from helpers import make_synthetic_dataset
from models.cnn import build_model
from server.server import Server
from utils.metrics import set_seed


def _run_tiny_simulation(num_clients: int, num_rounds: int, clients_per_round: int, seed: int = 42):
    set_seed(seed)
    device = torch.device("cpu")
    train_dataset = make_synthetic_dataset(num_samples=200, num_classes=10, seed=seed)
    test_dataset = make_synthetic_dataset(num_samples=60, num_classes=10, seed=seed + 1)

    shards = partition_iid(train_dataset, num_clients=num_clients, seed=seed)
    subsets = make_client_subsets(train_dataset, shards)

    global_model = build_model("cnn", in_channels=1, num_classes=10)
    clients = {
        client_id: Client(
            client_id=client_id,
            model_template=global_model,
            dataset=subset,
            device=device,
            batch_size=16,
            learning_rate=0.1,
            momentum=0.9,
            local_epochs=2,
        )
        for client_id, subset in subsets.items()
    }
    server = Server(
        global_model, clients, test_dataset, device, clients_per_round=clients_per_round, seed=seed
    )

    return [server.run_round(round_num) for round_num in range(1, num_rounds + 1)]


@pytest.mark.parametrize("num_clients", [2, 4])
def test_runs_without_errors_for_multiple_client_counts(num_clients):
    history = _run_tiny_simulation(num_clients=num_clients, num_rounds=2, clients_per_round=min(2, num_clients))
    assert len(history) == 2


@pytest.mark.parametrize("num_rounds", [1, 4])
def test_runs_without_errors_for_multiple_round_counts(num_rounds):
    history = _run_tiny_simulation(num_clients=4, num_rounds=num_rounds, clients_per_round=2)
    assert len(history) == num_rounds


def test_deterministic_runs_with_same_seed():
    history_a = _run_tiny_simulation(num_clients=4, num_rounds=3, clients_per_round=2, seed=123)
    history_b = _run_tiny_simulation(num_clients=4, num_rounds=3, clients_per_round=2, seed=123)
    for round_a, round_b in zip(history_a, history_b):
        assert round_a["train_loss"] == pytest.approx(round_b["train_loss"])
        assert round_a["test_accuracy"] == pytest.approx(round_b["test_accuracy"])


def test_global_accuracy_improves_over_rounds():
    history = _run_tiny_simulation(num_clients=4, num_rounds=6, clients_per_round=3, seed=7)
    assert history[-1]["test_accuracy"] > history[0]["test_accuracy"]

"""End-to-end test: a full federated run on synthetic IoT-shaped tabular data."""

from unittest.mock import patch

import numpy as np
import pytest

from data.iot_datasets import IoTExperimentData
from data.partition import partition_iid, make_client_subsets
from data.tabular_dataset import TabularDataset
from simulation.run import DataConfig, FederatedConfig, LoggingConfig, ModelConfig, SimulationConfig, run_simulation


def _synthetic_iot_experiment_data(seed: int) -> IoTExperimentData:
    rng = np.random.default_rng(seed)
    num_samples = 200
    num_features = 6
    labels = rng.integers(0, 3, size=num_samples)
    offsets = labels.astype(np.float32).reshape(-1, 1) * 0.5
    features = (offsets + rng.normal(0, 0.3, size=(num_samples, num_features))).astype(np.float32)

    full_dataset = TabularDataset(features, labels)
    client_indices = partition_iid(full_dataset, num_clients=6, seed=seed)
    train_client_datasets = make_client_subsets(full_dataset, client_indices)

    test_labels = rng.integers(0, 3, size=40)
    test_offsets = test_labels.astype(np.float32).reshape(-1, 1) * 0.5
    test_features = (test_offsets + rng.normal(0, 0.3, size=(40, num_features))).astype(np.float32)
    test_dataset = TabularDataset(test_features, test_labels)

    attack_labels = np.ones(30, dtype=np.int64)
    attack_features = (0.5 + rng.normal(0, 0.3, size=(30, num_features))).astype(np.float32)
    attack_pool = TabularDataset(attack_features, attack_labels)

    return IoTExperimentData(
        train_client_datasets=train_client_datasets,
        test_dataset=test_dataset,
        attack_pool=attack_pool,
        input_dim=num_features,
        num_classes=3,
        class_names=["normal", "attack_a", "attack_b"],
    )


def _iot_config() -> SimulationConfig:
    return SimulationConfig(
        seed=7,
        device="cpu",
        data=DataConfig(
            dataset="ciciot2023", data_dir="./unused", num_clients=6, partition_type="iid",
            dirichlet_alpha=0.5, batch_size=16, max_rows=None,
        ),
        model=ModelConfig(name="mlp"),
        federated=FederatedConfig(
            num_rounds=2, clients_per_round=4, local_epochs=1, learning_rate=0.05, momentum=0.9
        ),
        logging=LoggingConfig(log_dir="./results/test_iot/logs", log_level="INFO", results_dir="./results/test_iot"),
    )


def test_full_simulation_runs_on_synthetic_iot_data(tmp_path):
    config = _iot_config()
    config.logging.log_dir = str(tmp_path / "logs")
    config.logging.results_dir = str(tmp_path / "results")
    config.attacks.enabled = True
    config.attacks.active_attack = "real_traffic"
    config.attacks.malicious_client_fraction = 0.34  # 2 of 6 clients

    with patch("simulation.run.load_iot_experiment", return_value=_synthetic_iot_experiment_data(config.seed)):
        history = run_simulation(config)

    assert config.trust.enabled is True
    assert config.attacks.malicious_client_fraction > 0.0
    assert len(history) == config.federated.num_rounds
    for round_record in history:
        assert "test_accuracy" in round_record
        assert 0.0 <= round_record["test_accuracy"] <= 1.0


def test_full_simulation_runs_without_attacks(tmp_path):
    config = _iot_config()
    config.logging.log_dir = str(tmp_path / "logs")
    config.logging.results_dir = str(tmp_path / "results")

    with patch("simulation.run.load_iot_experiment", return_value=_synthetic_iot_experiment_data(config.seed)):
        history = run_simulation(config)

    assert len(history) == config.federated.num_rounds

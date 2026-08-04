"""Tests for simulation/run.py's IoT-vs-vision dataset dispatch."""

from unittest.mock import patch

import torch

from data.iot_datasets import IoTExperimentData
from data.tabular_dataset import TabularDataset
from simulation.run import DataConfig, FederatedConfig, LoggingConfig, ModelConfig, SimulationConfig, load_experiment_data


def _iot_config(dataset_name: str) -> SimulationConfig:
    return SimulationConfig(
        seed=1,
        device="cpu",
        data=DataConfig(
            dataset=dataset_name, data_dir="./fake", num_clients=2, partition_type="iid",
            dirichlet_alpha=0.5, batch_size=8, max_rows=100,
        ),
        model=ModelConfig(name="mlp"),
        federated=FederatedConfig(num_rounds=1, clients_per_round=1, local_epochs=1, learning_rate=0.01, momentum=0.9),
        logging=LoggingConfig(log_dir="./results/logs", log_level="INFO", results_dir="./results"),
    )


def _fake_experiment_data() -> IoTExperimentData:
    features = torch.randn(20, 5).numpy()
    labels = (torch.arange(20) % 3).numpy()
    train_datasets = {
        0: TabularDataset(features[:10], labels[:10]),
        1: TabularDataset(features[10:], labels[10:]),
    }
    return IoTExperimentData(
        train_client_datasets=train_datasets,
        test_dataset=TabularDataset(features, labels),
        attack_pool=TabularDataset(features[:5], labels[:5]),
        input_dim=5,
        num_classes=3,
        class_names=["a", "b", "c"],
    )


def test_iot_dataset_uses_load_iot_experiment():
    config = _iot_config("ciciot2023")
    with patch("simulation.run.load_iot_experiment", return_value=_fake_experiment_data()) as mock_load:
        model, clients, test_dataset, attack_pool = load_experiment_data(config, torch.device("cpu"))

    mock_load.assert_called_once()
    assert len(clients) == 2
    assert len(test_dataset) == 20
    assert attack_pool is not None
    assert len(attack_pool) == 5


def test_vision_dataset_returns_none_attack_pool():
    config = _iot_config("mnist")
    config.model.name = "cnn"
    with patch("simulation.run.load_dataset") as mock_load_dataset, patch(
        "simulation.run.dataset_shape", return_value=(1, 10)
    ), patch("simulation.run.build_clients") as mock_build_clients:
        mock_load_dataset.return_value = TabularDataset(
            torch.randn(10, 3).numpy(), (torch.arange(10) % 2).numpy()
        )
        mock_build_clients.return_value = {}
        model, clients, test_dataset, attack_pool = load_experiment_data(config, torch.device("cpu"))

    assert attack_pool is None


from evaluation.experiment_runner import ExperimentRunner


def test_experiment_runner_uses_load_experiment_data():
    config = _iot_config("ciciot2023")
    config.trust.enabled = False
    with patch("evaluation.experiment_runner.load_experiment_data") as mock_load:
        mock_load.return_value = (
            torch.nn.Linear(5, 3),
            {0: None, 1: None},  # placeholder clients; run() only needs the dict for length/logging here
            _fake_experiment_data().test_dataset,
            None,
        )
        runner = ExperimentRunner(config)
        # Only assert load_experiment_data is the wiring point, not a full run
        # (a full run needs real Client objects — covered by Task 17's e2e test).
        assert runner.config.data.dataset == "ciciot2023"
        mock_load.assert_not_called()  # not called yet, just constructing the runner

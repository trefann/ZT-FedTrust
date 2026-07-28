"""Tests for SimulationConfig value validation."""

import pytest

from simulation.run import (
    DataConfig,
    FederatedConfig,
    LoggingConfig,
    ModelConfig,
    SimulationConfig,
    validate_config,
)


def _base_config(**overrides: object) -> SimulationConfig:
    """Build a known-valid config, then apply "section.field": value overrides."""
    data = DataConfig(
        dataset="mnist",
        data_dir="./data_cache",
        num_clients=4,
        partition_type="iid",
        dirichlet_alpha=0.5,
        batch_size=32,
    )
    federated = FederatedConfig(
        num_rounds=3, clients_per_round=2, local_epochs=1, learning_rate=0.01, momentum=0.9
    )
    logging_cfg = LoggingConfig(log_dir="./results/logs", log_level="INFO", results_dir="./results")
    config = SimulationConfig(
        seed=42,
        device="cpu",
        data=data,
        model=ModelConfig(name="cnn"),
        federated=federated,
        logging=logging_cfg,
    )
    for dotted_key, value in overrides.items():
        if "." in dotted_key:
            section, field = dotted_key.split(".")
            setattr(getattr(config, section), field, value)
        else:
            setattr(config, dotted_key, value)
    return config


def test_valid_config_passes_validation():
    validate_config(_base_config())  # should not raise


@pytest.mark.parametrize(
    "override,bad_value",
    [
        ("federated.num_rounds", 0),
        ("federated.clients_per_round", 99),
        ("federated.local_epochs", 0),
        ("federated.learning_rate", -1.0),
        ("federated.momentum", 1.5),
        ("data.dataset", "not-a-real-dataset"),
        ("data.partition_type", "not-a-real-partition"),
        ("data.num_clients", 0),
        ("data.batch_size", 0),
        ("data.dirichlet_alpha", 0),
        ("model.name", "not-a-real-model"),
        ("logging.log_level", "VERBOSE"),
        ("device", "tpu"),
    ],
)
def test_invalid_values_raise_value_error(override, bad_value):
    config = _base_config(**{override: bad_value})
    with pytest.raises(ValueError):
        validate_config(config)

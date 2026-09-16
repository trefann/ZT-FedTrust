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


def test_iot_dataset_name_passes_validation():
    validate_config(_base_config(**{"data.dataset": "ciciot2023"}))  # should not raise


def test_max_rows_defaults_to_none():
    config = _base_config()
    assert config.data.max_rows is None


def test_real_traffic_attack_on_ciciot2023_fails_validation():
    """CICIoT2023 has no benign/attack split (see `load_ciciot2023`'s
    docstring in `data/iot_datasets.py`), so `real_traffic` must be
    rejected up front rather than failing deep inside `build_attack`."""
    config = _base_config(**{
        "data.dataset": "ciciot2023",
        "attacks.active_attack": "real_traffic",
    })
    with pytest.raises(ValueError):
        validate_config(config)


def test_real_traffic_attack_on_unsw_nb15_passes_validation():
    """UNSW-NB15 does have a benign/attack split, so `real_traffic` is valid."""
    config = _base_config(**{
        "data.dataset": "unsw_nb15",
        "attacks.active_attack": "real_traffic",
    })
    validate_config(config)  # should not raise


def test_real_traffic_probability_defaults_to_one():
    config = _base_config()
    assert config.attacks.real_traffic.probability == 1.0


def test_real_traffic_invalid_probability_raises_value_error():
    config = _base_config(**{
        "data.dataset": "unsw_nb15",
        "attacks.active_attack": "real_traffic",
    })
    config.attacks.real_traffic.probability = 1.5
    with pytest.raises(ValueError):
        validate_config(config)


def test_model_hidden_sizes_defaults_to_none():
    config = _base_config()
    assert config.model.hidden_sizes is None


def test_training_class_weighted_loss_defaults_to_false():
    config = _base_config()
    assert config.training.class_weighted_loss is False


def test_negative_hidden_size_raises():
    config = _base_config()
    config.model.hidden_sizes = (128, -1)
    with pytest.raises(ValueError):
        validate_config(config)


def test_data_binary_labels_defaults_to_false():
    config = _base_config()
    assert config.data.binary_labels is False

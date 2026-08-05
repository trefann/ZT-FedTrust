"""Smoke tests: every IoT config parses and validates (no data loading, no training)."""

import pytest
import yaml

from simulation.run import load_config, validate_config

_IOT_CONFIGS = [
    "configs/iot/ciciot2023.yaml",
    "configs/iot/unsw_nb15.yaml",
    "configs/iot/ton_iot_network.yaml",
    "configs/iot/ton_iot_devices.yaml",
    "configs/iot/nbaiot.yaml",
]


@pytest.mark.parametrize("config_path", _IOT_CONFIGS)
def test_config_loads_and_validates(config_path):
    config = load_config(config_path)
    validate_config(config)  # should not raise


def test_botiot_config_dataset_name_is_botiot():
    """Verify botiot config's dataset field is set correctly.
    Uses direct YAML load since load_config auto-validates."""
    with open("configs/iot/botiot.yaml", "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    assert raw["data"]["dataset"] == "botiot"


def test_botiot_config_fails_validation():
    """botiot is a disabled placeholder — validate_config rejects it up front
    (it's not in IOT_SUPPORTED_DATASETS), rather than failing deep inside a run."""
    with pytest.raises(ValueError, match="data.dataset"):
        load_config("configs/iot/botiot.yaml")

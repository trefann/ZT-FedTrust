"""Entry point for running a federated learning simulation from a YAML config."""

from __future__ import annotations

import argparse
import csv
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

import torch
import yaml

# Allow running as a standalone script (`python simulation/run.py`) by putting
# the project root on sys.path so sibling top-level packages are importable.
sys.path.append(str(Path(__file__).resolve().parent.parent))

from clients.client import Client
from data.dataset import SUPPORTED_DATASETS, dataset_shape, load_dataset
from data.partition import (
    SUPPORTED_PARTITION_TYPES,
    make_client_subsets,
    partition_dirichlet,
    partition_iid,
)
from models.cnn import available_models, build_model
from server.server import Server
from utils.logger import setup_logger
from utils.metrics import set_seed
from utils.visualization import save_plots

_VALID_LOG_LEVELS = ("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL")
_VALID_DEVICES = ("cpu", "cuda")


@dataclass
class DataConfig:
    dataset: str
    data_dir: str
    num_clients: int
    partition_type: str
    dirichlet_alpha: float
    batch_size: int


@dataclass
class ModelConfig:
    name: str


@dataclass
class FederatedConfig:
    num_rounds: int
    clients_per_round: int
    local_epochs: int
    learning_rate: float
    momentum: float


@dataclass
class LoggingConfig:
    log_dir: str
    log_level: str
    results_dir: str


@dataclass
class SimulationConfig:
    seed: int
    device: str
    data: DataConfig
    model: ModelConfig
    federated: FederatedConfig
    logging: LoggingConfig


def validate_config(config: SimulationConfig) -> None:
    """Raise `ValueError` if any config field is out of its valid range/set.

    Kept separate from `load_config` so it can also be run against a
    programmatically-constructed `SimulationConfig` (e.g. in tests).
    """
    errors: List[str] = []

    if config.seed < 0:
        errors.append(f"seed must be >= 0, got {config.seed}")
    if config.device not in _VALID_DEVICES:
        errors.append(f"device must be one of {_VALID_DEVICES}, got {config.device!r}")

    if config.data.dataset not in SUPPORTED_DATASETS:
        errors.append(f"data.dataset must be one of {SUPPORTED_DATASETS}, got {config.data.dataset!r}")
    if config.data.num_clients <= 0:
        errors.append(f"data.num_clients must be > 0, got {config.data.num_clients}")
    if config.data.partition_type not in SUPPORTED_PARTITION_TYPES:
        errors.append(
            f"data.partition.type must be one of {SUPPORTED_PARTITION_TYPES}, "
            f"got {config.data.partition_type!r}"
        )
    if config.data.dirichlet_alpha <= 0:
        errors.append(f"data.partition.alpha must be > 0, got {config.data.dirichlet_alpha}")
    if config.data.batch_size <= 0:
        errors.append(f"data.batch_size must be > 0, got {config.data.batch_size}")

    available = available_models()
    if config.model.name not in available:
        errors.append(f"model.name must be one of {available}, got {config.model.name!r}")

    if config.federated.num_rounds <= 0:
        errors.append(f"federated.num_rounds must be > 0, got {config.federated.num_rounds}")
    if not 1 <= config.federated.clients_per_round <= config.data.num_clients:
        errors.append(
            "federated.clients_per_round must be between 1 and data.num_clients "
            f"({config.data.num_clients}), got {config.federated.clients_per_round}"
        )
    if config.federated.local_epochs <= 0:
        errors.append(f"federated.local_epochs must be > 0, got {config.federated.local_epochs}")
    if config.federated.learning_rate <= 0:
        errors.append(f"federated.learning_rate must be > 0, got {config.federated.learning_rate}")
    if not 0 <= config.federated.momentum < 1:
        errors.append(f"federated.momentum must be in [0, 1), got {config.federated.momentum}")

    if config.logging.log_level.upper() not in _VALID_LOG_LEVELS:
        errors.append(
            f"logging.log_level must be one of {_VALID_LOG_LEVELS}, got {config.logging.log_level!r}"
        )

    if errors:
        formatted = "\n  - ".join(errors)
        raise ValueError(f"Invalid configuration:\n  - {formatted}")


def load_config(config_path: str) -> SimulationConfig:
    """Load, parse, and validate a YAML config file into a `SimulationConfig`."""
    with open(config_path, "r", encoding="utf-8") as f:
        raw: Dict[str, Any] = yaml.safe_load(f)

    data_raw = raw["data"]
    partition_raw = data_raw["partition"]

    config = SimulationConfig(
        seed=raw["seed"],
        device=raw["device"],
        data=DataConfig(
            dataset=data_raw["dataset"],
            data_dir=data_raw["data_dir"],
            num_clients=data_raw["num_clients"],
            partition_type=partition_raw["type"],
            dirichlet_alpha=partition_raw.get("alpha", 0.5),
            batch_size=data_raw["batch_size"],
        ),
        model=ModelConfig(name=raw["model"]["name"]),
        federated=FederatedConfig(
            num_rounds=raw["federated"]["num_rounds"],
            clients_per_round=raw["federated"]["clients_per_round"],
            local_epochs=raw["federated"]["local_epochs"],
            learning_rate=raw["federated"]["learning_rate"],
            momentum=raw["federated"]["momentum"],
        ),
        logging=LoggingConfig(
            log_dir=raw["logging"]["log_dir"],
            log_level=raw["logging"]["log_level"],
            results_dir=raw["logging"]["results_dir"],
        ),
    )
    validate_config(config)
    return config


def resolve_device(requested: str) -> torch.device:
    """Fall back to CPU if CUDA was requested but is unavailable."""
    if requested == "cuda" and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(requested)


def save_history(history: List[Dict[str, float]], results_dir: str) -> None:
    """Write per-round metrics to a CSV file in the results directory."""
    Path(results_dir).mkdir(parents=True, exist_ok=True)
    output_path = Path(results_dir) / "history.csv"
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(history[0].keys()))
        writer.writeheader()
        writer.writerows(history)


def build_clients(
    config: SimulationConfig, device: torch.device, model_template: torch.nn.Module
) -> Dict[int, Client]:
    """Load the training set, partition it across clients, and build `Client` objects.

    Each `Client` deep-copies `model_template` once at construction time to use
    as its own persistent local model.
    """
    train_dataset = load_dataset(config.data.dataset, config.data.data_dir, train=True)

    if config.data.partition_type == "iid":
        client_indices = partition_iid(train_dataset, config.data.num_clients, config.seed)
    elif config.data.partition_type == "dirichlet":
        client_indices = partition_dirichlet(
            train_dataset, config.data.num_clients, config.data.dirichlet_alpha, config.seed
        )
    else:
        raise ValueError(f"Unsupported partition type: {config.data.partition_type!r}")

    client_subsets = make_client_subsets(train_dataset, client_indices)
    return {
        client_id: Client(
            client_id=client_id,
            model_template=model_template,
            dataset=subset,
            device=device,
            batch_size=config.data.batch_size,
            learning_rate=config.federated.learning_rate,
            momentum=config.federated.momentum,
            local_epochs=config.federated.local_epochs,
        )
        for client_id, subset in client_subsets.items()
    }


def run_simulation(config: SimulationConfig) -> List[Dict[str, float]]:
    """Build all components from `config` and run the full federated simulation."""
    set_seed(config.seed)
    device = resolve_device(config.device)
    logger = setup_logger("simulation", config.logging.log_dir, config.logging.log_level)
    logger.info("Starting simulation on device=%s", device)
    logger.info("Config: %s", config)

    in_channels, num_classes = dataset_shape(config.data.dataset)
    test_dataset = load_dataset(config.data.dataset, config.data.data_dir, train=False)

    global_model = build_model(config.model.name, in_channels, num_classes)
    clients = build_clients(config, device, global_model)

    server = Server(
        global_model=global_model,
        clients=clients,
        test_dataset=test_dataset,
        device=device,
        clients_per_round=config.federated.clients_per_round,
        seed=config.seed,
    )

    history: List[Dict[str, float]] = []
    for round_num in range(1, config.federated.num_rounds + 1):
        history.append(server.run_round(round_num))

    save_history(history, config.logging.results_dir)
    save_plots(history, config.logging.results_dir)
    logger.info("Simulation complete. Results saved to %s", config.logging.results_dir)
    return history


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a federated learning simulation.")
    parser.add_argument(
        "--config",
        type=str,
        default=str(Path(__file__).resolve().parent.parent / "configs" / "config.yaml"),
        help="Path to the YAML configuration file.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    run_simulation(config)


if __name__ == "__main__":
    main()

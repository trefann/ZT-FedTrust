"""Entry point for running a federated learning simulation from a YAML config."""

from __future__ import annotations

import argparse
import csv
import sys
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Dict, List, Optional

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
from server.selection_metrics import (
    save_accuracy_comparison_plot,
    save_aggregation_weights_csv,
    save_selection_csv,
    save_selection_plots,
)
from server.selector import ClientSelector
from server.server import Server
from server.trust_fedavg import SUPPORTED_AGGREGATION_STRATEGIES
from trust.trust_engine import TrustEngine, TrustWeights
from trust.trust_metrics import save_trust_history, save_trust_plots, save_trust_scores
from trust.trust_policy import SUPPORTED_THRESHOLD_MODES, TrustPolicy
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
class TrustWeightsConfig:
    identity: float = 0.25
    integrity: float = 0.20
    behaviour: float = 0.20
    reputation: float = 0.15
    update_quality: float = 0.20


@dataclass
class TrustConfig:
    """Zero Trust evaluation + trust-based selection settings.

    Phase 2 introduced scoring only. Phase 3 adds `threshold_mode`,
    `threshold` (renamed from Phase 2's unused `trust_threshold`, now
    actually enforced), and `participation_fraction` to drive real
    trust-based client selection.
    """

    enabled: bool = True
    initial_reputation: float = 0.80
    reputation_alpha: float = 0.80
    threshold_mode: str = "fixed"
    threshold: float = 0.60
    participation_fraction: float = 0.50
    weights: TrustWeightsConfig = field(default_factory=TrustWeightsConfig)


@dataclass
class AggregationConfig:
    """Selects which aggregation strategy `Server` uses (see `server/trust_fedavg.py`)."""

    strategy: str = "fedavg"


@dataclass
class SimulationConfig:
    seed: int
    device: str
    data: DataConfig
    model: ModelConfig
    federated: FederatedConfig
    logging: LoggingConfig
    # Defaulted so existing call sites (and configs without a `trust:`/
    # `aggregation:` block) keep working unmodified — both are additive.
    trust: TrustConfig = field(default_factory=TrustConfig)
    aggregation: AggregationConfig = field(default_factory=AggregationConfig)


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

    if not 0.0 <= config.trust.initial_reputation <= 1.0:
        errors.append(
            f"trust.initial_reputation must be in [0, 1], got {config.trust.initial_reputation}"
        )
    if not 0.0 <= config.trust.reputation_alpha <= 1.0:
        errors.append(f"trust.reputation_alpha must be in [0, 1], got {config.trust.reputation_alpha}")
    if not 0.0 <= config.trust.threshold <= 1.0:
        errors.append(f"trust.threshold must be in [0, 1], got {config.trust.threshold}")
    if config.trust.threshold_mode not in SUPPORTED_THRESHOLD_MODES:
        errors.append(
            f"trust.threshold_mode must be one of {SUPPORTED_THRESHOLD_MODES}, "
            f"got {config.trust.threshold_mode!r}"
        )
    if not 0.0 < config.trust.participation_fraction <= 1.0:
        errors.append(
            f"trust.participation_fraction must be in (0, 1], got {config.trust.participation_fraction}"
        )
    weight_values = {
        "identity": config.trust.weights.identity,
        "integrity": config.trust.weights.integrity,
        "behaviour": config.trust.weights.behaviour,
        "reputation": config.trust.weights.reputation,
        "update_quality": config.trust.weights.update_quality,
    }
    if any(value < 0 for value in weight_values.values()):
        errors.append(f"trust.weights values must all be >= 0, got {weight_values}")
    if sum(weight_values.values()) <= 0:
        errors.append("trust.weights must sum to a positive number")

    if config.aggregation.strategy not in SUPPORTED_AGGREGATION_STRATEGIES:
        errors.append(
            f"aggregation.strategy must be one of {SUPPORTED_AGGREGATION_STRATEGIES}, "
            f"got {config.aggregation.strategy!r}"
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
    trust_raw = raw.get("trust", {})
    weights_raw = trust_raw.get("weights", {})
    default_weights = TrustWeightsConfig()

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
        trust=TrustConfig(
            enabled=trust_raw.get("enabled", True),
            initial_reputation=trust_raw.get("initial_reputation", 0.80),
            reputation_alpha=trust_raw.get("reputation_alpha", 0.80),
            threshold_mode=trust_raw.get("threshold_mode", "fixed"),
            threshold=trust_raw.get("threshold", 0.60),
            participation_fraction=trust_raw.get("participation_fraction", 0.50),
            weights=TrustWeightsConfig(
                identity=weights_raw.get("identity", default_weights.identity),
                integrity=weights_raw.get("integrity", default_weights.integrity),
                behaviour=weights_raw.get("behaviour", default_weights.behaviour),
                reputation=weights_raw.get("reputation", default_weights.reputation),
                update_quality=weights_raw.get("update_quality", default_weights.update_quality),
            ),
        ),
        aggregation=AggregationConfig(strategy=raw.get("aggregation", {}).get("strategy", "fedavg")),
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

    trust_engine: Optional[TrustEngine] = None
    client_selector: Optional[ClientSelector] = None
    if config.trust.enabled:
        trust_weights = TrustWeights(
            identity=config.trust.weights.identity,
            integrity=config.trust.weights.integrity,
            behaviour=config.trust.weights.behaviour,
            reputation=config.trust.weights.reputation,
            update_quality=config.trust.weights.update_quality,
        )
        trust_engine = TrustEngine(
            weights=trust_weights,
            initial_reputation=config.trust.initial_reputation,
            reputation_alpha=config.trust.reputation_alpha,
            seed=config.seed,
        )
        trust_engine.register_clients(clients.keys())

        # Trust-based selection (Phase 3): replaces random `clients_per_round`
        # sampling. Gated on the same `trust.enabled` flag as `TrustEngine`,
        # since selecting by trust without ever computing trust wouldn't
        # mean anything (everyone would sit at the neutral 1.0 default).
        policy = TrustPolicy(threshold_mode=config.trust.threshold_mode, threshold=config.trust.threshold)
        client_selector = ClientSelector(
            policy=policy,
            participation_fraction=config.trust.participation_fraction,
            seed=config.seed,
        )

    server = Server(
        global_model=global_model,
        clients=clients,
        test_dataset=test_dataset,
        device=device,
        clients_per_round=config.federated.clients_per_round,
        seed=config.seed,
        trust_engine=trust_engine,
        client_selector=client_selector,
        aggregation_strategy=config.aggregation.strategy,
    )

    history: List[Dict[str, float]] = []
    for round_num in range(1, config.federated.num_rounds + 1):
        history.append(server.run_round(round_num))

    save_history(history, config.logging.results_dir)
    save_plots(history, config.logging.results_dir)

    if trust_engine is not None:
        save_trust_scores(trust_engine.records, config.logging.results_dir)
        save_trust_history(trust_engine.records, config.logging.results_dir)
        save_trust_plots(trust_engine.records, config.logging.results_dir)

    if client_selector is not None:
        save_selection_csv(server.selection_records, config.logging.results_dir)
        save_aggregation_weights_csv(server.aggregation_weight_records, config.logging.results_dir)
        final_trust_scores = [client.trust_score for client in clients.values()]
        save_selection_plots(
            server.selection_records,
            server.aggregation_weight_records,
            final_trust_scores,
            config.logging.results_dir,
        )

    logger.info("Simulation complete. Results saved to %s", config.logging.results_dir)
    return history


def run_comparison(config_path: str) -> None:
    """Run the same trust-based client selection under both aggregation
    strategies (`fedavg` vs `trust_fedavg`) and plot their global accuracy
    side by side.

    Selection is held identical between the two runs (same seed, same trust
    thresholding) so the only variable being compared is the aggregation
    formula itself — isolating the effect Phase 3 introduces.

    Note: each `run_simulation` call writes the standard per-run outputs
    (history.csv, trust_scores.csv, etc.) to `logging.results_dir`, so after
    both runs those files reflect only the *second* run (trust_fedavg). The
    comparison plot itself is captured from each run's in-memory history
    before that happens, so it always reflects both.
    """
    base_config = load_config(config_path)
    if not base_config.trust.enabled:
        raise ValueError("run_comparison requires trust.enabled: true (selection must be trust-based).")

    histories: Dict[str, List[Dict[str, float]]] = {}
    for strategy in ("fedavg", "trust_fedavg"):
        config = replace(base_config, aggregation=AggregationConfig(strategy=strategy))
        display_name = "FedAvg" if strategy == "fedavg" else "TrustFedAvg"
        histories[display_name] = run_simulation(config)

    save_accuracy_comparison_plot(histories, base_config.logging.results_dir)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a federated learning simulation.")
    parser.add_argument(
        "--config",
        type=str,
        default=str(Path(__file__).resolve().parent.parent / "configs" / "config.yaml"),
        help="Path to the YAML configuration file.",
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Run both fedavg and trust_fedavg (same trust-based selection) and plot a global "
        "accuracy comparison, instead of a single run using the config's aggregation.strategy.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.compare:
        run_comparison(args.config)
    else:
        config = load_config(args.config)
        run_simulation(config)


if __name__ == "__main__":
    main()

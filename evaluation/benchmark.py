"""BenchmarkRunner: the Phase 5 research-evaluation orchestrator.

Runs the full experiment matrix (attack type x malicious fraction x seed x
algorithm), plus ablation and sensitivity sweeps, entirely by constructing
different `simulation.run.SimulationConfig` objects and driving them through
Phase 4's *unmodified* `ExperimentRunner`. No Phase 1-4 code changes were
needed for this: the four algorithm/attack conditions from the spec map
directly onto `ExperimentRunner`'s existing three modes —

    A. FedAvg,      no attack  -> ExperimentRunner.run("baseline")
    B. FedAvg,      attack     -> ExperimentRunner.run("fedavg_attack")
    C. ZT-FedTrust, no attack  -> ExperimentRunner.run("ztfedtrust_attack"), attacks.enabled=False
    D. ZT-FedTrust, attack     -> ExperimentRunner.run("ztfedtrust_attack"), attacks.enabled=True

`ExperimentRunner.run()` already computes `security_records`/
`attack_history_records` for every mode (with `is_malicious` simply always
False when no attack was assigned) — so every field in the Phase 5 master
schema is derivable from data Phase 4 already produces, plus the new
`Server.round_timings` (Phase 5's only source change, in `server/server.py`).

Research-integrity behavior (spec section 28): a failing experiment is
caught, recorded with status "FAILED" and its error message, and the sweep
continues — never silently dropped, never allowed to fabricate a result.
"""

from __future__ import annotations

import csv
import json
import statistics
import subprocess
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

from evaluation.experiment_runner import ExperimentResult, ExperimentRunner
from evaluation.metrics import estimate_communication_cost, estimate_model_size_bytes
from simulation.run import (
    AggregationConfig,
    AttackConfig,
    ByzantineConfig,
    DataConfig,
    FederatedConfig,
    LabelFlippingConfig,
    LoggingConfig,
    ModelConfig,
    ModelPoisoningConfig,
    SimulationConfig,
    SybilConfig,
    TrustConfig,
    TrustWeightsConfig,
)

MASTER_SCHEMA_FIELDNAMES = (
    "ExperimentID",
    "Algorithm",
    "AttackType",
    "MaliciousFraction",
    "Seed",
    "Round",
    "Accuracy",
    "Loss",
    "AverageTrust",
    "BenignTrust",
    "MaliciousTrust",
    "SelectedClients",
    "RejectedClients",
    "MaliciousSelected",
    "BenignSelected",
    "MaliciousAggregationWeight",
    "DetectionRate",
    "Precision",
    "Recall",
    "F1",
    "CommunicationCost",
    "RoundTime",
    "TrustEvaluationTime",
    "AggregationTime",
    # Beyond the spec's required 23 columns: section 14 also asks for Local
    # Training Time and Selection Time, which the required schema has no
    # column for. Appended here (not interleaved) so the first 23 columns
    # still exactly match the spec's order for anyone consuming a prefix.
    "SelectionTime",
    "LocalTrainingTime",
    "AttackSuccessRate",
    "FalsePositiveRate",
)

SUPPORTED_ALGORITHMS = ("fedavg", "ztfedtrust")


# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------


@dataclass
class ExperimentSettings:
    """Conditions held identical across every algorithm/attack comparison (spec section 5)."""

    num_clients: int = 20
    rounds: int = 30
    local_epochs: int = 2
    batch_size: int = 32
    learning_rate: float = 0.01
    momentum: float = 0.9
    participation_fraction: float = 0.5
    dataset: str = "mnist"
    data_dir: str = "./data_cache"
    partition_type: str = "iid"
    dirichlet_alpha: float = 0.5
    model_name: str = "cnn"
    device: str = "cpu"
    threshold_mode: str = "fixed"
    threshold: float = 0.60
    initial_reputation: float = 0.80
    reputation_alpha: float = 0.80
    aggregation_strategy: str = "trust_fedavg"
    trust_weights: Dict[str, float] = field(
        default_factory=lambda: {
            "identity": 0.25,
            "integrity": 0.20,
            "behaviour": 0.20,
            "reputation": 0.15,
            "update_quality": 0.20,
        }
    )
    bytes_per_param: int = 4


@dataclass
class BenchmarkConfig:
    """Top-level Phase 5 experiment configuration (spec section 25)."""

    experiment: ExperimentSettings = field(default_factory=ExperimentSettings)
    seeds: List[int] = field(default_factory=lambda: [42])
    algorithms: List[str] = field(default_factory=lambda: list(SUPPORTED_ALGORITHMS))
    attacks: List[str] = field(default_factory=lambda: ["none"])
    malicious_fractions: List[float] = field(default_factory=lambda: [0.20])
    results_dir: str = "results"
    convergence_target_fraction: float = 0.95
    convergence_stability_rounds: int = 3


def load_benchmark_config(path: str) -> BenchmarkConfig:
    """Load a Phase 5 experiment YAML into a `BenchmarkConfig`, defaults filling any gaps."""
    with open(path, "r", encoding="utf-8") as f:
        raw: Dict[str, Any] = yaml.safe_load(f) or {}

    experiment_raw = raw.get("experiment", {})
    defaults = ExperimentSettings()
    experiment = ExperimentSettings(
        num_clients=experiment_raw.get("num_clients", defaults.num_clients),
        rounds=experiment_raw.get("rounds", defaults.rounds),
        local_epochs=experiment_raw.get("local_epochs", defaults.local_epochs),
        batch_size=experiment_raw.get("batch_size", defaults.batch_size),
        learning_rate=experiment_raw.get("learning_rate", defaults.learning_rate),
        momentum=experiment_raw.get("momentum", defaults.momentum),
        participation_fraction=experiment_raw.get(
            "participation_fraction", defaults.participation_fraction
        ),
        dataset=experiment_raw.get("dataset", defaults.dataset),
        data_dir=experiment_raw.get("data_dir", defaults.data_dir),
        partition_type=experiment_raw.get("partition_type", defaults.partition_type),
        dirichlet_alpha=experiment_raw.get("dirichlet_alpha", defaults.dirichlet_alpha),
        model_name=experiment_raw.get("model_name", defaults.model_name),
        device=experiment_raw.get("device", defaults.device),
        threshold_mode=experiment_raw.get("threshold_mode", defaults.threshold_mode),
        threshold=experiment_raw.get("threshold", defaults.threshold),
        initial_reputation=experiment_raw.get("initial_reputation", defaults.initial_reputation),
        reputation_alpha=experiment_raw.get("reputation_alpha", defaults.reputation_alpha),
        aggregation_strategy=experiment_raw.get("aggregation_strategy", defaults.aggregation_strategy),
        trust_weights=experiment_raw.get("trust_weights", defaults.trust_weights),
        bytes_per_param=experiment_raw.get("bytes_per_param", defaults.bytes_per_param),
    )

    return BenchmarkConfig(
        experiment=experiment,
        seeds=raw.get("seeds", [42]),
        algorithms=raw.get("algorithms", list(SUPPORTED_ALGORITHMS)),
        attacks=raw.get("attacks", ["none"]),
        malicious_fractions=raw.get("malicious_fractions", [0.20]),
        results_dir=raw.get("results_dir", "results"),
        convergence_target_fraction=raw.get("convergence_target_fraction", 0.95),
        convergence_stability_rounds=raw.get("convergence_stability_rounds", 3),
    )


# --------------------------------------------------------------------------
# Master schema record
# --------------------------------------------------------------------------


@dataclass
class MasterRecord:
    """One row of the Phase 5 master results schema (spec section 20)."""

    experiment_id: str
    algorithm: str
    attack_type: str
    malicious_fraction: float
    seed: int
    round_num: int
    accuracy: float
    loss: float
    average_trust: float
    benign_trust: float
    malicious_trust: float
    selected_clients: int
    rejected_clients: int
    malicious_selected: int
    benign_selected: int
    malicious_aggregation_weight: float
    detection_rate: float
    precision: float
    recall: float
    f1: float
    communication_cost: int
    round_time: float
    trust_evaluation_time: float
    aggregation_time: float
    selection_time: float
    local_training_time: float
    attack_success_rate: float
    false_positive_rate: float

    def to_csv_row(self) -> List[Any]:
        return [
            self.experiment_id, self.algorithm, self.attack_type, self.malicious_fraction, self.seed,
            self.round_num, self.accuracy, self.loss, self.average_trust, self.benign_trust,
            self.malicious_trust, self.selected_clients, self.rejected_clients, self.malicious_selected,
            self.benign_selected, self.malicious_aggregation_weight, self.detection_rate, self.precision,
            self.recall, self.f1, self.communication_cost, self.round_time, self.trust_evaluation_time,
            self.aggregation_time, self.selection_time, self.local_training_time, self.attack_success_rate,
            self.false_positive_rate,
        ]


@dataclass
class ExperimentOutcome:
    """The result of one `BenchmarkRunner._run_single` call — success or failure, never silently dropped."""

    experiment_id: str
    algorithm: str
    attack_type: str
    malicious_fraction: float
    seed: int
    status: str  # "OK" or "FAILED"
    records: List[MasterRecord] = field(default_factory=list)
    error: Optional[str] = None


def _safe_mean(values: List[float]) -> float:
    return statistics.mean(values) if values else 0.0


def get_git_commit_hash() -> Optional[str]:
    """Return the current git commit hash, or `None` if unavailable (not a repo, git missing, etc.)."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, timeout=5, check=True
        )
        return result.stdout.strip()
    except Exception:
        return None


def _select_mode(algorithm: str, attack_type: str) -> str:
    """Map (algorithm, attack) onto one of Phase 4 ExperimentRunner's three existing modes."""
    if algorithm == "fedavg":
        return "baseline" if attack_type == "none" else "fedavg_attack"
    if algorithm == "ztfedtrust":
        return "ztfedtrust_attack"  # attacks.enabled (set by the caller) controls whether attacks apply
    raise ValueError(f"Unknown algorithm: {algorithm!r}. Supported: {SUPPORTED_ALGORITHMS}")


def build_simulation_config(
    settings: ExperimentSettings,
    algorithm: str,
    attack_type: str,
    malicious_fraction: float,
    seed: int,
    results_dir: str,
    weight_overrides: Optional[Dict[str, float]] = None,
    threshold_override: Optional[float] = None,
) -> SimulationConfig:
    """Build one `SimulationConfig` for a specific (algorithm, attack, fraction, seed) cell.

    `weight_overrides`/`threshold_override` let ablation/sensitivity sweeps
    reuse this exact same config-building path, varying only the one field
    under study — no new trust logic, just a different config value.
    """
    weights = dict(settings.trust_weights)
    if weight_overrides:
        weights.update(weight_overrides)

    attacks_enabled = attack_type != "none"
    active_attack = attack_type if attacks_enabled else "label_flipping"  # unused when disabled

    return SimulationConfig(
        seed=seed,
        device=settings.device,
        data=DataConfig(
            dataset=settings.dataset,
            data_dir=settings.data_dir,
            num_clients=settings.num_clients,
            partition_type=settings.partition_type,
            dirichlet_alpha=settings.dirichlet_alpha,
            batch_size=settings.batch_size,
        ),
        model=ModelConfig(name=settings.model_name),
        federated=FederatedConfig(
            num_rounds=settings.rounds,
            clients_per_round=max(1, round(settings.num_clients * settings.participation_fraction)),
            local_epochs=settings.local_epochs,
            learning_rate=settings.learning_rate,
            momentum=settings.momentum,
        ),
        logging=LoggingConfig(
            log_dir=f"{results_dir}/logs", log_level="WARNING", results_dir=results_dir
        ),
        trust=TrustConfig(
            enabled=(algorithm == "ztfedtrust"),
            initial_reputation=settings.initial_reputation,
            reputation_alpha=settings.reputation_alpha,
            threshold_mode=settings.threshold_mode,
            threshold=threshold_override if threshold_override is not None else settings.threshold,
            participation_fraction=settings.participation_fraction,
            weights=TrustWeightsConfig(**weights),
        ),
        aggregation=AggregationConfig(
            strategy=settings.aggregation_strategy if algorithm == "ztfedtrust" else "fedavg"
        ),
        attacks=AttackConfig(
            enabled=attacks_enabled,
            malicious_client_fraction=malicious_fraction,
            active_attack=active_attack,
            label_flipping=LabelFlippingConfig(enabled=(attack_type == "label_flipping")),
            model_poisoning=ModelPoisoningConfig(enabled=(attack_type == "model_poisoning")),
            byzantine=ByzantineConfig(enabled=(attack_type == "byzantine")),
            sybil=SybilConfig(enabled=(attack_type == "sybil")),
        ),
    )


def _records_from_result(
    result: ExperimentResult,
    experiment_id: str,
    algorithm: str,
    attack_type: str,
    malicious_fraction: float,
    seed: int,
    bytes_per_param: int,
) -> List[MasterRecord]:
    """Convert one `ExperimentResult` into master-schema rows, one per round.

    Every field is derived from data Phase 3/4 already computed
    (`security_records`, `attack_history_records`, `selection_records`) plus
    Phase 5's `Server.round_timings` — nothing here recomputes federated
    learning or trust logic, only reshapes existing outputs.
    """
    model_size_bytes = estimate_model_size_bytes(result.server.global_model, bytes_per_param)
    timing_by_round = {t["round"]: t for t in result.server.round_timings}

    records: List[MasterRecord] = []
    for i, round_metrics in enumerate(result.history):
        round_num = round_metrics["round"]
        security_record = result.security_records[i]
        round_history = [r for r in result.attack_history_records if r.round_num == round_num]

        selected_clients = sum(1 for r in round_history if r.selected)
        malicious_selected = sum(1 for r in round_history if r.is_malicious and r.selected)
        benign_selected = sum(1 for r in round_history if not r.is_malicious and r.selected)
        # "Rejected" = failed the trust threshold specifically (TP + FP in the
        # detection confusion matrix), matching Phase 3's own
        # `SelectionRecord.rejected_clients` semantics — 0 for algorithms
        # with no threshold mechanism (plain FedAvg).
        rejected_clients = security_record.detected_malicious + security_record.false_positives

        all_trust = [r.trust_score for r in round_history]
        benign_trust_values = [r.trust_score for r in round_history if not r.is_malicious]
        malicious_trust_values = [r.trust_score for r in round_history if r.is_malicious]

        timing = timing_by_round.get(round_num, {})
        comm_cost = estimate_communication_cost(model_size_bytes, selected_clients).total_bytes

        records.append(
            MasterRecord(
                experiment_id=experiment_id,
                algorithm=algorithm,
                attack_type=attack_type,
                malicious_fraction=malicious_fraction,
                seed=seed,
                round_num=round_num,
                accuracy=round_metrics["test_accuracy"],
                loss=round_metrics["test_loss"],
                average_trust=_safe_mean(all_trust),
                benign_trust=_safe_mean(benign_trust_values),
                malicious_trust=_safe_mean(malicious_trust_values),
                selected_clients=selected_clients,
                rejected_clients=rejected_clients,
                malicious_selected=malicious_selected,
                benign_selected=benign_selected,
                malicious_aggregation_weight=security_record.malicious_aggregation_weight,
                detection_rate=security_record.detection_rate,
                precision=security_record.precision,
                recall=security_record.recall,
                f1=security_record.f1,
                communication_cost=comm_cost,
                round_time=round_metrics["round_time_sec"],
                trust_evaluation_time=timing.get("trust_evaluation_time", 0.0),
                aggregation_time=timing.get("aggregation_time", 0.0),
                selection_time=timing.get("selection_time", 0.0),
                local_training_time=timing.get("local_training_time", 0.0),
                attack_success_rate=security_record.attack_success_rate,
                false_positive_rate=(
                    security_record.false_positives
                    / (security_record.false_positives + security_record.true_negatives)
                    if (security_record.false_positives + security_record.true_negatives) > 0
                    else 0.0
                ),
            )
        )
    return records


# --------------------------------------------------------------------------
# BenchmarkRunner
# --------------------------------------------------------------------------


class BenchmarkRunner:
    """Orchestrates the full Phase 5 experiment matrix, ablation, and sensitivity sweeps.

    Args:
        config: the loaded `BenchmarkConfig`.
        results_dir: overrides `config.results_dir` if given (e.g. for a
            reduced-scale validation run writing to a separate directory).
    """

    def __init__(self, config: BenchmarkConfig, results_dir: Optional[str] = None) -> None:
        self.config = config
        self.results_dir = results_dir or config.results_dir
        self.outcomes: List[ExperimentOutcome] = []

    # -- core single-experiment execution, shared by every sweep -----------

    def _run_single(
        self,
        experiment_id: str,
        algorithm: str,
        attack_type: str,
        malicious_fraction: float,
        seed: int,
        weight_overrides: Optional[Dict[str, float]] = None,
        threshold_override: Optional[float] = None,
    ) -> ExperimentOutcome:
        """Run one (algorithm, attack, fraction, seed) cell; never raises — failures are recorded."""
        try:
            sim_config = build_simulation_config(
                self.config.experiment, algorithm, attack_type, malicious_fraction, seed,
                results_dir=f"{self.results_dir}/_scratch",
                weight_overrides=weight_overrides, threshold_override=threshold_override,
            )
            mode = _select_mode(algorithm, attack_type)
            result = ExperimentRunner(sim_config).run(mode)
            records = _records_from_result(
                result, experiment_id, algorithm, attack_type, malicious_fraction, seed,
                self.config.experiment.bytes_per_param,
            )
            outcome = ExperimentOutcome(
                experiment_id, algorithm, attack_type, malicious_fraction, seed, "OK", records=records
            )
        except Exception as exc:  # noqa: BLE001 - deliberately broad: a failing cell must not abort the matrix
            outcome = ExperimentOutcome(
                experiment_id, algorithm, attack_type, malicious_fraction, seed, "FAILED", error=str(exc)
            )

        self.outcomes.append(outcome)
        self._save_raw(outcome, weight_overrides=weight_overrides, threshold_override=threshold_override)
        return outcome

    def _save_raw(
        self,
        outcome: ExperimentOutcome,
        weight_overrides: Optional[Dict[str, float]] = None,
        threshold_override: Optional[float] = None,
    ) -> None:
        """Save one experiment's full raw result (or failure) as a self-contained JSON file."""
        raw_dir = Path(self.results_dir) / "raw"
        raw_dir.mkdir(parents=True, exist_ok=True)
        payload = {
            "experiment_id": outcome.experiment_id,
            "algorithm": outcome.algorithm,
            "attack_type": outcome.attack_type,
            "malicious_fraction": outcome.malicious_fraction,
            "seed": outcome.seed,
            "status": outcome.status,
            "error": outcome.error,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "git_commit_hash": get_git_commit_hash(),
            "weight_overrides": weight_overrides,
            "threshold_override": threshold_override,
            "experiment_settings": asdict(self.config.experiment),
            "records": [asdict(r) for r in outcome.records],
        }
        output_path = raw_dir / f"{outcome.experiment_id}.json"
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, default=str)

    # -- matrix / ablation / sensitivity sweeps -----------------------------

    def run_matrix(self) -> List[ExperimentOutcome]:
        """Run the full algorithm x attack x malicious-fraction x seed matrix.

        The "none" attack is run once per (algorithm, seed) — it doesn't
        depend on `malicious_fractions` — while every other configured
        attack is run once per (algorithm, fraction, seed), matching the
        spec's experiment matrix (a single "None / 0%" row, not repeated
        per attack type).
        """
        for seed in self.config.seeds:
            for algorithm in self.config.algorithms:
                if "none" in self.config.attacks:
                    experiment_id = f"{algorithm}_none_0.0_seed{seed}"
                    self._run_single(experiment_id, algorithm, "none", 0.0, seed)

                for attack_type in self.config.attacks:
                    if attack_type == "none":
                        continue
                    for fraction in self.config.malicious_fractions:
                        experiment_id = f"{algorithm}_{attack_type}_{fraction}_seed{seed}"
                        self._run_single(experiment_id, algorithm, attack_type, fraction, seed)

        self.save_master_csv()
        return self.outcomes

    def run_ablation(
        self, attack_type: str, malicious_fraction: float, seeds: Optional[List[int]] = None
    ) -> List[ExperimentOutcome]:
        """Compare full ZT-FedTrust against 5 single-dimension-disabled variants.

        Disabling a dimension means setting its weight to 0.0 (renormalised
        automatically by `TrustEngine._compose_trust`, which divides by the
        *sum* of configured weights) — the dimension's computation still
        runs internally (e.g. reputation still updates), but it can no
        longer influence the composite trust score, selection, or
        aggregation. This reuses Phase 2's TrustEngine entirely unmodified.
        """
        seeds = seeds if seeds is not None else self.config.seeds
        variants = {
            "full": {},
            "no_identity": {"identity": 0.0},
            "no_integrity": {"integrity": 0.0},
            "no_behaviour": {"behaviour": 0.0},
            "no_reputation": {"reputation": 0.0},
            "no_update_quality": {"update_quality": 0.0},
        }
        outcomes: List[ExperimentOutcome] = []
        for seed in seeds:
            for variant_name, overrides in variants.items():
                experiment_id = f"ablation_{variant_name}_{attack_type}_{malicious_fraction}_seed{seed}"
                outcome = self._run_single(
                    experiment_id, "ztfedtrust", attack_type, malicious_fraction, seed,
                    weight_overrides=overrides,
                )
                outcome.algorithm = f"ztfedtrust_{variant_name}"  # tag for downstream grouping
                outcomes.append(outcome)

        self.save_master_csv(filename="ablation_master.csv", outcomes=outcomes)
        return outcomes

    def run_threshold_sensitivity(
        self,
        attack_type: str,
        malicious_fraction: float,
        thresholds: Optional[List[float]] = None,
        seeds: Optional[List[int]] = None,
    ) -> List[ExperimentOutcome]:
        """Sweep `trust.threshold` to measure the security/availability trade-off."""
        thresholds = thresholds if thresholds is not None else [0.40, 0.50, 0.60, 0.70, 0.80]
        seeds = seeds if seeds is not None else self.config.seeds
        outcomes: List[ExperimentOutcome] = []
        for seed in seeds:
            for threshold in thresholds:
                experiment_id = f"threshold_{threshold}_{attack_type}_{malicious_fraction}_seed{seed}"
                outcome = self._run_single(
                    experiment_id, "ztfedtrust", attack_type, malicious_fraction, seed,
                    threshold_override=threshold,
                )
                outcome.algorithm = f"threshold_{threshold}"
                outcomes.append(outcome)

        self.save_master_csv(filename="sensitivity_master.csv", outcomes=outcomes)
        return outcomes

    def run_weight_sensitivity(
        self,
        weight_configs: Dict[str, Dict[str, float]],
        attack_type: str,
        malicious_fraction: float,
        seeds: Optional[List[int]] = None,
    ) -> List[ExperimentOutcome]:
        """Compare named trust-weight configurations (spec section 16)."""
        seeds = seeds if seeds is not None else self.config.seeds
        outcomes: List[ExperimentOutcome] = []
        for seed in seeds:
            for config_name, weights in weight_configs.items():
                experiment_id = f"weights_{config_name}_{attack_type}_{malicious_fraction}_seed{seed}"
                outcome = self._run_single(
                    experiment_id, "ztfedtrust", attack_type, malicious_fraction, seed,
                    weight_overrides=weights,
                )
                outcome.algorithm = f"weights_{config_name}"
                outcomes.append(outcome)

        self.save_master_csv(filename="weight_sensitivity_master.csv", outcomes=outcomes)
        return outcomes

    # -- output -------------------------------------------------------------

    def save_master_csv(
        self, filename: str = "master_results.csv", outcomes: Optional[List[ExperimentOutcome]] = None
    ) -> None:
        """Write every successful outcome's rows to `results/raw/<filename>`, and failures alongside."""
        outcomes = outcomes if outcomes is not None else self.outcomes
        raw_dir = Path(self.results_dir) / "raw"
        raw_dir.mkdir(parents=True, exist_ok=True)

        with open(raw_dir / filename, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(MASTER_SCHEMA_FIELDNAMES)
            for outcome in outcomes:
                if outcome.status == "OK":
                    for record in outcome.records:
                        writer.writerow(record.to_csv_row())

        failed = [o for o in outcomes if o.status == "FAILED"]
        if failed:
            failed_path = raw_dir / "failed_experiments.csv"
            file_exists = failed_path.exists()
            with open(failed_path, "a", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                if not file_exists:
                    writer.writerow(["ExperimentID", "Algorithm", "AttackType", "MaliciousFraction", "Seed", "Error"])
                for outcome in failed:
                    writer.writerow(
                        [
                            outcome.experiment_id, outcome.algorithm, outcome.attack_type,
                            outcome.malicious_fraction, outcome.seed, outcome.error,
                        ]
                    )

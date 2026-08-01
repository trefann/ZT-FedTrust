"""Security metrics CSV logging + Phase 4 visualisations.

Ties `evaluation.attack_metrics`' detection-metric formulas to per-round
record types, CSV writers matching the spec'd columns exactly, and the
plots comparing malicious vs benign behaviour and (via
`save_loss_comparison_plot`, alongside Phase 3's reused
`save_accuracy_comparison_plot`) FedAvg / FedAvg-under-attack /
ZT-FedTrust-under-attack.

`build_round_security_records` is the single place ground-truth
`is_malicious` labels get combined with a round's observed outcome — shared
by both `simulation.run.run_simulation` and
`evaluation.experiment_runner.ExperimentRunner` so the two entry points can
never compute these numbers differently.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import matplotlib

matplotlib.use("Agg")  # headless-safe backend; must be set before pyplot import
import matplotlib.pyplot as plt

from clients.client import Client
from clients.malicious_client import MaliciousClient
from evaluation.attack_metrics import (
    compute_attack_success_rate,
    compute_detection_counts,
    compute_detection_metrics,
    compute_malicious_aggregation_weight,
)
from server.selector import ClientSelector

_SECURITY_CSV_FIELDNAMES = (
    "Round",
    "AttackType",
    "TotalClients",
    "MaliciousClients",
    "BenignClients",
    "DetectedMalicious",
    "UndetectedMalicious",
    "FalsePositives",
    "TrueNegatives",
    "Precision",
    "Recall",
    "F1",
    "DetectionRate",
    "AttackSuccessRate",
    "MaliciousAggregationWeight",
)

_HISTORY_CSV_FIELDNAMES = (
    "Round",
    "ClientID",
    "IsMalicious",
    "AttackType",
    "TrustScore",
    "Selected",
    "AggregationWeight",
)


@dataclass
class SecurityMetricsRecord:
    """One round's aggregate detection/attack-impact metrics."""

    round_num: int
    attack_type: str
    total_clients: int
    malicious_clients: int
    benign_clients: int
    detected_malicious: int
    undetected_malicious: int
    false_positives: int
    true_negatives: int
    precision: float
    recall: float
    f1: float
    detection_rate: float
    attack_success_rate: float
    malicious_aggregation_weight: float


@dataclass
class AttackHistoryRecord:
    """One client's ground-truth label, trust, and participation for one round."""

    round_num: int
    client_id: int
    is_malicious: bool
    attack_type: str
    trust_score: float
    selected: bool
    aggregation_weight: float


def build_round_security_records(
    round_num: int,
    attack_type: str,
    clients: Dict[int, Union[Client, MaliciousClient]],
    client_selector: Optional[ClientSelector],
    selected_ids: List[int],
    aggregation_weights: Dict[int, float],
) -> Tuple[SecurityMetricsRecord, List[AttackHistoryRecord]]:
    """Compute one round's security metrics + per-client attack history.

    "Accepted" (for detection purposes) means trust-eligible, i.e. passed
    the threshold `client_selector.policy` computed this round — see
    `evaluation.attack_metrics`'s module docstring for why that's the right
    notion of "accepted", not merely "sampled in stage 2". When
    `client_selector` is `None` (plain FedAvg, no trust filtering at all),
    every client is trivially "accepted": there's no mechanism to reject
    anyone, which is exactly the (informative) baseline this is meant to
    expose.
    """
    all_ids = sorted(clients.keys())
    is_malicious = {cid: isinstance(clients[cid], MaliciousClient) for cid in all_ids}

    if client_selector is not None:
        eligible_ids = {cid for cid in all_ids if client_selector.policy.is_trusted(clients[cid])}
    else:
        eligible_ids = set(all_ids)

    counts = compute_detection_counts(is_malicious, eligible_ids, all_ids)
    metrics = compute_detection_metrics(counts)
    attack_success_rate = compute_attack_success_rate(is_malicious, eligible_ids, all_ids)
    malicious_weight = compute_malicious_aggregation_weight(is_malicious, aggregation_weights)

    malicious_count = sum(1 for v in is_malicious.values() if v)
    security_record = SecurityMetricsRecord(
        round_num=round_num,
        attack_type=attack_type,
        total_clients=len(all_ids),
        malicious_clients=malicious_count,
        benign_clients=len(all_ids) - malicious_count,
        detected_malicious=counts.true_positives,
        undetected_malicious=counts.false_negatives,
        false_positives=counts.false_positives,
        true_negatives=counts.true_negatives,
        precision=metrics.precision,
        recall=metrics.recall,
        f1=metrics.f1,
        detection_rate=metrics.detection_rate,
        attack_success_rate=attack_success_rate,
        malicious_aggregation_weight=malicious_weight,
    )

    selected_set = set(selected_ids)
    history_records = [
        AttackHistoryRecord(
            round_num=round_num,
            client_id=cid,
            is_malicious=is_malicious[cid],
            attack_type=(clients[cid].attack_type if isinstance(clients[cid], MaliciousClient) else "none"),
            trust_score=clients[cid].trust_score,
            selected=cid in selected_set,
            aggregation_weight=aggregation_weights.get(cid, 0.0),
        )
        for cid in all_ids
    ]
    return security_record, history_records


def save_security_metrics_csv(records: List[SecurityMetricsRecord], results_dir: str) -> None:
    """Write per-round security metrics to `security_metrics.csv`."""
    if not records:
        return
    Path(results_dir).mkdir(parents=True, exist_ok=True)
    output_path = Path(results_dir) / "security_metrics.csv"
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(_SECURITY_CSV_FIELDNAMES)
        for r in records:
            writer.writerow(
                [
                    r.round_num, r.attack_type, r.total_clients, r.malicious_clients, r.benign_clients,
                    r.detected_malicious, r.undetected_malicious, r.false_positives, r.true_negatives,
                    r.precision, r.recall, r.f1, r.detection_rate, r.attack_success_rate,
                    r.malicious_aggregation_weight,
                ]
            )


def save_attack_history_csv(records: List[AttackHistoryRecord], results_dir: str) -> None:
    """Write per-client, per-round attack history to `attack_history.csv`."""
    if not records:
        return
    Path(results_dir).mkdir(parents=True, exist_ok=True)
    output_path = Path(results_dir) / "attack_history.csv"
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(_HISTORY_CSV_FIELDNAMES)
        for r in records:
            writer.writerow(
                [r.round_num, r.client_id, r.is_malicious, r.attack_type, r.trust_score, r.selected, r.aggregation_weight]
            )


def print_security_summary(record: SecurityMetricsRecord) -> None:
    """Print a concise one-line security summary for a round."""
    print(
        f"[security] Round {record.round_num} ({record.attack_type}): "
        f"detected {record.detected_malicious}/{record.malicious_clients} malicious, "
        f"F1={record.f1:.2f}, attack success rate={record.attack_success_rate:.2f}, "
        f"malicious agg. weight={record.malicious_aggregation_weight:.2f}"
    )


def _plot_trust_malicious_vs_benign(history_records: List[AttackHistoryRecord], output_path: Path) -> None:
    by_round: Dict[int, List[AttackHistoryRecord]] = {}
    for r in history_records:
        by_round.setdefault(r.round_num, []).append(r)

    rounds = sorted(by_round)
    benign_avg, malicious_avg = [], []
    for round_num in rounds:
        round_records = by_round[round_num]
        benign_scores = [r.trust_score for r in round_records if not r.is_malicious]
        malicious_scores = [r.trust_score for r in round_records if r.is_malicious]
        benign_avg.append(sum(benign_scores) / len(benign_scores) if benign_scores else float("nan"))
        malicious_avg.append(
            sum(malicious_scores) / len(malicious_scores) if malicious_scores else float("nan")
        )

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(rounds, benign_avg, marker="o", color="tab:green", label="Benign")
    ax.plot(rounds, malicious_avg, marker="o", color="tab:red", label="Malicious")
    ax.set_xlabel("Round")
    ax.set_ylabel("Average Trust")
    ax.set_ylim(0.0, 1.05)
    ax.set_title("Trust: Malicious vs Benign")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def _plot_metric_vs_round(records: List[SecurityMetricsRecord], key: str, title: str, ylabel: str, output_path: Path, color: str) -> None:
    rounds = [r.round_num for r in records]
    values = [getattr(r, key) for r in records]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(rounds, values, marker="o", color=color, label=ylabel)
    ax.set_xlabel("Round")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def _plot_trust_distribution_by_class(history_records: List[AttackHistoryRecord], output_path: Path) -> None:
    final_round = max(r.round_num for r in history_records)
    final_records = [r for r in history_records if r.round_num == final_round]
    benign_scores = [r.trust_score for r in final_records if not r.is_malicious]
    malicious_scores = [r.trust_score for r in final_records if r.is_malicious]

    fig, ax = plt.subplots(figsize=(8, 5))
    bins = [i / 10 for i in range(11)]
    if benign_scores:
        ax.hist(benign_scores, bins=bins, alpha=0.6, color="tab:green", edgecolor="black", label="Benign")
    if malicious_scores:
        ax.hist(malicious_scores, bins=bins, alpha=0.6, color="tab:red", edgecolor="black", label="Malicious")
    ax.set_xlabel("Trust Score")
    ax.set_ylabel("Number of Clients")
    ax.set_title(f"Client Trust Distribution (Round {final_round})")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def save_security_plots(
    security_records: List[SecurityMetricsRecord],
    history_records: List[AttackHistoryRecord],
    results_dir: str,
) -> None:
    """Generate and save the single-experiment Phase 4 visualisations.

    Writes into `results/security_plots/`:
      - `trust_malicious_vs_benign.png`
      - `attack_detection_rate.png`
      - `malicious_aggregation_weight.png`
      - `client_trust_distribution_by_class.png`

    The two cross-experiment comparison plots (global accuracy, global
    loss) need multiple experiment runs and are produced separately by
    `save_loss_comparison_plot` / `server.selection_metrics.save_accuracy_comparison_plot`
    — see `evaluation/experiment_runner.py`.
    """
    if not security_records or not history_records:
        return
    plots_dir = Path(results_dir) / "security_plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    _plot_trust_malicious_vs_benign(history_records, plots_dir / "trust_malicious_vs_benign.png")
    _plot_metric_vs_round(
        security_records, "detection_rate", "Attack Detection Rate", "Detection Rate",
        plots_dir / "attack_detection_rate.png", "tab:blue",
    )
    _plot_metric_vs_round(
        security_records, "malicious_aggregation_weight", "Malicious Aggregation Weight", "Weight",
        plots_dir / "malicious_aggregation_weight.png", "tab:purple",
    )
    _plot_trust_distribution_by_class(history_records, plots_dir / "client_trust_distribution_by_class.png")


def save_loss_comparison_plot(histories: Dict[str, List[Dict[str, float]]], results_dir: str) -> None:
    """Plot global test loss vs round for one or more named experiment runs.

    Mirrors `server.selection_metrics.save_accuracy_comparison_plot` (reused
    as-is for the accuracy comparison — it already accepts an arbitrary
    number of named histories, 2-way or 3-way, without modification).
    """
    if not histories:
        return
    plots_dir = Path(results_dir) / "security_plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 5))
    colors = ["tab:blue", "tab:orange", "tab:green", "tab:red"]
    for (label, history), color in zip(histories.items(), colors):
        rounds = [entry["round"] for entry in history]
        loss = [entry["test_loss"] for entry in history]
        ax.plot(rounds, loss, marker="o", color=color, label=label)
    ax.set_xlabel("Round")
    ax.set_ylabel("Test Loss")
    ax.set_title("Global Loss Comparison")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(plots_dir / "global_loss_comparison.png")
    plt.close(fig)

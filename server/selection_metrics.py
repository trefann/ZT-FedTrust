"""Client-selection and aggregation-weight records: CSV logging, console summary, plots.

Mirrors the role `trust/trust_metrics.py` played for Phase 2: this module
owns no selection or aggregation logic itself, only how the results of that
logic get recorded and visualised — so it can be tested independently of
`Server`, `ClientSelector`, or `TrustPolicy`.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import matplotlib

matplotlib.use("Agg")  # headless-safe backend; must be set before pyplot import
import matplotlib.pyplot as plt

_SELECTION_CSV_FIELDNAMES = (
    "Round",
    "Threshold",
    "EligibleClients",
    "SelectedClients",
    "RejectedClients",
    "AverageTrust",
    "MinTrust",
    "MaxTrust",
)

_WEIGHT_CSV_FIELDNAMES = ("Round", "ClientID", "Trust", "DatasetSize", "AggregationWeight")

_STRATEGY_DISPLAY_NAMES = {"fedavg": "FedAvg", "trust_fedavg": "TrustFedAvg"}


@dataclass
class SelectionRecord:
    """One round's client-selection outcome and trust statistics.

    `total_clients` isn't one of the required CSV columns but is kept here
    for the console summary's "Clients Available" line, and so
    `rejected_clients` (= `total_clients - eligible_clients`, i.e. clients
    that failed the trust threshold specifically) is self-documenting
    alongside the numbers it was derived from.
    """

    round_num: int
    threshold: float
    total_clients: int
    eligible_clients: int
    selected_clients: int
    rejected_clients: int
    average_trust: float
    min_trust: float
    max_trust: float


@dataclass
class AggregationWeightRecord:
    """One client's aggregation contribution for one round."""

    round_num: int
    client_id: int
    trust: float
    dataset_size: int
    aggregation_weight: float


def save_selection_csv(records: List[SelectionRecord], results_dir: str) -> None:
    """Write per-round client-selection statistics to `client_selection.csv`."""
    if not records:
        return
    Path(results_dir).mkdir(parents=True, exist_ok=True)
    output_path = Path(results_dir) / "client_selection.csv"
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(_SELECTION_CSV_FIELDNAMES)
        for record in records:
            writer.writerow(
                [
                    record.round_num,
                    record.threshold,
                    record.eligible_clients,
                    record.selected_clients,
                    record.rejected_clients,
                    record.average_trust,
                    record.min_trust,
                    record.max_trust,
                ]
            )


def save_aggregation_weights_csv(records: List[AggregationWeightRecord], results_dir: str) -> None:
    """Write per-client, per-round aggregation weights to `aggregation_weights.csv`."""
    if not records:
        return
    Path(results_dir).mkdir(parents=True, exist_ok=True)
    output_path = Path(results_dir) / "aggregation_weights.csv"
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(_WEIGHT_CSV_FIELDNAMES)
        for record in records:
            writer.writerow(
                [
                    record.round_num,
                    record.client_id,
                    record.trust,
                    record.dataset_size,
                    record.aggregation_weight,
                ]
            )


def print_selection_summary(
    record: SelectionRecord,
    aggregation_strategy: str,
    global_accuracy: float,
) -> None:
    """Print the concise end-of-round selection + aggregation summary."""
    strategy_display = _STRATEGY_DISPLAY_NAMES.get(aggregation_strategy, aggregation_strategy)
    print("-" * 40)
    print(f"Round {record.round_num}")
    print(f"Threshold: {record.threshold:.2f}")
    print(f"Clients Available: {record.total_clients}")
    print(f"Eligible: {record.eligible_clients}")
    print(f"Selected: {record.selected_clients}")
    print(f"Rejected: {record.rejected_clients}")
    print(f"Average Trust: {record.average_trust:.2f}")
    print(f"Aggregation: {strategy_display}")
    print(f"Global Accuracy: {global_accuracy * 100:.1f}%")
    print("-" * 40)


def _plot_line(rounds, values, title, ylabel, output_path, color, label=None):
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(rounds, values, marker="o", color=color, label=label or ylabel)
    ax.set_xlabel("Round")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def _plot_trust_distribution(client_trust_scores: List[float], round_num: int, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(
        client_trust_scores,
        bins=10,
        range=(0.0, 1.0),
        color="tab:cyan",
        edgecolor="black",
        label=f"Round {round_num} clients",
    )
    ax.set_xlabel("Trust Score")
    ax.set_ylabel("Number of Clients")
    ax.set_title(f"Client Trust Distribution (Round {round_num})")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def _plot_selection_timeline(records: List[SelectionRecord], output_path: Path) -> None:
    rounds = [r.round_num for r in records]
    selected = [r.selected_clients for r in records]
    rejected = [r.rejected_clients for r in records]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(rounds, selected, color="tab:green", label="Selected")
    ax.bar(rounds, rejected, bottom=selected, color="tab:red", label="Rejected")
    ax.set_xlabel("Round")
    ax.set_ylabel("Number of Clients")
    ax.set_title("Client Selection Timeline")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def _plot_aggregation_weight_distribution(records: List[AggregationWeightRecord], output_path: Path) -> None:
    final_round = max(record.round_num for record in records)
    final_records = sorted(
        (r for r in records if r.round_num == final_round), key=lambda r: r.client_id
    )
    client_labels = [f"Client {r.client_id}" for r in final_records]
    weights = [r.aggregation_weight for r in final_records]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(client_labels, weights, color="tab:purple", label=f"Round {final_round} weights")
    ax.set_xlabel("Client")
    ax.set_ylabel("Aggregation Weight")
    ax.set_title(f"Aggregation Weight Distribution (Round {final_round})")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def save_selection_plots(
    selection_records: List[SelectionRecord],
    weight_records: List[AggregationWeightRecord],
    final_client_trust_scores: List[float],
    results_dir: str,
) -> None:
    """Generate and save the required selection/aggregation visualisations.

    Writes into `results/selection_plots/`:
      - `client_trust_distribution.png`
      - `client_selection_timeline.png`
      - `trust_threshold_evolution.png`
      - `aggregation_weight_distribution.png`

    The 5th required visualisation (Global Accuracy Comparison: FedAvg vs
    TrustFedAvg) needs two full simulation runs to compare, so it's produced
    separately by `save_accuracy_comparison_plot` — see `simulation/run.py`'s
    `--compare` mode.
    """
    if not selection_records:
        return
    plots_dir = Path(results_dir) / "selection_plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    final_round = selection_records[-1].round_num
    _plot_trust_distribution(final_client_trust_scores, final_round, plots_dir / "client_trust_distribution.png")
    _plot_selection_timeline(selection_records, plots_dir / "client_selection_timeline.png")
    _plot_line(
        [r.round_num for r in selection_records],
        [r.threshold for r in selection_records],
        "Trust Threshold Evolution",
        "Threshold",
        plots_dir / "trust_threshold_evolution.png",
        "tab:blue",
    )
    if weight_records:
        _plot_aggregation_weight_distribution(weight_records, plots_dir / "aggregation_weight_distribution.png")


def save_accuracy_comparison_plot(
    histories: Dict[str, List[Dict[str, float]]], results_dir: str
) -> None:
    """Plot global test accuracy vs round for one or more named strategy runs.

    `histories` maps a display label (e.g. `"FedAvg"`, `"TrustFedAvg"`) to
    that run's per-round history (as returned by `simulation.run.run_simulation`).
    """
    if not histories:
        return
    plots_dir = Path(results_dir) / "selection_plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 5))
    colors = ["tab:blue", "tab:orange", "tab:green", "tab:red"]
    for (label, history), color in zip(histories.items(), colors):
        rounds = [entry["round"] for entry in history]
        accuracy = [entry["test_accuracy"] for entry in history]
        ax.plot(rounds, accuracy, marker="o", color=color, label=label)
    ax.set_xlabel("Round")
    ax.set_ylabel("Test Accuracy")
    ax.set_title("Global Accuracy Comparison")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(plots_dir / "global_accuracy_comparison.png")
    plt.close(fig)

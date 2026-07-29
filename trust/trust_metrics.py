"""Trust record type, CSV logging, console summaries, and trust visualisations.

This module has no dependency on `TrustEngine` — it only knows about
`TrustRecord`, so it can log/plot output from any source of trust records,
and can be unit-tested without constructing a full simulation.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import matplotlib

matplotlib.use("Agg")  # headless-safe backend; must be set before pyplot import
import matplotlib.pyplot as plt

_CSV_FIELDNAMES = (
    "Round",
    "ClientID",
    "Identity",
    "Integrity",
    "Behaviour",
    "Reputation",
    "UpdateQuality",
    "TrustScore",
)


@dataclass
class TrustRecord:
    """One client's complete Zero Trust evaluation for a single round."""

    round_num: int
    client_id: int
    identity: float
    integrity: float
    behaviour: float
    reputation: float
    update_quality: float
    trust_score: float


def save_trust_scores(records: List[TrustRecord], results_dir: str) -> None:
    """Write every per-client, per-round trust record to `trust_scores.csv`."""
    if not records:
        return
    Path(results_dir).mkdir(parents=True, exist_ok=True)
    output_path = Path(results_dir) / "trust_scores.csv"
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(_CSV_FIELDNAMES)
        for record in records:
            writer.writerow(
                [
                    record.round_num,
                    record.client_id,
                    record.identity,
                    record.integrity,
                    record.behaviour,
                    record.reputation,
                    record.update_quality,
                    record.trust_score,
                ]
            )


def _round_summary_rows(records: List[TrustRecord]) -> List[Dict[str, float]]:
    """Aggregate per-client records into one row of round-level statistics.

    Shared by `save_trust_history` and `save_trust_plots` so the two never
    disagree about how a round's numbers are summarised.
    """
    by_round: Dict[int, List[TrustRecord]] = {}
    for record in records:
        by_round.setdefault(record.round_num, []).append(record)

    rows: List[Dict[str, float]] = []
    for round_num in sorted(by_round):
        round_records = by_round[round_num]
        count = len(round_records)
        rows.append(
            {
                "round": round_num,
                "average_trust": sum(r.trust_score for r in round_records) / count,
                "min_trust": min(r.trust_score for r in round_records),
                "max_trust": max(r.trust_score for r in round_records),
                "average_identity": sum(r.identity for r in round_records) / count,
                "average_integrity": sum(r.integrity for r in round_records) / count,
                "average_behaviour": sum(r.behaviour for r in round_records) / count,
                "average_reputation": sum(r.reputation for r in round_records) / count,
                "average_update_quality": sum(r.update_quality for r in round_records) / count,
            }
        )
    return rows


def save_trust_history(records: List[TrustRecord], results_dir: str) -> List[Dict[str, float]]:
    """Write per-round trust-evolution statistics to `trust_history.csv`.

    Returns the same rows so `save_trust_plots` can reuse them without
    recomputing the aggregation.
    """
    rows = _round_summary_rows(records)
    if not rows:
        return rows
    Path(results_dir).mkdir(parents=True, exist_ok=True)
    output_path = Path(results_dir) / "trust_history.csv"
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    return rows


def print_round_summary(round_num: int, records: List[TrustRecord]) -> None:
    """Print the concise end-of-round trust + FedAvg summary to the console."""
    if not records:
        return
    average_trust = sum(r.trust_score for r in records) / len(records)
    average_reputation = sum(r.reputation for r in records) / len(records)
    highest = max(records, key=lambda r: r.trust_score)
    lowest = min(records, key=lambda r: r.trust_score)

    print(f"\nRound {round_num}")
    print(f"Average Trust: {average_trust:.2f}")
    print(f"Highest Trust Client: Client {highest.client_id} ({highest.trust_score:.2f})")
    print(f"Lowest Trust Client: Client {lowest.client_id} ({lowest.trust_score:.2f})")
    print(f"Average Reputation: {average_reputation:.2f}")
    print("FedAvg Completed Successfully")


def _plot_metric_vs_round(
    rows: List[Dict[str, float]],
    key: str,
    title: str,
    ylabel: str,
    output_path: Path,
    color: str,
) -> None:
    """Shared line-plot renderer for every "<metric> vs Round" chart."""
    rounds = [row["round"] for row in rows]
    values = [row[key] for row in rows]

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


def _plot_trust_distribution(records: List[TrustRecord], output_path: Path) -> None:
    """Histogram of individual client trust scores in the most recent round."""
    final_round = max(record.round_num for record in records)
    final_scores = [record.trust_score for record in records if record.round_num == final_round]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(
        final_scores,
        bins=10,
        range=(0.0, 1.0),
        color="tab:cyan",
        edgecolor="black",
        label=f"Round {final_round} clients",
    )
    ax.set_xlabel("Trust Score")
    ax.set_ylabel("Number of Clients")
    ax.set_title(f"Trust Score Distribution (Round {final_round})")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def save_trust_plots(records: List[TrustRecord], results_dir: str) -> None:
    """Generate and save all required trust visualisations to `results/trust_plots/`."""
    if not records:
        return
    plots_dir = Path(results_dir) / "trust_plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    rows = _round_summary_rows(records)

    _plot_metric_vs_round(
        rows, "average_trust", "Average Trust vs Round", "Average Trust",
        plots_dir / "average_trust_vs_round.png", "tab:blue",
    )
    _plot_metric_vs_round(
        rows, "average_reputation", "Reputation vs Round", "Average Reputation",
        plots_dir / "reputation_vs_round.png", "tab:green",
    )
    _plot_metric_vs_round(
        rows, "average_behaviour", "Behaviour Score vs Round", "Average Behaviour Score",
        plots_dir / "behaviour_vs_round.png", "tab:orange",
    )
    _plot_metric_vs_round(
        rows, "average_update_quality", "Update Quality vs Round", "Average Update Quality",
        plots_dir / "update_quality_vs_round.png", "tab:purple",
    )
    _plot_trust_distribution(records, plots_dir / "trust_distribution.png")

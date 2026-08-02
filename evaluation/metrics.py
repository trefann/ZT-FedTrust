"""Core ML, robustness, and communication-cost metrics (Phase 5).

Pure functions operating on plain data (a `history` list as returned by
`Server.run_round`, or plain floats) so they're trivially unit-testable and
reusable from both live experiment runs and reloaded CSV results.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import torch.nn as nn


# --------------------------------------------------------------------------
# Core ML metrics
# --------------------------------------------------------------------------


def best_accuracy(history: List[Dict[str, float]]) -> float:
    """Maximum `test_accuracy` reached at any point during training."""
    if not history:
        return 0.0
    return max(entry["test_accuracy"] for entry in history)


def final_accuracy(history: List[Dict[str, float]]) -> float:
    """`test_accuracy` after the final round."""
    if not history:
        return 0.0
    return history[-1]["test_accuracy"]


def final_loss(history: List[Dict[str, float]]) -> float:
    """`test_loss` after the final round."""
    if not history:
        return 0.0
    return history[-1]["test_loss"]


def convergence_round(
    history: List[Dict[str, float]],
    target_fraction: float = 0.95,
    stability_rounds: int = 3,
    reference: str = "best",
) -> Optional[int]:
    """First round where accuracy reaches `target_fraction` of a reference value
    and stays at or above it for `stability_rounds` consecutive rounds.

    Args:
        target_fraction: e.g. 0.95 -> "reached 95% of the reference accuracy".
        stability_rounds: how many consecutive rounds must hold at/above the
            threshold before it counts as converged (guards against a single
            lucky round being mistaken for convergence).
        reference: `"best"` (accuracy ever reached) or `"final"` (last
            round's accuracy) — which value `target_fraction` is relative to.

    Returns `None` if the history is empty, `reference` is invalid, or no
    window of `stability_rounds` consecutive rounds ever meets the
    threshold (e.g. the run is too short, or accuracy never stabilises).
    """
    if not history:
        return None
    if reference not in ("best", "final"):
        raise ValueError(f"reference must be 'best' or 'final', got {reference!r}")

    accuracies = [entry["test_accuracy"] for entry in history]
    rounds = [entry["round"] for entry in history]
    reference_accuracy = max(accuracies) if reference == "best" else accuracies[-1]
    threshold = reference_accuracy * target_fraction

    for i in range(len(accuracies)):
        window = accuracies[i : i + stability_rounds]
        if len(window) < stability_rounds:
            break  # not enough remaining rounds to confirm stability
        if all(acc >= threshold for acc in window):
            return rounds[i]
    return None


# --------------------------------------------------------------------------
# Robustness metrics
# --------------------------------------------------------------------------


def accuracy_degradation(accuracy_without_attack: float, accuracy_under_attack: float) -> float:
    """Accuracy_without_attack - Accuracy_under_attack. Positive means the attack hurt accuracy."""
    return accuracy_without_attack - accuracy_under_attack


def robustness_ratio(accuracy_under_attack: float, accuracy_without_attack: float) -> float:
    """Accuracy_under_attack / Accuracy_without_attack. 1.0 = no measurable degradation.

    Returns 0.0 (rather than raising) when `accuracy_without_attack` is 0,
    since that denominator being zero means "no meaningful baseline to
    compare against" rather than a division error worth crashing over.
    """
    if accuracy_without_attack == 0:
        return 0.0
    return accuracy_under_attack / accuracy_without_attack


def malicious_influence_ratio(
    malicious_aggregation_weight: float, total_aggregation_weight: float = 1.0
) -> float:
    """Malicious aggregation weight / total aggregation weight.

    `total_aggregation_weight` defaults to 1.0 because
    `server.trust_fedavg.compute_aggregation_weights` always normalises
    weights to sum to 1.0 across participants — so this is usually just
    `malicious_aggregation_weight` itself. Kept as an explicit ratio (rather
    than a bare pass-through) so the formula stays correct even if a future
    aggregation strategy doesn't normalise its weights.
    """
    if total_aggregation_weight == 0:
        return 0.0
    return malicious_aggregation_weight / total_aggregation_weight


# --------------------------------------------------------------------------
# Communication cost — SIMULATED/ESTIMATED, not a real network measurement
# --------------------------------------------------------------------------


@dataclass
class CommunicationCost:
    """Simulated/estimated communication cost for one round.

    Computed analytically from model parameter count x a configurable
    bytes-per-parameter precision x participant count. The simulator runs
    entirely in-process and never serialises weights over an actual
    network — these numbers are an estimate of what a real deployment
    would transfer, not a measurement of anything the simulator itself
    sent or received.
    """

    num_participating_clients: int
    model_size_bytes: int
    uploaded_bytes: int
    downloaded_bytes: int
    total_bytes: int


def estimate_model_size_bytes(model: nn.Module, bytes_per_param: int = 4) -> int:
    """Estimate serialized model size: parameter count x bytes per parameter.

    `bytes_per_param` defaults to 4 (float32). Pass 2 for float16, 1 for
    int8-quantised, etc., to model a different transmission precision.
    """
    num_params = sum(p.numel() for p in model.parameters())
    return num_params * bytes_per_param


def estimate_communication_cost(
    model_size_bytes: int, num_participating_clients: int
) -> CommunicationCost:
    """Estimate one round's communication cost.

    Each participating client both downloads the broadcast global model and
    uploads its own trained update — one model-sized transfer each way.
    """
    uploaded = model_size_bytes * num_participating_clients
    downloaded = model_size_bytes * num_participating_clients
    return CommunicationCost(
        num_participating_clients=num_participating_clients,
        model_size_bytes=model_size_bytes,
        uploaded_bytes=uploaded,
        downloaded_bytes=downloaded,
        total_bytes=uploaded + downloaded,
    )

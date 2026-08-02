"""Tests for evaluation/metrics.py: core ML, robustness, and communication-cost formulas."""

import pytest
import torch.nn as nn

from evaluation.metrics import (
    accuracy_degradation,
    best_accuracy,
    convergence_round,
    estimate_communication_cost,
    estimate_model_size_bytes,
    final_accuracy,
    final_loss,
    malicious_influence_ratio,
    robustness_ratio,
)


def _history(accuracies, losses=None):
    losses = losses or [1.0] * len(accuracies)
    return [
        {"round": i + 1, "test_accuracy": acc, "test_loss": loss}
        for i, (acc, loss) in enumerate(zip(accuracies, losses))
    ]


def test_best_and_final_accuracy():
    history = _history([0.5, 0.9, 0.7])
    assert best_accuracy(history) == 0.9
    assert final_accuracy(history) == 0.7


def test_final_loss():
    history = _history([0.5, 0.9], losses=[1.0, 0.2])
    assert final_loss(history) == 0.2


def test_empty_history_metrics_default_to_zero():
    assert best_accuracy([]) == 0.0
    assert final_accuracy([]) == 0.0
    assert final_loss([]) == 0.0
    assert convergence_round([]) is None


def test_convergence_round_basic_case():
    # Reaches 95% of best (0.9 -> threshold 0.855) at round 3 and stays there.
    history = _history([0.5, 0.6, 0.86, 0.88, 0.9])
    assert convergence_round(history, target_fraction=0.95, stability_rounds=3) == 3


def test_convergence_round_requires_stability():
    # Round 2 spikes to threshold but round 3 dips back below it -> not yet converged there.
    history = _history([0.5, 0.9, 0.5, 0.9, 0.9, 0.9])
    result = convergence_round(history, target_fraction=0.95, stability_rounds=3)
    assert result == 4  # rounds 4,5,6 are the first stable window >= 0.855


def test_convergence_round_never_converges_returns_none():
    history = _history([0.1, 0.2, 0.3])
    assert convergence_round(history, target_fraction=0.95, stability_rounds=3, reference="final") is None


def test_convergence_round_rejects_invalid_reference():
    with pytest.raises(ValueError):
        convergence_round(_history([0.5]), reference="not-a-real-option")


def test_accuracy_degradation_and_robustness_ratio():
    assert accuracy_degradation(0.9, 0.7) == pytest.approx(0.2)
    assert robustness_ratio(0.7, 0.9) == pytest.approx(0.7 / 0.9)


def test_robustness_ratio_zero_baseline_returns_zero_not_error():
    assert robustness_ratio(0.5, 0.0) == 0.0


def test_malicious_influence_ratio():
    assert malicious_influence_ratio(0.3, 1.0) == pytest.approx(0.3)
    assert malicious_influence_ratio(0.3, 0.0) == 0.0


def test_estimate_model_size_bytes_matches_param_count():
    model = nn.Linear(10, 5)  # 10*5 + 5 = 55 params
    expected_params = sum(p.numel() for p in model.parameters())
    assert estimate_model_size_bytes(model, bytes_per_param=4) == expected_params * 4
    assert estimate_model_size_bytes(model, bytes_per_param=2) == expected_params * 2


def test_estimate_communication_cost():
    cost = estimate_communication_cost(model_size_bytes=1000, num_participating_clients=5)
    assert cost.uploaded_bytes == 5000
    assert cost.downloaded_bytes == 5000
    assert cost.total_bytes == 10000
    assert cost.num_participating_clients == 5

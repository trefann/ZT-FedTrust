"""Tests for evaluation/attack_metrics.py and evaluation/security_metrics.py."""

import torch

from clients.client import Client
from clients.malicious_client import MaliciousClient
from evaluation.attack_metrics import (
    compute_attack_success_rate,
    compute_detection_counts,
    compute_detection_metrics,
    compute_malicious_aggregation_weight,
)
from evaluation.security_metrics import (
    build_round_security_records,
    save_attack_history_csv,
    save_security_metrics_csv,
    save_security_plots,
)
from attacks.label_flipping import LabelFlippingAttack
from helpers import make_synthetic_dataset
from models.cnn import build_model


# --------------------------------------------------------------------------
# Detection counts / metrics (hand-computed confusion matrices)
# --------------------------------------------------------------------------


def test_detection_counts_hand_computed():
    is_malicious = {0: True, 1: True, 2: False, 3: False, 4: True}
    eligible_ids = {2, 3, 4}  # client 4 (malicious) slipped through; 0, 1 correctly rejected
    counts = compute_detection_counts(is_malicious, eligible_ids, [0, 1, 2, 3, 4])
    assert counts.true_positives == 2   # 0, 1: malicious + rejected
    assert counts.false_positives == 0  # no benign rejected
    assert counts.true_negatives == 2   # 2, 3: benign + accepted
    assert counts.false_negatives == 1  # 4: malicious + accepted


def test_detection_metrics_formulas():
    from evaluation.attack_metrics import DetectionCounts

    counts = DetectionCounts(true_positives=2, false_positives=1, true_negatives=5, false_negatives=2)
    metrics = compute_detection_metrics(counts)
    assert metrics.precision == 2 / 3
    assert metrics.recall == 2 / 4
    assert metrics.f1 == 2 * (2 / 3 * 0.5) / (2 / 3 + 0.5)
    assert metrics.false_positive_rate == 1 / 6
    assert metrics.false_negative_rate == 2 / 4
    assert metrics.detection_rate == metrics.recall
    assert metrics.accuracy == (2 + 5) / 10


def test_detection_metrics_handles_zero_denominators():
    from evaluation.attack_metrics import DetectionCounts

    counts = DetectionCounts(0, 0, 0, 0)
    metrics = compute_detection_metrics(counts)
    assert metrics.precision == 0.0
    assert metrics.recall == 0.0
    assert metrics.f1 == 0.0
    assert metrics.accuracy == 0.0


def test_attack_success_rate_hand_computed():
    is_malicious = {0: True, 1: True, 2: False}
    eligible_ids = {0, 2}  # only one of the two malicious clients got through
    assert compute_attack_success_rate(is_malicious, eligible_ids, [0, 1, 2]) == 0.5


def test_attack_success_rate_zero_malicious_clients():
    assert compute_attack_success_rate({0: False}, {0}, [0]) == 0.0


def test_malicious_aggregation_weight_hand_computed():
    is_malicious = {0: True, 1: False, 2: True}
    weights = {0: 0.5, 1: 0.3, 2: 0.2}
    assert compute_malicious_aggregation_weight(is_malicious, weights) == 0.7


# --------------------------------------------------------------------------
# build_round_security_records
# --------------------------------------------------------------------------


def _make_pool(num_malicious: int, num_benign: int):
    clients = {}
    cid = 0
    for _ in range(num_benign):
        dataset = make_synthetic_dataset(num_samples=8, seed=cid)
        model = build_model("cnn", in_channels=1, num_classes=10)
        clients[cid] = Client(cid, model, dataset, torch.device("cpu"))
        cid += 1
    for _ in range(num_malicious):
        dataset = make_synthetic_dataset(num_samples=8, seed=cid)
        model = build_model("cnn", in_channels=1, num_classes=10)
        inner = Client(cid, model, dataset, torch.device("cpu"))
        clients[cid] = MaliciousClient(inner, LabelFlippingAttack(), seed=cid)
        cid += 1
    return clients


def test_build_round_security_records_without_selector_everyone_accepted():
    clients = _make_pool(num_malicious=2, num_benign=3)
    record, history = build_round_security_records(
        round_num=1,
        attack_type="label_flipping",
        clients=clients,
        client_selector=None,  # plain FedAvg: no rejection mechanism at all
        selected_ids=list(clients.keys()),
        aggregation_weights={cid: 1 / len(clients) for cid in clients},
    )
    assert record.total_clients == 5
    assert record.malicious_clients == 2
    assert record.detected_malicious == 0  # nothing is ever rejected without a selector
    assert record.undetected_malicious == 2
    assert len(history) == 5


def test_build_round_security_records_history_fields():
    clients = _make_pool(num_malicious=1, num_benign=1)
    malicious_id = next(cid for cid, c in clients.items() if isinstance(c, MaliciousClient))
    record, history = build_round_security_records(
        round_num=3,
        attack_type="label_flipping",
        clients=clients,
        client_selector=None,
        selected_ids=[malicious_id],
        aggregation_weights={malicious_id: 1.0},
    )
    malicious_record = next(r for r in history if r.client_id == malicious_id)
    assert malicious_record.is_malicious is True
    assert malicious_record.attack_type == "label_flipping"
    assert malicious_record.selected is True
    assert malicious_record.aggregation_weight == 1.0

    other_id = next(cid for cid in clients if cid != malicious_id)
    other_record = next(r for r in history if r.client_id == other_id)
    assert other_record.is_malicious is False
    assert other_record.attack_type == "none"
    assert other_record.selected is False
    assert other_record.aggregation_weight == 0.0


# --------------------------------------------------------------------------
# CSV + plot generation
# --------------------------------------------------------------------------


def test_security_csv_and_plots_generated(tmp_path):
    clients = _make_pool(num_malicious=2, num_benign=3)
    security_records = []
    history_records = []
    for round_num in range(1, 3):
        record, round_history = build_round_security_records(
            round_num, "label_flipping", clients, None, list(clients.keys()),
            {cid: 1 / len(clients) for cid in clients},
        )
        security_records.append(record)
        history_records.extend(round_history)

    results_dir = str(tmp_path)
    save_security_metrics_csv(security_records, results_dir)
    save_attack_history_csv(history_records, results_dir)
    save_security_plots(security_records, history_records, results_dir)

    assert (tmp_path / "security_metrics.csv").exists()
    assert (tmp_path / "attack_history.csv").exists()
    plots_dir = tmp_path / "security_plots"
    for filename in (
        "trust_malicious_vs_benign.png",
        "attack_detection_rate.png",
        "malicious_aggregation_weight.png",
        "client_trust_distribution_by_class.png",
    ):
        assert (plots_dir / filename).exists()

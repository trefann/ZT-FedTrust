"""Security/attack-detection metrics: ground-truth malicious labels vs ZT-FedTrust's observed decisions.

This module is the one sanctioned place ground-truth `is_malicious` labels
are read — purely to *score* detection performance after each round, never
to influence it. Nothing in `server/`, `trust/`, or `clients/client.py`
reads `is_malicious`; see `clients/malicious_client.py`'s module docstring.

"Accepted" here means *trust-eligible* (passed the trust threshold), not
merely "selected this round" — matching `server/selection_metrics.py`'s own
`RejectedClients = total - eligible` semantics from Phase 3. A malicious
client that passes the threshold but simply isn't sampled in stage 2 is not
a detection failure; one that passes the threshold *and gets aggregated*
is.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable


@dataclass
class DetectionCounts:
    """Confusion-matrix counts for one round's trust-threshold decisions."""

    true_positives: int   # malicious, rejected (correctly identified)
    false_positives: int  # benign, rejected (wrongly identified)
    true_negatives: int   # benign, accepted (correctly trusted)
    false_negatives: int  # malicious, accepted (missed)


def compute_detection_counts(
    is_malicious: Dict[int, bool], eligible_ids: Iterable[int], all_client_ids: Iterable[int]
) -> DetectionCounts:
    """Compare ground-truth malicious labels against this round's trust-eligibility outcome."""
    eligible = set(eligible_ids)
    true_positives = false_positives = true_negatives = false_negatives = 0
    for client_id in all_client_ids:
        malicious = is_malicious.get(client_id, False)
        accepted = client_id in eligible
        if malicious and not accepted:
            true_positives += 1
        elif not malicious and not accepted:
            false_positives += 1
        elif not malicious and accepted:
            true_negatives += 1
        else:  # malicious and accepted
            false_negatives += 1
    return DetectionCounts(true_positives, false_positives, true_negatives, false_negatives)


@dataclass
class DetectionMetrics:
    """Standard detection-quality metrics derived from `DetectionCounts`."""

    accuracy: float
    precision: float
    recall: float
    f1: float
    false_positive_rate: float
    false_negative_rate: float
    detection_rate: float  # identical formula to recall (TP / (TP + FN)); named separately per spec


def compute_detection_metrics(counts: DetectionCounts) -> DetectionMetrics:
    """Derive accuracy/precision/recall/F1/FPR/FNR/detection-rate from `counts`.

    Every ratio defaults to 0.0 when its denominator is 0 (e.g. no
    malicious clients this round -> recall/detection_rate/FNR are 0.0
    rather than undefined), so callers never need to special-case NaN.
    """
    tp, fp = counts.true_positives, counts.false_positives
    tn, fn = counts.true_negatives, counts.false_negatives
    total = tp + fp + tn + fn

    accuracy = (tp + tn) / total if total else 0.0
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    false_positive_rate = fp / (fp + tn) if (fp + tn) else 0.0
    false_negative_rate = fn / (fn + tp) if (fn + tp) else 0.0

    return DetectionMetrics(
        accuracy=accuracy,
        precision=precision,
        recall=recall,
        f1=f1,
        false_positive_rate=false_positive_rate,
        false_negative_rate=false_negative_rate,
        detection_rate=recall,
    )


def compute_attack_success_rate(
    is_malicious: Dict[int, bool], eligible_ids: Iterable[int], all_client_ids: Iterable[int]
) -> float:
    """Fraction of malicious clients that were trust-eligible (accepted) this round.

        Attack Success Rate = Malicious Clients Accepted / Total Malicious Clients

    Returns 0.0 when there are no malicious clients at all.
    """
    eligible = set(eligible_ids)
    malicious_ids = [cid for cid in all_client_ids if is_malicious.get(cid, False)]
    if not malicious_ids:
        return 0.0
    accepted_malicious = sum(1 for cid in malicious_ids if cid in eligible)
    return accepted_malicious / len(malicious_ids)


def compute_malicious_aggregation_weight(
    is_malicious: Dict[int, bool], aggregation_weights: Dict[int, float]
) -> float:
    """Total normalised aggregation weight contributed by malicious clients this round.

    `aggregation_weights` are already normalised to sum to 1.0 across
    participating clients (see `server.trust_fedavg.compute_aggregation_weights`),
    so this is directly "the fraction of the global model update malicious
    clients are responsible for" — the key number for evaluating whether
    TrustFedAvg actually reduces malicious influence.
    """
    return sum(weight for cid, weight in aggregation_weights.items() if is_malicious.get(cid, False))

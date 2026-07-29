"""Behavioural analysis: Zero Trust dimension 3.

Answers "does this client behave the way a healthy participant should?"
Phase 2 has no real network telemetry, so every round it simulates four
signals — communication latency, packet loss, participation frequency, and
model divergence — from a seeded `numpy.random.Generator`, then combines
them into one Behaviour Score.

Determinism: this class owns its own RNG stream (seeded independently, the
same pattern `data/partition.py` already uses), so re-running the simulator
with the same global seed reproduces identical behavioural metrics, without
disturbing or being disturbed by any other seeded randomness in the pipeline
(client selection, partitioning, local SGD, etc. each have their own streams).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import numpy as np


@dataclass
class BehaviourMetrics:
    """Raw simulated behavioural signals collected for one client in one round."""

    round_num: int
    latency_ms: float
    packet_loss: float
    participation: float
    model_divergence: float
    score: float


class BehaviourAnalyzer:
    """Simulates and scores per-round client behaviour, returning a score in [0, 1].

    Args:
        seed: seed for this analyzer's own RNG stream (independent of any
            other seeded randomness in the simulation).
    """

    # Weights for combining the four behavioural signals into one score.
    # Latency and packet loss are the strongest connectivity-quality signals
    # (0.3 each); participation and divergence are secondary (0.2 each).
    _LATENCY_WEIGHT = 0.3
    _PACKET_LOSS_WEIGHT = 0.3
    _PARTICIPATION_WEIGHT = 0.2
    _DIVERGENCE_WEIGHT = 0.2

    def __init__(self, seed: int = 42) -> None:
        self._rng = np.random.default_rng(seed)
        self.history: Dict[int, List[BehaviourMetrics]] = {}

    def evaluate(self, client_id: int, round_num: int) -> float:
        """Simulate this round's behavioural signals for `client_id` and score them."""
        latency_ms = max(float(self._rng.normal(loc=120.0, scale=30.0)), 0.0)
        packet_loss = float(np.clip(self._rng.normal(loc=0.02, scale=0.02), 0.0, 1.0))
        participation = float(np.clip(self._rng.normal(loc=0.95, scale=0.05), 0.0, 1.0))
        model_divergence = float(np.clip(self._rng.normal(loc=0.10, scale=0.05), 0.0, 1.0))

        # Each raw signal is converted to a "higher is better" sub-score
        # before being combined, so the weighted sum below is meaningful.
        latency_score = float(np.clip(1.0 - latency_ms / 500.0, 0.0, 1.0))
        packet_loss_score = float(np.clip(1.0 - packet_loss, 0.0, 1.0))
        divergence_score = float(np.clip(1.0 - model_divergence, 0.0, 1.0))

        score = float(
            np.clip(
                self._LATENCY_WEIGHT * latency_score
                + self._PACKET_LOSS_WEIGHT * packet_loss_score
                + self._PARTICIPATION_WEIGHT * participation
                + self._DIVERGENCE_WEIGHT * divergence_score,
                0.0,
                1.0,
            )
        )

        self.history.setdefault(client_id, []).append(
            BehaviourMetrics(
                round_num=round_num,
                latency_ms=latency_ms,
                packet_loss=packet_loss,
                participation=participation,
                model_divergence=model_divergence,
                score=score,
            )
        )
        return score

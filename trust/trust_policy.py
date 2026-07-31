"""Trust policy: eligibility rules and threshold strategies for client selection.

Kept isolated from `Server`/`TrustEngine` so selection policy — what counts
as "trusted enough", and how the threshold itself is derived — can evolve
independently of trust *scoring* (Phase 2) and of aggregation (Phase 3's
`server/trust_fedavg.py`). Neither of those modules needs to change if this
one does.
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional

from clients.client import Client


@dataclass
class SelectionStatistics:
    """Summary statistics over the most recently evaluated set of clients."""

    total_clients: int
    eligible_clients: int
    average_trust: float
    min_trust: float
    max_trust: float
    threshold_used: float


def fixed_threshold_strategy(clients: List[Client], configured_threshold: float) -> float:
    """Always return the configured threshold, ignoring current trust scores."""
    return configured_threshold


def adaptive_threshold_strategy(clients: List[Client], configured_threshold: float) -> float:
    """Derive a threshold from the current round's trust distribution.

        Threshold = Mean(Trust) - 0.5 * Stdev(Trust)

    Falls back to `configured_threshold` when fewer than 2 clients are given
    (a standard deviation isn't meaningful over 0 or 1 samples).
    `configured_threshold` is otherwise unused here, but kept in the
    signature so both strategies share one interface (`ThresholdStrategy`)
    and callers never need to special-case which mode is active.
    """
    if len(clients) < 2:
        return configured_threshold
    scores = [client.trust_score for client in clients]
    mean_trust = statistics.mean(scores)
    stdev_trust = statistics.stdev(scores)
    return mean_trust - 0.5 * stdev_trust


ThresholdStrategy = Callable[[List[Client], float], float]

_THRESHOLD_STRATEGIES: Dict[str, ThresholdStrategy] = {
    "fixed": fixed_threshold_strategy,
    "adaptive": adaptive_threshold_strategy,
}

SUPPORTED_THRESHOLD_MODES = tuple(_THRESHOLD_STRATEGIES.keys())


class TrustPolicy:
    """Decides which clients are trusted enough to participate, each round.

    Args:
        threshold_mode: `"fixed"` or `"adaptive"` — selects a `ThresholdStrategy`.
        threshold: the configured threshold. Used directly in `"fixed"` mode;
            used only as a fallback in `"adaptive"` mode (see
            `adaptive_threshold_strategy`).

    Usage: call `get_trusted_clients(clients)` once per round to compute the
    threshold and get the eligible subset; `is_trusted`/
    `calculate_selection_statistics` then report on that same evaluation.
    """

    def __init__(self, threshold_mode: str = "fixed", threshold: float = 0.60) -> None:
        if threshold_mode not in _THRESHOLD_STRATEGIES:
            raise ValueError(
                f"Unknown threshold_mode: {threshold_mode!r}. Available: {SUPPORTED_THRESHOLD_MODES}"
            )
        self.threshold_mode = threshold_mode
        self.configured_threshold = threshold
        self._strategy = _THRESHOLD_STRATEGIES[threshold_mode]
        self._last_threshold: float = threshold
        self._last_clients: List[Client] = []

    def is_trusted(self, client: Client) -> bool:
        """Return whether `client` meets the most recently computed threshold."""
        return client.trust_score >= self._last_threshold

    def get_trusted_clients(self, clients: List[Client]) -> List[Client]:
        """Compute this round's threshold from `clients`, then return the eligible subset.

        Subsequent calls to `is_trusted` and `calculate_selection_statistics`
        report against this same evaluation, until the next call.
        """
        self._last_threshold = self._strategy(clients, self.configured_threshold)
        self._last_clients = list(clients)
        return [client for client in self._last_clients if self.is_trusted(client)]

    def calculate_selection_statistics(self) -> SelectionStatistics:
        """Summarise the most recent `get_trusted_clients` evaluation."""
        clients = self._last_clients
        scores = [client.trust_score for client in clients]
        eligible = [client for client in clients if self.is_trusted(client)]
        return SelectionStatistics(
            total_clients=len(clients),
            eligible_clients=len(eligible),
            average_trust=statistics.mean(scores) if scores else 0.0,
            min_trust=min(scores) if scores else 0.0,
            max_trust=max(scores) if scores else 0.0,
            threshold_used=self._last_threshold,
        )

"""Trust-weighted aggregation, and the aggregation strategy pattern.

Standard FedAvg (`server/fedavg.py`, unchanged since Phase 1):

    Sum(ClientWeights * Samples) / Sum(Samples)

TrustFedAvg (this module):

    Sum(ClientWeights * Samples * Trust) / Sum(Samples * Trust)

Both strategies are unified behind one calling convention — a list of
`ClientUpdate` and a shared `AggregationStrategy` signature returning
`(aggregated_state, per_client_weights)` — so `Server` can dispatch to
either without an `if` statement anywhere near the training loop, and so a
new strategy only needs to be written once and registered in `_STRATEGIES`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Tuple

import torch

from server.fedavg import federated_average


@dataclass
class ClientUpdate:
    """One client's contribution to this round's aggregation."""

    client_id: int
    state_dict: Dict[str, torch.Tensor]
    num_samples: int
    trust: float = 1.0


def compute_aggregation_weights(updates: List[ClientUpdate], use_trust: bool) -> Dict[int, float]:
    """Compute each client's normalised aggregation weight for this round.

    Single source of truth for "how much does each client's update count",
    used both by the actual aggregation math below and by
    `server/selection_metrics.py` for `aggregation_weights.csv` — so the
    logged weights can never drift out of sync with the weights actually
    applied.

    Args:
        use_trust: if `True`, weight by `num_samples * trust` (TrustFedAvg);
            if `False`, weight by `num_samples` alone (plain FedAvg,
            mathematically identical to `server.fedavg.federated_average`).

    Returns weights normalised to sum to 1.0 across `updates`.
    """
    if not updates:
        raise ValueError("Cannot compute aggregation weights for an empty list of updates.")

    if use_trust:
        raw_weights = {update.client_id: update.num_samples * update.trust for update in updates}
    else:
        raw_weights = {update.client_id: float(update.num_samples) for update in updates}

    total = sum(raw_weights.values())
    if total <= 0:
        raise ValueError("Total aggregation weight must be positive (check samples/trust values).")

    return {client_id: value / total for client_id, value in raw_weights.items()}


def trust_weighted_average(
    updates: List[ClientUpdate],
) -> Tuple[Dict[str, torch.Tensor], Dict[int, float]]:
    """TrustFedAvg: aggregate client updates weighted by both sample count and trust.

    Higher-trust clients contribute proportionally more to the global model;
    a client with `trust=0` contributes nothing, without needing to be
    excluded from `updates` beforehand.
    """
    weights = compute_aggregation_weights(updates, use_trust=True)
    param_keys = updates[0].state_dict.keys()

    aggregated: Dict[str, torch.Tensor] = {}
    for key in param_keys:
        weighted_sum = sum(
            update.state_dict[key].float() * weights[update.client_id] for update in updates
        )
        aggregated[key] = weighted_sum.to(updates[0].state_dict[key].dtype)

    return aggregated, weights


def fedavg_strategy(
    updates: List[ClientUpdate],
) -> Tuple[Dict[str, torch.Tensor], Dict[int, float]]:
    """Standard FedAvg, adapted to the unified `ClientUpdate`/weights interface.

    Delegates the actual aggregation math to `server.fedavg.federated_average`
    (untouched since Phase 1) so there is exactly one implementation of plain
    sample-weighted averaging anywhere in the codebase; only the weights
    (used for `aggregation_weights.csv` logging) are computed here.
    """
    aggregated = federated_average([(update.state_dict, update.num_samples) for update in updates])
    weights = compute_aggregation_weights(updates, use_trust=False)
    return aggregated, weights


AggregationStrategy = Callable[[List[ClientUpdate]], Tuple[Dict[str, torch.Tensor], Dict[int, float]]]

_STRATEGIES: Dict[str, AggregationStrategy] = {
    "fedavg": fedavg_strategy,
    "trust_fedavg": trust_weighted_average,
}

SUPPORTED_AGGREGATION_STRATEGIES = tuple(_STRATEGIES.keys())


def get_aggregation_strategy(name: str) -> AggregationStrategy:
    """Look up an aggregation strategy by name, for config-driven dispatch.

    This is the only place a new strategy needs to be registered — callers
    (e.g. `Server`) never branch on the strategy name themselves.
    """
    if name not in _STRATEGIES:
        raise ValueError(
            f"Unknown aggregation strategy: {name!r}. Available: {SUPPORTED_AGGREGATION_STRATEGIES}"
        )
    return _STRATEGIES[name]

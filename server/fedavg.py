"""Federated Averaging (FedAvg) aggregation algorithm."""

from __future__ import annotations

from typing import Dict, List, Tuple

import torch


def federated_average(
    updates: List[Tuple[Dict[str, torch.Tensor], int]]
) -> Dict[str, torch.Tensor]:
    """Aggregate client model updates via sample-size-weighted averaging.

    Args:
        updates: list of (state_dict, num_samples) pairs, one per participating
            client, as produced by `Client.train`.

    Returns:
        The aggregated state dict, ready to load into the global model.
    """
    if not updates:
        raise ValueError("Cannot aggregate an empty list of client updates.")

    total_samples = sum(num_samples for _, num_samples in updates)
    if total_samples == 0:
        raise ValueError("Total sample count across clients is zero.")

    param_keys = updates[0][0].keys()
    aggregated: Dict[str, torch.Tensor] = {}
    for key in param_keys:
        weighted_sum = sum(
            state_dict[key].float() * (num_samples / total_samples)
            for state_dict, num_samples in updates
        )
        aggregated[key] = weighted_sum.to(updates[0][0][key].dtype)
    return aggregated

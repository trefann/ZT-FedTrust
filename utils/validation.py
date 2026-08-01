"""Tensor validation utilities: guard against corrupted model weights (Phase 4, safety)."""

from __future__ import annotations

from typing import Dict

import torch


def validate_state_dict(state_dict: Dict[str, torch.Tensor], context: str) -> None:
    """Raise `ValueError` if any tensor in `state_dict` contains NaN or infinite values.

    Called after an attack manipulates a client's update
    (`clients/malicious_client.py`) and again after aggregation
    (`server/server.py`), so a misconfigured or numerically unstable attack
    can never silently propagate a corrupted state dict into the global
    model.
    """
    for key, tensor in state_dict.items():
        if not torch.isfinite(tensor).all():
            raise ValueError(
                f"{context}: non-finite values (NaN or Inf) detected in parameter {key!r} — "
                "refusing to proceed with a corrupted state dict."
            )

"""Base interface for all adversarial attack strategies (Phase 4).

`MaliciousClient` (`clients/malicious_client.py`) is the only thing that
calls into these hooks — the server, `Client`, and `trust/` code never
import from this package at all, keeping attack logic completely separate
from the normal federated learning pipeline.
"""

from __future__ import annotations

from typing import Dict

import torch
from torch.utils.data import Dataset


class BaseAttack:
    """Common interface for dataset-level and update-level attacks.

    Subclasses override only the hook(s) their attack needs:
      - dataset-level attacks (e.g. label flipping) override `apply_to_dataset`
      - update-level attacks (e.g. model poisoning, Byzantine) override `apply_to_update`

    Both hooks default to a no-op passthrough here, rather than being
    `@abstractmethod`: making both abstract would force every attack to
    implement both, which contradicts "not every attack needs both
    methods". `MaliciousClient` always calls both hooks unconditionally;
    the no-op defaults make that safe for any attack that only cares about
    one of them.
    """

    name: str = "base"

    def apply_to_dataset(self, dataset: Dataset) -> Dataset:
        """Return a dataset for local training. Must not mutate `dataset` in place."""
        return dataset

    def apply_to_update(
        self, state_dict: Dict[str, torch.Tensor], reference_state_dict: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """Return a (possibly modified) copy of a client's trained state dict.

        Args:
            state_dict: the client's honestly-trained local weights.
            reference_state_dict: the global weights the client trained
                from this round — available as a baseline for attacks that
                need one (e.g. computing `trained - reference` as the
                client's real update before manipulating it).
        """
        return state_dict

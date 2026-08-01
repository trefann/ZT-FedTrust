"""Sybil attack: Phase 4 simulation-level attack.

Simulates one adversarial entity controlling multiple client identities
("Sybils"), each performing the same underlying malicious behaviour. This
class itself is a thin `BaseAttack` delegate — the group of identities
(shared `sybil_group_id`, common `parent_attacker_id`) is assembled by
`clients.malicious_client.assign_attacks`, not by this class.

This is a purely simulation-level construct: it does not implement any
real-world identity attack (certificate forgery, Sybil-resistant consensus
bypass, Eclipse attacks, etc.) — just multiple `MaliciousClient` instances
that happen to share bookkeeping metadata.
"""

from __future__ import annotations

from typing import Dict

import torch
from torch.utils.data import Dataset

from attacks.base_attack import BaseAttack


class SybilAttack(BaseAttack):
    """Delegates to an underlying attack; exists to be tagged with Sybil identity.

    Args:
        underlying_attack: the actual malicious behaviour each Sybil
            identity performs (e.g. a `LabelFlippingAttack()`).
    """

    name = "sybil"

    def __init__(self, underlying_attack: BaseAttack) -> None:
        self.underlying_attack = underlying_attack

    def apply_to_dataset(self, dataset: Dataset) -> Dataset:
        return self.underlying_attack.apply_to_dataset(dataset)

    def apply_to_update(
        self, state_dict: Dict[str, torch.Tensor], reference_state_dict: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        return self.underlying_attack.apply_to_update(state_dict, reference_state_dict)

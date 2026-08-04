"""Real-attack-traffic attack: a fifth Phase-4-style adversarial strategy,
added alongside label flipping / model poisoning / Byzantine / Sybil.

Unlike those four (all synthetic transformations of benign data or honest
updates), this models a compromised IoT device whose local dataset is
itself real malicious traffic — e.g. genuine Mirai/Gafgyt-infected N-BaIoT
captures, or any IoT dataset's own attack-labeled rows — rather than a
label-flip/noise/scale perturbation.

Ground-truth isolation is preserved exactly like `label_flipping`: this
attack only ever reads the dataset's own labels (real content, not
simulation ground truth), never `client.is_malicious` — the server, trust
engine, and selector still cannot distinguish this client from any other
without the same `evaluation/`-only `isinstance` check every other attack
already relies on.
"""

from __future__ import annotations

import random

from torch.utils.data import Dataset, Subset

from attacks.base_attack import BaseAttack


class RealAttackTrafficAttack(BaseAttack):
    """Replaces a client's local training data with a sample of real
    attack-labeled traffic, sized to match the client's original dataset
    (keeps per-round training time comparable to the other attacks)."""

    name = "real_traffic"

    def __init__(self, attack_pool: Dataset, seed: int = 42) -> None:
        if len(attack_pool) == 0:
            raise ValueError("RealAttackTrafficAttack requires a non-empty attack_pool")
        self.attack_pool = attack_pool
        self._rng = random.Random(seed)

    def apply_to_dataset(self, dataset: Dataset) -> Dataset:
        sample_size = min(len(dataset), len(self.attack_pool))
        indices = self._rng.sample(range(len(self.attack_pool)), sample_size)
        return Subset(self.attack_pool, indices)

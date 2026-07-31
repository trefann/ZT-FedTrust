"""Trust-aware client selection: stage 1 (trust threshold) + stage 2 (participation fraction).

    All Clients -> Evaluate Trust Scores -> Evaluate Trust Threshold ->
    Eligible Clients -> Participation Fraction -> Selected Clients

Stage 1 delegates entirely to `TrustPolicy` (kept isolated in `trust/`, per
the Zero Trust design). Stage 2 — sampling a fraction of the eligible pool —
is this module's own responsibility, since it's a participation-scheduling
concern, not a trust-policy one.
"""

from __future__ import annotations

import random
from typing import Dict, List

from clients.client import Client
from trust.trust_policy import TrustPolicy


class ClientSelector:
    """Selects which clients participate in a round via trust threshold + participation fraction.

    Args:
        policy: the `TrustPolicy` used for stage 1 (trust-threshold filtering).
        participation_fraction: fraction of the *eligible* pool to sample in
            stage 2, in (0, 1].
        seed: seed for this selector's own `random.Random` stream, kept
            independent of `Server`'s legacy selection RNG so trust-based and
            random selection never interfere with each other's sequences.
    """

    def __init__(self, policy: TrustPolicy, participation_fraction: float, seed: int = 42) -> None:
        if not 0.0 < participation_fraction <= 1.0:
            raise ValueError(
                f"participation_fraction must be in (0, 1], got {participation_fraction}"
            )
        self.policy = policy
        self.participation_fraction = participation_fraction
        self._rng = random.Random(seed)

    def select(self, clients: Dict[int, Client]) -> List[int]:
        """Return the client IDs selected to participate in the current round.

        Stage 1: `self.policy.get_trusted_clients` filters out clients below
        the trust threshold (fixed or adaptive — the policy decides).
        Stage 2: sample `participation_fraction` of the remaining eligible
        pool, rounded to the nearest whole client (at least one, if any are
        eligible).

        Returns an empty list if no client meets the trust threshold this
        round — callers should treat that as a real (if rare) failure state,
        not paper over it, since it means the whole client population has
        become untrusted.
        """
        eligible_clients = self.policy.get_trusted_clients(list(clients.values()))
        if not eligible_clients:
            return []

        num_to_select = max(1, round(len(eligible_clients) * self.participation_fraction))
        num_to_select = min(num_to_select, len(eligible_clients))

        eligible_ids = sorted(client.client_id for client in eligible_clients)
        return self._rng.sample(eligible_ids, num_to_select)

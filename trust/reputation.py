"""Historical reputation tracking: Zero Trust dimension 4.

Answers "how has this client behaved over time, not just this round?"
Reputation is an exponential moving average over each round's trust signal,
so a single bad round dents it without erasing a long history of good
behaviour, and a client can't instantly recover full trust after one good
round either.

    Reputation_new = alpha * Reputation_old + (1 - alpha) * CurrentTrust

`CurrentTrust` here is deliberately the *per-round* signal computed from the
other four dimensions (see `TrustEngine`), not the final composite trust
score — the composite score itself uses this round's reputation as an input,
so feeding it back into its own update would be circular.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List


@dataclass
class ReputationRecord:
    """One client's reputation value after a given round."""

    round_num: int
    reputation: float


class ReputationManager:
    """Maintains an exponentially-smoothed reputation score per client.

    Args:
        initial_reputation: reputation assigned to a client the first time
            it's seen, before any round has been observed.
        alpha: smoothing factor in [0, 1]. Higher alpha weighs history more
            heavily; lower alpha reacts faster to the current round.
    """

    def __init__(self, initial_reputation: float = 0.80, alpha: float = 0.80) -> None:
        if not 0.0 <= initial_reputation <= 1.0:
            raise ValueError(f"initial_reputation must be in [0, 1], got {initial_reputation}")
        if not 0.0 <= alpha <= 1.0:
            raise ValueError(f"alpha must be in [0, 1], got {alpha}")
        self.initial_reputation = initial_reputation
        self.alpha = alpha
        self._current: Dict[int, float] = {}
        self.history: Dict[int, List[ReputationRecord]] = {}

    def get(self, client_id: int) -> float:
        """Return the client's current reputation, initializing it if unseen."""
        return self._current.setdefault(client_id, self.initial_reputation)

    def update(self, client_id: int, round_num: int, current_trust: float) -> float:
        """Apply the smoothing update for one round and return the new reputation."""
        old_reputation = self.get(client_id)
        new_reputation = self.alpha * old_reputation + (1.0 - self.alpha) * current_trust
        new_reputation = float(min(max(new_reputation, 0.0), 1.0))

        self._current[client_id] = new_reputation
        self.history.setdefault(client_id, []).append(ReputationRecord(round_num, new_reputation))
        return new_reputation

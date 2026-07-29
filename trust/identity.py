"""Identity verification: Zero Trust dimension 1.

Answers "is this client who it claims to be?" Phase 2 uses a deterministic,
registry-based scheme — every client ID the simulator itself created is
treated as verified — because there is no real network identity to check in
a single-process simulation. The class exists as a seam: a later phase can
swap in certificate- or token-based authentication by replacing the body of
`verify` without touching `TrustEngine` or any other trust module, since the
public contract (`verify(client_id) -> float`) stays the same.
"""

from __future__ import annotations

from typing import Iterable, Optional, Set


class IdentityVerifier:
    """Verifies client identity and returns a confidence score in [0, 1].

    Args:
        known_client_ids: client IDs considered identity-verified. If `None`,
            every client ID is treated as verified (useful before the
            simulator has registered its client set, or in tests).
        verified_score: score returned for a known/verified client.
        unknown_score: score returned for an unrecognised client — configurable
            rather than hardcoded, since how harshly to penalise an unknown
            identity is a policy decision, not a fact derivable from data.
    """

    def __init__(
        self,
        known_client_ids: Optional[Set[int]] = None,
        verified_score: float = 1.0,
        unknown_score: float = 0.3,
    ) -> None:
        self._known_client_ids = known_client_ids
        self.verified_score = verified_score
        self.unknown_score = unknown_score

    def register_known_clients(self, client_ids: Iterable[int]) -> None:
        """Register the set of client IDs considered identity-verified."""
        self._known_client_ids = set(client_ids)

    def verify(self, client_id: int) -> float:
        """Return an identity confidence score for `client_id`, in [0, 1]."""
        if self._known_client_ids is None or client_id in self._known_client_ids:
            return self.verified_score
        return self.unknown_score

"""Device integrity evaluation: Zero Trust dimension 2.

Answers "can this client's device be trusted to run training faithfully?"
Phase 2 has no real device attestation to query, so each client carries a
simulated `DeviceProfile` (firmware/secure-boot/integrity-flag booleans plus
a continuous health score) that defaults to "fully healthy". The scoring
function is a simple, transparent combination so it stays easy to reason
about, and easy to replace once real attestation data (TPM quotes, MDM
posture, etc.) is available — only `evaluate` would need to change.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional


@dataclass
class DeviceProfile:
    """Simulated device health signals for one client."""

    firmware_ok: bool = True
    secure_boot_enabled: bool = True
    integrity_flag_clear: bool = True
    health_score: float = 1.0  # e.g. combined OS/patch-level health, in [0, 1]


class DeviceIntegrityEvaluator:
    """Evaluates simulated device health/integrity, returning a score in [0, 1].

    Args:
        profiles: optional initial mapping of client ID to `DeviceProfile`.
            Clients without a registered profile default to a fully-healthy
            `DeviceProfile()`.
    """

    def __init__(self, profiles: Optional[Dict[int, DeviceProfile]] = None) -> None:
        self._profiles: Dict[int, DeviceProfile] = dict(profiles) if profiles else {}

    def set_profile(self, client_id: int, profile: DeviceProfile) -> None:
        """Register or override the simulated device profile for a client."""
        self._profiles[client_id] = profile

    def evaluate(self, client_id: int) -> float:
        """Return a device integrity score for `client_id`, in [0, 1].

        Combines the three boolean checks (equally weighted) with the
        continuous health score (equally weighted against the boolean
        block), so a single failed check degrades trust but doesn't zero it
        out outright.
        """
        profile = self._profiles.get(client_id, DeviceProfile())
        boolean_score = sum(
            (profile.firmware_ok, profile.secure_boot_enabled, profile.integrity_flag_clear)
        ) / 3.0
        combined = 0.5 * boolean_score + 0.5 * profile.health_score
        return float(min(max(combined, 0.0), 1.0))

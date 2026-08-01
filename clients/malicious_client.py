"""Malicious client wrapper and attack-assignment logic (Phase 4).

`MaliciousClient` wraps a benign `Client` via composition — Client + Attack
Strategy — rather than duplicating or subclassing the whole `Client`
implementation. It exposes the same attributes/methods `Server` and
`TrustEngine` use on a plain `Client`, so it's a drop-in replacement
wherever `Server.clients` expects one: no server or trust-engine changes
were needed to support malicious clients.

Ground-truth honesty (critical, see Phase 4 spec requirement #11):
`is_malicious`/`attack_type` exist on this class purely as labels for
`evaluation/attack_metrics.py` to score detection performance after the
fact. Nothing in `server/`, `trust/`, or `clients/client.py` ever reads
them — `Server` and `TrustEngine` only ever see the same duck-typed
surface a benign `Client` exposes (trust scores, `get_weights`,
`set_weights`, `train`), with no way to distinguish a `MaliciousClient`
from a `Client` without explicitly checking `isinstance`.
"""

from __future__ import annotations

import random
from typing import Any, Dict, List, Optional, Set, Union

import torch
import torch.nn as nn

from attacks.base_attack import BaseAttack
from attacks.byzantine import ByzantineAttack
from attacks.label_flipping import LabelFlippingAttack
from attacks.model_poisoning import ModelPoisoningAttack
from attacks.sybil import SybilAttack
from clients.client import Client, LocalTrainResult
from utils.validation import validate_state_dict


class MaliciousClient:
    """Wraps a `Client` with an attack strategy applied during training.

    Each round, with probability `attack_probability` (its own seeded
    decision, independent of every other stream in the simulation), the
    configured attack is applied: `attack.apply_to_dataset` before local
    training (for dataset-level attacks), and `attack.apply_to_update`
    after (for update-level attacks). With the complementary probability,
    this client trains and reports exactly like a benign one — modelling
    an intermittent/stealthy attacker rather than an always-on one.

    Args:
        client: the wrapped benign `Client`.
        attack: the attack strategy to apply.
        attack_probability: per-round probability of actually attacking, in [0, 1].
        sybil_group_id: shared identifier for all identities controlled by
            the same simulated Sybil attacker (`None` for non-Sybil attacks).
        parent_attacker_id: the "real" client ID behind a Sybil group
            (`None` for non-Sybil attacks).
        seed: seed for this client's own per-round attack-trigger RNG.
    """

    def __init__(
        self,
        client: Client,
        attack: BaseAttack,
        attack_probability: float = 1.0,
        sybil_group_id: Optional[str] = None,
        parent_attacker_id: Optional[int] = None,
        seed: int = 42,
    ) -> None:
        if not 0.0 <= attack_probability <= 1.0:
            raise ValueError(f"attack_probability must be in [0, 1], got {attack_probability}")

        self.client = client
        self.attack = attack
        self.is_malicious = True
        self.attack_type = attack.name
        self.attack_probability = attack_probability
        self.sybil_group_id = sybil_group_id
        self.parent_attacker_id = parent_attacker_id

        self.client_id = client.client_id
        self.device = client.device
        self.dataset = client.dataset

        self._rng = random.Random(seed)
        self._reference_state: Optional[Dict[str, torch.Tensor]] = None
        self.last_round_attacked: bool = False

    # -- Trust-attribute proxies: read/write straight through to the -------
    # -- wrapped Client, so TrustEngine's writes (`client.trust_score = x`) -
    # -- land on the same object this client's own training logic reads. --
    @property
    def identity_score(self) -> float:
        return self.client.identity_score

    @identity_score.setter
    def identity_score(self, value: float) -> None:
        self.client.identity_score = value

    @property
    def integrity_score(self) -> float:
        return self.client.integrity_score

    @integrity_score.setter
    def integrity_score(self, value: float) -> None:
        self.client.integrity_score = value

    @property
    def behaviour_score(self) -> float:
        return self.client.behaviour_score

    @behaviour_score.setter
    def behaviour_score(self, value: float) -> None:
        self.client.behaviour_score = value

    @property
    def reputation_score(self) -> float:
        return self.client.reputation_score

    @reputation_score.setter
    def reputation_score(self, value: float) -> None:
        self.client.reputation_score = value

    @property
    def update_quality_score(self) -> float:
        return self.client.update_quality_score

    @update_quality_score.setter
    def update_quality_score(self, value: float) -> None:
        self.client.update_quality_score = value

    @property
    def trust_score(self) -> float:
        return self.client.trust_score

    @trust_score.setter
    def trust_score(self, value: float) -> None:
        self.client.trust_score = value

    @property
    def trust_history(self) -> List[Dict[str, Any]]:
        return self.client.trust_history

    @property
    def model(self) -> nn.Module:
        return self.client.model

    def get_weights(self) -> Dict[str, torch.Tensor]:
        return self.client.get_weights()

    def set_weights(self, state_dict: Dict[str, torch.Tensor]) -> None:
        # Keep a reference copy of the pre-training global weights for
        # update-level attacks (they need a baseline to compute the
        # client's real update before manipulating it).
        self._reference_state = {key: value.detach().clone() for key, value in state_dict.items()}
        self.client.set_weights(state_dict)

    def train(self) -> LocalTrainResult:
        """Train locally, applying this round's attack with probability `attack_probability`."""
        self.last_round_attacked = self._rng.random() < self.attack_probability

        original_dataset = self.client.dataset
        if self.last_round_attacked:
            self.client.dataset = self.attack.apply_to_dataset(original_dataset)
        try:
            result = self.client.train()
        finally:
            self.client.dataset = original_dataset  # never leave the wrapped Client permanently altered

        if self.last_round_attacked:
            if self._reference_state is None:
                raise RuntimeError(f"Client {self.client_id}: train() called before set_weights().")
            poisoned_state = self.attack.apply_to_update(result.state_dict, self._reference_state)
            validate_state_dict(poisoned_state, context=f"Client {self.client_id} ({self.attack_type})")
            result = LocalTrainResult(
                client_id=result.client_id,
                state_dict=poisoned_state,
                num_samples=result.num_samples,
                average_loss=result.average_loss,
            )
        return result


def build_attack(active_attack: str, attack_settings: Dict[str, Dict[str, Any]], seed: int) -> BaseAttack:
    """Construct the configured attack strategy by name.

    `attack_settings` is the flattened `{"label_flipping": {...}, ...}` dict
    produced by `simulation.run.AttackConfig` (kept as plain dicts here,
    rather than importing that dataclass, to avoid a circular import between
    `simulation/run.py` and this module).
    """
    if active_attack == "label_flipping":
        settings = attack_settings.get("label_flipping", {})
        return LabelFlippingAttack(label_map=settings.get("label_map"))
    if active_attack == "model_poisoning":
        settings = attack_settings.get("model_poisoning", {})
        return ModelPoisoningAttack(scale_factor=settings.get("scale_factor", 5.0), seed=seed)
    if active_attack == "byzantine":
        settings = attack_settings.get("byzantine", {})
        return ByzantineAttack(strategy=settings.get("strategy", "random"), seed=seed)
    if active_attack == "sybil":
        # Sybils need a concrete underlying behaviour; label flipping is the
        # simplest deterministic default. See `attacks/sybil.py`.
        return SybilAttack(LabelFlippingAttack())
    raise ValueError(f"Unknown active_attack: {active_attack!r}")


def assign_attacks(
    clients: Dict[int, Client],
    malicious_client_fraction: float,
    active_attack: str,
    attack_settings: Dict[str, Dict[str, Any]],
    seed: int,
) -> Dict[int, Union[Client, MaliciousClient]]:
    """Deterministically mark a fraction of `clients` as malicious.

    Returns a new dict with the same keys as `clients`: malicious client IDs
    map to a `MaliciousClient` wrapping the original `Client`; the rest map
    to their original, untouched `Client`. Uses its own seeded RNG,
    independent of every other stream in the simulation, so which clients
    are malicious (and, for Sybil, how they're grouped) is reproducible
    given the same seed — and, since sampling always draws from the full
    sorted client-ID list first, is the *same* set of malicious clients
    regardless of which experiment mode (fedavg_attack vs ztfedtrust_attack)
    calls this with the same config, satisfying the fairness requirement
    that both see identical attackers.

    Attack assignment is fixed for the whole run — this function is called
    once at simulation startup, not per round.
    """
    if not 0.0 <= malicious_client_fraction <= 1.0:
        raise ValueError(
            f"malicious_client_fraction must be in [0, 1], got {malicious_client_fraction}"
        )

    client_ids = sorted(clients.keys())
    num_malicious = round(len(client_ids) * malicious_client_fraction)
    rng = random.Random(seed)
    malicious_ids = set(rng.sample(client_ids, num_malicious))

    result: Dict[int, Union[Client, MaliciousClient]] = {}

    if active_attack == "sybil" and malicious_ids:
        sybil_settings = attack_settings.get("sybil", {})
        group_size = max(1, int(sybil_settings.get("number_of_identities", 3)))
        probability = sybil_settings.get("probability", 1.0)
        malicious_ids_sorted = sorted(malicious_ids)
        for start in range(0, len(malicious_ids_sorted), group_size):
            group_ids = malicious_ids_sorted[start : start + group_size]
            parent_id = group_ids[0]
            group_label = f"sybil-{parent_id}"
            for client_id in group_ids:
                attack = build_attack("sybil", attack_settings, seed=seed + client_id)
                result[client_id] = MaliciousClient(
                    clients[client_id],
                    attack,
                    attack_probability=probability,
                    sybil_group_id=group_label,
                    parent_attacker_id=parent_id,
                    seed=seed + client_id,
                )
    else:
        settings = attack_settings.get(active_attack, {})
        probability = settings.get("probability", 1.0)
        for client_id in malicious_ids:
            attack = build_attack(active_attack, attack_settings, seed=seed + client_id)
            result[client_id] = MaliciousClient(
                clients[client_id], attack, attack_probability=probability, seed=seed + client_id
            )

    for client_id, client in clients.items():
        if client_id not in result:
            result[client_id] = client
    return result


def known_identity_ids(clients: Dict[int, Union[Client, MaliciousClient]]) -> Set[int]:
    """Client IDs that should be registered as identity-verified.

    Reuses Phase 2's *existing* `IdentityVerifier` known/unknown
    distinction (`trust/identity.py`) rather than adding new trust logic:
    every client is "known" except non-parent Sybil identities, which are
    deliberately left unregistered so they fall back to
    `IdentityVerifier`'s configurable `unknown_score`. This models a
    realistic identity-verification signal — e.g. multiple registrations
    sharing a device fingerprint — not a ground-truth "this client is
    malicious" leak: `sybil_group_id`/`parent_attacker_id` describe an
    observable structural fact (shared identity), never `is_malicious`.
    """
    known: Set[int] = set()
    for client_id, client in clients.items():
        if isinstance(client, MaliciousClient) and client.parent_attacker_id is not None:
            if client_id == client.parent_attacker_id:
                known.add(client_id)
            # else: a non-parent Sybil duplicate — left unregistered.
        else:
            known.add(client_id)
    return known

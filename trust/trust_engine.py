"""Zero Trust engine: orchestrates all five trust dimensions into one composite score.

This is the only module that imports all five evaluators — each dimension
module (`identity`, `integrity`, `behaviour`, `reputation`, `update_quality`)
knows nothing about the others, so they can be tested, replaced, or extended
independently. `TrustEngine` wires them together via constructor injection
(see `__init__`) rather than constructing its own evaluators, so tests (and
later phases) can swap in fakes or reconfigured instances without editing
this class.

Phase 2 scope: this engine only *computes and records* trust. Nothing here
selects clients, weights aggregation, or rejects updates — that is Phase 3+.
`Server` calls `evaluate_round` purely for its side effects (writing onto
each `Client` and appending to `self.records`); its return value is not fed
back into FedAvg.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, Iterable, List

import torch

from clients.client import Client, LocalTrainResult
from trust.behaviour import BehaviourAnalyzer
from trust.identity import IdentityVerifier
from trust.integrity import DeviceIntegrityEvaluator
from trust.reputation import ReputationManager
from trust.trust_metrics import TrustRecord, print_round_summary
from trust.update_quality import UpdateQualityEvaluator


@dataclass
class TrustWeights:
    """Weights for combining the five trust dimensions into one composite score.

    Need not sum to exactly 1.0 — `TrustEngine` normalises by their total,
    so relative proportions are what matter.
    """

    identity: float = 0.25
    integrity: float = 0.20
    behaviour: float = 0.20
    reputation: float = 0.15
    update_quality: float = 0.20

    def as_dict(self) -> Dict[str, float]:
        """Return the weights as a plain dict, for iteration/summation."""
        return {
            "identity": self.identity,
            "integrity": self.integrity,
            "behaviour": self.behaviour,
            "reputation": self.reputation,
            "update_quality": self.update_quality,
        }


class TrustEngine:
    """Computes a dynamic Zero Trust score for every client, every round.

    Args:
        weights: relative importance of each trust dimension in the final
            composite score.
        initial_reputation: starting reputation for a client's first round.
        reputation_alpha: reputation smoothing factor (see `ReputationManager`).
        seed: seed for `BehaviourAnalyzer`'s simulated metrics, so re-running
            with the same global seed reproduces identical trust evolution.
        identity_verifier, integrity_evaluator, behaviour_analyzer,
            reputation_manager, update_quality_evaluator: optional overrides
            for dependency injection (e.g. in tests, or to plug in a
            differently-configured evaluator). Each defaults to a standard
            instance built from the arguments above.
    """

    def __init__(
        self,
        weights: TrustWeights,
        initial_reputation: float = 0.80,
        reputation_alpha: float = 0.80,
        seed: int = 42,
        identity_verifier: IdentityVerifier | None = None,
        integrity_evaluator: DeviceIntegrityEvaluator | None = None,
        behaviour_analyzer: BehaviourAnalyzer | None = None,
        reputation_manager: ReputationManager | None = None,
        update_quality_evaluator: UpdateQualityEvaluator | None = None,
    ) -> None:
        self.weights = weights
        self.identity_verifier = identity_verifier or IdentityVerifier()
        self.integrity_evaluator = integrity_evaluator or DeviceIntegrityEvaluator()
        self.behaviour_analyzer = behaviour_analyzer or BehaviourAnalyzer(seed=seed)
        self.reputation_manager = reputation_manager or ReputationManager(
            initial_reputation, reputation_alpha
        )
        self.update_quality_evaluator = update_quality_evaluator or UpdateQualityEvaluator()
        self.records: List[TrustRecord] = []
        self._logger = logging.getLogger(__name__)

    def register_clients(self, client_ids: Iterable[int]) -> None:
        """Register every participating client ID as identity-known up front."""
        self.identity_verifier.register_known_clients(client_ids)

    def evaluate_round(
        self,
        round_num: int,
        results: List[LocalTrainResult],
        clients: Dict[int, Client],
        global_state_before: Dict[str, torch.Tensor],
    ) -> List[TrustRecord]:
        """Evaluate all five trust dimensions for every client that trained this round.

        Writes the computed scores onto each corresponding `Client` (and
        appends to its `trust_history`), appends to `self.records`, and
        returns this round's records.

        Args:
            round_num: the current communication round (1-indexed).
            results: this round's `LocalTrainResult`s, as returned by each
                participating `Client.train()`.
            clients: the full client registry, keyed by client ID.
            global_state_before: the global model weights clients trained
                from this round (captured before aggregation).
        """
        client_states = {r.client_id: r.state_dict for r in results}
        sample_counts = {r.client_id: r.num_samples for r in results}
        average_losses = {r.client_id: r.average_loss for r in results}

        update_quality_scores = self.update_quality_evaluator.evaluate_round(
            client_states, sample_counts, global_state_before, average_losses
        )

        round_records: List[TrustRecord] = []
        for result in results:
            client_id = result.client_id
            identity = self.identity_verifier.verify(client_id)
            integrity = self.integrity_evaluator.evaluate(client_id)
            behaviour = self.behaviour_analyzer.evaluate(client_id, round_num)
            update_quality = update_quality_scores[client_id]

            # Reputation is updated from a per-round trust *signal* — a
            # simple, equally-weighted average of this round's four fresh
            # dimensions — deliberately distinct from the final composite
            # trust score below, since that score uses the reputation we're
            # about to compute as one of its own inputs. Using it here too
            # would make the update self-referential.
            current_round_signal = (identity + integrity + behaviour + update_quality) / 4.0
            reputation = self.reputation_manager.update(client_id, round_num, current_round_signal)

            trust_score = self._compose_trust(identity, integrity, behaviour, reputation, update_quality)

            client = clients[client_id]
            client.identity_score = identity
            client.integrity_score = integrity
            client.behaviour_score = behaviour
            client.reputation_score = reputation
            client.update_quality_score = update_quality
            client.trust_score = trust_score
            client.trust_history.append(
                {
                    "round": round_num,
                    "identity": identity,
                    "integrity": integrity,
                    "behaviour": behaviour,
                    "reputation": reputation,
                    "update_quality": update_quality,
                    "trust_score": trust_score,
                }
            )

            record = TrustRecord(
                round_num=round_num,
                client_id=client_id,
                identity=identity,
                integrity=integrity,
                behaviour=behaviour,
                reputation=reputation,
                update_quality=update_quality,
                trust_score=trust_score,
            )
            round_records.append(record)
            self._logger.debug(
                "Client %d trust: identity=%.2f integrity=%.2f behaviour=%.2f "
                "reputation=%.2f update_quality=%.2f -> trust=%.2f",
                client_id, identity, integrity, behaviour, reputation, update_quality, trust_score,
            )

        self.records.extend(round_records)
        return round_records

    def _compose_trust(
        self,
        identity: float,
        integrity: float,
        behaviour: float,
        reputation: float,
        update_quality: float,
    ) -> float:
        """Combine the five dimensions into one normalised, clipped trust score.

            Trust = w1*Identity + w2*Integrity + w3*Behaviour
                    + w4*Reputation + w5*UpdateQuality

        Dividing by the total configured weight normalises the result even
        if the configured weights don't sum to exactly 1.0; clipping guards
        against any floating-point overshoot at the [0, 1] boundary.
        """
        weights = self.weights.as_dict()
        total_weight = sum(weights.values())
        if total_weight <= 0:
            raise ValueError("Trust weights must sum to a positive number.")

        raw_score = (
            weights["identity"] * identity
            + weights["integrity"] * integrity
            + weights["behaviour"] * behaviour
            + weights["reputation"] * reputation
            + weights["update_quality"] * update_quality
        )
        normalised = raw_score / total_weight
        return float(min(max(normalised, 0.0), 1.0))

    def print_round_summary(self, round_num: int, records: List[TrustRecord]) -> None:
        """Print the concise end-of-round trust + FedAvg summary (see `trust_metrics`)."""
        print_round_summary(round_num, records)

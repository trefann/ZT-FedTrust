"""Central server orchestrating federated training rounds."""

from __future__ import annotations

import logging
import random
import time
from typing import Dict, List, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from clients.client import Client, LocalTrainResult
from server.selection_metrics import (
    AggregationWeightRecord,
    SelectionRecord,
    print_selection_summary,
)
from server.selector import ClientSelector
from server.trust_fedavg import ClientUpdate, get_aggregation_strategy
from trust.trust_engine import TrustEngine
from utils.metrics import evaluate_model
from utils.validation import validate_state_dict


class Server:
    """Coordinates client selection, local training, aggregation, and evaluation.

    The server holds the authoritative global model and never sees client
    data directly — only the state dicts and sample counts clients return.

    Two optional collaborators layer additively on top of Phase 1's plain
    FedAvg, each defaulting to `None`/`"fedavg"` so existing callers (and
    all Phase 1/2 tests) get byte-for-byte identical behavior:

      - `trust_engine` (Phase 2): if set, evaluates every participating
        client's trust dimensions each round.
      - `client_selector` (Phase 3): if set, replaces random
        `clients_per_round` sampling with trust-threshold + participation-
        fraction selection; combined with `aggregation_strategy="trust_fedavg"`,
        aggregation is weighted by trust as well as sample count.

    `aggregation_strategy` always goes through the same strategy-pattern
    dispatch (`server.trust_fedavg.get_aggregation_strategy`); `"fedavg"`
    (the default) delegates to `server.fedavg.federated_average` unchanged,
    so it is numerically identical to Phase 1/2's original inline call.
    """

    def __init__(
        self,
        global_model: nn.Module,
        clients: Dict[int, Client],
        test_dataset: Dataset,
        device: torch.device,
        clients_per_round: int,
        eval_batch_size: int = 256,
        seed: int = 42,
        trust_engine: Optional[TrustEngine] = None,
        client_selector: Optional[ClientSelector] = None,
        aggregation_strategy: str = "fedavg",
    ) -> None:
        self.global_model = global_model.to(device)
        self.clients = clients
        self.test_loader = DataLoader(test_dataset, batch_size=eval_batch_size, shuffle=False)
        self.device = device
        self.clients_per_round = min(clients_per_round, len(clients))
        self._rng = random.Random(seed)
        self._logger = logging.getLogger(__name__)

        self.trust_engine = trust_engine
        self.client_selector = client_selector
        self.aggregation_strategy_name = aggregation_strategy
        self._aggregation_fn = get_aggregation_strategy(aggregation_strategy)

        # Phase 3 per-round logs, mirroring how `trust_engine.records`
        # accumulates across a run. `selection_records` (and the console
        # summary) are only populated when `client_selector` is set, since
        # they describe threshold-based selection specifically.
        # `aggregation_weight_records`/`last_selected_ids` are always kept
        # up to date (Phase 4 needs per-client weights and this round's
        # participants regardless of which selection mode is active).
        self.selection_records: List[SelectionRecord] = []
        self.aggregation_weight_records: List[AggregationWeightRecord] = []
        self.last_selected_ids: List[int] = []

    def select_clients(self) -> List[int]:
        """Sample a subset of client IDs to participate in the current round.

        Legacy Phase 1/2 random selection, used only when `client_selector`
        is `None`.
        """
        return self._rng.sample(list(self.clients.keys()), self.clients_per_round)

    def run_round(self, round_num: int) -> Dict[str, float]:
        """Run a single federated round.

        Phase 3 workflow when `client_selector` is set: determine the trust
        threshold and select trusted clients, broadcast, train locally,
        collect updates, aggregate (trust-weighted or plain, per
        `aggregation_strategy`), evaluate the global model, then update
        trust/reputation for this round's participants.

        Falls back to Phase 1/2's exact behavior (random `clients_per_round`
        sampling) when `client_selector` is `None`.
        """
        start_time = time.time()

        if self.client_selector is not None:
            selected_ids = self.client_selector.select(self.clients)
            if not selected_ids:
                raise RuntimeError(
                    f"Round {round_num}: no client meets the trust threshold; cannot proceed."
                )
        else:
            selected_ids = self.select_clients()
        self.last_selected_ids = selected_ids

        global_state = self.global_model.state_dict()

        results: List[LocalTrainResult] = []
        for client_id in selected_ids:
            client = self.clients[client_id]
            client.set_weights(global_state)
            results.append(client.train())

        # Snapshot trust *before* this round's evaluation runs: aggregation
        # must weight by trust as it stood when these clients were selected
        # to train, not by a score derived from the very update it's about
        # to help weight (that would be circular).
        trust_at_selection_time = {
            client_id: self.clients[client_id].trust_score for client_id in selected_ids
        }
        updates = [
            ClientUpdate(
                client_id=r.client_id,
                state_dict=r.state_dict,
                num_samples=r.num_samples,
                trust=trust_at_selection_time[r.client_id],
            )
            for r in results
        ]
        aggregated_state, aggregation_weights = self._aggregation_fn(updates)
        validate_state_dict(aggregated_state, context=f"Round {round_num} aggregation")
        self.global_model.load_state_dict(aggregated_state)

        # Recorded unconditionally (not just when `client_selector` is set):
        # Phase 4's security metrics need per-client aggregation weights
        # even for plain-FedAvg-under-attack experiments.
        num_samples_by_client = {u.client_id: u.num_samples for u in updates}
        for client_id, weight in aggregation_weights.items():
            self.aggregation_weight_records.append(
                AggregationWeightRecord(
                    round_num=round_num,
                    client_id=client_id,
                    trust=trust_at_selection_time[client_id],
                    dataset_size=num_samples_by_client[client_id],
                    aggregation_weight=weight,
                )
            )

        eval_loss, eval_accuracy = evaluate_model(self.global_model, self.test_loader, self.device)
        avg_train_loss = sum(r.average_loss for r in results) / len(results)
        round_time = time.time() - start_time

        self._logger.info(
            "Round %d | clients=%s | train_loss=%.4f | test_loss=%.4f | test_acc=%.4f | time=%.2fs",
            round_num,
            selected_ids,
            avg_train_loss,
            eval_loss,
            eval_accuracy,
            round_time,
        )

        round_trust_records = None
        if self.trust_engine is not None:
            # Only reads `results` and the pre-training `global_state`
            # snapshot — neither aggregation nor evaluation touch those —
            # so trust scores come out identical regardless of exactly
            # where in the round this runs (see
            # tests/test_trust.py::test_trust_evaluation_does_not_change_fedavg_result).
            round_trust_records = self.trust_engine.evaluate_round(
                round_num, results, self.clients, global_state
            )

        if self.client_selector is not None:
            stats = self.client_selector.policy.calculate_selection_statistics()
            selection_record = SelectionRecord(
                round_num=round_num,
                threshold=stats.threshold_used,
                total_clients=stats.total_clients,
                eligible_clients=stats.eligible_clients,
                selected_clients=len(selected_ids),
                rejected_clients=stats.total_clients - stats.eligible_clients,
                average_trust=stats.average_trust,
                min_trust=stats.min_trust,
                max_trust=stats.max_trust,
            )
            self.selection_records.append(selection_record)
            print_selection_summary(selection_record, self.aggregation_strategy_name, eval_accuracy)
        elif self.trust_engine is not None:
            # Phase 2 standalone (no Phase 3 selection active): keep its
            # own simpler summary exactly as before.
            self.trust_engine.print_round_summary(round_num, round_trust_records)

        return {
            "round": round_num,
            "train_loss": avg_train_loss,
            "test_loss": eval_loss,
            "test_accuracy": eval_accuracy,
            "round_time_sec": round_time,
        }

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
from server.fedavg import federated_average
from trust.trust_engine import TrustEngine
from utils.metrics import evaluate_model


class Server:
    """Coordinates client selection, local training, aggregation, and evaluation.

    The server holds the authoritative global model and never sees client
    data directly — only the state dicts and sample counts clients return.

    `trust_engine` is optional and defaults to `None`, in which case
    `run_round` behaves exactly as in Phase 1: no trust evaluation, no
    change to FedAvg. Passed a `TrustEngine`, `run_round` evaluates every
    participating client's trust dimensions after local training and before
    aggregation, purely as a side effect (writing onto each `Client` and
    `trust_engine.records`) — the aggregation and evaluation logic below is
    identical either way.
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
    ) -> None:
        self.global_model = global_model.to(device)
        self.clients = clients
        self.test_loader = DataLoader(test_dataset, batch_size=eval_batch_size, shuffle=False)
        self.device = device
        self.clients_per_round = min(clients_per_round, len(clients))
        self._rng = random.Random(seed)
        self._logger = logging.getLogger(__name__)
        self.trust_engine = trust_engine

    def select_clients(self) -> List[int]:
        """Sample a subset of client IDs to participate in the current round."""
        return self._rng.sample(list(self.clients.keys()), self.clients_per_round)

    def run_round(self, round_num: int) -> Dict[str, float]:
        """Run a single federated round: broadcast, local training, aggregation, evaluation."""
        start_time = time.time()
        selected_ids = self.select_clients()
        global_state = self.global_model.state_dict()

        results: List[LocalTrainResult] = []
        for client_id in selected_ids:
            client = self.clients[client_id]
            client.set_weights(global_state)
            results.append(client.train())

        round_trust_records = None
        if self.trust_engine is not None:
            # Evaluated on the pre-aggregation global state and this round's
            # fresh local results, before FedAvg — matches the specified
            # workflow (trust evaluation happens between local training and
            # aggregation) without altering what FedAvg itself computes.
            round_trust_records = self.trust_engine.evaluate_round(
                round_num, results, self.clients, global_state
            )

        aggregated_state = federated_average([(r.state_dict, r.num_samples) for r in results])
        self.global_model.load_state_dict(aggregated_state)

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

        if self.trust_engine is not None:
            self.trust_engine.print_round_summary(round_num, round_trust_records)

        return {
            "round": round_num,
            "train_loss": avg_train_loss,
            "test_loss": eval_loss,
            "test_accuracy": eval_accuracy,
            "round_time_sec": round_time,
        }

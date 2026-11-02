"""Simulated federated learning client with local training logic."""

from __future__ import annotations

import copy
import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset


@dataclass
class LocalTrainResult:
    """Outcome of one client's local training round."""

    client_id: int
    state_dict: Dict[str, torch.Tensor]
    num_samples: int
    average_loss: float


class Client:
    """A single simulated federated learning participant.

    Holds a private data shard and its own persistent local model, which the
    server syncs via `set_weights` before each round and reads back via
    `get_weights`/`train`.

    The `*_score`/`trust_score`/`trust_history` attributes are populated by
    `trust.trust_engine.TrustEngine` (Phase 2's Zero Trust evaluation layer)
    once per round. They default to a neutral 1.0 ("not yet evaluated") and
    are inert otherwise — nothing in FedAvg reads them in this phase.
    """

    def __init__(
        self,
        client_id: int,
        model_template: nn.Module,
        dataset: Dataset,
        device: torch.device,
        batch_size: int = 32,
        learning_rate: float = 0.01,
        momentum: float = 0.9,
        local_epochs: int = 1,
        class_weights: Optional[torch.Tensor] = None,
    ) -> None:
        self.client_id = client_id
        self.dataset = dataset
        self.device = device
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.local_epochs = local_epochs
        self.class_weights = class_weights
        self.model = copy.deepcopy(model_template).to(device)
        self._logger = logging.getLogger(__name__)

        # Zero Trust evaluation state (Phase 2), all in [0.0, 1.0].
        self.identity_score: float = 1.0
        self.integrity_score: float = 1.0
        self.behaviour_score: float = 1.0
        self.reputation_score: float = 1.0
        self.update_quality_score: float = 1.0
        self.trust_score: float = 1.0
        self.trust_history: List[Dict[str, Any]] = []

    def get_weights(self) -> Dict[str, torch.Tensor]:
        """Return a detached CPU copy of the client's current local model weights."""
        return {key: value.detach().cpu().clone() for key, value in self.model.state_dict().items()}

    def set_weights(self, state_dict: Dict[str, torch.Tensor]) -> None:
        """Load `state_dict` (e.g. the broadcast global weights) into the local model."""
        self.model.load_state_dict(state_dict)

    def train(self) -> LocalTrainResult:
        """Train the local model in place for `local_epochs`, starting from its current weights."""
        self.model.train()

        # Drop a leftover final batch of exactly 1 sample (but only when a
        # full batch is still available without it) so BatchNorm-based
        # models (e.g. `models/tabular_mlp.py::TabularMLP`) never receive a
        # training batch of size 1 — `nn.BatchNorm1d.forward` raises
        # "Expected more than 1 value per channel when training" for that
        # case, since per-channel variance is undefined for N=1. Batches of
        # size >= 2 are numerically fine for BatchNorm, so this only trims
        # the specific size-1 remainder rather than every partial batch.
        # Datasets that don't even fill one batch (`len(self.dataset) <=
        # self.batch_size`) still get their single, necessarily under-full,
        # batch — same as before.
        remainder = len(self.dataset) % self.batch_size
        drop_last = remainder == 1 and len(self.dataset) > self.batch_size
        loader = DataLoader(self.dataset, batch_size=self.batch_size, shuffle=True, drop_last=drop_last)
        optimizer = torch.optim.SGD(
            self.model.parameters(), lr=self.learning_rate, momentum=self.momentum
        )
        criterion = nn.CrossEntropyLoss(
            weight=self.class_weights.to(self.device) if self.class_weights is not None else None
        )

        total_loss = 0.0
        total_batches = 0
        for _ in range(self.local_epochs):
            for inputs, targets in loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                total_batches += 1

        average_loss = total_loss / max(total_batches, 1)
        self._logger.debug(
            "Client %d finished local training: %d samples, avg loss %.4f",
            self.client_id,
            len(self.dataset),
            average_loss,
        )
        return LocalTrainResult(
            client_id=self.client_id,
            state_dict=self.get_weights(),
            num_samples=len(self.dataset),
            average_loss=average_loss,
        )

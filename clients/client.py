"""Simulated federated learning client with local training logic."""

from __future__ import annotations

import copy
import logging
from dataclasses import dataclass
from typing import Dict

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
    ) -> None:
        self.client_id = client_id
        self.dataset = dataset
        self.device = device
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.local_epochs = local_epochs
        self.model = copy.deepcopy(model_template).to(device)
        self._logger = logging.getLogger(__name__)

    def get_weights(self) -> Dict[str, torch.Tensor]:
        """Return a detached CPU copy of the client's current local model weights."""
        return {key: value.detach().cpu().clone() for key, value in self.model.state_dict().items()}

    def set_weights(self, state_dict: Dict[str, torch.Tensor]) -> None:
        """Load `state_dict` (e.g. the broadcast global weights) into the local model."""
        self.model.load_state_dict(state_dict)

    def train(self) -> LocalTrainResult:
        """Train the local model in place for `local_epochs`, starting from its current weights."""
        self.model.train()

        loader = DataLoader(self.dataset, batch_size=self.batch_size, shuffle=True)
        optimizer = torch.optim.SGD(
            self.model.parameters(), lr=self.learning_rate, momentum=self.momentum
        )
        criterion = nn.CrossEntropyLoss()

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

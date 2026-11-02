"""Feedforward MLP for tabular IoT attack-detection features."""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class TabularMLP(nn.Module):
    """A small MLP for tabular classification, with configurable hidden-layer sizes.

    Uses the same `in_channels`/`num_classes` constructor-keyword names as
    `models/cnn.py`'s `SimpleCNN` (here `in_channels` means "input feature
    count") so both models share one `build_model(name, in_channels,
    num_classes)` calling convention — `simulation/run.py` and
    `evaluation/experiment_runner.py` need no changes to support this model.

    `hidden_sizes` defaults to `(128, 64)`, the original fixed two-layer
    architecture, so existing callers that don't pass it are unaffected.
    Each entry in `self.layers` is an `nn.ModuleList([Linear, BatchNorm1d])`
    pair; `forward` applies ReLU + dropout after each in sequence.
    """

    def __init__(
        self,
        in_channels: int = 32,
        num_classes: int = 10,
        hidden_sizes: Tuple[int, ...] = (128, 64),
    ) -> None:
        super().__init__()
        if not hidden_sizes:
            raise ValueError("hidden_sizes must contain at least one layer size")

        self.layers = nn.ModuleList()
        previous_size = in_channels
        for size in hidden_sizes:
            self.layers.append(nn.ModuleList([nn.Linear(previous_size, size), nn.BatchNorm1d(size)]))
            previous_size = size

        self.output = nn.Linear(previous_size, num_classes)
        self.dropout = nn.Dropout(0.25)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for linear, batch_norm in self.layers:
            x = F.relu(batch_norm(linear(x)))
            x = self.dropout(x)
        return self.output(x)

    @property
    def fc1(self) -> nn.Linear:
        """Backward compatibility: access first hidden layer."""
        return self.layers[0][0]

    @property
    def bn1(self) -> nn.BatchNorm1d:
        """Backward compatibility: access first batch norm."""
        return self.layers[0][1]

    @property
    def fc2(self) -> nn.Linear:
        """Backward compatibility: access second hidden layer (if present)."""
        if len(self.layers) < 2:
            raise AttributeError("fc2 does not exist (only 1 hidden layer)")
        return self.layers[1][0]

    @property
    def bn2(self) -> nn.BatchNorm1d:
        """Backward compatibility: access second batch norm (if present)."""
        if len(self.layers) < 2:
            raise AttributeError("bn2 does not exist (only 1 hidden layer)")
        return self.layers[1][1]

    @property
    def fc3(self) -> nn.Linear:
        """Backward compatibility: access output layer."""
        return self.output

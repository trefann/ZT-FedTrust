"""Feedforward MLP for tabular IoT attack-detection features."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class TabularMLP(nn.Module):
    """A small MLP for tabular classification.

    Uses the same `in_channels`/`num_classes` constructor-keyword names as
    `models/cnn.py`'s `SimpleCNN` (here `in_channels` means "input feature
    count") so both models share one `build_model(name, in_channels,
    num_classes)` calling convention — `simulation/run.py` and
    `evaluation/experiment_runner.py` need no changes to support this model.
    """

    def __init__(self, in_channels: int = 32, num_classes: int = 10) -> None:
        super().__init__()
        self.fc1 = nn.Linear(in_channels, 128)
        self.bn1 = nn.BatchNorm1d(128)
        self.fc2 = nn.Linear(128, 64)
        self.bn2 = nn.BatchNorm1d(64)
        self.fc3 = nn.Linear(64, num_classes)
        self.dropout = nn.Dropout(0.25)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.bn1(self.fc1(x)))
        x = self.dropout(x)
        x = F.relu(self.bn2(self.fc2(x)))
        x = self.dropout(x)
        return self.fc3(x)

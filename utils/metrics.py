"""Evaluation metrics and reproducibility helpers."""

from __future__ import annotations

import random
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader


def set_seed(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch RNGs for reproducible experiments."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def evaluate_model(
    model: nn.Module, data_loader: DataLoader, device: torch.device
) -> Tuple[float, float]:
    """Evaluate `model` on `data_loader`, returning (average_loss, accuracy)."""
    model.eval()
    criterion = nn.CrossEntropyLoss(reduction="sum")

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for inputs, targets in data_loader:
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = model(inputs)
        total_loss += criterion(outputs, targets).item()
        predictions = outputs.argmax(dim=1)
        total_correct += (predictions == targets).sum().item()
        total_samples += targets.size(0)

    average_loss = total_loss / max(total_samples, 1)
    accuracy = total_correct / max(total_samples, 1)
    return average_loss, accuracy

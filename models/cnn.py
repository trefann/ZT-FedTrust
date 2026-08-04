"""Convolutional neural network architectures for federated learning experiments."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.tabular_mlp import TabularMLP


class SimpleCNN(nn.Module):
    """A small CNN suitable for MNIST/CIFAR-style image classification.

    Uses adaptive average pooling before the classifier head so the same
    architecture works for both 28x28 grayscale and 32x32 RGB inputs.
    """

    def __init__(self, in_channels: int = 1, num_classes: int = 10) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.adaptive_pool = nn.AdaptiveAvgPool2d((7, 7))
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, num_classes)
        self.dropout = nn.Dropout(0.25)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.adaptive_pool(x)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        return self.fc2(x)


_MODEL_REGISTRY = {"cnn": SimpleCNN, "mlp": TabularMLP}


def build_model(name: str, in_channels: int, num_classes: int) -> nn.Module:
    """Factory for constructing a registered model by name.

    Adding a new architecture only requires implementing an `nn.Module` and
    registering it in `_MODEL_REGISTRY` — no other code needs to change.
    """
    if name not in _MODEL_REGISTRY:
        raise ValueError(f"Unknown model name: {name!r}. Available: {list(_MODEL_REGISTRY)}")
    return _MODEL_REGISTRY[name](in_channels=in_channels, num_classes=num_classes)


def available_models() -> list[str]:
    """List model names registered in `_MODEL_REGISTRY`."""
    return list(_MODEL_REGISTRY)

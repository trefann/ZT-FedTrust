"""Synthetic in-memory dataset used across tests to avoid network dependencies."""

from __future__ import annotations

import torch
from torch.utils.data import TensorDataset


def make_synthetic_dataset(
    num_samples: int = 400,
    num_classes: int = 10,
    in_channels: int = 1,
    image_size: int = 28,
    seed: int = 0,
) -> TensorDataset:
    """Build a small, learnable image classification dataset.

    Each label determines a per-channel brightness offset baked into the
    image, plus noise, so a CNN can actually learn to separate classes —
    unlike pure random noise, which no amount of training could classify.
    """
    generator = torch.Generator().manual_seed(seed)
    labels = torch.randint(0, num_classes, (num_samples,), generator=generator)
    noise = torch.randn(num_samples, in_channels, image_size, image_size, generator=generator) * 0.3
    offsets = (labels.float() / num_classes).view(-1, 1, 1, 1)
    images = offsets.expand(-1, in_channels, image_size, image_size) + noise
    return TensorDataset(images, labels)

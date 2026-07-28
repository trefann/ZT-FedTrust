"""Dataset loading utilities for the federated learning simulator."""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import torchvision
from torch.utils.data import Dataset
from torchvision import transforms

_MNIST_TRANSFORM = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ]
)

_CIFAR10_TRANSFORM = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    ]
)

_SHAPES = {"mnist": (1, 10), "cifar10": (3, 10)}

SUPPORTED_DATASETS = tuple(_SHAPES.keys())


def load_dataset(name: str, data_dir: str, train: bool) -> Dataset:
    """Load a supported vision dataset, downloading it to `data_dir` if needed.

    Args:
        name: dataset identifier, one of "mnist" or "cifar10".
        data_dir: directory used to cache/download the raw dataset files.
        train: whether to load the training split (True) or test split (False).
    """
    name = name.lower()
    Path(data_dir).mkdir(parents=True, exist_ok=True)

    if name == "mnist":
        return torchvision.datasets.MNIST(
            root=data_dir, train=train, download=True, transform=_MNIST_TRANSFORM
        )
    if name == "cifar10":
        return torchvision.datasets.CIFAR10(
            root=data_dir, train=train, download=True, transform=_CIFAR10_TRANSFORM
        )
    raise ValueError(f"Unsupported dataset: {name!r}. Supported: {list(_SHAPES)}")


def dataset_shape(name: str) -> Tuple[int, int]:
    """Return (in_channels, num_classes) for a supported dataset name."""
    name = name.lower()
    if name not in _SHAPES:
        raise ValueError(f"Unsupported dataset: {name!r}. Supported: {list(_SHAPES)}")
    return _SHAPES[name]

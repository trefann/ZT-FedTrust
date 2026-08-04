"""Torch Dataset wrapper for preprocessed tabular IoT feature matrices."""

from __future__ import annotations

from typing import List, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset


class TabularDataset(Dataset):
    """Wraps a preprocessed `(features, labels)` array pair.

    Exposes `.targets`, matching the attribute `data/partition.py`'s
    `_extract_labels` already looks for on torchvision-style datasets — so
    Dirichlet/IID partitioning works on IoT data with no changes to
    `data/partition.py`.
    """

    def __init__(self, features: np.ndarray, labels: np.ndarray) -> None:
        if len(features) != len(labels):
            raise ValueError(
                f"features/labels length mismatch: {len(features)} vs {len(labels)}"
            )
        self.features = torch.from_numpy(np.asarray(features, dtype=np.float32))
        self.labels = torch.from_numpy(np.asarray(labels, dtype=np.int64))
        self.targets: List[int] = self.labels.tolist()

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.features[index], self.labels[index]

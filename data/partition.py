"""Strategies for partitioning a dataset across simulated federated clients."""

from __future__ import annotations

from typing import Dict, List

import numpy as np
from torch.utils.data import Dataset, Subset

SUPPORTED_PARTITION_TYPES = ("iid", "dirichlet")


def partition_iid(dataset: Dataset, num_clients: int, seed: int) -> Dict[int, List[int]]:
    """Split dataset indices uniformly at random across clients (IID)."""
    rng = np.random.default_rng(seed)
    indices = rng.permutation(len(dataset))
    shards = np.array_split(indices, num_clients)
    return {client_id: shard.tolist() for client_id, shard in enumerate(shards)}


def partition_dirichlet(
    dataset: Dataset, num_clients: int, alpha: float, seed: int
) -> Dict[int, List[int]]:
    """Split dataset indices across clients using a Dirichlet label distribution.

    Lower `alpha` produces more label skew (more non-IID) between clients;
    as `alpha` grows large the split approaches the IID case.
    """
    rng = np.random.default_rng(seed)
    labels = np.array(_extract_labels(dataset))
    num_classes = int(labels.max()) + 1

    client_indices: Dict[int, List[int]] = {i: [] for i in range(num_clients)}
    for class_id in range(num_classes):
        class_indices = np.where(labels == class_id)[0]
        rng.shuffle(class_indices)

        proportions = rng.dirichlet(alpha=np.repeat(alpha, num_clients))
        split_points = (np.cumsum(proportions) * len(class_indices)).astype(int)[:-1]
        for client_id, shard in enumerate(np.split(class_indices, split_points)):
            client_indices[client_id].extend(shard.tolist())

    for client_id in client_indices:
        rng.shuffle(client_indices[client_id])
    return client_indices


def _extract_labels(dataset: Dataset) -> List[int]:
    """Best-effort extraction of integer labels from a torchvision-style dataset."""
    if hasattr(dataset, "targets"):
        return [int(t) for t in dataset.targets]
    return [int(dataset[i][1]) for i in range(len(dataset))]


def make_client_subsets(
    dataset: Dataset, client_indices: Dict[int, List[int]]
) -> Dict[int, Subset]:
    """Wrap raw index lists into `Subset` datasets, one per client."""
    return {client_id: Subset(dataset, indices) for client_id, indices in client_indices.items()}

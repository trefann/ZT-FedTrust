"""Label-flipping attack: Phase 4 dataset-level attack.

Wraps a client's local training dataset so labels are remapped at *read*
time, according to a configurable mapping. The underlying dataset — and
transitively the shared MNIST/CIFAR-10 object every client's `Subset`
points into, and the global test set — is never mutated.
"""

from __future__ import annotations

from typing import Dict, Optional

from torch.utils.data import Dataset

from attacks.base_attack import BaseAttack

# Default adjacent-pair swap: 0<->1, 2<->3, 4<->5, 6<->7, 8<->9.
DEFAULT_LABEL_MAP: Dict[int, int] = {i: (i + 1 if i % 2 == 0 else i - 1) for i in range(10)}


class _LabelFlippingDataset(Dataset):
    """Read-time label remapper. Never modifies the wrapped dataset."""

    def __init__(self, base_dataset: Dataset, label_map: Dict[int, int]) -> None:
        self.base_dataset = base_dataset
        self.label_map = label_map

    def __len__(self) -> int:
        return len(self.base_dataset)

    def __getitem__(self, index: int):
        image, label = self.base_dataset[index]
        return image, self.label_map.get(int(label), int(label))


class LabelFlippingAttack(BaseAttack):
    """Flips training labels according to a configurable mapping.

    Args:
        label_map: source label -> flipped label. Defaults to swapping each
            even digit with the next odd one (0->1, 1->0, 2->3, 3->2, ...).
    """

    name = "label_flipping"

    def __init__(self, label_map: Optional[Dict[int, int]] = None) -> None:
        self.label_map = dict(label_map) if label_map else dict(DEFAULT_LABEL_MAP)

    def apply_to_dataset(self, dataset: Dataset) -> Dataset:
        return _LabelFlippingDataset(dataset, self.label_map)

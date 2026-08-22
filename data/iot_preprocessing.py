"""Dataset-agnostic tabular preprocessing: feature encoding, label encoding,
subsampling, and train/test splitting shared by every IoT dataset loader in
`data/iot_datasets.py`.

Column type (categorical vs. numeric) is inferred from pandas dtype rather
than hand-classified per dataset — the five IoT source datasets have
dozens of columns each with no consistent naming convention, so dtype
introspection is far less error-prone than a hardcoded column-type list
per dataset.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from pandas.api.types import is_numeric_dtype


@dataclass
class FeatureEncoder:
    """Fitted feature transform: categorical columns become ordinal codes
    (0 reserved for unseen values), numeric columns are standardized.

    Column order in `feature_names` is fixed at fit time (alphabetical) so
    every dataset split (train/test, or every natural client's own
    dataframe) produces feature vectors with identical column semantics.
    """

    feature_names: List[str]
    categorical_columns: List[str]
    numeric_columns: List[str]
    categorical_maps: Dict[str, Dict[str, int]] = field(default_factory=dict)
    numeric_stats: Dict[str, Tuple[float, float]] = field(default_factory=dict)

    @property
    def input_dim(self) -> int:
        return len(self.feature_names)

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        columns = []
        for name in self.feature_names:
            if name in self.categorical_columns:
                mapping = self.categorical_maps[name]
                codes = df[name].astype(str).map(mapping).fillna(0).astype(np.float32)
                columns.append(codes.to_numpy())
            else:
                mean, std = self.numeric_stats[name]
                values = pd.to_numeric(df[name], errors="coerce").fillna(mean).astype(np.float64)
                columns.append(((values - mean) / (std + 1e-8)).to_numpy(dtype=np.float32))
        return np.stack(columns, axis=1).astype(np.float32)


def fit_feature_encoder(train_df: pd.DataFrame) -> FeatureEncoder:
    """Fit a `FeatureEncoder` from training data only (never from test rows,
    to avoid leaking test-set statistics into feature scaling)."""
    feature_names = sorted(train_df.columns)
    numeric_columns = [c for c in feature_names if is_numeric_dtype(train_df[c])]
    categorical_columns = [c for c in feature_names if c not in numeric_columns]

    categorical_maps: Dict[str, Dict[str, int]] = {}
    for col in categorical_columns:
        uniques = sorted(train_df[col].astype(str).unique())
        categorical_maps[col] = {value: index + 1 for index, value in enumerate(uniques)}

    numeric_stats: Dict[str, Tuple[float, float]] = {}
    for col in numeric_columns:
        values = pd.to_numeric(train_df[col], errors="coerce")
        numeric_stats[col] = (float(values.mean()), float(values.std(ddof=0)))

    return FeatureEncoder(
        feature_names=feature_names,
        categorical_columns=categorical_columns,
        numeric_columns=numeric_columns,
        categorical_maps=categorical_maps,
        numeric_stats=numeric_stats,
    )


def fit_label_classes(labels: pd.Series) -> List[str]:
    """Sorted unique label strings — the fixed class-index vocabulary for a dataset.

    Fit on the *full* dataset (train+test pooled), not train alone: knowing
    which class names exist isn't statistical leakage the way feature
    scaling would be — it's the same as knowing MNIST has 10 digit classes
    in advance.
    """
    return sorted(labels.astype(str).unique())


def encode_labels(labels: pd.Series, class_list: List[str]) -> np.ndarray:
    index = {value: i for i, value in enumerate(class_list)}
    str_labels = labels.astype(str)
    unknown = sorted(set(str_labels.unique()) - set(class_list))
    if unknown:
        raise ValueError(f"Labels not in class_list: {unknown}")
    return str_labels.map(index).to_numpy(dtype=np.int64)


def subsample_dataframe(df: pd.DataFrame, max_rows: Optional[int], seed: int) -> pd.DataFrame:
    if max_rows is None or len(df) <= max_rows:
        return df.reset_index(drop=True)
    return df.sample(n=max_rows, random_state=seed).reset_index(drop=True)


def split_train_test(
    df: pd.DataFrame, test_fraction: float, seed: int
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    shuffled = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    split_point = int(len(shuffled) * (1.0 - test_fraction))
    return (
        shuffled.iloc[:split_point].reset_index(drop=True),
        shuffled.iloc[split_point:].reset_index(drop=True),
    )


def compute_class_weights(labels: Sequence[int], num_classes: int) -> torch.Tensor:
    """Inverse-frequency class weights for `nn.CrossEntropyLoss(weight=...)`.

    `weight[c] = total_samples / (num_classes * count[c])` — the standard
    "balanced" class-weight formula, giving rarer classes a larger weight.
    A class with zero observed samples gets weight `0.0` (there's no
    gradient signal for it regardless; this only avoids a division by
    zero).
    """
    counts = np.bincount(np.asarray(labels, dtype=np.int64), minlength=num_classes)
    total = counts.sum()
    weights = np.zeros(num_classes, dtype=np.float32)
    nonzero = counts > 0
    weights[nonzero] = total / (num_classes * counts[nonzero])
    return torch.from_numpy(weights)

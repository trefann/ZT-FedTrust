"""Real IoT attack-detection dataset loaders for ZT-FedTrust.

Two shared builders handle every dataset:

  - `_build_partitioned_experiment`: for datasets with no natural per-device
    split (CICIoT2023, UNSW-NB15, TON_IoT network) — one pooled dataframe,
    Dirichlet/IID-partitioned across `num_clients` simulated clients via
    the existing `data/partition.py`, exactly like MNIST today.
  - `_build_natural_client_experiment`: for datasets whose real devices
    become clients directly (N-BaIoT's 9 devices, TON_IoT's 7 IoT device
    telemetry sources) — no partitioning; each device's own data is one
    client's dataset.

Both fit one shared `FeatureEncoder`/label vocabulary across the relevant
pooled training data (required so every client in a population shares one
model's input dimensionality — federated averaging cannot mix clients with
different feature counts) and return one `IoTExperimentData` bundle.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import pandas as pd
from torch.utils.data import Dataset

from data.iot_preprocessing import (
    FeatureEncoder,
    encode_labels,
    fit_feature_encoder,
    fit_label_classes,
    split_train_test,
    subsample_dataframe,
)
from data.partition import make_client_subsets, partition_dirichlet, partition_iid
from data.tabular_dataset import TabularDataset

_TEST_FRACTION = 0.2


@dataclass
class IoTExperimentData:
    """Everything `simulation/run.py::load_experiment_data` needs for one IoT experiment."""

    train_client_datasets: Dict[int, Dataset]
    test_dataset: Dataset
    attack_pool: Optional[Dataset]
    input_dim: int
    num_classes: int
    class_names: List[str]


def _build_attack_pool(
    train_df: pd.DataFrame,
    feature_columns: List[str],
    label_column: str,
    encoder: FeatureEncoder,
    class_names: List[str],
    benign_label: Optional[str],
) -> Optional[Dataset]:
    """Rows whose label isn't `benign_label` — used by `attacks.real_traffic`.

    Returns None when the dataset has no benign/attack distinction to draw
    from (e.g. this CICIoT2023 slice contains only DDoS-attack rows), so
    `real_traffic` cleanly isn't offered as an option for that dataset
    rather than silently returning a meaningless pool.
    """
    if benign_label is None or benign_label not in class_names:
        return None
    attack_rows = train_df[train_df[label_column] != benign_label]
    if attack_rows.empty:
        return None
    features = encoder.transform(attack_rows[feature_columns])
    labels = encode_labels(attack_rows[label_column], class_names)
    return TabularDataset(features, labels)


def _build_partitioned_experiment(
    df: pd.DataFrame,
    label_column: str,
    benign_label: Optional[str],
    num_clients: int,
    partition_type: str,
    dirichlet_alpha: float,
    max_rows: Optional[int],
    seed: int,
) -> IoTExperimentData:
    df = subsample_dataframe(df, max_rows, seed)
    feature_columns = [c for c in df.columns if c != label_column]
    class_names = fit_label_classes(df[label_column])

    train_df, test_df = split_train_test(df, _TEST_FRACTION, seed)
    encoder = fit_feature_encoder(train_df[feature_columns])

    train_dataset = TabularDataset(
        encoder.transform(train_df[feature_columns]),
        encode_labels(train_df[label_column], class_names),
    )
    test_dataset = TabularDataset(
        encoder.transform(test_df[feature_columns]),
        encode_labels(test_df[label_column], class_names),
    )

    if partition_type == "iid":
        client_indices = partition_iid(train_dataset, num_clients, seed)
    elif partition_type == "dirichlet":
        client_indices = partition_dirichlet(train_dataset, num_clients, dirichlet_alpha, seed)
    else:
        raise ValueError(f"Unsupported partition type: {partition_type!r}")
    train_client_datasets = make_client_subsets(train_dataset, client_indices)

    attack_pool = _build_attack_pool(
        train_df, feature_columns, label_column, encoder, class_names, benign_label
    )

    return IoTExperimentData(
        train_client_datasets=train_client_datasets,
        test_dataset=test_dataset,
        attack_pool=attack_pool,
        input_dim=encoder.input_dim,
        num_classes=len(class_names),
        class_names=class_names,
    )


def _build_natural_client_experiment(
    device_dataframes: Dict[str, pd.DataFrame],
    label_column: str,
    benign_label: Optional[str],
    max_rows_per_device: Optional[int],
    seed: int,
) -> IoTExperimentData:
    device_names = sorted(device_dataframes)
    sampled = {
        name: subsample_dataframe(device_dataframes[name], max_rows_per_device, seed)
        for name in device_names
    }
    feature_columns = [c for c in next(iter(sampled.values())).columns if c != label_column]

    full_pool = pd.concat(sampled.values(), ignore_index=True)
    class_names = fit_label_classes(full_pool[label_column])

    train_parts: Dict[str, pd.DataFrame] = {}
    test_parts: Dict[str, pd.DataFrame] = {}
    for name, device_df in sampled.items():
        train_df, test_df = split_train_test(device_df, _TEST_FRACTION, seed)
        train_parts[name] = train_df
        test_parts[name] = test_df

    pooled_train = pd.concat(train_parts.values(), ignore_index=True)
    encoder = fit_feature_encoder(pooled_train[feature_columns])

    train_client_datasets: Dict[int, Dataset] = {}
    for client_id, name in enumerate(device_names):
        features = encoder.transform(train_parts[name][feature_columns])
        labels = encode_labels(train_parts[name][label_column], class_names)
        train_client_datasets[client_id] = TabularDataset(features, labels)

    pooled_test = pd.concat(test_parts.values(), ignore_index=True)
    test_dataset = TabularDataset(
        encoder.transform(pooled_test[feature_columns]),
        encode_labels(pooled_test[label_column], class_names),
    )

    attack_pool = _build_attack_pool(
        pooled_train, feature_columns, label_column, encoder, class_names, benign_label
    )

    return IoTExperimentData(
        train_client_datasets=train_client_datasets,
        test_dataset=test_dataset,
        attack_pool=attack_pool,
        input_dim=encoder.input_dim,
        num_classes=len(class_names),
        class_names=class_names,
    )

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

import zipfile
from dataclasses import dataclass
from pathlib import Path
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
from data.rar_extract import extract_rar
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


_CICIOT2023_FILENAME = "CICIoT2023 (2).csv"


def _load_ciciot2023_dataframe(data_dir: str) -> pd.DataFrame:
    path = Path(data_dir) / _CICIOT2023_FILENAME
    return pd.read_csv(path, encoding="utf-8-sig")


def load_ciciot2023(
    data_dir: str,
    num_clients: int,
    partition_type: str,
    dirichlet_alpha: float,
    max_rows: Optional[int],
    seed: int,
) -> IoTExperimentData:
    """No benign traffic exists in this CICIoT2023 slice, so `benign_label`
    is None — `_build_attack_pool` correctly returns no attack pool."""
    df = _load_ciciot2023_dataframe(data_dir)
    return _build_partitioned_experiment(
        df,
        label_column="label",
        benign_label=None,
        num_clients=num_clients,
        partition_type=partition_type,
        dirichlet_alpha=dirichlet_alpha,
        max_rows=max_rows,
        seed=seed,
    )


_UNSW_NB15_ZIP_FILENAME = "UNSW-NB15 dataset.zip"
_UNSW_NB15_TRAIN_ENTRY = "CSV Files/Training and Testing Sets/UNSW_NB15_training-set.csv"
_UNSW_NB15_TEST_ENTRY = "CSV Files/Training and Testing Sets/UNSW_NB15_testing-set.csv"


def _load_unsw_nb15_dataframe(data_dir: str) -> pd.DataFrame:
    zip_path = Path(data_dir) / _UNSW_NB15_ZIP_FILENAME
    with zipfile.ZipFile(zip_path) as zf:
        with zf.open(_UNSW_NB15_TRAIN_ENTRY) as f:
            train_df = pd.read_csv(f, encoding="utf-8-sig")
        with zf.open(_UNSW_NB15_TEST_ENTRY) as f:
            test_df = pd.read_csv(f, encoding="utf-8-sig")
    df = pd.concat([train_df, test_df], ignore_index=True)
    return df.drop(columns=["id", "label"])


def load_unsw_nb15(
    data_dir: str,
    num_clients: int,
    partition_type: str,
    dirichlet_alpha: float,
    max_rows: Optional[int],
    seed: int,
) -> IoTExperimentData:
    df = _load_unsw_nb15_dataframe(data_dir)
    return _build_partitioned_experiment(
        df,
        label_column="attack_cat",
        benign_label="Normal",
        num_clients=num_clients,
        partition_type=partition_type,
        dirichlet_alpha=dirichlet_alpha,
        max_rows=max_rows,
        seed=seed,
    )


_TON_IOT_ZIP_FILENAME = "Train_Test_datasets.zip"
_TON_IOT_NETWORK_ENTRY = "Train_Test_datasets/Train_Test_Network_dataset/train_test_network.csv"
_TON_IOT_NETWORK_DROP_COLUMNS = [
    "src_ip", "dst_ip", "dns_query", "ssl_subject", "ssl_issuer",
    "http_uri", "http_user_agent", "label",
]


def _load_ton_iot_network_dataframe(data_dir: str) -> pd.DataFrame:
    zip_path = Path(data_dir) / _TON_IOT_ZIP_FILENAME
    with zipfile.ZipFile(zip_path) as zf:
        with zf.open(_TON_IOT_NETWORK_ENTRY) as f:
            df = pd.read_csv(f, encoding="utf-8-sig")
    return df.drop(columns=[c for c in _TON_IOT_NETWORK_DROP_COLUMNS if c in df.columns])


def load_ton_iot_network(
    data_dir: str,
    num_clients: int,
    partition_type: str,
    dirichlet_alpha: float,
    max_rows: Optional[int],
    seed: int,
) -> IoTExperimentData:
    df = _load_ton_iot_network_dataframe(data_dir)
    return _build_partitioned_experiment(
        df,
        label_column="type",
        benign_label="normal",
        num_clients=num_clients,
        partition_type=partition_type,
        dirichlet_alpha=dirichlet_alpha,
        max_rows=max_rows,
        seed=seed,
    )


_TON_IOT_DEVICE_ENTRIES = {
    "fridge": "Train_Test_datasets/Train_Test_IoT_dataset/Train_Test_IoT_Fridge.csv",
    "garage_door": "Train_Test_datasets/Train_Test_IoT_dataset/Train_Test_IoT_Garage_Door.csv",
    "gps_tracker": "Train_Test_datasets/Train_Test_IoT_dataset/Train_Test_IoT_GPS_Tracker.csv",
    "modbus": "Train_Test_datasets/Train_Test_IoT_dataset/Train_Test_IoT_Modbus.csv",
    "motion_light": "Train_Test_datasets/Train_Test_IoT_dataset/Train_Test_IoT_Motion_Light.csv",
    "thermostat": "Train_Test_datasets/Train_Test_IoT_dataset/Train_Test_IoT_Thermostat.csv",
    "weather": "Train_Test_datasets/Train_Test_IoT_dataset/Train_Test_IoT_Weather.csv",
}
_TON_IOT_DEVICE_DROP_COLUMNS = ["date", "time", "label"]


def _load_ton_iot_device_dataframes(data_dir: str) -> Dict[str, pd.DataFrame]:
    zip_path = Path(data_dir) / _TON_IOT_ZIP_FILENAME
    raw: Dict[str, pd.DataFrame] = {}
    with zipfile.ZipFile(zip_path) as zf:
        for device_name, entry in _TON_IOT_DEVICE_ENTRIES.items():
            with zf.open(entry) as f:
                df = pd.read_csv(f, encoding="utf-8-sig")
            raw[device_name] = df.drop(
                columns=[c for c in _TON_IOT_DEVICE_DROP_COLUMNS if c in df.columns]
            )

    all_feature_columns = sorted(
        {col for df in raw.values() for col in df.columns if col != "type"}
    )
    reindexed: Dict[str, pd.DataFrame] = {}
    for device_name, df in raw.items():
        missing = [c for c in all_feature_columns if c not in df.columns]
        for col in missing:
            df[col] = 0.0
        reindexed[device_name] = df[all_feature_columns + ["type"]]
    return reindexed


def load_ton_iot_devices(
    data_dir: str, max_rows_per_device: Optional[int], seed: int
) -> IoTExperimentData:
    """Each of TON_IoT's 7 real device sources becomes one client — no
    Dirichlet/IID partitioning. Devices don't share sensor columns (Fridge
    reports temperature, Garage_Door reports door state, ...), so every
    device's dataframe is reindexed onto the union of all 7 devices'
    columns before fitting one shared `FeatureEncoder` — a device's
    non-applicable columns are filled with 0.0, which `FeatureEncoder`
    then standardizes like any other numeric value.
    """
    device_dataframes = _load_ton_iot_device_dataframes(data_dir)
    return _build_natural_client_experiment(
        device_dataframes,
        label_column="type",
        benign_label="normal",
        max_rows_per_device=max_rows_per_device,
        seed=seed,
    )


_NBAIOT_ZIP_FILENAME = "detection+of+iot+botnet+attacks+n+baiot.zip"
_NBAIOT_CACHE_DIR = "data_cache/iot/nbaiot"
_NBAIOT_DEVICE_DIRS = (
    "Danmini_Doorbell", "Ecobee_Thermostat", "Ennio_Doorbell",
    "Philips_B120N10_Baby_Monitor", "Provision_PT_737E_Security_Camera",
    "Provision_PT_838_Security_Camera", "Samsung_SNH_1011_N_Webcam",
    "SimpleHome_XCS7_1002_WHT_Security_Camera", "SimpleHome_XCS7_1003_WHT_Security_Camera",
)


def _read_family_attack_csvs(extracted_dir: Path, family: str) -> pd.DataFrame:
    """Read every CSV an extracted attack archive contains, labeling each
    row by its source filename (e.g. `combo.csv` -> "gafgyt_combo")."""
    frames = []
    for csv_path in sorted(extracted_dir.glob("*.csv")):
        part = pd.read_csv(csv_path, encoding="utf-8-sig")
        part["label"] = f"{family}_{csv_path.stem}"
        frames.append(part)
    return pd.concat(frames, ignore_index=True)


def _load_nbaiot_device_dataframe(zip_path: Path, device_name: str, cache_dir: Path) -> pd.DataFrame:
    frames = []
    with zipfile.ZipFile(zip_path) as zf:
        entries = set(zf.namelist())
        benign_entry = f"{device_name}/benign_traffic.csv"
        with zf.open(benign_entry) as f:
            benign_df = pd.read_csv(f, encoding="utf-8-sig")
        benign_df["label"] = "benign"
        frames.append(benign_df)

        for family in ("gafgyt", "mirai"):
            rar_entry = f"{device_name}/{family}_attacks.rar"
            if rar_entry not in entries:
                continue
            extracted_dir = cache_dir / device_name / family
            if not any(extracted_dir.glob("*.csv")):
                archive_path = cache_dir / device_name / f"{family}_attacks.rar"
                archive_path.parent.mkdir(parents=True, exist_ok=True)
                with zf.open(rar_entry) as src, open(archive_path, "wb") as dst:
                    dst.write(src.read())
                extract_rar(archive_path, extracted_dir)
            frames.append(_read_family_attack_csvs(extracted_dir, family))

    return pd.concat(frames, ignore_index=True)


def _load_nbaiot_device_dataframes(data_dir: str) -> Dict[str, pd.DataFrame]:
    zip_path = Path(data_dir) / _NBAIOT_ZIP_FILENAME
    cache_dir = Path(_NBAIOT_CACHE_DIR)
    with zipfile.ZipFile(zip_path) as zf:
        present_devices = {
            name for name in _NBAIOT_DEVICE_DIRS if f"{name}/benign_traffic.csv" in zf.namelist()
        }
    return {
        name: _load_nbaiot_device_dataframe(zip_path, name, cache_dir) for name in present_devices
    }


def load_nbaiot(data_dir: str, max_rows_per_device: Optional[int], seed: int) -> IoTExperimentData:
    """Each of N-BaIoT's 9 real devices becomes one client. Every device
    ships `benign_traffic.csv` and a `gafgyt_attacks.rar`, but only 7 of the
    9 also ship a `mirai_attacks.rar` (Ennio_Doorbell and
    Samsung_SNH_1011_N_Webcam have no Mirai traffic in the real dataset) —
    that per-family absence is a documented characteristic of N-BaIoT, not a
    corrupted/incomplete download, so it's skipped rather than raised. A
    missing `benign_traffic.csv`, by contrast, would indicate a genuinely
    broken archive and is read unconditionally so it fails loudly instead.
    """
    device_dataframes = _load_nbaiot_device_dataframes(data_dir)
    return _build_natural_client_experiment(
        device_dataframes,
        label_column="label",
        benign_label="benign",
        max_rows_per_device=max_rows_per_device,
        seed=seed,
    )

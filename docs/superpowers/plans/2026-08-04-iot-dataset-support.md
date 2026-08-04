# IoT Dataset Support Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace ZT-FedTrust's MNIST/CIFAR-10 proof-of-concept substitute with real IoT attack-detection datasets (CICIoT2023, N-BaIoT, TON_IoT, UNSW-NB15), fully additive to the existing trust engine, aggregation, attacks, and evaluation framework.

**Architecture:** A new `data/iot_preprocessing.py` + `data/tabular_dataset.py` provide dataset-agnostic fit/transform helpers and a `torch.utils.data.Dataset` wrapper. Five per-dataset loader functions in `data/iot_datasets.py` each produce a raw pandas DataFrame (or one DataFrame per real device), then hand off to one of two shared builders (`_build_partitioned_experiment` for Dirichlet-style datasets, `_build_natural_client_experiment` for datasets whose real devices become clients directly) that fit encoders, split train/test, and return one `IoTExperimentData` bundle (per-client training datasets, a pooled global test set, and — where the dataset has a benign/attack split — a pool of real attack-labeled rows). `simulation/run.py` gains one new function, `load_experiment_data`, that both `simulation/run.py` and `evaluation/experiment_runner.py` call instead of their current dataset-loading lines, branching internally between the untouched vision path and the new IoT path. A new `models/tabular_mlp.py` slots into the existing model registry. A new `attacks/real_traffic.py` attack lets malicious clients train on real attack-labeled traffic instead of a synthetic perturbation, wired into the existing `build_attack`/`assign_attacks` machinery via one new optional parameter.

**Tech Stack:** Python, PyTorch, pandas (already a dependency), pytest. No new dependencies — RAR extraction shells out to an external `7z`/`unrar` binary rather than adding a Python RAR library.

## Global Constraints

- No new pip dependencies (`requirements.txt` stays as-is: torch, torchvision, numpy, PyYAML, matplotlib, pandas, scipy, tabulate).
- Existing MNIST/CIFAR-10 code (`data/dataset.py`, `models/cnn.py`'s `SimpleCNN`, existing tests, existing demo configs) is never modified except to add the new model to `_MODEL_REGISTRY` — no removals, no behavior changes to vision runs.
- Every new module gets tests using tiny synthetic/fixture data — no test reads the real multi-GB dataset files in `ZT-Fedtrust Datasets/` (that folder is outside git and may not exist in CI).
- All new code follows the existing style: `from __future__ import annotations`, type hints, dataclasses for config-shaped data, no comments except where a non-obvious WHY needs explaining (matches this codebase's existing docstring/comment density).
- CSV reads use `encoding="utf-8-sig"` (several source files carry a UTF-8 BOM, confirmed on the TON_IoT files).
- Real source filenames (exact, spaces and all): `"CICIoT2023 (2).csv"`, `"UNSW-NB15 dataset.zip"`, `"Train_Test_datasets.zip"`, `"detection+of+iot+botnet+attacks+n+baiot.zip"`.

---

## File Structure

**New files:**
- `data/iot_preprocessing.py` — dtype-driven feature/label encoding, subsampling, train/test split. Dataset-agnostic.
- `data/tabular_dataset.py` — `TabularDataset(Dataset)`, the tabular analogue of a torchvision dataset.
- `data/rar_extract.py` — finds a `7z`/`unrar` binary and shells out to extract a `.rar` archive.
- `models/tabular_mlp.py` — `TabularMLP(nn.Module)`.
- `data/iot_datasets.py` — `IoTExperimentData`, the two shared experiment builders, five per-dataset loaders, and the top-level `IOT_SUPPORTED_DATASETS` / `load_iot_experiment` registry.
- `attacks/real_traffic.py` — `RealAttackTrafficAttack(BaseAttack)`.
- `configs/iot/ciciot2023.yaml`, `configs/iot/unsw_nb15.yaml`, `configs/iot/ton_iot_network.yaml`, `configs/iot/ton_iot_devices.yaml`, `configs/iot/nbaiot.yaml`, `configs/iot/botiot.yaml`.
- `tests/test_iot_preprocessing.py`, `tests/test_tabular_dataset.py`, `tests/test_rar_extract.py`, `tests/test_tabular_mlp.py`, `tests/test_iot_datasets.py`, `tests/test_real_traffic_attack.py`, `tests/test_iot_configs.py`, `tests/test_iot_integration.py`.

**Modified files:**
- `models/cnn.py` — register `TabularMLP` under `"mlp"`.
- `simulation/run.py` — `DataConfig` gains `max_rows`; `validate_config` accepts IoT dataset names; new `load_experiment_data` function; `run_simulation` uses it; `assign_attacks` call gains `attack_dataset`.
- `evaluation/experiment_runner.py` — uses `load_experiment_data` instead of its own `dataset_shape`/`load_dataset`/`build_model` calls; `assign_attacks` call gains `attack_dataset`.
- `attacks/__init__.py` — `SUPPORTED_ATTACK_TYPES` gains `"real_traffic"`.
- `clients/malicious_client.py` — `build_attack`/`assign_attacks` gain an optional `attack_dataset` parameter and a `"real_traffic"` branch.

---

### Task 1: Shared tabular preprocessing helpers

**Files:**
- Create: `data/iot_preprocessing.py`
- Test: `tests/test_iot_preprocessing.py`

**Interfaces:**
- Produces: `FeatureEncoder` (dataclass with `.transform(df) -> np.ndarray[float32]` and `.input_dim: int`), `fit_feature_encoder(train_df: pd.DataFrame) -> FeatureEncoder`, `fit_label_classes(labels: pd.Series) -> List[str]`, `encode_labels(labels: pd.Series, class_list: List[str]) -> np.ndarray[int64]`, `subsample_dataframe(df: pd.DataFrame, max_rows: Optional[int], seed: int) -> pd.DataFrame`, `split_train_test(df: pd.DataFrame, test_fraction: float, seed: int) -> Tuple[pd.DataFrame, pd.DataFrame]`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_iot_preprocessing.py
"""Tests for dataset-agnostic tabular preprocessing helpers."""

import numpy as np
import pandas as pd
import pytest

from data.iot_preprocessing import (
    encode_labels,
    fit_feature_encoder,
    fit_label_classes,
    split_train_test,
    subsample_dataframe,
)


def _sample_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "proto": ["tcp", "udp", "tcp", "icmp", "udp", "tcp"],
            "duration": [1.0, 2.5, 0.1, 9.9, 3.3, 0.5],
            "bytes": [100, 200, 50, 900, 300, 60],
        }
    )


def test_fit_feature_encoder_splits_categorical_and_numeric():
    encoder = fit_feature_encoder(_sample_df())
    assert set(encoder.categorical_columns) == {"proto"}
    assert set(encoder.numeric_columns) == {"duration", "bytes"}
    assert encoder.input_dim == 3


def test_transform_produces_expected_shape_and_dtype():
    df = _sample_df()
    encoder = fit_feature_encoder(df)
    matrix = encoder.transform(df)
    assert matrix.shape == (6, 3)
    assert matrix.dtype == np.float32


def test_transform_maps_unseen_categorical_value_to_zero():
    train_df = _sample_df()
    encoder = fit_feature_encoder(train_df)
    proto_col_index = encoder.feature_names.index("proto")

    unseen_df = pd.DataFrame({"proto": ["quic"], "duration": [1.0], "bytes": [10]})
    matrix = encoder.transform(unseen_df)
    assert matrix[0, proto_col_index] == 0.0


def test_transform_numeric_column_is_standardized():
    df = pd.DataFrame({"value": [1.0, 2.0, 3.0, 4.0, 5.0]})
    encoder = fit_feature_encoder(df)
    matrix = encoder.transform(df)
    assert matrix.mean() == pytest.approx(0.0, abs=1e-5)


def test_fit_label_classes_returns_sorted_unique_values():
    labels = pd.Series(["ddos", "normal", "ddos", "backdoor"])
    assert fit_label_classes(labels) == ["backdoor", "ddos", "normal"]


def test_encode_labels_maps_to_class_indices():
    labels = pd.Series(["ddos", "normal", "backdoor"])
    class_list = ["backdoor", "ddos", "normal"]
    encoded = encode_labels(labels, class_list)
    assert encoded.tolist() == [1, 2, 0]
    assert encoded.dtype == np.int64


def test_encode_labels_raises_on_unseen_label():
    labels = pd.Series(["ddos", "totally-new-attack"])
    with pytest.raises(ValueError):
        encode_labels(labels, ["ddos", "normal"])


def test_subsample_dataframe_returns_unchanged_when_under_cap():
    df = _sample_df()
    result = subsample_dataframe(df, max_rows=100, seed=1)
    assert len(result) == len(df)


def test_subsample_dataframe_caps_row_count():
    df = pd.DataFrame({"x": range(100)})
    result = subsample_dataframe(df, max_rows=10, seed=1)
    assert len(result) == 10


def test_subsample_dataframe_none_means_unlimited():
    df = _sample_df()
    result = subsample_dataframe(df, max_rows=None, seed=1)
    assert len(result) == len(df)


def test_split_train_test_covers_all_rows_without_overlap():
    df = pd.DataFrame({"x": range(50)})
    train_df, test_df = split_train_test(df, test_fraction=0.2, seed=1)
    assert len(train_df) + len(test_df) == 50
    assert set(train_df["x"]).isdisjoint(set(test_df["x"]))


def test_split_train_test_respects_fraction():
    df = pd.DataFrame({"x": range(100)})
    train_df, test_df = split_train_test(df, test_fraction=0.3, seed=1)
    assert len(test_df) == 30
    assert len(train_df) == 70
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_iot_preprocessing.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'data.iot_preprocessing'`

- [ ] **Step 3: Implement `data/iot_preprocessing.py`**

```python
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
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


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
    categorical_columns = [c for c in feature_names if train_df[c].dtype == object]
    numeric_columns = [c for c in feature_names if c not in categorical_columns]

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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_iot_preprocessing.py -v`
Expected: PASS (12 tests)

- [ ] **Step 5: Commit**

```bash
git add data/iot_preprocessing.py tests/test_iot_preprocessing.py
git commit -m "feat: add dataset-agnostic tabular preprocessing helpers"
```

---

### Task 2: `TabularDataset`

**Files:**
- Create: `data/tabular_dataset.py`
- Test: `tests/test_tabular_dataset.py`

**Interfaces:**
- Consumes: nothing new (plain `numpy`/`torch`).
- Produces: `TabularDataset(features: np.ndarray, labels: np.ndarray)` — a `torch.utils.data.Dataset` with `.targets: List[int]` (so `data/partition.py`'s existing `_extract_labels` works unmodified for Dirichlet partitioning of IoT data).

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_tabular_dataset.py
"""Tests for TabularDataset: the torch.utils.data.Dataset wrapper for tabular IoT data."""

import numpy as np
import pytest
import torch

from data.tabular_dataset import TabularDataset


def _make_dataset(num_samples=10, num_features=4):
    features = np.arange(num_samples * num_features, dtype=np.float64).reshape(
        num_samples, num_features
    )
    labels = np.array([i % 3 for i in range(num_samples)])
    return TabularDataset(features, labels)


def test_len_matches_number_of_rows():
    dataset = _make_dataset(num_samples=7)
    assert len(dataset) == 7


def test_getitem_returns_float_tensor_and_int_label():
    dataset = _make_dataset(num_samples=5, num_features=3)
    features, label = dataset[0]
    assert isinstance(features, torch.Tensor)
    assert features.dtype == torch.float32
    assert features.shape == (3,)
    assert isinstance(label, torch.Tensor)
    assert label.dtype == torch.int64


def test_targets_attribute_matches_labels():
    dataset = _make_dataset(num_samples=6)
    labels = [i % 3 for i in range(6)]
    assert dataset.targets == labels


def test_mismatched_lengths_raise_value_error():
    features = np.zeros((5, 2))
    labels = np.zeros(4)
    with pytest.raises(ValueError):
        TabularDataset(features, labels)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_tabular_dataset.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'data.tabular_dataset'`

- [ ] **Step 3: Implement `data/tabular_dataset.py`**

```python
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_tabular_dataset.py -v`
Expected: PASS (4 tests)

- [ ] **Step 5: Commit**

```bash
git add data/tabular_dataset.py tests/test_tabular_dataset.py
git commit -m "feat: add TabularDataset for preprocessed IoT feature matrices"
```

---

### Task 3: RAR extraction utility

**Files:**
- Create: `data/rar_extract.py`
- Test: `tests/test_rar_extract.py`

**Interfaces:**
- Produces: `RarToolNotFoundError(RuntimeError)`, `find_rar_tool() -> Optional[str]`, `extract_rar(archive_path: Path, dest_dir: Path) -> None`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_rar_extract.py
"""Tests for the 7z/unrar detection-and-extraction utility (mocked — no real archives)."""

from pathlib import Path
from unittest.mock import patch

import pytest

from data.rar_extract import RarToolNotFoundError, extract_rar, find_rar_tool


def test_find_rar_tool_returns_first_match_on_path():
    with patch("shutil.which", side_effect=lambda name: "/usr/bin/7z" if name == "7z" else None):
        assert find_rar_tool() == "/usr/bin/7z"


def test_find_rar_tool_returns_none_when_nothing_found():
    with patch("shutil.which", return_value=None), patch("pathlib.Path.exists", return_value=False):
        assert find_rar_tool() is None


def test_extract_rar_raises_clear_error_when_no_tool_found(tmp_path):
    with patch("data.rar_extract.find_rar_tool", return_value=None):
        with pytest.raises(RarToolNotFoundError, match="7-Zip"):
            extract_rar(tmp_path / "archive.rar", tmp_path / "out")


def test_extract_rar_invokes_7z_with_expected_arguments(tmp_path):
    archive = tmp_path / "archive.rar"
    archive.write_bytes(b"fake")
    dest = tmp_path / "out"

    with patch("data.rar_extract.find_rar_tool", return_value="/usr/bin/7z"), patch(
        "subprocess.run"
    ) as mock_run:
        extract_rar(archive, dest)

    mock_run.assert_called_once()
    command = mock_run.call_args[0][0]
    assert command[0] == "/usr/bin/7z"
    assert str(archive) in command
    assert dest.exists()


def test_extract_rar_invokes_unrar_with_expected_arguments(tmp_path):
    archive = tmp_path / "archive.rar"
    archive.write_bytes(b"fake")
    dest = tmp_path / "out"

    with patch("data.rar_extract.find_rar_tool", return_value="/usr/bin/unrar"), patch(
        "subprocess.run"
    ) as mock_run:
        extract_rar(archive, dest)

    command = mock_run.call_args[0][0]
    assert command[0] == "/usr/bin/unrar"
    assert command[1] == "x"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_rar_extract.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'data.rar_extract'`

- [ ] **Step 3: Implement `data/rar_extract.py`**

```python
"""Detects and shells out to a 7z/unrar binary to extract .rar archives.

No Python RAR library is used on purpose — `rarfile` still requires an
external `unrar`/`bsdtar` binary to do the actual decompression, so it adds
a dependency without removing the real requirement: some extraction tool
must be installed on the machine. N-BaIoT's Mirai/Gafgyt attack traffic
ships as `.rar` (not `.zip`) inside its outer zip archive.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from typing import Optional

_CANDIDATES = ("7z", "7za", "unrar", r"C:\Program Files\7-Zip\7z.exe")


class RarToolNotFoundError(RuntimeError):
    pass


def find_rar_tool() -> Optional[str]:
    """Return a path to a usable 7z/unrar binary, or None if none is found."""
    for candidate in _CANDIDATES:
        resolved = shutil.which(candidate)
        if resolved:
            return resolved
        if Path(candidate).exists():
            return candidate
    return None


def extract_rar(archive_path: Path, dest_dir: Path) -> None:
    """Extract `archive_path` into `dest_dir`, creating it if needed."""
    tool = find_rar_tool()
    if tool is None:
        raise RarToolNotFoundError(
            "No RAR extraction tool found (looked for 7z/7za/unrar on PATH). "
            "Install 7-Zip (e.g. `winget install 7zip.7zip`) and retry."
        )
    dest_dir.mkdir(parents=True, exist_ok=True)
    tool_name = Path(tool).stem.lower()
    if "unrar" in tool_name:
        command = [tool, "x", "-y", str(archive_path), str(dest_dir) + "/"]
    else:
        command = [tool, "x", str(archive_path), f"-o{dest_dir}", "-y"]
    subprocess.run(command, check=True, capture_output=True)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_rar_extract.py -v`
Expected: PASS (5 tests)

- [ ] **Step 5: Commit**

```bash
git add data/rar_extract.py tests/test_rar_extract.py
git commit -m "feat: add 7z/unrar detection-and-extraction utility"
```

---

### Task 4: `TabularMLP` model

**Files:**
- Create: `models/tabular_mlp.py`
- Modify: `models/cnn.py` (register the new model)
- Test: `tests/test_tabular_mlp.py`

**Interfaces:**
- Consumes: nothing new.
- Produces: `TabularMLP(in_channels: int, num_classes: int)` — an `nn.Module` using the same constructor-keyword convention as `SimpleCNN` (`in_channels` here means "input feature count", reusing the same parameter name so `models/cnn.py::build_model`'s existing `_MODEL_REGISTRY[name](in_channels=in_channels, num_classes=num_classes)` call works for both models with zero changes to `build_model` itself). Registered as `"mlp"`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_tabular_mlp.py
"""Tests for TabularMLP construction and forward-pass behavior."""

import pytest
import torch

from models.cnn import available_models, build_model


@pytest.mark.parametrize("input_dim,num_classes", [(39, 6), (115, 11), (6, 8)])
def test_forward_pass_output_shape(input_dim, num_classes):
    model = build_model("mlp", in_channels=input_dim, num_classes=num_classes)
    batch = torch.randn(4, input_dim)
    output = model(batch)
    assert output.shape == (4, num_classes)


def test_mlp_registered_in_available_models():
    assert "mlp" in available_models()


def test_mlp_handles_batch_size_one_in_eval_mode():
    model = build_model("mlp", in_channels=10, num_classes=3)
    model.eval()
    output = model(torch.randn(1, 10))
    assert output.shape == (1, 3)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_tabular_mlp.py -v`
Expected: FAIL with `ValueError: Unknown model name: 'mlp'`

- [ ] **Step 3: Implement `models/tabular_mlp.py`**

```python
"""Feedforward MLP for tabular IoT attack-detection features."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class TabularMLP(nn.Module):
    """A small MLP for tabular classification.

    Uses the same `in_channels`/`num_classes` constructor-keyword names as
    `models/cnn.py`'s `SimpleCNN` (here `in_channels` means "input feature
    count") so both models share one `build_model(name, in_channels,
    num_classes)` calling convention — `simulation/run.py` and
    `evaluation/experiment_runner.py` need no changes to support this model.
    """

    def __init__(self, in_channels: int = 32, num_classes: int = 10) -> None:
        super().__init__()
        self.fc1 = nn.Linear(in_channels, 128)
        self.bn1 = nn.BatchNorm1d(128)
        self.fc2 = nn.Linear(128, 64)
        self.bn2 = nn.BatchNorm1d(64)
        self.fc3 = nn.Linear(64, num_classes)
        self.dropout = nn.Dropout(0.25)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.bn1(self.fc1(x)))
        x = self.dropout(x)
        x = F.relu(self.bn2(self.fc2(x)))
        x = self.dropout(x)
        return self.fc3(x)
```

- [ ] **Step 4: Register it in `models/cnn.py`**

In `models/cnn.py`, add the import near the top and one entry to `_MODEL_REGISTRY`:

```python
# Add after the existing `import torch.nn.functional as F` line:
from models.tabular_mlp import TabularMLP
```

```python
# Change:
_MODEL_REGISTRY = {"cnn": SimpleCNN}
# To:
_MODEL_REGISTRY = {"cnn": SimpleCNN, "mlp": TabularMLP}
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `pytest tests/test_tabular_mlp.py tests/test_model.py -v`
Expected: PASS (6 tests total — 3 new + 3 existing CNN tests still passing, confirming no regression)

- [ ] **Step 6: Commit**

```bash
git add models/tabular_mlp.py models/cnn.py tests/test_tabular_mlp.py
git commit -m "feat: add TabularMLP model, registered as 'mlp'"
```

---

### Task 5: Shared IoT experiment builders

**Files:**
- Create: `data/iot_datasets.py` (this task only — `IoTExperimentData`, `_build_partitioned_experiment`, `_build_natural_client_experiment`, `_build_attack_pool`; no real dataset loaders yet)
- Test: `tests/test_iot_datasets.py` (this task's section only)

**Interfaces:**
- Consumes: `FeatureEncoder`/`fit_feature_encoder`/`fit_label_classes`/`encode_labels`/`subsample_dataframe`/`split_train_test` (Task 1), `TabularDataset` (Task 2), `data.partition.partition_iid`/`partition_dirichlet`/`make_client_subsets` (existing, unchanged).
- Produces: `IoTExperimentData` (dataclass: `train_client_datasets: Dict[int, Dataset]`, `test_dataset: Dataset`, `attack_pool: Optional[Dataset]`, `input_dim: int`, `num_classes: int`, `class_names: List[str]`); `_build_partitioned_experiment(df, label_column, benign_label, num_clients, partition_type, dirichlet_alpha, max_rows, seed) -> IoTExperimentData`; `_build_natural_client_experiment(device_dataframes: Dict[str, pd.DataFrame], label_column, benign_label, max_rows_per_device, seed) -> IoTExperimentData`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_iot_datasets.py
"""Tests for the shared IoT experiment builders and, in later tasks, per-dataset loaders."""

import numpy as np
import pandas as pd
import pytest

from data.iot_datasets import _build_natural_client_experiment, _build_partitioned_experiment


def _make_partitioned_df(num_rows=200, seed=0):
    rng = np.random.default_rng(seed)
    labels = rng.choice(["normal", "ddos", "backdoor"], size=num_rows, p=[0.6, 0.25, 0.15])
    return pd.DataFrame(
        {
            "proto": rng.choice(["tcp", "udp"], size=num_rows),
            "duration": rng.random(num_rows) * 10,
            "bytes": rng.integers(0, 1000, size=num_rows),
            "label": labels,
        }
    )


class TestBuildPartitionedExperiment:
    def test_produces_requested_number_of_clients(self):
        df = _make_partitioned_df()
        result = _build_partitioned_experiment(
            df,
            label_column="label",
            benign_label="normal",
            num_clients=5,
            partition_type="iid",
            dirichlet_alpha=0.5,
            max_rows=None,
            seed=42,
        )
        assert len(result.train_client_datasets) == 5

    def test_input_dim_excludes_label_column(self):
        df = _make_partitioned_df()
        result = _build_partitioned_experiment(
            df, "label", "normal", num_clients=3, partition_type="iid",
            dirichlet_alpha=0.5, max_rows=None, seed=42,
        )
        assert result.input_dim == 3  # proto, duration, bytes

    def test_num_classes_matches_label_vocabulary(self):
        df = _make_partitioned_df()
        result = _build_partitioned_experiment(
            df, "label", "normal", num_clients=3, partition_type="iid",
            dirichlet_alpha=0.5, max_rows=None, seed=42,
        )
        assert result.num_classes == 3
        assert result.class_names == ["backdoor", "ddos", "normal"]

    def test_attack_pool_excludes_benign_rows(self):
        df = _make_partitioned_df()
        result = _build_partitioned_experiment(
            df, "label", "normal", num_clients=3, partition_type="iid",
            dirichlet_alpha=0.5, max_rows=None, seed=42,
        )
        assert result.attack_pool is not None
        assert len(result.attack_pool) > 0
        assert len(result.attack_pool) < len(df)

    def test_attack_pool_none_when_benign_label_absent(self):
        df = _make_partitioned_df()
        result = _build_partitioned_experiment(
            df, "label", benign_label=None, num_clients=3, partition_type="iid",
            dirichlet_alpha=0.5, max_rows=None, seed=42,
        )
        assert result.attack_pool is None

    def test_max_rows_caps_dataset_size(self):
        df = _make_partitioned_df(num_rows=200)
        result = _build_partitioned_experiment(
            df, "label", "normal", num_clients=2, partition_type="iid",
            dirichlet_alpha=0.5, max_rows=50, seed=42,
        )
        total_train = sum(len(d) for d in result.train_client_datasets.values())
        assert total_train + len(result.test_dataset) == 50

    def test_dirichlet_partition_type_works(self):
        df = _make_partitioned_df()
        result = _build_partitioned_experiment(
            df, "label", "normal", num_clients=4, partition_type="dirichlet",
            dirichlet_alpha=0.5, max_rows=None, seed=42,
        )
        assert len(result.train_client_datasets) == 4


class TestBuildNaturalClientExperiment:
    def _device_frames(self):
        rng = np.random.default_rng(1)
        return {
            "device_a": pd.DataFrame(
                {
                    "sensor_x": rng.random(40),
                    "type": rng.choice(["normal", "ddos"], size=40, p=[0.7, 0.3]),
                }
            ),
            "device_b": pd.DataFrame(
                {
                    "sensor_x": rng.random(30),
                    "type": rng.choice(["normal", "backdoor"], size=30, p=[0.7, 0.3]),
                }
            ),
        }

    def test_one_client_per_device(self):
        result = _build_natural_client_experiment(
            self._device_frames(), label_column="type", benign_label="normal",
            max_rows_per_device=None, seed=7,
        )
        assert len(result.train_client_datasets) == 2

    def test_class_vocabulary_spans_all_devices(self):
        result = _build_natural_client_experiment(
            self._device_frames(), label_column="type", benign_label="normal",
            max_rows_per_device=None, seed=7,
        )
        assert result.class_names == ["backdoor", "ddos", "normal"]

    def test_test_dataset_pools_all_devices(self):
        result = _build_natural_client_experiment(
            self._device_frames(), label_column="type", benign_label="normal",
            max_rows_per_device=None, seed=7,
        )
        assert len(result.test_dataset) > 0

    def test_max_rows_per_device_caps_each_device_independently(self):
        result = _build_natural_client_experiment(
            self._device_frames(), label_column="type", benign_label="normal",
            max_rows_per_device=10, seed=7,
        )
        for dataset in result.train_client_datasets.values():
            assert len(dataset) <= 10
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_iot_datasets.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'data.iot_datasets'`

- [ ] **Step 3: Implement `data/iot_datasets.py`** (this task's portion)

```python
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_iot_datasets.py -v`
Expected: PASS (11 tests)

- [ ] **Step 5: Commit**

```bash
git add data/iot_datasets.py tests/test_iot_datasets.py
git commit -m "feat: add shared IoT experiment builders (partitioned + natural-client)"
```

---

### Task 6: CICIoT2023 loader

**Files:**
- Modify: `data/iot_datasets.py` (add `_load_ciciot2023_dataframe`, `load_ciciot2023`)
- Modify: `tests/test_iot_datasets.py` (add this dataset's tests)

**Interfaces:**
- Consumes: `_build_partitioned_experiment` (Task 5).
- Produces: `load_ciciot2023(data_dir: str, num_clients: int, partition_type: str, dirichlet_alpha: float, max_rows: Optional[int], seed: int) -> IoTExperimentData`.

This dataset's source CSV (`CICIoT2023 (2).csv`) contains only DDoS-attack rows (6 subtypes, no benign traffic — confirmed by inspecting the real file) — so it has no `benign_label`, and its `attack_pool` will always be `None` (verified: `real_traffic` isn't a meaningful attack for this particular slice).

- [ ] **Step 1: Write the failing test**

Append to `tests/test_iot_datasets.py`:

```python
from pathlib import Path

from data.iot_datasets import load_ciciot2023


class TestLoadCiciot2023:
    def _write_fixture(self, tmp_path: Path) -> Path:
        content = (
            "flow_duration,Header_Length,Protocol Type,Rate,label\n"
            "0,54,6,17.3,DDoS-TCP_Flood\n"
            "0,53,5,0.3,DDoS-TCP_Flood\n"
            "1,60,17,5.0,DDoS-UDP_Flood\n"
            "2,70,1,9.0,DDoS-ICMP_Flood\n"
            "0,54,6,3.9,DDoS-TCP_Flood\n"
            "1,61,17,4.4,DDoS-UDP_Flood\n"
            "2,71,1,8.8,DDoS-ICMP_Flood\n"
            "0,55,6,2.1,DDoS-TCP_Flood\n"
        )
        data_dir = tmp_path / "ZT-Fedtrust Datasets"
        data_dir.mkdir()
        (data_dir / "CICIoT2023 (2).csv").write_text(content, encoding="utf-8")
        return data_dir

    def test_loads_and_produces_expected_shape(self, tmp_path):
        data_dir = self._write_fixture(tmp_path)
        result = load_ciciot2023(
            str(data_dir), num_clients=2, partition_type="iid",
            dirichlet_alpha=0.5, max_rows=None, seed=1,
        )
        assert result.input_dim == 4  # flow_duration, Header_Length, Protocol Type, Rate
        assert result.num_classes == 3
        assert len(result.train_client_datasets) == 2

    def test_no_benign_traffic_means_no_attack_pool(self, tmp_path):
        data_dir = self._write_fixture(tmp_path)
        result = load_ciciot2023(
            str(data_dir), num_clients=2, partition_type="iid",
            dirichlet_alpha=0.5, max_rows=None, seed=1,
        )
        assert result.attack_pool is None
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_iot_datasets.py::TestLoadCiciot2023 -v`
Expected: FAIL with `ImportError: cannot import name 'load_ciciot2023'`

- [ ] **Step 3: Add to `data/iot_datasets.py`**

```python
# Add near the top with the other imports:
from pathlib import Path

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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_iot_datasets.py -v`
Expected: PASS (13 tests)

- [ ] **Step 5: Commit**

```bash
git add data/iot_datasets.py tests/test_iot_datasets.py
git commit -m "feat: add CICIoT2023 loader"
```

---

### Task 7: UNSW-NB15 loader

**Files:**
- Modify: `data/iot_datasets.py` (add `_load_unsw_nb15_dataframe`, `load_unsw_nb15`)
- Modify: `tests/test_iot_datasets.py`

**Interfaces:**
- Consumes: `_build_partitioned_experiment` (Task 5).
- Produces: `load_unsw_nb15(data_dir, num_clients, partition_type, dirichlet_alpha, max_rows, seed) -> IoTExperimentData`.

Source: `UNSW-NB15 dataset.zip` → `CSV Files/Training and Testing Sets/UNSW_NB15_training-set.csv` + `.../UNSW_NB15_testing-set.csv` (both already labeled with `attack_cat`/`label`; concatenated here and re-split by the shared builder so `max_rows`/seed behave consistently with every other loader). Drops `id` (row index, not predictive) and the binary `label` column (redundant with `attack_cat`, the actual multi-class target). Target: `attack_cat`. Benign class: `"Normal"`.

- [ ] **Step 1: Write the failing test**

```python
import io
import zipfile

from data.iot_datasets import load_unsw_nb15


class TestLoadUnswNb15:
    def _write_fixture(self, tmp_path):
        header = (
            "id,dur,proto,service,state,spkts,dpkts,sbytes,dbytes,attack_cat,label\n"
        )
        rows = [
            "1,0.12,tcp,-,FIN,6,4,258,172,Normal,0\n",
            "2,0.65,tcp,-,FIN,14,38,734,42014,Normal,0\n",
            "3,1.62,tcp,-,FIN,8,16,364,13186,Exploits,1\n",
            "4,0.02,udp,dns,CON,2,2,146,178,Reconnaissance,1\n",
            "5,0.09,tcp,http,FIN,10,8,600,900,Normal,0\n",
            "6,0.44,tcp,-,FIN,12,20,700,1200,Exploits,1\n",
            "7,0.31,udp,dns,CON,3,3,150,190,Reconnaissance,1\n",
            "8,0.77,tcp,http,FIN,9,7,620,880,Normal,0\n",
        ]
        data_dir = tmp_path / "ZT-Fedtrust Datasets"
        data_dir.mkdir()
        with zipfile.ZipFile(data_dir / "UNSW-NB15 dataset.zip", "w") as zf:
            zf.writestr(
                "CSV Files/Training and Testing Sets/UNSW_NB15_training-set.csv",
                header + "".join(rows[:6]),
            )
            zf.writestr(
                "CSV Files/Training and Testing Sets/UNSW_NB15_testing-set.csv",
                header + "".join(rows[6:]),
            )
        return data_dir

    def test_loads_and_drops_id_and_binary_label(self, tmp_path):
        data_dir = self._write_fixture(tmp_path)
        result = load_unsw_nb15(
            str(data_dir), num_clients=2, partition_type="iid",
            dirichlet_alpha=0.5, max_rows=None, seed=1,
        )
        assert result.input_dim == 7  # dur, proto, service, state, spkts, dpkts, sbytes, dbytes minus id/label/attack_cat = 8-1(dbytes stays)
        assert "Normal" in result.class_names

    def test_attack_pool_excludes_normal_rows(self, tmp_path):
        data_dir = self._write_fixture(tmp_path)
        result = load_unsw_nb15(
            str(data_dir), num_clients=2, partition_type="iid",
            dirichlet_alpha=0.5, max_rows=None, seed=1,
        )
        assert result.attack_pool is not None
        assert len(result.attack_pool) > 0
```

Note: `dbytes` is a numeric feature column too; recount when implementing — the fixture has columns `id, dur, proto, service, state, spkts, dpkts, sbytes, dbytes, attack_cat, label` (11 total), minus `id`, `label`, `attack_cat` (dropped/target) leaves `dur, proto, service, state, spkts, dpkts, sbytes, dbytes` = 8 feature columns. Fix the assertion to `assert result.input_dim == 8` before running.

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_iot_datasets.py::TestLoadUnswNb15 -v`
Expected: FAIL with `ImportError: cannot import name 'load_unsw_nb15'`

- [ ] **Step 3: Add to `data/iot_datasets.py`**

```python
# Add near the top:
import zipfile

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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_iot_datasets.py -v`
Expected: PASS (15 tests)

- [ ] **Step 5: Commit**

```bash
git add data/iot_datasets.py tests/test_iot_datasets.py
git commit -m "feat: add UNSW-NB15 loader"
```

---

### Task 8: TON_IoT network loader

**Files:**
- Modify: `data/iot_datasets.py` (add `_load_ton_iot_network_dataframe`, `load_ton_iot_network`)
- Modify: `tests/test_iot_datasets.py`

**Interfaces:**
- Consumes: `_build_partitioned_experiment` (Task 5).
- Produces: `load_ton_iot_network(data_dir, num_clients, partition_type, dirichlet_alpha, max_rows, seed) -> IoTExperimentData`.

Source: `Train_Test_datasets.zip` → `Train_Test_datasets/Train_Test_Network_dataset/train_test_network.csv`. Drops identity/free-text columns that would either leak identity or blow up categorical cardinality: `src_ip`, `dst_ip`, `dns_query`, `ssl_subject`, `ssl_issuer`, `http_uri`, `http_user_agent`. Also drops the binary `label` (redundant with the multi-class `type` target). Target: `type`. Benign class: `"normal"`.

- [ ] **Step 1: Write the failing test**

```python
from data.iot_datasets import load_ton_iot_network


class TestLoadTonIotNetwork:
    def _write_fixture(self, tmp_path):
        header = (
            "src_ip,src_port,dst_ip,dst_port,proto,service,duration,src_bytes,"
            "dst_bytes,conn_state,dns_query,ssl_subject,ssl_issuer,http_uri,"
            "http_user_agent,label,type\n"
        )
        rows = [
            "192.168.1.1,80,192.168.1.2,443,tcp,-,0.1,100,200,SF,-,-,-,-,normal\n".replace(
                "SF,-,-,-,-,normal", "SF,-,-,-,-,0,normal"
            ),
            "192.168.1.3,53,192.168.1.4,53,udp,dns,0.05,60,80,SF,example.com,-,-,-,0,normal\n",
            "192.168.1.5,4444,192.168.1.6,80,tcp,-,2.3,900,50,REJ,-,-,-,-,1,backdoor\n",
            "192.168.1.7,80,192.168.1.8,80,tcp,http,0.9,500,600,SF,-,-,/x,curl,1,ddos\n",
            "192.168.1.9,80,192.168.1.10,443,tcp,-,0.2,110,210,SF,-,-,-,-,0,normal\n",
            "192.168.1.11,4444,192.168.1.12,80,tcp,-,3.1,950,40,REJ,-,-,-,-,1,backdoor\n",
        ]
        data_dir = tmp_path / "ZT-Fedtrust Datasets"
        data_dir.mkdir()
        import zipfile

        with zipfile.ZipFile(data_dir / "Train_Test_datasets.zip", "w") as zf:
            zf.writestr(
                "Train_Test_datasets/Train_Test_Network_dataset/train_test_network.csv",
                header + "".join(rows),
            )
        return data_dir

    def test_drops_identity_columns(self, tmp_path):
        data_dir = self._write_fixture(tmp_path)
        result = load_ton_iot_network(
            str(data_dir), num_clients=2, partition_type="iid",
            dirichlet_alpha=0.5, max_rows=None, seed=1,
        )
        # src_port,dst_port,proto,service,duration,src_bytes,dst_bytes,conn_state = 8 features
        assert result.input_dim == 8
        assert "normal" in result.class_names

    def test_attack_pool_present(self, tmp_path):
        data_dir = self._write_fixture(tmp_path)
        result = load_ton_iot_network(
            str(data_dir), num_clients=2, partition_type="iid",
            dirichlet_alpha=0.5, max_rows=None, seed=1,
        )
        assert result.attack_pool is not None
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_iot_datasets.py::TestLoadTonIotNetwork -v`
Expected: FAIL with `ImportError: cannot import name 'load_ton_iot_network'`

- [ ] **Step 3: Add to `data/iot_datasets.py`**

```python
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_iot_datasets.py -v`
Expected: PASS (17 tests)

- [ ] **Step 5: Commit**

```bash
git add data/iot_datasets.py tests/test_iot_datasets.py
git commit -m "feat: add TON_IoT network loader"
```

---

### Task 9: TON_IoT devices loader (natural clients)

**Files:**
- Modify: `data/iot_datasets.py` (add `_load_ton_iot_device_dataframes`, `load_ton_iot_devices`)
- Modify: `tests/test_iot_datasets.py`

**Interfaces:**
- Consumes: `_build_natural_client_experiment` (Task 5).
- Produces: `load_ton_iot_devices(data_dir, max_rows_per_device, seed) -> IoTExperimentData`. No `num_clients`/`partition_type` params — natural-client datasets always produce one client per real device (7, fixed).

Source: 7 device CSVs inside `Train_Test_datasets.zip`, under `Train_Test_datasets/Train_Test_IoT_dataset/`: `Train_Test_IoT_Fridge.csv`, `..._Garage_Door.csv`, `..._GPS_Tracker.csv`, `..._Modbus.csv`, `..._Motion_Light.csv`, `..._Thermostat.csv`, `..._Weather.csv`. Each drops `date`/`time`/binary `label`. `_build_natural_client_experiment` unions each device's distinct sensor columns into one shared `FeatureEncoder` (confirmed real schemas: Fridge has `fridge_temperature, temp_condition`; Garage_Door has `door_state, sphone_signal`; etc. — no two devices share a sensor column, so the encoder's `feature_names` end up being the union of all of them, and `FeatureEncoder.transform` naturally leaves a device's non-applicable columns absent from ITS OWN dataframe — see Step 3 for how per-device dataframes get reindexed to the full shared column set before being handed to the encoder). Target: `type`. Benign class: `"normal"` (confirmed present in all 7 device files).

- [ ] **Step 1: Write the failing test**

```python
from data.iot_datasets import load_ton_iot_devices


class TestLoadTonIotDevices:
    def _write_fixture(self, tmp_path):
        import zipfile

        data_dir = tmp_path / "ZT-Fedtrust Datasets"
        data_dir.mkdir()
        with zipfile.ZipFile(data_dir / "Train_Test_datasets.zip", "w") as zf:
            zf.writestr(
                "Train_Test_datasets/Train_Test_IoT_dataset/Train_Test_IoT_Fridge.csv",
                "date,time,fridge_temperature,temp_condition,label,type\n"
                "25-Apr-19,19:19:40,9,high,1,ddos\n"
                "25-Apr-19,19:19:45,4.6,low,0,normal\n"
                "25-Apr-19,19:19:50,5.1,low,0,normal\n",
            )
            zf.writestr(
                "Train_Test_datasets/Train_Test_IoT_dataset/Train_Test_IoT_Garage_Door.csv",
                "date,time,door_state,sphone_signal,label,type\n"
                "25-Apr-19,19:19:40,open,1,1,backdoor\n"
                "25-Apr-19,19:19:45,closed,0,0,normal\n"
                "25-Apr-19,19:19:50,closed,0,0,normal\n",
            )
        return data_dir

    def test_produces_one_client_per_device(self, tmp_path):
        data_dir = self._write_fixture(tmp_path)
        result = load_ton_iot_devices(str(data_dir), max_rows_per_device=None, seed=1)
        assert len(result.train_client_datasets) == 2

    def test_shared_feature_space_across_devices(self, tmp_path):
        data_dir = self._write_fixture(tmp_path)
        result = load_ton_iot_devices(str(data_dir), max_rows_per_device=None, seed=1)
        # union of fridge_temperature, temp_condition, door_state, sphone_signal
        assert result.input_dim == 4
        for dataset in result.train_client_datasets.values():
            features, _ = dataset[0]
            assert features.shape == (4,)

    def test_class_vocabulary_spans_both_devices(self, tmp_path):
        data_dir = self._write_fixture(tmp_path)
        result = load_ton_iot_devices(str(data_dir), max_rows_per_device=None, seed=1)
        assert result.class_names == ["backdoor", "ddos", "normal"]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_iot_datasets.py::TestLoadTonIotDevices -v`
Expected: FAIL with `ImportError: cannot import name 'load_ton_iot_devices'`

- [ ] **Step 3: Add to `data/iot_datasets.py`**

```python
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_iot_datasets.py -v`
Expected: PASS (20 tests)

- [ ] **Step 5: Commit**

```bash
git add data/iot_datasets.py tests/test_iot_datasets.py
git commit -m "feat: add TON_IoT device-telemetry loader (natural clients)"
```

---

### Task 10: N-BaIoT loader (natural clients, RAR extraction)

**Files:**
- Modify: `data/iot_datasets.py` (add `_read_family_attack_csvs`, `_load_nbaiot_device_dataframe`, `_load_nbaiot_device_dataframes`, `load_nbaiot`)
- Modify: `tests/test_iot_datasets.py`

**Interfaces:**
- Consumes: `_build_natural_client_experiment` (Task 5), `extract_rar` (Task 3).
- Produces: `load_nbaiot(data_dir, max_rows_per_device, seed) -> IoTExperimentData`. 9 natural clients (or fewer, for the 2 devices with no Mirai traffic — still both present, just missing the `mirai_*` label rows).

Source: `detection+of+iot+botnet+attacks+n+baiot.zip`. Per device folder: `benign_traffic.csv` (plain, in the outer zip) plus `gafgyt_attacks.rar` and (for 7 of the 9 devices) `mirai_attacks.rar`, each containing several attack-subtype CSVs (e.g. `combo.csv`, `scan.csv`) sharing the same 115-feature schema as `benign_traffic.csv`. Rather than hardcoding the exact subtype filenames (undocumented in the dataset's own description file, and not verifiable on this machine without an installed extractor), the loader discovers whatever `.csv` files an extracted `.rar` actually contains and derives each one's label from its filename — `f"{family}_{path.stem}"` (e.g. `"gafgyt_combo"`, `"mirai_syn"`), with `benign_traffic.csv` labeled `"benign"`.

- [ ] **Step 1: Write the failing tests**

```python
from unittest.mock import patch

from data.iot_datasets import _read_family_attack_csvs, load_nbaiot


class TestReadFamilyAttackCsvs:
    def test_labels_derived_from_filename(self, tmp_path):
        extracted = tmp_path / "gafgyt"
        extracted.mkdir()
        (extracted / "combo.csv").write_text("f1,f2\n1.0,2.0\n1.1,2.1\n", encoding="utf-8")
        (extracted / "scan.csv").write_text("f1,f2\n3.0,4.0\n", encoding="utf-8")

        df = _read_family_attack_csvs(extracted, family="gafgyt")
        assert set(df["label"]) == {"gafgyt_combo", "gafgyt_scan"}
        assert len(df) == 3


class TestLoadNbaiot:
    def _write_fixture(self, tmp_path):
        import zipfile

        data_dir = tmp_path / "ZT-Fedtrust Datasets"
        data_dir.mkdir()
        zip_path = data_dir / "detection+of+iot+botnet+attacks+n+baiot.zip"
        with zipfile.ZipFile(zip_path, "w") as zf:
            zf.writestr(
                "Danmini_Doorbell/benign_traffic.csv",
                "f1,f2\n0.1,0.2\n0.15,0.22\n0.12,0.19\n",
            )
            zf.writestr("Danmini_Doorbell/gafgyt_attacks.rar", b"placeholder")
            zf.writestr(
                "Ecobee_Thermostat/benign_traffic.csv",
                "f1,f2\n0.3,0.4\n0.31,0.41\n0.29,0.39\n",
            )
            zf.writestr("Ecobee_Thermostat/gafgyt_attacks.rar", b"placeholder")
        return data_dir

    def test_produces_one_client_per_device(self, tmp_path):
        data_dir = self._write_fixture(tmp_path)

        def fake_extract_rar(archive_path, dest_dir):
            dest_dir.mkdir(parents=True, exist_ok=True)
            (dest_dir / "combo.csv").write_text("f1,f2\n9.0,9.1\n9.2,9.3\n", encoding="utf-8")

        with patch("data.iot_datasets.extract_rar", side_effect=fake_extract_rar):
            result = load_nbaiot(str(data_dir), max_rows_per_device=None, seed=1)

        assert len(result.train_client_datasets) == 2
        assert "benign" in result.class_names
        assert "gafgyt_combo" in result.class_names
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_iot_datasets.py::TestReadFamilyAttackCsvs tests/test_iot_datasets.py::TestLoadNbaiot -v`
Expected: FAIL with `ImportError: cannot import name '_read_family_attack_csvs'`

- [ ] **Step 3: Add to `data/iot_datasets.py`**

```python
from data.rar_extract import extract_rar

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
    device_dataframes = _load_nbaiot_device_dataframes(data_dir)
    return _build_natural_client_experiment(
        device_dataframes,
        label_column="label",
        benign_label="benign",
        max_rows_per_device=max_rows_per_device,
        seed=seed,
    )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_iot_datasets.py -v`
Expected: PASS (23 tests)

- [ ] **Step 5: Commit**

```bash
git add data/iot_datasets.py tests/test_iot_datasets.py
git commit -m "feat: add N-BaIoT loader with cached RAR extraction"
```

---

### Task 11: IoT dataset registry

**Files:**
- Modify: `data/iot_datasets.py` (add `IOT_SUPPORTED_DATASETS`, `load_iot_experiment`)
- Modify: `tests/test_iot_datasets.py`

**Interfaces:**
- Consumes: all five loaders (Tasks 6–10).
- Produces: `IOT_SUPPORTED_DATASETS: Tuple[str, ...]` (`"ciciot2023"`, `"unsw_nb15"`, `"ton_iot_network"`, `"ton_iot_devices"`, `"nbaiot"` — `"botiot"` deliberately excluded, see below), `load_iot_experiment(name, data_dir, max_rows, num_clients, partition_type, dirichlet_alpha, seed) -> IoTExperimentData`.

`"botiot"` is deliberately **not** included in `IOT_SUPPORTED_DATASETS` yet — the design spec calls it a disabled placeholder pending the user supplying real BoT-IoT data; selecting it in a config should fail with a clear, dataset-specific message rather than a generic "unsupported dataset" error.

- [ ] **Step 1: Write the failing tests**

```python
from data.iot_datasets import IOT_SUPPORTED_DATASETS, load_iot_experiment


class TestLoadIotExperimentDispatch:
    def test_supported_datasets_tuple_contents(self):
        assert set(IOT_SUPPORTED_DATASETS) == {
            "ciciot2023", "unsw_nb15", "ton_iot_network", "ton_iot_devices", "nbaiot",
        }

    def test_dispatches_to_ciciot2023(self, tmp_path):
        data_dir = TestLoadCiciot2023()._write_fixture(tmp_path)
        result = load_iot_experiment(
            "ciciot2023", str(data_dir), max_rows=None, num_clients=2,
            partition_type="iid", dirichlet_alpha=0.5, seed=1,
        )
        assert len(result.train_client_datasets) == 2

    def test_unknown_dataset_raises_value_error(self, tmp_path):
        with pytest.raises(ValueError, match="botiot"):
            load_iot_experiment(
                "botiot", str(tmp_path), max_rows=None, num_clients=2,
                partition_type="iid", dirichlet_alpha=0.5, seed=1,
            )
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_iot_datasets.py::TestLoadIotExperimentDispatch -v`
Expected: FAIL with `ImportError: cannot import name 'IOT_SUPPORTED_DATASETS'`

- [ ] **Step 3: Add to `data/iot_datasets.py`**

```python
IOT_SUPPORTED_DATASETS = ("ciciot2023", "unsw_nb15", "ton_iot_network", "ton_iot_devices", "nbaiot")

_NATURAL_CLIENT_DATASETS = ("ton_iot_devices", "nbaiot")


def load_iot_experiment(
    name: str,
    data_dir: str,
    max_rows: Optional[int],
    num_clients: int,
    partition_type: str,
    dirichlet_alpha: float,
    seed: int,
) -> IoTExperimentData:
    """Dispatch to the right per-dataset loader by name.

    `num_clients`/`partition_type`/`dirichlet_alpha` are ignored for
    natural-client datasets (`ton_iot_devices`, `nbaiot`) — those always
    produce one client per real device.
    """
    if name == "ciciot2023":
        return load_ciciot2023(data_dir, num_clients, partition_type, dirichlet_alpha, max_rows, seed)
    if name == "unsw_nb15":
        return load_unsw_nb15(data_dir, num_clients, partition_type, dirichlet_alpha, max_rows, seed)
    if name == "ton_iot_network":
        return load_ton_iot_network(data_dir, num_clients, partition_type, dirichlet_alpha, max_rows, seed)
    if name == "ton_iot_devices":
        return load_ton_iot_devices(data_dir, max_rows, seed)
    if name == "nbaiot":
        return load_nbaiot(data_dir, max_rows, seed)
    if name == "botiot":
        raise ValueError(
            "botiot is a disabled placeholder: the real BoT-IoT dataset has not been "
            "supplied yet. Use one of: " + ", ".join(IOT_SUPPORTED_DATASETS)
        )
    raise ValueError(f"Unknown IoT dataset: {name!r}. Supported: {list(IOT_SUPPORTED_DATASETS)}")
```

`iot_dataset_shape` is intentionally *not* added here: nothing in this plan needs shape without the rest of the experiment bundle — Task 12's `load_experiment_data` calls `load_iot_experiment` directly and reads `.input_dim`/`.num_classes` off the result. Adding a shape-only wrapper with no caller would be dead code.

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_iot_datasets.py -v`
Expected: PASS (26 tests)

- [ ] **Step 5: Commit**

```bash
git add data/iot_datasets.py tests/test_iot_datasets.py
git commit -m "feat: add IoT dataset registry and load_iot_experiment dispatcher"
```

---

### Task 12: Integrate into `simulation/run.py`

**Files:**
- Modify: `simulation/run.py`
- Modify: `tests/test_config_validation.py` (add IoT dataset acceptance cases)

**Interfaces:**
- Consumes: `IOT_SUPPORTED_DATASETS`, `load_iot_experiment` (Task 11); `IoTExperimentData` (Task 5).
- Produces: `DataConfig.max_rows: Optional[int] = None` (new field); `load_experiment_data(config: SimulationConfig, device: torch.device) -> Tuple[nn.Module, Dict[int, Client], Dataset, Optional[Dataset]]` (returns `global_model, clients, test_dataset, attack_pool`) — the single function both this file and `evaluation/experiment_runner.py` now call instead of their own `dataset_shape`/`load_dataset`/`build_model`/`build_clients` sequence.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_config_validation.py`:

```python
def test_iot_dataset_name_passes_validation():
    validate_config(_base_config(**{"data.dataset": "ciciot2023"}))  # should not raise


def test_max_rows_defaults_to_none():
    config = _base_config()
    assert config.data.max_rows is None
```

Create `tests/test_simulation_iot_integration.py`:

```python
"""Tests for simulation/run.py's IoT-vs-vision dataset dispatch."""

from unittest.mock import patch

import pandas as pd
import pytest
import torch

from data.iot_datasets import IoTExperimentData
from data.tabular_dataset import TabularDataset
from simulation.run import DataConfig, FederatedConfig, LoggingConfig, ModelConfig, SimulationConfig, load_experiment_data


def _iot_config(dataset_name: str) -> SimulationConfig:
    return SimulationConfig(
        seed=1,
        device="cpu",
        data=DataConfig(
            dataset=dataset_name, data_dir="./fake", num_clients=2, partition_type="iid",
            dirichlet_alpha=0.5, batch_size=8, max_rows=100,
        ),
        model=ModelConfig(name="mlp"),
        federated=FederatedConfig(num_rounds=1, clients_per_round=1, local_epochs=1, learning_rate=0.01, momentum=0.9),
        logging=LoggingConfig(log_dir="./results/logs", log_level="INFO", results_dir="./results"),
    )


def _fake_experiment_data() -> IoTExperimentData:
    features = torch.randn(20, 5).numpy()
    labels = (torch.arange(20) % 3).numpy()
    train_datasets = {
        0: TabularDataset(features[:10], labels[:10]),
        1: TabularDataset(features[10:], labels[10:]),
    }
    return IoTExperimentData(
        train_client_datasets=train_datasets,
        test_dataset=TabularDataset(features, labels),
        attack_pool=TabularDataset(features[:5], labels[:5]),
        input_dim=5,
        num_classes=3,
        class_names=["a", "b", "c"],
    )


def test_iot_dataset_uses_load_iot_experiment():
    config = _iot_config("ciciot2023")
    with patch("simulation.run.load_iot_experiment", return_value=_fake_experiment_data()) as mock_load:
        model, clients, test_dataset, attack_pool = load_experiment_data(config, torch.device("cpu"))

    mock_load.assert_called_once()
    assert len(clients) == 2
    assert len(test_dataset) == 20
    assert attack_pool is not None
    assert len(attack_pool) == 5


def test_vision_dataset_returns_none_attack_pool():
    config = _iot_config("mnist")
    config.model.name = "cnn"
    with patch("simulation.run.load_dataset") as mock_load_dataset, patch(
        "simulation.run.dataset_shape", return_value=(1, 10)
    ), patch("simulation.run.build_clients") as mock_build_clients:
        mock_load_dataset.return_value = TabularDataset(
            torch.randn(10, 3).numpy(), (torch.arange(10) % 2).numpy()
        )
        mock_build_clients.return_value = {}
        model, clients, test_dataset, attack_pool = load_experiment_data(config, torch.device("cpu"))

    assert attack_pool is None
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_config_validation.py tests/test_simulation_iot_integration.py -v`
Expected: FAIL — `test_max_rows_defaults_to_none` fails with `TypeError` (unexpected keyword `max_rows`), `test_iot_dataset_name_passes_validation` fails with `ValueError`, and the new integration test file fails with `ImportError: cannot import name 'load_experiment_data'`.

- [ ] **Step 3: Modify `simulation/run.py`**

Add the import (near the existing `from data.dataset import ...` line):

```python
from data.iot_datasets import IOT_SUPPORTED_DATASETS, load_iot_experiment
```

Add `Tuple` to the existing `from typing import Any, Dict, List, Optional` line (becomes `from typing import Any, Dict, List, Optional, Tuple`), and add `from torch.utils.data import Dataset` near the other imports (check first — it may already be imported indirectly; if `Dataset` isn't already a top-level name in this file, add the import explicitly).

Add `max_rows` to `DataConfig` (find the existing dataclass and add one field at the end, so it stays keyword-compatible with every existing call site):

```python
@dataclass
class DataConfig:
    dataset: str
    data_dir: str
    num_clients: int
    partition_type: str
    dirichlet_alpha: float
    batch_size: int
    max_rows: Optional[int] = None
```

Update `validate_config`'s dataset check (replace the existing `if config.data.dataset not in SUPPORTED_DATASETS:` block):

```python
    if config.data.dataset not in SUPPORTED_DATASETS and config.data.dataset not in IOT_SUPPORTED_DATASETS:
        errors.append(
            f"data.dataset must be one of {SUPPORTED_DATASETS + IOT_SUPPORTED_DATASETS}, "
            f"got {config.data.dataset!r}"
        )
```

Update `load_config` to parse `max_rows` (in the `DataConfig(...)` construction inside `load_config`, add one line):

```python
        data=DataConfig(
            dataset=data_raw["dataset"],
            data_dir=data_raw["data_dir"],
            num_clients=data_raw["num_clients"],
            partition_type=partition_raw["type"],
            dirichlet_alpha=partition_raw.get("alpha", 0.5),
            batch_size=data_raw["batch_size"],
            max_rows=data_raw.get("max_rows"),
        ),
```

Add `load_experiment_data`, placed right after `build_clients`:

```python
def load_experiment_data(
    config: SimulationConfig, device: torch.device
) -> Tuple[torch.nn.Module, Dict[int, Client], Dataset, Optional[Dataset]]:
    """Resolve the global model, per-client training data, the global test
    set, and (where available) a real-attack-traffic pool for `config`.

    Branches on whether `config.data.dataset` is one of `IOT_SUPPORTED_DATASETS`
    (loads once via `load_iot_experiment`, which returns pre-partitioned or
    natural per-client datasets already split from one deterministic pass
    over the data) or a vision dataset (existing Phase 1 path, unchanged).
    Both `simulation/run.py::run_simulation` and
    `evaluation/experiment_runner.py::ExperimentRunner.run` call this
    instead of duplicating the branch.
    """
    if config.data.dataset in IOT_SUPPORTED_DATASETS:
        experiment_data = load_iot_experiment(
            name=config.data.dataset,
            data_dir=config.data.data_dir,
            max_rows=config.data.max_rows,
            num_clients=config.data.num_clients,
            partition_type=config.data.partition_type,
            dirichlet_alpha=config.data.dirichlet_alpha,
            seed=config.seed,
        )
        global_model = build_model(config.model.name, experiment_data.input_dim, experiment_data.num_classes)
        clients = {
            client_id: Client(
                client_id=client_id,
                model_template=global_model,
                dataset=dataset,
                device=device,
                batch_size=config.data.batch_size,
                learning_rate=config.federated.learning_rate,
                momentum=config.federated.momentum,
                local_epochs=config.federated.local_epochs,
            )
            for client_id, dataset in experiment_data.train_client_datasets.items()
        }
        return global_model, clients, experiment_data.test_dataset, experiment_data.attack_pool

    in_channels, num_classes = dataset_shape(config.data.dataset)
    test_dataset = load_dataset(config.data.dataset, config.data.data_dir, train=False)
    global_model = build_model(config.model.name, in_channels, num_classes)
    clients = build_clients(config, device, global_model)
    return global_model, clients, test_dataset, None
```

In `run_simulation`, replace these two existing lines:

```python
    in_channels, num_classes = dataset_shape(config.data.dataset)
    test_dataset = load_dataset(config.data.dataset, config.data.data_dir, train=False)

    global_model = build_model(config.model.name, in_channels, num_classes)
    clients = build_clients(config, device, global_model)
```

with:

```python
    global_model, clients, test_dataset, attack_pool = load_experiment_data(config, device)
```

Then find the existing `assign_attacks(...)` call inside `run_simulation` and add the new keyword argument:

```python
        clients = assign_attacks(
            clients,
            malicious_client_fraction=config.attacks.malicious_client_fraction,
            active_attack=config.attacks.active_attack,
            attack_settings=_attack_settings(config),
            seed=config.seed,
            attack_dataset=attack_pool,
        )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_config_validation.py tests/test_simulation_iot_integration.py -v`
Expected: These specific new tests will still fail at this step on `attack_dataset` (an unexpected keyword to `assign_attacks`, since Task 15 hasn't added it yet) and on `"real_traffic"` not existing. That's expected — re-run with `-k "not attack_dataset"` is not needed; instead confirm only the dataset-shape/max_rows/dispatch-related assertions pass by running:

Run: `pytest tests/test_config_validation.py::test_iot_dataset_name_passes_validation tests/test_config_validation.py::test_max_rows_defaults_to_none tests/test_simulation_iot_integration.py::test_vision_dataset_returns_none_attack_pool -v`
Expected: PASS (3 tests)

Run: `pytest tests/test_simulation_iot_integration.py::test_iot_dataset_uses_load_iot_experiment -v`
Expected: FAIL with `TypeError: assign_attacks() got an unexpected keyword argument 'attack_dataset'` — this is the expected, temporary state; Task 15 resolves it. Confirm the failure is exactly this `TypeError` (not an import or logic error) before moving on.

- [ ] **Step 5: Commit**

```bash
git add simulation/run.py tests/test_config_validation.py tests/test_simulation_iot_integration.py
git commit -m "feat: wire IoT datasets into simulation/run.py via load_experiment_data"
```

---

### Task 13: Integrate into `evaluation/experiment_runner.py`

**Files:**
- Modify: `evaluation/experiment_runner.py`
- Modify: `tests/test_simulation_iot_integration.py` (add an experiment_runner case)

**Interfaces:**
- Consumes: `load_experiment_data` (Task 12).

- [ ] **Step 1: Write the failing test**

Append to `tests/test_simulation_iot_integration.py`:

```python
from evaluation.experiment_runner import ExperimentRunner


def test_experiment_runner_uses_load_experiment_data():
    config = _iot_config("ciciot2023")
    config.trust.enabled = False
    with patch("evaluation.experiment_runner.load_experiment_data") as mock_load:
        mock_load.return_value = (
            torch.nn.Linear(5, 3),
            {0: None, 1: None},  # placeholder clients; run() only needs the dict for length/logging here
            _fake_experiment_data().test_dataset,
            None,
        )
        runner = ExperimentRunner(config)
        # Only assert load_experiment_data is the wiring point, not a full run
        # (a full run needs real Client objects — covered by Task 17's e2e test).
        assert runner.config.data.dataset == "ciciot2023"
        mock_load.assert_not_called()  # not called yet, just constructing the runner
```

(This is a light wiring check — Task 17's end-to-end test is what actually exercises a full `ExperimentRunner.run()` on IoT data.)

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_simulation_iot_integration.py::test_experiment_runner_uses_load_experiment_data -v`
Expected: FAIL — passes construction but `mock_load` patch target `evaluation.experiment_runner.load_experiment_data` doesn't exist yet (`AttributeError`/`ModuleNotFoundError`-style patch failure).

- [ ] **Step 3: Modify `evaluation/experiment_runner.py`**

Delete this line entirely (no longer used in this file):

```python
from data.dataset import dataset_shape, load_dataset
```

Delete this line entirely too (no longer used):

```python
from models.cnn import build_model
```

Change the existing `from simulation.run import SimulationConfig, build_clients, resolve_device` line to:

```python
from simulation.run import SimulationConfig, load_experiment_data, resolve_device
```

Replace these existing lines inside `ExperimentRunner.run`:

```python
        in_channels, num_classes = dataset_shape(config.data.dataset)
        test_dataset = load_dataset(config.data.dataset, config.data.data_dir, train=False)

        global_model = build_model(config.model.name, in_channels, num_classes)
        clients: Dict[int, Union[Client, MaliciousClient]] = build_clients(config, device, global_model)
```

with:

```python
        global_model, built_clients, test_dataset, attack_pool = load_experiment_data(config, device)
        clients: Dict[int, Union[Client, MaliciousClient]] = built_clients
```

Then find the existing `assign_attacks(...)` call in this file and add the keyword argument, matching Task 12:

```python
            clients = assign_attacks(
                clients,
                malicious_client_fraction=config.attacks.malicious_client_fraction,
                active_attack=config.attacks.active_attack,
                attack_settings=_attack_settings(config),
                seed=config.seed,
                attack_dataset=attack_pool,
            )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_simulation_iot_integration.py -v`
Expected: `test_experiment_runner_uses_load_experiment_data` PASSes. `test_iot_dataset_uses_load_iot_experiment` still fails on the same expected `attack_dataset` `TypeError` as Task 12 (resolved in Task 15).

Run the full existing suite to confirm zero regressions on the vision path:

Run: `pytest tests/ -v -k "not iot and not tabular_mlp and not real_traffic"`
Expected: PASS (all pre-existing tests still pass — `experiment_runner.py`'s vision behavior is unchanged, only its internal wiring moved into `load_experiment_data`)

- [ ] **Step 5: Commit**

```bash
git add evaluation/experiment_runner.py tests/test_simulation_iot_integration.py
git commit -m "refactor: route evaluation/experiment_runner.py through load_experiment_data"
```

---

### Task 14: `RealAttackTrafficAttack`

**Files:**
- Create: `attacks/real_traffic.py`
- Test: `tests/test_real_traffic_attack.py`

**Interfaces:**
- Consumes: `BaseAttack` (existing, unchanged).
- Produces: `RealAttackTrafficAttack(attack_pool: Dataset, seed: int = 42)` with `.apply_to_dataset(dataset) -> Dataset`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_real_traffic_attack.py
"""Tests for RealAttackTrafficAttack: dataset-level, trains on real attack traffic."""

import pytest
from torch.utils.data import TensorDataset
import torch

from attacks.real_traffic import RealAttackTrafficAttack


def _pool(size=20):
    features = torch.randn(size, 4)
    labels = torch.ones(size, dtype=torch.int64)  # all "attack"-labeled
    return TensorDataset(features, labels)


def _client_dataset(size=8):
    features = torch.zeros(size, 4)
    labels = torch.zeros(size, dtype=torch.int64)
    return TensorDataset(features, labels)


def test_empty_pool_raises_value_error():
    empty = TensorDataset(torch.empty(0, 4), torch.empty(0, dtype=torch.int64))
    with pytest.raises(ValueError):
        RealAttackTrafficAttack(attack_pool=empty)


def test_apply_to_dataset_returns_pool_sized_sample():
    attack = RealAttackTrafficAttack(attack_pool=_pool(20), seed=1)
    result = attack.apply_to_dataset(_client_dataset(8))
    assert len(result) == 8


def test_apply_to_dataset_never_exceeds_pool_size():
    attack = RealAttackTrafficAttack(attack_pool=_pool(5), seed=1)
    result = attack.apply_to_dataset(_client_dataset(8))
    assert len(result) == 5


def test_apply_to_dataset_draws_from_pool_not_original():
    attack = RealAttackTrafficAttack(attack_pool=_pool(20), seed=1)
    result = attack.apply_to_dataset(_client_dataset(8))
    _, label = result[0]
    assert label.item() == 1  # pool's label, not the client's original 0


def test_apply_to_update_is_default_noop():
    attack = RealAttackTrafficAttack(attack_pool=_pool(20), seed=1)
    state = {"w": torch.tensor([1.0, 2.0])}
    reference = {"w": torch.tensor([0.0, 0.0])}
    assert attack.apply_to_update(state, reference) is state
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_real_traffic_attack.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'attacks.real_traffic'`

- [ ] **Step 3: Implement `attacks/real_traffic.py`**

```python
"""Real-attack-traffic attack: a fifth Phase-4-style adversarial strategy,
added alongside label flipping / model poisoning / Byzantine / Sybil.

Unlike those four (all synthetic transformations of benign data or honest
updates), this models a compromised IoT device whose local dataset is
itself real malicious traffic — e.g. genuine Mirai/Gafgyt-infected N-BaIoT
captures, or any IoT dataset's own attack-labeled rows — rather than a
label-flip/noise/scale perturbation.

Ground-truth isolation is preserved exactly like `label_flipping`: this
attack only ever reads the dataset's own labels (real content, not
simulation ground truth), never `client.is_malicious` — the server, trust
engine, and selector still cannot distinguish this client from any other
without the same `evaluation/`-only `isinstance` check every other attack
already relies on.
"""

from __future__ import annotations

import random

from torch.utils.data import Dataset, Subset

from attacks.base_attack import BaseAttack


class RealAttackTrafficAttack(BaseAttack):
    """Replaces a client's local training data with a sample of real
    attack-labeled traffic, sized to match the client's original dataset
    (keeps per-round training time comparable to the other attacks)."""

    name = "real_traffic"

    def __init__(self, attack_pool: Dataset, seed: int = 42) -> None:
        if len(attack_pool) == 0:
            raise ValueError("RealAttackTrafficAttack requires a non-empty attack_pool")
        self.attack_pool = attack_pool
        self._rng = random.Random(seed)

    def apply_to_dataset(self, dataset: Dataset) -> Dataset:
        sample_size = min(len(dataset), len(self.attack_pool))
        indices = self._rng.sample(range(len(self.attack_pool)), sample_size)
        return Subset(self.attack_pool, indices)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_real_traffic_attack.py -v`
Expected: PASS (5 tests)

- [ ] **Step 5: Commit**

```bash
git add attacks/real_traffic.py tests/test_real_traffic_attack.py
git commit -m "feat: add RealAttackTrafficAttack (dataset-level, real attack traffic)"
```

---

### Task 15: Wire `real_traffic` into `build_attack`/`assign_attacks`

**Files:**
- Modify: `attacks/__init__.py`
- Modify: `clients/malicious_client.py`
- Test: `tests/test_attacks.py` (existing file — add cases; check its current content first for import/style conventions before editing)

**Interfaces:**
- Consumes: `RealAttackTrafficAttack` (Task 14).
- Produces: `build_attack(active_attack, attack_settings, seed, attack_dataset=None)`, `assign_attacks(clients, malicious_client_fraction, active_attack, attack_settings, seed, attack_dataset=None)`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_attacks.py` (match the existing file's import style — it already imports from `clients.malicious_client` and `helpers`; if it doesn't already import `pytest`, add that import too):

```python
from unittest.mock import MagicMock

from torch.utils.data import TensorDataset
import torch

from attacks import SUPPORTED_ATTACK_TYPES
from attacks.real_traffic import RealAttackTrafficAttack
from clients.malicious_client import build_attack


def test_real_traffic_in_supported_attack_types():
    assert "real_traffic" in SUPPORTED_ATTACK_TYPES


def test_build_attack_real_traffic_with_dataset():
    pool = TensorDataset(torch.randn(10, 3), torch.ones(10, dtype=torch.int64))
    attack = build_attack("real_traffic", {}, seed=1, attack_dataset=pool)
    assert isinstance(attack, RealAttackTrafficAttack)


def test_build_attack_real_traffic_without_dataset_raises():
    with pytest.raises(ValueError, match="attack_dataset"):
        build_attack("real_traffic", {}, seed=1, attack_dataset=None)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_attacks.py -k real_traffic -v`
Expected: FAIL — `"real_traffic" not in SUPPORTED_ATTACK_TYPES`, then `ValueError: Unknown active_attack: 'real_traffic'` from `build_attack`.

- [ ] **Step 3: Modify `attacks/__init__.py`**

```python
"""Adversarial attack strategies (Phase 4): label flipping, model poisoning, Byzantine, Sybil, real-attack-traffic."""

SUPPORTED_ATTACK_TYPES = ("label_flipping", "model_poisoning", "byzantine", "sybil", "real_traffic")
```

- [ ] **Step 4: Modify `clients/malicious_client.py`**

Add the import:

```python
from attacks.real_traffic import RealAttackTrafficAttack
```

Add `Dataset` to the existing `torch`-related imports if not already present:

```python
from torch.utils.data import Dataset
```

Update `build_attack`'s signature and add the new branch (insert before the final `raise ValueError`):

```python
def build_attack(
    active_attack: str,
    attack_settings: Dict[str, Dict[str, Any]],
    seed: int,
    attack_dataset: Optional[Dataset] = None,
) -> BaseAttack:
    """Construct the configured attack strategy by name.

    `attack_dataset` is only consumed by `"real_traffic"` — the other four
    attacks are synthetic transformations that don't need real data.
    """
    if active_attack == "label_flipping":
        settings = attack_settings.get("label_flipping", {})
        return LabelFlippingAttack(label_map=settings.get("label_map"))
    if active_attack == "model_poisoning":
        settings = attack_settings.get("model_poisoning", {})
        return ModelPoisoningAttack(scale_factor=settings.get("scale_factor", 5.0), seed=seed)
    if active_attack == "byzantine":
        settings = attack_settings.get("byzantine", {})
        return ByzantineAttack(strategy=settings.get("strategy", "random"), seed=seed)
    if active_attack == "sybil":
        return SybilAttack(LabelFlippingAttack())
    if active_attack == "real_traffic":
        if attack_dataset is None:
            raise ValueError(
                "active_attack='real_traffic' requires attack_dataset (a pool of real "
                "attack-labeled samples) — not available for this dataset."
            )
        return RealAttackTrafficAttack(attack_pool=attack_dataset, seed=seed)
    raise ValueError(f"Unknown active_attack: {active_attack!r}")
```

Update `assign_attacks`'s signature and thread `attack_dataset` through its two `build_attack(...)` call sites:

```python
def assign_attacks(
    clients: Dict[int, Client],
    malicious_client_fraction: float,
    active_attack: str,
    attack_settings: Dict[str, Dict[str, Any]],
    seed: int,
    attack_dataset: Optional[Dataset] = None,
) -> Dict[int, Union[Client, MaliciousClient]]:
```

In the `sybil` branch, `build_attack("sybil", attack_settings, seed=seed + client_id)` stays unchanged (Sybil's underlying attack is always label flipping, per its existing design — not affected by `attack_dataset`). In the non-Sybil branch, update the call:

```python
        for client_id in malicious_ids:
            attack = build_attack(active_attack, attack_settings, seed=seed + client_id, attack_dataset=attack_dataset)
            result[client_id] = MaliciousClient(
                clients[client_id], attack, attack_probability=probability, seed=seed + client_id
            )
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `pytest tests/test_attacks.py -v`
Expected: PASS (all existing tests plus 3 new ones)

Run the two integration tests left pending from Tasks 12–13:

Run: `pytest tests/test_simulation_iot_integration.py -v`
Expected: PASS (all tests, including the two that were expected to fail on `attack_dataset` before this task)

- [ ] **Step 6: Commit**

```bash
git add attacks/__init__.py clients/malicious_client.py tests/test_attacks.py
git commit -m "feat: wire real_traffic attack into build_attack/assign_attacks"
```

---

### Task 16: IoT experiment configs

**Files:**
- Create: `configs/iot/ciciot2023.yaml`, `configs/iot/unsw_nb15.yaml`, `configs/iot/ton_iot_network.yaml`, `configs/iot/ton_iot_devices.yaml`, `configs/iot/nbaiot.yaml`, `configs/iot/botiot.yaml`
- Test: `tests/test_iot_configs.py`

**Interfaces:**
- Consumes: `load_config`, `validate_config` (Task 12).

Each config points `data.data_dir` directly at the user's `ZT-Fedtrust Datasets` folder (the loaders in `data/iot_datasets.py` read the known archive filenames from there and cache extraction artifacts under `data_cache/iot/<name>/`, matching the RAR-extraction cache path hardcoded in Task 10). `max_rows: 5000` gives fast demo/dev iteration; the plan's own testing never runs these against real data (no such file exists in CI), but running them locally requires setting `data_dir` to wherever the user's `ZT-Fedtrust Datasets` folder actually is. For final paper-quality numbers, set `max_rows: null`.

`num_clients` for `ton_iot_devices` and `nbaiot` is set to their real device counts (7 and 9) — natural-client loaders ignore this value for partitioning, but `validate_config`'s `1 <= clients_per_round <= data.num_clients` check still uses it as the upper bound, so it must match reality.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_iot_configs.py
"""Smoke tests: every IoT config parses and validates (no data loading, no training)."""

import pytest

from simulation.run import load_config, validate_config

_IOT_CONFIGS = [
    "configs/iot/ciciot2023.yaml",
    "configs/iot/unsw_nb15.yaml",
    "configs/iot/ton_iot_network.yaml",
    "configs/iot/ton_iot_devices.yaml",
    "configs/iot/nbaiot.yaml",
]


@pytest.mark.parametrize("config_path", _IOT_CONFIGS)
def test_config_loads_and_validates(config_path):
    config = load_config(config_path)
    validate_config(config)  # should not raise


def test_botiot_config_dataset_name_is_botiot():
    config = load_config("configs/iot/botiot.yaml")
    assert config.data.dataset == "botiot"


def test_botiot_config_fails_validation():
    """botiot is a disabled placeholder — validate_config rejects it up front
    (it's not in IOT_SUPPORTED_DATASETS), rather than failing deep inside a run."""
    config = load_config("configs/iot/botiot.yaml")
    with pytest.raises(ValueError, match="data.dataset"):
        validate_config(config)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_iot_configs.py -v`
Expected: FAIL with `FileNotFoundError` (configs don't exist yet)

- [ ] **Step 3: Create the config files**

`configs/iot/ciciot2023.yaml`:

```yaml
# CICIoT2023 multi-class DDoS-subtype classification. This source slice
# contains only DDoS-family attack traffic (no benign rows), so
# active_attack: real_traffic is not usable here (see data/iot_datasets.py's
# load_ciciot2023 docstring) — use label_flipping/model_poisoning/byzantine/sybil.
#
# Usage:
#   python simulation/run.py --config configs/iot/ciciot2023.yaml --experiment compare_all

seed: 42
device: "cpu"

data:
  dataset: "ciciot2023"
  data_dir: "./ZT-Fedtrust Datasets"   # point this at the folder containing the real archives
  num_clients: 10
  batch_size: 32
  max_rows: 5000   # set to null for full-dataset, paper-quality runs
  partition:
    type: "dirichlet"
    alpha: 0.5

model:
  name: "mlp"

federated:
  num_rounds: 5
  clients_per_round: 6
  local_epochs: 1
  learning_rate: 0.01
  momentum: 0.9

logging:
  log_dir: "./results/iot_ciciot2023/logs"
  log_level: "INFO"
  results_dir: "./results/iot_ciciot2023"

trust:
  enabled: true
  initial_reputation: 0.80
  reputation_alpha: 0.80
  threshold_mode: adaptive
  threshold: 0.60
  participation_fraction: 0.70
  weights:
    identity: 0.25
    integrity: 0.20
    behaviour: 0.20
    reputation: 0.15
    update_quality: 0.20

aggregation:
  strategy: trust_fedavg

attacks:
  enabled: true
  malicious_client_fraction: 0.20
  active_attack: byzantine
  label_flipping:
    enabled: false
    probability: 1.0
  model_poisoning:
    enabled: false
    probability: 1.0
    scale_factor: 5.0
  byzantine:
    enabled: true
    probability: 1.0
    strategy: negate
  sybil:
    enabled: false
    number_of_identities: 3
    probability: 1.0
```

`configs/iot/unsw_nb15.yaml`:

```yaml
# UNSW-NB15 multi-class attack_cat classification (10 categories incl.
# Normal). Supports active_attack: real_traffic (real Normal/attack split).
#
# Usage:
#   python simulation/run.py --config configs/iot/unsw_nb15.yaml --experiment compare_all

seed: 42
device: "cpu"

data:
  dataset: "unsw_nb15"
  data_dir: "./ZT-Fedtrust Datasets"
  num_clients: 10
  batch_size: 32
  max_rows: 5000
  partition:
    type: "dirichlet"
    alpha: 0.5

model:
  name: "mlp"

federated:
  num_rounds: 5
  clients_per_round: 6
  local_epochs: 1
  learning_rate: 0.01
  momentum: 0.9

logging:
  log_dir: "./results/iot_unsw_nb15/logs"
  log_level: "INFO"
  results_dir: "./results/iot_unsw_nb15"

trust:
  enabled: true
  initial_reputation: 0.80
  reputation_alpha: 0.80
  threshold_mode: adaptive
  threshold: 0.60
  participation_fraction: 0.70
  weights:
    identity: 0.25
    integrity: 0.20
    behaviour: 0.20
    reputation: 0.15
    update_quality: 0.20

aggregation:
  strategy: trust_fedavg

attacks:
  enabled: true
  malicious_client_fraction: 0.20
  active_attack: real_traffic
  label_flipping:
    enabled: false
    probability: 1.0
  model_poisoning:
    enabled: false
    probability: 1.0
    scale_factor: 5.0
  byzantine:
    enabled: false
    probability: 1.0
    strategy: negate
  sybil:
    enabled: false
    number_of_identities: 3
    probability: 1.0
```

`configs/iot/ton_iot_network.yaml`:

```yaml
# TON_IoT network-flow multi-class `type` classification. Supports
# active_attack: real_traffic (real normal/attack split).
#
# Usage:
#   python simulation/run.py --config configs/iot/ton_iot_network.yaml --experiment compare_all

seed: 42
device: "cpu"

data:
  dataset: "ton_iot_network"
  data_dir: "./ZT-Fedtrust Datasets"
  num_clients: 10
  batch_size: 32
  max_rows: 5000
  partition:
    type: "dirichlet"
    alpha: 0.5

model:
  name: "mlp"

federated:
  num_rounds: 5
  clients_per_round: 6
  local_epochs: 1
  learning_rate: 0.01
  momentum: 0.9

logging:
  log_dir: "./results/iot_ton_iot_network/logs"
  log_level: "INFO"
  results_dir: "./results/iot_ton_iot_network"

trust:
  enabled: true
  initial_reputation: 0.80
  reputation_alpha: 0.80
  threshold_mode: adaptive
  threshold: 0.60
  participation_fraction: 0.70
  weights:
    identity: 0.25
    integrity: 0.20
    behaviour: 0.20
    reputation: 0.15
    update_quality: 0.20

aggregation:
  strategy: trust_fedavg

attacks:
  enabled: true
  malicious_client_fraction: 0.20
  active_attack: real_traffic
  label_flipping:
    enabled: false
    probability: 1.0
  model_poisoning:
    enabled: false
    probability: 1.0
    scale_factor: 5.0
  byzantine:
    enabled: false
    probability: 1.0
    strategy: negate
  sybil:
    enabled: false
    number_of_identities: 3
    probability: 1.0
```

`configs/iot/ton_iot_devices.yaml` (natural clients — `num_clients: 7` documents the real device count for `clients_per_round`'s validation bound; the loader ignores it for partitioning):

```yaml
# TON_IoT device telemetry: 7 real IoT devices (Fridge, Garage_Door,
# GPS_Tracker, Modbus, Motion_Light, Thermostat, Weather) as 7 natural
# federated clients, sharing a zero-padded union feature schema. Supports
# active_attack: real_traffic.
#
# num_clients: 7 is not a partitioning knob here (natural-client datasets
# ignore it) — it exists only so federated.clients_per_round's upper-bound
# validation matches the real device count.
#
# Usage:
#   python simulation/run.py --config configs/iot/ton_iot_devices.yaml --experiment compare_all

seed: 42
device: "cpu"

data:
  dataset: "ton_iot_devices"
  data_dir: "./ZT-Fedtrust Datasets"
  num_clients: 7
  batch_size: 32
  max_rows: 5000
  partition:
    type: "iid"
    alpha: 0.5

model:
  name: "mlp"

federated:
  num_rounds: 5
  clients_per_round: 5
  local_epochs: 1
  learning_rate: 0.01
  momentum: 0.9

logging:
  log_dir: "./results/iot_ton_iot_devices/logs"
  log_level: "INFO"
  results_dir: "./results/iot_ton_iot_devices"

trust:
  enabled: true
  initial_reputation: 0.80
  reputation_alpha: 0.80
  threshold_mode: adaptive
  threshold: 0.60
  participation_fraction: 0.70
  weights:
    identity: 0.25
    integrity: 0.20
    behaviour: 0.20
    reputation: 0.15
    update_quality: 0.20

aggregation:
  strategy: trust_fedavg

attacks:
  enabled: true
  malicious_client_fraction: 0.20
  active_attack: real_traffic
  label_flipping:
    enabled: false
    probability: 1.0
  model_poisoning:
    enabled: false
    probability: 1.0
    scale_factor: 5.0
  byzantine:
    enabled: false
    probability: 1.0
    strategy: negate
  sybil:
    enabled: false
    number_of_identities: 3
    probability: 1.0
```

`configs/iot/nbaiot.yaml` (natural clients — `num_clients: 9`):

```yaml
# N-BaIoT: 9 real IoT devices as 9 natural federated clients, benign vs.
# Mirai/Gafgyt-subtype multi-class classification. Supports
# active_attack: real_traffic (this is the dataset the design spec
# specifically calls out for it — malicious clients train on genuine
# Mirai/Gafgyt-infected traffic).
#
# Requires 7-Zip or unrar installed on the machine the first time this
# runs (extracts the .rar attack archives; cached afterward). See
# data/rar_extract.py.
#
# Usage:
#   python simulation/run.py --config configs/iot/nbaiot.yaml --experiment compare_all

seed: 42
device: "cpu"

data:
  dataset: "nbaiot"
  data_dir: "./ZT-Fedtrust Datasets"
  num_clients: 9
  batch_size: 32
  max_rows: 5000
  partition:
    type: "iid"
    alpha: 0.5

model:
  name: "mlp"

federated:
  num_rounds: 5
  clients_per_round: 7
  local_epochs: 1
  learning_rate: 0.01
  momentum: 0.9

logging:
  log_dir: "./results/iot_nbaiot/logs"
  log_level: "INFO"
  results_dir: "./results/iot_nbaiot"

trust:
  enabled: true
  initial_reputation: 0.80
  reputation_alpha: 0.80
  threshold_mode: adaptive
  threshold: 0.60
  participation_fraction: 0.70
  weights:
    identity: 0.25
    integrity: 0.20
    behaviour: 0.20
    reputation: 0.15
    update_quality: 0.20

aggregation:
  strategy: trust_fedavg

attacks:
  enabled: true
  malicious_client_fraction: 0.20
  active_attack: real_traffic
  label_flipping:
    enabled: false
    probability: 1.0
  model_poisoning:
    enabled: false
    probability: 1.0
    scale_factor: 5.0
  byzantine:
    enabled: false
    probability: 1.0
    strategy: negate
  sybil:
    enabled: false
    number_of_identities: 3
    probability: 1.0
```

`configs/iot/botiot.yaml` (disabled placeholder — loads and parses fine, `dataset: "botiot"`, but fails `validate_config` on purpose since `"botiot"` isn't in `IOT_SUPPORTED_DATASETS` yet):

```yaml
# DISABLED PLACEHOLDER. The file originally supplied as BoT-IoT data
# turned out to be a duplicate of UNSW-NB15 (confirmed: matching column
# schema, not real BoT-IoT's pkSeqID/stime/flgs/... schema). This config
# exists so the experiment slot is visible and ready, but validate_config
# deliberately rejects "botiot" (see data/iot_datasets.py's
# IOT_SUPPORTED_DATASETS) until the real BoT-IoT CSVs are supplied and a
# matching loader is added following the same pattern as the other five.
#
# This config is NOT runnable yet.

seed: 42
device: "cpu"

data:
  dataset: "botiot"
  data_dir: "./ZT-Fedtrust Datasets"
  num_clients: 10
  batch_size: 32
  max_rows: 5000
  partition:
    type: "dirichlet"
    alpha: 0.5

model:
  name: "mlp"

federated:
  num_rounds: 5
  clients_per_round: 6
  local_epochs: 1
  learning_rate: 0.01
  momentum: 0.9

logging:
  log_dir: "./results/iot_botiot/logs"
  log_level: "INFO"
  results_dir: "./results/iot_botiot"

trust:
  enabled: true
  initial_reputation: 0.80
  reputation_alpha: 0.80
  threshold_mode: adaptive
  threshold: 0.60
  participation_fraction: 0.70
  weights:
    identity: 0.25
    integrity: 0.20
    behaviour: 0.20
    reputation: 0.15
    update_quality: 0.20

aggregation:
  strategy: trust_fedavg

attacks:
  enabled: false
  malicious_client_fraction: 0.20
  active_attack: label_flipping
  label_flipping:
    enabled: false
    probability: 1.0
  model_poisoning:
    enabled: false
    probability: 1.0
    scale_factor: 5.0
  byzantine:
    enabled: false
    probability: 1.0
    strategy: negate
  sybil:
    enabled: false
    number_of_identities: 3
    probability: 1.0
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_iot_configs.py -v`
Expected: PASS (7 tests)

- [ ] **Step 5: Commit**

```bash
git add configs/iot/ tests/test_iot_configs.py
git commit -m "feat: add 6 IoT experiment configs (5 runnable, botiot disabled placeholder)"
```

---

### Task 17: End-to-end integration test

**Files:**
- Test: `tests/test_iot_integration.py`

**Interfaces:**
- Consumes: everything from Tasks 1–15.

This is the plan's final check: a full `run_simulation()` call on a tiny synthetic IoT-shaped dataset (monkeypatching `load_iot_experiment` so no real data files or 7-Zip installation are needed), confirming the entire wiring — trust engine, trust-based selection, TrustFedAvg, and the `real_traffic` attack — works end to end on the new tabular path, exactly as it already does for MNIST.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_iot_integration.py
"""End-to-end test: a full federated run on synthetic IoT-shaped tabular data."""

from unittest.mock import patch

import numpy as np
import pytest

from data.iot_datasets import IoTExperimentData
from data.partition import partition_iid, make_client_subsets
from data.tabular_dataset import TabularDataset
from simulation.run import DataConfig, FederatedConfig, LoggingConfig, ModelConfig, SimulationConfig, run_simulation


def _synthetic_iot_experiment_data(seed: int) -> IoTExperimentData:
    rng = np.random.default_rng(seed)
    num_samples = 200
    num_features = 6
    labels = rng.integers(0, 3, size=num_samples)
    offsets = labels.astype(np.float32).reshape(-1, 1) * 0.5
    features = (offsets + rng.normal(0, 0.3, size=(num_samples, num_features))).astype(np.float32)

    full_dataset = TabularDataset(features, labels)
    client_indices = partition_iid(full_dataset, num_clients=6, seed=seed)
    train_client_datasets = make_client_subsets(full_dataset, client_indices)

    test_labels = rng.integers(0, 3, size=40)
    test_offsets = test_labels.astype(np.float32).reshape(-1, 1) * 0.5
    test_features = (test_offsets + rng.normal(0, 0.3, size=(40, num_features))).astype(np.float32)
    test_dataset = TabularDataset(test_features, test_labels)

    attack_labels = np.ones(30, dtype=np.int64)
    attack_features = (0.5 + rng.normal(0, 0.3, size=(30, num_features))).astype(np.float32)
    attack_pool = TabularDataset(attack_features, attack_labels)

    return IoTExperimentData(
        train_client_datasets=train_client_datasets,
        test_dataset=test_dataset,
        attack_pool=attack_pool,
        input_dim=num_features,
        num_classes=3,
        class_names=["normal", "attack_a", "attack_b"],
    )


def _iot_config() -> SimulationConfig:
    return SimulationConfig(
        seed=7,
        device="cpu",
        data=DataConfig(
            dataset="ciciot2023", data_dir="./unused", num_clients=6, partition_type="iid",
            dirichlet_alpha=0.5, batch_size=16, max_rows=None,
        ),
        model=ModelConfig(name="mlp"),
        federated=FederatedConfig(
            num_rounds=2, clients_per_round=4, local_epochs=1, learning_rate=0.05, momentum=0.9
        ),
        logging=LoggingConfig(log_dir="./results/test_iot/logs", log_level="INFO", results_dir="./results/test_iot"),
    )


def test_full_simulation_runs_on_synthetic_iot_data(tmp_path):
    config = _iot_config()
    config.logging.log_dir = str(tmp_path / "logs")
    config.logging.results_dir = str(tmp_path / "results")
    config.attacks.enabled = True
    config.attacks.active_attack = "real_traffic"
    config.attacks.malicious_client_fraction = 0.34  # 2 of 6 clients

    with patch("simulation.run.load_iot_experiment", return_value=_synthetic_iot_experiment_data(config.seed)):
        history = run_simulation(config)

    assert len(history) == config.federated.num_rounds
    for round_record in history:
        assert "accuracy" in round_record
        assert 0.0 <= round_record["accuracy"] <= 1.0


def test_full_simulation_runs_without_attacks(tmp_path):
    config = _iot_config()
    config.logging.log_dir = str(tmp_path / "logs")
    config.logging.results_dir = str(tmp_path / "results")

    with patch("simulation.run.load_iot_experiment", return_value=_synthetic_iot_experiment_data(config.seed)):
        history = run_simulation(config)

    assert len(history) == config.federated.num_rounds
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_iot_integration.py -v`
Expected: FAIL — check the actual failure carefully. If `run_simulation`'s return shape or `history` record keys differ from `"accuracy"`, adjust the assertion to match `Server`'s actual per-round record (inspect `server/server.py::evaluate`/`run_simulation`'s history-building code if this fails on a `KeyError` rather than an import error).

- [ ] **Step 3: No new implementation needed** — this task exercises Tasks 1–15's existing code. If Step 2 reveals a real bug (not just a wrong assertion), fix it in the relevant task's file and note the fix in the commit message.

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_iot_integration.py -v`
Expected: PASS (2 tests)

Then run the complete test suite to confirm zero regressions anywhere:

Run: `pytest tests/ -v`
Expected: PASS (every existing test plus every new one from this plan)

- [ ] **Step 5: Commit**

```bash
git add tests/test_iot_integration.py
git commit -m "test: add end-to-end IoT federated simulation integration test"
```

---

## Post-plan manual verification (not automated — requires the user's real data)

Once all 17 tasks are merged, these steps require the user's machine and real data, so they're not part of the automated plan:

1. `winget install 7zip.7zip` (if not already done).
2. Point each `configs/iot/*.yaml`'s `data.data_dir` at the actual `ZT-Fedtrust Datasets` folder location.
3. Run one demo-scale experiment per dataset (`max_rows: 5000`, the checked-in default) to confirm real data loads correctly: `python simulation/run.py --config configs/iot/unsw_nb15.yaml --experiment compare_all`.
4. For paper-quality numbers, set `max_rows: null` in the relevant config and re-run — full-dataset runs will take substantially longer, especially for `nbaiot` (first run also pays the one-time RAR extraction cost).

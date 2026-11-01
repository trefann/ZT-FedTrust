# IoT Model Accuracy Tuning Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Get all three experiment scenarios (`baseline`, `fedavg_attack`, `ztfedtrust_attack`) on all four real IoT datasets (`nbaiot`, `ton_iot_devices`, `unsw_nb15`, `ton_iot_network`) to 85%+ test accuracy, via a larger training budget, class-weighted loss, and configurable model capacity — reported honestly if any scenario/dataset falls short after the bounded tuning budget.

**Architecture:** Three orthogonal, additive levers, all optional/defaulted so nothing outside the four IoT configs is affected: (1) `TabularMLP`'s hidden-layer sizes become a config knob instead of hardcoded; (2) `Client` accepts an optional per-class loss weight vector; (3) `data/iot_datasets.py` computes that weight vector once per dataset (inverse-frequency, from the pooled training-label distribution, before client partitioning) and `simulation/run.py` wires both new knobs through to model/client construction. The four IoT configs get a first boosted training budget, then a bounded, per-dataset empirical tuning loop (tune against `baseline`, verify the other two scenarios once) pushes each dataset toward the 85% target, reporting honestly wherever it falls short.

**Tech Stack:** Python, PyTorch, pandas/numpy (already dependencies), pytest.

## Global Constraints

- No new pip dependencies.
- `Client.__init__`/`train()`'s new `class_weights` parameter must default to `None` and preserve today's unweighted-loss behavior exactly when `None` — existing callers/tests that don't pass it must be unaffected.
- `TabularMLP.__init__`'s new `hidden_sizes` parameter must default to `(128, 64)` — today's fixed architecture — so existing callers/tests that don't pass it produce an identically-shaped model.
- `build_model()`'s new `hidden_sizes` parameter must default to `None` and only be forwarded to the underlying model constructor when not `None` (so `SimpleCNN`, which has no such parameter, is never passed one).
- The new `training.class_weighted_loss` config key must default to `False` so the two non-IoT configs (MNIST/CIFAR-10) are unaffected without any change to their own config files.
- `IoTExperimentData`'s new `class_weights` field is computed once per dataset from the pooled training-label distribution *before* client partitioning — every client in a federation shares the same class-weight vector, never a per-client one.
- The full existing test suite (264 tests as of this plan's baseline — confirm the exact number is current with `pytest tests/ -q` before starting, per this project's established practice of not assuming a stale baseline number) must stay green throughout every task.
- The bounded per-dataset tuning loop (Tasks 5-8) is capped at 4 attempts per dataset and must report the actual final numbers honestly — no adjusting results to force a favorable outcome, no silently extending the attempt budget past 4, matching this project's established practice (see `docs/superpowers/plans/2026-08-17-peer-cluster-trust-validation-report.md`).

---

## File Structure

**Modified files:**
- `models/tabular_mlp.py` — `TabularMLP.__init__` takes `hidden_sizes: Tuple[int, ...] = (128, 64)` instead of two hardcoded layers.
- `models/cnn.py` — `build_model()` gains an optional `hidden_sizes` parameter.
- `clients/client.py` — `Client.__init__`/`train()` accept an optional `class_weights: Optional[torch.Tensor]`.
- `data/iot_preprocessing.py` — new `compute_class_weights(labels, num_classes) -> torch.Tensor` helper.
- `data/iot_datasets.py` — `IoTExperimentData` gains a `class_weights: torch.Tensor` field, computed by both builder functions.
- `simulation/run.py` — `ModelConfig` gains `hidden_sizes`; new `TrainingConfig` dataclass; `SimulationConfig` gains `training`; `load_config`/`validate_config` updated; `load_experiment_data`'s IoT branch wires both new knobs through.
- `configs/iot/nbaiot.yaml`, `configs/iot/ton_iot_devices.yaml`, `configs/iot/unsw_nb15.yaml`, `configs/iot/ton_iot_network.yaml` — boosted training budget, `model.hidden_sizes`, `training.class_weighted_loss: true`; then per-dataset empirical tuning adjustments in Tasks 5-8.

**Modified/new test files:**
- `tests/test_tabular_mlp.py` — append `hidden_sizes` tests.
- `tests/test_iot_preprocessing.py` — append `compute_class_weights` tests.
- `tests/test_client.py` — append a `class_weights` test.
- `tests/test_iot_datasets.py` — append `class_weights` field tests for both builder paths.
- `tests/test_config_validation.py` — append `hidden_sizes`/`training.class_weighted_loss` tests.

**New report file:**
- `docs/superpowers/plans/2026-08-21-iot-accuracy-tuning-validation-report.md` — one shared report; Task 5 creates it, Tasks 6-8 each append their own dataset's section.

---

### Task 1: Configurable model capacity

**Files:**
- Modify: `models/tabular_mlp.py`
- Modify: `models/cnn.py`
- Test: `tests/test_tabular_mlp.py`

**Interfaces:**
- Consumes: nothing new.
- Produces: `TabularMLP(in_channels, num_classes, hidden_sizes=(128, 64))` and `build_model(name, in_channels, num_classes, hidden_sizes=None)`. Task 3 depends on `build_model`'s exact new signature.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_tabular_mlp.py`:

```python
def test_default_hidden_sizes_match_original_architecture():
    model = build_model("mlp", in_channels=20, num_classes=5)
    assert [layer[0].out_features for layer in model.layers] == [128, 64]
    batch = torch.randn(4, 20)
    output = model(batch)
    assert output.shape == (4, 5)


def test_custom_hidden_sizes_change_layer_widths():
    model = build_model("mlp", in_channels=20, num_classes=5, hidden_sizes=(256, 128, 32))
    assert [layer[0].out_features for layer in model.layers] == [256, 128, 32]
    batch = torch.randn(4, 20)
    output = model(batch)
    assert output.shape == (4, 5)


def test_single_hidden_layer_is_valid():
    model = build_model("mlp", in_channels=10, num_classes=3, hidden_sizes=(64,))
    batch = torch.randn(4, 10)
    output = model(batch)
    assert output.shape == (4, 3)


def test_empty_hidden_sizes_raises():
    with pytest.raises(ValueError):
        build_model("mlp", in_channels=10, num_classes=3, hidden_sizes=())
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_tabular_mlp.py -v -k "hidden_sizes or single_hidden or empty_hidden"`
Expected: FAIL — `build_model() got an unexpected keyword argument 'hidden_sizes'`, and `model.layers` doesn't exist yet (`TabularMLP` currently uses `fc1`/`fc2`/`fc3`, not a `layers` list).

- [ ] **Step 3: Rewrite `models/tabular_mlp.py`**

Replace the entire file with:

```python
"""Feedforward MLP for tabular IoT attack-detection features."""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class TabularMLP(nn.Module):
    """A small MLP for tabular classification, with configurable hidden-layer sizes.

    Uses the same `in_channels`/`num_classes` constructor-keyword names as
    `models/cnn.py`'s `SimpleCNN` (here `in_channels` means "input feature
    count") so both models share one `build_model(name, in_channels,
    num_classes)` calling convention — `simulation/run.py` and
    `evaluation/experiment_runner.py` need no changes to support this model.

    `hidden_sizes` defaults to `(128, 64)`, the original fixed two-layer
    architecture, so existing callers that don't pass it are unaffected.
    Each entry in `self.layers` is an `nn.ModuleList([Linear, BatchNorm1d])`
    pair; `forward` applies ReLU + dropout after each in sequence.
    """

    def __init__(
        self,
        in_channels: int = 32,
        num_classes: int = 10,
        hidden_sizes: Tuple[int, ...] = (128, 64),
    ) -> None:
        super().__init__()
        if not hidden_sizes:
            raise ValueError("hidden_sizes must contain at least one layer size")

        self.layers = nn.ModuleList()
        previous_size = in_channels
        for size in hidden_sizes:
            self.layers.append(nn.ModuleList([nn.Linear(previous_size, size), nn.BatchNorm1d(size)]))
            previous_size = size

        self.output = nn.Linear(previous_size, num_classes)
        self.dropout = nn.Dropout(0.25)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for linear, batch_norm in self.layers:
            x = F.relu(batch_norm(linear(x)))
            x = self.dropout(x)
        return self.output(x)
```

- [ ] **Step 4: Modify `models/cnn.py`**

Update the imports at the top (add `Any`, `Dict`, `Optional`, `Tuple`):

```python
from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.tabular_mlp import TabularMLP
```

Replace `build_model`:

```python
def build_model(
    name: str, in_channels: int, num_classes: int, hidden_sizes: Optional[Tuple[int, ...]] = None
) -> nn.Module:
    """Factory for constructing a registered model by name.

    Adding a new architecture only requires implementing an `nn.Module` and
    registering it in `_MODEL_REGISTRY` — no other code needs to change.

    `hidden_sizes` is forwarded only when given, so it's a no-op for models
    (like `SimpleCNN`) that don't accept it.
    """
    if name not in _MODEL_REGISTRY:
        raise ValueError(f"Unknown model name: {name!r}. Available: {list(_MODEL_REGISTRY)}")
    kwargs: Dict[str, Any] = {"in_channels": in_channels, "num_classes": num_classes}
    if hidden_sizes is not None:
        kwargs["hidden_sizes"] = hidden_sizes
    return _MODEL_REGISTRY[name](**kwargs)
```

(Leave `SimpleCNN`, `_MODEL_REGISTRY`, and `available_models` exactly as they are.)

- [ ] **Step 5: Run tests to verify they pass**

Run: `python -m pytest tests/test_tabular_mlp.py tests/test_model.py -v`
Expected: PASS (existing tests + 4 new ones).

- [ ] **Step 6: Run the full suite to confirm no regressions**

Run: `python -m pytest tests/ -q`
Expected: all existing tests still pass, plus the 4 new ones.

- [ ] **Step 7: Commit**

```bash
git add models/tabular_mlp.py models/cnn.py tests/test_tabular_mlp.py
git commit -m "feat: make TabularMLP hidden-layer sizes configurable"
```

---

### Task 2: Class-weighted loss primitives

**Files:**
- Modify: `clients/client.py`
- Modify: `data/iot_preprocessing.py`
- Test: `tests/test_client.py`
- Test: `tests/test_iot_preprocessing.py`

**Interfaces:**
- Consumes: nothing new from Task 1.
- Produces: `Client(..., class_weights: Optional[torch.Tensor] = None)` and `compute_class_weights(labels: Sequence[int], num_classes: int) -> torch.Tensor`. Task 3 depends on both exact signatures.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_iot_preprocessing.py` (add `import torch` to the existing imports at the top of the file, alongside `numpy`/`pandas`, and add `compute_class_weights` to the existing `from data.iot_preprocessing import (...)` block):

```python
def test_compute_class_weights_gives_rarer_classes_higher_weight():
    # class 0 appears 8 times, class 1 appears 2 times, class 2 never appears
    labels = [0] * 8 + [1] * 2
    weights = compute_class_weights(labels, num_classes=3)
    assert weights.shape == (3,)
    assert weights[1] > weights[0]


def test_compute_class_weights_zero_count_class_gets_zero_weight():
    labels = [0, 0, 1, 1]
    weights = compute_class_weights(labels, num_classes=3)
    assert weights[2] == 0.0


def test_compute_class_weights_all_one_class_does_not_crash():
    labels = [4, 4, 4, 4]
    weights = compute_class_weights(labels, num_classes=5)
    assert weights.shape == (5,)
    assert weights[4] > 0.0
    assert torch.equal(weights[[0, 1, 2, 3]], torch.zeros(4))


def test_compute_class_weights_is_a_float_tensor():
    weights = compute_class_weights([0, 1, 0, 1], num_classes=2)
    assert weights.dtype == torch.float32
```

Append to `tests/test_client.py` (uses the existing `_make_tabular_dataset` helper already defined in that file):

```python
def test_class_weights_change_the_loss_for_an_imbalanced_batch():
    torch.manual_seed(0)
    dataset = _make_tabular_dataset(num_samples=64, num_features=10, seed=1)
    model_unweighted = build_model("mlp", in_channels=10, num_classes=10)
    model_weighted = build_model("mlp", in_channels=10, num_classes=10)
    model_weighted.load_state_dict(model_unweighted.state_dict())

    class_weights = torch.ones(10)
    class_weights[0] = 5.0  # heavily up-weight class 0

    client_unweighted = Client(
        client_id=0, model_template=model_unweighted, dataset=dataset, device=torch.device("cpu"),
        batch_size=16, learning_rate=0.01, momentum=0.9, local_epochs=1,
    )
    client_weighted = Client(
        client_id=1, model_template=model_weighted, dataset=dataset, device=torch.device("cpu"),
        batch_size=16, learning_rate=0.01, momentum=0.9, local_epochs=1, class_weights=class_weights,
    )

    torch.manual_seed(1)
    result_unweighted = client_unweighted.train()
    torch.manual_seed(1)
    result_weighted = client_weighted.train()

    assert result_unweighted.average_loss != result_weighted.average_loss
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_iot_preprocessing.py tests/test_client.py -v -k "class_weight"`
Expected: FAIL — `ImportError: cannot import name 'compute_class_weights'`, and `Client(...)` raises `TypeError: __init__() got an unexpected keyword argument 'class_weights'`.

- [ ] **Step 3: Modify `data/iot_preprocessing.py`**

Add `torch` and `Sequence` to the imports at the top:

```python
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from pandas.api.types import is_numeric_dtype
```

Add this function at the end of the file:

```python
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
```

- [ ] **Step 4: Modify `clients/client.py`**

Add `Optional` to the existing `typing` import:

```python
from typing import Any, Dict, List, Optional
```

Update `Client.__init__`'s signature and body — add the new parameter after `local_epochs` and store it:

```python
    def __init__(
        self,
        client_id: int,
        model_template: nn.Module,
        dataset: Dataset,
        device: torch.device,
        batch_size: int = 32,
        learning_rate: float = 0.01,
        momentum: float = 0.9,
        local_epochs: int = 1,
        class_weights: Optional[torch.Tensor] = None,
    ) -> None:
        self.client_id = client_id
        self.dataset = dataset
        self.device = device
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.local_epochs = local_epochs
        self.class_weights = class_weights
        self.model = copy.deepcopy(model_template).to(device)
        self._logger = logging.getLogger(__name__)
```

(The rest of `__init__` — the Zero Trust evaluation state block — is unchanged.)

In `train()`, replace this line:

```python
        criterion = nn.CrossEntropyLoss()
```

with:

```python
        criterion = nn.CrossEntropyLoss(
            weight=self.class_weights.to(self.device) if self.class_weights is not None else None
        )
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `python -m pytest tests/test_iot_preprocessing.py tests/test_client.py -v`
Expected: PASS (existing tests + 8 new ones: 4 in `test_iot_preprocessing.py`, 1 in `test_client.py`, plus this step also re-runs the 3 pre-existing tests in each file — all should pass).

- [ ] **Step 6: Run the full suite to confirm no regressions**

Run: `python -m pytest tests/ -q`
Expected: all existing tests still pass, plus the 5 new ones from this task (on top of Task 1's 4).

- [ ] **Step 7: Commit**

```bash
git add clients/client.py data/iot_preprocessing.py tests/test_client.py tests/test_iot_preprocessing.py
git commit -m "feat: support class-weighted loss in Client training"
```

---

### Task 3: Wire class weights and model capacity through the IoT data pipeline

**Files:**
- Modify: `data/iot_datasets.py`
- Modify: `simulation/run.py`
- Test: `tests/test_iot_datasets.py`
- Test: `tests/test_config_validation.py`

**Interfaces:**
- Consumes: `compute_class_weights(labels, num_classes) -> torch.Tensor` from Task 2's `data/iot_preprocessing.py`; `build_model(name, in_channels, num_classes, hidden_sizes=None)` from Task 1's `models/cnn.py`; `Client(..., class_weights=None)` from Task 2's `clients/client.py`.
- Produces: `IoTExperimentData.class_weights: torch.Tensor`; `SimulationConfig.model.hidden_sizes: Optional[Tuple[int, ...]]`; `SimulationConfig.training.class_weighted_loss: bool`. Tasks 4-8 depend on these three config surfaces existing and being parsed correctly from YAML.

- [ ] **Step 1: Write the failing tests**

First, read `data/iot_datasets.py` and `simulation/run.py` in full (both already read during planning; re-read here for current line numbers, since planning summaries can drift) so your edits land in the right place.

This file already has the two fixtures you need: module-level `_make_partitioned_df(num_rows=200, seed=0)` (labels skewed `normal:0.60, ddos:0.25, backdoor:0.15` — already unbalanced, no new fixture needed) and `TestBuildNaturalClientExperiment._device_frames(self)` (device_a: `normal:0.7, ddos:0.3` over 40 rows; device_b: `normal:0.7, backdoor:0.3` over 30 rows — pooled, `normal` is clearly the most common class). Add one new test method to each existing test class (do not add module-level test functions or new fixture helpers):

In `TestBuildPartitionedExperiment`, add:

```python
    def test_class_weights_favor_rarer_class(self):
        df = _make_partitioned_df()
        result = _build_partitioned_experiment(
            df, "label", "normal", num_clients=3, partition_type="iid",
            dirichlet_alpha=0.5, max_rows=None, seed=42,
        )
        assert result.class_weights.shape == (result.num_classes,)
        backdoor_index = result.class_names.index("backdoor")  # rarest, p=0.15
        normal_index = result.class_names.index("normal")      # most common, p=0.60
        assert result.class_weights[backdoor_index] > result.class_weights[normal_index]
```

In `TestBuildNaturalClientExperiment`, add:

```python
    def test_class_weights_pool_across_devices(self):
        result = _build_natural_client_experiment(
            self._device_frames(), label_column="type", benign_label="normal",
            max_rows_per_device=None, seed=7,
        )
        assert result.class_weights.shape == (result.num_classes,)
        normal_index = result.class_names.index("normal")  # p=0.7 on both devices, most common overall
        ddos_index = result.class_names.index("ddos")      # only appears on device_a, rarer overall
        assert result.class_weights[ddos_index] > result.class_weights[normal_index]
```

Also add `torch` to this file's imports (it currently imports `numpy`, `pandas`, `pytest`, `zipfile`, `Path`, `patch` — add `import torch` alongside them) — not strictly required by the two assertions above (which only compare tensor elements with `>`), but needed if you choose to assert `class_weights.dtype == torch.float32` too, which is encouraged for parity with Task 2's `compute_class_weights` tests.

Append to `tests/test_config_validation.py`:

```python
def test_model_hidden_sizes_defaults_to_none():
    config = _base_config()
    assert config.model.hidden_sizes is None


def test_training_class_weighted_loss_defaults_to_false():
    config = _base_config()
    assert config.training.class_weighted_loss is False


def test_negative_hidden_size_raises():
    config = _base_config()
    config.model.hidden_sizes = (128, -1)
    with pytest.raises(ValueError):
        validate_config(config)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_iot_datasets.py tests/test_config_validation.py -v -k "class_weight or hidden_sizes"`
Expected: FAIL — `AttributeError`/`TypeError` for the not-yet-existing `class_weights` field and `model.hidden_sizes`/`training` config surfaces.

- [ ] **Step 3: Modify `data/iot_datasets.py`**

Add `torch` to the imports and import `compute_class_weights`:

```python
import pandas as pd
import torch
from torch.utils.data import Dataset

from data.iot_preprocessing import (
    FeatureEncoder,
    compute_class_weights,
    encode_labels,
    fit_feature_encoder,
    fit_label_classes,
    split_train_test,
    subsample_dataframe,
)
```

Add the new field to `IoTExperimentData`:

```python
@dataclass
class IoTExperimentData:
    """Everything `simulation/run.py::load_experiment_data` needs for one IoT experiment."""

    train_client_datasets: Dict[int, Dataset]
    test_dataset: Dataset
    attack_pool: Optional[Dataset]
    input_dim: int
    num_classes: int
    class_names: List[str]
    class_weights: torch.Tensor
```

In `_build_partitioned_experiment`, replace this block:

```python
    train_dataset = TabularDataset(
        encoder.transform(train_df[feature_columns]),
        encode_labels(train_df[label_column], class_names),
    )
```

with (capturing the labels so they can be reused for the weight computation below, instead of encoding them twice):

```python
    train_labels = encode_labels(train_df[label_column], class_names)
    train_dataset = TabularDataset(encoder.transform(train_df[feature_columns]), train_labels)
```

Then, right before the `return IoTExperimentData(...)` statement, add:

```python
    class_weights = compute_class_weights(train_labels, num_classes=len(class_names))
```

And add `class_weights=class_weights` to that function's `return IoTExperimentData(...)` call.

In `_build_natural_client_experiment`, right after the existing `pooled_train = pd.concat(train_parts.values(), ignore_index=True)` line, add:

```python
    pooled_train_labels = encode_labels(pooled_train[label_column], class_names)
    class_weights = compute_class_weights(pooled_train_labels, num_classes=len(class_names))
```

And add `class_weights=class_weights` to that function's `return IoTExperimentData(...)` call.

- [ ] **Step 4: Modify `simulation/run.py`**

Update `ModelConfig`:

```python
@dataclass
class ModelConfig:
    name: str
    hidden_sizes: Optional[Tuple[int, ...]] = None
```

Add `Tuple` to the existing `from typing import Any, Dict, List, Optional, Tuple` import at the top of the file (it currently imports `Any, Dict, List, Optional` — add `Tuple`).

Add a new `TrainingConfig` dataclass, placed after `ModelConfig`:

```python
@dataclass
class TrainingConfig:
    """Training-recipe knobs orthogonal to the federated schedule in `FederatedConfig`."""

    class_weighted_loss: bool = False
```

Update `SimulationConfig` to include it:

```python
@dataclass
class SimulationConfig:
    seed: int
    device: str
    data: DataConfig
    model: ModelConfig
    federated: FederatedConfig
    logging: LoggingConfig
    # Defaulted so existing call sites (and configs without a `trust:`/
    # `aggregation:`/`attacks:`/`training:` block) keep working unmodified —
    # all four are additive.
    trust: TrustConfig = field(default_factory=TrustConfig)
    aggregation: AggregationConfig = field(default_factory=AggregationConfig)
    attacks: AttackConfig = field(default_factory=AttackConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
```

In `validate_config`, add this check right after the existing `model.name` check (`if config.model.name not in available: ...`):

```python
    if config.model.hidden_sizes is not None and any(size <= 0 for size in config.model.hidden_sizes):
        errors.append(f"model.hidden_sizes values must all be > 0, got {config.model.hidden_sizes}")
```

In `load_config`, update the `model=ModelConfig(...)` line:

```python
        model=ModelConfig(
            name=raw["model"]["name"],
            hidden_sizes=tuple(raw["model"]["hidden_sizes"]) if "hidden_sizes" in raw["model"] else None,
        ),
```

And add a new `training=TrainingConfig(...)` entry to the `SimulationConfig(...)` constructor call (add it after the existing `attacks=AttackConfig(...)` block, before the closing `)`):

```python
        training=TrainingConfig(
            class_weighted_loss=raw.get("training", {}).get("class_weighted_loss", False),
        ),
```

In `load_experiment_data`'s IoT branch, replace:

```python
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
```

with:

```python
        global_model = build_model(
            config.model.name,
            experiment_data.input_dim,
            experiment_data.num_classes,
            hidden_sizes=config.model.hidden_sizes,
        )
        class_weights = experiment_data.class_weights if config.training.class_weighted_loss else None
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
                class_weights=class_weights,
            )
            for client_id, dataset in experiment_data.train_client_datasets.items()
        }
        return global_model, clients, experiment_data.test_dataset, experiment_data.attack_pool
```

Leave the non-IoT branch (`in_channels, num_classes = dataset_shape(...)` onward) unchanged — `hidden_sizes` stays `None` and `class_weighted_loss` stays `False` for MNIST/CIFAR-10 configs, since neither sets those new keys.

- [ ] **Step 5: Run tests to verify they pass**

Run: `python -m pytest tests/test_iot_datasets.py tests/test_config_validation.py -v`
Expected: PASS — all existing tests in both files plus the new ones from this task.

- [ ] **Step 6: Run the full suite to confirm no regressions**

Run: `python -m pytest tests/ -q`
Expected: all existing tests still pass, plus this task's new tests on top of Tasks 1-2's.

- [ ] **Step 7: Commit**

```bash
git add data/iot_datasets.py simulation/run.py tests/test_iot_datasets.py tests/test_config_validation.py
git commit -m "feat: wire class-weighted loss and configurable model capacity through the IoT pipeline"
```

---

### Task 4: Raise the training budget in the four real IoT configs

**Files:**
- Modify: `configs/iot/nbaiot.yaml`
- Modify: `configs/iot/ton_iot_devices.yaml`
- Modify: `configs/iot/unsw_nb15.yaml`
- Modify: `configs/iot/ton_iot_network.yaml`

**Interfaces:**
- Consumes: `model.hidden_sizes` and `training.class_weighted_loss` config keys from Task 3's `simulation/run.py`.
- Produces: the four configs' new "attempt 1" starting point for Tasks 5-8's tuning loops. `clients_per_round` and `model.hidden_sizes` values from this task are read (and, for `hidden_sizes`, potentially further widened) by Tasks 5-8.

This task has no unit tests of its own — `tests/test_iot_configs.py` (unmodified, already exists) already parametrizes over all four of these files and calls `load_config`/`validate_config` on each, so it automatically re-validates these edits.

- [ ] **Step 1: Edit `configs/iot/nbaiot.yaml`**

In the `data:` block, change `max_rows: 5000` to `max_rows: 15000`.

In the `model:` block, change:
```yaml
model:
  name: "mlp"
```
to:
```yaml
model:
  name: "mlp"
  hidden_sizes: [128, 64]
```

In the `federated:` block, change `num_rounds: 5` to `num_rounds: 20`, change `local_epochs: 1` to `local_epochs: 2`, and change `clients_per_round: 7` to `clients_per_round: 9` (this dataset's `data.num_clients` is 9 — see the file's own comment on that field).

Add a new `training:` block (place it after the `model:` block, before `federated:`, matching this file's existing top-to-bottom section order):
```yaml
training:
  class_weighted_loss: true
```

- [ ] **Step 2: Edit `configs/iot/ton_iot_devices.yaml`**

Same pattern. `data.max_rows: 5000` → `15000`. `model:` block gains `hidden_sizes: [128, 64]`. `federated.num_rounds: 5` → `20`, `federated.local_epochs: 1` → `2`, `federated.clients_per_round: 5` → `7` (this dataset's `data.num_clients` is 7). Add the same `training: class_weighted_loss: true` block.

- [ ] **Step 3: Edit `configs/iot/unsw_nb15.yaml`**

Same pattern. `data.max_rows: 5000` → `15000`. `model:` block gains `hidden_sizes: [128, 64]`. `federated.num_rounds: 5` → `20`, `federated.local_epochs: 1` → `2`, `federated.clients_per_round: 6` → `10` (this dataset's `data.num_clients` is 10). Add the same `training: class_weighted_loss: true` block.

- [ ] **Step 4: Edit `configs/iot/ton_iot_network.yaml`**

Same pattern. `data.max_rows: 5000` → `15000`. `model:` block gains `hidden_sizes: [128, 64]`. `federated.num_rounds: 5` → `20`, `federated.local_epochs: 1` → `2`, `federated.clients_per_round: 6` → `10` (this dataset's `data.num_clients` is 10). Add the same `training: class_weighted_loss: true` block.

- [ ] **Step 5: Run the existing config smoke tests**

Run: `python -m pytest tests/test_iot_configs.py -v`
Expected: PASS — all four edited configs still parse and validate.

- [ ] **Step 6: Run the full suite to confirm no regressions**

Run: `python -m pytest tests/ -q`
Expected: all tests still pass (this task adds no new tests; count is unchanged from Task 3's).

- [ ] **Step 7: Commit**

```bash
git add configs/iot/nbaiot.yaml configs/iot/ton_iot_devices.yaml configs/iot/unsw_nb15.yaml configs/iot/ton_iot_network.yaml
git commit -m "feat: raise training budget and enable class-weighted loss for IoT configs"
```

---

### Task 5: Tune and validate N-BaIoT toward 85% accuracy

**Files:**
- Modify: `configs/iot/nbaiot.yaml` (further tuning adjustments only, per the bounded loop below — do not touch any other dataset's config).
- Report: create `docs/superpowers/plans/2026-08-21-iot-accuracy-tuning-validation-report.md`.

**Interfaces:**
- Consumes: Task 4's boosted `configs/iot/nbaiot.yaml` (attempt 1) and the full pipeline from Tasks 1-3.
- Produces: the report file (created fresh by this task; Tasks 6-8 each append their own section to it — read this task's final report section as the template for structure/tone before writing your own).

This task has no unit tests of its own — it's an empirical, bounded tuning loop using the project's own `ExperimentRunner`, the same tool used throughout this project's history for exactly this kind of measurement. **Report the actual numbers you get. Do not adjust anything to force a particular outcome — if 85% isn't reached after the 4-attempt budget, report the real final number and a technical explanation, exactly as this project's validation reports have always done.**

- [ ] **Step 1: Confirm the full suite is green before starting**

Run: `python -m pytest tests/ -q`
Expected: all tests pass (confirms Tasks 1-4 landed cleanly).

- [ ] **Step 2: Attempt 1 — measure the `baseline` scenario with Task 4's boosted config**

Run (adjust the Python invocation for your environment; this worktree has no `.venv` — plain `python` on the system interpreter has all required dependencies already installed):

```bash
python -c "
from simulation.run import load_config
from evaluation.experiment_runner import ExperimentRunner

cfg = load_config('configs/iot/nbaiot.yaml')
result = ExperimentRunner(cfg).run('baseline')
acc = result.history[-1]['test_accuracy'] * 100
print(f'nbaiot baseline attempt 1: accuracy={acc:.1f}%')
"
```

This will take real wall-clock time (20 rounds x 2 local epochs x up to 15,000 rows/client, x9 clients) — expect several times longer than the ~2-3 minute runs from before this plan. Let it run to completion; there is no time limit on this task.

If `accuracy >= 85.0`, skip to Step 4 (this dataset is done — no further tuning attempts needed).

- [ ] **Step 3: Attempts 2-4 — bounded ordered adjustment loop (only if Step 2 was under 85%)**

Apply these adjustments to `configs/iot/nbaiot.yaml` **one at a time**, re-running the same Step 2 command after each, stopping as soon as one attempt clears 85% or all three adjustments have been tried:

- **Attempt 2:** `federated.local_epochs: 2` → `5`.
- **Attempt 3** (only if attempt 2 was still under 85%): `federated.num_rounds: 20` → `50`.
- **Attempt 4** (only if attempt 3 was still under 85%): `model.hidden_sizes: [128, 64]` → `[256, 128]`.

If, when you reach any of these three steps, you judge (from the attempt-so-far trend, e.g. loss/accuracy plateauing rather than still climbing) that a different single change is more likely to help than the one listed — most likely substituting `federated.learning_rate: 0.01` → `0.005` for whichever of the three above doesn't fit this dataset — you may substitute it for that attempt's slot. Document why in the report if you do. Do not exceed 4 total attempts regardless.

After Step 2 + up to 3 more attempts (4 total), stop — whether or not 85% was reached — and move to Step 4 with whatever the final config and final `baseline` accuracy are.

- [ ] **Step 4: Lock the config, verify the other two scenarios once**

With whichever config version was in place for the best (or final, if none cleared 85%) `baseline` attempt, run:

```bash
python -c "
from simulation.run import load_config
from evaluation.experiment_runner import ExperimentRunner

cfg = load_config('configs/iot/nbaiot.yaml')
for mode in ('baseline', 'fedavg_attack', 'ztfedtrust_attack'):
    result = ExperimentRunner(cfg).run(mode)
    acc = result.history[-1]['test_accuracy'] * 100
    print(f'nbaiot {mode}: accuracy={acc:.1f}%')
"
```

Do not adjust the config further based on these results — this is a one-time verification of the config Step 2/3 already settled on, not a fifth tuning attempt. If `fedavg_attack` or `ztfedtrust_attack` falls short of 85% even though `baseline` cleared it, that is itself a reportable finding (see Step 5), not something to tune away.

- [ ] **Step 5: Write the validation report**

Create `docs/superpowers/plans/2026-08-21-iot-accuracy-tuning-validation-report.md` with a `# IoT Model Accuracy Tuning — Validation Report` heading, then an `## N-BaIoT` section covering:
- The full sequence of attempts tried (attempt number, what changed, `baseline` accuracy measured), with the actual command output for each.
- The final config's `baseline`/`fedavg_attack`/`ztfedtrust_attack` accuracies from Step 4, and whether each cleared 85%.
- If 85% wasn't reached on `baseline` or either of the other two scenarios: an honest, specific technical explanation (e.g. class-imbalance severity even after weighting, feature separability limits, or — for `ztfedtrust_attack` specifically — a reference to the peer-cluster-trust validation report's documented N-BaIoT trust-mechanism findings, if that's a plausible contributing factor for that scenario specifically).

- [ ] **Step 6: Commit**

```bash
git add configs/iot/nbaiot.yaml docs/superpowers/plans/2026-08-21-iot-accuracy-tuning-validation-report.md
git commit -m "docs: tune and validate nbaiot accuracy"
```

---

### Task 6: Tune and validate TON_IoT devices toward 85% accuracy

**Files:**
- Modify: `configs/iot/ton_iot_devices.yaml` only.
- Report: append to `docs/superpowers/plans/2026-08-21-iot-accuracy-tuning-validation-report.md` (read the current file first — Task 5 already created it with an `## N-BaIoT` section; add a new `## TON_IoT Devices` section without disturbing the existing one).

**Interfaces:**
- Consumes: Task 4's boosted `configs/iot/ton_iot_devices.yaml` (attempt 1) and the full pipeline from Tasks 1-3. Independent of Task 5's N-BaIoT tuning result — do not read or depend on Task 5's chosen config values, only its report section as a structural template.

Identical procedure to Task 5, applied to `ton_iot_devices` instead of `nbaiot`, with the same 4-attempt cap and the same ordered adjustment list (`local_epochs` 2→5, then `num_rounds` 20→50, then `hidden_sizes` [128,64]→[256,128], with the same learning-rate-substitution allowance and the same honest-reporting requirement).

- [ ] **Step 1: Confirm the full suite is still green**

Run: `python -m pytest tests/ -q`
Expected: all tests pass.

- [ ] **Step 2: Attempt 1 — measure `baseline`**

```bash
python -c "
from simulation.run import load_config
from evaluation.experiment_runner import ExperimentRunner

cfg = load_config('configs/iot/ton_iot_devices.yaml')
result = ExperimentRunner(cfg).run('baseline')
acc = result.history[-1]['test_accuracy'] * 100
print(f'ton_iot_devices baseline attempt 1: accuracy={acc:.1f}%')
"
```

If `>= 85.0`, skip to Step 4.

- [ ] **Step 3: Attempts 2-4 — bounded ordered adjustment loop**

Same three ordered adjustments as Task 5 Step 3, applied to `configs/iot/ton_iot_devices.yaml`, same substitution allowance, same 4-total-attempts cap.

- [ ] **Step 4: Lock the config, verify the other two scenarios once**

```bash
python -c "
from simulation.run import load_config
from evaluation.experiment_runner import ExperimentRunner

cfg = load_config('configs/iot/ton_iot_devices.yaml')
for mode in ('baseline', 'fedavg_attack', 'ztfedtrust_attack'):
    result = ExperimentRunner(cfg).run(mode)
    acc = result.history[-1]['test_accuracy'] * 100
    print(f'ton_iot_devices {mode}: accuracy={acc:.1f}%')
"
```

- [ ] **Step 5: Append the validation report section**

Read `docs/superpowers/plans/2026-08-21-iot-accuracy-tuning-validation-report.md` first, then append a `## TON_IoT Devices` section in the same structure as Task 5's `## N-BaIoT` section (attempt sequence with real output, final three-scenario accuracies, honest explanation for any shortfall).

- [ ] **Step 6: Commit**

```bash
git add configs/iot/ton_iot_devices.yaml docs/superpowers/plans/2026-08-21-iot-accuracy-tuning-validation-report.md
git commit -m "docs: tune and validate ton_iot_devices accuracy"
```

---

### Task 7: Tune and validate UNSW-NB15 toward 85% accuracy

**Files:**
- Modify: `configs/iot/unsw_nb15.yaml` only.
- Report: append to `docs/superpowers/plans/2026-08-21-iot-accuracy-tuning-validation-report.md` (add a `## UNSW-NB15` section; don't disturb the two existing sections from Tasks 5-6).

**Interfaces:**
- Consumes: Task 4's boosted `configs/iot/unsw_nb15.yaml` (attempt 1) and the full pipeline from Tasks 1-3. Independent of Tasks 5-6's results.

Identical procedure to Task 5, applied to `unsw_nb15` instead of `nbaiot`, same 4-attempt cap, same ordered adjustment list, same substitution allowance, same honest-reporting requirement.

- [ ] **Step 1: Confirm the full suite is still green**

Run: `python -m pytest tests/ -q`

- [ ] **Step 2: Attempt 1 — measure `baseline`**

```bash
python -c "
from simulation.run import load_config
from evaluation.experiment_runner import ExperimentRunner

cfg = load_config('configs/iot/unsw_nb15.yaml')
result = ExperimentRunner(cfg).run('baseline')
acc = result.history[-1]['test_accuracy'] * 100
print(f'unsw_nb15 baseline attempt 1: accuracy={acc:.1f}%')
"
```

If `>= 85.0`, skip to Step 4.

- [ ] **Step 3: Attempts 2-4 — bounded ordered adjustment loop**

Same three ordered adjustments as Task 5 Step 3, applied to `configs/iot/unsw_nb15.yaml`.

- [ ] **Step 4: Lock the config, verify the other two scenarios once**

```bash
python -c "
from simulation.run import load_config
from evaluation.experiment_runner import ExperimentRunner

cfg = load_config('configs/iot/unsw_nb15.yaml')
for mode in ('baseline', 'fedavg_attack', 'ztfedtrust_attack'):
    result = ExperimentRunner(cfg).run(mode)
    acc = result.history[-1]['test_accuracy'] * 100
    print(f'unsw_nb15 {mode}: accuracy={acc:.1f}%')
"
```

- [ ] **Step 5: Append the validation report section**

Read the report file first, then append a `## UNSW-NB15` section in the same structure as the prior two sections.

- [ ] **Step 6: Commit**

```bash
git add configs/iot/unsw_nb15.yaml docs/superpowers/plans/2026-08-21-iot-accuracy-tuning-validation-report.md
git commit -m "docs: tune and validate unsw_nb15 accuracy"
```

---

### Task 8: Tune and validate TON_IoT network toward 85% accuracy

**Files:**
- Modify: `configs/iot/ton_iot_network.yaml` only.
- Report: append to `docs/superpowers/plans/2026-08-21-iot-accuracy-tuning-validation-report.md` (add a `## TON_IoT Network` section; don't disturb the three existing sections).

**Interfaces:**
- Consumes: Task 4's boosted `configs/iot/ton_iot_network.yaml` (attempt 1) and the full pipeline from Tasks 1-3. Independent of Tasks 5-7's results.

Identical procedure to Task 5, applied to `ton_iot_network` instead of `nbaiot`, same 4-attempt cap, same ordered adjustment list, same substitution allowance, same honest-reporting requirement.

- [ ] **Step 1: Confirm the full suite is still green**

Run: `python -m pytest tests/ -q`

- [ ] **Step 2: Attempt 1 — measure `baseline`**

```bash
python -c "
from simulation.run import load_config
from evaluation.experiment_runner import ExperimentRunner

cfg = load_config('configs/iot/ton_iot_network.yaml')
result = ExperimentRunner(cfg).run('baseline')
acc = result.history[-1]['test_accuracy'] * 100
print(f'ton_iot_network baseline attempt 1: accuracy={acc:.1f}%')
"
```

If `>= 85.0`, skip to Step 4.

- [ ] **Step 3: Attempts 2-4 — bounded ordered adjustment loop**

Same three ordered adjustments as Task 5 Step 3, applied to `configs/iot/ton_iot_network.yaml`.

- [ ] **Step 4: Lock the config, verify the other two scenarios once**

```bash
python -c "
from simulation.run import load_config
from evaluation.experiment_runner import ExperimentRunner

cfg = load_config('configs/iot/ton_iot_network.yaml')
for mode in ('baseline', 'fedavg_attack', 'ztfedtrust_attack'):
    result = ExperimentRunner(cfg).run(mode)
    acc = result.history[-1]['test_accuracy'] * 100
    print(f'ton_iot_network {mode}: accuracy={acc:.1f}%')
"
```

- [ ] **Step 5: Append the validation report section, and write the report's overall summary**

Read the report file first, then append a `## TON_IoT Network` section in the same structure as the prior three sections. Since this is the last of the four datasets, also add a final `## Overall Summary` section at the end of the report: a small table of all four datasets x three scenarios' final accuracies and whether each cleared 85%, plus one honest paragraph on the overall outcome — do not claim full success if any scenario/dataset fell short; state plainly which did and didn't, matching this project's established reporting practice.

- [ ] **Step 6: Commit**

```bash
git add configs/iot/ton_iot_network.yaml docs/superpowers/plans/2026-08-21-iot-accuracy-tuning-validation-report.md
git commit -m "docs: tune and validate ton_iot_network accuracy, add overall summary"
```

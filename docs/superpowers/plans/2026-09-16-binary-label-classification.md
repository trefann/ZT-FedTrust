# Binary Benign/Attack Label Classification Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an opt-in binary (benign vs. attack) classification mode for the IoT datasets, and empirically check whether N-BaIoT and UNSW-NB15 get close to 90% test accuracy under binary framing without any extra training budget — since multi-class tuning already showed more rounds/epochs/width doesn't close the gap.

**Architecture:** A new `binary_labels` flag threads from YAML config down through `simulation/run.py`'s `DataConfig` into `data/iot_datasets.py`'s two shared builder functions. When set, each dataset's label column is rewritten to exactly `"benign"`/`"attack"` (using the dataset's existing, already-known `benign_label` value) immediately before the existing multi-class label-fitting/encoding/class-weighting machinery runs — that machinery is otherwise completely unchanged. Two new sibling config files exercise this for N-BaIoT and UNSW-NB15 at their original (pre-tuning) training budget, and a single `baseline` measurement per dataset answers the accuracy question.

**Tech Stack:** Python, pandas, PyTorch, pytest — no new dependencies.

## Global Constraints

- `DataConfig.binary_labels` and every new function parameter it threads through must default to `False`/absent, so every existing config (both already-tuned multi-class IoT configs and the two non-IoT vision configs) is completely unaffected unless it explicitly opts in.
- The canonical collapsed labels are always the literal strings `"benign"` and `"attack"`, regardless of a dataset's original benign-label spelling (`"benign"`, `"normal"`, `"Normal"`).
- `binary_labels=True` with a dataset whose `benign_label` is `None` (only CICIoT2023 today) must raise a clear `ValueError` rather than silently collapsing every row to `"attack"` — a single-class dataset can't be meaningfully classified.
- The full existing test suite (278 tests as of this plan's baseline — confirm the exact number is current with `pytest tests/ -q` before starting, don't assume it's still accurate) must stay green throughout both tasks.
- Do not edit `configs/iot/nbaiot.yaml` or `configs/iot/unsw_nb15.yaml` (the already-tuned multi-class configs) — binary mode uses new, separate config files.
- This plan does not touch TON_IoT devices, TON_IoT network, `trust/`, `attacks/`, or `server/`.

---

## File Structure

**Modified files:**
- `data/iot_datasets.py` — both shared builders and all five per-dataset loader functions plus `load_iot_experiment` gain a `binary_labels: bool = False` parameter.
- `simulation/run.py` — `DataConfig` gains `binary_labels: bool = False`; `load_config` parses it; `load_experiment_data`'s IoT branch forwards it.
- `tests/test_iot_datasets.py` — new tests for binary collapsing on both builder paths.
- `tests/test_config_validation.py` — new test for the config default.

**New files:**
- `configs/iot/nbaiot_binary.yaml` — N-BaIoT at its original (pre-tuning) budget, `binary_labels: true`.
- `configs/iot/unsw_nb15_binary.yaml` — same pattern for UNSW-NB15.
- `docs/superpowers/plans/2026-09-16-binary-label-classification-validation-report.md` — the empirical findings.

---

### Task 1: Wire `binary_labels` through the IoT data pipeline

**Files:**
- Modify: `data/iot_datasets.py`
- Modify: `simulation/run.py`
- Test: `tests/test_iot_datasets.py`
- Test: `tests/test_config_validation.py`

**Interfaces:**
- Consumes: nothing new from other tasks.
- Produces: `_build_partitioned_experiment(..., binary_labels: bool = False)`, `_build_natural_client_experiment(..., binary_labels: bool = False)`, every `load_*` function and `load_iot_experiment(..., binary_labels: bool = False)`, `DataConfig.binary_labels: bool = False`. Task 2 depends on `data.binary_labels: true` being a valid, working YAML key by the end of this task.

- [ ] **Step 1: Write the failing tests**

First, read `data/iot_datasets.py`, `tests/test_iot_datasets.py`, `simulation/run.py`, and `tests/test_config_validation.py` in full (all already read during planning; re-read here since line numbers can drift) so your edits land in the right place.

Append to `tests/test_iot_datasets.py`. In `TestBuildPartitionedExperiment`, add:

```python
    def test_binary_labels_collapses_to_benign_and_attack(self):
        df = _make_partitioned_df()
        result = _build_partitioned_experiment(
            df, "label", "normal", num_clients=3, partition_type="iid",
            dirichlet_alpha=0.5, max_rows=None, seed=42, binary_labels=True,
        )
        assert result.class_names == ["attack", "benign"]
        assert result.num_classes == 2

    def test_binary_labels_with_no_benign_label_raises(self):
        df = _make_partitioned_df()
        with pytest.raises(ValueError):
            _build_partitioned_experiment(
                df, "label", None, num_clients=3, partition_type="iid",
                dirichlet_alpha=0.5, max_rows=None, seed=42, binary_labels=True,
            )
```

(Note: `_make_partitioned_df`'s label column uses `"normal"`/`"ddos"`/`"backdoor"`, but this dataset's real `benign_label` in these tests is passed as `"normal"` per the existing tests in this class, e.g. `test_produces_requested_number_of_clients` — that's consistent with the sample fixture, so `binary_labels=True` here collapses `"normal"` rows to `"benign"` and `"ddos"`/`"backdoor"` rows to `"attack"`.)

In `TestBuildNaturalClientExperiment`, add:

```python
    def test_binary_labels_collapses_to_benign_and_attack(self):
        result = _build_natural_client_experiment(
            self._device_frames(), label_column="type", benign_label="normal",
            max_rows_per_device=None, seed=7, binary_labels=True,
        )
        assert result.class_names == ["attack", "benign"]
        assert result.num_classes == 2

    def test_binary_labels_with_no_benign_label_raises(self):
        with pytest.raises(ValueError):
            _build_natural_client_experiment(
                self._device_frames(), label_column="type", benign_label=None,
                max_rows_per_device=None, seed=7, binary_labels=True,
            )
```

Append to `tests/test_config_validation.py`:

```python
def test_data_binary_labels_defaults_to_false():
    config = _base_config()
    assert config.data.binary_labels is False
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_iot_datasets.py tests/test_config_validation.py -v -k "binary_labels"`
Expected: FAIL — `TypeError: _build_partitioned_experiment() got an unexpected keyword argument 'binary_labels'` (and the same for `_build_natural_client_experiment`), and `AttributeError: 'DataConfig' object has no attribute 'binary_labels'`.

- [ ] **Step 3: Modify `data/iot_datasets.py`**

In `_build_partitioned_experiment`, change the signature to add the new parameter, and insert the collapsing step right after subsampling:

```python
def _build_partitioned_experiment(
    df: pd.DataFrame,
    label_column: str,
    benign_label: Optional[str],
    num_clients: int,
    partition_type: str,
    dirichlet_alpha: float,
    max_rows: Optional[int],
    seed: int,
    binary_labels: bool = False,
) -> IoTExperimentData:
    df = subsample_dataframe(df, max_rows, seed)
    if binary_labels:
        if benign_label is None:
            raise ValueError(
                "binary_labels=True requires a benign_label to collapse against, but "
                "this dataset has none (e.g. CICIoT2023's benign-free slice)."
            )
        df = df.assign(**{
            label_column: df[label_column].apply(
                lambda value: "benign" if value == benign_label else "attack"
            )
        })
        benign_label = "benign"
    feature_columns = [c for c in df.columns if c != label_column]
    class_names = fit_label_classes(df[label_column])
```

(Everything else in this function — from `train_df, test_df = split_train_test(...)` through the `return IoTExperimentData(...)` — is unchanged.)

In `_build_natural_client_experiment`, change the signature and insert the collapsing step right after `sampled` is built, before `full_pool` is computed:

```python
def _build_natural_client_experiment(
    device_dataframes: Dict[str, pd.DataFrame],
    label_column: str,
    benign_label: Optional[str],
    max_rows_per_device: Optional[int],
    seed: int,
    binary_labels: bool = False,
) -> IoTExperimentData:
    device_names = sorted(device_dataframes)
    sampled = {
        name: subsample_dataframe(device_dataframes[name], max_rows_per_device, seed)
        for name in device_names
    }
    if binary_labels:
        if benign_label is None:
            raise ValueError(
                "binary_labels=True requires a benign_label to collapse against, but "
                "this dataset has none (e.g. CICIoT2023's benign-free slice)."
            )
        sampled = {
            name: df.assign(**{
                label_column: df[label_column].apply(
                    lambda value: "benign" if value == benign_label else "attack"
                )
            })
            for name, df in sampled.items()
        }
        benign_label = "benign"
    feature_columns = [c for c in next(iter(sampled.values())).columns if c != label_column]
```

(Everything else in this function — from `full_pool = pd.concat(...)` through the `return IoTExperimentData(...)` — is unchanged.)

Update every per-dataset loader function to accept and forward `binary_labels: bool = False`:

```python
def load_ciciot2023(
    data_dir: str,
    num_clients: int,
    partition_type: str,
    dirichlet_alpha: float,
    max_rows: Optional[int],
    seed: int,
    binary_labels: bool = False,
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
        binary_labels=binary_labels,
    )
```

```python
def load_unsw_nb15(
    data_dir: str,
    num_clients: int,
    partition_type: str,
    dirichlet_alpha: float,
    max_rows: Optional[int],
    seed: int,
    binary_labels: bool = False,
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
        binary_labels=binary_labels,
    )
```

```python
def load_ton_iot_network(
    data_dir: str,
    num_clients: int,
    partition_type: str,
    dirichlet_alpha: float,
    max_rows: Optional[int],
    seed: int,
    binary_labels: bool = False,
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
        binary_labels=binary_labels,
    )
```

```python
def load_ton_iot_devices(
    data_dir: str, max_rows_per_device: Optional[int], seed: int, binary_labels: bool = False
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
        binary_labels=binary_labels,
    )
```

```python
def load_nbaiot(
    data_dir: str, max_rows_per_device: Optional[int], seed: int, binary_labels: bool = False
) -> IoTExperimentData:
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
        binary_labels=binary_labels,
    )
```

Update `load_iot_experiment`'s signature and every dispatch line:

```python
def load_iot_experiment(
    name: str,
    data_dir: str,
    max_rows: Optional[int],
    num_clients: int,
    partition_type: str,
    dirichlet_alpha: float,
    seed: int,
    binary_labels: bool = False,
) -> IoTExperimentData:
    """Dispatch to the right per-dataset loader by name.

    `num_clients`/`partition_type`/`dirichlet_alpha` are ignored for
    natural-client datasets (`ton_iot_devices`, `nbaiot`) — those always
    produce one client per real device. `max_rows` also means something
    different for these two: it caps rows PER DEVICE (each of the 7-9
    real devices is independently subsampled to at most `max_rows` rows),
    not the dataset's total row count as it does for the partitioned
    datasets (`ciciot2023`, `unsw_nb15`, `ton_iot_network`) — the same
    `max_rows` value therefore yields a much larger total dataset for a
    natural-client dataset than for a partitioned one.
    """
    if name == "ciciot2023":
        return load_ciciot2023(
            data_dir, num_clients, partition_type, dirichlet_alpha, max_rows, seed, binary_labels
        )
    if name == "unsw_nb15":
        return load_unsw_nb15(
            data_dir, num_clients, partition_type, dirichlet_alpha, max_rows, seed, binary_labels
        )
    if name == "ton_iot_network":
        return load_ton_iot_network(
            data_dir, num_clients, partition_type, dirichlet_alpha, max_rows, seed, binary_labels
        )
    if name == "ton_iot_devices":
        return load_ton_iot_devices(data_dir, max_rows, seed, binary_labels)
    if name == "nbaiot":
        return load_nbaiot(data_dir, max_rows, seed, binary_labels)
    if name == "botiot":
        raise ValueError(
            "botiot is a disabled placeholder: the real BoT-IoT dataset has not been "
            "supplied yet. Use one of: " + ", ".join(IOT_SUPPORTED_DATASETS)
        )
    raise ValueError(f"Unknown IoT dataset: {name!r}. Supported: {list(IOT_SUPPORTED_DATASETS)}")
```

- [ ] **Step 4: Modify `simulation/run.py`**

Update `DataConfig`:

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
    binary_labels: bool = False
```

In `load_config`, update the `data=DataConfig(...)` block:

```python
        data=DataConfig(
            dataset=data_raw["dataset"],
            data_dir=data_raw["data_dir"],
            num_clients=data_raw["num_clients"],
            partition_type=partition_raw["type"],
            dirichlet_alpha=partition_raw.get("alpha", 0.5),
            batch_size=data_raw["batch_size"],
            max_rows=data_raw.get("max_rows"),
            binary_labels=data_raw.get("binary_labels", False),
        ),
```

In `load_experiment_data`'s IoT branch, update the `load_iot_experiment(...)` call:

```python
        experiment_data = load_iot_experiment(
            name=config.data.dataset,
            data_dir=config.data.data_dir,
            max_rows=config.data.max_rows,
            num_clients=config.data.num_clients,
            partition_type=config.data.partition_type,
            dirichlet_alpha=config.data.dirichlet_alpha,
            seed=config.seed,
            binary_labels=config.data.binary_labels,
        )
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `python -m pytest tests/test_iot_datasets.py tests/test_config_validation.py -v`
Expected: PASS — all existing tests in both files plus the 5 new ones (4 in `test_iot_datasets.py`, 1 in `test_config_validation.py`).

- [ ] **Step 6: Run the full suite to confirm no regressions**

Run: `python -m pytest tests/ -q`
Expected: all existing tests still pass, plus this task's 5 new tests.

- [ ] **Step 7: Commit**

```bash
git add data/iot_datasets.py simulation/run.py tests/test_iot_datasets.py tests/test_config_validation.py
git commit -m "feat: add opt-in binary benign/attack label classification for IoT datasets"
```

---

### Task 2: Binary-mode configs and empirical validation for N-BaIoT and UNSW-NB15

**Files:**
- Create: `configs/iot/nbaiot_binary.yaml`
- Create: `configs/iot/unsw_nb15_binary.yaml`
- Report: create `docs/superpowers/plans/2026-09-16-binary-label-classification-validation-report.md`

**Interfaces:**
- Consumes: `data.binary_labels: true` from Task 1's `simulation/run.py`/`data/iot_datasets.py`.

This task has no unit tests of its own — it's a single fast empirical measurement per dataset, not a tuning loop. **Report the actual numbers you get, whatever they are — do not adjust anything to force a particular outcome**, matching this project's established practice (see `docs/superpowers/plans/2026-08-21-iot-accuracy-tuning-validation-report.md`).

- [ ] **Step 1: Confirm the full suite is green before starting**

Run: `python -m pytest tests/ -q`
Expected: all tests pass (confirms Task 1 landed cleanly).

- [ ] **Step 2: Create `configs/iot/nbaiot_binary.yaml`**

This is `configs/iot/nbaiot.yaml`'s content from BEFORE the accuracy-tuning plan's Task 5 changed it — i.e. the original "attempt 1" budget, not the current (post-tuning) file. Read the current `configs/iot/nbaiot.yaml` to copy every section verbatim EXCEPT `data.max_rows`, `federated.num_rounds`, `federated.local_epochs`, and `model.hidden_sizes`, which must use these exact values instead of whatever the current tuned file has:

```yaml
# N-BaIoT, binary (benign vs. attack) classification. A sibling of
# nbaiot.yaml that collapses the multi-class Mirai/Gafgyt-subtype
# vocabulary down to two classes, at the ORIGINAL (pre-tuning) training
# budget — see docs/superpowers/specs/2026-09-16-binary-label-classification-design.md.
#
# Requires 7-Zip or unrar installed on the machine the first time this
# runs (extracts the .rar attack archives; cached afterward). See
# data/rar_extract.py.
#
# Usage:
#   python simulation/run.py --config configs/iot/nbaiot_binary.yaml --experiment compare_all

seed: 42
device: "cpu"

data:
  dataset: "nbaiot"
  data_dir: "./ZT-Fedtrust Datasets"
  num_clients: 9
  batch_size: 32
  max_rows: 15000  # per-device cap for natural-client datasets, not a dataset total (up to 9x this many rows overall)
  binary_labels: true
  partition:
    type: "iid"
    alpha: 0.5

model:
  name: "mlp"
  hidden_sizes: [128, 64]

training:
  class_weighted_loss: true

federated:
  num_rounds: 20
  clients_per_round: 9
  local_epochs: 2
  learning_rate: 0.01
  momentum: 0.9

logging:
  log_dir: "./results/iot_nbaiot_binary/logs"
  log_level: "INFO"
  results_dir: "./results/iot_nbaiot_binary"

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
  real_traffic:
    enabled: true
    probability: 1.0
```

- [ ] **Step 3: Create `configs/iot/unsw_nb15_binary.yaml`**

Same pattern for UNSW-NB15's original (pre-tuning) budget:

```yaml
# UNSW-NB15, binary (benign vs. attack) classification. A sibling of
# unsw_nb15.yaml that collapses the 10-category attack_cat vocabulary down
# to two classes, at the ORIGINAL (pre-tuning) training budget — see
# docs/superpowers/specs/2026-09-16-binary-label-classification-design.md.
#
# Usage:
#   python simulation/run.py --config configs/iot/unsw_nb15_binary.yaml --experiment compare_all

seed: 42
device: "cpu"

data:
  dataset: "unsw_nb15"
  data_dir: "./ZT-Fedtrust Datasets"
  num_clients: 10
  batch_size: 32
  max_rows: 15000
  binary_labels: true
  partition:
    type: "dirichlet"
    alpha: 0.5

model:
  name: "mlp"
  hidden_sizes: [128, 64]

training:
  class_weighted_loss: true

federated:
  num_rounds: 20
  clients_per_round: 10
  local_epochs: 2
  learning_rate: 0.01
  momentum: 0.9

logging:
  log_dir: "./results/iot_unsw_nb15_binary/logs"
  log_level: "INFO"
  results_dir: "./results/iot_unsw_nb15_binary"

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
  real_traffic:
    enabled: true
    probability: 1.0
```

- [ ] **Step 4: Run the config smoke check**

Both new files should parse and validate. Run:

```bash
python -c "
from simulation.run import load_config, validate_config
for path in ('configs/iot/nbaiot_binary.yaml', 'configs/iot/unsw_nb15_binary.yaml'):
    cfg = load_config(path)
    validate_config(cfg)
    print(f'{path}: OK, binary_labels={cfg.data.binary_labels}')
"
```

Expected: both print `OK, binary_labels=True` with no errors.

- [ ] **Step 5: Run the `baseline` measurement for both datasets**

```bash
python -c "
from simulation.run import load_config
from evaluation.experiment_runner import ExperimentRunner

for path, name in (('configs/iot/nbaiot_binary.yaml', 'nbaiot'), ('configs/iot/unsw_nb15_binary.yaml', 'unsw_nb15')):
    cfg = load_config(path)
    result = ExperimentRunner(cfg).run('baseline')
    acc = result.history[-1]['test_accuracy'] * 100
    print(f'{name} (binary) baseline: accuracy={acc:.2f}%')
"
```

This is a much smaller budget than the multi-class tuning runs (20 rounds, 2 local epochs) — expect a few minutes per dataset, not tens of minutes.

- [ ] **Step 6: Write the validation report**

Create `docs/superpowers/plans/2026-09-16-binary-label-classification-validation-report.md` with a `# Binary Benign/Attack Label Classification — Validation Report` heading, covering:
- The exact commands run and their full output from Steps 4-5.
- Both datasets' binary `baseline` accuracy, compared directly against their multi-class `baseline` numbers at the same original budget (N-BaIoT multi-class attempt 1 was 67.09%; UNSW-NB15 multi-class attempt 1 was 62.87% — both documented in `docs/superpowers/plans/2026-08-21-iot-accuracy-tuning-validation-report.md`).
- An honest assessment: does binary framing get either dataset close to or above 90%? If yes, say so plainly and note that a natural next step (not part of this task) would be measuring `fedavg_attack`/`ztfedtrust_attack` too. If no, say that plainly too, with your best technical read of why (e.g. residual feature overlap between benign and attack traffic, as already discussed for N-BaIoT's multi-class ceiling in the earlier report) — do not soften either outcome.

- [ ] **Step 7: Commit**

```bash
git add configs/iot/nbaiot_binary.yaml configs/iot/unsw_nb15_binary.yaml docs/superpowers/plans/2026-09-16-binary-label-classification-validation-report.md
git commit -m "docs: validate binary benign/attack classification for nbaiot and unsw_nb15"
```

# Binary Benign/Attack Label Classification — Validation Report

## Pre-flight

`python -m pytest tests/ -q` → 283 passed, 1 pre-existing scipy warning (confirms Task 1 landed cleanly before this empirical step).

## Config smoke check

Command:

```python
from simulation.run import load_config, validate_config
for path in ('configs/iot/nbaiot_binary.yaml', 'configs/iot/unsw_nb15_binary.yaml'):
    cfg = load_config(path)
    validate_config(cfg)
    print(f'{path}: OK, binary_labels={cfg.data.binary_labels}')
```

Output:

```
configs/iot/nbaiot_binary.yaml: OK, binary_labels=True
configs/iot/unsw_nb15_binary.yaml: OK, binary_labels=True
```

Both new configs parse and validate correctly with `binary_labels` wired through.

## `baseline` measurement

Command:

```python
from simulation.run import load_config
from evaluation.experiment_runner import ExperimentRunner

for path, name in (('configs/iot/nbaiot_binary.yaml', 'nbaiot'), ('configs/iot/unsw_nb15_binary.yaml', 'unsw_nb15')):
    cfg = load_config(path)
    result = ExperimentRunner(cfg).run('baseline')
    acc = result.history[-1]['test_accuracy'] * 100
    print(f'{name} (binary) baseline: accuracy={acc:.2f}%')
```

Output (PerformanceWarning lines are pre-existing pandas fragmentation warnings from `data/iot_datasets.py`, unrelated to this change):

```
nbaiot (binary) baseline: accuracy=99.89%
unsw_nb15 (binary) baseline: accuracy=88.53%
```

## Comparison against multi-class, same original budget

| Dataset | Multi-class `baseline` (attempt 1) | Binary `baseline` | Delta |
|---|---|---|---|
| N-BaIoT | 67.09% | **99.89%** | +32.80pts |
| UNSW-NB15 | 62.87% | **88.53%** | +25.66pts |

(Multi-class attempt-1 numbers are from `docs/superpowers/plans/2026-08-21-iot-accuracy-tuning-validation-report.md`'s N-BaIoT and UNSW-NB15 sections, run at the identical `num_rounds=20, local_epochs=2, max_rows=15000, hidden_sizes=[128,64]` budget as these binary runs, so the comparison is like-for-like — only the label granularity changed.)

## Honest assessment

**N-BaIoT clears the 90% target decisively: 99.89%.** This is a dramatic, unambiguous result. The gap between N-BaIoT's multi-class ceiling (never exceeded ~74% across a 4-attempt, 4x-larger training budget in the earlier tuning work) and this binary result (99.89% on the *smaller*, original budget) confirms the earlier hypothesis directly: the accuracy ceiling was never a training-budget or model-capacity problem — it was the difficulty of separating individual Mirai/Gafgyt attack subtypes from each other and from benign traffic. Detecting *that an attack occurred at all* is, on this dataset, nearly perfectly separable from the given features.

**UNSW-NB15 gets close but does not clear 90%: 88.53%.** This is still a very large improvement (+25.66pts) over the multi-class ceiling, and well ahead of anything the multi-class tuning loop achieved (its best multi-class result across 4 attempts was 66.57%). The remaining ~1.5-point gap to 90% is plausibly closeable: UNSW-NB15's multi-class tuning loop showed noisy, non-monotonic behavior across attempts (62.87% → 63.30% → 60.27% → 66.57%), suggesting some genuine sensitivity to training budget/capacity that a similar bounded tuning pass in binary mode might resolve. UNSW-NB15's attack traffic (fuzzers, reconnaissance, generic exploits, etc., per its own 10-category taxonomy) is also plausibly more heterogeneous in how closely various attack types resemble normal traffic than N-BaIoT's Mirai/Gafgyt botnet traffic, which could explain why some residual multi-class-shaped difficulty carries over into the binary framing.

## Extension: TON_IoT devices and TON_IoT network

Following the strong N-BaIoT/UNSW-NB15 results above, binary mode was extended to the remaining two datasets using the same mechanism (Task 1's `binary_labels` flag, already generic across all five IoT loaders — no code changes needed). New sibling configs `configs/iot/ton_iot_devices_binary.yaml` and `configs/iot/ton_iot_network_binary.yaml` were created at each dataset's original (pre-tuning) budget, matching the same pattern as `nbaiot_binary.yaml`/`unsw_nb15_binary.yaml`.

TON_IoT network had never been tuned in multi-class mode at all (Task 8 of the accuracy-tuning plan was never run), so its multi-class `baseline` at the original budget was measured for the first time here, for a fair comparison.

Command:

```python
from simulation.run import load_config
from evaluation.experiment_runner import ExperimentRunner

runs = [
    ('configs/iot/ton_iot_network.yaml', 'ton_iot_network (multi-class, attempt 1)'),
    ('configs/iot/ton_iot_devices_binary.yaml', 'ton_iot_devices (binary)'),
    ('configs/iot/ton_iot_network_binary.yaml', 'ton_iot_network (binary)'),
]
for path, name in runs:
    cfg = load_config(path)
    result = ExperimentRunner(cfg).run('baseline')
    acc = result.history[-1]['test_accuracy'] * 100
    print(f'{name} baseline: accuracy={acc:.2f}%')
```

Output:

```
ton_iot_network (multi-class, attempt 1) baseline: accuracy=70.83%
ton_iot_devices (binary) baseline: accuracy=62.99%
ton_iot_network (binary) baseline: accuracy=97.63%
```

| Dataset | Multi-class `baseline` (attempt 1) | Binary `baseline` | Delta | Clears 90%? |
|---|---|---|---|---|
| TON_IoT network | 70.83% | **97.63%** | +26.80pts | **Yes** |
| TON_IoT devices | 41.76% | **62.99%** | +21.23pts | No |

**TON_IoT network clears 90% decisively, joining N-BaIoT.** Same pattern as the other partitioned/homogeneous dataset (UNSW-NB15): a large jump from binary framing, here enough to clear the target outright. This makes three of four datasets where binary classification either clears or nearly clears 90%, reinforcing that multi-class attack-subtype difficulty — not training budget, not federation structure in general — was the dominant limiting factor.

**TON_IoT devices improves substantially (+21.23pts) but remains well short of 90%.** This is consistent with everything already known about this specific dataset from the multi-class tuning work: its 4-attempt tuning sequence was flat/noisy regardless of budget (41.76% → 39.13% → 39.02% → 41.96%), the `real_traffic` attack causes severe damage under `fedavg_attack` (24.73%), and the Zero Trust defense makes it actively worse under `ztfedtrust_attack` (14.31%) due to trust-based selection over-excluding clients on this small 7-device federation. Binary framing helps — the classification task itself was part of the problem — but something else specific to this dataset (federation size, feature quality from the zero-padded union schema across heterogeneous device types, or the attack's disproportionate impact on a small federation) is still capping it well below the other three datasets' binary results. This is a genuine, honest limitation, not something more label-simplification is expected to fix.

## Conclusion

Binary reformulation directly and substantially validates the hypothesis it was designed to test: this project's accuracy ceiling was driven primarily by multi-class attack-subtype classification difficulty, not by training budget, model capacity, or an inherent limit of the federated learning pipeline. **Three of four datasets (N-BaIoT: 99.89%, TON_IoT network: 97.63%, and UNSW-NB15 at 88.53% just short) now clear or nearly clear the 90% target** — a dramatic change from the multi-class ceiling, where none of the four ever exceeded 74%. **TON_IoT devices remains the exception**: substantially improved (41.76% → 62.99%) but still well below target, with a distinct, already-documented set of problems (severe attack damage, defense over-exclusion, a small and structurally unusual federation) that binary framing alone does not resolve.

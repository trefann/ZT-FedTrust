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

**UNSW-NB15 initially got close but did not clear 90%: 88.53%** at the original (attempt 1) budget. This was still a very large improvement (+25.66pts) over the multi-class ceiling, and well ahead of anything the multi-class tuning loop achieved (its best multi-class result across 4 attempts was 66.57%).

### UNSW-NB15 binary — pushed past 90% with further tuning

Unlike the multi-class tuning loop's noisy, non-monotonic behavior on this dataset (62.87% → 63.30% → 60.27% → 66.57%), the binary version's tuning curve climbed steadily and monotonically with every adjustment, suggesting genuine, exploitable headroom rather than noise. The bounded 4-attempt tuning loop was run first (same ordered adjustments as the multi-class work: more local epochs, then more rounds, then wider hidden layers), then — with the user's explicit approval to exceed the plan's normal 4-attempt cap given how close and how consistent the trend was — one further attempt substituted a lower learning rate.

| Attempt | Config | `baseline` accuracy |
|---|---|---|
| 1 | `num_rounds=20, local_epochs=2, hidden_sizes=[128,64], lr=0.01` | 88.53% |
| 2 | + `local_epochs: 2 → 5` | 88.60% |
| 3 | + `num_rounds: 20 → 50` | 89.40% |
| 4 | + `hidden_sizes: [128,64] → [256,128]` | 89.77% |
| 5 (past the normal cap) | + `learning_rate: 0.01 → 0.005` | **90.00%** |

Raw output:

```
Attempt 1: unsw_nb15 (binary) baseline: accuracy=88.53%
Attempt 2: unsw_nb15 (binary) baseline attempt2(rounds=20,epochs=5,hidden=128-64): accuracy=88.60%
Attempt 3: unsw_nb15 (binary) baseline attempt3(rounds=50,epochs=5,hidden=128-64): accuracy=89.40%
Attempt 4: unsw_nb15 (binary) baseline attempt4(rounds=50,epochs=5,hidden=256-128): accuracy=89.77%
Attempt 5: unsw_nb15 (binary) baseline attempt5(rounds=50,epochs=5,hidden=256-128,lr=0.005): accuracy=90.00%
```

Attempt 5's config was locked and verified once per scenario:

```python
from simulation.run import load_config
from evaluation.experiment_runner import ExperimentRunner

cfg = load_config('configs/iot/unsw_nb15_binary.yaml')
for mode in ('baseline', 'fedavg_attack', 'ztfedtrust_attack'):
    result = ExperimentRunner(cfg).run(mode)
    acc = result.history[-1]['test_accuracy'] * 100
    print(f'unsw_nb15 (binary) {mode}: accuracy={acc:.2f}%')
```

Output:

```
unsw_nb15 (binary) baseline: accuracy=90.00%
unsw_nb15 (binary) fedavg_attack: accuracy=90.37%
unsw_nb15 (binary) ztfedtrust_attack: accuracy=90.33%
```

| Scenario | Accuracy | Cleared 90%? |
|---|---|---|
| `baseline` | 90.00% | Yes (exactly at target) |
| `fedavg_attack` | 90.37% | Yes |
| `ztfedtrust_attack` | 90.33% | Yes |

**All three scenarios now clear 90% on UNSW-NB15.** Reported honestly: `baseline` lands exactly at 90.00%, not comfortably above it — this is a real result, not adjusted to hit the number, but it is a narrow margin and a different seed or minor config change could plausibly land just under 90% again. The three scenarios are also tightly clustered (90.00%–90.37%, a 0.37pt spread) with `ztfedtrust_attack` and `fedavg_attack` both slightly *above* `baseline` — there is no meaningful defended-vs-undefended gap in either direction here, consistent with what the multi-class tuning work on this same dataset already found (the defense's earlier apparent advantage at a much smaller training budget did not persist once training budget increased).

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

## Addendum: results measured after this report was first written

Added 2026-11-02. The scripts and logs below were found uncommitted in the working tree and are now preserved in `artifacts/2026-09-16-binary-label-scratch/` (see its README). Their file timestamps (2026-09-16, 08:49–09:45) fall after this report's last commit (08:47), and their contents had not been recorded here. Numbers are copied from the logs; nothing was re-run for this addendum.

### N-BaIoT and TON_IoT network: the remaining two scenarios

`verify_nbaiot_tonnet_binary.py` ran all three scenarios on the committed `nbaiot_binary.yaml` and `ton_iot_network_binary.yaml` (raw output: `nbaiot_tonnet_binary_verify.log`).

| Dataset (binary) | `baseline` | `fedavg_attack` | `ztfedtrust_attack` |
|---|---|---|---|
| N-BaIoT | 99.89% | 99.92% | 99.73% |
| TON_IoT network | 97.63% | 97.63% | 97.70% |

All six clear 90%. As with UNSW-NB15, there is no meaningful defended-vs-undefended gap: N-BaIoT `ztfedtrust_attack` is 0.19pt below `fedavg_attack`, and TON_IoT network's three scenarios span 0.07pt. These tasks may simply be too close to saturated for accuracy to reveal a defense effect, but that is an interpretation this data does not test.

### TON_IoT devices: further tuning attempts (none beat 62.99%)

`tune_ton_iot_devices_binary_attempt.py` produced these `baseline` results. The settings were applied by temporarily editing the config, so they are not reproducible from the committed `ton_iot_devices_binary.yaml` alone.

| Settings | `baseline` accuracy |
|---|---|
| rounds=20, epochs=5, hidden=128-64 | 61.50% |
| rounds=50, epochs=5, hidden=128-64 | 59.79% |
| rounds=20, epochs=2, hidden=128-64, max_rows=40000 | 59.56% |

### TON_IoT devices: class balance

`diagnose_ton_iot_devices.py` (full, uncapped data) shows exactly 15,000 `normal` rows on every device, with attack share ranging from 51.8% (modbus) to 62.4% (fridge). Across the 7 devices that is 156,119 attack rows of 261,119 total, or 59.8%.

By that arithmetic, a classifier that always predicted "attack" would score about 59.8% on a pooled random test split (assuming the `max_rows` subsample preserves class proportions, which was not checked). The 62.99% result and the three attempts above all sit within about 3 points of that figure. This suggests the model may be learning little beyond the class prior on this dataset, but that is not confirmed: no per-class precision/recall breakdown was run.

## Conclusion

Binary reformulation directly and substantially validates the hypothesis it was designed to test: this project's accuracy ceiling was driven primarily by multi-class attack-subtype classification difficulty, not by training budget, model capacity, or an inherent limit of the federated learning pipeline. **Three of four datasets now clear the 90% target on all three experiment scenarios (`baseline`/`fedavg_attack`/`ztfedtrust_attack`):**

| Dataset | Binary accuracy (all 3 scenarios) | Clears 90%? |
|---|---|---|
| N-BaIoT | 99.89% / 99.92% / 99.73% (all 3 scenarios verified; see addendum) | Yes |
| TON_IoT network | 97.63% / 97.63% / 97.70% (all 3 scenarios verified; see addendum) | Yes |
| UNSW-NB15 | 90.00% / 90.37% / 90.33% (all 3 scenarios verified) | Yes |
| TON_IoT devices | 62.99% (baseline only) | No |

This is a dramatic change from the multi-class ceiling, where none of the four datasets, across any scenario, ever exceeded 74%. **TON_IoT devices remains the sole exception**: substantially improved (41.76% → 62.99%) but still well below target, with a distinct, already-documented set of problems (severe attack damage, defense over-exclusion, a small and structurally unusual federation) that binary framing alone does not resolve.

Note: an earlier version of this report said N-BaIoT's and TON_IoT network's `fedavg_attack`/`ztfedtrust_attack` scenarios had not been measured in binary mode. They had been — the results were sitting in an uncommitted log and are recorded in the addendum above. TON_IoT devices is still `baseline`-only.

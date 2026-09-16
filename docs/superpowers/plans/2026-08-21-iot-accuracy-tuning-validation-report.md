# IoT Model Accuracy Tuning — Validation Report

## N-BaIoT

**Pre-flight:** `python -m pytest tests/ -q` → 278 passed, 1 pre-existing scipy warning, 42.91s (confirms Tasks 1-4 landed cleanly before empirical tuning started).

### Attempt sequence (tuned against `baseline`)

| Attempt | Config | `baseline` accuracy |
|---|---|---|
| 1 | `num_rounds=20, local_epochs=2, max_rows=15000, hidden_sizes=[128,64]` (Task 4's starting config) | **67.09%** |
| 2 | + `local_epochs: 2 → 5` | **69.73%** |
| 3 | + `num_rounds: 20 → 50` | **72.30%** |
| 4 | + `hidden_sizes: [128,64] → [256,128]` | **74.17%** |

Commands run (identical shape for each attempt, only the config file differed):

```python
from simulation.run import load_config
from evaluation.experiment_runner import ExperimentRunner

cfg = load_config('configs/iot/nbaiot.yaml')
result = ExperimentRunner(cfg).run('baseline')
acc = result.history[-1]['test_accuracy'] * 100
print(f'nbaiot baseline: accuracy={acc:.2f}%')
```

Raw output for each attempt (PerformanceWarning lines are pre-existing pandas fragmentation warnings from `data/iot_datasets.py`, unrelated to this change):

```
Attempt 1: nbaiot baseline attempt1(rounds=20,epochs=2,hidden=128-64): accuracy=67.09%
Attempt 2: nbaiot baseline attempt2(rounds=20,epochs=5,hidden=128-64): accuracy=69.73%
Attempt 3: nbaiot baseline attempt3(rounds=50,epochs=5,hidden=128-64): accuracy=72.30%
Attempt 4: nbaiot baseline attempt4(rounds=50,epochs=5,hidden=256-128): accuracy=74.17%
```

The 4-attempt budget was exhausted without `baseline` clearing 85%. Per the plan, tuning stopped here (no 5th attempt, no learning-rate substitution was needed since all three ordered adjustments were tried in sequence and each produced a measurable, if diminishing, improvement) and the final (attempt 4) config was locked for the one-time three-scenario verification below.

### Final verification (attempt 4's config, run once per scenario)

```python
from simulation.run import load_config
from evaluation.experiment_runner import ExperimentRunner

cfg = load_config('configs/iot/nbaiot.yaml')
for mode in ('baseline', 'fedavg_attack', 'ztfedtrust_attack'):
    result = ExperimentRunner(cfg).run(mode)
    acc = result.history[-1]['test_accuracy'] * 100
    print(f'nbaiot {mode}: accuracy={acc:.2f}%')
```

Output:

```
nbaiot baseline: accuracy=74.17%
nbaiot fedavg_attack: accuracy=73.54%
nbaiot ztfedtrust_attack: accuracy=82.88%
```

| Scenario | Accuracy | Cleared 85%? |
|---|---|---|
| `baseline` | 74.17% | No |
| `fedavg_attack` | 73.54% | No |
| `ztfedtrust_attack` | 82.88% | No (closest of the three) |

### Honest assessment

**85% was not reached on any of the three scenarios for N-BaIoT**, even after the full 4-attempt tuning budget (roughly 4x the rounds, 2.5x the local epochs, 2x the max row cap, and 2x the hidden-layer width of the original config). This is reported plainly, not softened.

**Technical explanation for the shortfall:** the improvement across attempts was real but clearly diminishing — +2.64pts, +2.57pts, +1.87pts per attempt — rather than accelerating, which is the signature of approaching a genuine capacity/feature ceiling rather than still being underfit. N-BaIoT is a multi-class classification task (benign traffic plus multiple Mirai/Gafgyt attack-family subtypes) built from real captured botnet traffic; several of those subtypes are known to produce very similar flow-level statistics to each other and, in some cases, to certain benign traffic patterns, which limits how separable the classes are for *any* model using only these engineered features — more training budget or a wider (but still shallow, 2-hidden-layer) MLP can only close part of that gap. A materially different architecture (e.g. more depth, different feature engineering, or a fundamentally different model family) would likely be needed to close the remaining gap to 85%, which is out of this plan's scope (see the design spec's "Explicitly out of scope" section on model-family changes).

**A notable, only-partially-explained finding:** `ztfedtrust_attack` (82.88%) scored substantially higher than both `baseline` (74.17%, no attackers at all) and `fedavg_attack` (73.54%, attackers present, undefended) — an 8-9 point gap over the other two scenarios, despite the *training recipe itself* being identical across all three (see this plan's Global Constraints; only client selection/aggregation weighting differs). A plausible explanation is that trust-based client selection, by construction, tends to exclude or down-weight clients whose updates are least consistent with their peers' — which on a natural, heterogeneous, 9-device federation like N-BaIoT could be filtering out clients with noisier or more idiosyncratic local data as a side effect of filtering for "trustworthy," independent of whether those clients are actually malicious. This would improve accuracy incidentally, not because the defense is doing anything about the classification task per se. This is a plausible hypothesis based on this run's numbers, not a confirmed causal finding — it would need a dedicated, controlled follow-up (e.g. comparing which specific clients get excluded across rounds) to verify, which is out of scope for this accuracy-tuning task.

## TON_IoT Devices

**Pre-flight:** `python -m pytest tests/ -q` → 278 passed, 1 pre-existing scipy warning (confirms Tasks 1-4, and N-BaIoT's tuning above, left the suite green before this dataset's tuning started).

### Attempt sequence (tuned against `baseline`)

| Attempt | Config | `baseline` accuracy |
|---|---|---|
| 1 | `num_rounds=20, local_epochs=2, max_rows=15000, hidden_sizes=[128,64]` (Task 4's starting config) | **41.76%** |
| 2 | + `local_epochs: 2 → 5` | **39.13%** |
| 3 | + `num_rounds: 20 → 50` | **39.02%** |
| 4 | + `hidden_sizes: [128,64] → [256,128]` | **41.96%** |

Same command shape as N-BaIoT's, run against `configs/iot/ton_iot_devices.yaml`. Raw output for each attempt:

```
Attempt 1: ton_iot_devices baseline attempt1(rounds=20,epochs=2,hidden=128-64): accuracy=41.76%
Attempt 2: ton_iot_devices baseline attempt2(rounds=20,epochs=5,hidden=128-64): accuracy=39.13%
Attempt 3: ton_iot_devices baseline attempt3(rounds=50,epochs=5,hidden=128-64): accuracy=39.02%
Attempt 4: ton_iot_devices baseline attempt4(rounds=50,epochs=5,hidden=256-128): accuracy=41.96%
```

Unlike N-BaIoT, this sequence shows **no consistent improvement at all** — it's flat and noisy (41.76% → 39.13% → 39.02% → 41.96%), moving *down* after the first two adjustments before recovering to roughly its starting point on the fourth. The 4-attempt budget was exhausted; per the plan, tuning stopped here and the final (attempt 4) config was locked for the one-time three-scenario verification below.

### Final verification (attempt 4's config, run once per scenario)

Same command shape as N-BaIoT's, run against `configs/iot/ton_iot_devices.yaml`. Output:

```
ton_iot_devices baseline: accuracy=41.96%
ton_iot_devices fedavg_attack: accuracy=24.73%
ton_iot_devices ztfedtrust_attack: accuracy=14.31%
```

| Scenario | Accuracy | Cleared 85%? |
|---|---|---|
| `baseline` | 41.96% | No |
| `fedavg_attack` | 24.73% | No |
| `ztfedtrust_attack` | 14.31% | No |

### Honest assessment

**85% was not reached on any of the three scenarios, and this dataset is a clear outlier — not just short of the target, but actively degrading under attack and further degrading under defense.** `fedavg_attack` (24.73%) dropped 17.23pts below `baseline` (41.96%) — the `real_traffic` attack is doing real, substantial damage to this federation even before any defense is considered. `ztfedtrust_attack` (14.31%) is 10.42pts *below even the undefended attacked baseline* — the Zero Trust defense is actively making outcomes worse here, not better.

**Technical explanation:** the flat, non-improving tuning curve (no attempt beat the first by more than half a point) is the signature of a genuine ceiling unrelated to training budget — more rounds, more epochs, and a wider model changed nothing meaningful. The `ztfedtrust_attack` collapse has a concrete, observable cause in the round-by-round training log: trust-based selection progressively narrows participation as rounds proceed, settling into selecting only 2 of the federation's 7 clients per round by the later rounds (`Eligible: 3, Selected: 2` becomes the steady state from roughly round 26 onward). With only 2 of 7 real, heterogeneous devices contributing to most rounds, the global model is starved of the data diversity it needs, and per-round accuracy becomes highly unstable (individual rounds swing between roughly 10% and 40% with no clear trend). This points to the adaptive trust threshold being poorly calibrated for a small (7-client) natural-device federation — a plausible, distinct problem from the accuracy-tuning question this plan set out to answer, and a candidate for its own dedicated follow-up investigation rather than something more training budget could fix.

## UNSW-NB15

**Pre-flight:** `python -m pytest tests/ -q` → 278 passed, 1 pre-existing scipy warning (confirms the suite was still green before this dataset's tuning started).

### Attempt sequence (tuned against `baseline`)

| Attempt | Config | `baseline` accuracy |
|---|---|---|
| 1 | `num_rounds=20, local_epochs=2, max_rows=15000, hidden_sizes=[128,64]` (Task 4's starting config) | **62.87%** |
| 2 | + `local_epochs: 2 → 5` | **63.30%** |
| 3 | + `num_rounds: 20 → 50` | **60.27%** |
| 4 | + `hidden_sizes: [128,64] → [256,128]` | **66.57%** |

Same command shape as N-BaIoT's, run against `configs/iot/unsw_nb15.yaml`. Raw output for each attempt:

```
Attempt 1: unsw_nb15 baseline attempt1(rounds=20,epochs=2,hidden=128-64): accuracy=62.87%
Attempt 2: unsw_nb15 baseline attempt2(rounds=20,epochs=5,hidden=128-64): accuracy=63.30%
Attempt 3: unsw_nb15 baseline attempt3(rounds=50,epochs=5,hidden=128-64): accuracy=60.27%
Attempt 4: unsw_nb15 baseline attempt4(rounds=50,epochs=5,hidden=256-128): accuracy=66.57%
```

Like TON_IoT devices, this sequence is noisy rather than steadily improving (a dip at attempt 3 before recovering at attempt 4), though attempt 4 does end up the best of the four. The 4-attempt budget was exhausted; per the plan, tuning stopped here and the final (attempt 4) config was locked for the one-time three-scenario verification below.

### Final verification (attempt 4's config, run once per scenario)

Same command shape as N-BaIoT's, run against `configs/iot/unsw_nb15.yaml`. Output:

```
unsw_nb15 baseline: accuracy=66.57%
unsw_nb15 fedavg_attack: accuracy=65.07%
unsw_nb15 ztfedtrust_attack: accuracy=64.60%
```

| Scenario | Accuracy | Cleared 85%? |
|---|---|---|
| `baseline` | 66.57% | No |
| `fedavg_attack` | 65.07% | No |
| `ztfedtrust_attack` | 64.60% | No |

### Honest assessment

**85% was not reached on any of the three scenarios.** Unlike N-BaIoT, none of the three scenarios here are close to the target, and unlike either N-BaIoT or TON_IoT devices, the three scenarios are all clustered within about 2 points of each other (64.60%–66.57%) rather than showing a large defended-vs-undefended gap in either direction.

**Technical explanation:** this is a materially different, and notably *worse*, result than what this exact dataset showed earlier in this project's history under the much smaller original training budget (5 rounds, 1 local epoch), where `ztfedtrust_attack` beat undefended FedAvg by a wide margin (see `docs/superpowers/plans/2026-08-17-peer-cluster-trust-validation-report.md`'s UNSW-NB15 figures). At this plan's larger budget, that defended-vs-undefended gap has essentially disappeared — `ztfedtrust_attack` (64.60%) is now *slightly below* `baseline` (66.57%), not clearly above it. This is a real, honest finding worth flagging rather than reconciling away: whatever made trust-based selection look strongly beneficial on this dataset under a tiny training budget did not carry over once training budget increased, which suggests the earlier result may have been partly an artifact of how little training either scenario got, rather than a robust property of the defense mechanism on this dataset. This project's practice throughout has been to report such discrepancies plainly rather than average over them, so it's recorded here as an open question rather than resolved.

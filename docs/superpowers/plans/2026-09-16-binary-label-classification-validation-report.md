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

## Conclusion

Binary reformulation directly and substantially validates the hypothesis it was designed to test: this project's accuracy ceiling was driven by multi-class attack-subtype classification difficulty, not by training budget, model capacity, or an inherent limit of the federated learning pipeline. One dataset (N-BaIoT) now clears the 90% target with room to spare; the other (UNSW-NB15) falls just short but is close enough that a further bounded tuning pass (not part of this task, per the design's explicit scope) is a reasonable next step, rather than continuing to work on the multi-class formulation.

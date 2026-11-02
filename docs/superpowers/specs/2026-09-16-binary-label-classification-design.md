# Binary Benign/Attack Label Classification — Design

**Date:** 2026-09-16
**Status:** Approved by user, pending implementation plan

## Purpose

The IoT accuracy-tuning work (see `2026-08-21-iot-accuracy-tuning-design.md`) raised the training budget, added class-weighted loss, and made model capacity configurable, then ran a bounded 4-attempt tuning loop per dataset. None of the three datasets tested so far reached even 85% test accuracy on any of the three experiment scenarios (`baseline`, `fedavg_attack`, `ztfedtrust_attack`):

- N-BaIoT: 74.17% / 73.54% / 82.88% (best result overall)
- UNSW-NB15: 66.57% / 65.07% / 64.60%
- TON_IoT devices: 41.96% / 24.73% / 14.31% (the defense actively hurts this dataset)

The pattern across attempts (diminishing or flat/noisy returns as training budget increased) indicates the bottleneck is not underfitting — more rounds/epochs/width will not close this gap. All four datasets are currently classified into their full multi-class attack-subtype vocabulary (e.g. N-BaIoT's benign plus multiple Mirai/Gafgyt subtypes; UNSW-NB15's 10 attack categories), which is a substantially harder task than detecting whether an attack occurred at all. This project adds an opt-in binary (benign vs. attack) classification mode to test whether that's the real ceiling, without abandoning the project's real IoT-security dataset scope.

## Decisions made during design

1. **Opt-in toggle, default off, existing multi-class path untouched.** A new `DataConfig.binary_labels: bool = False` field. Every existing config (both already-tuned multi-class IoT configs and the two non-IoT vision configs) is unaffected unless it explicitly sets `data.binary_labels: true`. This preserves every number already measured and committed.
2. **Reuse each dataset's existing `benign_label` metadata — no new per-dataset configuration.** `data/iot_datasets.py` already threads a `benign_label` string (`"benign"` for N-BaIoT, `"normal"`/`"Normal"` for the others) through its two shared builder functions, currently used only to build the `real_traffic` attack pool. Binary mode reuses this exact value to decide which rows collapse to `"benign"` (everything else becomes `"attack"`).
3. **Collapse labels before the existing multi-class machinery runs, don't replace that machinery.** When `binary_labels` is true, the dataset's label column is rewritten to exactly two string values (`"benign"` / `"attack"`) right before `fit_label_classes`/`encode_labels` are called. Those functions, and the already-existing `compute_class_weights`, run completely unchanged on top of the collapsed column — no new label-fitting or encoding logic needed.
4. **The canonical collapsed benign label is always the literal string `"benign"`, regardless of the dataset's original label spelling.** After collapsing, callers (specifically `_build_attack_pool`, which filters on `benign_label`) must be given `"benign"` instead of the dataset's original benign-label string, since the original string no longer appears in the collapsed column.
5. **Scoped first to N-BaIoT and UNSW-NB15 only, as new sibling config files, not edits to the existing tuned configs.** `configs/iot/nbaiot_binary.yaml` and `configs/iot/unsw_nb15_binary.yaml` are new files, copies of each dataset's original (pre-tuning) "attempt 1" budget (`num_rounds=20, local_epochs=2, max_rows=15000, hidden_sizes=[128,64]`) with `binary_labels: true` added. TON_IoT devices and TON_IoT network are explicitly out of scope for this round — revisit only if binary mode on the first two datasets looks promising.
6. **Validate with a single fast `baseline` measurement per dataset first, not the full 4-attempt tuning loop.** The question this round is "does binary framing get close to 90% without extra training budget," not "what's the absolute best binary accuracy achievable." If either dataset's `baseline` run is close to or clears 90%, that's the answer; if not, that's reported plainly too, and a further tuning loop (or extending to the other two datasets) is a separate follow-up decision, not committed to upfront.

## Changes

**`data/iot_datasets.py`:**
- Both `_build_partitioned_experiment` and `_build_natural_client_experiment` gain a `binary_labels: bool = False` parameter.
- When `True`, right after the dataset's dataframe(s) are assembled (before `fit_label_classes` is called), the label column's values are rewritten: `"benign"` where the value equals the dataset's original `benign_label`, else `"attack"`.
- `benign_label` used in the rest of that function's body (specifically the `_build_attack_pool` call) is reassigned to the literal `"benign"` when `binary_labels` is `True`, so the attack-pool filter still matches correctly against the now-collapsed column.
- Each per-dataset loader function (`load_nbaiot`, `load_unsw_nb15`, `load_ton_iot_devices`, `load_ton_iot_network`, `load_ciciot2023`) gains a `binary_labels: bool = False` parameter, forwarded to whichever shared builder it calls. `load_iot_experiment` (the dispatcher) forwards it too.

**`simulation/run.py`:**
- `DataConfig` gains `binary_labels: bool = False`.
- `load_config` parses `data.binary_labels` from YAML (defaulting to `False` when absent).
- `load_experiment_data`'s IoT branch passes `config.data.binary_labels` through to `load_iot_experiment`.

**New config files:**
- `configs/iot/nbaiot_binary.yaml` — copy of `nbaiot.yaml`'s pre-tuning state (`num_rounds: 20, local_epochs: 2, max_rows: 15000, hidden_sizes: [128, 64]`, `class_weighted_loss: true`) plus `data.binary_labels: true`.
- `configs/iot/unsw_nb15_binary.yaml` — same pattern for UNSW-NB15's pre-tuning state.

## Testing

- Unit tests appended to `tests/test_iot_datasets.py`: one exercising `_build_partitioned_experiment(..., binary_labels=True)` (reusing that file's existing `_make_partitioned_df()` fixture) asserting `result.class_names == ["attack", "benign"]` and that the label collapse is correct; one exercising `_build_natural_client_experiment(..., binary_labels=True)` (reusing `TestBuildNaturalClientExperiment._device_frames`) with the same assertions, pooled across devices.
- A test appended to `tests/test_config_validation.py` confirming `DataConfig.binary_labels` defaults to `False`.
- The existing `tests/test_iot_configs.py`-style smoke coverage extends naturally to the two new config files (parse + validate).
- The full existing suite (278 tests as of this plan's baseline — confirm current with `pytest tests/ -q` before starting, not assumed) must stay green throughout.
- The 90%-proximity question itself is not a unit test — it's the empirical validation step below.

## Validation plan

1. Run `ExperimentRunner(load_config('configs/iot/nbaiot_binary.yaml')).run('baseline')` and the same for `unsw_nb15_binary.yaml` — one measurement each, no tuning loop.
2. Report both real numbers plainly, whatever they are.
3. If either is close to or clears 90%: decide as a follow-up (not part of this plan) whether to also measure `fedavg_attack`/`ztfedtrust_attack` for that dataset, and/or extend binary mode to TON_IoT devices/network.
4. If neither comes close: report that honestly too — it would mean the accuracy ceiling on these datasets isn't primarily a multi-class-vs-binary artifact, and further options (the TON_IoT-devices over-exclusion bug, or accepting the current honest multi-class results as the paper's actual finding) become the live alternatives again.

## Explicitly out of scope

- TON_IoT devices and TON_IoT network — not touched in this round.
- The 4-attempt training-budget tuning loop — not run for binary mode in this round; only a single fast baseline measurement per dataset.
- The `fedavg_attack`/`ztfedtrust_attack` scenarios — not run in this round; `baseline` only.
- Editing the existing, already-tuned multi-class config files (`nbaiot.yaml`, `unsw_nb15.yaml`) — binary mode uses new, separate config files.
- The TON_IoT-devices trust-selection over-exclusion issue found during multi-class validation — a separate, already-identified follow-up, not part of this design.

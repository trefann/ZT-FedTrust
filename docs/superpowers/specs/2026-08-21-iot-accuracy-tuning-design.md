# IoT Model Accuracy Tuning — Design

**Date:** 2026-08-21
**Status:** Approved by user, pending implementation plan

## Purpose

The four real-IoT-dataset configs added earlier in this project's history
(`configs/iot/nbaiot.yaml`, `ton_iot_devices.yaml`, `unsw_nb15.yaml`,
`ton_iot_network.yaml`) were deliberately scoped for fast iteration during
development: `num_rounds: 5`, `local_epochs: 1`, `max_rows: 5000` per
client. That budget produced usable, honest results for validating the
Zero Trust trust mechanism, but final-round test accuracy currently tops
out at 25-67% across the four datasets — far short of what's needed for
this work to be presented credibly (target: **85%+ test accuracy on all
three experiment scenarios — `baseline`, `fedavg_attack`,
`ztfedtrust_attack` — across all four datasets**).

Two concrete, verified causes, not speculation:

1. **Severe underfitting.** 5 rounds × 1 local epoch × a 5,000-row-per-client
   cap is a tiny training budget for a multi-class tabular classification
   task (N-BaIoT: benign + multiple Mirai/Gafgyt subtypes; UNSW-NB15: 10
   attack categories, etc.).
2. **No class-imbalance handling.** `clients/client.py`'s `train()` uses
   plain, unweighted `nn.CrossEntropyLoss()`. These are naturally
   imbalanced intrusion-detection datasets (a few common classes, several
   rare attack subtypes); unweighted loss caps achievable accuracy well
   below what the (already-normalized) features could support.

Feature normalization (`data/tabular_dataset.py` /
`data/iot_preprocessing.py`'s z-score scaling) is already correct and not a
contributing factor.

## Decisions made during design

1. **Three levers, applied together from the start** (not staged/conditional):
   raise the training budget, add class-weighted loss, and make model
   capacity (hidden layer sizes) a config-level knob so it can be tuned
   per dataset without new model classes. The user explicitly chose to
   commit to all three upfront rather than start minimal and escalate only
   if needed, given no time constraint on this work.
2. **No wall-clock budget.** The user explicitly removed the time
   constraint ("no real limit") — training runs may take as long as needed,
   including overnight/background runs, to reach the accuracy target.
3. **Tune against the `baseline` (clean, no-attack) scenario first, per
   dataset.** All three experiment modes (`baseline`, `fedavg_attack`,
   `ztfedtrust_attack`, defined in `evaluation/experiment_runner.py`)
   share the identical model architecture and local training recipe —
   they differ only in which clients participate and how their updates are
   aggregated (plain average vs. trust-weighted). `baseline` isolates "can
   this model/recipe learn the task at all" from "how much do attackers
   and the defense affect the outcome," so it's the right signal to tune
   against; a recipe that clears 85% on `baseline` should carry over to the
   other two scenarios, which are then verified (not re-tuned) once per
   dataset.
4. **Bounded per-dataset tuning loop, capped at 4 attempts total.** Attempt
   1 is a reasonable "boosted" starting config (higher rounds/epochs/
   max_rows, class-weighted loss on), measured against `baseline` test
   accuracy. If under 85%, attempts 2-4 apply, one at a time, the first
   three items from a fixed ordered adjustment list — more local epochs,
   then more rounds, then wider hidden layers — re-measuring after each.
   "Lower the learning rate" is a fourth candidate adjustment, used to
   replace whichever earlier item doesn't make sense for a given dataset
   (e.g., epochs are already high) so the three-slot budget stays
   productive; it does not add a 5th attempt. Stop after 4 attempts
   regardless of outcome, to keep this an executable, boundable plan task
   rather than an open-ended search.
5. **Honest reporting, no forcing.** For any dataset/scenario still under
   85% after the 4-attempt budget, the validation report states the actual
   number and a technical explanation for the shortfall — matching this
   project's established practice (see the peer-cluster-trust validation
   report's corrected N-BaIoT findings) of reporting unflattering results
   plainly rather than adjusting anything to force a favorable number or
   quietly extending the attempt budget.

## Changes

**`clients/client.py`:** `Client.__init__`/`train()` accept an optional
`class_weights: torch.Tensor | None = None`, passed into
`nn.CrossEntropyLoss(weight=class_weights)` (falls back to today's
unweighted behavior when `None`, so existing callers/tests that don't pass
it are unaffected).

**`models/tabular_mlp.py`:** `TabularMLP.__init__` takes
`hidden_sizes: Tuple[int, ...] = (128, 64)` instead of two hardcoded
`nn.Linear` layers, building the same BatchNorm+ReLU+Dropout stack for
however many sizes are given, so per-dataset tuning can widen/deepen the
network purely via config.

**New helper (file TBD at implementation time — likely
`data/iot_preprocessing.py` or a new small `data/class_weights.py`):**
`compute_class_weights(labels: Sequence[int]) -> torch.Tensor` — inverse-
frequency weighting from the full pre-split label distribution (computed
once per dataset before client partitioning, not per-client, so every
client trains against the same global class-weight vector).

**`simulation/run.py` / `evaluation/experiment_runner.py`:** wire
`compute_class_weights` and the configurable `hidden_sizes` through to
where `Client` and `TabularMLP` get constructed for a run.

**All four `configs/iot/*.yaml`:** raised `num_rounds`, `local_epochs`,
`max_rows`, `clients_per_round`; new `model.hidden_sizes` key; new
`training.class_weighted_loss: true` toggle (default `false`, so the
non-IoT configs — MNIST/CIFAR-10 — are unaffected without needing any
change to their own config files; only the four IoT configs opt in).
Starting values are an implementation-time decision (informed by the
tuning loop itself), not fixed by this spec.

## Testing

- Unit tests for `compute_class_weights` (known label distribution →
  expected inverse-frequency weights; an all-one-class edge case).
- Unit tests for `TabularMLP` with non-default `hidden_sizes` (correct
  layer shapes; forward pass produces the right output shape for varying
  depth/width).
- The full existing suite (`tests/test_iot_datasets.py`,
  `tests/test_attacks.py`, `tests/test_experiment_runner.py`, etc.) must
  stay green — these changes alter *how much* training happens and *how
  it's weighted*, not *what* gets trained on or the trust/attack mechanics
  already validated.
- The 85% accuracy target itself is **not** a unit test — it's empirical,
  per-dataset validation via the bounded tuning loop (Decision 4), reported
  the same way Task 3 of the peer-cluster-trust plan reported its
  real-data findings.

## Validation plan

For each of the four datasets (`nbaiot`, `ton_iot_devices`, `unsw_nb15`,
`ton_iot_network`):

1. Run the bounded tuning loop (Decision 4) against `baseline`.
2. Once `baseline` clears 85% (or the 4-attempt budget is exhausted), lock
   that dataset's final config and run `fedavg_attack` and
   `ztfedtrust_attack` once each with it.
3. Record all three scenarios' final test accuracy, whether each cleared
   85%, and — for anything that didn't — an honest technical explanation
   (e.g., "the trust mechanism's exclusion behavior on this dataset caps
   `ztfedtrust_attack` accuracy independent of model capacity, see
   `docs/superpowers/plans/2026-08-17-peer-cluster-trust-validation-report.md`").

Total: up to 4 (datasets) × (4 tuning attempts + 2 verification runs) = up
to 24 experiment runs, plus this is explicitly allowed to take significant
wall-clock time per the user's "no real limit" decision.

## Explicitly out of scope

- The peer-cluster-relative trust redesign (device-type-based acceptable
  ranges, homogeneous-device support) discussed alongside this work — the
  user explicitly chose to sequence accuracy work first and return to the
  trust redesign afterward as separate, later work.
- Switching the model family (e.g., to random forest) — evaluated and
  rejected during discussion: incompatible with this project's
  weight-averaging FedAvg/TrustFedAvg aggregation and with the trust
  engine's continuous-weight-vector-based metrics
  (`cosine_similarity`/`gradient_norm`/`peer_cluster_similarity` in
  `trust/update_quality.py`), which would all need to be redesigned from
  scratch for a non-differentiable ensemble model.
- Optimizer choice (e.g. switching SGD to Adam) or a learning-rate
  *schedule* (warmup/decay) — a one-time, static lower learning-rate value
  is explicitly in scope as Decision 4's fourth candidate adjustment, but
  swapping optimizers or adding a schedule is not.
- Any change to `trust/`, `attacks/`, or `server/` — this work only
  touches model capacity, training budget, and loss weighting.

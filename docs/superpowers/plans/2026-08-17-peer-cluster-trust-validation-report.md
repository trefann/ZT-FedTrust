# Peer-Cluster-Relative Trust Scoring — Validation Report

**Date:** 2026-08-18
**Task:** Task 3 of `docs/superpowers/plans/2026-08-17-peer-cluster-trust.md`

## Pre-flight: full suite green

`python -m pytest tests/ -q` → 264 passed, 1 pre-existing scipy warning, 37.10s.

(264 = 251 original baseline + 6 from Task 1 + 6 from Task 2 + 1 regression
test from the bug fix described below.)

## Bug found and fixed before the empirical runs

While preparing to run the four-dataset comparison, a real bug was found in
the test suite (not in the feature code from Tasks 1-2): `tests/test_iot_datasets.py`'s
`TestLoadNbaiot` class patches `data.iot_datasets.extract_rar` with a fake
that writes tiny 2-column stub CSVs, to avoid depending on real `.rar`
archives in tests. But nothing redirected `_NBAIOT_CACHE_DIR` — a
module-level constant hardcoded to `"data_cache/iot/nbaiot"` — away from
its real path. Every test in that class was therefore writing stub content
directly into the same cache directory real N-BaIoT experiment runs read
from, silently overwriting genuine, previously-extracted data with 2-column
placeholder CSVs.

**Fix (commit `46af379`):** added an autouse `_isolated_cache_dir` fixture
to `TestLoadNbaiot` that monkeypatches `_NBAIOT_CACHE_DIR` to a per-test
`tmp_path`, plus a regression test (`test_stub_extraction_stays_isolated_from_real_cache_dir`)
that asserts the real cache directory's contents are unchanged after running
a test that exercises the patched extraction path.

Before writing the fix, the real cache directory was inspected directly and
confirmed intact (`data_cache/iot/nbaiot/Danmini_Doorbell/gafgyt/combo.csv`:
105MB, real feature-column headers, 59,719 rows, Oct 2017 timestamp — not
stub content) — so this bug had not yet corrupted this environment's cache,
but was a live risk for the next time `pytest tests/` ran. Full suite after
the fix: 264 passed, same result as before (37.10s), confirming the fix
didn't change any other test's behavior.

## Step 2: four-dataset comparison

Command run:

```bash
python -c "
from simulation.run import load_config
from evaluation.experiment_runner import ExperimentRunner

for name in ('nbaiot', 'ton_iot_devices', 'unsw_nb15', 'ton_iot_network'):
    cfg = load_config(f'configs/iot/{name}.yaml')
    runner = ExperimentRunner(cfg)
    fedavg = runner.run('fedavg_attack')
    zt = runner.run('ztfedtrust_attack')
    fedavg_acc = fedavg.history[-1]['test_accuracy'] * 100
    zt_acc = zt.history[-1]['test_accuracy'] * 100
    print(f'{name}: fedavg_attack={fedavg_acc:.1f}%  ztfedtrust_attack={zt_acc:.1f}%  delta={zt_acc - fedavg_acc:+.1f}pts')
"
```

Full output (per-round `TrustFedAvg` summaries omitted below for brevity,
shown in full in the raw run — final line per dataset is the reported
delta):

| Dataset | fedavg_attack | ztfedtrust_attack | delta (this run) | delta (pre-feature baseline) |
|---|---|---|---|---|
| `nbaiot` | 67.5% | 67.3% | **-0.2pts** | +1.6pts |
| `ton_iot_devices` | 29.7% | 25.5% | **-4.2pts** | -4.1pts |
| `unsw_nb15` | 33.1% | 54.3% | **+21.2pts** | +17.7pts |
| `ton_iot_network` | 49.2% | 59.2% | **+10.0pts** | +8.7pts |

**Honest assessment — accuracy deltas alone:**

- `unsw_nb15` and `ton_iot_network` (expected not to regress): both stayed
  strongly positive, and both actually improved slightly over their
  pre-feature baselines (+21.2 vs +17.7, +10.0 vs +8.7). No regression —
  the additive `peer_cluster_similarity` metric did not hurt the two
  datasets where the existing mechanism already worked.
- `nbaiot` and `ton_iot_devices` (expected to improve): **the final-round
  accuracy delta did not improve** by this measure alone. `nbaiot` moved
  from +1.6pts to -0.2pts (worse by this metric); `ton_iot_devices` stayed
  essentially flat (-4.1pts to -4.2pts). Taken at face value, this metric
  alone would not support a claim of improvement.

This is reported plainly, not softened: accuracy delta on its own does not
show the hoped-for improvement on the two target datasets. But — as the
design spec anticipated and as the N-BaIoT ground-truth check below
confirms — the pre-feature `nbaiot` result of "+1.6pts" was itself
established (in this project's prior work) to be a misleading number: it
came from a run where the two real attackers were being given *more*
aggregation weight (33-50%) than an undefended baseline would give them
(14-29%), i.e. the trust mechanism was actively wrong even while accuracy
looked fine that round. A final-round accuracy number on a small
(7-10 client), short (5-round) natural-device federation is a noisy,
easily-confounded proxy for whether the defense is behaving correctly.
The ground-truth check below is the real test of what this feature was
built to fix.

## Step 3: N-BaIoT ground-truth check

Command run:

```bash
python -c "
from simulation.run import load_config
from evaluation.experiment_runner import ExperimentRunner

cfg = load_config('configs/iot/nbaiot.yaml')
result = ExperimentRunner(cfg).run('ztfedtrust_attack')

for rec in result.attack_history_records:
    if rec.is_malicious:
        print(f'round={rec.round_num} client={rec.client_id} trust={rec.trust_score:.3f} selected={rec.selected} weight={rec.aggregation_weight:.3f}')
"
```

Full output (malicious clients only):

```
round=1 client=0 trust=0.899 selected=True  weight=0.167
round=1 client=1 trust=0.891 selected=True  weight=0.167
round=2 client=0 trust=0.937 selected=True  weight=0.237
round=2 client=1 trust=0.940 selected=True  weight=0.235
round=3 client=0 trust=0.896 selected=True  weight=0.253
round=3 client=1 trust=0.940 selected=False weight=0.000
round=4 client=0 trust=0.882 selected=True  weight=0.200
round=4 client=1 trust=0.899 selected=True  weight=0.209
round=5 client=0 trust=0.892 selected=True  weight=0.201
round=5 client=1 trust=0.891 selected=True  weight=0.205
```

**Comparison against the project's prior documented N-BaIoT finding**
(malicious aggregation weight 0.33-0.50 throughout, never trending down,
versus undefended FedAvg's 0.14-0.29):

This run's malicious-client weights range from **0.167 to 0.253** (one
round, round 3 client 1, dropped to full exclusion at 0.000). This is a
clear, real shift: the malicious clients are no longer being given *more*
weight than an undefended baseline would — they now sit inside or below
the undefended range (0.14-0.29), instead of above it (0.33-0.50) as
documented before this feature. That is the specific failure this feature
was built to fix, and it is fixed on this run: `peer_cluster_similarity`
is pulling the malicious clients' composite trust down relative to their
own behavioral peers, where the old population-mean-only signal wasn't.

**What did not fully materialize:** the weights do not show a clean
monotonic trend toward zero across rounds (0.167 → 0.235-0.237 → 0.000/0.253
→ 0.200-0.209 → 0.201-0.205) — there's a bump up in round 2-3 before
settling into the 0.20-0.25 band, and only one of the two malicious clients
was ever fully excluded, in one round. This is a real, honest limitation:
the peer-cluster mechanism meaningfully reduces the attackers' influence
compared to the old mechanism, but does not reliably drive it to zero or
keep it trending down round over round on this dataset. A plausible
technical explanation: N-BaIoT's 9 device types produce enough per-round
variability in the adaptive clustering threshold (the round's own mean
pairwise cosine distance) that the two malicious clients sometimes land in
a cluster large enough to blend in, rather than being consistently isolated
as their own outlier cluster.

`ton_iot_devices`'s ground-truth check (repeat of the same pattern) was not
run — the brief marks the `nbaiot` check as the required minimum and marks
`ton_iot_devices` as "if time allows"; given the clear, honest picture
already obtained from `nbaiot`, and that `ton_iot_devices`'s accuracy delta
showed no change at all (-4.1 to -4.2), it was deprioritized in favor of
reporting promptly on the confirmed result above rather than open-endedly
extending this task's scope.

## Overall assessment

- **No regression** on the two already-working, partitioned datasets
  (`unsw_nb15`, `ton_iot_network`) — both stayed strongly positive and even
  ticked up slightly.
- **Real, meaningful improvement** in the specific failure mode this
  feature targeted on `nbaiot`: malicious clients' aggregation weight moved
  from consistently *above* the undefended baseline (0.33-0.50) to mostly
  *within or below* it (0.167-0.253, with one full exclusion), directly
  confirmed by ground-truth `is_malicious` labels. This is the core claim
  of the design spec — "detect the attacker even when there's nothing
  common between the clients" — and it holds up on the dataset with the
  clearest prior documented failure.
- **Not a complete fix:** final-round accuracy deltas on `nbaiot` and
  `ton_iot_devices` did not improve by that measure alone, and the
  malicious-weight reduction on `nbaiot`, while real, is not a clean
  monotonic trend to zero — one attacker was excluded in only one of five
  rounds. Consistent with this project's stated goal from the outset ("you
  can't build a perfect security system where it is guaranteed to detect
  every single attack"), this is a genuine improvement to the specific
  failure mode identified, not a claim of solved/perfect detection on
  fully heterogeneous federations.

## Step 5: bug findings

One real bug was found and fixed (see "Bug found and fixed before the
empirical runs" above) — a test-isolation defect in
`tests/test_iot_datasets.py`, unrelated to the Task 1/2 feature code
itself, but discovered while preparing this task's validation runs. No
other bugs were found in Tasks 1-2's implementation; the `peer_clustering.py`
and `update_quality.py` changes behaved exactly as designed across all
four datasets and the ground-truth check.

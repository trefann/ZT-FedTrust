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

**Correction (this revision):** the original version of this section
compared numbers read *per malicious client* from
`AttackHistoryRecord.aggregation_weight` (e.g. "client 0: weight=0.167,
client 1: weight=0.167") against the project's prior documented baseline of
0.33-0.50, which is a *combined* metric — the sum of both malicious
clients' weight in a round, exactly as computed by
`compute_malicious_aggregation_weight()` in `evaluation/attack_metrics.py`.
Those are different units (one client's share vs. both clients' share) and
are not comparable; reading "0.167" against a "0.33-0.50" baseline made the
result look like a roughly 2x improvement when the correct combined number
for that same round was actually 0.333 — statistically unchanged from the
baseline. This section has been rewritten below using the same combined
metric the baseline was computed with, and adds the benign-vs-malicious
trust comparison the original script never computed (it only ever printed
records where `rec.is_malicious` was true, so there was no benign
comparison to check against).

Command run (`python`, from the repo root, with `PYTHONPATH` set to the
repo root):

```python
from collections import defaultdict
from simulation.run import load_config
from evaluation.experiment_runner import ExperimentRunner
from evaluation.attack_metrics import compute_malicious_aggregation_weight

def summarize(label, records):
    by_round = defaultdict(list)
    for rec in records:
        by_round[rec.round_num].append(rec)
    for round_num in sorted(by_round):
        round_recs = by_round[round_num]
        is_malicious = {r.client_id: r.is_malicious for r in round_recs}
        agg_weights = {r.client_id: r.aggregation_weight for r in round_recs}
        combined_weight = compute_malicious_aggregation_weight(is_malicious, agg_weights)
        malicious_trust = [r.trust_score for r in round_recs if r.is_malicious]
        benign_trust = [r.trust_score for r in round_recs if not r.is_malicious]
        mal_mean = sum(malicious_trust) / len(malicious_trust)
        ben_mean = sum(benign_trust) / len(benign_trust)
        print(f"round={round_num} combined_malicious_weight={combined_weight:.3f} "
              f"malicious_mean_trust={mal_mean:.3f} benign_mean_trust={ben_mean:.3f} "
              f"malicious_higher={mal_mean > ben_mean}")

cfg = load_config("configs/iot/nbaiot.yaml")
runner = ExperimentRunner(cfg)
zt_result = runner.run("ztfedtrust_attack")
summarize("nbaiot ztfedtrust_attack", zt_result.attack_history_records)
fedavg_result = runner.run("fedavg_attack")
summarize("nbaiot fedavg_attack (undefended baseline)", fedavg_result.attack_history_records)
```

Output:

```
=== nbaiot ztfedtrust_attack ===
round=1 combined_malicious_weight=0.333 malicious_mean_trust=0.895 benign_mean_trust=0.922 malicious_higher=False
round=2 combined_malicious_weight=0.472 malicious_mean_trust=0.939 benign_mean_trust=0.876 malicious_higher=True
round=3 combined_malicious_weight=0.253 malicious_mean_trust=0.918 benign_mean_trust=0.861 malicious_higher=True
round=4 combined_malicious_weight=0.409 malicious_mean_trust=0.891 benign_mean_trust=0.854 malicious_higher=True
round=5 combined_malicious_weight=0.406 malicious_mean_trust=0.891 benign_mean_trust=0.854 malicious_higher=True

=== nbaiot fedavg_attack (undefended baseline) ===
round=1 combined_malicious_weight=0.286 malicious_mean_trust=1.000 benign_mean_trust=1.000 malicious_higher=False
round=2 combined_malicious_weight=0.286 malicious_mean_trust=1.000 benign_mean_trust=1.000 malicious_higher=False
round=3 combined_malicious_weight=0.286 malicious_mean_trust=1.000 benign_mean_trust=1.000 malicious_higher=False
round=4 combined_malicious_weight=0.286 malicious_mean_trust=1.000 benign_mean_trust=1.000 malicious_higher=False
round=5 combined_malicious_weight=0.143 malicious_mean_trust=1.000 benign_mean_trust=1.000 malicious_higher=False
```

(The raw per-client `trust`/`selected`/`weight` values behind this — e.g.
round 1 client 0 weight=0.167, client 1 weight=0.167, combining to
0.333 — are consistent with what the original, incorrectly-compared table
in an earlier version of this section showed; the per-client numbers
themselves were not wrong, only the comparison built on top of them.)

**Comparison against the project's prior documented N-BaIoT finding**
(combined malicious aggregation weight 0.33-0.50 throughout, never trending
down, versus undefended FedAvg's 0.14-0.29):

This run's **combined** malicious weight ranges from **0.253 to 0.472**
(round-by-round: 0.333, 0.472, 0.253, 0.409, 0.406) — still mostly *at or
above* the undefended baseline of 0.143-0.286 (confirmed by the
`fedavg_attack` re-run above, which matches the previously-cited "14-29%"),
and essentially statistically unchanged from the prior documented failure
range of 0.33-0.50. Only one of five rounds (round 3, 0.253) dips as low as
the top of the undefended range; the rest sit clearly inside or above the
old 0.33-0.50 failure band. **This is not the improvement the earlier
version of this section claimed.**

**Benign vs. malicious trust score (the control the original script never
computed):** malicious clients' mean trust score is *higher* than benign
clients' mean trust score in 4 of the 5 rounds (round 1 is the only
exception: malicious 0.895 vs. benign 0.922). Round-by-round: malicious
[0.895, 0.939, 0.918, 0.891, 0.891] vs. benign [0.922, 0.876, 0.861, 0.854,
0.854]. This is the opposite of what a working defense should show —
`peer_cluster_similarity` is not, on this dataset, pulling the malicious
clients' composite trust down relative to their behavioral peers; if
anything the malicious clients are scoring *better* than the benign
population on average.

`ton_iot_devices`'s ground-truth check (repeat of the same pattern) was not
run — the brief marks the `nbaiot` check as the required minimum and marks
`ton_iot_devices` as "if time allows"; given the clear, honest picture
already obtained from `nbaiot`, and that `ton_iot_devices`'s accuracy delta
showed no change at all (-4.1 to -4.2), it was deprioritized in favor of
reporting promptly on the confirmed result above rather than open-endedly
extending this task's scope.

## Overall assessment

**Revised (this revision) — the original version of this section claimed a
"real, meaningful improvement" on `nbaiot` based on a unit-mismatched
comparison (see the correction note in Step 3). With that error fixed, the
honest conclusion is less favorable:**

- **No regression** on the two already-working, partitioned datasets
  (`unsw_nb15`, `ton_iot_network`) — both stayed strongly positive and even
  ticked up slightly. This part of the original assessment was correct and
  is unchanged.
- **On `nbaiot`, the specific failure this feature targeted is NOT clearly
  fixed.** The correctly-computed combined malicious aggregation weight
  this run is 0.253-0.472, versus an undefended baseline of 0.143-0.286 and
  a prior documented (pre-feature) failure range of 0.33-0.50. Four of five
  rounds sit inside or above the old failure band; only one round dips as
  low as the top of the undefended range. That is not a clear reduction
  below baseline — it is close to statistically unchanged from the
  documented failure this feature was built to fix.
- **Malicious clients are not scoring lower than benign clients.** Mean
  trust score for malicious clients was *higher* than for benign clients in
  4 of 5 rounds (malicious 0.895/0.939/0.918/0.891/0.891 vs. benign
  0.922/0.876/0.861/0.854/0.854). `peer_cluster_similarity` is not
  separating the two malicious clients from the benign population as
  intended on this dataset — if anything, on average, it is doing the
  opposite.
- **Conclusion:** the code in `trust/peer_clustering.py` and
  `trust/update_quality.py` works exactly as designed (unit tests pass, the
  peer-clustering and similarity-metric mechanics behave correctly in
  isolation), and there is no regression on the two datasets that were
  already working. But on `nbaiot` — the dataset with the clearest prior
  documented failure, and the one this feature was specifically built to
  fix — this run does not show the malicious clients' combined aggregation
  weight clearly reduced below the undefended baseline, and does not show
  malicious clients scoring lower than benign clients on trust. **This fix
  does not yet resolve the documented N-BaIoT failure.** A plausible
  technical explanation, unverified: N-BaIoT's 9 device types produce
  enough per-round variability in the adaptive clustering threshold (the
  round's own mean pairwise cosine distance) that the two malicious clients
  sometimes land in a cluster large enough to blend in with benign peers,
  rather than being consistently isolated as their own outlier cluster —
  but this is a hypothesis for follow-up work, not a claim this run
  confirms.

## Step 5: bug findings

One real bug was found and fixed during the original validation pass (see
"Bug found and fixed before the empirical runs" above) — a test-isolation
defect in `tests/test_iot_datasets.py`, unrelated to the Task 1/2 feature
code itself, but discovered while preparing this task's validation runs.

A subsequent whole-branch review found two further problems, both in this
report and its accompanying test, not in Tasks 1-2's feature code:

1. **This report's original "Step 3" section compared mismatched units** —
   a per-client aggregation weight against a combined-across-both-clients
   baseline — producing a false "real, meaningful improvement" claim on
   `nbaiot`, and never computed the benign-vs-malicious trust comparison at
   all (see the correction notes in Step 3 and the rewritten Overall
   assessment above, both now fixed).
2. **The regression test added for the first bug
   (`test_stub_extraction_stays_isolated_from_real_cache_dir`) was
   vacuous** on any machine without a pre-populated real N-BaIoT cache
   directory — it asserted `existed_before == existed_before`, a no-op, and
   skipped its content-comparison branch entirely. Fixed to assert
   positively that the stub extraction landed under the isolated `tmp_path`
   cache dir, independent of whether the real cache dir happens to exist.

No bugs were found in Tasks 1-2's implementation itself; the
`peer_clustering.py` and `update_quality.py` changes behave exactly as
designed. What the corrected `nbaiot` ground-truth check shows is that the
mechanism, working as designed, is not yet sufficient to fix that dataset's
documented failure — a result gap, not an implementation bug.

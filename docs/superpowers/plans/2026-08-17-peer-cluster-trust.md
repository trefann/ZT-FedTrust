# Peer-Cluster-Relative Trust Scoring Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a peer-cluster-relative trust signal so the Zero Trust engine can detect a misbehaving client by comparing it to others *similar to it*, not just to the whole population's average — fixing a real, evidence-backed failure on genuinely heterogeneous real-device federations (N-BaIoT, TON_IoT devices).

**Architecture:** A new standalone module, `trust/peer_clustering.py`, groups this round's participating clients by the similarity of their own update vectors (hierarchical clustering, cosine distance, an adaptive per-round threshold — no device labels, no fixed cluster count). `trust/update_quality.py` gains one new metric, `peer_cluster_similarity`, added alongside its three existing metrics: cosine similarity between a client's update and its own cluster's mean update, instead of the whole round's mean. Both existing metrics (`cosine_similarity`, `gradient_norm`) and the composite trust engine are untouched — this is purely additive.

**Tech Stack:** Python, PyTorch, `scipy` (already a project dependency — `scipy.cluster.hierarchy`, `scipy.spatial.distance`), pytest.

## Global Constraints

- No new pip dependencies — `scipy` is already declared in `requirements.txt` and used elsewhere (`evaluation/statistical_analysis.py`).
- `trust/update_quality.py`'s existing `cosine_similarity` and `gradient_norm` metrics, and every other trust dimension (`identity`, `integrity`, `behaviour`, `reputation`), are unmodified — this plan only adds one new metric to one file plus one new standalone module.
- `trust/peer_clustering.py` must never read `is_malicious`, client identity, or any declared/self-reported metadata — it only ever sees update vectors, matching this project's structural ground-truth-isolation rule (documented in `attacks/base_attack.py` and `clients/malicious_client.py`).
- Metric weight values in `update_quality.py`'s `metric_weights` dict are relative — the existing code already normalizes by their sum (`total_weight = sum(self.metric_weights.values())`), so exact numbers aren't fragile, but must match: `loss_improvement: 0.30`, `cosine_similarity: 0.20`, `peer_cluster_similarity: 0.30`, `gradient_norm: 0.20`.
- Clients with no meaningful peer group to compare against (alone in their own cluster, or fewer than 3 participating clients that round) get the neutral score `0.5` for `peer_cluster_similarity` — the same convention this file's existing metrics already use (`_cosine_similarity_score`'s zero-norm case, `_gradient_norm_score`'s single-client case).
- Existing tests in `tests/test_trust.py` covering `UpdateQualityEvaluator` must keep passing unmodified.

---

## File Structure

**New files:**
- `trust/peer_clustering.py` — `cluster_clients(client_vectors, min_clients=3) -> Dict[int, int]`, the standalone clustering function.
- `tests/test_peer_clustering.py` — tests for the clustering module in isolation.

**Modified files:**
- `trust/update_quality.py` — new `peer_cluster_similarity` metric, rebalanced `metric_weights`, two new private helpers (`_compute_cluster_means`, `_peer_cluster_similarity_score`).
- `tests/test_trust.py` — new tests for the new metric, appended to the existing "Update quality" section.

---

### Task 1: `trust/peer_clustering.py`

**Files:**
- Create: `trust/peer_clustering.py`
- Test: `tests/test_peer_clustering.py`

**Interfaces:**
- Consumes: nothing new — takes plain `Dict[int, torch.Tensor]` (the same flattened update vectors `trust/update_quality.py`'s `_flatten_state_dict` already produces).
- Produces: `cluster_clients(client_vectors: Dict[int, torch.Tensor], min_clients: int = 3) -> Dict[int, int]` — maps every input client ID to a cluster ID. Task 2 depends on this exact signature.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_peer_clustering.py
"""Tests for peer-cluster-relative grouping of client update vectors.

Clustering is behavior-derived (from the update vectors themselves), never
from a declared/self-reported client attribute — see the module docstring
in trust/peer_clustering.py for why.
"""

import torch

from trust.peer_clustering import cluster_clients


def test_two_distinct_groups_of_aligned_vectors_form_two_clusters():
    vectors = {
        0: torch.tensor([1.0, 0.0, 0.0]),
        1: torch.tensor([0.9, 0.1, 0.0]),
        2: torch.tensor([-1.0, 0.0, 0.0]),
        3: torch.tensor([-0.9, -0.1, 0.0]),
    }
    clusters = cluster_clients(vectors)
    assert clusters[0] == clusters[1]
    assert clusters[2] == clusters[3]
    assert clusters[0] != clusters[2]


def test_returns_a_cluster_id_for_every_input_client():
    vectors = {i: torch.randn(5) for i in range(6)}
    clusters = cluster_clients(vectors)
    assert set(clusters.keys()) == set(vectors.keys())


def test_identical_vectors_all_land_in_one_cluster():
    vectors = {i: torch.tensor([1.0, 2.0, 3.0]) for i in range(5)}
    clusters = cluster_clients(vectors)
    assert len(set(clusters.values())) == 1


def test_fewer_than_min_clients_get_distinct_singleton_clusters():
    vectors = {0: torch.tensor([1.0, 0.0]), 1: torch.tensor([1.0, 0.0])}
    clusters = cluster_clients(vectors, min_clients=3)
    assert clusters[0] != clusters[1]


def test_zero_vector_does_not_crash_clustering():
    vectors = {
        0: torch.zeros(4),
        1: torch.tensor([1.0, 0.0, 0.0, 0.0]),
        2: torch.tensor([0.9, 0.1, 0.0, 0.0]),
    }
    clusters = cluster_clients(vectors)
    assert set(clusters.keys()) == {0, 1, 2}


def test_single_client_gets_its_own_cluster():
    vectors = {0: torch.tensor([1.0, 0.0, 0.0])}
    clusters = cluster_clients(vectors)
    assert clusters == {0: 0}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_peer_clustering.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'trust.peer_clustering'`

- [ ] **Step 3: Implement `trust/peer_clustering.py`**

```python
"""Behavior-derived peer grouping for the update-quality trust dimension.

Groups this round's participating clients by the similarity of their own
update vectors — never by a declared or self-reported client attribute
(e.g. "device type"). This is deliberate: a declared label could be
spoofed by an attacker, and behavior-derived grouping works even in a
federation where no such label exists at all, which is what makes it
useful specifically on genuinely heterogeneous real-device federations
(e.g. N-BaIoT, TON_IoT devices) where a single population-wide baseline
stops being meaningful (see `docs/superpowers/specs/2026-08-17-peer-cluster-trust-design.md`).

Only ever reads update vectors — no client identity, no `is_malicious`,
no declared metadata — matching this project's structural ground-truth
isolation rule used throughout `attacks/` and `clients/malicious_client.py`.
"""

from __future__ import annotations

from typing import Dict

import numpy as np
import torch
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import pdist


def cluster_clients(client_vectors: Dict[int, torch.Tensor], min_clients: int = 3) -> Dict[int, int]:
    """Group clients by update-vector similarity via hierarchical clustering.

    Uses cosine distance and average linkage. The merge threshold is
    adaptive — the round's own mean pairwise distance — rather than a
    fixed constant, so it doesn't need per-dataset tuning: on a
    homogeneous round (every update points a similar direction), this
    naturally collapses toward one cluster containing everyone, the same
    way this project's trust-selection threshold is already
    self-calibrating (`mean - 0.5*stdev` of the round's own trust
    distribution) rather than a hardcoded number.

    With fewer than `min_clients` participants there isn't enough
    pairwise-distance signal to calibrate a threshold from (a single
    distance value isn't a distribution), so every client is placed in
    its own singleton cluster — callers that treat a singleton cluster as
    "no peers available" (see `update_quality.py`'s
    `_peer_cluster_similarity_score`) then correctly fall back to a
    neutral score for this case too, with no separate special-casing needed.
    """
    client_ids = sorted(client_vectors)
    if len(client_ids) < min_clients:
        return {client_id: index for index, client_id in enumerate(client_ids)}

    matrix = np.stack([client_vectors[client_id].detach().cpu().numpy() for client_id in client_ids])
    distances = pdist(matrix, metric="cosine")
    # A zero-vector client makes cosine distance NaN (division by a zero
    # norm) rather than raise — treat it as maximally distant (cosine
    # distance's own max value) instead of letting NaN propagate into the
    # linkage/threshold math.
    distances = np.nan_to_num(distances, nan=2.0)

    linkage_matrix = linkage(distances, method="average")
    threshold = float(distances.mean())
    cluster_ids = fcluster(linkage_matrix, t=threshold, criterion="distance")

    return {client_id: int(cluster_id) for client_id, cluster_id in zip(client_ids, cluster_ids)}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_peer_clustering.py -v`
Expected: PASS (6 tests)

- [ ] **Step 5: Run the full suite to confirm no regressions**

Run: `pytest tests/ -q`
Expected: all existing tests still pass, plus the 6 new ones (baseline before this task: 251 passed — confirm this with `git log`/`git stash` if the number looks different, don't assume 251 is still accurate without checking)

- [ ] **Step 6: Commit**

```bash
git add trust/peer_clustering.py tests/test_peer_clustering.py
git commit -m "feat: add behavior-derived peer clustering for trust scoring"
```

---

### Task 2: Add `peer_cluster_similarity` to `trust/update_quality.py`

**Files:**
- Modify: `trust/update_quality.py`
- Modify: `tests/test_trust.py` (append to the existing "Update quality" section, after `test_update_quality_rewards_loss_improvement_across_rounds`)

**Interfaces:**
- Consumes: `cluster_clients(client_vectors: Dict[int, torch.Tensor], min_clients: int = 3) -> Dict[int, int]` from Task 1's `trust/peer_clustering.py`.
- Produces: `UpdateQualityEvaluator.evaluate_round(...)` return value is unchanged in shape (`Dict[int, float]`, one combined score per client) — this task only changes what feeds into that combination. Also produces two new static/private methods other tasks don't need but this task's own tests exercise directly: `UpdateQualityEvaluator._peer_cluster_similarity_score(client_vector: torch.Tensor, cluster_mean_vector: torch.Tensor, cluster_size: int) -> float` and `UpdateQualityEvaluator._compute_cluster_means(client_vectors, sample_counts, cluster_assignments) -> Dict[int, torch.Tensor]`.

- [ ] **Step 1: Write the failing tests**

First, read `trust/update_quality.py` and `tests/test_trust.py`'s existing "Update quality" section (around line 125-163) so your edits land in the right place and match the file's existing style exactly.

Append these to `tests/test_trust.py`, directly after `test_update_quality_rewards_loss_improvement_across_rounds`:

```python
def test_update_quality_metric_weights_include_peer_cluster_similarity():
    evaluator = UpdateQualityEvaluator()
    assert "peer_cluster_similarity" in evaluator.metric_weights


def test_peer_cluster_similarity_score_rewards_alignment_with_cluster_mean():
    score = UpdateQualityEvaluator._peer_cluster_similarity_score(
        client_vector=torch.tensor([1.0, 0.0, 0.0]),
        cluster_mean_vector=torch.tensor([0.9, 0.1, 0.0]),
        cluster_size=2,
    )
    assert score > 0.9


def test_peer_cluster_similarity_score_penalizes_misalignment_with_cluster_mean():
    score = UpdateQualityEvaluator._peer_cluster_similarity_score(
        client_vector=torch.tensor([1.0, 0.0, 0.0]),
        cluster_mean_vector=torch.tensor([-1.0, 0.0, 0.0]),
        cluster_size=2,
    )
    assert score < 0.1


def test_peer_cluster_similarity_score_neutral_when_alone_in_cluster():
    score = UpdateQualityEvaluator._peer_cluster_similarity_score(
        client_vector=torch.tensor([1.0, 0.0, 0.0]),
        cluster_mean_vector=torch.tensor([1.0, 0.0, 0.0]),
        cluster_size=1,
    )
    assert score == 0.5


def test_compute_cluster_means_is_sample_weighted_average_per_cluster():
    client_vectors = {
        0: torch.tensor([1.0, 0.0]),
        1: torch.tensor([3.0, 0.0]),
        2: torch.tensor([0.0, 5.0]),
    }
    sample_counts = {0: 10, 1: 30, 2: 99}
    cluster_assignments = {0: 1, 1: 1, 2: 2}  # clients 0,1 share cluster 1; client 2 alone in cluster 2

    means = UpdateQualityEvaluator._compute_cluster_means(client_vectors, sample_counts, cluster_assignments)

    assert set(means.keys()) == {1, 2}
    # Cluster 1: sample-weighted mean of [1,0]*10 and [3,0]*30 over 40 samples -> [2.5, 0]
    assert torch.allclose(means[1], torch.tensor([2.5, 0.0]))
    assert torch.allclose(means[2], torch.tensor([0.0, 5.0]))


def test_update_quality_scores_still_bounded_with_more_participants():
    """Wiring-level check: evaluate_round produces one bounded score per
    client even with enough participants (>=3) to actually trigger clustering."""
    evaluator = UpdateQualityEvaluator()
    global_state = {"w": torch.zeros(6)}
    client_states = {
        0: {"w": torch.tensor([1.0, 0.0, 0.0, 0.0, 0.0, 0.0]) * 0.1},
        1: {"w": torch.tensor([0.95, 0.05, 0.0, 0.0, 0.0, 0.0]) * 0.1},
        2: {"w": torch.tensor([0.9, 0.1, 0.0, 0.0, 0.0, 0.0]) * 0.1},
        3: {"w": torch.tensor([0.0, 0.0, 0.0, 0.0, -1.0, 0.0]) * 0.1},
        4: {"w": torch.tensor([0.0, 0.0, 0.0, 0.0, -0.95, -0.05]) * 0.1},
    }
    sample_counts = {i: 10 for i in range(5)}
    average_losses = {i: 1.0 for i in range(5)}

    scores = evaluator.evaluate_round(client_states, sample_counts, global_state, average_losses)
    assert set(scores.keys()) == {0, 1, 2, 3, 4}
    for score in scores.values():
        assert 0.0 <= score <= 1.0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_trust.py -v -k "peer_cluster or compute_cluster_means"`
Expected: FAIL — `metric_weights` test fails on a missing key; the `_peer_cluster_similarity_score`/`_compute_cluster_means` tests fail with `AttributeError: type object 'UpdateQualityEvaluator' has no attribute ...`

- [ ] **Step 3: Modify `trust/update_quality.py`**

Add the import near the top, alongside the existing `import torch`:

```python
from trust.peer_clustering import cluster_clients
```

Update the module docstring's bullet list (the one starting "Three metrics are combined into one score") to add a fourth bullet, so the file's own documentation stays accurate:

```python
  - training loss improvement vs. this client's own previous round
  - cosine similarity between the client's update direction and the round's
    sample-weighted mean update direction (a proxy for "consensus" before
    FedAvg aggregation actually runs)
  - cosine similarity between the client's update direction and its own
    behavior-derived peer cluster's mean update direction — catches a
    misbehaving client even when the whole population has no shared
    baseline to deviate from (see `trust/peer_clustering.py`)
  - update magnitude (L2 norm of the pseudo-gradient), min-max normalised
    against the other participating clients this round
```

Change the `metric_weights` dict in `__init__`:

```python
        self.metric_weights: Dict[str, float] = {
            "loss_improvement": 0.30,
            "cosine_similarity": 0.20,
            "peer_cluster_similarity": 0.30,
            "gradient_norm": 0.20,
        }
```

Replace the entire `evaluate_round` method (signature, docstring, and body — this is the complete replacement, not a partial edit) with:

```python
    def evaluate_round(
        self,
        client_states: Dict[int, Dict[str, torch.Tensor]],
        sample_counts: Dict[int, int],
        global_state_before: Dict[str, torch.Tensor],
        average_losses: Dict[int, float],
    ) -> Dict[int, float]:
        """Score every participating client's update quality for one round.

        Args:
            client_states: each client's returned local model weights.
            sample_counts: each client's local dataset size (for the
                sample-weighted mean update direction, matching how FedAvg
                itself will weight these same updates).
            global_state_before: the global model weights clients trained
                from at the start of this round (i.e. before aggregation).
            average_losses: each client's average local training loss this round.
        """
        global_vector = _flatten_state_dict(global_state_before)
        client_vectors = {
            client_id: _flatten_state_dict(state) - global_vector
            for client_id, state in client_states.items()
        }

        total_samples = sum(sample_counts[client_id] for client_id in client_vectors)
        mean_vector = sum(
            (client_vectors[client_id] * (sample_counts[client_id] / total_samples))
            for client_id in client_vectors
        )

        norms = {client_id: float(torch.linalg.norm(vec)) for client_id, vec in client_vectors.items()}
        norm_values = list(norms.values())

        cluster_assignments = cluster_clients(client_vectors)
        cluster_means = self._compute_cluster_means(client_vectors, sample_counts, cluster_assignments)
        cluster_peer_counts: Dict[int, int] = {}
        for cluster_id in cluster_assignments.values():
            cluster_peer_counts[cluster_id] = cluster_peer_counts.get(cluster_id, 0) + 1

        scores: Dict[int, float] = {}
        for client_id, vector in client_vectors.items():
            cluster_id = cluster_assignments[client_id]
            metrics = {
                "loss_improvement": self._loss_improvement_score(client_id, average_losses[client_id]),
                "cosine_similarity": self._cosine_similarity_score(vector, mean_vector),
                "peer_cluster_similarity": self._peer_cluster_similarity_score(
                    vector, cluster_means[cluster_id], cluster_peer_counts[cluster_id]
                ),
                "gradient_norm": self._gradient_norm_score(norms[client_id], norm_values),
            }
            self._previous_loss[client_id] = average_losses[client_id]

            total_weight = sum(self.metric_weights.values())
            combined = sum(metrics[name] * self.metric_weights[name] for name in metrics) / total_weight
            scores[client_id] = float(min(max(combined, 0.0), 1.0))

        return scores
```

Add the two new methods after `_cosine_similarity_score` and before `_gradient_norm_score`:

```python
    @staticmethod
    def _compute_cluster_means(
        client_vectors: Dict[int, torch.Tensor],
        sample_counts: Dict[int, int],
        cluster_assignments: Dict[int, int],
    ) -> Dict[int, torch.Tensor]:
        """Sample-weighted mean update vector per cluster (same weighting
        convention as the round's own global mean_vector above)."""
        cluster_totals: Dict[int, torch.Tensor] = {}
        cluster_sample_totals: Dict[int, int] = {}
        for client_id, vector in client_vectors.items():
            cluster_id = cluster_assignments[client_id]
            weighted = vector * sample_counts[client_id]
            if cluster_id not in cluster_totals:
                cluster_totals[cluster_id] = weighted.clone()
                cluster_sample_totals[cluster_id] = sample_counts[client_id]
            else:
                cluster_totals[cluster_id] = cluster_totals[cluster_id] + weighted
                cluster_sample_totals[cluster_id] += sample_counts[client_id]
        return {
            cluster_id: total / cluster_sample_totals[cluster_id]
            for cluster_id, total in cluster_totals.items()
        }

    @staticmethod
    def _peer_cluster_similarity_score(
        client_vector: torch.Tensor, cluster_mean_vector: torch.Tensor, cluster_size: int
    ) -> float:
        """Score how aligned this client's update is with its own cluster's mean update.

        A client alone in its own cluster (`cluster_size == 1`) has no
        peers to compare against — neutral score, the same convention this
        file's other "no meaningful comparison available" cases already
        use (see `_cosine_similarity_score` and `_gradient_norm_score`).
        """
        if cluster_size < 2:
            return 0.5
        if torch.linalg.norm(client_vector) < 1e-12 or torch.linalg.norm(cluster_mean_vector) < 1e-12:
            return 0.5
        cosine = torch.nn.functional.cosine_similarity(
            client_vector.unsqueeze(0), cluster_mean_vector.unsqueeze(0)
        ).item()
        return float((cosine + 1.0) / 2.0)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_trust.py -v -k "peer_cluster or compute_cluster_means or update_quality"`
Expected: PASS — all new tests plus the 3 pre-existing update-quality tests (`test_update_quality_scores_are_bounded_and_present_for_every_client`, `test_update_quality_single_client_gradient_norm_defaults_to_full_score`, `test_update_quality_rewards_loss_improvement_across_rounds`) still passing unmodified.

- [ ] **Step 5: Run the full suite to confirm no regressions**

Run: `pytest tests/ -q`
Expected: all tests pass, zero regressions, count increased by 5 new tests from Task 1's baseline.

- [ ] **Step 6: Commit**

```bash
git add trust/update_quality.py tests/test_trust.py
git commit -m "feat: add peer-cluster-relative update-quality metric"
```

---

### Task 3: Real-data validation — confirm it helps where it should, doesn't regress where it shouldn't

**Files:**
- No source files modified in this task unless a real bug is found while validating (if so: fix it in whichever file it actually lives in, following the same minimal-scoped-fix approach used elsewhere in this project's history, and explain the bug and fix fully in the report).
- Report: write full findings to whatever report-file path your dispatch prompt specifies (if run standalone, write to `docs/superpowers/plans/2026-08-17-peer-cluster-trust-validation-report.md`).

**Interfaces:**
- Consumes: Task 1 + Task 2's `trust/peer_clustering.py` and updated `trust/update_quality.py` — used transparently by the existing `TrustEngine`/`UpdateQualityEvaluator` wiring already in `simulation/run.py` and `evaluation/experiment_runner.py`. No new integration code needed — this task is purely about running the existing simulator and reporting real numbers.

This task has no unit tests of its own — it's an empirical check using the project's own `ExperimentRunner`, the same tool already used throughout this project's history for exactly this kind of before/after comparison. **Report the actual numbers you get. Do not adjust anything to force a particular outcome — an honest "didn't help as much as hoped" result is a valid, useful outcome for this task, matching this project's established practice of reporting unflattering results rather than hiding them.**

- [ ] **Step 1: Confirm the full test suite is green before starting**

Run: `pytest tests/ -q`
Expected: all tests pass (this confirms Tasks 1-2 landed cleanly before you start empirical validation on top of them)

- [ ] **Step 2: Re-run the four-dataset comparison this feature is meant to affect**

Run this from the repository root (adjust the venv activation line for your shell if needed):

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

Expected direction (not a hard pass/fail gate — report what actually happens): `nbaiot` and `ton_iot_devices` deltas should be less negative (or more positive) than the pre-this-feature baseline deltas already on record (N-BaIoT was +1.6pts but with ground-truth detection failure, per the project's own prior findings; TON_IoT devices was -4.1pts). `unsw_nb15` and `ton_iot_network` deltas should stay comparably positive to their pre-this-feature numbers (+17.7pts and +8.7pts respectively) — a large negative swing on either would indicate a real regression worth investigating before declaring this task done.

- [ ] **Step 3: Ground-truth check — are the real attackers actually being excluded now on the datasets that previously failed?**

Run this for `nbaiot` (repeat the same pattern for `ton_iot_devices` if time allows — `nbaiot`'s check is the required minimum since it's the dataset with the clearest prior documented failure: both real attackers previously kept 33-50% aggregation weight throughout, higher than undefended FedAvg's 14-29%):

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

Compare the printed `weight` values for the malicious clients across rounds against this project's prior documented N-BaIoT finding (malicious aggregation weight 0.33–0.50 throughout, never trending down) — report plainly whether the trend now shows real exclusion (weight trending toward 0 in later rounds) or not.

- [ ] **Step 4: Write the validation report**

Write a report covering: the exact commands run and their full output from Steps 2-3, an honest assessment of whether each of the four datasets moved in the expected direction, whether the N-BaIoT ground-truth check shows genuine improvement in attacker exclusion, and — if the result is mixed or doesn't show the hoped-for improvement — a clear, honest statement of that, with your best technical explanation of why (e.g. referencing the adaptive-threshold's specific behavior on that dataset's actual cluster structure), exactly the way this project has reported every other honest, non-flattering result throughout its history. Do not soften or omit a negative finding.

- [ ] **Step 5: If Step 2 or 3 reveals a genuine bug** (not just a disappointing-but-correct result — an actual defect, e.g. a crash, an exception, or output that contradicts what the code is supposed to do per Tasks 1-2's design), fix it with a minimal, scoped change in whichever file it actually lives in, add a regression test for it following that file's existing test conventions, and re-run the full suite (`pytest tests/ -q`) to confirm the fix didn't break anything else. Document the bug and fix fully in the validation report. If no bug is found, skip this step and say so in the report.

- [ ] **Step 6: Commit**

```bash
git add docs/superpowers/plans/2026-08-17-peer-cluster-trust-validation-report.md
git commit -m "docs: validate peer-cluster-relative trust scoring against real data"
```

(If Step 5 produced a code fix, include those files in this commit too, or split into two commits — implementation fix, then report — whichever keeps the history clearer; either is fine.)

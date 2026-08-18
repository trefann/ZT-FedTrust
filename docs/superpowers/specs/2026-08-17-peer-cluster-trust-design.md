# Peer-Cluster-Relative Trust Scoring — Design

**Date:** 2026-08-17
**Status:** Approved by user, pending implementation plan

## Purpose

Post-launch evaluation of the IoT dataset support work (see
`2026-08-04-iot-dataset-support-design.md`) found a real, reproducible
limitation: on datasets where federated clients are genuinely different
real devices (N-BaIoT's 9 device types, TON_IoT's 7 device types), the
Zero Trust engine's `real_traffic`-attack detection is unreliable —
sometimes catching an attacker, often missing one, and in at least one
run (N-BaIoT) giving the two real attackers *more* aggregation weight
than an undefended baseline would have.

Root cause, confirmed by reading the code directly: `trust/update_quality.py`
scores each client's update by comparing it to **one global,
sample-weighted mean update vector**, computed across every participating
client that round (`cosine_similarity` metric), and normalises update
magnitude against that same global population (`gradient_norm` metric).
On a partitioned dataset (UNSW-NB15, TON_IoT network), every client's
"normal" is a slice of the same underlying distribution, so deviating
from the global mean is a meaningful anomaly signal. On a natural-client
dataset, clients are different device types by construction — a camera's
normal traffic was never going to look like a thermostat's — so the
global mean is an average over qualitatively different things, and
"distance from it" stops distinguishing an attacker from an honestly
different device.

This project adds a peer-cluster-relative trust signal so the engine can
ask "does this client behave like others *similar to it*?" instead of
"does this client behave like the population average?" — without
assuming any prior knowledge of device types.

## Decisions made during design

1. **Clustering is behavior-derived, not declared.** Clients are grouped
   by clustering their own update vectors (the same flattened
   `state_after − state_before` vectors `update_quality.py` already
   computes each round) — never by a self-reported "device type" field.
   This is deliberate: a declared type could be spoofed by an attacker,
   and behavior-derived clustering generalizes to federations where
   device type isn't known at all, which is the more defensible and more
   novel claim.
2. **Additive, not a replacement.** The existing `cosine_similarity`
   (global-mean) and `gradient_norm` (global-population) metrics in
   `update_quality.py` are left untouched. A new `peer_cluster_similarity`
   metric is added alongside them in the existing `metric_weights` dict —
   a mechanism `update_quality.py` was already explicitly designed to
   extend ("adding a new metric is just: compute it and add one entry
   here"). This is zero-risk to the two datasets where the existing
   signal already works (UNSW-NB15, TON_IoT network), and produces a
   clean, direct before/after comparison for validation.
3. **Hierarchical clustering with an adaptive distance threshold, not a
   fixed cluster count.** A fixed *k* (e.g. k-means) would require
   assuming in advance how many device types exist, which contradicts the
   goal of working even when nothing about the client population is
   known. Hierarchical clustering instead asks "how far apart do two
   clients' updates need to be before they stop being peers?" and lets
   the number of groups emerge from the round's own data. The threshold
   itself is adaptive (derived from the round's own pairwise-distance
   distribution, e.g. the mean pairwise cosine distance that round) —
   matching this project's existing convention of self-calibrating
   thresholds (the trust-selection threshold is already `mean − 0.5·stdev`
   of the round's own trust distribution) rather than a hardcoded magic
   number that would need per-dataset tuning.
4. **No new dependency.** `scipy` is already a dependency (used by
   `evaluation/statistical_analysis.py`). Clustering uses
   `scipy.cluster.hierarchy` (linkage + `fcluster`) over pairwise cosine
   distances (`scipy.spatial.distance.pdist`), not a new library.
5. **Behaviour dimension's simulated randomness is explicitly out of
   scope.** While reading the trust engine for this design, `trust/behaviour.py`
   was found to compute its entire score from a seeded random-number
   generator — no real signal at all, a leftover from an early phase
   before real training data existed. This is a real weakness but
   unrelated to the heterogeneity problem this project addresses; it's
   logged here as a separate, later follow-up, not pulled into this scope.

## New component: `trust/peer_clustering.py`

One function: given this round's client update vectors, return a
`{client_id: cluster_id}` mapping.

- **Input:** `client_vectors: Dict[int, torch.Tensor]` — the same
  flattened update vectors `update_quality.py`'s `evaluate_round` already
  builds (no new computation of the vectors themselves).
- **Method:** pairwise cosine distance between every pair of client
  vectors → average-linkage hierarchical clustering
  (`scipy.cluster.hierarchy.linkage`) → cut into clusters via
  `fcluster(criterion="distance", t=<adaptive threshold>)`, where the
  threshold is derived from the round's own pairwise-distance
  distribution (mean pairwise distance this round).
- **Output:** `Dict[int, int]` mapping every input client ID to a cluster
  ID. On a homogeneous round (every client's update points in a similar
  direction), this should naturally collapse toward one cluster
  containing everyone — the mechanism degrades gracefully rather than
  needing to detect "am I on a homogeneous dataset" explicitly.
- **Ground-truth isolation preserved:** this module only ever reads
  update vectors. It has no access to `is_malicious`, client identity, or
  any declared metadata — consistent with every other trust-scoring
  component in this project.

## Change to `trust/update_quality.py`

- New metric `peer_cluster_similarity`, added to the existing
  `metric_weights` dict.
- Computed as: cosine similarity between a client's own update vector and
  the sample-weighted mean update vector of *its own cluster only*
  (computed via `trust/peer_clustering.py`'s output for that round),
  rescaled to `[0, 1]` the same way the existing `cosine_similarity`
  metric already is (`(cosine + 1) / 2`).
- **Edge cases**, both defaulting to the neutral `0.5` this file's
  existing metrics already use when there's no meaningful comparison
  available (e.g. `_cosine_similarity_score`'s zero-norm case):
  - A client that ends up alone in its own cluster (no peers to compare
    against).
  - Fewer than 3 participating clients in a round — too few for the
    adaptive threshold's own pairwise-distance statistics to be
    meaningful (a single pairwise distance can't describe a
    distribution). Clustering is skipped entirely for that round and
    every client gets the neutral score for this metric only.
- **Rebalanced `metric_weights`** (values are relative — the existing
  code already normalises by their sum, so exact numbers aren't fragile):
  `loss_improvement: 0.30`, `cosine_similarity: 0.20`,
  `peer_cluster_similarity: 0.30`, `gradient_norm: 0.20`. The new metric
  gets the largest single share since demonstrating it moves the outcome
  is the point of this work.

## Validation plan

Re-run the exact comparisons already established as this project's real
baseline numbers, using the same configs and the same
`ExperimentRunner`/`compare_all` machinery already exercised:

- **Expected to improve:** `nbaiot`, `ton_iot_devices` — the two
  natural-client, genuinely heterogeneous datasets where the existing
  mechanism currently fails or underperforms undefended FedAvg.
- **Expected not to regress:** `unsw_nb15`, `ton_iot_network` — the two
  partitioned, homogeneous datasets where the existing mechanism already
  works. A regression here would mean the new signal is actively hurting
  the case it isn't needed for, which would be a real problem to catch
  before claiming this as an improvement.
- Ground-truth attacker-exclusion checks (which client IDs are actually
  malicious vs. which get excluded) are re-run the same way they already
  were for N-BaIoT and the UNSW-NB15 full-scale deep dive earlier in this
  project's history — this is what actually confirms detection improved,
  not just that an accuracy number moved.

## Explicitly out of scope

- `trust/behaviour.py`'s simulated-randomness weakness (flagged above,
  separate follow-up).
- Any change to `identity`, `integrity`, or `reputation` dimensions.
- Any change to `TrustPolicy`'s threshold logic, `ClientSelector`, or
  `TrustFedAvg` aggregation — this work only changes one input signal
  feeding into the existing composite trust score, not how that score is
  used downstream.
- Multi-round clustering history (e.g. clustering based on several
  rounds' worth of accumulated update vectors rather than just the
  current round's) — the design uses only the current round's vectors,
  matching every other per-round metric in `update_quality.py`; a
  rolling-history version is a possible future enhancement, not part of
  this scope.

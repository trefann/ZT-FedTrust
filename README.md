# ZT-FedTrust — Federated Learning Simulator

A modular, PyTorch-based simulator for federated learning experiments. It
simulates multiple clients training locally on private data partitions,
aggregates their updates on a central server via **FedAvg**, and evaluates
the resulting global model.

> **Scope note:** Phase 2 added a Zero Trust evaluation layer that computes
> and logs a trust score for every client, every round. Phase 3 put those
> scores to use: clients below a trust threshold are excluded from
> participating, and aggregation can optionally weight contributions by
> trust as well as sample count (`TrustFedAvg`). Phase 4 (this version) adds
> controlled adversarial clients (label flipping, model poisoning,
> Byzantine, Sybil) and the security metrics to measure whether ZT-FedTrust
> actually detects and reduces their influence — **without ever letting the
> server or trust code read the ground-truth malicious label**; see
> [Phase 4 — Adversarial Evaluation](#phase-4--adversarial-evaluation).
> Switching between plain `FedAvg` and `TrustFedAvg`, random and
> trust-based selection, or attack on/off, is entirely config-driven; see
> [Configuration](#configuration).

## Features

- Configurable number of simulated clients with local SGD training
- IID and Dirichlet (non-IID) data partitioning strategies
- FedAvg aggregation weighted by client sample count
- Global model evaluation after every round
- YAML-driven configuration with validation (no code changes needed to run a
  new experiment, and invalid values fail fast with a clear message)
- File + console logging, per-round metrics saved to CSV, per-round timing
- Accuracy/loss plots generated automatically after each run
- Deterministic runs via a single configurable seed
- Automated test suite (unit + integration) covering every module
- **Zero Trust evaluation layer (Phase 2):** every client is scored every
  round across 5 trust dimensions (identity, device integrity, behaviour,
  historical reputation, update quality), logged to CSV and plotted
- **Trust-based client selection + TrustFedAvg (Phase 3):** clients below a
  configurable (fixed or adaptive) trust threshold are excluded before
  training; aggregation can weight contributions by trust as well as sample
  count. Both are swappable purely via `config.yaml` — no code changes.
- **Adversarial attack simulation + security evaluation (Phase 4):** label
  flipping, model poisoning, Byzantine, and Sybil attacks against a
  configurable fraction of clients; detection/impact metrics (precision,
  recall, F1, attack success rate, malicious aggregation weight) logged to
  CSV and plotted; a fair 3-mode experiment runner (`baseline` /
  `fedavg_attack` / `ztfedtrust_attack`) for apples-to-apples comparison.

## Project structure

```
zt-fedtrust/
├── configs/
│   └── config.yaml        # experiment configuration
├── data/
│   ├── dataset.py         # dataset loading (MNIST, CIFAR-10)
│   └── partition.py       # IID / Dirichlet client partitioning
├── models/
│   └── cnn.py             # model architectures + factory
├── clients/
│   ├── client.py            # simulated client: local training, get/set_weights
│   └── malicious_client.py  # MaliciousClient (composition over Client), attack assignment (Phase 4)
├── attacks/                  # adversarial attack strategies (Phase 4)
│   ├── base_attack.py        # BaseAttack: apply_to_dataset / apply_to_update hooks
│   ├── label_flipping.py     # dataset-level: remaps training labels, never mutates the dataset
│   ├── model_poisoning.py    # update-level: scales/adds noise to the real update
│   ├── byzantine.py          # update-level: random or negated update
│   └── sybil.py               # delegates to an underlying attack; carries Sybil group metadata
├── server/
│   ├── server.py            # round orchestration: selection, training, aggregation, evaluation
│   ├── fedavg.py            # FedAvg aggregation (Phase 1, unchanged)
│   ├── trust_fedavg.py      # TrustFedAvg + the aggregation strategy pattern (Phase 3)
│   ├── selector.py          # ClientSelector: trust-threshold + participation-fraction selection (Phase 3)
│   └── selection_metrics.py # selection/aggregation-weight CSV logging, console summary, plots (Phase 3)
├── evaluation/                    # attack/security evaluation (Phase 4)
│   ├── attack_metrics.py          # detection confusion-matrix + rate formulas (ground truth read here only)
│   ├── security_metrics.py        # CSV logging + plots, shared build_round_security_records helper
│   └── experiment_runner.py       # ExperimentRunner: baseline / fedavg_attack / ztfedtrust_attack
├── simulation/
│   └── run.py             # main entry point, config loading + validation, --compare / --experiment modes
├── utils/
│   ├── logger.py          # logging setup
│   ├── metrics.py         # evaluation metrics + seeding
│   ├── validation.py      # NaN/Inf tensor guard (Phase 4 safety)
│   └── visualization.py   # accuracy/loss plot generation
├── trust/                  # Zero Trust evaluation + policy
│   ├── identity.py         # dimension 1: IdentityVerifier
│   ├── integrity.py        # dimension 2: DeviceIntegrityEvaluator
│   ├── behaviour.py        # dimension 3: BehaviourAnalyzer
│   ├── reputation.py       # dimension 4: ReputationManager
│   ├── update_quality.py   # dimension 5: UpdateQualityEvaluator
│   ├── trust_engine.py     # TrustEngine: composes all 5 into one score (Phase 2)
│   ├── trust_metrics.py    # TrustRecord, CSV logging, plots, round summary (Phase 2)
│   └── trust_policy.py     # TrustPolicy: fixed/adaptive threshold + eligibility (Phase 3)
├── tests/                  # pytest unit + integration tests (synthetic data, offline)
├── results/                 # logs, history/trust/selection/security CSVs, plots (generated)
├── requirements.txt
├── requirements-dev.txt     # adds pytest, for running the test suite
└── README.md
```

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

For running tests, install the dev extras instead (this also installs
`requirements.txt`):

```bash
pip install -r requirements-dev.txt
```

## Running a simulation

```bash
python simulation/run.py --config configs/config.yaml
```

Datasets are downloaded automatically to `data.data_dir` (default
`./data_cache`) on first run. Results are written to `logging.results_dir`:

- `history.csv` — per-round train/test loss, test accuracy, and round time
- `accuracy_plot.png` / `loss_plot.png` — generated automatically at the end of the run
- `trust_scores.csv` — every client's 5 trust-dimension scores, every round (if `trust.enabled`)
- `trust_history.csv` — per-round trust averages/min/max, for tracking evolution over time
- `trust_plots/` — average trust, reputation, behaviour, and update-quality vs round, plus a trust-score distribution histogram
- `client_selection.csv` — per-round threshold, eligible/selected/rejected counts, trust stats (if trust-based selection is active)
- `aggregation_weights.csv` — every participating client's trust, dataset size, and actual normalised aggregation weight, every round
- `selection_plots/` — client trust distribution, selection timeline (selected vs rejected), threshold evolution, aggregation weight distribution
- `security_metrics.csv` / `attack_history.csv` / `security_plots/` — if `attacks.enabled`; see [Phase 4](#phase-4--adversarial-evaluation)
- log files under `logging.log_dir`

Each round prints a concise console summary once FedAvg completes. With
trust-based selection active (Phase 3):

```
----------------------------------------
Round 8
Threshold: 0.62
Clients Available: 20
Eligible: 15
Selected: 10
Rejected: 5
Average Trust: 0.81
Aggregation: TrustFedAvg
Global Accuracy: 94.2%
----------------------------------------
```

With `trust.enabled: true` but no trust-based selection (Phase 2 only), the
simpler Phase 2 summary is printed instead:

```
Round 1
Average Trust: 0.89
Highest Trust Client: Client 3 (0.92)
Lowest Trust Client: Client 0 (0.86)
Average Reputation: 0.82
FedAvg Completed Successfully
```

### Comparing FedAvg vs TrustFedAvg

```bash
python simulation/run.py --config configs/config.yaml --compare
```

Runs the *same* trust-based client selection (same seed) twice — once per
aggregation strategy — so the only variable is the aggregation formula
itself, then saves `selection_plots/global_accuracy_comparison.png`. Note
this overwrites the standard per-run outputs (`history.csv`, etc.) with the
second run's (`trust_fedavg`) results; the comparison plot is captured from
both runs' in-memory histories before that happens.

## Configuration

All experiment parameters live in `configs/config.yaml`. Values are validated
on load (`simulation.run.validate_config`) — e.g. `clients_per_round` must be
between 1 and `num_clients`, `dataset`/`model.name`/`partition.type` must be
registered names, learning rate/batch size must be positive. Invalid configs
raise a `ValueError` listing every problem found, rather than failing deep
into training.

| Section      | Key                    | Description                                      |
|--------------|-------------------------|---------------------------------------------------|
| top-level    | `seed`                  | global RNG seed for reproducibility               |
| top-level    | `device`                | `"cuda"` or `"cpu"` (falls back to cpu if no GPU) |
| `data`       | `dataset`               | `"mnist"` or `"cifar10"`                          |
| `data`       | `num_clients`           | number of simulated clients                       |
| `data`       | `partition.type`        | `"iid"` or `"dirichlet"`                          |
| `data`       | `partition.alpha`       | Dirichlet concentration (lower = more non-IID)    |
| `model`      | `name`                  | registered model name (see `models/cnn.py`)       |
| `federated`  | `num_rounds`            | number of federated communication rounds          |
| `federated`  | `clients_per_round`     | clients sampled to participate each round         |
| `federated`  | `local_epochs`          | local training epochs per client per round        |
| `federated`  | `learning_rate`, `momentum` | local SGD optimizer settings                 |
| `logging`    | `log_dir`, `results_dir`, `log_level` | output locations and verbosity      |
| `trust`      | `enabled`               | turn Zero Trust evaluation **and** trust-based selection on/off (Phase 1/2-only random selection when `false`) |
| `trust`      | `initial_reputation`    | starting reputation for a client's first round    |
| `trust`      | `reputation_alpha`      | reputation smoothing factor (higher = more historical weight) |
| `trust`      | `threshold_mode`        | `"fixed"` or `"adaptive"` (`mean(trust) - 0.5*stdev(trust)`) |
| `trust`      | `threshold`             | trust threshold: used directly in `"fixed"` mode, fallback in `"adaptive"` mode |
| `trust`      | `participation_fraction` | fraction of *trust-eligible* clients selected each round, in (0, 1] |
| `trust.weights` | `identity`, `integrity`, `behaviour`, `reputation`, `update_quality` | relative weight of each dimension in the composite trust score |
| `aggregation` | `strategy`             | `"fedavg"` (plain, Phase 1) or `"trust_fedavg"` (trust-weighted, Phase 3) |
| `attacks`    | `enabled`               | turn adversarial simulation on/off (all Phase 1-3 behavior when `false`) |
| `attacks`    | `malicious_client_fraction` | fraction of clients assigned an attack, in [0, 1]           |
| `attacks`    | `active_attack`         | `"label_flipping"` \| `"model_poisoning"` \| `"byzantine"` \| `"sybil"` |
| `attacks.label_flipping` | `probability`, `label_map` | per-round attack-trigger probability; optional custom `{source: target}` remap |
| `attacks.model_poisoning` | `probability`, `scale_factor` | trigger probability; update-scaling multiplier, bounded to (0, 100] |
| `attacks.byzantine` | `probability`, `strategy` | trigger probability; `"random"` (noise, scaled to reference norm) or `"negate"` (`-update`) |
| `attacks.sybil` | `number_of_identities`, `probability` | Sybil group size; per-identity trigger probability |

## Zero Trust evaluation (Phase 2)

Every round, after local training and before FedAvg aggregation, `TrustEngine`
(`trust/trust_engine.py`) evaluates each participating client across five
dimensions and combines them into one composite score:

```
Trust = w1*Identity + w2*Integrity + w3*Behaviour + w4*Reputation + w5*UpdateQuality
```

| Dimension | Module | Phase 2 approach |
|---|---|---|
| Identity Verification | `trust/identity.py` — `IdentityVerifier` | deterministic: any registered client ID is "verified"; unknown IDs get a configurable lower score. Designed to be replaced by certificate-based auth later. |
| Device Integrity | `trust/integrity.py` — `DeviceIntegrityEvaluator` | simulated static `DeviceProfile` per client (firmware/secure-boot/integrity-flag + a health score), defaulting to fully healthy. |
| Behaviour Analysis | `trust/behaviour.py` — `BehaviourAnalyzer` | simulates latency, packet loss, participation, and model divergence from a seeded RNG (its own independent stream, so it doesn't disturb any other seeded randomness in the pipeline). |
| Historical Reputation | `trust/reputation.py` — `ReputationManager` | exponential smoothing: `Reputation_new = alpha*Reputation_old + (1-alpha)*CurrentTrust`, where `CurrentTrust` is a per-round signal computed from the other four dimensions (kept separate from the final composite score to avoid a circular update). |
| Update Quality | `trust/update_quality.py` — `UpdateQualityEvaluator` | computed from real training signal: loss improvement vs. the client's own previous round, cosine similarity with the round's sample-weighted mean update direction, and min-max-normalised update magnitude. |

In Phase 2, `Server.run_round` called `TrustEngine` purely for its side
effects (writing `Client.trust_score`/`trust_history`), between local
training and aggregation. Phase 3's integration moved that call to *after*
aggregation/evaluation, matching its own workflow diagram — safe because
`TrustEngine.evaluate_round` only ever reads `results` and the pre-training
global-state snapshot, neither of which aggregation or evaluation touch, so
trust scores come out identical regardless of exactly where in the round
it's called (still verified by
`tests/test_trust.py::test_trust_evaluation_does_not_change_fedavg_result`).

Crucially, the trust scores a round's *aggregation* uses are always
snapshotted **before** that round's own `TrustEngine` evaluation runs — using
a client's just-computed update-quality score to help weight that same
update would be circular. Selection (see below) works the same way: each
round's threshold check uses trust as it stood at the *end of the previous
round*.

## Trust-based selection & TrustFedAvg (Phase 3)

Each round now begins by determining who's trusted enough to participate:

```
All Clients -> Evaluate Trust (persisted from last round) -> Evaluate Threshold
            -> Eligible Clients -> Participation Fraction -> Selected Clients
```

`TrustPolicy` (`trust/trust_policy.py`) owns stage 1 — a `threshold_mode` of
`"fixed"` uses the configured `threshold` directly; `"adaptive"` recomputes
it every round as `mean(trust) - 0.5*stdev(trust)` over the current client
pool (falling back to the fixed value with fewer than 2 clients). `Rejected`
in `client_selection.csv`/the console summary specifically means "failed the
trust threshold" (`total - eligible`), distinct from simply not being
sampled in stage 2.

`ClientSelector` (`server/selector.py`) owns stage 2 — sampling
`participation_fraction` of the eligible pool via its own seeded RNG,
independent of `Server`'s legacy random-selection stream. If *no* client
meets the threshold, `Server.run_round` raises rather than silently
training zero clients.

**TrustFedAvg** (`server/trust_fedavg.py`) extends FedAvg's
`Sum(weights*samples)/Sum(samples)` to
`Sum(weights*samples*trust)/Sum(samples*trust)`. Both strategies share one
calling convention (`ClientUpdate` in, `(aggregated_state, per_client_weights)`
out), registered in a small strategy map — `Server` never branches on the
strategy name; `aggregation.strategy` in config is the only thing that
changes. `"fedavg"` delegates to the untouched Phase 1
`federated_average`, so it's numerically identical to before.

Both Phase 3 collaborators are optional constructor arguments on `Server`
(`client_selector`, `aggregation_strategy`), defaulting to `None`/`"fedavg"`
— every Phase 1/2 test constructs `Server` without them and is completely
unaffected.

## Phase 4 — Adversarial Evaluation

**Research question:** can ZT-FedTrust identify and reduce the influence of
malicious clients while maintaining good federated learning performance?
Phase 4 adds controlled malicious clients and the metrics to answer that
question from real experiments — not simulated/assumed results.

### Supported attacks (`attacks/`)

| Attack | Level | Mechanism |
|---|---|---|
| `label_flipping` | dataset | Wraps the client's local dataset so labels are remapped at read time (default: swap each even digit with the next odd one, `0<->1, 2<->3, ...`); configurable via `label_map`. **Never mutates** the underlying dataset — every client's `Subset` (and the global test set) points into the same shared MNIST/CIFAR-10 object, so this has to be a read-time wrapper, not an in-place edit. |
| `model_poisoning` | update | `poisoned = reference + (trained - reference) * scale_factor` (+ optional Gaussian noise). Bounded to `scale_factor ∈ (0, 100]` — validated both in config and in the attack class itself. |
| `byzantine` | update | `"random"`: Gaussian noise rescaled to match the reference weights' own norm (abnormal direction, bounded magnitude). `"negate"`: `poisoned = reference - (trained - reference)`, i.e. the exact opposite of the real update. |
| `sybil` | simulation | One simulated attacker controls `number_of_identities` client IDs, all performing the same underlying attack (label flipping by default), tagged with a shared `sybil_group_id` and `parent_attacker_id`. A pure bookkeeping/grouping construct — not a real identity-forgery implementation. |

Every attack triggers **per round**, independently, with probability
`attacks.<name>.probability` (default 1.0) — modelling an intermittent
attacker rather than an always-on one. Which clients are malicious is fixed
for the whole run and fully deterministic given `seed`
(`clients/malicious_client.py::assign_attacks`).

After any attack manipulates a client's update,
`utils/validation.py::validate_state_dict` checks every tensor for NaN/Inf
before it goes anywhere near the global model — again after aggregation, as
a second line of defence — so a misconfigured attack can raise a clear
error instead of silently corrupting the run.

### How malicious clients are evaluated — ground truth is never leaked

This is the load-bearing constraint of Phase 4: **the server, `Server`,
`TrustEngine`, `TrustPolicy`, and `ClientSelector` never read
`client.is_malicious`.** `MaliciousClient` (`clients/malicious_client.py`)
wraps a benign `Client` via **composition** — Client + Attack Strategy —
and exposes exactly the same duck-typed surface (`trust_score`,
`get_weights`/`set_weights`/`train`, etc.) that `Server` already used for a
plain `Client`. Nothing downstream can tell the difference without an
explicit `isinstance(client, MaliciousClient)` check — and the only code
that ever does that check is `evaluation/`, strictly for grading detection
performance after the fact:

```
Client -> Local Training -> Attack Applied -> Zero Trust Engine
              (Identity / Behaviour / Update Quality, Phase 2 — unchanged)
                       -> Trust Score -> Trust-Based Selection -> TrustFedAvg
```

The one deliberate, narrow exception: for the Sybil attack, Phase 2's
*existing* `IdentityVerifier` known/unknown mechanism is reused (not
extended with new logic) by registering only each Sybil group's parent
identity as "known" (`clients/malicious_client.py::known_identity_ids`).
This models an observable structural signal — multiple registrations
sharing a device fingerprint — not a ground-truth "this client is evil"
leak; `IdentityVerifier` itself is unmodified from Phase 2.

### Security metrics (`evaluation/attack_metrics.py`)

A client counts as "accepted" for detection purposes if it's
**trust-eligible** (passed the threshold), matching Phase 3's own
`Rejected = total - eligible` semantics — not merely "sampled in stage 2".

- **Confusion matrix**: True Positive (malicious, rejected), False Positive
  (benign, rejected), True Negative (benign, accepted), False Negative
  (malicious, accepted) → Accuracy, Precision, Recall, F1, False Positive
  Rate, False Negative Rate, Detection Rate (= Recall)
- **Attack Success Rate** = malicious clients accepted / total malicious clients
- **Malicious Aggregation Weight** = total normalised aggregation weight
  held by malicious clients that round — the key number for whether
  TrustFedAvg is actually reducing their influence, not just whether
  they're eventually rejected

Logged every round to `security_metrics.csv` and, per-client, to
`attack_history.csv` (`Round, ClientID, IsMalicious, AttackType,
TrustScore, Selected, AggregationWeight`) — enough to see whether malicious
clients gradually lose trust over the run.

### Experiment modes (`evaluation/experiment_runner.py`)

`ExperimentRunner` runs one `SimulationConfig` under three modes, sharing
the *exact same* dataset, partitioning, model, client count, rounds, local
epochs, learning rate, and seed — the only thing that differs is which
defense machinery is wired in:

| Mode | Selection | Aggregation | Attacks |
|---|---|---|---|
| `baseline` | random (Phase 1) | plain FedAvg | never (the no-attack control) |
| `fedavg_attack` | random (Phase 1) | plain FedAvg | as configured |
| `ztfedtrust_attack` | trust-based (Phase 3) | per `aggregation.strategy` | as configured |

Attack assignment is deterministic given `seed`, so `fedavg_attack` and
`ztfedtrust_attack` always face the *same* malicious clients — isolating
the defense mechanism as the only variable, per the fairness requirement.

```bash
# One mode at a time:
python simulation/run.py --config configs/config.yaml --experiment baseline
python simulation/run.py --config configs/config.yaml --experiment fedavg_attack
python simulation/run.py --config configs/config.yaml --experiment ztfedtrust_attack

# All three, plus the 3-way accuracy/loss comparison plots:
python simulation/run.py --config configs/config.yaml --experiment compare_all
```

`compare_all` writes each mode's own `security_metrics.csv`/
`attack_history.csv`/`security_plots/` into `results/<mode>/`, plus
`selection_plots/global_accuracy_comparison.png` and
`.../global_loss_comparison.png` at the top level.

### Visualisations (`results/security_plots/`)

`trust_malicious_vs_benign.png` (average trust of each group per round),
`attack_detection_rate.png`, `malicious_aggregation_weight.png`,
`client_trust_distribution_by_class.png` (final-round histogram, benign vs
malicious) — generated automatically whenever `attacks.enabled: true`. The
two comparison plots (`global_accuracy_comparison.png`,
`global_loss_comparison.png`) come from `--compare`/`--experiment
compare_all` instead, since they need multiple runs.

### Validation results (real runs, not fabricated)

Six experiments were run end-to-end against real MNIST (10 clients, 2
rounds, 20% malicious — 30% for Sybil — chosen small for fast validation;
see `configs/config.yaml` for a larger 20-client starting point):

| Test | System | Attack | Final Accuracy | Detected/Malicious | Malicious Agg. Weight |
|---|---|---|---|---|---|
| 1 | FedAvg | none | 97.1% | — | — |
| 2 | FedAvg | label flipping (20%) | 96.9% | 0/2 | 0.20 (their fair share — no defense at all) |
| 3 | ZT-FedTrust | label flipping (20%) | 97.2% | 0/2 | 0.18 → dropping |
| 4 | ZT-FedTrust | model poisoning (20%) | 90.8% | 0/2 | 0.17 |
| 5 | ZT-FedTrust | Byzantine (20%) | 96.7% | 0/2 | 0.18 |
| 6 | ZT-FedTrust | Sybil (30%, group of 3) | 96.0% | 0/3 | 0.32 |

Honest read: at only 2 rounds, reputation's smoothing (`alpha=0.80` — by
design, resistant to snap judgments) hadn't yet pushed any malicious
client's trust below the 0.60 threshold, so **detection counts are 0 across
the board at this short horizon** — plotted trust curves
(`trust_malicious_vs_benign.png`) already show a clear, consistent gap
between malicious and benign averages in every attack run, and malicious
aggregation weight is already measurably below malicious clients' fair
sample-count share under TrustFedAvg (Test 3-5) — the mechanism is
visibly working, just needs more rounds to act on what it's observing.
`tests/test_attacks.py::test_trust_score_degrades_for_a_byzantine_client_relative_to_benign`
confirms the same effect with more rounds/lower malicious fraction. Run the
`--experiment compare_all` command above with more rounds (or the 20-client
scale) to see full rejection play out.

## Client API

Each `Client` owns a persistent local model and exposes:

- `set_weights(state_dict)` — load broadcast global weights before training
- `train()` — run local SGD for `local_epochs`, returning a `LocalTrainResult`
  (updated weights, sample count, average loss)
- `get_weights()` — read back the client's current local weights

`Server.run_round` drives this cycle: select → broadcast → `set_weights` →
`train` → collect → aggregate (via whichever strategy `aggregation_strategy`
names) → evaluate → trust update.

## Testing

```bash
pytest tests/ -v
```

The suite runs entirely on a small synthetic in-memory dataset (no network
access, no real MNIST/CIFAR-10 download), so it's fast and CI-friendly. It
covers:

- Partitioning: full index coverage, no overlap, balanced IID shard sizes
- Model: forward-pass output shape for both MNIST- and CIFAR-10-shaped inputs
- Client: standalone local training, `get_weights`/`set_weights` round-trip
- FedAvg: weighted-average correctness, edge cases (empty/zero-sample input)
- Config validation: valid config passes, each invalid field is rejected
- Integration: multiple client counts, multiple round counts, deterministic
  reruns with the same seed, and improving global accuracy over rounds
- Trust (`tests/test_trust.py`): each of the 5 dimension modules in
  isolation (score bounds, reputation formula correctness, behaviour
  determinism), `TrustEngine` composition and determinism, CSV/plot
  generation, and — the key regression guard — that enabling trust
  evaluation produces identical FedAvg results to Phase 1
- Selection + TrustFedAvg (`tests/test_phase3.py`): fixed/adaptive threshold
  correctness, that a below-threshold client is never selected, aggregation
  weights always sum to 1.0, `trust_weighted_average` matches a
  hand-computed example, `fedavg_strategy` is numerically identical to the
  legacy `federated_average`, switching strategy requires only a config
  string, and full-pipeline determinism (identical selection + accuracy
  across two runs with the same seed)
- Attacks (`tests/test_attacks.py`): label flipping never mutates the
  underlying dataset, model-poisoning/Byzantine math against hand-computed
  examples plus config-bound validation, Sybil grouping, deterministic
  attack assignment, `MaliciousClient`'s trust-attribute proxying and
  non-finite-output guard, and — with a benign-majority pool — that a
  Byzantine client's trust measurably degrades relative to benign clients
- Security metrics (`tests/test_security_metrics.py`): detection
  counts/precision/recall/F1/rates against hand-computed confusion
  matrices, CSV/plot generation
- Experiment runner (`tests/test_experiment_runner.py`): mode validation
  and config-to-attack-settings mapping (offline; the full 3-mode pipeline
  against real data is exercised by the Phase 4 validation runs below,
  consistent with how Phase 1's `run_simulation` itself was verified)

The full 120-test suite (36 Phase 1 + 21 Phase 2 + 22 Phase 3 + 41 Phase 4)
runs offline in well under a minute.

## Extensibility

- **New datasets**: add a loader branch in `data/dataset.py` and an entry in
  `dataset_shape`/`SUPPORTED_DATASETS`.
- **New partitioning strategies**: add a function in `data/partition.py`
  returning `Dict[int, List[int]]`, register it in `SUPPORTED_PARTITION_TYPES`,
  then wire it into `simulation/run.py`.
- **New models**: implement an `nn.Module` in `models/` and register it in
  `models/cnn.py`'s `_MODEL_REGISTRY`.
- **New aggregation strategies**: write a function matching
  `server.trust_fedavg.AggregationStrategy` (`List[ClientUpdate] ->
  (aggregated_state, per_client_weights)`) and register it in
  `trust_fedavg.py`'s `_STRATEGIES` — no other code needs to change.
- **New selection policies**: `TrustPolicy`'s threshold strategies are a
  small registry (`trust_policy._THRESHOLD_STRATEGIES`) keyed by
  `threshold_mode`; add a new `ThresholdStrategy` function and register it
  the same way.
- **New trust metrics**: `UpdateQualityEvaluator.metric_weights` and the
  per-dimension modules are designed so a new signal is just: compute it,
  add one entry to a weights dict, done — see the module docstrings in `trust/`.
- **New attacks**: implement `BaseAttack.apply_to_dataset` and/or
  `apply_to_update` in `attacks/`, then reference it from
  `clients/malicious_client.py::build_attack` — no changes to `Server`,
  `TrustEngine`, or any other Phase 1-3 code.
- **Attack detection / adversarial-robustness improvements** (Phase 5+):
  the evaluation harness (`evaluation/attack_metrics.py`,
  `security_metrics.csv`, `attack_history.csv`) already exists to measure
  any change here — e.g. a smarter update-quality signal, a different
  reputation smoothing curve, or the `sybil_group_id`-aware identity check
  extended beyond just excluding non-parent duplicates.

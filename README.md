# ZT-FedTrust — Federated Learning Simulator

A modular, PyTorch-based simulator for federated learning experiments. It
simulates multiple clients training locally on private data partitions,
aggregates their updates on a central server via **FedAvg**, and evaluates
the resulting global model.

> **Scope note:** Phase 2 added a Zero Trust evaluation layer that computes
> and logs a trust score for every client, every round. Phase 3 (this
> version) puts those scores to use: clients below a trust threshold are
> excluded from participating, and aggregation can optionally weight
> contributions by trust as well as sample count (`TrustFedAvg`). What's
> still out of scope: no malicious-client simulation and no attack
> detection — that's Phase 4. Switching between plain `FedAvg` and
> `TrustFedAvg`, or between random and trust-based selection, is entirely
> config-driven; see [Configuration](#configuration).

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
│   └── client.py          # simulated client: local training, get/set_weights
├── server/
│   ├── server.py            # round orchestration: selection, training, aggregation, evaluation
│   ├── fedavg.py            # FedAvg aggregation (Phase 1, unchanged)
│   ├── trust_fedavg.py      # TrustFedAvg + the aggregation strategy pattern (Phase 3)
│   ├── selector.py          # ClientSelector: trust-threshold + participation-fraction selection (Phase 3)
│   └── selection_metrics.py # selection/aggregation-weight CSV logging, console summary, plots (Phase 3)
├── simulation/
│   └── run.py             # main entry point, config loading + validation, --compare mode
├── utils/
│   ├── logger.py          # logging setup
│   ├── metrics.py         # evaluation metrics + seeding
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
├── results/                 # logs, history/trust/selection CSVs, plots (generated)
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
- **Malicious-client simulation / attack detection** (Phase 4): the pieces
  to build on are all here — inject bad behavior/updates via `Client`,
  observe how `TrustPolicy`/`TrustEngine` scores and (fails to) reject them,
  and compare against `ClientSelector`'s decisions in `client_selection.csv`.
  No refactoring of this phase's code should be required.

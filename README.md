# ZT-FedTrust — Federated Learning Simulator

A modular, PyTorch-based simulator for federated learning experiments. It
simulates multiple clients training locally on private data partitions,
aggregates their updates on a central server via **FedAvg**, and evaluates
the resulting global model.

> **Scope note:** Phase 2 adds a Zero Trust evaluation layer that *computes
> and logs* a trust score for every client, every round — but nothing yet
> *acts* on those scores. There is no trust-based client selection,
> trust-weighted aggregation, client rejection, or attack simulation. FedAvg
> is byte-for-byte unchanged from Phase 1. Those behaviors are intentionally
> left for a later phase to build on top of the scores computed here.

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
  historical reputation, update quality), logged to CSV and plotted —
  scoring only, no effect on client selection or aggregation yet

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
│   ├── server.py          # round orchestration, client selection, evaluation
│   └── fedavg.py          # FedAvg aggregation
├── simulation/
│   └── run.py             # main entry point, config loading + validation
├── utils/
│   ├── logger.py          # logging setup
│   ├── metrics.py         # evaluation metrics + seeding
│   └── visualization.py   # accuracy/loss plot generation
├── trust/                  # Zero Trust evaluation layer (Phase 2, scoring only)
│   ├── identity.py         # dimension 1: IdentityVerifier
│   ├── integrity.py        # dimension 2: DeviceIntegrityEvaluator
│   ├── behaviour.py        # dimension 3: BehaviourAnalyzer
│   ├── reputation.py       # dimension 4: ReputationManager
│   ├── update_quality.py   # dimension 5: UpdateQualityEvaluator
│   ├── trust_engine.py     # TrustEngine: composes all 5 into one score
│   └── trust_metrics.py    # TrustRecord, CSV logging, plots, round summary
├── tests/                  # pytest unit + integration tests (synthetic data, offline)
├── results/                 # logs, history.csv, plots, trust_scores/history/plots (generated)
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
- log files under `logging.log_dir`

Each round also prints a concise console summary once FedAvg completes:

```
Round 1
Average Trust: 0.89
Highest Trust Client: Client 3 (0.92)
Lowest Trust Client: Client 0 (0.86)
Average Reputation: 0.82
FedAvg Completed Successfully
```

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
| `trust`      | `enabled`               | turn Zero Trust evaluation on/off (Phase 1 behavior when `false`) |
| `trust`      | `initial_reputation`    | starting reputation for a client's first round    |
| `trust`      | `reputation_alpha`      | reputation smoothing factor (higher = more historical weight) |
| `trust`      | `trust_threshold`       | defined for a future phase; **not used for filtering yet** |
| `trust.weights` | `identity`, `integrity`, `behaviour`, `reputation`, `update_quality` | relative weight of each dimension in the composite trust score |

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

**This phase is observation-only.** `Server.run_round` calls `TrustEngine`
purely for its side effects (writing `Client.trust_score`/`trust_history`,
appending to `TrustEngine.records`); its return value is never read by
FedAvg. A test (`tests/test_trust.py::test_trust_evaluation_does_not_change_fedavg_result`)
verifies that enabling trust evaluation produces byte-identical aggregation
results to running with `trust.enabled: false`.

`Client.get_weights()`/`set_weights()` plus `TrustEngine.records` /
`Client.trust_score` are the intended integration points for a future phase
to add trust-based client selection or trust-weighted aggregation, without
needing to touch this phase's code.

## Client API

Each `Client` owns a persistent local model and exposes:

- `set_weights(state_dict)` — load broadcast global weights before training
- `train()` — run local SGD for `local_epochs`, returning a `LocalTrainResult`
  (updated weights, sample count, average loss)
- `get_weights()` — read back the client's current local weights

`Server.run_round` drives this cycle: broadcast → `set_weights` → `train` →
collect → `federated_average` → evaluate.

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

## Extensibility

- **New datasets**: add a loader branch in `data/dataset.py` and an entry in
  `dataset_shape`/`SUPPORTED_DATASETS`.
- **New partitioning strategies**: add a function in `data/partition.py`
  returning `Dict[int, List[int]]`, register it in `SUPPORTED_PARTITION_TYPES`,
  then wire it into `simulation/run.py`.
- **New models**: implement an `nn.Module` in `models/` and register it in
  `models/cnn.py`'s `_MODEL_REGISTRY`.
- **New aggregation rules**: implement an alternative to
  `server/fedavg.py::federated_average` and swap it into `Server.run_round`.
- **New trust metrics**: `UpdateQualityEvaluator.metric_weights` and the
  per-dimension modules are designed so a new signal is just: compute it,
  add one entry to a weights dict, done — see the module docstrings in `trust/`.
- **Trust-based client selection / trust-weighted aggregation / attack
  simulation** (future phases): build directly on `Client.trust_score`,
  `Client.trust_history`, and `TrustEngine.records` — e.g. filter
  `Server.select_clients()`'s output by `trust_threshold`, or weight
  `federated_average`'s inputs by trust instead of only sample count. No
  refactoring of this phase's code should be required.

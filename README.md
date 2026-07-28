# ZT-FedTrust — Federated Learning Simulator

A modular, PyTorch-based simulator for federated learning experiments. It
simulates multiple clients training locally on private data partitions,
aggregates their updates on a central server via **FedAvg**, and evaluates
the resulting global model.

> **Scope note:** this is the baseline simulation layer only. It does not
> implement any trust, reputation, robust-aggregation, or security mechanism
> against malicious/faulty clients — those are intentionally out of scope
> for now and can be layered on top of this architecture later.

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
├── tests/                  # pytest unit + integration tests (synthetic data, offline)
├── results/                 # logs, history.csv, plots (generated)
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
- log files under `logging.log_dir`

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
- **Trust/security mechanisms** (future work): can be introduced as a layer
  between client updates and aggregation (e.g. filtering or reweighting the
  `updates` list passed to `federated_average`) without touching client or
  data-loading code.

# IoT Dataset Support for ZT-FedTrust — Design

**Date:** 2026-08-04
**Status:** Approved by user, pending implementation plan

## Purpose

ZT-FedTrust's proof-of-concept (Phases 1–6) validated the Zero Trust
federated-learning mechanism against MNIST/CIFAR-10, a generic vision
benchmark used as a fast substitute for the paper's actual target
application: *"Ensuring Zero Trust Security in IoT with Federated
Learning-Based Attack Detection."* This project replaces that substitute
with real IoT network-security datasets, while keeping every existing
trust/aggregation/attack/evaluation component untouched — this is an
additive extension, not a rewrite, matching the pattern every prior phase
in `PROCESS.md` followed.

## Source data inventory

Located in `ZT-Fedtrust Datasets/` (outside git, per `.gitignore` — raw
data is never committed):

| Archive | Real content | Size |
|---|---|---|
| `CICIoT2023 (2).csv` | Single CSV, 127K rows, 39 flow-stat features, multi-class attack labels | 20 MB |
| `detection+of+iot+botnet+attacks+n+baiot.zip` | N-BaIoT: 9 real IoT devices, each with `benign_traffic.csv` + Mirai/Gafgyt attack traffic **as `.rar` archives**, 115 shared statistical features | 1.7 GB |
| `Train_Test_datasets.zip` | TON_IoT: 7 IoT device telemetry CSVs (Fridge, Garage_Door, GPS_Tracker, Modbus, Motion_Light, Thermostat, Weather), 3 Linux host-log CSVs, 2 Windows host-log CSVs, 1 network-flow CSV — each with an incompatible column schema | 90 MB |
| `UNSW-NB15 dataset.zip` | UNSW-NB15: 49-feature network flow dataset, combined training+testing CSVs plus 4 raw capture CSVs | 720 MB |
| `BoT-IoT Dataset.zip` | **Mislabeled** — contains UNSW-NB15 CSVs, not real BoT-IoT data (confirmed: BoT-IoT's real schema uses `pkSeqID`/`stime`/`flgs`/... which this file does not have) | 156 MB |
| `Description_stats_datasets.zip` | Not a valid zip (corrupted/truncated); assumed non-critical documentation, not raw data | 532 KB |
| `cifar-10-python.tar.gz` | Existing CIFAR-10 archive, already supported, unrelated to this work | 170 MB |

The user will supply the real BoT-IoT dataset separately; until then it is
a disabled placeholder (see below).

## Decisions made during design

1. **BoT-IoT**: real dataset to be supplied later. Design leaves an
   extensible slot for it; it is not usable today.
2. **Client mapping**: each dataset is its own federated population/set of
   experiments — no unified cross-dataset feature space. Reasoning: the
   schemas are structurally incompatible (49 network-flow fields vs. 115
   statistical aggregates vs. 4–6 device-telemetry fields), a unified
   schema would discard nearly all dataset-specific signal, and per-dataset
   populations better match the original research framing (N-BaIoT as a
   *cross-dataset generalization test*, per project memory
   `intended_iot_datasets_vs_implementation`).
3. **Label target**: multi-class attack-type classification, using each
   dataset's own label taxonomy (not unified across datasets). This is
   compatible with the trust engine, which scores updates via loss
   improvement / cosine similarity / update magnitude — signals that don't
   depend on what the classification task's classes mean.
4. **TON_IoT scope**: split into two experiments, `ton_iot_network` and
   `ton_iot_devices`, because TON_IoT's 13 raw sources have mutually
   incompatible schemas (verified: `Garage_Door` has 4 columns,
   `train_test_network.csv` has 44) and FedAvg requires one shared model
   architecture per client population.
5. **RAR extraction**: shells out to `7z` or `unrar`, whichever is found
   on the machine (neither is currently installed — user will run
   `winget install 7zip.7zip` before N-BaIoT's attack traffic can be
   extracted). Extraction is cached so it happens once.
6. **Data scale**: every loader takes a `max_rows` subsampling knob. Demo/
   dev configs set a small cap for fast iteration; final paper-quality
   results are produced by setting `max_rows: null` (full dataset), per
   the project's existing "never fabricate a result" rule (`PROCESS.md`)
   — real numbers only, from real full-scale runs, saved to `results/`.
7. **Vision pipeline (MNIST/CIFAR-10)**: left fully in place, code and
   tests untouched, but no new configs, demos, or paper results use it
   going forward.
8. **New attack — real attack traffic**: malicious clients can train on
   *genuine* attack-labeled rows from the dataset itself (real
   Mirai/Gafgyt-infected traffic for N-BaIoT, real attack-labeled rows for
   the others) instead of a synthetic label-flip/noise/scale
   transformation. This is a fifth attack type alongside the existing four.

## Experiment matrix

| Experiment config | Source | Client construction | Features | Label classes |
|---|---|---|---|---|
| `ciciot2023` | Single CSV | Dirichlet-partitioned into `num_clients` (config), reusing `data/partition.py` unchanged | 39 numeric | multi-class attack type |
| `nbaiot` | 9 device folders | **9 natural clients**, one per real device — no partitioning | 115 shared statistical features | benign / Mirai-subtype / Gafgyt-subtype |
| `unsw_nb15` | Combined train+test CSVs | Dirichlet-partitioned into `num_clients` | 49 features | multi-class `attack_cat` (~10 categories) |
| `ton_iot_network` | `train_test_network.csv` (211K rows) | Dirichlet-partitioned into `num_clients` | 44 features | multi-class `type` (10 categories: normal/backdoor/ddos/dos/injection/mitm/password/ransomware/scanning/xss) |
| `ton_iot_devices` | 7 device CSVs | **7 natural clients**, one per real device | union of all 7 devices' sensor columns, zero-padded per client to the shared schema (each client's own columns populated, the rest zero) | shared 8-category taxonomy confirmed present across all 7 device files (normal/backdoor/ddos/injection/password/ransomware/scanning/xss) |
| `botiot` | *(pending real data)* | disabled placeholder | TBD | TBD |

Natural-client datasets (`nbaiot`, `ton_iot_devices`) skip
`data/partition.py` entirely — each real device file directly becomes one
client's local dataset, which is a more realistic non-IID split than a
synthetic Dirichlet partition over pooled data.

## New components

### `data/iot_datasets.py`

Parallel to the existing `data/dataset.py` (which is untouched). One
loader function per dataset:

- Extracts the source zip (and, for N-BaIoT, the nested per-device `.rar`
  attack archives) into `data_cache/`, caching after first run.
- Parses CSV(s), drops non-predictive identity columns (source/dest IPs,
  timestamps, row IDs — these would let a model memorize identity instead
  of learning attack patterns), encodes categorical fields
  (protocol/service/state/...), standardizes numeric features.
- Encodes the multi-class label against a fixed, dataset-specific class
  list (needed so every client's label encoding agrees, and so
  `label_flipping`'s configured `label_map` can target real class indices).
- Accepts a `max_rows` subsampling knob; `None` loads the full dataset.

A small RAR-extraction utility (used only by the N-BaIoT loader) detects
`7z`/`unrar` on `PATH` (or common Windows install locations) and raises a
clear, actionable error if neither is found.

### `models/tabular_mlp.py`

A small feedforward MLP (Linear → BatchNorm → ReLU → Dropout, stacked),
registered in the existing `_MODEL_REGISTRY` as `"mlp"`. It uses the same
`build_model(name, first_dim, num_classes)` calling convention the CNN
already uses — `simulation/run.py` and `evaluation/experiment_runner.py`
need no changes; only a `dataset_shape`-equivalent lookup needs an entry
per IoT dataset returning `(input_dim, num_classes)`.

### `attacks/real_traffic.py` — `RealAttackTrafficAttack(BaseAttack)`

A **dataset-level** attack (overrides `apply_to_dataset`, like
`LabelFlippingAttack`) rather than an update-level one:

- Constructed with a pool of real attack-labeled samples drawn from the
  experiment's own training dataset (every row whose label is not
  "normal"/"benign"), built once when the experiment's dataset is loaded.
- On an attacked round, the malicious client's local training dataset is
  replaced by a sample drawn from that pool, sized to match the client's
  original dataset size (keeps per-round training time comparable to the
  other attacks, avoiding a benchmarking confound). Sampling uses this
  client's own seeded RNG so different malicious clients don't
  deterministically see identical rows.
- `apply_to_update` is left at `BaseAttack`'s no-op default — this attack
  only manipulates training data, not the resulting weights.
- Ground-truth isolation is preserved: which *rows* are attack-labeled is
  intrinsic dataset content (exactly like `label_flipping` reading
  labels), never a read of `client.is_malicious` — the server, trust
  engine, and selector still cannot distinguish this from any other
  client without the same `evaluation/`-only `isinstance` check every
  other attack already relies on.
- Only meaningful for IoT datasets that have a benign/attack split
  available; not selectable for `mnist`/`cifar10` configs.

**Integration into the existing attack machinery**: `build_attack()` and
`assign_attacks()` (`clients/malicious_client.py`) currently take only
scalar config plus a seed — no dataset access, because the other four
attacks are purely synthetic transformations that don't need real data.
Both gain a small additive extension: an optional `attack_dataset`
parameter (default `None`, unused by the existing four attacks), threaded
through from `simulation/run.py`/`experiment_runner.py` where the
experiment's training dataset is already loaded. Selecting
`active_attack: real_traffic` without an available attack pool (e.g. on a
vision dataset) raises a clear config-validation error rather than failing
deep inside training.

`attacks/__init__.py`'s `SUPPORTED_ATTACK_TYPES` gains `"real_traffic"`
alongside the existing four.

## Explicitly unchanged

Trust engine, `TrustPolicy`, `TrustFedAvg`, `ClientSelector`,
`data/partition.py`, the other four attack strategies, the vision
(MNIST/CIFAR-10) data path, and the Phase 5 evaluation/benchmarking/
report-generation framework. New IoT experiment configs plug into that
framework exactly the way existing MNIST configs do — no framework changes
needed.

## Testing strategy

New unit tests per loader use tiny synthetic fixture CSVs, not the real
multi-GB files — keeps the suite offline and fast, matching the existing
"no network access needed for CI" rule. Additional coverage:

- Shape-correctness test for `TabularMLP` across a few `(input_dim,
  num_classes)` combinations.
- RAR-tool-detection fallback logic, mocked (doesn't require 7-Zip
  actually installed in CI).
- `RealAttackTrafficAttack.apply_to_dataset` — pool construction excludes
  benign rows, sampled dataset size matches the original, ground-truth
  isolation (no read of any `is_malicious`-like attribute).
- One regression test confirming `real_traffic` on a dataset with no
  attack pool raises a clear config error rather than an opaque failure.

## Out of scope

- Unified cross-dataset feature space.
- Real BoT-IoT integration (blocked on the user supplying correct data).
- Any change to trust scoring, aggregation math, or the four existing
  attack strategies.
- Removing MNIST/CIFAR-10 code, tests, or demo configs.

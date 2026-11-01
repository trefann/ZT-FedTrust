# Binary-label classification: scratch scripts and raw logs (2026-09-16)

Leftover scripts and raw output from the work documented in
`../2026-09-16-binary-label-classification-validation-report.md`, preserved
here (moved out of the repo root) as evidence. Nothing below was written up
in that report after commit `8897c23` — the numbers are recorded here
exactly as the logs show them.

Run scripts from the repo root with `PYTHONPATH=.` (they import
`simulation.run` and use repo-relative config/data paths).

## `verify_nbaiot_tonnet_binary.py` -> `nbaiot_tonnet_binary_verify.log`

Three-scenario run for the two datasets the report's closing note lists as
"not yet measured in binary mode". Log was written after that report's last
commit; **the report has not been updated with these numbers.**

| Dataset (binary) | baseline | fedavg_attack | ztfedtrust_attack |
|---|---|---|---|
| N-BaIoT | 99.89% | 99.92% | 99.73% |
| TON_IoT network | 97.63% | 97.63% | 97.70% |

## `tune_ton_iot_devices_binary_attempt.py` -> `ton_iot_devices_binary_*.log`

Attempts to improve TON_IoT devices (binary) `baseline` beyond the reported
62.99%. The label string in each log records the settings used; those
settings were applied by temporarily editing the config, so they are not
reproducible from the committed `ton_iot_devices_binary.yaml` alone.

| Log | Settings | baseline accuracy |
|---|---|---|
| `..._attempt2.log` | rounds=20, epochs=5, hidden=128-64 | 61.50% |
| `..._attempt3.log` | rounds=50, epochs=5, hidden=128-64 | 59.79% |
| `..._maxrows.log` | rounds=20, epochs=2, hidden=128-64, max_rows=40000 | 59.56% |

None beat the reported 62.99%.

## `diagnose_ton_iot_devices.py` -> `ton_iot_devices_diagnosis.log`

Per-device normal/attack row counts for TON_IoT devices (full, uncapped):
each device has exactly 15,000 `normal` rows; attack share ranges from
51.8% (modbus) to 62.4% (fridge).

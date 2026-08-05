"""Tests for evaluation/benchmark.py: config building, ablation/sensitivity overrides,
master schema shape, and failure handling — all offline (no dataset download).
"""

from unittest.mock import patch

import pytest

from evaluation.benchmark import (
    MASTER_SCHEMA_FIELDNAMES,
    BenchmarkConfig,
    BenchmarkRunner,
    ExperimentSettings,
    MasterRecord,
    _select_mode,
    build_simulation_config,
    load_benchmark_config,
)


def _settings(**overrides) -> ExperimentSettings:
    settings = ExperimentSettings(num_clients=10, rounds=2)
    for key, value in overrides.items():
        setattr(settings, key, value)
    return settings


# --------------------------------------------------------------------------
# The four algorithm/attack conditions map onto Phase 4's existing modes
# --------------------------------------------------------------------------


def test_condition_a_fedavg_no_attack():
    assert _select_mode("fedavg", "none") == "baseline"
    config = build_simulation_config(_settings(), "fedavg", "none", 0.0, 42, "results")
    assert config.trust.enabled is False
    assert config.aggregation.strategy == "fedavg"
    assert config.attacks.enabled is False


def test_condition_b_fedavg_under_attack():
    assert _select_mode("fedavg", "label_flipping") == "fedavg_attack"
    config = build_simulation_config(_settings(), "fedavg", "label_flipping", 0.2, 42, "results")
    assert config.trust.enabled is False
    assert config.aggregation.strategy == "fedavg"
    assert config.attacks.enabled is True
    assert config.attacks.malicious_client_fraction == 0.2
    assert config.attacks.label_flipping.enabled is True
    assert config.attacks.model_poisoning.enabled is False


def test_condition_c_ztfedtrust_no_attack():
    assert _select_mode("ztfedtrust", "none") == "ztfedtrust_attack"
    config = build_simulation_config(_settings(), "ztfedtrust", "none", 0.0, 42, "results")
    assert config.trust.enabled is True
    assert config.aggregation.strategy == "trust_fedavg"
    assert config.attacks.enabled is False


def test_condition_d_ztfedtrust_under_attack():
    assert _select_mode("ztfedtrust", "byzantine") == "ztfedtrust_attack"
    config = build_simulation_config(_settings(), "ztfedtrust", "byzantine", 0.3, 42, "results")
    assert config.trust.enabled is True
    assert config.aggregation.strategy == "trust_fedavg"
    assert config.attacks.enabled is True
    assert config.attacks.byzantine.enabled is True


def test_select_mode_rejects_unknown_algorithm():
    with pytest.raises(ValueError):
        _select_mode("not-a-real-algorithm", "none")


# --------------------------------------------------------------------------
# Ablation / sensitivity overrides are pure config variations
# --------------------------------------------------------------------------


def test_weight_override_zeroes_one_dimension_only():
    config = build_simulation_config(
        _settings(), "ztfedtrust", "none", 0.0, 42, "results", weight_overrides={"identity": 0.0}
    )
    assert config.trust.weights.identity == 0.0
    assert config.trust.weights.integrity == _settings().trust_weights["integrity"]


def test_threshold_override_replaces_configured_threshold():
    settings = _settings(threshold=0.60)
    config = build_simulation_config(
        settings, "ztfedtrust", "none", 0.0, 42, "results", threshold_override=0.80
    )
    assert config.trust.threshold == 0.80


def test_no_override_uses_settings_defaults():
    settings = _settings(threshold=0.65)
    config = build_simulation_config(settings, "ztfedtrust", "none", 0.0, 42, "results")
    assert config.trust.threshold == 0.65
    assert config.trust.weights.identity == settings.trust_weights["identity"]


# --------------------------------------------------------------------------
# max_rows: caps IoT dataset size for benchmark cells (MNIST/CIFAR-10 ignore it)
# --------------------------------------------------------------------------


def test_experiment_settings_max_rows_defaults_to_none():
    assert ExperimentSettings().max_rows is None


def test_build_simulation_config_threads_max_rows_into_data_config():
    settings = _settings(dataset="unsw_nb15", max_rows=5000)
    config = build_simulation_config(settings, "fedavg", "none", 0.0, 42, "results")
    assert config.data.max_rows == 5000


def test_build_simulation_config_defaults_max_rows_to_none():
    config = build_simulation_config(_settings(), "fedavg", "none", 0.0, 42, "results")
    assert config.data.max_rows is None


def test_load_benchmark_config_parses_max_rows(tmp_path):
    config_path = tmp_path / "benchmark.yaml"
    config_path.write_text(
        "experiment:\n"
        "  dataset: unsw_nb15\n"
        "  max_rows: 5000\n",
        encoding="utf-8",
    )
    config = load_benchmark_config(str(config_path))
    assert config.experiment.max_rows == 5000


def test_load_benchmark_config_defaults_max_rows_to_none(tmp_path):
    config_path = tmp_path / "benchmark.yaml"
    config_path.write_text("experiment:\n  dataset: mnist\n", encoding="utf-8")
    config = load_benchmark_config(str(config_path))
    assert config.experiment.max_rows is None


# --------------------------------------------------------------------------
# Master schema shape
# --------------------------------------------------------------------------


def test_master_record_row_matches_fieldname_count():
    record = MasterRecord(
        experiment_id="e", algorithm="fedavg", attack_type="none", malicious_fraction=0.0, seed=42,
        round_num=1, accuracy=0.9, loss=0.1, average_trust=1.0, benign_trust=1.0, malicious_trust=0.0,
        selected_clients=5, rejected_clients=0, malicious_selected=0, benign_selected=5,
        malicious_aggregation_weight=0.0, detection_rate=0.0, precision=0.0, recall=0.0, f1=0.0,
        communication_cost=1000, round_time=1.0, trust_evaluation_time=0.0, aggregation_time=0.0,
        selection_time=0.0, local_training_time=1.0, attack_success_rate=0.0, false_positive_rate=0.0,
    )
    assert len(record.to_csv_row()) == len(MASTER_SCHEMA_FIELDNAMES)


def test_master_schema_contains_all_spec_required_columns():
    required = {
        "ExperimentID", "Algorithm", "AttackType", "MaliciousFraction", "Seed", "Round", "Accuracy",
        "Loss", "AverageTrust", "BenignTrust", "MaliciousTrust", "SelectedClients", "RejectedClients",
        "MaliciousSelected", "BenignSelected", "MaliciousAggregationWeight", "DetectionRate",
        "Precision", "Recall", "F1", "CommunicationCost", "RoundTime", "TrustEvaluationTime",
        "AggregationTime",
    }
    assert required.issubset(set(MASTER_SCHEMA_FIELDNAMES))


# --------------------------------------------------------------------------
# Failure handling: a failing experiment is recorded, not silently dropped,
# and does not abort the rest of the matrix
# --------------------------------------------------------------------------


def test_failed_experiment_is_recorded_not_raised(tmp_path):
    config = BenchmarkConfig(
        experiment=_settings(), seeds=[1, 2], algorithms=["fedavg"], attacks=["none"],
        malicious_fractions=[0.0], results_dir=str(tmp_path),
    )
    runner = BenchmarkRunner(config)

    with patch("evaluation.benchmark.ExperimentRunner") as mock_runner_cls:
        mock_runner_cls.return_value.run.side_effect = RuntimeError("simulated failure for testing")
        outcomes = runner.run_matrix()

    assert len(outcomes) == 2  # both seeds attempted, neither crashed the loop
    assert all(o.status == "FAILED" for o in outcomes)
    assert all("simulated failure for testing" in o.error for o in outcomes)

    failed_csv = tmp_path / "raw" / "failed_experiments.csv"
    assert failed_csv.exists()
    content = failed_csv.read_text()
    assert "simulated failure for testing" in content


def test_raw_json_saved_for_every_outcome(tmp_path):
    config = BenchmarkConfig(
        experiment=_settings(), seeds=[1], algorithms=["fedavg"], attacks=["none"],
        malicious_fractions=[0.0], results_dir=str(tmp_path),
    )
    runner = BenchmarkRunner(config)
    with patch("evaluation.benchmark.ExperimentRunner") as mock_runner_cls:
        mock_runner_cls.return_value.run.side_effect = RuntimeError("boom")
        runner.run_matrix()

    raw_files = list((tmp_path / "raw").glob("*.json"))
    assert len(raw_files) == 1
    import json

    payload = json.loads(raw_files[0].read_text())
    assert payload["status"] == "FAILED"
    assert payload["error"] == "boom"
    assert "timestamp" in payload
    assert "experiment_settings" in payload

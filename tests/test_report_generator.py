"""Tests for evaluation/report_generator.py, using a hand-crafted synthetic master CSV
(no live experiment run / no dataset download needed — ReportGenerator reads only from CSV)."""

import csv

from evaluation.benchmark import MASTER_SCHEMA_FIELDNAMES
from evaluation.report_generator import ReportGenerator


def _write_master_csv(results_dir, rows):
    raw_dir = results_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    with open(raw_dir / "master_results.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(MASTER_SCHEMA_FIELDNAMES)
        for row in rows:
            writer.writerow(row)


def _synthetic_rows(seeds=(1, 2)):
    """(Algorithm, AttackType, Fraction, Seed, Round) grid with plausible values."""
    rows = []
    for algorithm in ("fedavg", "ztfedtrust"):
        for attack_type, fraction in (("none", 0.0), ("label_flipping", 0.2)):
            if algorithm == "fedavg":
                trust = 1.0
                benign_trust = 1.0
                malicious_trust = 1.0 if attack_type == "none" else 1.0
            else:
                trust = 0.85
                benign_trust = 0.90
                malicious_trust = 1.0 if attack_type == "none" else 0.60
            for seed_index, seed in enumerate(seeds):
                # Per-seed, per-algorithm jitter: without it, fedavg and
                # ztfedtrust rows are numerically identical at every seed, so
                # a paired t-test's differences all have zero variance (a
                # degenerate case scipy reports as NaN, not "not significant").
                algorithm_sign = 1 if algorithm == "ztfedtrust" else -1
                jitter = 0.01 * (seed_index + 1) * algorithm_sign
                for round_num in (1, 2, 3):
                    accuracy = 0.5 + 0.1 * round_num - (0.05 if attack_type != "none" else 0.0) + jitter
                    loss = 1.0 - 0.1 * round_num
                    malicious_selected = 0 if attack_type == "none" else 1
                    benign_selected = 5 if attack_type == "none" else 4
                    if attack_type == "none":
                        malicious_weight = 0.0
                    else:
                        # ZT-FedTrust holds malicious clients to a lower share than
                        # FedAvg — exercises the "reduced" branch in key findings.
                        malicious_weight = 0.2 if algorithm == "fedavg" else 0.1
                    rows.append(
                        [
                            f"{algorithm}_{attack_type}_{fraction}_seed{seed}", algorithm, attack_type, fraction,
                            seed, round_num, accuracy, loss, trust, benign_trust,
                            (malicious_trust if attack_type != "none" else 0.0),
                            malicious_selected + benign_selected, 0, malicious_selected, benign_selected,
                            malicious_weight, 0.1 * round_num if attack_type != "none" else 0.0,
                            0.5, 0.3, 0.4, 13000000, 10.0, 0.01, 0.02, 0.001, 9.9,
                            1.0 if attack_type != "none" else 0.0, 0.05 if attack_type != "none" else 0.0,
                        ]
                    )
    return rows


def test_report_generator_loads_master_csv(tmp_path):
    _write_master_csv(tmp_path, _synthetic_rows())
    generator = ReportGenerator(str(tmp_path))
    assert generator.master_df is not None
    assert len(generator.master_df) == len(_synthetic_rows())


def test_report_generator_missing_csvs_are_none_not_errors(tmp_path):
    (tmp_path / "raw").mkdir(parents=True)
    generator = ReportGenerator(str(tmp_path))
    assert generator.master_df is None
    assert generator.ablation_df is None
    # Should not raise even with nothing to generate from.
    generator.generate_all()


def test_generate_figures_creates_expected_files(tmp_path):
    _write_master_csv(tmp_path, _synthetic_rows())
    generator = ReportGenerator(str(tmp_path))
    written = generator.generate_figures()
    assert "figure1_accuracy_vs_round.png" in written
    assert "figure2_loss_vs_round.png" in written
    assert "figure3_trust_evolution.png" in written
    assert "figure4_detection_rate.png" in written
    assert "figure5_malicious_influence.png" in written
    assert "figure9_communication_overhead.png" in written
    assert "figure10_computational_overhead.png" in written
    for filename in written:
        assert (tmp_path / "figures" / filename).exists()


def test_generate_tables_creates_csv_and_markdown(tmp_path):
    _write_master_csv(tmp_path, _synthetic_rows())
    generator = ReportGenerator(str(tmp_path))
    written = generator.generate_tables()
    assert "table1_overall_performance.csv" in written
    assert "table1_overall_performance.md" in written
    assert "table2_security_performance.csv" in written
    assert "table3_malicious_influence.csv" in written
    assert "table4_overhead.csv" in written
    for filename in written:
        assert (tmp_path / "tables" / filename).exists()


def test_generate_processed_summaries(tmp_path):
    _write_master_csv(tmp_path, _synthetic_rows())
    generator = ReportGenerator(str(tmp_path))
    written = generator.generate_processed_summaries()
    assert "summary.csv" in written
    assert "security_summary.csv" in written
    assert (tmp_path / "processed" / "summary.csv").exists()


def test_generate_report_creates_markdown_with_key_sections(tmp_path):
    _write_master_csv(tmp_path, _synthetic_rows())
    generator = ReportGenerator(str(tmp_path))
    report_text = generator.generate_report(experiment_settings={"num_clients": 10})
    assert "# ZT-FedTrust Final Research Report" in report_text
    assert "## Results" in report_text
    assert "## Key Findings" in report_text
    assert (tmp_path / "reports" / "final_report.md").exists()


def test_report_never_claims_superiority_without_supporting_data(tmp_path):
    """With ztfedtrust's malicious weight LOWER than fedavg's in the synthetic data,
    the report should say so — but only because the numbers actually show it."""
    _write_master_csv(tmp_path, _synthetic_rows())
    generator = ReportGenerator(str(tmp_path))
    report_text = generator.generate_report()
    assert "reduced mean malicious aggregation weight" in report_text


def test_generate_all_does_not_raise_with_only_master_csv(tmp_path):
    _write_master_csv(tmp_path, _synthetic_rows())
    generator = ReportGenerator(str(tmp_path))
    generator.generate_all(experiment_settings={"num_clients": 10, "rounds": 3})
    assert (tmp_path / "reports" / "final_report.md").exists()


def test_significance_section_reports_insufficient_data_below_minimum(tmp_path):
    # Default fixture uses 2 seeds, below MIN_PAIRED_SAMPLES (3).
    _write_master_csv(tmp_path, _synthetic_rows(seeds=(1, 2)))
    generator = ReportGenerator(str(tmp_path))
    report_text = generator.generate_report()
    assert "### Statistical Significance" in report_text
    assert "below the minimum" in report_text
    assert "p = " not in report_text


def test_significance_section_computes_real_result_with_enough_seeds(tmp_path):
    _write_master_csv(tmp_path, _synthetic_rows(seeds=(1, 2, 3, 4, 5)))
    generator = ReportGenerator(str(tmp_path))
    report_text = generator.generate_report()
    assert "### Statistical Significance" in report_text
    assert "n=5 matched seeds" in report_text
    assert "p = " in report_text
    assert "mean difference" in report_text

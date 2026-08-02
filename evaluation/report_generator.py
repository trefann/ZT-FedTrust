"""ReportGenerator: figures, tables, and the final Markdown report (Phase 5).

Reads exclusively from saved CSV results (`results/raw/*.csv`) via pandas —
never from live `ExperimentResult`/`Server` objects — so `--report` can
regenerate everything from a previous run's output without rerunning any
experiments. If a given CSV doesn't exist (e.g. no ablation sweep was run),
the corresponding figures/tables/report sections are skipped, not
fabricated.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")  # headless-safe backend; must be set before pyplot import
import matplotlib.pyplot as plt
import pandas as pd

from evaluation.metrics import (
    accuracy_degradation,
    best_accuracy,
    convergence_round,
    final_accuracy,
    final_loss,
    robustness_ratio,
)
from evaluation.statistical_analysis import MIN_PAIRED_SAMPLES, paired_significance_test, summarize

_COLORS = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple", "tab:brown"]


def _history_from_group(group: pd.DataFrame) -> List[Dict[str, float]]:
    """Reconstruct an in-memory `history` list (round/test_accuracy/test_loss) from a CSV group."""
    return [
        {"round": int(row.Round), "test_accuracy": float(row.Accuracy), "test_loss": float(row.Loss)}
        for row in group.sort_values("Round").itertuples()
    ]


class ReportGenerator:
    """Generates figures, tables, and the final research report from saved Phase 5 results.

    Args:
        results_dir: directory containing `raw/master_results.csv` etc.
            (the same `results_dir` a `BenchmarkRunner` was given).
        convergence_target_fraction, convergence_stability_rounds: passed
            through to `evaluation.metrics.convergence_round`.
    """

    def __init__(
        self,
        results_dir: str,
        convergence_target_fraction: float = 0.95,
        convergence_stability_rounds: int = 3,
    ) -> None:
        self.results_dir = Path(results_dir)
        self.raw_dir = self.results_dir / "raw"
        self.figures_dir = self.results_dir / "figures"
        self.tables_dir = self.results_dir / "tables"
        self.reports_dir = self.results_dir / "reports"
        for directory in (self.figures_dir, self.tables_dir, self.reports_dir):
            directory.mkdir(parents=True, exist_ok=True)

        self.convergence_target_fraction = convergence_target_fraction
        self.convergence_stability_rounds = convergence_stability_rounds

        self.master_df = self._load_csv("master_results.csv")
        self.ablation_df = self._load_csv("ablation_master.csv")
        self.sensitivity_df = self._load_csv("sensitivity_master.csv")
        self.weight_sensitivity_df = self._load_csv("weight_sensitivity_master.csv")
        self.failed_df = self._load_csv("failed_experiments.csv")

    def _load_csv(self, filename: str) -> Optional[pd.DataFrame]:
        path = self.raw_dir / filename
        if not path.exists():
            return None
        df = pd.read_csv(path)
        return df if not df.empty else None

    def _primary_attack(self, df: pd.DataFrame) -> Tuple[Optional[str], Optional[float]]:
        """Auto-pick the most-represented non-"none" (AttackType, MaliciousFraction) pair."""
        non_none = df[df["AttackType"] != "none"]
        if non_none.empty:
            return None, None
        counts = non_none.groupby(["AttackType", "MaliciousFraction"]).size()
        attack_type, fraction = counts.idxmax()
        return attack_type, float(fraction)

    def _line_plot(
        self, series: Dict[str, pd.Series], title: str, xlabel: str, ylabel: str, output_name: str
    ) -> bool:
        series = {label: s for label, s in series.items() if s is not None and not s.empty}
        if not series:
            return False
        fig, ax = plt.subplots(figsize=(8, 5))
        for (label, s), color in zip(series.items(), _COLORS):
            ax.plot(s.index, s.values, marker="o", label=label, color=color)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(self.figures_dir / output_name)
        plt.close(fig)
        return True

    def _bar_plot(
        self, labels: List[str], values: List[float], title: str, xlabel: str, ylabel: str, output_name: str
    ) -> bool:
        if not labels:
            return False
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.bar(labels, values, color=_COLORS[0])
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(True, alpha=0.3, axis="y")
        fig.tight_layout()
        fig.savefig(self.figures_dir / output_name)
        plt.close(fig)
        return True

    # -- Figures -------------------------------------------------------------

    def generate_figures(self) -> List[str]:
        """Generate every figure derivable from the loaded CSVs. Returns filenames actually written."""
        written: List[str] = []
        if self.master_df is not None:
            written += self._figures_from_master(self.master_df)
        if self.ablation_df is not None:
            written += self._figure_ablation(self.ablation_df)
        if self.sensitivity_df is not None:
            written += self._figure_threshold_sensitivity(self.sensitivity_df)
        return written

    def _figures_from_master(self, df: pd.DataFrame) -> List[str]:
        written = []
        attack_type, fraction = self._primary_attack(df)

        fedavg_none = df[(df.Algorithm == "fedavg") & (df.AttackType == "none")]
        fedavg_attack = (
            df[(df.Algorithm == "fedavg") & (df.AttackType == attack_type) & (df.MaliciousFraction == fraction)]
            if attack_type else pd.DataFrame()
        )
        zt_attack = (
            df[(df.Algorithm == "ztfedtrust") & (df.AttackType == attack_type) & (df.MaliciousFraction == fraction)]
            if attack_type else pd.DataFrame()
        )

        # Figure 1 — Accuracy vs Round
        if self._line_plot(
            {
                "FedAvg (no attack)": fedavg_none.groupby("Round")["Accuracy"].mean(),
                f"FedAvg + {attack_type}": fedavg_attack.groupby("Round")["Accuracy"].mean() if not fedavg_attack.empty else None,
                f"ZT-FedTrust + {attack_type}": zt_attack.groupby("Round")["Accuracy"].mean() if not zt_attack.empty else None,
            },
            "Figure 1: Accuracy vs Communication Round", "Round", "Accuracy", "figure1_accuracy_vs_round.png",
        ):
            written.append("figure1_accuracy_vs_round.png")

        # Figure 2 — Loss vs Round
        if self._line_plot(
            {
                "FedAvg (no attack)": fedavg_none.groupby("Round")["Loss"].mean(),
                f"FedAvg + {attack_type}": fedavg_attack.groupby("Round")["Loss"].mean() if not fedavg_attack.empty else None,
                f"ZT-FedTrust + {attack_type}": zt_attack.groupby("Round")["Loss"].mean() if not zt_attack.empty else None,
            },
            "Figure 2: Loss vs Communication Round", "Round", "Loss", "figure2_loss_vs_round.png",
        ):
            written.append("figure2_loss_vs_round.png")

        # Figure 3 — Trust Evolution (ZT-FedTrust under the primary attack)
        if not zt_attack.empty:
            if self._line_plot(
                {
                    "Benign": zt_attack.groupby("Round")["BenignTrust"].mean(),
                    "Malicious": zt_attack.groupby("Round")["MaliciousTrust"].mean(),
                },
                "Figure 3: Trust Evolution (Benign vs Malicious)", "Round", "Average Trust",
                "figure3_trust_evolution.png",
            ):
                written.append("figure3_trust_evolution.png")

            # Figure 4 — Attack Detection Rate vs Round
            if self._line_plot(
                {"Detection Rate": zt_attack.groupby("Round")["DetectionRate"].mean()},
                "Figure 4: Attack Detection Rate vs Round", "Round", "Detection Rate",
                "figure4_detection_rate.png",
            ):
                written.append("figure4_detection_rate.png")

        # Figure 5 — Malicious Aggregation Influence: FedAvg vs ZT-FedTrust
        if self._line_plot(
            {
                "FedAvg": fedavg_attack.groupby("Round")["MaliciousAggregationWeight"].mean() if not fedavg_attack.empty else None,
                "ZT-FedTrust": zt_attack.groupby("Round")["MaliciousAggregationWeight"].mean() if not zt_attack.empty else None,
            },
            "Figure 5: Malicious Aggregation Influence", "Round", "Malicious Aggregation Weight",
            "figure5_malicious_influence.png",
        ):
            written.append("figure5_malicious_influence.png")

        # Figure 6 — Accuracy under different attack strengths
        if attack_type is not None:
            subset = df[df.AttackType == attack_type]
            last_round_idx = subset.groupby(["Algorithm", "MaliciousFraction", "Seed"])["Round"].idxmax()
            final_rows = subset.loc[last_round_idx]
            grouped = final_rows.groupby(["Algorithm", "MaliciousFraction"])["Accuracy"].mean().reset_index()
            if not grouped.empty:
                fig, ax = plt.subplots(figsize=(8, 5))
                for algorithm, group in grouped.groupby("Algorithm"):
                    group = group.sort_values("MaliciousFraction")
                    ax.plot(group.MaliciousFraction, group.Accuracy, marker="o", label=algorithm)
                ax.set_xlabel("Malicious Fraction")
                ax.set_ylabel("Final Accuracy")
                ax.set_title(f"Figure 6: Accuracy vs Attack Strength ({attack_type})")
                ax.legend()
                ax.grid(True, alpha=0.3)
                fig.tight_layout()
                fig.savefig(self.figures_dir / "figure6_accuracy_vs_attack_strength.png")
                plt.close(fig)
                written.append("figure6_accuracy_vs_attack_strength.png")

        # Figure 9 — Communication overhead: FedAvg vs ZT-FedTrust
        comm = df.groupby("Algorithm")["CommunicationCost"].mean()
        if self._bar_plot(
            list(comm.index), list(comm.values),
            "Figure 9: Communication Overhead (mean bytes/round)", "Algorithm", "Bytes (simulated/estimated)",
            "figure9_communication_overhead.png",
        ):
            written.append("figure9_communication_overhead.png")

        # Figure 10 — Computational overhead breakdown
        time_cols = ["LocalTrainingTime", "TrustEvaluationTime", "SelectionTime", "AggregationTime"]
        available_time_cols = [c for c in time_cols if c in df.columns]
        if available_time_cols:
            means = df.groupby("Algorithm")[available_time_cols].mean()
            fig, ax = plt.subplots(figsize=(8, 5))
            bottom = pd.Series(0.0, index=means.index)
            for col, color in zip(available_time_cols, _COLORS):
                ax.bar(means.index, means[col], bottom=bottom, label=col, color=color)
                bottom = bottom + means[col]
            ax.set_xlabel("Algorithm")
            ax.set_ylabel("Seconds per round (mean)")
            ax.set_title("Figure 10: Computational Overhead Breakdown")
            ax.legend()
            ax.grid(True, alpha=0.3, axis="y")
            fig.tight_layout()
            fig.savefig(self.figures_dir / "figure10_computational_overhead.png")
            plt.close(fig)
            written.append("figure10_computational_overhead.png")

        return written

    def _figure_ablation(self, df: pd.DataFrame) -> List[str]:
        last_round_idx = df.groupby(["Algorithm", "Seed"])["Round"].idxmax()
        final_rows = df.loc[last_round_idx]
        grouped = final_rows.groupby("Algorithm")[["Accuracy", "DetectionRate", "F1", "MaliciousAggregationWeight"]].mean()
        if grouped.empty:
            return []
        fig, axes = plt.subplots(2, 2, figsize=(11, 8))
        metrics = ["Accuracy", "DetectionRate", "F1", "MaliciousAggregationWeight"]
        for ax, metric in zip(axes.flat, metrics):
            ax.bar(grouped.index, grouped[metric], color=_COLORS[0])
            ax.set_title(metric)
            ax.tick_params(axis="x", rotation=30)
            ax.grid(True, alpha=0.3, axis="y")
        fig.suptitle("Figure 7: Ablation Study — ZT-FedTrust Component Variants")
        fig.tight_layout()
        fig.savefig(self.figures_dir / "figure7_ablation_study.png")
        plt.close(fig)
        return ["figure7_ablation_study.png"]

    def _figure_threshold_sensitivity(self, df: pd.DataFrame) -> List[str]:
        df = df.copy()
        df["Threshold"] = df["Algorithm"].str.replace("threshold_", "", regex=False).astype(float)
        last_round_idx = df.groupby(["Threshold", "Seed"])["Round"].idxmax()
        final_rows = df.loc[last_round_idx]
        grouped = final_rows.groupby("Threshold")[["Accuracy", "DetectionRate"]].mean().sort_index()
        if grouped.empty:
            return []
        fig, ax1 = plt.subplots(figsize=(8, 5))
        ax1.plot(grouped.index, grouped["DetectionRate"], marker="o", color="tab:red", label="Detection Rate")
        ax1.set_xlabel("Trust Threshold")
        ax1.set_ylabel("Detection Rate", color="tab:red")
        ax2 = ax1.twinx()
        ax2.plot(grouped.index, grouped["Accuracy"], marker="s", color="tab:blue", label="Accuracy")
        ax2.set_ylabel("Accuracy", color="tab:blue")
        ax1.set_title("Figure 8: Threshold Sensitivity")
        ax1.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(self.figures_dir / "figure8_threshold_sensitivity.png")
        plt.close(fig)
        return ["figure8_threshold_sensitivity.png"]

    # -- Tables --------------------------------------------------------------

    def generate_tables(self) -> List[str]:
        """Generate every table derivable from the loaded CSVs (CSV + Markdown). Returns filenames written."""
        written: List[str] = []
        if self.master_df is not None:
            written += self._table_overall_performance(self.master_df)
            written += self._table_security_performance(self.master_df)
            written += self._table_malicious_influence(self.master_df)
            written += self._table_overhead(self.master_df)
        if self.ablation_df is not None:
            written += self._table_ablation(self.ablation_df)
        return written

    def _write_table(self, rows: List[Dict], name: str) -> List[str]:
        if not rows:
            return []
        df = pd.DataFrame(rows)
        csv_path = self.tables_dir / f"{name}.csv"
        md_path = self.tables_dir / f"{name}.md"
        df.to_csv(csv_path, index=False)
        with open(md_path, "w", encoding="utf-8") as f:
            f.write(df.to_markdown(index=False))
        return [f"{name}.csv", f"{name}.md"]

    def _table_overall_performance(self, df: pd.DataFrame) -> List[str]:
        rows = []
        for (algorithm, attack, fraction), group in df.groupby(["Algorithm", "AttackType", "MaliciousFraction"]):
            for seed, seed_group in group.groupby("Seed"):
                history = _history_from_group(seed_group)
                rows.append(
                    {
                        "Algorithm": algorithm, "Attack": attack, "MaliciousFraction": fraction, "Seed": seed,
                        "FinalAccuracy": final_accuracy(history), "BestAccuracy": best_accuracy(history),
                        "FinalLoss": final_loss(history),
                        "ConvergenceRound": convergence_round(
                            history, self.convergence_target_fraction, self.convergence_stability_rounds
                        ),
                    }
                )
        return self._write_table(rows, "table1_overall_performance")

    def _table_security_performance(self, df: pd.DataFrame) -> List[str]:
        attacked = df[df.AttackType != "none"]
        if attacked.empty:
            return []
        rows = []
        for (algorithm, attack, fraction), group in attacked.groupby(["Algorithm", "AttackType", "MaliciousFraction"]):
            last_round_idx = group.groupby("Seed")["Round"].idxmax()
            final_rows = group.loc[last_round_idx]
            rows.append(
                {
                    "Algorithm": algorithm, "Attack": attack, "MaliciousFraction": fraction,
                    "DetectionRate": final_rows["DetectionRate"].mean(),
                    "Precision": final_rows["Precision"].mean(),
                    "Recall": final_rows["Recall"].mean(),
                    "F1": final_rows["F1"].mean(),
                    "FalsePositiveRate": final_rows["FalsePositiveRate"].mean(),
                    "AttackSuccessRate": final_rows["AttackSuccessRate"].mean(),
                }
            )
        return self._write_table(rows, "table2_security_performance")

    def _table_malicious_influence(self, df: pd.DataFrame) -> List[str]:
        attacked = df[df.AttackType != "none"]
        if attacked.empty:
            return []
        rows = []
        for (algorithm, attack, fraction), group in attacked.groupby(["Algorithm", "AttackType", "MaliciousFraction"]):
            total_selected = group["MaliciousSelected"] + group["BenignSelected"]
            malicious_selection_rate = (group["MaliciousSelected"] / total_selected.replace(0, pd.NA)).fillna(0.0).mean()
            rows.append(
                {
                    "Algorithm": algorithm, "Attack": attack, "MaliciousFraction": fraction,
                    "MaliciousSelectionRate": malicious_selection_rate,
                    "MaliciousAggregationWeight": group["MaliciousAggregationWeight"].mean(),
                    "AverageMaliciousTrust": group["MaliciousTrust"].mean(),
                }
            )
        return self._write_table(rows, "table3_malicious_influence")

    def _table_overhead(self, df: pd.DataFrame) -> List[str]:
        rows = []
        runtime_by_algorithm = {}
        for algorithm, group in df.groupby("Algorithm"):
            runtime = group.groupby("Seed")["RoundTime"].sum().mean()
            runtime_by_algorithm[algorithm] = runtime
        fedavg_runtime = runtime_by_algorithm.get("fedavg")
        for algorithm, group in df.groupby("Algorithm"):
            runtime = runtime_by_algorithm[algorithm]
            trust_overhead = (runtime - fedavg_runtime) if fedavg_runtime is not None else None
            trust_overhead_pct = (
                (trust_overhead / fedavg_runtime * 100) if trust_overhead is not None and fedavg_runtime else None
            )
            rows.append(
                {
                    "Algorithm": algorithm,
                    "Runtime": runtime,
                    "CommunicationCost": group["CommunicationCost"].sum() / group["Seed"].nunique(),
                    "TrustOverhead": trust_overhead,
                    "TrustOverheadPercent": trust_overhead_pct,
                    "AggregationTime": group["AggregationTime"].mean(),
                }
            )
        return self._write_table(rows, "table4_overhead")

    def _table_ablation(self, df: pd.DataFrame) -> List[str]:
        rows = []
        last_round_idx = df.groupby(["Algorithm", "Seed"])["Round"].idxmax()
        final_rows = df.loc[last_round_idx]
        for variant, group in final_rows.groupby("Algorithm"):
            rows.append(
                {
                    "Variant": variant,
                    "Accuracy": group["Accuracy"].mean(),
                    "DetectionRate": group["DetectionRate"].mean(),
                    "F1": group["F1"].mean(),
                    "MaliciousInfluence": group["MaliciousAggregationWeight"].mean(),
                }
            )
        return self._write_table(rows, "table5_ablation")

    # -- Processed summaries (section 19) ------------------------------------

    def generate_processed_summaries(self) -> List[str]:
        """Write results/processed/{summary,security_summary,ablation_summary,sensitivity_summary}.csv."""
        processed_dir = self.results_dir / "processed"
        processed_dir.mkdir(parents=True, exist_ok=True)
        written = []

        if self.master_df is not None:
            rows = []
            for (algorithm, attack, fraction), group in self.master_df.groupby(
                ["Algorithm", "AttackType", "MaliciousFraction"]
            ):
                seed_finals = [
                    final_accuracy(_history_from_group(seed_group))
                    for _, seed_group in group.groupby("Seed")
                ]
                stats = summarize(seed_finals)
                rows.append(
                    {
                        "Algorithm": algorithm, "AttackType": attack, "MaliciousFraction": fraction,
                        "NumSeeds": stats.n, "MeanFinalAccuracy": stats.mean, "StdFinalAccuracy": stats.stdev,
                        "MedianFinalAccuracy": stats.median, "MinFinalAccuracy": stats.minimum,
                        "MaxFinalAccuracy": stats.maximum, "CI95Low": stats.ci95_low, "CI95High": stats.ci95_high,
                    }
                )
            pd.DataFrame(rows).to_csv(processed_dir / "summary.csv", index=False)
            written.append("summary.csv")

            attacked = self.master_df[self.master_df.AttackType != "none"]
            if not attacked.empty:
                sec_rows = []
                for (algorithm, attack, fraction), group in attacked.groupby(
                    ["Algorithm", "AttackType", "MaliciousFraction"]
                ):
                    last_round_idx = group.groupby("Seed")["Round"].idxmax()
                    final_rows = group.loc[last_round_idx]
                    sec_rows.append(
                        {
                            "Algorithm": algorithm, "AttackType": attack, "MaliciousFraction": fraction,
                            "MeanDetectionRate": final_rows["DetectionRate"].mean(),
                            "MeanF1": final_rows["F1"].mean(),
                            "MeanMaliciousAggregationWeight": final_rows["MaliciousAggregationWeight"].mean(),
                        }
                    )
                pd.DataFrame(sec_rows).to_csv(processed_dir / "security_summary.csv", index=False)
                written.append("security_summary.csv")

        if self.ablation_df is not None:
            last_round_idx = self.ablation_df.groupby(["Algorithm", "Seed"])["Round"].idxmax()
            final_rows = self.ablation_df.loc[last_round_idx]
            summary = final_rows.groupby("Algorithm")[["Accuracy", "DetectionRate", "F1", "MaliciousAggregationWeight"]].mean()
            summary.to_csv(processed_dir / "ablation_summary.csv")
            written.append("ablation_summary.csv")

        if self.sensitivity_df is not None:
            df = self.sensitivity_df.copy()
            df["Threshold"] = df["Algorithm"].str.replace("threshold_", "", regex=False).astype(float)
            last_round_idx = df.groupby(["Threshold", "Seed"])["Round"].idxmax()
            final_rows = df.loc[last_round_idx]
            summary = final_rows.groupby("Threshold")[["Accuracy", "DetectionRate", "Precision", "Recall"]].mean()
            summary.to_csv(processed_dir / "sensitivity_summary.csv")
            written.append("sensitivity_summary.csv")

        return written

    # -- Report ----------------------------------------------------------------

    def generate_report(self, experiment_settings: Optional[Dict] = None) -> str:
        """Write `results/reports/final_report.md`. Returns the report text."""
        lines: List[str] = ["# ZT-FedTrust Final Research Report", ""]
        lines.append(
            "This report is generated entirely from data saved under "
            f"`{self.results_dir}` — no value below is hand-entered or assumed."
        )
        lines.append("")

        lines.append("## Experimental Setup")
        if experiment_settings:
            for key, value in experiment_settings.items():
                lines.append(f"- **{key}**: {value}")
        else:
            lines.append("_Experiment settings not provided to the report generator._")
        lines.append("")

        if self.failed_df is not None:
            lines.append(f"> **{len(self.failed_df)} experiment(s) FAILED** and are excluded from the results "
                         "below (see `raw/failed_experiments.csv` for details) — not silently dropped, "
                         "flagged explicitly here.")
            lines.append("")

        if self.master_df is None:
            lines.append("## Results")
            lines.append("No `master_results.csv` found — no experiment matrix has been run yet.")
        else:
            lines.append("## Results")
            lines.append("")
            lines.append(self._report_accuracy_section(self.master_df))
            lines.append(self._report_robustness_section(self.master_df))
            lines.append(self._report_significance_section(self.master_df))
            lines.append(self._report_detection_section(self.master_df))
            lines.append(self._report_overhead_section(self.master_df))

        if self.ablation_df is not None:
            lines.append("## Ablation Results")
            lines.append(self._report_ablation_section(self.ablation_df))

        if self.sensitivity_df is not None:
            lines.append("## Sensitivity Analysis")
            lines.append(self._report_sensitivity_section(self.sensitivity_df))

        lines.append("## Key Findings")
        lines.append(self._report_key_findings(self.master_df))

        report_text = "\n".join(lines)
        output_path = self.reports_dir / "final_report.md"
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(report_text)
        return report_text

    def _report_accuracy_section(self, df: pd.DataFrame) -> str:
        lines = ["### Accuracy"]
        for (algorithm, attack, fraction), group in df.groupby(["Algorithm", "AttackType", "MaliciousFraction"]):
            finals = [final_accuracy(_history_from_group(g)) for _, g in group.groupby("Seed")]
            stats = summarize(finals)
            lines.append(
                f"- **{algorithm}**, attack=`{attack}`, fraction={fraction}: "
                f"final accuracy = {stats.mean:.4f} (n={stats.n} seed(s), std={stats.stdev:.4f}, "
                f"95% CI [{stats.ci95_low:.4f}, {stats.ci95_high:.4f}])"
            )
        return "\n".join(lines) + "\n"

    def _report_robustness_section(self, df: pd.DataFrame) -> str:
        lines = ["### Robustness"]
        none_group = df[df.AttackType == "none"]
        for algorithm in df["Algorithm"].unique():
            baseline_rows = none_group[none_group.Algorithm == algorithm]
            if baseline_rows.empty:
                continue
            baseline_acc = summarize(
                [final_accuracy(_history_from_group(g)) for _, g in baseline_rows.groupby("Seed")]
            ).mean
            attacked = df[(df.Algorithm == algorithm) & (df.AttackType != "none")]
            for (attack, fraction), group in attacked.groupby(["AttackType", "MaliciousFraction"]):
                attacked_acc = summarize(
                    [final_accuracy(_history_from_group(g)) for _, g in group.groupby("Seed")]
                ).mean
                degradation = accuracy_degradation(baseline_acc, attacked_acc)
                ratio = robustness_ratio(attacked_acc, baseline_acc)
                lines.append(
                    f"- **{algorithm}** under `{attack}` @ {fraction}: accuracy degradation = "
                    f"{degradation:.4f}, robustness ratio = {ratio:.4f}"
                )
        return "\n".join(lines) + "\n"

    def _report_significance_section(self, df: pd.DataFrame) -> str:
        """Paired t-test (FedAvg vs ZT-FedTrust final accuracy, matched by seed) per attack condition.

        Only reported when both algorithms share the same seed set for a
        given (attack, fraction) and there are enough pairs — otherwise
        `paired_significance_test` returns `None` and that line is skipped
        entirely rather than showing a fabricated result.
        """
        lines = ["### Statistical Significance"]
        attack_conditions = df[["AttackType", "MaliciousFraction"]].drop_duplicates()
        any_result = False

        for _, row in attack_conditions.iterrows():
            attack, fraction = row["AttackType"], row["MaliciousFraction"]
            condition = df[(df.AttackType == attack) & (df.MaliciousFraction == fraction)]
            fedavg_rows = condition[condition.Algorithm == "fedavg"]
            zt_rows = condition[condition.Algorithm == "ztfedtrust"]
            if fedavg_rows.empty or zt_rows.empty:
                continue

            fedavg_by_seed = {
                seed: final_accuracy(_history_from_group(g)) for seed, g in fedavg_rows.groupby("Seed")
            }
            zt_by_seed = {
                seed: final_accuracy(_history_from_group(g)) for seed, g in zt_rows.groupby("Seed")
            }
            shared_seeds = sorted(set(fedavg_by_seed) & set(zt_by_seed))
            if not shared_seeds:
                continue

            baseline_values = [fedavg_by_seed[s] for s in shared_seeds]
            comparison_values = [zt_by_seed[s] for s in shared_seeds]
            result = paired_significance_test(baseline_values, comparison_values)
            if result is None:
                lines.append(
                    f"- `{attack}` @ {fraction}: only {len(shared_seeds)} matched seed(s) — "
                    f"below the minimum ({MIN_PAIRED_SAMPLES}) needed to report significance."
                )
                continue

            any_result = True
            significance = "significant" if result.significant_at_0_05 else "not significant"
            lines.append(
                f"- `{attack}` @ {fraction} (n={result.n_pairs} matched seeds): ZT-FedTrust vs FedAvg "
                f"final-accuracy paired t-test, mean difference = {result.mean_difference:+.4f}, "
                f"p = {result.p_value:.4g} ({significance} at α=0.05)"
            )

        if not any_result and len(lines) == 1:
            lines.append(
                "No attack condition had matching FedAvg/ZT-FedTrust seeds with enough repeats "
                "to compute significance."
            )
        return "\n".join(lines) + "\n"

    def _report_detection_section(self, df: pd.DataFrame) -> str:
        attacked = df[df.AttackType != "none"]
        if attacked.empty:
            return "### Detection\nNo attack experiments in this result set.\n"
        lines = ["### Detection"]
        for (algorithm, attack, fraction), group in attacked.groupby(["Algorithm", "AttackType", "MaliciousFraction"]):
            last_round_idx = group.groupby("Seed")["Round"].idxmax()
            final_rows = group.loc[last_round_idx]
            lines.append(
                f"- **{algorithm}** vs `{attack}` @ {fraction}: detection rate = "
                f"{final_rows['DetectionRate'].mean():.4f}, F1 = {final_rows['F1'].mean():.4f}, "
                f"malicious aggregation weight = {final_rows['MaliciousAggregationWeight'].mean():.4f}"
            )
        return "\n".join(lines) + "\n"

    def _report_overhead_section(self, df: pd.DataFrame) -> str:
        lines = ["### Overhead"]
        runtime_by_algorithm = df.groupby(["Algorithm", "Seed"])["RoundTime"].sum().groupby("Algorithm").mean()
        for algorithm, runtime in runtime_by_algorithm.items():
            lines.append(f"- **{algorithm}**: mean total runtime = {runtime:.2f}s across all rounds")
        if "fedavg" in runtime_by_algorithm and "ztfedtrust" in runtime_by_algorithm:
            overhead = runtime_by_algorithm["ztfedtrust"] - runtime_by_algorithm["fedavg"]
            pct = (overhead / runtime_by_algorithm["fedavg"] * 100) if runtime_by_algorithm["fedavg"] else 0.0
            lines.append(f"- Trust overhead: {overhead:.2f}s ({pct:.1f}%)")
        return "\n".join(lines) + "\n"

    def _report_ablation_section(self, df: pd.DataFrame) -> str:
        last_round_idx = df.groupby(["Algorithm", "Seed"])["Round"].idxmax()
        final_rows = df.loc[last_round_idx]
        summary = final_rows.groupby("Algorithm")[["Accuracy", "DetectionRate", "F1"]].mean()
        lines = ["| Variant | Accuracy | Detection Rate | F1 |", "|---|---|---|---|"]
        for variant, row in summary.iterrows():
            lines.append(f"| {variant} | {row['Accuracy']:.4f} | {row['DetectionRate']:.4f} | {row['F1']:.4f} |")
        return "\n".join(lines) + "\n"

    def _report_sensitivity_section(self, df: pd.DataFrame) -> str:
        df = df.copy()
        df["Threshold"] = df["Algorithm"].str.replace("threshold_", "", regex=False).astype(float)
        last_round_idx = df.groupby(["Threshold", "Seed"])["Round"].idxmax()
        final_rows = df.loc[last_round_idx]
        summary = final_rows.groupby("Threshold")[["Accuracy", "DetectionRate"]].mean().sort_index()
        lines = ["| Threshold | Accuracy | Detection Rate |", "|---|---|---|"]
        for threshold, row in summary.iterrows():
            lines.append(f"| {threshold} | {row['Accuracy']:.4f} | {row['DetectionRate']:.4f} |")
        return "\n".join(lines) + "\n"

    def _report_key_findings(self, df: Optional[pd.DataFrame]) -> str:
        """Only states what the loaded data actually shows — no claims beyond the numbers."""
        if df is None:
            return "No results available to draw findings from.\n"

        findings: List[str] = []
        none_group = df[df.AttackType == "none"]
        attacked = df[df.AttackType != "none"]

        for algorithm in df["Algorithm"].unique():
            baseline_rows = none_group[none_group.Algorithm == algorithm]
            attacked_rows = attacked[attacked.Algorithm == algorithm]
            if baseline_rows.empty or attacked_rows.empty:
                continue
            baseline_acc = summarize(
                [final_accuracy(_history_from_group(g)) for _, g in baseline_rows.groupby("Seed")]
            ).mean
            attacked_acc = summarize(
                [final_accuracy(_history_from_group(g)) for _, g in attacked_rows.groupby("Seed")]
            ).mean
            ratio = robustness_ratio(attacked_acc, baseline_acc)
            findings.append(
                f"- **{algorithm}**: accuracy under attack was {ratio * 100:.1f}% of its no-attack "
                f"accuracy in this result set ({attacked_acc:.4f} vs {baseline_acc:.4f})."
            )

        if not attacked.empty:
            for algorithm in ("fedavg", "ztfedtrust"):
                group = attacked[attacked.Algorithm == algorithm]
                if group.empty:
                    continue
                last_round_idx = group.groupby("Seed")["Round"].idxmax()
                final_rows = group.loc[last_round_idx]
                findings.append(
                    f"- **{algorithm}**: mean detection rate against attacks in this result set was "
                    f"{final_rows['DetectionRate'].mean():.4f}, with mean malicious aggregation weight "
                    f"{final_rows['MaliciousAggregationWeight'].mean():.4f}."
                )

            if "fedavg" in attacked.Algorithm.values and "ztfedtrust" in attacked.Algorithm.values:
                fedavg_weight = attacked[attacked.Algorithm == "fedavg"]["MaliciousAggregationWeight"].mean()
                zt_weight = attacked[attacked.Algorithm == "ztfedtrust"]["MaliciousAggregationWeight"].mean()
                if zt_weight < fedavg_weight:
                    findings.append(
                        f"- ZT-FedTrust reduced mean malicious aggregation weight from {fedavg_weight:.4f} "
                        f"(FedAvg) to {zt_weight:.4f} in this result set."
                    )
                elif zt_weight > fedavg_weight:
                    findings.append(
                        f"- In this result set, ZT-FedTrust's mean malicious aggregation weight "
                        f"({zt_weight:.4f}) was *not* lower than FedAvg's ({fedavg_weight:.4f}) — "
                        "review the underlying data before drawing conclusions."
                    )

        if not findings:
            return "Insufficient data to draw findings (need both attacked and non-attacked runs for at least one algorithm).\n"
        return "\n".join(findings) + "\n"

    def generate_all(self, experiment_settings: Optional[Dict] = None) -> None:
        """Generate processed summaries, figures, tables, and the report in one call."""
        self.generate_processed_summaries()
        self.generate_figures()
        self.generate_tables()
        self.generate_report(experiment_settings)

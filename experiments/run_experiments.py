"""Phase 5 CLI: run the benchmark matrix, ablation/sensitivity sweeps, or regenerate a report.

Examples:
    python experiments/run_experiments.py --config experiments/configs/label_flipping.yaml
    python experiments/run_experiments.py --all
    python experiments/run_experiments.py --ablation
    python experiments/run_experiments.py --sensitivity
    python experiments/run_experiments.py --report

Small-scale override flags (--num-clients/--rounds/--seeds) let you run a
fast validation pass without editing the committed config files, which are
sized per the spec's suggested defaults (20 clients, 30 rounds, 5 seeds).
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import asdict
from pathlib import Path
from typing import List

# Allow running as a standalone script by putting the project root on
# sys.path, matching the pattern used by simulation/run.py.
sys.path.append(str(Path(__file__).resolve().parent.parent))

from evaluation.benchmark import BenchmarkConfig, BenchmarkRunner, load_benchmark_config
from evaluation.report_generator import ReportGenerator

CONFIGS_DIR = Path(__file__).resolve().parent / "configs"


def merge_configs(config_paths: List[Path]) -> BenchmarkConfig:
    """Merge multiple experiment configs into one full-matrix `BenchmarkConfig`.

    Used by `--all`: every config file shares the same `experiment`/`seeds`
    block (the fairness requirement — see the config files' comments), so
    the first file's values are used for those; `attacks`/
    `malicious_fractions` are unioned across all files (in order,
    de-duplicated) to produce the complete experiment matrix.
    """
    configs = [load_benchmark_config(str(p)) for p in config_paths]
    base = configs[0]

    attacks: List[str] = []
    fractions: List[float] = []
    for config in configs:
        for attack in config.attacks:
            if attack not in attacks:
                attacks.append(attack)
        for fraction in config.malicious_fractions:
            if fraction not in fractions:
                fractions.append(fraction)

    return BenchmarkConfig(
        experiment=base.experiment,
        seeds=base.seeds,
        algorithms=base.algorithms,
        attacks=attacks,
        malicious_fractions=fractions,
        results_dir=base.results_dir,
        convergence_target_fraction=base.convergence_target_fraction,
        convergence_stability_rounds=base.convergence_stability_rounds,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run ZT-FedTrust Phase 5 research experiments.")
    parser.add_argument("--config", type=str, default=None, help="Path to one experiment YAML config.")
    parser.add_argument("--all", action="store_true", help="Run every config in experiments/configs/.")
    parser.add_argument(
        "--ablation", action="store_true",
        help="Run the trust-component ablation study (full vs each dimension disabled).",
    )
    parser.add_argument(
        "--sensitivity", action="store_true", help="Run the trust-threshold sensitivity sweep."
    )
    parser.add_argument(
        "--report", action="store_true",
        help="Regenerate figures/tables/report from existing results, without running any experiments.",
    )
    parser.add_argument(
        "--results-dir", type=str, default=None, help="Override the config's results_dir."
    )
    parser.add_argument(
        "--attack", type=str, default="label_flipping",
        help="Attack type used by --ablation/--sensitivity (default: label_flipping).",
    )
    parser.add_argument(
        "--malicious-fraction", type=float, default=0.20,
        help="Malicious fraction used by --ablation/--sensitivity (default: 0.20).",
    )
    parser.add_argument("--num-clients", type=int, default=None, help="Override experiment.num_clients.")
    parser.add_argument("--rounds", type=int, default=None, help="Override experiment.rounds.")
    parser.add_argument(
        "--seeds", type=str, default=None, help="Comma-separated seed list, overriding the config's."
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.report:
        results_dir = args.results_dir or "results"
        generator = ReportGenerator(results_dir)
        generator.generate_all()
        print(f"Report regenerated from existing results at {results_dir}/reports/final_report.md")
        return

    if args.all:
        config_paths = sorted(CONFIGS_DIR.glob("*.yaml"))
        if not config_paths:
            raise FileNotFoundError(f"No config files found in {CONFIGS_DIR}")
        config = merge_configs(config_paths)
    elif args.config:
        config = load_benchmark_config(args.config)
    elif args.ablation or args.sensitivity:
        config = load_benchmark_config(str(CONFIGS_DIR / "baseline.yaml"))
    else:
        raise SystemExit(
            "Specify one of --config <path>, --all, --ablation, --sensitivity, or --report."
        )

    if args.num_clients is not None:
        config.experiment.num_clients = args.num_clients
    if args.rounds is not None:
        config.experiment.rounds = args.rounds
    if args.seeds is not None:
        config.seeds = [int(s.strip()) for s in args.seeds.split(",")]

    results_dir = args.results_dir or config.results_dir
    runner = BenchmarkRunner(config, results_dir=results_dir)

    if args.ablation:
        runner.run_ablation(attack_type=args.attack, malicious_fraction=args.malicious_fraction)
    elif args.sensitivity:
        runner.run_threshold_sensitivity(attack_type=args.attack, malicious_fraction=args.malicious_fraction)
    else:
        runner.run_matrix()

    failed_count = sum(1 for o in runner.outcomes if o.status == "FAILED")
    if failed_count:
        print(f"WARNING: {failed_count} experiment(s) FAILED — see {results_dir}/raw/failed_experiments.csv")

    generator = ReportGenerator(
        results_dir, config.convergence_target_fraction, config.convergence_stability_rounds
    )
    generator.generate_all(experiment_settings=asdict(config.experiment))
    print(f"Done. Report at {results_dir}/reports/final_report.md")


if __name__ == "__main__":
    main()

"""ExperimentRunner: baseline / fedavg_attack / ztfedtrust_attack — a fair comparison harness.

All three modes share the exact same dataset, partitioning, model
architecture, client count, rounds, local epochs, learning rate, and random
seed (all read from one `SimulationConfig`) — the *only* thing that differs
between modes is which defensive/attack machinery is wired into `Server`,
isolating exactly the variable each experiment is meant to test. This
module reuses `simulation.run`'s `load_experiment_data`/`resolve_device`
rather than duplicating them.

Import direction note: this module imports FROM `simulation.run`.
`simulation/run.py` only ever imports this module back inside a function
body (`main()`), never at top level, to avoid a circular import.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union

from clients.client import Client
from clients.malicious_client import MaliciousClient, assign_attacks, known_identity_ids
from evaluation.security_metrics import (
    AttackHistoryRecord,
    SecurityMetricsRecord,
    build_round_security_records,
    save_attack_history_csv,
    save_loss_comparison_plot,
    save_security_metrics_csv,
    save_security_plots,
)
from server.selection_metrics import save_accuracy_comparison_plot
from server.selector import ClientSelector
from server.server import Server
from simulation.run import SimulationConfig, load_experiment_data, resolve_device
from trust.trust_engine import TrustEngine, TrustWeights
from trust.trust_policy import TrustPolicy
from utils.metrics import set_seed

EXPERIMENT_MODES = ("baseline", "fedavg_attack", "ztfedtrust_attack")

_MODE_DISPLAY_NAMES = {
    "baseline": "FedAvg (no attack)",
    "fedavg_attack": "FedAvg (attack)",
    "ztfedtrust_attack": "ZT-FedTrust (attack)",
}


def _attack_settings(config: SimulationConfig) -> Dict[str, Dict[str, Any]]:
    """Flatten `AttackConfig` into the plain-dict shape `assign_attacks`/`build_attack` expect."""
    attacks = config.attacks
    return {
        "label_flipping": {
            "probability": attacks.label_flipping.probability,
            "label_map": attacks.label_flipping.label_map,
        },
        "model_poisoning": {
            "probability": attacks.model_poisoning.probability,
            "scale_factor": attacks.model_poisoning.scale_factor,
        },
        "byzantine": {
            "probability": attacks.byzantine.probability,
            "strategy": attacks.byzantine.strategy,
        },
        "sybil": {
            "probability": attacks.sybil.probability,
            "number_of_identities": attacks.sybil.number_of_identities,
        },
        "real_traffic": {
            "probability": attacks.real_traffic.probability,
        },
    }


@dataclass
class ExperimentResult:
    """Everything produced by one `ExperimentRunner.run(mode)` call."""

    mode: str
    history: List[Dict[str, float]]
    security_records: List[SecurityMetricsRecord]
    attack_history_records: List[AttackHistoryRecord]
    server: Server


class ExperimentRunner:
    """Runs one of `baseline`/`fedavg_attack`/`ztfedtrust_attack` from a single shared config.

    - `baseline`: plain Phase 1 FedAvg, random selection, no attacks —
      the no-attack control, regardless of what `config.trust`/`config.attacks` say.
    - `fedavg_attack`: plain FedAvg + random selection, WITH attacks —
      tests whether an undefended system is vulnerable.
    - `ztfedtrust_attack`: trust-based selection + the configured
      aggregation strategy (Phase 2/3), WITH attacks — the full defense
      stack under the same adversarial conditions as `fedavg_attack`.

    Attack assignment (which clients are malicious) is deterministic given
    `config.seed` (see `clients.malicious_client.assign_attacks`), so
    `fedavg_attack` and `ztfedtrust_attack` always face the *same* malicious
    clients — the fairness requirement that isolates the defense mechanism
    as the only variable.
    """

    def __init__(self, config: SimulationConfig) -> None:
        self.config = config

    def run(self, mode: str) -> ExperimentResult:
        """Run one experiment mode and return its full result."""
        if mode not in EXPERIMENT_MODES:
            raise ValueError(f"mode must be one of {EXPERIMENT_MODES}, got {mode!r}")

        config = self.config
        set_seed(config.seed)
        device = resolve_device(config.device)

        global_model, built_clients, test_dataset, attack_pool = load_experiment_data(config, device)
        clients: Dict[int, Union[Client, MaliciousClient]] = built_clients

        use_attacks = mode != "baseline" and config.attacks.enabled
        if use_attacks:
            clients = assign_attacks(
                clients,
                malicious_client_fraction=config.attacks.malicious_client_fraction,
                active_attack=config.attacks.active_attack,
                attack_settings=_attack_settings(config),
                seed=config.seed,
                attack_dataset=attack_pool,
            )

        use_trust = mode == "ztfedtrust_attack"
        trust_engine: Optional[TrustEngine] = None
        client_selector: Optional[ClientSelector] = None
        aggregation_strategy = "fedavg"
        if use_trust:
            trust_engine = TrustEngine(
                weights=TrustWeights(
                    identity=config.trust.weights.identity,
                    integrity=config.trust.weights.integrity,
                    behaviour=config.trust.weights.behaviour,
                    reputation=config.trust.weights.reputation,
                    update_quality=config.trust.weights.update_quality,
                ),
                initial_reputation=config.trust.initial_reputation,
                reputation_alpha=config.trust.reputation_alpha,
                seed=config.seed,
            )
            # Reuses Phase 2's EXISTING known/unknown identity distinction —
            # no new trust logic — so non-parent Sybil identities naturally
            # score lower on identity. See known_identity_ids's docstring.
            trust_engine.register_clients(known_identity_ids(clients))

            policy = TrustPolicy(
                threshold_mode=config.trust.threshold_mode, threshold=config.trust.threshold
            )
            client_selector = ClientSelector(
                policy=policy,
                participation_fraction=config.trust.participation_fraction,
                seed=config.seed,
            )
            aggregation_strategy = config.aggregation.strategy

        server = Server(
            global_model=global_model,
            clients=clients,
            test_dataset=test_dataset,
            device=device,
            clients_per_round=config.federated.clients_per_round,
            seed=config.seed,
            trust_engine=trust_engine,
            client_selector=client_selector,
            aggregation_strategy=aggregation_strategy,
        )

        attack_type = config.attacks.active_attack if use_attacks else "none"
        history: List[Dict[str, float]] = []
        security_records: List[SecurityMetricsRecord] = []
        attack_history_records: List[AttackHistoryRecord] = []
        for round_num in range(1, config.federated.num_rounds + 1):
            metrics = server.run_round(round_num)
            history.append(metrics)

            round_weights = {
                r.client_id: r.aggregation_weight
                for r in server.aggregation_weight_records
                if r.round_num == round_num
            }
            security_record, round_history = build_round_security_records(
                round_num, attack_type, clients, client_selector, server.last_selected_ids, round_weights
            )
            security_records.append(security_record)
            attack_history_records.extend(round_history)

        return ExperimentResult(mode, history, security_records, attack_history_records, server)

    def run_all_and_compare(self) -> Dict[str, ExperimentResult]:
        """Run all three modes and save per-mode outputs plus the 3-way comparison plots.

        Each mode's `security_metrics.csv`/`attack_history.csv`/
        `security_plots/` are written into a mode-named subdirectory of
        `logging.results_dir` (unlike Phase 3's 2-way `--compare`, so the
        three runs' individual outputs don't overwrite each other); the
        combined 3-way accuracy/loss comparison plots go at the top level.
        """
        results: Dict[str, ExperimentResult] = {}
        histories: Dict[str, List[Dict[str, float]]] = {}

        for mode in EXPERIMENT_MODES:
            result = self.run(mode)
            results[mode] = result
            histories[_MODE_DISPLAY_NAMES[mode]] = result.history

            mode_dir = f"{self.config.logging.results_dir}/{mode}"
            save_security_metrics_csv(result.security_records, mode_dir)
            save_attack_history_csv(result.attack_history_records, mode_dir)
            save_security_plots(result.security_records, result.attack_history_records, mode_dir)

        save_accuracy_comparison_plot(histories, self.config.logging.results_dir)
        save_loss_comparison_plot(histories, self.config.logging.results_dir)
        return results

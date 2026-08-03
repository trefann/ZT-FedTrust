"""Tests for Phase 3: trust-based selection, TrustFedAvg, and their integration.

Covers `TrustPolicy`, `ClientSelector`, the `server.trust_fedavg` strategy
pattern, `selection_metrics` CSV/plot generation, and — the key regression
guards — that switching aggregation strategy requires only a config string
(no code branching at call sites), and that the whole pipeline stays
deterministic under a fixed seed.
"""

import copy

import pytest
import torch

from clients.client import Client
from data.partition import make_client_subsets, partition_iid
from helpers import make_synthetic_dataset
from models.cnn import build_model
from server.selection_metrics import (
    save_aggregation_weights_csv,
    save_selection_csv,
    save_selection_plots,
)
from server.selector import ClientSelector
from server.server import Server
from server.trust_fedavg import (
    ClientUpdate,
    compute_aggregation_weights,
    fedavg_strategy,
    get_aggregation_strategy,
    trust_weighted_average,
)
from server.fedavg import federated_average
from trust.trust_engine import TrustEngine, TrustWeights
from trust.trust_policy import TrustPolicy
from utils.metrics import set_seed


# --------------------------------------------------------------------------
# TrustPolicy
# --------------------------------------------------------------------------


def _client_with_trust(client_id: int, trust: float) -> Client:
    dataset = make_synthetic_dataset(num_samples=8, seed=client_id)
    model = build_model("cnn", in_channels=1, num_classes=10)
    client = Client(client_id, model, dataset, torch.device("cpu"))
    client.trust_score = trust
    return client


def test_fixed_threshold_only_admits_clients_at_or_above_it():
    policy = TrustPolicy(threshold_mode="fixed", threshold=0.60)
    clients = [
        _client_with_trust(0, 0.91),
        _client_with_trust(1, 0.82),
        _client_with_trust(2, 0.47),
        _client_with_trust(3, 0.69),
        _client_with_trust(4, 0.38),
    ]
    eligible = policy.get_trusted_clients(clients)
    assert {c.client_id for c in eligible} == {0, 1, 3}
    assert all(c.trust_score >= 0.60 for c in eligible)


def test_is_trusted_reflects_last_computed_threshold():
    policy = TrustPolicy(threshold_mode="fixed", threshold=0.60)
    trusted = _client_with_trust(0, 0.9)
    untrusted = _client_with_trust(1, 0.3)
    policy.get_trusted_clients([trusted, untrusted])
    assert policy.is_trusted(trusted) is True
    assert policy.is_trusted(untrusted) is False


def test_adaptive_threshold_matches_mean_minus_half_stdev():
    import statistics

    policy = TrustPolicy(threshold_mode="adaptive", threshold=0.60)
    scores = [0.9, 0.8, 0.5, 0.7, 0.4]
    clients = [_client_with_trust(i, s) for i, s in enumerate(scores)]
    policy.get_trusted_clients(clients)
    expected = statistics.mean(scores) - 0.5 * statistics.stdev(scores)
    assert policy.calculate_selection_statistics().threshold_used == pytest.approx(expected)


def test_adaptive_threshold_falls_back_with_fewer_than_two_clients():
    policy = TrustPolicy(threshold_mode="adaptive", threshold=0.60)
    policy.get_trusted_clients([_client_with_trust(0, 0.9)])
    assert policy.calculate_selection_statistics().threshold_used == 0.60


def test_calculate_selection_statistics_counts_are_consistent():
    policy = TrustPolicy(threshold_mode="fixed", threshold=0.60)
    clients = [_client_with_trust(i, s) for i, s in enumerate([0.9, 0.8, 0.4, 0.3])]
    policy.get_trusted_clients(clients)
    stats = policy.calculate_selection_statistics()
    assert stats.total_clients == 4
    assert stats.eligible_clients == 2
    assert stats.min_trust == 0.3
    assert stats.max_trust == 0.9


def test_trust_policy_rejects_unknown_threshold_mode():
    with pytest.raises(ValueError):
        TrustPolicy(threshold_mode="not-a-real-mode")


def test_eligibility_survives_trust_score_mutation_after_selection():
    """Regression test: `Server.run_round` calls `TrustEngine.evaluate_round`
    (which mutates `client.trust_score` in place) *after* selection but
    *before* `calculate_selection_statistics()` is read for the console
    summary / CSV row. If TrustPolicy read trust scores live instead of a
    frozen snapshot, eligibility/average-trust would silently reflect a mix
    of this round's fresh scores (for clients that just trained) and stale
    scores (for clients that didn't) — evaluated against a threshold that
    was itself computed before any of that mutation happened.
    """
    policy = TrustPolicy(threshold_mode="adaptive", threshold=0.60)
    clients = [_client_with_trust(i, 1.0) for i in range(10)]  # all tied -> threshold == 1.0
    eligible = policy.get_trusted_clients(clients)
    assert len(eligible) == 10  # everyone exactly meets an exactly-1.0 threshold

    stats_before_mutation = policy.calculate_selection_statistics()

    # Simulate TrustEngine overwriting trust_score for a subset of clients
    # (as if they'd just trained and been freshly scored) after selection.
    for client in clients[:7]:
        client.trust_score = 0.85

    stats_after_mutation = policy.calculate_selection_statistics()
    assert stats_after_mutation.eligible_clients == stats_before_mutation.eligible_clients == 10
    assert stats_after_mutation.average_trust == stats_before_mutation.average_trust == 1.0
    assert all(policy.is_trusted(client) for client in clients)


# --------------------------------------------------------------------------
# ClientSelector
# --------------------------------------------------------------------------


def test_selector_never_selects_a_client_below_threshold():
    policy = TrustPolicy(threshold_mode="fixed", threshold=0.60)
    selector = ClientSelector(policy, participation_fraction=1.0, seed=1)
    clients = {
        0: _client_with_trust(0, 0.91),
        1: _client_with_trust(1, 0.82),
        2: _client_with_trust(2, 0.47),
        3: _client_with_trust(3, 0.69),
        4: _client_with_trust(4, 0.38),
    }
    selected = selector.select(clients)
    assert set(selected).issubset({0, 1, 3})
    assert 2 not in selected and 4 not in selected


def test_selector_applies_participation_fraction_to_eligible_pool():
    policy = TrustPolicy(threshold_mode="fixed", threshold=0.0)  # everyone eligible
    selector = ClientSelector(policy, participation_fraction=0.5, seed=1)
    clients = {i: _client_with_trust(i, 0.9) for i in range(12)}
    selected = selector.select(clients)
    assert len(selected) == 6  # 12 eligible * 0.5


def test_selector_is_deterministic_given_same_seed():
    def run():
        policy = TrustPolicy(threshold_mode="fixed", threshold=0.0)
        selector = ClientSelector(policy, participation_fraction=0.5, seed=7)
        clients = {i: _client_with_trust(i, 0.9) for i in range(10)}
        return selector.select(clients)

    assert run() == run()


def test_selector_returns_empty_when_nobody_meets_threshold():
    policy = TrustPolicy(threshold_mode="fixed", threshold=0.99)
    selector = ClientSelector(policy, participation_fraction=1.0, seed=1)
    clients = {0: _client_with_trust(0, 0.5), 1: _client_with_trust(1, 0.4)}
    assert selector.select(clients) == []


def test_selector_rejects_invalid_participation_fraction():
    policy = TrustPolicy()
    with pytest.raises(ValueError):
        ClientSelector(policy, participation_fraction=0.0)
    with pytest.raises(ValueError):
        ClientSelector(policy, participation_fraction=1.5)


# --------------------------------------------------------------------------
# Aggregation weights + TrustFedAvg math
# --------------------------------------------------------------------------


def test_aggregation_weights_sum_to_one_with_trust():
    updates = [
        ClientUpdate(0, {"w": torch.zeros(2)}, num_samples=100, trust=0.9),
        ClientUpdate(1, {"w": torch.zeros(2)}, num_samples=50, trust=0.5),
        ClientUpdate(2, {"w": torch.zeros(2)}, num_samples=25, trust=0.2),
    ]
    weights = compute_aggregation_weights(updates, use_trust=True)
    assert sum(weights.values()) == pytest.approx(1.0)


def test_aggregation_weights_sum_to_one_without_trust():
    updates = [
        ClientUpdate(0, {"w": torch.zeros(2)}, num_samples=100, trust=0.1),
        ClientUpdate(1, {"w": torch.zeros(2)}, num_samples=50, trust=0.9),
    ]
    weights = compute_aggregation_weights(updates, use_trust=False)
    assert sum(weights.values()) == pytest.approx(1.0)
    # Ignoring trust: weights should be purely sample-proportional.
    assert weights[0] == pytest.approx(100 / 150)
    assert weights[1] == pytest.approx(50 / 150)


def test_trust_weighted_average_matches_hand_computed_example():
    state_a = {"w": torch.tensor([1.0, 2.0])}
    state_b = {"w": torch.tensor([3.0, 4.0])}
    updates = [
        ClientUpdate(0, state_a, num_samples=10, trust=0.5),
        ClientUpdate(1, state_b, num_samples=10, trust=1.0),
    ]
    # Sum(w*n*trust)/Sum(n*trust): (a*10*0.5 + b*10*1.0) / (10*0.5 + 10*1.0)
    expected = (state_a["w"] * 5.0 + state_b["w"] * 10.0) / 15.0
    aggregated, weights = trust_weighted_average(updates)
    assert torch.allclose(aggregated["w"], expected)
    assert weights[0] == pytest.approx(5.0 / 15.0)
    assert weights[1] == pytest.approx(10.0 / 15.0)


def test_fedavg_strategy_matches_legacy_federated_average_exactly():
    state_a = {"w": torch.tensor([1.0, 2.0])}
    state_b = {"w": torch.tensor([3.0, 4.0])}
    updates = [
        ClientUpdate(0, state_a, num_samples=30, trust=0.1),  # trust must be ignored
        ClientUpdate(1, state_b, num_samples=70, trust=0.9),
    ]
    aggregated_new, _ = fedavg_strategy(updates)
    aggregated_legacy = federated_average([(state_a, 30), (state_b, 70)])
    assert torch.allclose(aggregated_new["w"], aggregated_legacy["w"])


def test_get_aggregation_strategy_dispatch():
    assert get_aggregation_strategy("fedavg") is fedavg_strategy
    assert get_aggregation_strategy("trust_fedavg") is trust_weighted_average
    with pytest.raises(ValueError):
        get_aggregation_strategy("not-a-real-strategy")


def test_compute_aggregation_weights_rejects_empty_updates():
    with pytest.raises(ValueError):
        compute_aggregation_weights([], use_trust=True)


# --------------------------------------------------------------------------
# Server integration: strategy switching, threshold enforcement, determinism
# --------------------------------------------------------------------------


def _build_server(
    num_clients: int, aggregation_strategy: str, seed: int = 42, threshold: float = 0.0
) -> Server:
    device = torch.device("cpu")
    train_dataset = make_synthetic_dataset(num_samples=160, num_classes=10, seed=seed)
    test_dataset = make_synthetic_dataset(num_samples=40, num_classes=10, seed=seed + 1)

    shards = partition_iid(train_dataset, num_clients=num_clients, seed=seed)
    subsets = make_client_subsets(train_dataset, shards)

    global_model = build_model("cnn", in_channels=1, num_classes=10)
    clients = {
        cid: Client(cid, global_model, subset, device, batch_size=16, learning_rate=0.1, local_epochs=1)
        for cid, subset in subsets.items()
    }

    trust_engine = TrustEngine(weights=TrustWeights(), seed=seed)
    trust_engine.register_clients(clients.keys())

    policy = TrustPolicy(threshold_mode="fixed", threshold=threshold)
    selector = ClientSelector(policy, participation_fraction=0.5, seed=seed)

    return Server(
        global_model,
        clients,
        test_dataset,
        device,
        clients_per_round=2,
        seed=seed,
        trust_engine=trust_engine,
        client_selector=selector,
        aggregation_strategy=aggregation_strategy,
    )


def _build_and_run(num_clients: int, aggregation_strategy: str, seed: int = 42, num_rounds: int = 3):
    """Same seed-isolation discipline as Phase 2's regression test: reset,
    build, and run to completion in one atomic step before comparing runs."""
    set_seed(seed)
    server = _build_server(num_clients, aggregation_strategy, seed)
    history = [server.run_round(r) for r in range(1, num_rounds + 1)]
    return history, server


def test_switching_aggregation_strategy_requires_only_a_config_string():
    # Same construction, only the `aggregation_strategy` string differs —
    # proving no code branching is required anywhere else to switch.
    history_fedavg, _ = _build_and_run(4, "fedavg")
    history_trust_fedavg, _ = _build_and_run(4, "trust_fedavg")
    assert len(history_fedavg) == len(history_trust_fedavg) == 3
    for round_metrics in (*history_fedavg, *history_trust_fedavg):
        assert 0.0 <= round_metrics["test_accuracy"] <= 1.0


def test_no_client_below_threshold_ever_participates():
    # A threshold above every possible trust score (max is 1.0) means zero
    # clients are eligible; Server must refuse to silently train anyone
    # rather than let an untrusted (or in this case, unevaluated) client through.
    set_seed(42)
    server_strict = _build_server(4, "trust_fedavg", seed=42, threshold=1.01)
    with pytest.raises(RuntimeError):
        server_strict.run_round(1)


def test_selection_and_weight_records_recorded_every_round():
    _, server = _build_and_run(4, "trust_fedavg", seed=42, num_rounds=3)
    assert len(server.selection_records) == 3
    assert len(server.aggregation_weight_records) > 0
    for record in server.selection_records:
        assert record.selected_clients <= record.eligible_clients <= record.total_clients


def test_full_pipeline_is_deterministic_given_same_seed():
    history_a, server_a = _build_and_run(4, "trust_fedavg", seed=99)
    history_b, server_b = _build_and_run(4, "trust_fedavg", seed=99)

    for round_a, round_b in zip(history_a, history_b):
        assert round_a["test_accuracy"] == pytest.approx(round_b["test_accuracy"])

    selection_a = [(r.selected_clients, r.eligible_clients) for r in server_a.selection_records]
    selection_b = [(r.selected_clients, r.eligible_clients) for r in server_b.selection_records]
    assert selection_a == selection_b


# --------------------------------------------------------------------------
# CSV + plot generation
# --------------------------------------------------------------------------


def test_selection_csv_and_plots_generated(tmp_path):
    _, server = _build_and_run(4, "trust_fedavg", seed=42, num_rounds=2)
    results_dir = str(tmp_path)

    save_selection_csv(server.selection_records, results_dir)
    save_aggregation_weights_csv(server.aggregation_weight_records, results_dir)
    final_scores = [c.trust_score for c in server.clients.values()]
    save_selection_plots(server.selection_records, server.aggregation_weight_records, final_scores, results_dir)

    assert (tmp_path / "client_selection.csv").exists()
    assert (tmp_path / "aggregation_weights.csv").exists()
    plots_dir = tmp_path / "selection_plots"
    for filename in (
        "client_trust_distribution.png",
        "client_selection_timeline.png",
        "trust_threshold_evolution.png",
        "aggregation_weight_distribution.png",
    ):
        assert (plots_dir / filename).exists()

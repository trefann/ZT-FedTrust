"""Tests for the Phase 2 Zero Trust evaluation layer.

Covers each dimension module in isolation, `TrustEngine` composition and
determinism, CSV/plot generation, and — critically — that wiring a
`TrustEngine` into `Server` does not change FedAvg's aggregation result at
all (Phase 2 must be pure observation).
"""

import copy

import pytest
import torch

from clients.client import Client, LocalTrainResult
from data.partition import make_client_subsets, partition_iid
from helpers import make_synthetic_dataset
from models.cnn import build_model
from server.server import Server
from trust.behaviour import BehaviourAnalyzer
from trust.identity import IdentityVerifier
from trust.integrity import DeviceIntegrityEvaluator, DeviceProfile
from trust.reputation import ReputationManager
from trust.trust_engine import TrustEngine, TrustWeights
from trust.trust_metrics import save_trust_history, save_trust_plots, save_trust_scores
from trust.update_quality import UpdateQualityEvaluator
from utils.metrics import set_seed


# --------------------------------------------------------------------------
# Identity
# --------------------------------------------------------------------------


def test_identity_verifier_known_client_returns_verified_score():
    verifier = IdentityVerifier(verified_score=1.0, unknown_score=0.3)
    verifier.register_known_clients([0, 1, 2])
    assert verifier.verify(1) == 1.0


def test_identity_verifier_unknown_client_returns_configured_score():
    verifier = IdentityVerifier(verified_score=1.0, unknown_score=0.3)
    verifier.register_known_clients([0, 1, 2])
    assert verifier.verify(99) == 0.3


def test_identity_verifier_defaults_to_verified_when_registry_unset():
    verifier = IdentityVerifier()
    assert verifier.verify(123) == 1.0


# --------------------------------------------------------------------------
# Device integrity
# --------------------------------------------------------------------------


def test_integrity_evaluator_default_profile_is_fully_healthy():
    evaluator = DeviceIntegrityEvaluator()
    assert evaluator.evaluate(0) == 1.0


def test_integrity_evaluator_degraded_profile_scores_lower():
    evaluator = DeviceIntegrityEvaluator()
    evaluator.set_profile(0, DeviceProfile(firmware_ok=False, health_score=0.5))
    score = evaluator.evaluate(0)
    assert 0.0 <= score < 1.0


# --------------------------------------------------------------------------
# Behaviour
# --------------------------------------------------------------------------


def test_behaviour_analyzer_scores_are_bounded():
    analyzer = BehaviourAnalyzer(seed=1)
    for round_num in range(1, 6):
        score = analyzer.evaluate(client_id=0, round_num=round_num)
        assert 0.0 <= score <= 1.0


def test_behaviour_analyzer_is_deterministic_given_same_seed():
    analyzer_a = BehaviourAnalyzer(seed=7)
    analyzer_b = BehaviourAnalyzer(seed=7)
    scores_a = [analyzer_a.evaluate(0, r) for r in range(1, 4)]
    scores_b = [analyzer_b.evaluate(0, r) for r in range(1, 4)]
    assert scores_a == scores_b


def test_behaviour_analyzer_records_history():
    analyzer = BehaviourAnalyzer(seed=1)
    analyzer.evaluate(client_id=5, round_num=1)
    assert len(analyzer.history[5]) == 1


# --------------------------------------------------------------------------
# Reputation
# --------------------------------------------------------------------------


def test_reputation_update_matches_formula():
    manager = ReputationManager(initial_reputation=0.8, alpha=0.8)
    new_reputation = manager.update(client_id=0, round_num=1, current_trust=0.5)
    expected = 0.8 * 0.8 + 0.2 * 0.5
    assert new_reputation == pytest.approx(expected)


def test_reputation_clamped_to_unit_interval():
    manager = ReputationManager(initial_reputation=1.0, alpha=0.0)
    # alpha=0 means reputation instantly tracks current_trust each round.
    assert manager.update(0, 1, current_trust=1.5) == 1.0
    assert manager.update(0, 2, current_trust=-1.0) == 0.0


def test_reputation_manager_rejects_invalid_alpha():
    with pytest.raises(ValueError):
        ReputationManager(alpha=1.5)


def test_reputation_history_accumulates_per_client():
    manager = ReputationManager()
    manager.update(0, 1, 0.5)
    manager.update(0, 2, 0.6)
    assert [r.round_num for r in manager.history[0]] == [1, 2]


# --------------------------------------------------------------------------
# Update quality
# --------------------------------------------------------------------------


def test_update_quality_scores_are_bounded_and_present_for_every_client():
    evaluator = UpdateQualityEvaluator()
    global_state = {"w": torch.zeros(10)}
    client_states = {
        0: {"w": torch.ones(10) * 0.1},
        1: {"w": torch.ones(10) * -0.1},
    }
    sample_counts = {0: 100, 1: 50}
    average_losses = {0: 1.0, 1: 1.2}

    scores = evaluator.evaluate_round(client_states, sample_counts, global_state, average_losses)
    assert set(scores.keys()) == {0, 1}
    for score in scores.values():
        assert 0.0 <= score <= 1.0


def test_update_quality_single_client_gradient_norm_defaults_to_full_score():
    evaluator = UpdateQualityEvaluator()
    global_state = {"w": torch.zeros(4)}
    client_states = {0: {"w": torch.ones(4)}}
    scores = evaluator.evaluate_round(client_states, {0: 10}, global_state, {0: 1.0})
    assert scores[0] > 0.0


def test_update_quality_rewards_loss_improvement_across_rounds():
    evaluator = UpdateQualityEvaluator()
    global_state = {"w": torch.zeros(4)}
    client_states = {0: {"w": torch.ones(4) * 0.1}, 1: {"w": torch.ones(4) * 0.1}}
    sample_counts = {0: 10, 1: 10}

    evaluator.evaluate_round(client_states, sample_counts, global_state, {0: 1.0, 1: 1.0})
    improved = evaluator.evaluate_round(client_states, sample_counts, global_state, {0: 0.5, 1: 1.0})
    assert improved[0] > improved[1]


def test_update_quality_metric_weights_include_peer_cluster_similarity():
    evaluator = UpdateQualityEvaluator()
    assert "peer_cluster_similarity" in evaluator.metric_weights


def test_peer_cluster_similarity_score_rewards_alignment_with_cluster_mean():
    score = UpdateQualityEvaluator._peer_cluster_similarity_score(
        client_vector=torch.tensor([1.0, 0.0, 0.0]),
        cluster_mean_vector=torch.tensor([0.9, 0.1, 0.0]),
        cluster_size=2,
    )
    assert score > 0.9


def test_peer_cluster_similarity_score_penalizes_misalignment_with_cluster_mean():
    score = UpdateQualityEvaluator._peer_cluster_similarity_score(
        client_vector=torch.tensor([1.0, 0.0, 0.0]),
        cluster_mean_vector=torch.tensor([-1.0, 0.0, 0.0]),
        cluster_size=2,
    )
    assert score < 0.1


def test_peer_cluster_similarity_score_neutral_when_alone_in_cluster():
    score = UpdateQualityEvaluator._peer_cluster_similarity_score(
        client_vector=torch.tensor([1.0, 0.0, 0.0]),
        cluster_mean_vector=torch.tensor([1.0, 0.0, 0.0]),
        cluster_size=1,
    )
    assert score == 0.5


def test_compute_cluster_means_is_sample_weighted_average_per_cluster():
    client_vectors = {
        0: torch.tensor([1.0, 0.0]),
        1: torch.tensor([3.0, 0.0]),
        2: torch.tensor([0.0, 5.0]),
    }
    sample_counts = {0: 10, 1: 30, 2: 99}
    cluster_assignments = {0: 1, 1: 1, 2: 2}  # clients 0,1 share cluster 1; client 2 alone in cluster 2

    means = UpdateQualityEvaluator._compute_cluster_means(client_vectors, sample_counts, cluster_assignments)

    assert set(means.keys()) == {1, 2}
    # Cluster 1: sample-weighted mean of [1,0]*10 and [3,0]*30 over 40 samples -> [2.5, 0]
    assert torch.allclose(means[1], torch.tensor([2.5, 0.0]))
    assert torch.allclose(means[2], torch.tensor([0.0, 5.0]))


def test_update_quality_scores_still_bounded_with_more_participants():
    """Wiring-level check: evaluate_round produces one bounded score per
    client even with enough participants (>=3) to actually trigger clustering."""
    evaluator = UpdateQualityEvaluator()
    global_state = {"w": torch.zeros(6)}
    client_states = {
        0: {"w": torch.tensor([1.0, 0.0, 0.0, 0.0, 0.0, 0.0]) * 0.1},
        1: {"w": torch.tensor([0.95, 0.05, 0.0, 0.0, 0.0, 0.0]) * 0.1},
        2: {"w": torch.tensor([0.9, 0.1, 0.0, 0.0, 0.0, 0.0]) * 0.1},
        3: {"w": torch.tensor([0.0, 0.0, 0.0, 0.0, -1.0, 0.0]) * 0.1},
        4: {"w": torch.tensor([0.0, 0.0, 0.0, 0.0, -0.95, -0.05]) * 0.1},
    }
    sample_counts = {i: 10 for i in range(5)}
    average_losses = {i: 1.0 for i in range(5)}

    scores = evaluator.evaluate_round(client_states, sample_counts, global_state, average_losses)
    assert set(scores.keys()) == {0, 1, 2, 3, 4}
    for score in scores.values():
        assert 0.0 <= score <= 1.0


# --------------------------------------------------------------------------
# TrustEngine composition + determinism
# --------------------------------------------------------------------------


def _make_engine(seed: int = 42) -> TrustEngine:
    return TrustEngine(
        weights=TrustWeights(),
        initial_reputation=0.8,
        reputation_alpha=0.8,
        seed=seed,
    )


def _make_client(client_id: int, num_samples: int = 32) -> Client:
    dataset = make_synthetic_dataset(num_samples=num_samples, seed=client_id)
    model = build_model("cnn", in_channels=1, num_classes=10)
    return Client(client_id, model, dataset, torch.device("cpu"), batch_size=16, local_epochs=1)


def test_trust_engine_produces_bounded_scores_and_updates_client():
    engine = _make_engine()
    client = _make_client(0)
    engine.register_clients([0])
    global_state = client.get_weights()

    client.set_weights(global_state)
    result = client.train()

    records = engine.evaluate_round(1, [result], {0: client}, global_state)
    assert len(records) == 1
    record = records[0]
    for value in (
        record.identity, record.integrity, record.behaviour,
        record.reputation, record.update_quality, record.trust_score,
    ):
        assert 0.0 <= value <= 1.0

    assert client.trust_score == record.trust_score
    assert len(client.trust_history) == 1


def test_trust_engine_is_deterministic_given_same_seed():
    def run_two_rounds(seed: int):
        set_seed(seed)
        engine = _make_engine(seed=seed)
        client = _make_client(0)
        engine.register_clients([0])
        scores = []
        for round_num in range(1, 3):
            global_state = client.get_weights()
            client.set_weights(global_state)
            result = client.train()
            records = engine.evaluate_round(round_num, [result], {0: client}, global_state)
            scores.append(records[0].trust_score)
        return scores

    assert run_two_rounds(123) == pytest.approx(run_two_rounds(123))


def test_trust_engine_rejects_zero_total_weight():
    engine = TrustEngine(weights=TrustWeights(0, 0, 0, 0, 0))
    with pytest.raises(ValueError):
        engine._compose_trust(1.0, 1.0, 1.0, 1.0, 1.0)


# --------------------------------------------------------------------------
# CSV + plot generation
# --------------------------------------------------------------------------


def test_trust_csv_and_plots_generated(tmp_path):
    engine = _make_engine()
    client = _make_client(0)
    engine.register_clients([0])

    for round_num in range(1, 3):
        global_state = client.get_weights()
        client.set_weights(global_state)
        result = client.train()
        engine.evaluate_round(round_num, [result], {0: client}, global_state)

    results_dir = str(tmp_path)
    save_trust_scores(engine.records, results_dir)
    save_trust_history(engine.records, results_dir)
    save_trust_plots(engine.records, results_dir)

    assert (tmp_path / "trust_scores.csv").exists()
    assert (tmp_path / "trust_history.csv").exists()
    plots_dir = tmp_path / "trust_plots"
    for filename in (
        "average_trust_vs_round.png",
        "reputation_vs_round.png",
        "behaviour_vs_round.png",
        "update_quality_vs_round.png",
        "trust_distribution.png",
    ):
        assert (plots_dir / filename).exists()


# --------------------------------------------------------------------------
# Integration: trust evaluation must not change FedAvg's result at all
# --------------------------------------------------------------------------


def _build_server(num_clients: int, trust_engine, seed: int = 42) -> Server:
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
    if trust_engine is not None:
        trust_engine.register_clients(clients.keys())
    return Server(
        global_model, clients, test_dataset, device,
        clients_per_round=2, seed=seed, trust_engine=trust_engine,
    )


def _build_and_run(num_clients: int, trust_engine, seed: int = 42, num_rounds: int = 3):
    """Reset the seed, build a server, and run it to completion in one atomic step.

    Building both servers up front and running them one after the other
    would let the first server's training (DataLoader shuffling draws from
    PyTorch's *global* RNG) advance shared state that the second server's
    training then inherits — an artifact of comparing two runs in the same
    process, not something that happens in a real single-server simulation.
    Resetting immediately before each full build-and-run avoids that.
    """
    set_seed(seed)
    server = _build_server(num_clients, trust_engine, seed)
    history = [server.run_round(r) for r in range(1, num_rounds + 1)]
    return history, server


def test_trust_evaluation_does_not_change_fedavg_result():
    history_without, server_without_trust = _build_and_run(num_clients=4, trust_engine=None)
    history_with, server_with_trust = _build_and_run(num_clients=4, trust_engine=_make_engine())

    for round_without, round_with in zip(history_without, history_with):
        assert round_without["train_loss"] == pytest.approx(round_with["train_loss"])
        assert round_without["test_accuracy"] == pytest.approx(round_with["test_accuracy"])

    for key in server_without_trust.global_model.state_dict():
        assert torch.allclose(
            server_without_trust.global_model.state_dict()[key],
            server_with_trust.global_model.state_dict()[key],
        )


def test_server_without_trust_engine_leaves_client_trust_attributes_untouched():
    _, server = _build_and_run(num_clients=4, trust_engine=None, num_rounds=1)
    for client in server.clients.values():
        assert client.trust_history == []
        assert client.trust_score == 1.0  # untouched default

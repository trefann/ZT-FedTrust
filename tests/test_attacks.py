"""Tests for Phase 4 attack strategies, MaliciousClient, and attack assignment."""

import copy

import pytest
import torch

from attacks.base_attack import BaseAttack
from attacks.byzantine import ByzantineAttack
from attacks.label_flipping import DEFAULT_LABEL_MAP, LabelFlippingAttack
from attacks.model_poisoning import MAX_SCALE_FACTOR, ModelPoisoningAttack
from attacks.sybil import SybilAttack
from clients.client import Client
from clients.malicious_client import MaliciousClient, assign_attacks, known_identity_ids
from helpers import make_synthetic_dataset
from models.cnn import build_model
from server.selector import ClientSelector
from server.server import Server
from trust.trust_engine import TrustEngine, TrustWeights
from trust.trust_policy import TrustPolicy
from utils.metrics import set_seed


# --------------------------------------------------------------------------
# Label flipping
# --------------------------------------------------------------------------


def test_default_label_map_matches_spec_pairs():
    assert DEFAULT_LABEL_MAP[0] == 1
    assert DEFAULT_LABEL_MAP[1] == 0
    assert DEFAULT_LABEL_MAP[2] == 3
    assert DEFAULT_LABEL_MAP[3] == 2


def test_label_flipping_flips_labels_at_read_time():
    dataset = make_synthetic_dataset(num_samples=20, num_classes=10, seed=1)
    attack = LabelFlippingAttack()
    flipped = attack.apply_to_dataset(dataset)
    for i in range(len(dataset)):
        _, original_label = dataset[i]
        _, flipped_label = flipped[i]
        assert flipped_label == DEFAULT_LABEL_MAP[int(original_label)]


def test_label_flipping_never_mutates_underlying_dataset():
    dataset = make_synthetic_dataset(num_samples=20, num_classes=10, seed=1)
    original_labels = [int(dataset[i][1]) for i in range(len(dataset))]

    attack = LabelFlippingAttack()
    flipped = attack.apply_to_dataset(dataset)
    for i in range(len(flipped)):
        _ = flipped[i]  # force reads

    labels_after = [int(dataset[i][1]) for i in range(len(dataset))]
    assert labels_after == original_labels


def test_label_flipping_custom_map():
    attack = LabelFlippingAttack(label_map={0: 5})
    dataset = make_synthetic_dataset(num_samples=1, num_classes=10, seed=1)
    flipped = attack.apply_to_dataset(dataset)
    # Whatever the original label is, only 0 is remapped; everything else passes through.
    _, original_label = dataset[0]
    _, flipped_label = flipped[0]
    expected = 5 if int(original_label) == 0 else int(original_label)
    assert flipped_label == expected


# --------------------------------------------------------------------------
# Model poisoning
# --------------------------------------------------------------------------


def test_model_poisoning_scales_update_relative_to_reference():
    reference = {"w": torch.tensor([1.0, 1.0])}
    trained = {"w": torch.tensor([2.0, 3.0])}  # real update = [1.0, 2.0]
    attack = ModelPoisoningAttack(scale_factor=5.0)
    poisoned = attack.apply_to_update(trained, reference)
    expected = reference["w"] + (trained["w"] - reference["w"]) * 5.0
    assert torch.allclose(poisoned["w"], expected)


@pytest.mark.parametrize("bad_scale", [0.0, -0.0, float("inf"), float("nan"), MAX_SCALE_FACTOR + 1])
def test_model_poisoning_rejects_invalid_scale_factor(bad_scale):
    with pytest.raises(ValueError):
        ModelPoisoningAttack(scale_factor=bad_scale)


def test_model_poisoning_rejects_negative_noise_std():
    with pytest.raises(ValueError):
        ModelPoisoningAttack(noise_std=-1.0)


def test_model_poisoning_noise_is_deterministic_given_same_seed():
    reference = {"w": torch.zeros(10)}
    trained = {"w": torch.ones(10)}
    attack_a = ModelPoisoningAttack(scale_factor=1.0, noise_std=0.1, seed=7)
    attack_b = ModelPoisoningAttack(scale_factor=1.0, noise_std=0.1, seed=7)
    result_a = attack_a.apply_to_update(trained, reference)
    result_b = attack_b.apply_to_update(trained, reference)
    assert torch.allclose(result_a["w"], result_b["w"])


# --------------------------------------------------------------------------
# Byzantine
# --------------------------------------------------------------------------


def test_byzantine_negate_produces_opposite_update():
    reference = {"w": torch.tensor([1.0, 1.0])}
    trained = {"w": torch.tensor([2.0, 3.0])}  # real update = [1.0, 2.0]
    attack = ByzantineAttack(strategy="negate")
    poisoned = attack.apply_to_update(trained, reference)
    expected = reference["w"] - (trained["w"] - reference["w"])
    assert torch.allclose(poisoned["w"], expected)


def test_byzantine_random_produces_bounded_finite_update():
    reference = {"w": torch.ones(20) * 3.0}
    trained = {"w": torch.ones(20) * 3.5}
    attack = ByzantineAttack(strategy="random", seed=1)
    poisoned = attack.apply_to_update(trained, reference)
    assert torch.isfinite(poisoned["w"]).all()
    # Scaled to roughly the reference's own norm, not unbounded.
    assert torch.linalg.norm(poisoned["w"] - reference["w"]) == pytest.approx(
        torch.linalg.norm(reference["w"]).item(), rel=0.05
    )


def test_byzantine_rejects_unknown_strategy():
    with pytest.raises(ValueError):
        ByzantineAttack(strategy="not-a-real-strategy")


# --------------------------------------------------------------------------
# Sybil
# --------------------------------------------------------------------------


def test_sybil_delegates_dataset_and_update_hooks():
    underlying = LabelFlippingAttack()
    sybil = SybilAttack(underlying)
    dataset = make_synthetic_dataset(num_samples=5, seed=1)
    wrapped = sybil.apply_to_dataset(dataset)
    assert type(wrapped) is type(underlying.apply_to_dataset(dataset))

    reference = {"w": torch.zeros(3)}
    trained = {"w": torch.ones(3)}
    assert torch.equal(sybil.apply_to_update(trained, reference)["w"], trained["w"])  # no-op passthrough


# --------------------------------------------------------------------------
# BaseAttack defaults
# --------------------------------------------------------------------------


def test_base_attack_hooks_are_no_ops_by_default():
    attack = BaseAttack()
    dataset = make_synthetic_dataset(num_samples=3, seed=1)
    assert attack.apply_to_dataset(dataset) is dataset
    state = {"w": torch.tensor([1.0])}
    assert attack.apply_to_update(state, state) is state


# --------------------------------------------------------------------------
# MaliciousClient
# --------------------------------------------------------------------------


def _make_malicious(attack, attack_probability=1.0, num_samples=32, seed=1):
    dataset = make_synthetic_dataset(num_samples=num_samples, seed=seed)
    model = build_model("cnn", in_channels=1, num_classes=10)
    inner = Client(0, model, dataset, torch.device("cpu"), batch_size=16, local_epochs=1)
    return MaliciousClient(inner, attack, attack_probability=attack_probability, seed=seed)


def test_malicious_client_proxies_trust_attributes_both_ways():
    malicious = _make_malicious(LabelFlippingAttack())
    malicious.trust_score = 0.42
    assert malicious.client.trust_score == 0.42
    malicious.client.reputation_score = 0.77
    assert malicious.reputation_score == 0.77


def test_malicious_client_restores_original_dataset_after_training():
    malicious = _make_malicious(LabelFlippingAttack())
    original_dataset = malicious.client.dataset
    malicious.set_weights(malicious.get_weights())
    malicious.train()
    assert malicious.client.dataset is original_dataset


def test_malicious_client_applies_update_attack_when_triggered():
    malicious = _make_malicious(ModelPoisoningAttack(scale_factor=10.0), attack_probability=1.0)
    global_state = malicious.get_weights()
    malicious.set_weights(global_state)
    result = malicious.train()
    assert malicious.last_round_attacked is True
    # Poisoned update should differ meaningfully from the reference weights.
    diff = sum(
        torch.linalg.norm(result.state_dict[k].float() - global_state[k].float()).item()
        for k in global_state
    )
    assert diff > 0


def test_malicious_client_never_attacks_when_probability_zero():
    malicious = _make_malicious(ByzantineAttack(strategy="negate"), attack_probability=0.0)
    for _ in range(3):
        malicious.set_weights(malicious.get_weights())
        malicious.train()
        assert malicious.last_round_attacked is False


def test_malicious_client_rejects_invalid_probability():
    with pytest.raises(ValueError):
        _make_malicious(LabelFlippingAttack(), attack_probability=1.5)


def test_malicious_client_raises_on_non_finite_attack_output():
    class _ExplodingAttack(BaseAttack):
        name = "exploding"

        def apply_to_update(self, state_dict, reference_state_dict):
            return {k: v * float("inf") for k, v in state_dict.items()}

    malicious = _make_malicious(_ExplodingAttack(), attack_probability=1.0)
    malicious.set_weights(malicious.get_weights())
    with pytest.raises(ValueError):
        malicious.train()


# --------------------------------------------------------------------------
# Attack assignment
# --------------------------------------------------------------------------


def _make_client_pool(num_clients: int) -> dict:
    clients = {}
    for cid in range(num_clients):
        dataset = make_synthetic_dataset(num_samples=16, seed=cid)
        model = build_model("cnn", in_channels=1, num_classes=10)
        clients[cid] = Client(cid, model, dataset, torch.device("cpu"))
    return clients


def test_assign_attacks_fraction_correctness():
    clients = _make_client_pool(20)
    result = assign_attacks(clients, 0.20, "label_flipping", {"label_flipping": {"probability": 1.0}}, seed=42)
    malicious_count = sum(1 for c in result.values() if isinstance(c, MaliciousClient))
    assert malicious_count == 4


def test_assign_attacks_is_deterministic_given_same_seed():
    def malicious_ids():
        clients = _make_client_pool(20)
        result = assign_attacks(
            clients, 0.25, "label_flipping", {"label_flipping": {"probability": 1.0}}, seed=7
        )
        return sorted(cid for cid, c in result.items() if isinstance(c, MaliciousClient))

    assert malicious_ids() == malicious_ids()


def test_assign_attacks_sybil_grouping():
    clients = _make_client_pool(12)
    settings = {"sybil": {"probability": 1.0, "number_of_identities": 3}}
    result = assign_attacks(clients, 0.5, "sybil", settings, seed=3)  # 6 malicious -> 2 groups of 3

    malicious = {cid: c for cid, c in result.items() if isinstance(c, MaliciousClient)}
    assert len(malicious) == 6
    groups = {}
    for cid, client in malicious.items():
        groups.setdefault(client.sybil_group_id, []).append(cid)
        assert client.parent_attacker_id in malicious
    assert all(len(members) == 3 for members in groups.values())


def test_assign_attacks_rejects_invalid_fraction():
    clients = _make_client_pool(4)
    with pytest.raises(ValueError):
        assign_attacks(clients, 1.5, "label_flipping", {}, seed=1)


def test_known_identity_ids_excludes_non_parent_sybils():
    clients = _make_client_pool(6)
    settings = {"sybil": {"probability": 1.0, "number_of_identities": 3}}
    result = assign_attacks(clients, 0.5, "sybil", settings, seed=3)  # one group of 3
    known = known_identity_ids(result)

    sybils = {cid: c for cid, c in result.items() if isinstance(c, MaliciousClient)}
    parent_id = next(iter(sybils.values())).parent_attacker_id
    assert parent_id in known
    non_parent_sybils = [cid for cid, c in sybils.items() if cid != parent_id]
    assert all(cid not in known for cid in non_parent_sybils)
    # Every benign client remains known.
    benign_ids = [cid for cid, c in result.items() if not isinstance(c, MaliciousClient)]
    assert all(cid in known for cid in benign_ids)


# --------------------------------------------------------------------------
# Trust responds to malicious behaviour (integration, synthetic data)
# --------------------------------------------------------------------------


def test_trust_score_degrades_for_a_byzantine_client_relative_to_benign():
    """Ground truth is only read here, for the test's own assertion — never by Server/TrustEngine.

    Uses a benign majority (4 benign, 1 malicious — the spec's typical 20%
    malicious fraction), not a 50/50 split: the update-quality signal
    compares each client's update against the *sample-weighted mean* of all
    participants, so with only 2 total clients the malicious update would
    already dominate its own comparison baseline, masking the effect this
    test is checking for.
    """
    set_seed(11)
    device = torch.device("cpu")
    global_model = build_model("cnn", in_channels=1, num_classes=10)
    test_dataset = make_synthetic_dataset(num_samples=32, num_classes=10, seed=100)

    clients = {}
    for cid in range(4):
        dataset = make_synthetic_dataset(num_samples=64, num_classes=10, seed=cid)
        clients[cid] = Client(cid, global_model, dataset, device, batch_size=16, learning_rate=0.1)

    malicious_dataset = make_synthetic_dataset(num_samples=64, num_classes=10, seed=99)
    malicious_inner = Client(4, global_model, malicious_dataset, device, batch_size=16, learning_rate=0.1)
    malicious_client = MaliciousClient(
        malicious_inner, ByzantineAttack(strategy="negate"), attack_probability=1.0, seed=99
    )
    clients[4] = malicious_client

    trust_engine = TrustEngine(weights=TrustWeights(), seed=11)
    trust_engine.register_clients(clients.keys())
    policy = TrustPolicy(threshold_mode="fixed", threshold=0.0)  # never reject, just observe scores
    selector = ClientSelector(policy, participation_fraction=1.0, seed=11)

    server = Server(
        global_model, clients, test_dataset, device,
        clients_per_round=5, seed=11, trust_engine=trust_engine, client_selector=selector,
        aggregation_strategy="trust_fedavg",
    )
    for round_num in range(1, 4):
        server.run_round(round_num)

    average_benign_trust = sum(clients[cid].trust_score for cid in range(4)) / 4
    assert average_benign_trust > malicious_client.trust_score

"""Offline tests for evaluation/experiment_runner.py's config-level logic.

`ExperimentRunner.run()` needs a real dataset download (MNIST/CIFAR-10) to
exercise end-to-end, which the rest of this test suite deliberately avoids
(see tests/helpers.py) to stay fast and network-free. The full three-mode
pipeline against real data is exercised separately as part of Phase 4's
"Final Validation" real-MNIST runs. What's tested here — mode validation,
config-to-attack-settings mapping, and that attack assignment (already unit
tested in test_attacks.py) is what guarantees fairness across modes — is
everything that doesn't require a dataset.
"""

import pytest

from evaluation.experiment_runner import EXPERIMENT_MODES, ExperimentRunner, _attack_settings
from simulation.run import (
    AggregationConfig,
    AttackConfig,
    ByzantineConfig,
    DataConfig,
    FederatedConfig,
    LabelFlippingConfig,
    LoggingConfig,
    ModelConfig,
    ModelPoisoningConfig,
    SimulationConfig,
    SybilConfig,
    TrustConfig,
)


def _base_config(**attack_overrides) -> SimulationConfig:
    attacks = AttackConfig(
        enabled=True,
        malicious_client_fraction=0.2,
        active_attack="label_flipping",
        label_flipping=LabelFlippingConfig(enabled=True, probability=1.0, label_map={0: 1}),
        model_poisoning=ModelPoisoningConfig(enabled=False, probability=1.0, scale_factor=5.0),
        byzantine=ByzantineConfig(enabled=False, probability=1.0, strategy="random"),
        sybil=SybilConfig(enabled=False, number_of_identities=3, probability=1.0),
    )
    for key, value in attack_overrides.items():
        setattr(attacks, key, value)
    return SimulationConfig(
        seed=42,
        device="cpu",
        data=DataConfig(
            dataset="mnist", data_dir="./data_cache", num_clients=4,
            partition_type="iid", dirichlet_alpha=0.5, batch_size=32,
        ),
        model=ModelConfig(name="cnn"),
        federated=FederatedConfig(
            num_rounds=1, clients_per_round=2, local_epochs=1, learning_rate=0.01, momentum=0.9
        ),
        logging=LoggingConfig(log_dir="./results/logs", log_level="INFO", results_dir="./results"),
        trust=TrustConfig(),
        aggregation=AggregationConfig(strategy="trust_fedavg"),
        attacks=attacks,
    )


def test_experiment_modes_are_exactly_the_spec_three():
    assert EXPERIMENT_MODES == ("baseline", "fedavg_attack", "ztfedtrust_attack")


def test_experiment_runner_rejects_unknown_mode():
    runner = ExperimentRunner(_base_config())
    with pytest.raises(ValueError):
        runner.run("not-a-real-mode")


def test_attack_settings_flattens_config_correctly():
    config = _base_config()
    settings = _attack_settings(config)
    assert settings["label_flipping"] == {"probability": 1.0, "label_map": {0: 1}}
    assert settings["model_poisoning"]["scale_factor"] == 5.0
    assert settings["byzantine"]["strategy"] == "random"
    assert settings["sybil"]["number_of_identities"] == 3

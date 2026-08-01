"""Byzantine attack: Phase 4 update-level attack.

Produces a deliberately abnormal update while remaining numerically stable
— bounded in magnitude, matched to the reference weights' own scale, rather
than an arbitrary unbounded value.
"""

from __future__ import annotations

from typing import Dict

import torch

from attacks.base_attack import BaseAttack

SUPPORTED_BYZANTINE_STRATEGIES = ("random", "negate")


class ByzantineAttack(BaseAttack):
    """Replaces a client's real update with a clearly abnormal one.

    Args:
        strategy: `"random"` — replace the update with Gaussian noise,
            rescaled to match the reference weights' own norm (abnormal in
            direction, but not wildly out of scale); `"negate"` — send the
            exact opposite of the real update (`update = -normal_update`).
        seed: seed for `"random"`'s own noise RNG, independent of every
            other seeded stream in the simulation.
    """

    name = "byzantine"

    def __init__(self, strategy: str = "random", seed: int = 42) -> None:
        if strategy not in SUPPORTED_BYZANTINE_STRATEGIES:
            raise ValueError(
                f"strategy must be one of {SUPPORTED_BYZANTINE_STRATEGIES}, got {strategy!r}"
            )
        self.strategy = strategy
        self._generator = torch.Generator().manual_seed(seed)

    def apply_to_update(
        self, state_dict: Dict[str, torch.Tensor], reference_state_dict: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        poisoned: Dict[str, torch.Tensor] = {}
        for key, trained_value in state_dict.items():
            reference_value = reference_state_dict[key].float()

            if self.strategy == "negate":
                real_update = trained_value.float() - reference_value
                poisoned[key] = (reference_value - real_update).to(trained_value.dtype)
            else:  # "random"
                reference_scale = torch.linalg.norm(reference_value) + 1e-8
                noise = torch.randn(reference_value.shape, generator=self._generator)
                noise_norm = torch.linalg.norm(noise) + 1e-8
                scaled_noise = noise / noise_norm * reference_scale
                poisoned[key] = (reference_value + scaled_noise).to(trained_value.dtype)

        return poisoned

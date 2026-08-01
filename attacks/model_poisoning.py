"""Model poisoning attack: Phase 4 update-level attack.

Scales a client's real local update by a configurable factor (optionally
adding noise), expressed relative to what the client actually learned this
round rather than as an arbitrary absolute weight value:

    real_update = trained_weights - reference_weights
    malicious_update = real_update * scale_factor (+ noise)
    poisoned_weights = reference_weights + malicious_update
"""

from __future__ import annotations

from typing import Dict

import torch

from attacks.base_attack import BaseAttack

# Upper bound on |scale_factor|, enforced both here and in config validation
# (simulation/run.py), so a misconfigured value can't produce exploding
# weights that then corrupt the global model.
MAX_SCALE_FACTOR = 100.0


class ModelPoisoningAttack(BaseAttack):
    """Scales (and optionally adds noise to) a client's local update.

    Args:
        scale_factor: multiplier applied to the client's real update.
            Validated to be finite and non-zero, within `MAX_SCALE_FACTOR`
            in magnitude.
        noise_std: standard deviation of optional zero-mean Gaussian noise
            added to the scaled update. Must be >= 0.
        seed: seed for this attack's own noise-generation RNG, independent
            of every other seeded stream in the simulation.
    """

    name = "model_poisoning"

    def __init__(self, scale_factor: float = 5.0, noise_std: float = 0.0, seed: int = 42) -> None:
        if not torch.isfinite(torch.tensor(float(scale_factor))).item():
            raise ValueError(f"scale_factor must be finite, got {scale_factor}")
        if not 0.0 < abs(scale_factor) <= MAX_SCALE_FACTOR:
            raise ValueError(
                f"scale_factor must be in (0, {MAX_SCALE_FACTOR}] in magnitude, got {scale_factor}"
            )
        if noise_std < 0.0:
            raise ValueError(f"noise_std must be >= 0, got {noise_std}")
        self.scale_factor = scale_factor
        self.noise_std = noise_std
        self._generator = torch.Generator().manual_seed(seed)

    def apply_to_update(
        self, state_dict: Dict[str, torch.Tensor], reference_state_dict: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        poisoned: Dict[str, torch.Tensor] = {}
        for key, trained_value in state_dict.items():
            reference_value = reference_state_dict[key].float()
            real_update = trained_value.float() - reference_value
            malicious_update = real_update * self.scale_factor
            if self.noise_std > 0.0:
                noise = torch.randn(malicious_update.shape, generator=self._generator) * self.noise_std
                malicious_update = malicious_update + noise
            poisoned[key] = (reference_value + malicious_update).to(trained_value.dtype)
        return poisoned

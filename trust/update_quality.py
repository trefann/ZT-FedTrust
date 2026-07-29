"""Update quality assessment: Zero Trust dimension 5.

Answers "is this client's model update actually useful?" Unlike the other
four dimensions, this one is computed from real training signal rather than
simulated — the client's returned weights and reported training loss — so it
is the dimension most directly indicative of an honest, well-behaved update.

Three metrics are combined into one score, each mapped to [0, 1]:

  - training loss improvement vs. this client's own previous round
  - cosine similarity between the client's update direction and the round's
    sample-weighted mean update direction (a proxy for "consensus" before
    FedAvg aggregation actually runs)
  - update magnitude (L2 norm of the pseudo-gradient), min-max normalised
    against the other participating clients this round

Not yet implemented: validation-accuracy improvement. That would need each
`Client` to hold a local validation split, which Phase 1's `Client` doesn't
have — a natural Phase 3+ addition. The `metric_weights` design below means
adding it later is a matter of computing the metric and adding one entry to
`metric_weights`; `evaluate_round` renormalises automatically over whatever
metrics are present, so no other code needs to change.
"""

from __future__ import annotations

from typing import Dict, List

import torch


def _flatten_state_dict(state_dict: Dict[str, torch.Tensor]) -> torch.Tensor:
    """Concatenate every tensor in a state dict into one flat 1-D vector.

    Relies on `state_dict` preserving insertion order (guaranteed since
    Python 3.7 dicts, and PyTorch state dicts follow module definition
    order), so two state dicts from the same architecture flatten to
    directly comparable vectors.
    """
    return torch.cat([tensor.detach().float().reshape(-1) for tensor in state_dict.values()])


class UpdateQualityEvaluator:
    """Scores each client's local model update for one round, in [0, 1].

    Args:
        loss_improvement_scale: controls how sharply loss improvement
            saturates toward 0 or 1 (see `_loss_improvement_score`); larger
            values make the score more sensitive to small loss changes.
    """

    def __init__(self, loss_improvement_scale: float = 5.0) -> None:
        self.loss_improvement_scale = loss_improvement_scale
        # Relative weight of each metric in the combined score. Adding a new
        # metric is just: compute it in `evaluate_round` and add it here.
        self.metric_weights: Dict[str, float] = {
            "loss_improvement": 0.40,
            "cosine_similarity": 0.35,
            "gradient_norm": 0.25,
        }
        self._previous_loss: Dict[int, float] = {}

    def evaluate_round(
        self,
        client_states: Dict[int, Dict[str, torch.Tensor]],
        sample_counts: Dict[int, int],
        global_state_before: Dict[str, torch.Tensor],
        average_losses: Dict[int, float],
    ) -> Dict[int, float]:
        """Score every participating client's update quality for one round.

        Args:
            client_states: each client's returned local model weights.
            sample_counts: each client's local dataset size (for the
                sample-weighted mean update direction, matching how FedAvg
                itself will weight these same updates).
            global_state_before: the global model weights clients trained
                from at the start of this round (i.e. before aggregation).
            average_losses: each client's average local training loss this round.
        """
        global_vector = _flatten_state_dict(global_state_before)
        client_vectors = {
            client_id: _flatten_state_dict(state) - global_vector
            for client_id, state in client_states.items()
        }

        total_samples = sum(sample_counts[client_id] for client_id in client_vectors)
        mean_vector = sum(
            (client_vectors[client_id] * (sample_counts[client_id] / total_samples))
            for client_id in client_vectors
        )

        norms = {client_id: float(torch.linalg.norm(vec)) for client_id, vec in client_vectors.items()}
        norm_values = list(norms.values())

        scores: Dict[int, float] = {}
        for client_id, vector in client_vectors.items():
            metrics = {
                "loss_improvement": self._loss_improvement_score(client_id, average_losses[client_id]),
                "cosine_similarity": self._cosine_similarity_score(vector, mean_vector),
                "gradient_norm": self._gradient_norm_score(norms[client_id], norm_values),
            }
            self._previous_loss[client_id] = average_losses[client_id]

            total_weight = sum(self.metric_weights.values())
            combined = sum(metrics[name] * self.metric_weights[name] for name in metrics) / total_weight
            scores[client_id] = float(min(max(combined, 0.0), 1.0))

        return scores

    def _loss_improvement_score(self, client_id: int, average_loss: float) -> float:
        """Score how much this client's loss improved since its previous round.

        Maps the (unbounded) improvement through `tanh` so a score always
        lands in [0, 1]: large improvements saturate toward 1, large
        regressions toward 0, and no prior data (first round) is neutral.
        """
        previous_loss = self._previous_loss.get(client_id)
        if previous_loss is None:
            return 0.5
        improvement = previous_loss - average_loss
        scaled = torch.tanh(torch.tensor(improvement * self.loss_improvement_scale))
        return float(0.5 + 0.5 * scaled.item())

    @staticmethod
    def _cosine_similarity_score(client_vector: torch.Tensor, mean_vector: torch.Tensor) -> float:
        """Score how aligned this client's update is with the round's mean update.

        Cosine similarity naturally lands in [-1, 1]; rescaled to [0, 1] so
        it combines cleanly with the other metrics.
        """
        if torch.linalg.norm(client_vector) < 1e-12 or torch.linalg.norm(mean_vector) < 1e-12:
            return 0.5  # no meaningful direction to compare against
        cosine = torch.nn.functional.cosine_similarity(
            client_vector.unsqueeze(0), mean_vector.unsqueeze(0)
        ).item()
        return float((cosine + 1.0) / 2.0)

    @staticmethod
    def _gradient_norm_score(client_norm: float, round_norms: List[float]) -> float:
        """Score update magnitude via min-max normalisation against this round's peers.

        A smaller relative norm scores higher: unusually large updates are
        treated as a weaker (not necessarily malicious — that's a later
        phase's job) signal. With one client, or when every norm is equal,
        there's no meaningful spread to rank against, so it defaults to 1.0.
        """
        if len(round_norms) <= 1:
            return 1.0
        lowest, highest = min(round_norms), max(round_norms)
        if highest - lowest < 1e-12:
            return 1.0
        return float(1.0 - (client_norm - lowest) / (highest - lowest))

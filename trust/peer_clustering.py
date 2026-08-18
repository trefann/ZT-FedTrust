"""Behavior-derived peer grouping for the update-quality trust dimension.

Groups this round's participating clients by the similarity of their own
update vectors — never by a declared or self-reported client attribute
(e.g. "device type"). This is deliberate: a declared label could be
spoofed by an attacker, and behavior-derived grouping works even in a
federation where no such label exists at all, which is what makes it
useful specifically on genuinely heterogeneous real-device federations
(e.g. N-BaIoT, TON_IoT devices) where a single population-wide baseline
stops being meaningful (see `docs/superpowers/specs/2026-08-17-peer-cluster-trust-design.md`).

Only ever reads update vectors — no client identity, no `is_malicious`,
no declared metadata — matching this project's structural ground-truth
isolation rule used throughout `attacks/` and `clients/malicious_client.py`.
"""

from __future__ import annotations

from typing import Dict

import numpy as np
import torch
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import pdist


def cluster_clients(client_vectors: Dict[int, torch.Tensor], min_clients: int = 3) -> Dict[int, int]:
    """Group clients by update-vector similarity via hierarchical clustering.

    Uses cosine distance and average linkage. The merge threshold is
    adaptive — the round's own mean pairwise distance — rather than a
    fixed constant, so it doesn't need per-dataset tuning: on a
    homogeneous round (every update points a similar direction), this
    naturally collapses toward one cluster containing everyone, the same
    way this project's trust-selection threshold is already
    self-calibrating (`mean - 0.5*stdev` of the round's own trust
    distribution) rather than a hardcoded number.

    With fewer than `min_clients` participants there isn't enough
    pairwise-distance signal to calibrate a threshold from (a single
    distance value isn't a distribution), so every client is placed in
    its own singleton cluster — callers that treat a singleton cluster as
    "no peers available" (see `update_quality.py`'s
    `_peer_cluster_similarity_score`) then correctly fall back to a
    neutral score for this case too, with no separate special-casing needed.
    """
    client_ids = sorted(client_vectors)
    if len(client_ids) < min_clients:
        return {client_id: index for index, client_id in enumerate(client_ids)}

    matrix = np.stack([client_vectors[client_id].detach().cpu().numpy() for client_id in client_ids])
    distances = pdist(matrix, metric="cosine")
    # A zero-vector client makes cosine distance NaN (division by a zero
    # norm) rather than raise — treat it as maximally distant (cosine
    # distance's own max value) instead of letting NaN propagate into the
    # linkage/threshold math.
    distances = np.nan_to_num(distances, nan=2.0)

    linkage_matrix = linkage(distances, method="average")
    threshold = float(distances.mean())
    cluster_ids = fcluster(linkage_matrix, t=threshold, criterion="distance")

    return {client_id: int(cluster_id) for client_id, cluster_id in zip(client_ids, cluster_ids)}

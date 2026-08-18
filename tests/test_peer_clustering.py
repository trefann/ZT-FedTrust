"""Tests for peer-cluster-relative grouping of client update vectors.

Clustering is behavior-derived (from the update vectors themselves), never
from a declared/self-reported client attribute — see the module docstring
in trust/peer_clustering.py for why.
"""

import torch

from trust.peer_clustering import cluster_clients


def test_two_distinct_groups_of_aligned_vectors_form_two_clusters():
    vectors = {
        0: torch.tensor([1.0, 0.0, 0.0]),
        1: torch.tensor([0.9, 0.1, 0.0]),
        2: torch.tensor([-1.0, 0.0, 0.0]),
        3: torch.tensor([-0.9, -0.1, 0.0]),
    }
    clusters = cluster_clients(vectors)
    assert clusters[0] == clusters[1]
    assert clusters[2] == clusters[3]
    assert clusters[0] != clusters[2]


def test_returns_a_cluster_id_for_every_input_client():
    vectors = {i: torch.randn(5) for i in range(6)}
    clusters = cluster_clients(vectors)
    assert set(clusters.keys()) == set(vectors.keys())


def test_identical_vectors_all_land_in_one_cluster():
    vectors = {i: torch.tensor([1.0, 2.0, 3.0]) for i in range(5)}
    clusters = cluster_clients(vectors)
    assert len(set(clusters.values())) == 1


def test_fewer_than_min_clients_get_distinct_singleton_clusters():
    vectors = {0: torch.tensor([1.0, 0.0]), 1: torch.tensor([1.0, 0.0])}
    clusters = cluster_clients(vectors, min_clients=3)
    assert clusters[0] != clusters[1]


def test_zero_vector_does_not_crash_clustering():
    vectors = {
        0: torch.zeros(4),
        1: torch.tensor([1.0, 0.0, 0.0, 0.0]),
        2: torch.tensor([0.9, 0.1, 0.0, 0.0]),
    }
    clusters = cluster_clients(vectors)
    assert set(clusters.keys()) == {0, 1, 2}


def test_single_client_gets_its_own_cluster():
    vectors = {0: torch.tensor([1.0, 0.0, 0.0])}
    clusters = cluster_clients(vectors)
    assert clusters == {0: 0}

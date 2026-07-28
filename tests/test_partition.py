"""Tests for client data partitioning: coverage, balance, and subset wiring."""

from data.partition import make_client_subsets, partition_dirichlet, partition_iid
from helpers import make_synthetic_dataset


def test_partition_iid_covers_all_indices_without_overlap():
    dataset = make_synthetic_dataset(num_samples=100, seed=1)
    shards = partition_iid(dataset, num_clients=5, seed=42)
    all_indices = [i for shard in shards.values() for i in shard]
    assert sorted(all_indices) == list(range(100))


def test_partition_iid_equal_data_distribution():
    dataset = make_synthetic_dataset(num_samples=103, seed=1)
    shards = partition_iid(dataset, num_clients=5, seed=42)
    sizes = [len(v) for v in shards.values()]
    assert max(sizes) - min(sizes) <= 1


def test_partition_dirichlet_covers_all_indices_without_overlap():
    dataset = make_synthetic_dataset(num_samples=200, num_classes=10, seed=2)
    shards = partition_dirichlet(dataset, num_clients=4, alpha=0.5, seed=42)
    all_indices = [i for shard in shards.values() for i in shard]
    assert sorted(all_indices) == list(range(200))


def test_make_client_subsets_lengths_match_indices():
    dataset = make_synthetic_dataset(num_samples=50, seed=3)
    shards = partition_iid(dataset, num_clients=5, seed=42)
    subsets = make_client_subsets(dataset, shards)
    for client_id, subset in subsets.items():
        assert len(subset) == len(shards[client_id])

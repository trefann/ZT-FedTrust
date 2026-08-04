"""Tests for the shared IoT experiment builders and, in later tasks, per-dataset loaders."""

import numpy as np
import pandas as pd
import pytest

from data.iot_datasets import _build_natural_client_experiment, _build_partitioned_experiment


def _make_partitioned_df(num_rows=200, seed=0):
    rng = np.random.default_rng(seed)
    labels = rng.choice(["normal", "ddos", "backdoor"], size=num_rows, p=[0.6, 0.25, 0.15])
    return pd.DataFrame(
        {
            "proto": rng.choice(["tcp", "udp"], size=num_rows),
            "duration": rng.random(num_rows) * 10,
            "bytes": rng.integers(0, 1000, size=num_rows),
            "label": labels,
        }
    )


class TestBuildPartitionedExperiment:
    def test_produces_requested_number_of_clients(self):
        df = _make_partitioned_df()
        result = _build_partitioned_experiment(
            df,
            label_column="label",
            benign_label="normal",
            num_clients=5,
            partition_type="iid",
            dirichlet_alpha=0.5,
            max_rows=None,
            seed=42,
        )
        assert len(result.train_client_datasets) == 5

    def test_input_dim_excludes_label_column(self):
        df = _make_partitioned_df()
        result = _build_partitioned_experiment(
            df, "label", "normal", num_clients=3, partition_type="iid",
            dirichlet_alpha=0.5, max_rows=None, seed=42,
        )
        assert result.input_dim == 3  # proto, duration, bytes

    def test_num_classes_matches_label_vocabulary(self):
        df = _make_partitioned_df()
        result = _build_partitioned_experiment(
            df, "label", "normal", num_clients=3, partition_type="iid",
            dirichlet_alpha=0.5, max_rows=None, seed=42,
        )
        assert result.num_classes == 3
        assert result.class_names == ["backdoor", "ddos", "normal"]

    def test_attack_pool_excludes_benign_rows(self):
        df = _make_partitioned_df()
        result = _build_partitioned_experiment(
            df, "label", "normal", num_clients=3, partition_type="iid",
            dirichlet_alpha=0.5, max_rows=None, seed=42,
        )
        assert result.attack_pool is not None
        assert len(result.attack_pool) > 0
        assert len(result.attack_pool) < len(df)

    def test_attack_pool_none_when_benign_label_absent(self):
        df = _make_partitioned_df()
        result = _build_partitioned_experiment(
            df, "label", benign_label=None, num_clients=3, partition_type="iid",
            dirichlet_alpha=0.5, max_rows=None, seed=42,
        )
        assert result.attack_pool is None

    def test_max_rows_caps_dataset_size(self):
        df = _make_partitioned_df(num_rows=200)
        result = _build_partitioned_experiment(
            df, "label", "normal", num_clients=2, partition_type="iid",
            dirichlet_alpha=0.5, max_rows=50, seed=42,
        )
        total_train = sum(len(d) for d in result.train_client_datasets.values())
        assert total_train + len(result.test_dataset) == 50

    def test_dirichlet_partition_type_works(self):
        df = _make_partitioned_df()
        result = _build_partitioned_experiment(
            df, "label", "normal", num_clients=4, partition_type="dirichlet",
            dirichlet_alpha=0.5, max_rows=None, seed=42,
        )
        assert len(result.train_client_datasets) == 4


class TestBuildNaturalClientExperiment:
    def _device_frames(self):
        rng = np.random.default_rng(1)
        return {
            "device_a": pd.DataFrame(
                {
                    "sensor_x": rng.random(40),
                    "type": rng.choice(["normal", "ddos"], size=40, p=[0.7, 0.3]),
                }
            ),
            "device_b": pd.DataFrame(
                {
                    "sensor_x": rng.random(30),
                    "type": rng.choice(["normal", "backdoor"], size=30, p=[0.7, 0.3]),
                }
            ),
        }

    def test_one_client_per_device(self):
        result = _build_natural_client_experiment(
            self._device_frames(), label_column="type", benign_label="normal",
            max_rows_per_device=None, seed=7,
        )
        assert len(result.train_client_datasets) == 2

    def test_class_vocabulary_spans_all_devices(self):
        result = _build_natural_client_experiment(
            self._device_frames(), label_column="type", benign_label="normal",
            max_rows_per_device=None, seed=7,
        )
        assert result.class_names == ["backdoor", "ddos", "normal"]

    def test_test_dataset_pools_all_devices(self):
        result = _build_natural_client_experiment(
            self._device_frames(), label_column="type", benign_label="normal",
            max_rows_per_device=None, seed=7,
        )
        assert len(result.test_dataset) > 0

    def test_max_rows_per_device_caps_each_device_independently(self):
        result = _build_natural_client_experiment(
            self._device_frames(), label_column="type", benign_label="normal",
            max_rows_per_device=10, seed=7,
        )
        for dataset in result.train_client_datasets.values():
            assert len(dataset) <= 10

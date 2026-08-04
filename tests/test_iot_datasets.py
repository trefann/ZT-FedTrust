"""Tests for the shared IoT experiment builders and, in later tasks, per-dataset loaders."""

import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from data.iot_datasets import _build_natural_client_experiment, _build_partitioned_experiment, load_ciciot2023, load_unsw_nb15


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


class TestLoadCiciot2023:
    def _write_fixture(self, tmp_path: Path) -> Path:
        content = (
            "flow_duration,Header_Length,Protocol Type,Rate,label\n"
            "0,54,6,17.3,DDoS-TCP_Flood\n"
            "0,53,5,0.3,DDoS-TCP_Flood\n"
            "1,60,17,5.0,DDoS-UDP_Flood\n"
            "2,70,1,9.0,DDoS-ICMP_Flood\n"
            "0,54,6,3.9,DDoS-TCP_Flood\n"
            "1,61,17,4.4,DDoS-UDP_Flood\n"
            "2,71,1,8.8,DDoS-ICMP_Flood\n"
            "0,55,6,2.1,DDoS-TCP_Flood\n"
        )
        data_dir = tmp_path / "ZT-Fedtrust Datasets"
        data_dir.mkdir()
        (data_dir / "CICIoT2023 (2).csv").write_text(content, encoding="utf-8")
        return data_dir

    def test_loads_and_produces_expected_shape(self, tmp_path):
        data_dir = self._write_fixture(tmp_path)
        result = load_ciciot2023(
            str(data_dir), num_clients=2, partition_type="iid",
            dirichlet_alpha=0.5, max_rows=None, seed=1,
        )
        assert result.input_dim == 4  # flow_duration, Header_Length, Protocol Type, Rate
        assert result.num_classes == 3
        assert len(result.train_client_datasets) == 2

    def test_no_benign_traffic_means_no_attack_pool(self, tmp_path):
        data_dir = self._write_fixture(tmp_path)
        result = load_ciciot2023(
            str(data_dir), num_clients=2, partition_type="iid",
            dirichlet_alpha=0.5, max_rows=None, seed=1,
        )
        assert result.attack_pool is None


class TestLoadUnswNb15:
    def _write_fixture(self, tmp_path):
        header = (
            "id,dur,proto,service,state,spkts,dpkts,sbytes,dbytes,attack_cat,label\n"
        )
        rows = [
            "1,0.12,tcp,-,FIN,6,4,258,172,Normal,0\n",
            "2,0.65,tcp,-,FIN,14,38,734,42014,Normal,0\n",
            "3,1.62,tcp,-,FIN,8,16,364,13186,Exploits,1\n",
            "4,0.02,udp,dns,CON,2,2,146,178,Reconnaissance,1\n",
            "5,0.09,tcp,http,FIN,10,8,600,900,Normal,0\n",
            "6,0.44,tcp,-,FIN,12,20,700,1200,Exploits,1\n",
            "7,0.31,udp,dns,CON,3,3,150,190,Reconnaissance,1\n",
            "8,0.77,tcp,http,FIN,9,7,620,880,Normal,0\n",
        ]
        data_dir = tmp_path / "ZT-Fedtrust Datasets"
        data_dir.mkdir()
        with zipfile.ZipFile(data_dir / "UNSW-NB15 dataset.zip", "w") as zf:
            zf.writestr(
                "CSV Files/Training and Testing Sets/UNSW_NB15_training-set.csv",
                header + "".join(rows[:6]),
            )
            zf.writestr(
                "CSV Files/Training and Testing Sets/UNSW_NB15_testing-set.csv",
                header + "".join(rows[6:]),
            )
        return data_dir

    def test_loads_and_drops_id_and_binary_label(self, tmp_path):
        data_dir = self._write_fixture(tmp_path)
        result = load_unsw_nb15(
            str(data_dir), num_clients=2, partition_type="iid",
            dirichlet_alpha=0.5, max_rows=None, seed=1,
        )
        assert result.input_dim == 8  # dur, proto, service, state, spkts, dpkts, sbytes, dbytes
        assert "Normal" in result.class_names

    def test_attack_pool_excludes_normal_rows(self, tmp_path):
        data_dir = self._write_fixture(tmp_path)
        result = load_unsw_nb15(
            str(data_dir), num_clients=2, partition_type="iid",
            dirichlet_alpha=0.5, max_rows=None, seed=1,
        )
        assert result.attack_pool is not None
        assert len(result.attack_pool) > 0

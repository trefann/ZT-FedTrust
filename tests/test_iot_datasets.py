"""Tests for the shared IoT experiment builders and, in later tasks, per-dataset loaders."""

import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from data.iot_datasets import _build_natural_client_experiment, _build_partitioned_experiment, load_ciciot2023, load_ton_iot_devices, load_ton_iot_network, load_unsw_nb15


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


class TestLoadTonIotNetwork:
    def _write_fixture(self, tmp_path):
        header = (
            "src_ip,src_port,dst_ip,dst_port,proto,service,duration,src_bytes,"
            "dst_bytes,conn_state,dns_query,ssl_subject,ssl_issuer,http_uri,"
            "http_user_agent,label,type\n"
        )
        rows = [
            "192.168.1.1,80,192.168.1.2,443,tcp,-,0.1,100,200,SF,-,-,-,-,-,0,normal\n",
            "192.168.1.3,53,192.168.1.4,53,udp,dns,0.05,60,80,SF,example.com,-,-,-,-,0,normal\n",
            "192.168.1.5,4444,192.168.1.6,80,tcp,-,2.3,900,50,REJ,-,-,-,-,-,1,backdoor\n",
            "192.168.1.7,80,192.168.1.8,80,tcp,http,0.9,500,600,SF,-,-,-,/x,curl,1,ddos\n",
            "192.168.1.9,80,192.168.1.10,443,tcp,-,0.2,110,210,SF,-,-,-,-,-,0,normal\n",
            "192.168.1.11,4444,192.168.1.12,80,tcp,-,3.1,950,40,REJ,-,-,-,-,-,1,backdoor\n",
        ]
        data_dir = tmp_path / "ZT-Fedtrust Datasets"
        data_dir.mkdir()
        with zipfile.ZipFile(data_dir / "Train_Test_datasets.zip", "w") as zf:
            zf.writestr(
                "Train_Test_datasets/Train_Test_Network_dataset/train_test_network.csv",
                header + "".join(rows),
            )
        return data_dir

    def test_drops_identity_columns(self, tmp_path):
        data_dir = self._write_fixture(tmp_path)
        result = load_ton_iot_network(
            str(data_dir), num_clients=2, partition_type="iid",
            dirichlet_alpha=0.5, max_rows=None, seed=1,
        )
        # src_port,dst_port,proto,service,duration,src_bytes,dst_bytes,conn_state = 8 features
        assert result.input_dim == 8
        assert "normal" in result.class_names

    def test_attack_pool_present(self, tmp_path):
        data_dir = self._write_fixture(tmp_path)
        result = load_ton_iot_network(
            str(data_dir), num_clients=2, partition_type="iid",
            dirichlet_alpha=0.5, max_rows=None, seed=1,
        )
        assert result.attack_pool is not None


class TestLoadTonIotDevices:
    # The real TON_IoT zip has 7 device CSVs; this fixture only writes 2 of
    # them (Fridge, Garage_Door). `_load_ton_iot_device_dataframes` is
    # deliberately strict (unconditional `zf.open`, matching the sibling
    # loaders) so it fails loudly if any of `_TON_IOT_DEVICE_ENTRIES` is
    # missing from the zip. To exercise the reindex/union-schema logic
    # against this smaller fixture without weakening that production
    # strictness, every test in this class monkeypatches
    # `_TON_IOT_DEVICE_ENTRIES` down to just the 2 entries the fixture
    # actually writes.
    @pytest.fixture(autouse=True)
    def _two_device_entries(self, monkeypatch):
        monkeypatch.setattr(
            "data.iot_datasets._TON_IOT_DEVICE_ENTRIES",
            {
                "fridge": "Train_Test_datasets/Train_Test_IoT_dataset/Train_Test_IoT_Fridge.csv",
                "garage_door": "Train_Test_datasets/Train_Test_IoT_dataset/Train_Test_IoT_Garage_Door.csv",
            },
        )

    def _write_fixture(self, tmp_path):
        import zipfile

        data_dir = tmp_path / "ZT-Fedtrust Datasets"
        data_dir.mkdir()
        with zipfile.ZipFile(data_dir / "Train_Test_datasets.zip", "w") as zf:
            zf.writestr(
                "Train_Test_datasets/Train_Test_IoT_dataset/Train_Test_IoT_Fridge.csv",
                "date,time,fridge_temperature,temp_condition,label,type\n"
                "25-Apr-19,19:19:40,9,high,1,ddos\n"
                "25-Apr-19,19:19:45,4.6,low,0,normal\n"
                "25-Apr-19,19:19:50,5.1,low,0,normal\n",
            )
            zf.writestr(
                "Train_Test_datasets/Train_Test_IoT_dataset/Train_Test_IoT_Garage_Door.csv",
                "date,time,door_state,sphone_signal,label,type\n"
                "25-Apr-19,19:19:40,open,1,1,backdoor\n"
                "25-Apr-19,19:19:45,closed,0,0,normal\n"
                "25-Apr-19,19:19:50,closed,0,0,normal\n",
            )
        return data_dir

    def test_produces_one_client_per_device(self, tmp_path):
        data_dir = self._write_fixture(tmp_path)
        result = load_ton_iot_devices(str(data_dir), max_rows_per_device=None, seed=1)
        assert len(result.train_client_datasets) == 2

    def test_shared_feature_space_across_devices(self, tmp_path):
        data_dir = self._write_fixture(tmp_path)
        result = load_ton_iot_devices(str(data_dir), max_rows_per_device=None, seed=1)
        # union of fridge_temperature, temp_condition, door_state, sphone_signal
        assert result.input_dim == 4
        for dataset in result.train_client_datasets.values():
            features, _ = dataset[0]
            assert features.shape == (4,)

    def test_class_vocabulary_spans_both_devices(self, tmp_path):
        data_dir = self._write_fixture(tmp_path)
        result = load_ton_iot_devices(str(data_dir), max_rows_per_device=None, seed=1)
        assert result.class_names == ["backdoor", "ddos", "normal"]

    def test_missing_device_entry_raises_keyerror(self, tmp_path, monkeypatch):
        """A device zip entry missing from the archive must fail loudly
        (KeyError from zipfile), not be silently skipped."""
        data_dir = self._write_fixture(tmp_path)
        monkeypatch.setattr(
            "data.iot_datasets._TON_IOT_DEVICE_ENTRIES",
            {
                "fridge": "Train_Test_datasets/Train_Test_IoT_dataset/Train_Test_IoT_Fridge.csv",
                "garage_door": "Train_Test_datasets/Train_Test_IoT_dataset/Train_Test_IoT_Garage_Door.csv",
                "missing_device": "Train_Test_datasets/Train_Test_IoT_dataset/Train_Test_IoT_Does_Not_Exist.csv",
            },
        )
        with pytest.raises(KeyError):
            load_ton_iot_devices(str(data_dir), max_rows_per_device=None, seed=1)

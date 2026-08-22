"""Tests for dataset-agnostic tabular preprocessing helpers."""

import numpy as np
import pandas as pd
import pytest
import torch

from data.iot_preprocessing import (
    compute_class_weights,
    encode_labels,
    fit_feature_encoder,
    fit_label_classes,
    split_train_test,
    subsample_dataframe,
)


def _sample_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "proto": ["tcp", "udp", "tcp", "icmp", "udp", "tcp"],
            "duration": [1.0, 2.5, 0.1, 9.9, 3.3, 0.5],
            "bytes": [100, 200, 50, 900, 300, 60],
        }
    )


def test_fit_feature_encoder_splits_categorical_and_numeric():
    encoder = fit_feature_encoder(_sample_df())
    assert set(encoder.categorical_columns) == {"proto"}
    assert set(encoder.numeric_columns) == {"duration", "bytes"}
    assert encoder.input_dim == 3


def test_transform_produces_expected_shape_and_dtype():
    df = _sample_df()
    encoder = fit_feature_encoder(df)
    matrix = encoder.transform(df)
    assert matrix.shape == (6, 3)
    assert matrix.dtype == np.float32


def test_transform_maps_unseen_categorical_value_to_zero():
    train_df = _sample_df()
    encoder = fit_feature_encoder(train_df)
    proto_col_index = encoder.feature_names.index("proto")

    unseen_df = pd.DataFrame({"proto": ["quic"], "duration": [1.0], "bytes": [10]})
    matrix = encoder.transform(unseen_df)
    assert matrix[0, proto_col_index] == 0.0


def test_transform_numeric_column_is_standardized():
    df = pd.DataFrame({"value": [1.0, 2.0, 3.0, 4.0, 5.0]})
    encoder = fit_feature_encoder(df)
    matrix = encoder.transform(df)
    assert matrix.mean() == pytest.approx(0.0, abs=1e-5)


def test_fit_label_classes_returns_sorted_unique_values():
    labels = pd.Series(["ddos", "normal", "ddos", "backdoor"])
    assert fit_label_classes(labels) == ["backdoor", "ddos", "normal"]


def test_encode_labels_maps_to_class_indices():
    labels = pd.Series(["ddos", "normal", "backdoor"])
    class_list = ["backdoor", "ddos", "normal"]
    encoded = encode_labels(labels, class_list)
    assert encoded.tolist() == [1, 2, 0]
    assert encoded.dtype == np.int64


def test_encode_labels_raises_on_unseen_label():
    labels = pd.Series(["ddos", "totally-new-attack"])
    with pytest.raises(ValueError):
        encode_labels(labels, ["ddos", "normal"])


def test_subsample_dataframe_returns_unchanged_when_under_cap():
    df = _sample_df()
    result = subsample_dataframe(df, max_rows=100, seed=1)
    assert len(result) == len(df)


def test_subsample_dataframe_caps_row_count():
    df = pd.DataFrame({"x": range(100)})
    result = subsample_dataframe(df, max_rows=10, seed=1)
    assert len(result) == 10


def test_subsample_dataframe_none_means_unlimited():
    df = _sample_df()
    result = subsample_dataframe(df, max_rows=None, seed=1)
    assert len(result) == len(df)


def test_split_train_test_covers_all_rows_without_overlap():
    df = pd.DataFrame({"x": range(50)})
    train_df, test_df = split_train_test(df, test_fraction=0.2, seed=1)
    assert len(train_df) + len(test_df) == 50
    assert set(train_df["x"]).isdisjoint(set(test_df["x"]))


def test_split_train_test_respects_fraction():
    df = pd.DataFrame({"x": range(100)})
    train_df, test_df = split_train_test(df, test_fraction=0.3, seed=1)
    assert len(test_df) == 30
    assert len(train_df) == 70


def test_compute_class_weights_gives_rarer_classes_higher_weight():
    # class 0 appears 8 times, class 1 appears 2 times, class 2 never appears
    labels = [0] * 8 + [1] * 2
    weights = compute_class_weights(labels, num_classes=3)
    assert weights.shape == (3,)
    assert weights[1] > weights[0]


def test_compute_class_weights_zero_count_class_gets_zero_weight():
    labels = [0, 0, 1, 1]
    weights = compute_class_weights(labels, num_classes=3)
    assert weights[2] == 0.0


def test_compute_class_weights_all_one_class_does_not_crash():
    labels = [4, 4, 4, 4]
    weights = compute_class_weights(labels, num_classes=5)
    assert weights.shape == (5,)
    assert weights[4] > 0.0
    assert torch.equal(weights[[0, 1, 2, 3]], torch.zeros(4))


def test_compute_class_weights_is_a_float_tensor():
    weights = compute_class_weights([0, 1, 0, 1], num_classes=2)
    assert weights.dtype == torch.float32

"""Tests for TabularMLP construction and forward-pass behavior."""

import pytest
import torch

from models.cnn import available_models, build_model


@pytest.mark.parametrize("input_dim,num_classes", [(39, 6), (115, 11), (6, 8)])
def test_forward_pass_output_shape(input_dim, num_classes):
    model = build_model("mlp", in_channels=input_dim, num_classes=num_classes)
    batch = torch.randn(4, input_dim)
    output = model(batch)
    assert output.shape == (4, num_classes)


def test_mlp_registered_in_available_models():
    assert "mlp" in available_models()


def test_mlp_handles_batch_size_one_in_eval_mode():
    model = build_model("mlp", in_channels=10, num_classes=3)
    model.eval()
    output = model(torch.randn(1, 10))
    assert output.shape == (1, 3)


def test_default_hidden_sizes_match_original_architecture():
    model = build_model("mlp", in_channels=20, num_classes=5)
    assert [layer[0].out_features for layer in model.layers] == [128, 64]
    batch = torch.randn(4, 20)
    output = model(batch)
    assert output.shape == (4, 5)


def test_custom_hidden_sizes_change_layer_widths():
    model = build_model("mlp", in_channels=20, num_classes=5, hidden_sizes=(256, 128, 32))
    assert [layer[0].out_features for layer in model.layers] == [256, 128, 32]
    batch = torch.randn(4, 20)
    output = model(batch)
    assert output.shape == (4, 5)


def test_single_hidden_layer_is_valid():
    model = build_model("mlp", in_channels=10, num_classes=3, hidden_sizes=(64,))
    batch = torch.randn(4, 10)
    output = model(batch)
    assert output.shape == (4, 3)


def test_empty_hidden_sizes_raises():
    with pytest.raises(ValueError):
        build_model("mlp", in_channels=10, num_classes=3, hidden_sizes=())

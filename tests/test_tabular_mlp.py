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

"""Tests for CNN model construction and forward-pass behavior."""

import pytest
import torch

from models.cnn import available_models, build_model


@pytest.mark.parametrize("in_channels,image_size", [(1, 28), (3, 32)])
def test_forward_pass_output_shape(in_channels, image_size):
    model = build_model("cnn", in_channels=in_channels, num_classes=10)
    batch = torch.randn(4, in_channels, image_size, image_size)
    output = model(batch)
    assert output.shape == (4, 10)


def test_unknown_model_name_raises():
    with pytest.raises(ValueError):
        build_model("does-not-exist", in_channels=1, num_classes=10)


def test_available_models_includes_cnn():
    assert "cnn" in available_models()

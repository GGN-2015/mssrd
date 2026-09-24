from __future__ import annotations

import json
from importlib.resources import files

import numpy as np
import pytest
import torch

from mssrd.core import ScaleEstimate
from mssrd.paper.reproduce import _metrics, _paper_grayscale, _paper_scales
from mssrd.paper.unet import (
    build_true_bottleneck_model,
    build_unet_model,
    true_bottleneck_candidate,
    unet_candidates,
)


def test_paper_scales_match_protocol() -> None:
    assert _paper_scales(28, 28) == (2, 4, 7)
    assert _paper_scales(32, 32) == (2, 4, 8)
    assert _paper_scales(8, 8) == (2, 4, 8)


def test_paper_grayscale_and_optdigits_scale() -> None:
    rgb = np.zeros((2, 4, 4, 3), dtype=np.uint8)
    rgb[..., 0] = 255
    grayscale = _paper_grayscale(rgb, 255.0)
    np.testing.assert_allclose(grayscale, 0.299, rtol=1e-6)
    digits = np.full((2, 8, 8), 16, dtype=np.uint8)
    np.testing.assert_allclose(_paper_grayscale(digits, 16.0), 1.0)


def test_reference_manifest_contains_twelve_datasets() -> None:
    reference = json.loads(
        files("mssrd.paper").joinpath("reference_results.json").read_text(encoding="utf-8")
    )
    assert len(reference["datasets"]) == 12
    assert reference["datasets"]["cifar10"]["prediction"] == {
        "q": 8,
        "channels": 13,
        "latent_dimension": 208,
    }


def test_single_dataset_metrics_omit_undefined_correlation() -> None:
    metrics = _metrics([{"predicted_latent_dimension": 28, "empirical_latent_dimension": 30}], [])
    assert metrics["dataset_count"] == 1
    assert metrics["log_dimension_pearson"] is None


def _scale_estimate(scale: int, channels: int, image_size: int) -> ScaleEstimate:
    dimension = scale * scale
    basis = np.eye(dimension, dtype=np.float64)
    grid = image_size // scale
    return ScaleEstimate(
        scale=scale,
        grid_height=grid,
        grid_width=grid,
        patch_dimension=dimension,
        channels=channels,
        latent_scalars=grid * grid * channels,
        retained_variance=0.95,
        linear_nmse=0.05,
        active_rd_modes=channels,
        rd_bits_per_patch=1.0,
        entropy_rank=float(channels),
        participation_ratio=float(channels),
        eigenvalues=np.ones(dimension),
        basis=basis,
    )


def test_unet_candidates_cover_skip_ablation_and_bypass() -> None:
    prediction = _scale_estimate(scale=7, channels=22, image_size=28)
    candidates = unet_candidates(prediction, (28, 28))
    assert [(item.role, item.channels) for item in candidates] == [
        ("full_skip_zero_bottleneck", 0),
        ("full_skip_one_channel", 1),
        ("full_skip_prediction", 22),
    ]
    assert all(item.skip_scalars > prediction.latent_scalars for item in candidates)


@pytest.mark.parametrize(
    "role",
    [
        "full_skip_zero_bottleneck",
        "full_skip_one_channel",
        "full_skip_prediction",
    ],
)
def test_unet_models_preserve_image_shape(role: str) -> None:
    prediction = _scale_estimate(scale=8, channels=4, image_size=32)
    candidate = next(item for item in unet_candidates(prediction, (32, 32)) if item.role == role)
    model = build_unet_model(candidate, image_shape=(32, 32))
    output = model(torch.randn(2, 1, 32, 32))
    assert output.shape == (2, 1, 32, 32)


def test_true_bottleneck_unet_has_no_input_dependent_skip_path() -> None:
    candidate = true_bottleneck_candidate(3, scale=7, grid_height=4, grid_width=4)
    model = build_true_bottleneck_model(candidate, image_shape=(28, 28), basis=np.zeros((49, 3)))
    with torch.no_grad():
        first = model(torch.randn(2, 1, 28, 28))
        second = model(torch.randn(2, 1, 28, 28))
    torch.testing.assert_close(first, second)


def test_true_bottleneck_unet_starts_at_spectral_projection() -> None:
    candidate = true_bottleneck_candidate(3, scale=7, grid_height=4, grid_width=4)
    basis = np.eye(49, dtype=np.float32)[:, :3]
    model = build_true_bottleneck_model(candidate, image_shape=(28, 28), basis=basis)
    values = torch.randn(2, 1, 28, 28)
    with torch.no_grad():
        output = model(values)
        patches = torch.nn.functional.unfold(values, kernel_size=7, stride=7)
        latent = torch.matmul(patches.transpose(1, 2), model.basis)
        projected = torch.matmul(latent, model.basis.transpose(0, 1)).transpose(1, 2)
        expected = torch.nn.functional.fold(
            projected, output_size=(28, 28), kernel_size=7, stride=7
        )
    torch.testing.assert_close(output, expected)

from __future__ import annotations

import numpy as np
import pytest

from mssrd import MSSRD, audit_unet_capacity, predict_bottleneck


def low_rank_images(seed: int = 4) -> np.ndarray:
    rng = np.random.default_rng(seed)
    raw_basis = rng.normal(size=(16, 2))
    basis, _ = np.linalg.qr(raw_basis)
    coefficients = rng.normal(size=(240, 2, 2, 2))
    patches = coefficients @ basis.T
    return patches.reshape(240, 2, 2, 4, 4).transpose(0, 1, 3, 2, 4).reshape(240, 8, 8)


def test_known_rank_prediction_and_inverse() -> None:
    images = low_rank_images()
    estimator = MSSRD(retained_variance=0.95, scales=[2, 4], seed=11)
    result = estimator.fit(images)
    assert result.prediction.scale == 4
    assert result.prediction.channels == 2
    assert result.prediction.tensor_shape == (2, 2, 2)
    latent = estimator.transform(images)
    reconstructed = estimator.inverse_transform(latent)
    assert latent.shape == (240, 2, 2, 2)
    assert np.mean((images - reconstructed) ** 2) < 1e-20


def test_pareto_frontier_balances_latent_scalars_and_shared_parameters() -> None:
    result = predict_bottleneck(low_rank_images(), retained_variance=0.95, scales=[2, 4])
    assert [estimate.scale for estimate in result.pareto_frontier] == [2, 4]
    assert result.scales[0].linear_parameter_count == 2 * 4 * result.scales[0].channels
    payload = result.to_dict()
    assert payload["pareto_frontier_scales"] == [2, 4]


def test_nmse_and_retained_variance_parameterizations_match() -> None:
    images = low_rank_images()
    default = predict_bottleneck(images, scales=[2, 4], seed=11)
    direct = predict_bottleneck(images, target_nmse=0.05, scales=[2, 4], seed=11)
    legacy = predict_bottleneck(images, retained_variance=0.95, scales=[2, 4], seed=11)
    assert default.retained_variance == pytest.approx(0.99)
    assert direct.prediction.tensor_shape == legacy.prediction.tensor_shape
    with pytest.raises(ValueError, match="not both"):
        predict_bottleneck(
            images,
            target_nmse=0.05,
            retained_variance=0.95,
            scales=[2, 4],
        )


def test_constant_features_are_counted_and_ignored() -> None:
    rng = np.random.default_rng(9)
    images = rng.normal(size=(80, 8, 8))
    images[:, 0, :] = 3.0
    result = predict_bottleneck(images, scales=[2, 4], seed=9)
    assert result.constant_input_features == 8
    assert result.nonconstant_input_features == 56


def test_all_constant_dataset_is_rejected() -> None:
    with pytest.raises(ValueError, match="all input features are constant"):
        predict_bottleneck(np.ones((10, 8, 8)), scales=[2, 4])


def test_nchw_and_nhwc_color_inputs_match_in_grayscale() -> None:
    rng = np.random.default_rng(3)
    nhwc = rng.integers(0, 256, size=(40, 8, 8, 3), dtype=np.uint8)
    nchw = np.moveaxis(nhwc, -1, 1)
    first = predict_bottleneck(nhwc, scales=[2, 4], seed=3)
    second = predict_bottleneck(nchw, scales=[2, 4], channel_axis=1, seed=3)
    assert first.prediction.tensor_shape == second.prediction.tensor_shape
    np.testing.assert_allclose(
        first.prediction.eigenvalues, second.prediction.eigenvalues, rtol=1e-12, atol=1e-12
    )


def test_joint_channel_mode_uses_color_dimensions() -> None:
    rng = np.random.default_rng(8)
    images = rng.normal(size=(50, 8, 8, 3))
    result = predict_bottleneck(images, scales=[4], color_mode="channels")
    assert result.prediction.patch_dimension == 4 * 4 * 3


def test_nondivisor_scale_is_zero_padded_and_cropped() -> None:
    rng = np.random.default_rng(12)
    images = rng.normal(size=(80, 7, 10))
    estimator = MSSRD(retained_variance=1.0, scales=[4])
    result = estimator.fit(images)
    estimate = result.prediction
    assert estimate.tensor_shape[:2] == (2, 3)
    assert estimate.padding_bottom == 1
    assert estimate.padding_right == 2
    reconstructed = estimator.inverse_transform(estimator.transform(images))
    assert reconstructed.shape == images.shape
    np.testing.assert_allclose(reconstructed, images, rtol=1e-10, atol=1e-10)


def test_invalid_scale_has_actionable_error() -> None:
    images = np.arange(5 * 7 * 8, dtype=np.float64).reshape(5, 7, 8)
    with pytest.raises(ValueError, match="no larger"):
        predict_bottleneck(images, scales=[9])


def test_unet_capacity_audit_counts_bypass_paths() -> None:
    result = predict_bottleneck(low_rank_images(), scales=[4])
    audit = audit_unet_capacity(result, [(8, 8, 16), (4, 4, 32)])
    assert audit.predicted_bottleneck_shape == (2, 2, 2)
    assert audit.skip_path_scalars == 1536
    assert audit.total_transmitted_scalars == 1544
    assert audit.skip_to_bottleneck_ratio == 192.0
    assert audit.terminal_is_global_information_bottleneck is False

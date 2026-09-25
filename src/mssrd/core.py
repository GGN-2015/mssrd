"""Core MS-SRD estimator.

The implementation intentionally depends only on NumPy. It estimates local
patch covariance spectra and predicts a spatial bottleneck tensor before any
neural network is trained.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class ScaleEstimate:
    """Spectral bottleneck estimate for one square patch scale."""

    scale: int
    grid_height: int
    grid_width: int
    patch_dimension: int
    channels: int
    latent_scalars: int
    retained_variance: float
    linear_nmse: float
    active_rd_modes: int
    rd_bits_per_patch: float
    entropy_rank: float
    participation_ratio: float
    eigenvalues: FloatArray = field(repr=False, compare=False)
    basis: FloatArray = field(repr=False, compare=False)
    padding_bottom: int = 0
    padding_right: int = 0

    @property
    def tensor_shape(self) -> tuple[int, int, int]:
        return self.grid_height, self.grid_width, self.channels

    @property
    def linear_parameter_count(self) -> int:
        """Bias-free encoder and decoder parameters for the shared block map."""

        return 2 * self.patch_dimension * self.channels

    def to_dict(self, *, include_eigenvalues: bool = False) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "scale": self.scale,
            "grid_height": self.grid_height,
            "grid_width": self.grid_width,
            "channels": self.channels,
            "tensor_shape": list(self.tensor_shape),
            "patch_dimension": self.patch_dimension,
            "padding_bottom": self.padding_bottom,
            "padding_right": self.padding_right,
            "latent_scalars": self.latent_scalars,
            "linear_parameter_count": self.linear_parameter_count,
            "retained_variance": self.retained_variance,
            "linear_nmse": self.linear_nmse,
            "active_rd_modes": self.active_rd_modes,
            "rd_bits_per_patch": self.rd_bits_per_patch,
            "entropy_rank": self.entropy_rank,
            "participation_ratio": self.participation_ratio,
        }
        if include_eigenvalues:
            payload["eigenvalues"] = self.eigenvalues.tolist()
        return payload


@dataclass(frozen=True)
class MSSRDResult:
    """Complete result returned by :class:`MSSRD`."""

    sample_count: int
    input_shape: tuple[int, int, int]
    color_mode: str
    retained_variance: float
    seed: int
    constant_input_features: int
    nonconstant_input_features: int
    scales: tuple[ScaleEstimate, ...]
    prediction_index: int
    global_pca_dimension: int | None = None
    global_eigenvalues: FloatArray | None = field(default=None, repr=False, compare=False)

    @property
    def prediction(self) -> ScaleEstimate:
        return self.scales[self.prediction_index]

    @property
    def pareto_frontier(self) -> tuple[ScaleEstimate, ...]:
        """Scales not dominated in latent scalars and shared linear parameters."""

        frontier = []
        for candidate in self.scales:
            dominated = any(
                other.latent_scalars <= candidate.latent_scalars
                and other.linear_parameter_count <= candidate.linear_parameter_count
                and (
                    other.latent_scalars < candidate.latent_scalars
                    or other.linear_parameter_count < candidate.linear_parameter_count
                )
                for other in self.scales
            )
            if not dominated:
                frontier.append(candidate)
        return tuple(frontier)

    def to_dict(self, *, include_eigenvalues: bool = False) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "method": "MS-SRD",
            "sample_count": self.sample_count,
            "input_shape": list(self.input_shape),
            "color_mode": self.color_mode,
            "retained_variance": self.retained_variance,
            "target_nmse": 1.0 - self.retained_variance,
            "seed": self.seed,
            "constant_input_features": self.constant_input_features,
            "nonconstant_input_features": self.nonconstant_input_features,
            "prediction": self.prediction.to_dict(include_eigenvalues=include_eigenvalues),
            "selection_rule": "minimum latent scalars over the supplied candidate scales",
            "pareto_frontier_scales": [item.scale for item in self.pareto_frontier],
            "scales": [
                scale.to_dict(include_eigenvalues=include_eigenvalues) for scale in self.scales
            ],
            "preprocessing": {
                "centering": "subtract the training per-pixel mean",
                "variance_normalization": False,
                "constant_features": "zero-energy directions are ignored",
            },
        }
        if self.global_pca_dimension is not None:
            payload["global_pca_dimension"] = self.global_pca_dimension
        if include_eigenvalues and self.global_eigenvalues is not None:
            payload["global_eigenvalues"] = self.global_eigenvalues.tolist()
        return payload


def _as_float(images: NDArray[Any]) -> FloatArray:
    if np.issubdtype(images.dtype, np.integer):
        info = np.iinfo(images.dtype)
        scale = float(max(abs(info.min), abs(info.max)))
        return images.astype(np.float64) / scale
    if np.issubdtype(images.dtype, np.bool_):
        return images.astype(np.float64)
    return images.astype(np.float64, copy=False)


def _infer_channel_axis(images: NDArray[Any], channel_axis: int | None) -> int | None:
    if images.ndim == 3:
        return None
    if channel_axis is not None:
        axis = channel_axis % images.ndim
        if axis == 0:
            raise ValueError("channel_axis cannot be the sample axis")
        return axis
    last_is_channel = images.shape[-1] in {1, 2, 3, 4}
    first_is_channel = images.shape[1] in {1, 2, 3, 4}
    if last_is_channel:
        return images.ndim - 1
    if first_is_channel:
        return 1
    raise ValueError(
        "Cannot infer the channel axis. Pass channel_axis=-1 for NHWC or channel_axis=1 for NCHW."
    )


def prepare_images(
    images: ArrayLike,
    *,
    color_mode: str = "grayscale",
    channel_axis: int | None = None,
) -> FloatArray:
    """Convert an image collection to finite ``(N, H, W, C)`` float64 data.

    Integer arrays are scaled by their dtype range. Floating-point arrays are
    left on their original scale because MS-SRD is invariant to a global scalar
    multiplier.
    """

    array = np.asarray(images)
    if array.ndim not in {3, 4}:
        raise ValueError("images must have shape (N,H,W), (N,H,W,C), or (N,C,H,W)")
    if len(array) < 2:
        raise ValueError("at least two images are required")
    axis = _infer_channel_axis(array, channel_axis)
    if axis is None:
        array = array[..., np.newaxis]
    elif axis != array.ndim - 1:
        array = np.moveaxis(array, axis, -1)
    array = _as_float(array)
    if not np.all(np.isfinite(array)):
        raise ValueError("images contain NaN or infinite values")
    if color_mode == "grayscale":
        channels = array.shape[-1]
        if channels == 1:
            pass
        elif channels >= 3:
            array = (0.299 * array[..., 0] + 0.587 * array[..., 1] + 0.114 * array[..., 2])[
                ..., np.newaxis
            ]
        else:
            raise ValueError("grayscale conversion requires 1, 3, or 4 input channels")
    elif color_mode != "channels":
        raise ValueError("color_mode must be 'grayscale' or 'channels'")
    return np.ascontiguousarray(array, dtype=np.float64)


def _pca_dimension(eigenvalues: FloatArray, retained_variance: float) -> int:
    total = float(eigenvalues.sum())
    if total <= np.finfo(np.float64).eps:
        return 0
    return int(np.searchsorted(np.cumsum(eigenvalues), retained_variance * total) + 1)


def _effective_ranks(eigenvalues: FloatArray) -> tuple[float, float]:
    total = float(eigenvalues.sum())
    if total <= np.finfo(np.float64).eps:
        return 0.0, 0.0
    probabilities = eigenvalues[eigenvalues > 0.0] / total
    entropy_rank = float(np.exp(-np.sum(probabilities * np.log(probabilities))))
    participation = float(total * total / np.sum(eigenvalues * eigenvalues))
    return entropy_rank, participation


def _reverse_waterfill(
    eigenvalues: FloatArray, distortion_fraction: float
) -> tuple[int, float, float]:
    values = eigenvalues[eigenvalues > np.finfo(np.float64).eps]
    if values.size == 0:
        return 0, 0.0, 0.0
    target = distortion_fraction * float(values.sum())
    low, high = 0.0, float(values.max())
    for _ in range(100):
        level = 0.5 * (low + high)
        if float(np.minimum(values, level).sum()) < target:
            low = level
        else:
            high = level
    active = values > high
    rate = 0.5 * float(np.log2(values[active] / high).sum()) if np.any(active) else 0.0
    return int(np.count_nonzero(active)), high, rate


def _covariance_spectrum(samples: FloatArray) -> tuple[FloatArray, FloatArray]:
    centered = samples - samples.mean(axis=0, keepdims=True)
    covariance = centered.T @ centered / len(centered)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    order = np.argsort(eigenvalues)[::-1]
    return np.maximum(eigenvalues[order], 0.0), eigenvectors[:, order]


def _centered_covariance_eigenvalues(centered_samples: FloatArray) -> FloatArray:
    """Return descending covariance eigenvalues without materializing eigenvectors."""

    covariance = centered_samples.T @ centered_samples / len(centered_samples)
    eigenvalues = np.linalg.eigvalsh(covariance)
    return np.maximum(eigenvalues[::-1], 0.0)


def _patch_spectrum(
    centered_images: FloatArray,
    scale: int,
    *,
    batch_size: int,
) -> tuple[FloatArray, FloatArray]:
    _, height, width, channels = centered_images.shape
    grid_height = (height + scale - 1) // scale
    grid_width = (width + scale - 1) // scale
    padded_height = grid_height * scale
    padded_width = grid_width * scale
    dimension = scale * scale * channels
    total = np.zeros(dimension, dtype=np.float64)
    cross = np.zeros((dimension, dimension), dtype=np.float64)
    count = 0
    for start in range(0, len(centered_images), batch_size):
        batch = centered_images[start : start + batch_size]
        batch = np.pad(
            batch,
            (
                (0, 0),
                (0, padded_height - height),
                (0, padded_width - width),
                (0, 0),
            ),
        )
        patches = (
            batch.reshape(len(batch), grid_height, scale, grid_width, scale, channels)
            .transpose(0, 1, 3, 2, 4, 5)
            .reshape(-1, dimension)
        )
        total += patches.sum(axis=0)
        cross += patches.T @ patches
        count += len(patches)
    covariance = (cross - np.outer(total, total) / count) / count
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    order = np.argsort(eigenvalues)[::-1]
    return np.maximum(eigenvalues[order], 0.0), eigenvectors[:, order]


def _resolve_scales(
    height: int,
    width: int,
    scales: Sequence[int] | None,
    max_patch_size: int,
) -> tuple[int, ...]:
    if scales is None:
        resolved = tuple(range(2, min(height, width, max_patch_size) + 1))
    else:
        resolved = tuple(sorted(set(int(scale) for scale in scales)))
    if not resolved:
        raise ValueError(
            "No valid patch scales. Pass scales explicitly or increase max_patch_size."
        )
    invalid = [scale for scale in resolved if scale < 1 or scale > min(height, width)]
    if invalid:
        raise ValueError(
            "scales must be positive and no larger than the shorter image dimension; "
            f"invalid values: {invalid}"
        )
    return resolved


class MSSRD:
    """Estimate an autoencoder bottleneck tensor without training a network.

    Parameters
    ----------
    target_nmse:
        Requested linear NMSE budget. The default is ``0.01``. This is the
        direct parameterization of the distortion constraint.
    retained_variance:
        Backward-compatible complement of ``target_nmse``. Specify at most one
        of these arguments.
    scales:
        Candidate square patch sizes. Scales need not divide the image dimensions;
        centered images are zero-padded on the bottom and right before patching.
        If omitted, every integer from 2 through ``max_patch_size`` is used.
    color_mode:
        ``"grayscale"`` applies BT.601 luma to RGB data. ``"channels"`` keeps
        all channels and lets each local eigenmode mix space and color.
    max_samples:
        Optional deterministic sample cap, applied before estimation.
    compute_global:
        Also compute global PCA. This can be expensive for large images.
    """

    def __init__(
        self,
        *,
        target_nmse: float | None = None,
        retained_variance: float | None = None,
        scales: Sequence[int] | None = None,
        max_patch_size: int = 8,
        color_mode: str = "grayscale",
        channel_axis: int | None = None,
        max_samples: int | None = None,
        seed: int = 0,
        batch_size: int = 512,
        compute_global: bool = False,
        variance_epsilon: float = 1e-12,
    ) -> None:
        if target_nmse is not None and retained_variance is not None:
            raise ValueError("specify target_nmse or retained_variance, not both")
        if target_nmse is None and retained_variance is None:
            target_nmse = 0.01
        if target_nmse is not None:
            if not 0.0 <= target_nmse < 1.0:
                raise ValueError("target_nmse must be in [0, 1)")
            retained_variance = 1.0 - target_nmse
        assert retained_variance is not None
        if not 0.0 < retained_variance <= 1.0:
            raise ValueError("retained_variance must be in (0, 1]")
        if max_patch_size < 1 or batch_size < 1:
            raise ValueError("max_patch_size and batch_size must be positive")
        self.retained_variance = retained_variance
        self.scales = tuple(scales) if scales is not None else None
        self.max_patch_size = max_patch_size
        self.color_mode = color_mode
        self.channel_axis = channel_axis
        self.max_samples = max_samples
        self.seed = seed
        self.batch_size = batch_size
        self.compute_global = compute_global
        self.variance_epsilon = variance_epsilon

    def fit(self, images: ArrayLike) -> MSSRDResult:
        prepared = prepare_images(
            images, color_mode=self.color_mode, channel_axis=self.channel_axis
        )
        if self.max_samples is not None and len(prepared) > self.max_samples:
            rng = np.random.default_rng(self.seed)
            indices = np.sort(rng.choice(len(prepared), size=self.max_samples, replace=False))
            prepared = prepared[indices]
        mean_image = prepared.mean(axis=0, dtype=np.float64)
        centered = np.ascontiguousarray(prepared - mean_image)
        point_variances = np.mean(centered * centered, axis=0)
        nonconstant = int(np.count_nonzero(point_variances > self.variance_epsilon))
        total_features = int(np.prod(point_variances.shape))
        if nonconstant == 0:
            raise ValueError(
                "all input features are constant; no nonzero bottleneck is identifiable"
            )
        _, height, width, channels = centered.shape
        resolved_scales = _resolve_scales(height, width, self.scales, self.max_patch_size)
        estimates: list[ScaleEstimate] = []
        for scale in resolved_scales:
            eigenvalues, eigenvectors = _patch_spectrum(centered, scale, batch_size=self.batch_size)
            selected = _pca_dimension(eigenvalues, self.retained_variance)
            total_energy = float(eigenvalues.sum())
            tail = float(eigenvalues[selected:].sum())
            linear_nmse = tail / total_energy if total_energy > 0.0 else 0.0
            active, _, rate = _reverse_waterfill(eigenvalues, 1.0 - self.retained_variance)
            entropy_rank, participation = _effective_ranks(eigenvalues)
            grid_height = (height + scale - 1) // scale
            grid_width = (width + scale - 1) // scale
            estimates.append(
                ScaleEstimate(
                    scale=scale,
                    grid_height=grid_height,
                    grid_width=grid_width,
                    patch_dimension=scale * scale * channels,
                    channels=selected,
                    latent_scalars=grid_height * grid_width * selected,
                    retained_variance=1.0 - linear_nmse,
                    linear_nmse=linear_nmse,
                    active_rd_modes=active,
                    rd_bits_per_patch=rate,
                    entropy_rank=entropy_rank,
                    participation_ratio=participation,
                    eigenvalues=eigenvalues,
                    basis=eigenvectors,
                    padding_bottom=grid_height * scale - height,
                    padding_right=grid_width * scale - width,
                )
            )
        prediction_index = min(
            range(len(estimates)),
            key=lambda index: (estimates[index].latent_scalars, estimates[index].scale),
        )
        global_dimension: int | None = None
        global_eigenvalues: FloatArray | None = None
        if self.compute_global:
            global_eigenvalues = _centered_covariance_eigenvalues(
                centered.reshape(len(centered), -1)
            )
            global_dimension = _pca_dimension(global_eigenvalues, self.retained_variance)
        self.mean_image_ = mean_image
        self.input_shape_ = (height, width, channels)
        self.result_ = MSSRDResult(
            sample_count=len(centered),
            input_shape=self.input_shape_,
            color_mode=self.color_mode,
            retained_variance=self.retained_variance,
            seed=self.seed,
            constant_input_features=total_features - nonconstant,
            nonconstant_input_features=nonconstant,
            scales=tuple(estimates),
            prediction_index=prediction_index,
            global_pca_dimension=global_dimension,
            global_eigenvalues=global_eigenvalues,
        )
        return self.result_

    def _require_fitted(self) -> MSSRDResult:
        if not hasattr(self, "result_"):
            raise RuntimeError("call fit before transform or inverse_transform")
        return self.result_

    def transform(self, images: ArrayLike, *, scale: int | None = None) -> FloatArray:
        """Project images into the selected or requested MS-SRD tensor."""

        result = self._require_fitted()
        prepared = prepare_images(
            images, color_mode=self.color_mode, channel_axis=self.channel_axis
        )
        if tuple(prepared.shape[1:]) != self.input_shape_:
            raise ValueError(
                f"expected images with prepared shape {self.input_shape_}, got {prepared.shape[1:]}"
            )
        estimate = result.prediction
        if scale is not None:
            matches = [item for item in result.scales if item.scale == scale]
            if not matches:
                raise ValueError(f"scale {scale} was not fitted")
            estimate = matches[0]
        q = estimate.scale
        n, height, width, channels = prepared.shape
        centered = prepared - self.mean_image_
        centered = np.pad(
            centered,
            (
                (0, 0),
                (0, estimate.padding_bottom),
                (0, estimate.padding_right),
                (0, 0),
            ),
        )
        patches = (
            centered.reshape(n, estimate.grid_height, q, estimate.grid_width, q, channels)
            .transpose(0, 1, 3, 2, 4, 5)
            .reshape(n, estimate.grid_height, estimate.grid_width, -1)
        )
        return np.ascontiguousarray(patches @ estimate.basis[:, : estimate.channels])

    def inverse_transform(self, latent: ArrayLike, *, scale: int | None = None) -> FloatArray:
        """Apply the linear inverse mapping associated with a fitted scale."""

        result = self._require_fitted()
        estimate = result.prediction
        if scale is not None:
            matches = [item for item in result.scales if item.scale == scale]
            if not matches:
                raise ValueError(f"scale {scale} was not fitted")
            estimate = matches[0]
        encoded = np.asarray(latent, dtype=np.float64)
        expected = (estimate.grid_height, estimate.grid_width, estimate.channels)
        if encoded.ndim != 4 or tuple(encoded.shape[1:]) != expected:
            raise ValueError(f"expected latent shape (N,{expected[0]},{expected[1]},{expected[2]})")
        q = estimate.scale
        channels = self.input_shape_[2]
        patches = encoded @ estimate.basis[:, : estimate.channels].T
        padded_height = estimate.grid_height * q
        padded_width = estimate.grid_width * q
        reconstructed = (
            patches.reshape(len(encoded), estimate.grid_height, estimate.grid_width, q, q, channels)
            .transpose(0, 1, 3, 2, 4, 5)
            .reshape(len(encoded), padded_height, padded_width, channels)
        )
        reconstructed = reconstructed[:, : self.input_shape_[0], : self.input_shape_[1], :]
        reconstructed += self.mean_image_
        if self.color_mode == "grayscale":
            return reconstructed[..., 0]
        return reconstructed

    def fit_transform(self, images: ArrayLike) -> FloatArray:
        self.fit(images)
        return self.transform(images)


def predict_bottleneck(
    images: ArrayLike,
    *,
    target_nmse: float | None = None,
    retained_variance: float | None = None,
    scales: Iterable[int] | None = None,
    max_patch_size: int = 8,
    color_mode: str = "grayscale",
    channel_axis: int | None = None,
    max_samples: int | None = None,
    seed: int = 0,
    compute_global: bool = False,
) -> MSSRDResult:
    """Functional convenience wrapper around :class:`MSSRD`."""

    estimator = MSSRD(
        target_nmse=target_nmse,
        retained_variance=retained_variance,
        scales=tuple(scales) if scales is not None else None,
        max_patch_size=max_patch_size,
        color_mode=color_mode,
        channel_axis=channel_axis,
        max_samples=max_samples,
        seed=seed,
        compute_global=compute_global,
    )
    return estimator.fit(images)

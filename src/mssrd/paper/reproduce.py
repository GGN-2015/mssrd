"""End-to-end reproduction of the MS-SRD paper experiment."""

from __future__ import annotations

import csv
import json
from collections.abc import Sequence
from importlib.resources import files
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.lines import Line2D
from matplotlib.ticker import LogLocator, NullFormatter

from mssrd.core import MSSRD, MSSRDResult, ScaleEstimate
from mssrd.paper.datasets import PAPER_DATASETS, PaperDataset, load_paper_dataset
from mssrd.paper.model import coarse_channels, extract_patches, train_candidate
from mssrd.paper.unet import (
    TRUE_BOTTLENECK_PROTOCOL_VERSION,
    UNET_PROTOCOL_VERSION,
    train_true_bottleneck_candidate,
    train_unet_candidate,
    true_bottleneck_candidate,
    unet_candidates,
)

plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["ps.fonttype"] = 42

UNET_NMSE_TARGETS = (0.01, 0.02, 0.05, 0.10)
PRIMARY_NMSE_TARGET = 0.01


def _paper_scales(height: int, width: int) -> tuple[int, ...]:
    return tuple(range(2, min(height, width, 8) + 1))


def _paper_grayscale(images: np.ndarray, divisor: float) -> np.ndarray:
    source = images.astype(np.float32)
    if source.ndim == 4 and source.shape[-1] == 1:
        source = source[..., 0]
    elif source.ndim == 4 and source.shape[-1] >= 3:
        source = 0.299 * source[..., 0] + 0.587 * source[..., 1] + 0.114 * source[..., 2]
    elif source.ndim != 3:
        raise ValueError(f"unsupported paper image shape {source.shape}")
    return np.ascontiguousarray(source / np.float32(divisor))


def _bootstrap_geometry(
    images: np.ndarray,
    *,
    scales: Sequence[int],
    retained_variance: float,
    repetitions: int,
    seed: int,
) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    subset_size = min(len(images), 5000)
    outcomes: list[tuple[int, int, int]] = []
    for _ in range(repetitions):
        indices = rng.choice(len(images), size=subset_size, replace=True)
        result = MSSRD(
            retained_variance=retained_variance,
            scales=scales,
            color_mode="grayscale",
            seed=seed,
            batch_size=512,
        ).fit(images[indices])
        prediction = result.prediction
        outcomes.append((prediction.latent_scalars, prediction.scale, prediction.channels))
    counts: dict[str, int] = {}
    for latent, scale, channels in outcomes:
        key = f"q={scale},c={channels},m={latent}"
        counts[key] = counts.get(key, 0) + 1
    values = np.asarray([item[0] for item in outcomes], dtype=np.float64)
    return {
        "repetitions": repetitions,
        "selection_counts": counts,
        "latent_median": float(np.median(values)),
        "latent_q025": float(np.quantile(values, 0.025)),
        "latent_q975": float(np.quantile(values, 0.975)),
    }


def _json_default(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"cannot serialize {type(value)!r}")


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, default=_json_default, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    return torch.device(name)


def _validate_scale(
    *,
    split: PaperDataset,
    centered_train: np.ndarray,
    centered_test: np.ndarray,
    estimate: ScaleEstimate,
    device: torch.device,
    dataset_dir: Path,
    seed: int,
    steps: int,
    batch_size: int,
    target_nmse: float,
    max_patch_samples: int,
) -> list[dict[str, Any]]:
    cache_path = dataset_dir / "training_cache.json"
    cache: dict[str, dict[str, Any]] = {}
    if cache_path.exists():
        cache = json.loads(cache_path.read_text(encoding="utf-8"))
    train_full = extract_patches(centered_train, estimate.scale)
    test_patches = extract_patches(centered_test, estimate.scale)
    rng = np.random.default_rng(seed + 1000 * estimate.scale)
    if len(train_full) > max_patch_samples:
        selected = rng.choice(len(train_full), size=max_patch_samples, replace=False)
        train_patches = np.ascontiguousarray(train_full[selected])
    else:
        train_patches = train_full

    def run(channels: int) -> dict[str, Any]:
        key = (
            f"{split.slug}|q={estimate.scale}|c={channels}|seed={seed}|"
            f"steps={steps}|patches={max_patch_samples}"
        )
        if key not in cache:
            metrics = train_candidate(
                train_patches,
                test_patches,
                estimate.basis[:, :channels],
                device=device,
                seed=seed + estimate.scale * 100 + channels,
                steps=steps,
                batch_size=min(batch_size, len(train_patches)),
            )
            cache[key] = {
                "slug": split.slug,
                "dataset": split.name,
                "q": estimate.scale,
                "grid_h": estimate.grid_height,
                "grid_w": estimate.grid_width,
                "channels": channels,
                "latent_dimension": estimate.grid_height * estimate.grid_width * channels,
                "predicted_channels": estimate.channels,
                "predicted_latent_dimension": estimate.latent_scalars,
                "target_nmse": target_nmse,
                "passed": bool(float(metrics["final_nmse"]) <= target_nmse),
                "seed": seed,
                **metrics,
            }
            _write_json(cache_path, cache)
            print(
                f"[{split.slug}] q={estimate.scale} c={channels}: "
                f"NMSE {metrics['initial_nmse']:.4f} -> {metrics['final_nmse']:.4f}",
                flush=True,
            )
        row = dict(cache[key])
        row["target_nmse"] = target_nmse
        row["passed"] = bool(float(row["final_nmse"]) <= target_nmse)
        return row

    records = [
        run(channels) for channels in coarse_channels(estimate.patch_dimension, estimate.channels)
    ]
    passing = [row for row in records if bool(row["passed"])]
    if passing:
        first_pass = min(int(row["channels"]) for row in passing)
        lower_tested = max(
            [int(row["channels"]) for row in records if int(row["channels"]) < first_pass],
            default=0,
        )
        existing = {int(row["channels"]) for row in records}
        for channels in range(lower_tested + 1, first_pass):
            if channels not in existing:
                records.append(run(channels))
    return sorted(records, key=lambda row: int(row["channels"]))


def _repeat_boundaries(
    *,
    split: PaperDataset,
    centered_train: np.ndarray,
    centered_test: np.ndarray,
    result: MSSRDResult,
    empirical: dict[str, Any],
    device: torch.device,
    seed: int,
    steps: int,
    batch_size: int,
    target_nmse: float,
    max_patch_samples: int,
) -> list[dict[str, Any]]:
    candidates = {
        (result.prediction.scale, result.prediction.channels, "predicted"),
        (int(empirical["q"]), int(empirical["channels"]), "empirical_boundary"),
    }
    if int(empirical["channels"]) > 1:
        candidates.add((int(empirical["q"]), int(empirical["channels"]) - 1, "below_boundary"))
    estimates = {item.scale: item for item in result.scales}
    prepared: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    for scale, _, _ in candidates:
        train_full = extract_patches(centered_train, scale)
        test_patches = extract_patches(centered_test, scale)
        rng = np.random.default_rng(seed + 1000 * scale)
        if len(train_full) > max_patch_samples:
            indices = rng.choice(len(train_full), size=max_patch_samples, replace=False)
            train_patches = np.ascontiguousarray(train_full[indices])
        else:
            train_patches = train_full
        prepared[scale] = (train_patches, test_patches)
    rows: list[dict[str, Any]] = []
    for scale, channels, role in sorted(candidates):
        train_patches, test_patches = prepared[scale]
        estimate = estimates[scale]
        for repeat_seed in (seed + 1, seed + 2):
            metrics = train_candidate(
                train_patches,
                test_patches,
                estimate.basis[:, :channels],
                device=device,
                seed=repeat_seed + scale * 100 + channels,
                steps=steps,
                batch_size=min(batch_size, len(train_patches)),
            )
            rows.append(
                {
                    "slug": split.slug,
                    "dataset": split.name,
                    "role": role,
                    "q": scale,
                    "channels": channels,
                    "latent_dimension": estimate.grid_height * estimate.grid_width * channels,
                    "seed": repeat_seed,
                    "target_nmse": target_nmse,
                    "passed": float(metrics["final_nmse"]) <= target_nmse,
                    **metrics,
                }
            )
            print(
                f"[{split.slug}] repeat {role} q={scale} c={channels} "
                f"seed={repeat_seed} NMSE={metrics['final_nmse']:.4f}",
                flush=True,
            )
    return rows


def _validate_unet(
    *,
    split: PaperDataset,
    centered_train: np.ndarray,
    centered_test: np.ndarray,
    prediction: ScaleEstimate,
    device: torch.device,
    dataset_dir: Path,
    seed: int,
    steps: int,
    batch_size: int,
    target_nmse: float = PRIMARY_NMSE_TARGET,
) -> list[dict[str, Any]]:
    cache_path = dataset_dir / "unet_cache.json"
    cache: dict[str, dict[str, Any]] = {}
    if cache_path.exists():
        cache = json.loads(cache_path.read_text(encoding="utf-8"))
    records: list[dict[str, Any]] = []
    candidates = unet_candidates(prediction, centered_train.shape[1:])
    for candidate in candidates:
        role_offset = sum(ord(character) for character in candidate.role)
        candidate_seed = seed + 30000 + role_offset + candidate.channels
        key = (
            f"v={UNET_PROTOCOL_VERSION}|{split.slug}|role={candidate.role}|"
            f"c={candidate.channels}|"
            f"seed={candidate_seed}|steps={steps}|batch={batch_size}|"
            f"train={len(centered_train)}|test={len(centered_test)}"
        )
        if key not in cache:
            metrics = train_unet_candidate(
                centered_train,
                centered_test,
                candidate,
                device=device,
                seed=candidate_seed,
                steps=steps,
                batch_size=min(batch_size, len(centered_train)),
            )
            cache[key] = {
                "slug": split.slug,
                "dataset": split.name,
                "family": "unet",
                **candidate.to_dict(),
                "predicted_latent_scalars": prediction.latent_scalars,
                "budget_ratio": (
                    candidate.latent_scalars / prediction.latent_scalars
                    if prediction.latent_scalars
                    else None
                ),
                "total_transmitted_scalars": (candidate.latent_scalars + candidate.skip_scalars),
                "target_nmse": target_nmse,
                "passed": bool(float(metrics["final_nmse"]) <= target_nmse),
                "seed": candidate_seed,
                **metrics,
            }
            _write_json(cache_path, cache)
            print(
                f"[{split.slug}] U-Net/{candidate.role} "
                f"M={candidate.latent_scalars}, skips={candidate.skip_scalars}: "
                f"NMSE {metrics['initial_nmse']:.4f} -> {metrics['final_nmse']:.4f}",
                flush=True,
            )
        row = dict(cache[key])
        row["target_nmse"] = target_nmse
        row["passed"] = bool(float(row["final_nmse"]) <= target_nmse)
        records.append(row)
    return records


def _unet_dataset_summary(
    rows: list[dict[str, Any]], predicted_latent_scalars: int
) -> dict[str, Any]:
    def select(role: str) -> dict[str, Any]:
        return next(row for row in rows if row["role"] == role)

    zero = select("full_skip_zero_bottleneck")
    one = select("full_skip_one_channel")
    predicted = select("full_skip_prediction")
    return {
        "predicted_latent_scalars": predicted_latent_scalars,
        "full_skips": {
            "skip_scalars": int(zero["skip_scalars"]),
            "skip_to_predicted_bottleneck_ratio": (
                int(zero["skip_scalars"]) / predicted_latent_scalars
            ),
            "zero_bottleneck_nmse": float(zero["final_nmse"]),
            "zero_bottleneck_passed": bool(zero["passed"]),
            "one_channel_nmse": float(one["final_nmse"]),
            "predicted_nmse": float(predicted["final_nmse"]),
        },
    }


def _unet_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    slugs = sorted({str(row["slug"]) for row in rows})
    if not slugs:
        return {}

    def selected(role: str) -> dict[str, dict[str, Any]]:
        return {str(row["slug"]): row for row in rows if row["role"] == role}

    zero = selected("full_skip_zero_bottleneck")
    one = selected("full_skip_one_channel")
    full_predicted = selected("full_skip_prediction")
    skip_ratios = [
        float(row["skip_scalars"]) / float(row["predicted_latent_scalars"]) for row in zero.values()
    ]
    return {
        "dataset_count": len(slugs),
        "full_skip_zero_bottleneck_pass_fraction": float(
            np.mean([bool(row["passed"]) for row in zero.values()])
        ),
        "full_skip_zero_bottleneck_median_nmse": float(
            np.median([float(row["final_nmse"]) for row in zero.values()])
        ),
        "full_skip_one_channel_pass_fraction": float(
            np.mean([bool(row["passed"]) for row in one.values()])
        ),
        "full_skip_prediction_pass_fraction": float(
            np.mean([bool(row["passed"]) for row in full_predicted.values()])
        ),
        "full_skip_zero_bottleneck_nmse_by_dataset": {
            slug: float(zero[slug]["final_nmse"]) for slug in slugs
        },
        "skip_to_predicted_bottleneck_ratio_median": float(np.median(skip_ratios)),
        "skip_to_predicted_bottleneck_ratio_min": float(np.min(skip_ratios)),
        "full_skip_zero_to_prediction_nmse_median_difference": float(
            np.median(
                [
                    float(zero[slug]["final_nmse"]) - float(full_predicted[slug]["final_nmse"])
                    for slug in slugs
                ]
            )
        ),
    }


def _spectral_predictions_for_targets(
    result: MSSRDResult, targets: Sequence[float]
) -> list[dict[str, Any]]:
    predictions: list[dict[str, Any]] = []
    for target in targets:
        if not 0.0 < target < 1.0:
            raise ValueError("U-Net NMSE targets must lie in (0, 1)")
        candidates: list[dict[str, Any]] = []
        for estimate in result.scales:
            eigenvalues = estimate.eigenvalues
            total = float(eigenvalues.sum())
            channels = int(np.searchsorted(np.cumsum(eigenvalues), (1.0 - target) * total) + 1)
            channels = min(channels, estimate.patch_dimension)
            latent_scalars = estimate.grid_height * estimate.grid_width * channels
            linear_nmse = float(eigenvalues[channels:].sum() / total)
            candidates.append(
                {
                    "target_nmse": float(target),
                    "scale": estimate.scale,
                    "grid_height": estimate.grid_height,
                    "grid_width": estimate.grid_width,
                    "patch_dimension": estimate.patch_dimension,
                    "predicted_channels": channels,
                    "predicted_latent_scalars": latent_scalars,
                    "linear_nmse": linear_nmse,
                }
            )
        predictions.append(
            min(
                candidates,
                key=lambda row: (
                    int(row["predicted_latent_scalars"]),
                    int(row["scale"]),
                ),
            )
        )
    return predictions


def _model_validation_split(images: np.ndarray, *, seed: int) -> tuple[np.ndarray, np.ndarray]:
    validation_size = min(1000, max(64, len(images) // 10), len(images) - 1)
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(images))
    validation_indices = np.sort(order[:validation_size])
    training_indices = np.sort(order[validation_size:])
    return (
        np.ascontiguousarray(images[training_indices]),
        np.ascontiguousarray(images[validation_indices]),
    )


def _validate_true_bottleneck_unet(
    *,
    split: PaperDataset,
    centered_train: np.ndarray,
    centered_test: np.ndarray,
    predictions: list[dict[str, Any]],
    scale_estimates: Sequence[ScaleEstimate],
    device: torch.device,
    dataset_dir: Path,
    seed: int,
    steps: int,
    batch_size: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    grids = {(int(row["grid_height"]), int(row["grid_width"])) for row in predictions}
    if len(grids) != 1:
        raise ValueError(
            f"true-bottleneck U-Net requires one selected grid across targets; got {grids}"
        )
    grid_height, grid_width = next(iter(grids))
    selected_scales = {int(row["scale"]) for row in predictions}
    if len(selected_scales) != 1:
        raise ValueError(
            "true-bottleneck U-Net requires one spectral scale across targets; "
            f"got {selected_scales}"
        )
    scale = next(iter(selected_scales))
    estimate = next(item for item in scale_estimates if item.scale == scale)
    maximum_channels = estimate.patch_dimension
    model_train, model_validation = _model_validation_split(centered_train, seed=seed + 41000)
    cache_path = dataset_dir / "true_bottleneck_cache.json"
    cache: dict[str, dict[str, Any]] = {}
    if cache_path.exists():
        cache = json.loads(cache_path.read_text(encoding="utf-8"))

    def run(channels: int) -> dict[str, Any]:
        candidate = true_bottleneck_candidate(
            channels,
            scale=scale,
            grid_height=grid_height,
            grid_width=grid_width,
        )
        slug_offset = sum((index + 1) * ord(value) for index, value in enumerate(split.slug))
        candidate_seed = seed + 50000 + slug_offset + channels
        key = (
            f"v={TRUE_BOTTLENECK_PROTOCOL_VERSION}|{split.slug}|c={channels}|"
            f"seed={candidate_seed}|steps={steps}|batch={batch_size}|"
            f"train={len(model_train)}|validation={len(model_validation)}|"
            f"test={len(centered_test)}"
        )
        if key not in cache:
            metrics = train_true_bottleneck_candidate(
                model_train,
                model_validation,
                centered_test,
                candidate,
                estimate.basis[:, :channels],
                device=device,
                seed=candidate_seed,
                steps=steps,
                batch_size=min(batch_size, len(model_train)),
            )
            cache[key] = {
                "slug": split.slug,
                "dataset": split.name,
                "family": "true_bottleneck_unet",
                **candidate.to_dict(),
                "seed": candidate_seed,
                "train_images": len(model_train),
                "validation_images": len(model_validation),
                "test_images": len(centered_test),
                **metrics,
            }
            _write_json(cache_path, cache)
            print(
                f"[{split.slug}] true-bottleneck U-Net c={channels}: "
                f"validation={metrics['validation_nmse']:.4f}, "
                f"test={metrics['test_nmse']:.4f}",
                flush=True,
            )
        return cache[key]

    records_by_channel: dict[int, dict[str, Any]] = {}

    def cached_run(channels: int) -> dict[str, Any]:
        if channels not in records_by_channel:
            records_by_channel[channels] = run(channels)
        return records_by_channel[channels]

    for channels in sorted({int(row["predicted_channels"]) for row in predictions}):
        cached_run(channels)

    boundary_channels: dict[float, int | None] = {}
    for prediction in sorted(predictions, key=lambda row: float(row["target_nmse"]), reverse=True):
        target = float(prediction["target_nmse"])
        high = int(prediction["predicted_channels"])
        if float(cached_run(high)["validation_nmse"]) > target:
            high = maximum_channels
        if float(cached_run(high)["validation_nmse"]) > target:
            boundary_channels[target] = None
            continue
        low = 0
        while high - low > 1:
            middle = (low + high) // 2
            if float(cached_run(middle)["validation_nmse"]) <= target:
                high = middle
            else:
                low = middle
        boundary_channels[target] = high

    test_oracle_channels: dict[float, int | None] = {}
    for prediction in sorted(predictions, key=lambda row: float(row["target_nmse"]), reverse=True):
        target = float(prediction["target_nmse"])
        high = int(prediction["predicted_channels"])
        if float(cached_run(high)["test_nmse"]) > target:
            high = maximum_channels
        if float(cached_run(high)["test_nmse"]) > target:
            test_oracle_channels[target] = None
            continue
        low = 0
        while high - low > 1:
            middle = (low + high) // 2
            if float(cached_run(middle)["test_nmse"]) <= target:
                high = middle
            else:
                low = middle
        test_oracle_channels[target] = high
    records = sorted(records_by_channel.values(), key=lambda row: int(row["channels"]))

    boundaries: list[dict[str, Any]] = []
    for prediction in predictions:
        target = float(prediction["target_nmse"])
        empirical_channels = boundary_channels[target]
        empirical = (
            records_by_channel[empirical_channels] if empirical_channels is not None else None
        )
        test_oracle_channel = test_oracle_channels[target]
        test_oracle = (
            records_by_channel[test_oracle_channel] if test_oracle_channel is not None else None
        )
        predicted_run = next(
            row for row in records if int(row["channels"]) == int(prediction["predicted_channels"])
        )
        boundaries.append(
            {
                "slug": split.slug,
                "dataset": split.name,
                **prediction,
                "empirical_channels": int(empirical["channels"]) if empirical else None,
                "empirical_latent_scalars": (
                    int(empirical["latent_scalars"]) if empirical else None
                ),
                "boundary_validation_nmse": (
                    float(empirical["validation_nmse"]) if empirical else None
                ),
                "boundary_test_nmse": float(empirical["test_nmse"]) if empirical else None,
                "boundary_test_passed": (
                    bool(float(empirical["test_nmse"]) <= target) if empirical else None
                ),
                "test_oracle_channels": (int(test_oracle["channels"]) if test_oracle else None),
                "test_oracle_latent_scalars": (
                    int(test_oracle["latent_scalars"]) if test_oracle else None
                ),
                "test_oracle_nmse": float(test_oracle["test_nmse"]) if test_oracle else None,
                "predicted_test_nmse": float(predicted_run["test_nmse"]),
                "predicted_test_passed": bool(float(predicted_run["test_nmse"]) <= target),
            }
        )
    return records, boundaries


def _true_bottleneck_metrics(boundaries: list[dict[str, Any]]) -> dict[str, Any]:
    complete = [row for row in boundaries if row["test_oracle_channels"] is not None]
    if not complete:
        return {}
    predicted = np.asarray([float(row["predicted_channels"]) for row in complete])
    empirical = np.asarray([float(row["test_oracle_channels"]) for row in complete])
    ratios = empirical / predicted
    dataset_count = len({str(row["slug"]) for row in complete})
    calibrated: np.ndarray | None = None
    if dataset_count > 1:
        calibrated_predictions: list[int] = []
        for row in complete:
            calibration_rows = [
                other
                for other in complete
                if other["slug"] != row["slug"]
                and float(other["target_nmse"]) == float(row["target_nmse"])
            ]
            calibration_ratio = float(
                np.median(
                    [
                        float(other["test_oracle_channels"]) / float(other["predicted_channels"])
                        for other in calibration_rows
                    ]
                )
            )
            calibrated_predictions.append(
                max(1, int(np.ceil(calibration_ratio * float(row["predicted_channels"]))))
            )
        calibrated = np.asarray(calibrated_predictions, dtype=np.float64)
    by_target: dict[str, Any] = {}
    for target in sorted({float(row["target_nmse"]) for row in complete}, reverse=True):
        selected = [row for row in complete if float(row["target_nmse"]) == target]
        target_predicted = np.asarray(
            [float(row["predicted_channels"]) for row in selected], dtype=np.float64
        )
        target_empirical = np.asarray(
            [float(row["test_oracle_channels"]) for row in selected], dtype=np.float64
        )
        target_ratios = np.asarray(
            [
                float(row["test_oracle_channels"]) / float(row["predicted_channels"])
                for row in selected
            ]
        )
        target_calibrated: list[int] = []
        if len(selected) > 1:
            for row in selected:
                other_ratios = [
                    float(other["test_oracle_channels"]) / float(other["predicted_channels"])
                    for other in selected
                    if other["slug"] != row["slug"]
                ]
                target_calibrated.append(
                    max(
                        1,
                        int(
                            np.ceil(
                                float(np.median(other_ratios)) * float(row["predicted_channels"])
                            )
                        ),
                    )
                )
        validation_channels = np.asarray(
            [float(row["empirical_channels"]) for row in selected], dtype=np.float64
        )
        by_target[f"{target:.3f}"] = {
            "count": len(selected),
            "median_empirical_to_mssrd_channel_ratio": float(np.median(target_ratios)),
            "minimum_ratio": float(np.min(target_ratios)),
            "maximum_ratio": float(np.max(target_ratios)),
            "mssrd_channel_mape": float(
                np.mean(np.abs(target_predicted - target_empirical) / target_empirical)
            ),
            "mssrd_channel_mae": float(np.mean(np.abs(target_predicted - target_empirical))),
            "validation_boundary_channel_mape": float(
                np.mean(np.abs(validation_channels - target_empirical) / target_empirical)
            ),
            "leave_one_dataset_out_calibrated_channel_mape": (
                float(
                    np.mean(
                        np.abs(np.asarray(target_calibrated) - target_empirical) / target_empirical
                    )
                )
                if target_calibrated
                else None
            ),
            "leave_one_dataset_out_calibrated_channel_mae": (
                float(np.mean(np.abs(np.asarray(target_calibrated) - target_empirical)))
                if target_calibrated
                else None
            ),
            "mssrd_prediction_test_pass_fraction": float(
                np.mean([bool(row["predicted_test_passed"]) for row in selected])
            ),
            "validation_boundary_test_pass_fraction": float(
                np.mean([bool(row["boundary_test_passed"]) for row in selected])
            ),
        }
    return {
        "dataset_count": dataset_count,
        "target_count": len({float(row["target_nmse"]) for row in complete}),
        "comparison_count": len(complete),
        "mssrd_log_channel_pearson": (
            float(np.corrcoef(np.log(predicted), np.log(empirical))[0, 1])
            if len(complete) > 1 and np.ptp(predicted) > 0 and np.ptp(empirical) > 0
            else None
        ),
        "median_empirical_to_mssrd_channel_ratio": float(np.median(ratios)),
        "mssrd_channel_mape": float(np.mean(np.abs(predicted - empirical) / empirical)),
        "leave_one_dataset_out_calibrated_channel_mape": (
            float(np.mean(np.abs(calibrated - empirical) / empirical))
            if calibrated is not None
            else None
        ),
        "leave_one_dataset_out_calibrated_channel_mae": (
            float(np.mean(np.abs(calibrated - empirical))) if calibrated is not None else None
        ),
        "by_target": by_target,
    }


def _summary_row(
    *,
    split: PaperDataset,
    result: MSSRDResult,
    bootstrap: dict[str, Any],
    empirical: dict[str, Any] | None,
) -> dict[str, Any]:
    prediction = result.prediction
    row: dict[str, Any] = {
        "slug": split.slug,
        "dataset": split.name,
        "category": split.category,
        "original_shape": list(split.original_shape),
        "grayscale_shape": list(result.input_shape[:2]),
        "train_available": split.train_available,
        "test_available": split.test_available,
        "train_used": result.sample_count,
        "test_used": len(split.test),
        "grayscale_mapping": "BT.601 luma" if len(split.original_shape) == 3 else "identity",
        "preprocessing": (
            "scale to [0,1], subtract training per-pixel mean; no per-pixel variance normalization"
        ),
        "retained_variance_target": result.retained_variance,
        "global_pca_dimension": result.global_pca_dimension,
        "constant_input_features": result.constant_input_features,
        "scales": [item.to_dict(include_eigenvalues=False) for item in result.scales],
        "predicted_q": prediction.scale,
        "predicted_grid_h": prediction.grid_height,
        "predicted_grid_w": prediction.grid_width,
        "predicted_channels": prediction.channels,
        "predicted_latent_dimension": prediction.latent_scalars,
        "predicted_compression_ratio": (
            int(np.prod(result.input_shape)) / prediction.latent_scalars
        ),
        "bootstrap": bootstrap,
    }
    if empirical is not None:
        row.update(
            {
                "empirical_q": int(empirical["q"]),
                "empirical_grid_h": int(empirical["grid_h"]),
                "empirical_grid_w": int(empirical["grid_w"]),
                "empirical_channels": int(empirical["channels"]),
                "empirical_latent_dimension": int(empirical["latent_dimension"]),
                "empirical_nmse": float(empirical["final_nmse"]),
                "prediction_ratio": (
                    prediction.latent_scalars / int(empirical["latent_dimension"])
                ),
            }
        )
    return row


def _reference_check(
    summaries: list[dict[str, Any]],
    *,
    seed: int,
    max_train: int,
    max_test: int,
    target_nmse: float,
) -> dict[str, Any]:
    reference_path = files("mssrd.paper").joinpath("reference_results.json")
    reference = json.loads(reference_path.read_text(encoding="utf-8"))
    comparable = (
        seed == 20260924
        and max_train == 20000
        and max_test == 5000
        and np.isclose(target_nmse, float(reference["target_nmse"]))
    )
    checks: list[dict[str, Any]] = []
    for row in summaries:
        expected = reference["datasets"].get(str(row["slug"]))
        if expected is None:
            continue
        actual = {
            "q": row["predicted_q"],
            "channels": row["predicted_channels"],
            "latent_dimension": row["predicted_latent_dimension"],
        }
        checks.append(
            {
                "slug": row["slug"],
                "actual": actual,
                "expected": expected["prediction"],
                "matches": actual == expected["prediction"] if comparable else None,
            }
        )
    return {
        "comparable_to_paper_defaults": comparable,
        "all_spectral_predictions_match": (
            all(item["matches"] for item in checks) if comparable else None
        ),
        "checks": checks,
    }


def _metrics(summaries: list[dict[str, Any]], repeats: list[dict[str, Any]]) -> dict[str, Any]:
    complete = [row for row in summaries if "empirical_latent_dimension" in row]
    if not complete:
        return {}
    predicted = np.asarray([row["predicted_latent_dimension"] for row in complete], dtype=float)
    empirical = np.asarray([row["empirical_latent_dimension"] for row in complete], dtype=float)
    ratio = predicted / empirical
    metrics: dict[str, Any] = {
        "dataset_count": len(complete),
        "exact_match_fraction": float(np.mean(predicted == empirical)),
        "mape": float(np.mean(np.abs(predicted - empirical) / empirical)),
        "median_ape": float(np.median(np.abs(predicted - empirical) / empirical)),
        "max_ape": float(np.max(np.abs(predicted - empirical) / empirical)),
        "log_dimension_pearson": (
            float(np.corrcoef(np.log(predicted), np.log(empirical))[0, 1])
            if len(complete) > 1
            else None
        ),
        "within_10_percent_fraction": float(np.mean(np.maximum(ratio, 1.0 / ratio) <= 1.1)),
        "within_25_percent_fraction": float(np.mean(np.maximum(ratio, 1.0 / ratio) <= 1.25)),
    }
    if repeats:
        for role in ("predicted", "empirical_boundary", "below_boundary"):
            selected = [bool(row["passed"]) for row in repeats if row["role"] == role]
            metrics[f"{role}_pass_fraction_additional_seeds"] = float(np.mean(selected))
    return metrics


def _build_figures(
    summaries: list[dict[str, Any]],
    runs: list[dict[str, Any]],
    repeats: list[dict[str, Any]],
    unet_runs: list[dict[str, Any]],
    true_bottleneck_runs: list[dict[str, Any]],
    true_bottleneck_boundaries: list[dict[str, Any]],
    spectra: dict[str, dict[int, np.ndarray]],
    output_dir: Path,
    target_nmse: float,
) -> None:
    figure_dir = output_dir / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9})

    available = [str(row["slug"]) for row in summaries if str(row["slug"]) in spectra]
    representatives = [
        slug for slug in ("mnist", "fashion_mnist", "cifar10", "chestmnist") if slug in spectra
    ]
    representatives.extend(slug for slug in available if slug not in representatives)
    representatives = representatives[:4]
    if representatives:
        row_count = (len(representatives) + 1) // 2
        column_count = min(2, len(representatives))
        figure, axes = plt.subplots(
            row_count,
            column_count,
            figsize=(7.2, 2.85 * row_count),
            constrained_layout=True,
            squeeze=False,
        )
        for axis in axes.flat:
            axis.set_visible(False)
        for axis, slug in zip(axes.flat, representatives, strict=False):
            axis.set_visible(True)
            for scale, values in sorted(spectra[slug].items()):
                normalized = values / max(float(values.sum()), np.finfo(float).eps)
                axis.plot(np.arange(1, len(values) + 1), normalized, label=f"q={scale}")
            axis.set(yscale="log", xlabel="Patch eigenmode", ylabel="Variance fraction", title=slug)
            axis.grid(alpha=0.2)
            axis.legend(frameon=False, fontsize=7)
        figure.savefig(figure_dir / "multiscale_spectra.png", dpi=220)
        figure.savefig(figure_dir / "multiscale_spectra.pdf")
        plt.close(figure)

    if all("empirical_latent_dimension" in row for row in summaries):
        categories = sorted({str(row["category"]) for row in summaries})
        palette = dict(
            zip(
                categories,
                plt.cm.Set2(np.linspace(0.05, 0.95, len(categories))),
                strict=True,
            )
        )
        maximum = max(
            max(float(row["predicted_latent_dimension"]), float(row["empirical_latent_dimension"]))
            for row in summaries
        )
        minimum = min(
            min(float(row["predicted_latent_dimension"]), float(row["empirical_latent_dimension"]))
            for row in summaries
        )
        figure, axis = plt.subplots(figsize=(7.3, 5.2), constrained_layout=True)
        handles: list[Line2D] = []
        for index, row in enumerate(summaries, start=1):
            x = float(row["empirical_latent_dimension"])
            y = float(row["predicted_latent_dimension"])
            color = palette[str(row["category"])]
            axis.scatter(x, y, s=95, color=color, edgecolor="black", linewidth=0.45)
            x_offset = 5 if index % 2 else -6
            y_offset = 5 if index % 3 else -7
            axis.annotate(
                str(index),
                (x, y),
                xytext=(x_offset, y_offset),
                textcoords="offset points",
                ha="center",
                va="center",
                fontsize=7,
            )
            handles.append(
                Line2D(
                    [],
                    [],
                    marker="o",
                    linestyle="None",
                    markerfacecolor=color,
                    markeredgecolor="black",
                    markeredgewidth=0.4,
                    label=f"{index}. {row['dataset']}",
                )
            )
        axis.plot(
            [minimum / 1.25, maximum * 1.1],
            [minimum / 1.25, maximum * 1.1],
            color="#555555",
            linestyle="--",
            linewidth=1,
        )
        axis.set(
            xscale="log",
            yscale="log",
            xlabel="Empirical minimum tested latent scalars",
            ylabel="Training-free predicted latent scalars",
            title=f"Predicted versus empirical bottlenecks at NMSE <= {target_nmse:g}",
        )
        axis.grid(True, which="both", alpha=0.2)
        axis.set_xlim(minimum / 1.25, maximum * 1.15)
        axis.set_ylim(minimum / 1.25, maximum * 1.15)
        axis.legend(
            handles=handles,
            loc="upper left",
            bbox_to_anchor=(1.01, 1.0),
            frameon=False,
            fontsize=7.2,
        )
        figure.savefig(figure_dir / "prediction_vs_empirical.png", dpi=220)
        figure.savefig(figure_dir / "prediction_vs_empirical.pdf")
        plt.close(figure)

        x = np.arange(len(summaries))
        width = 0.26
        figure, axis = plt.subplots(figsize=(7.4, 4.1), constrained_layout=True)
        axis.bar(
            x - width,
            [
                np.nan
                if row.get("global_pca_dimension", row.get("global_pca95_dimension")) is None
                else float(row.get("global_pca_dimension", row.get("global_pca95_dimension")))
                for row in summaries
            ],
            width,
            label="Global PCA",
        )
        axis.bar(
            x,
            [float(row["predicted_latent_dimension"]) for row in summaries],
            width,
            label="MS-SRD prediction",
        )
        axis.bar(
            x + width,
            [float(row["empirical_latent_dimension"]) for row in summaries],
            width,
            label="Nonlinear CNN",
        )
        axis.set(
            yscale="log",
            ylabel="Latent scalar count",
            xticks=x,
            xticklabels=[str(row["slug"]) for row in summaries],
            title="Global, multiscale, and empirical bottleneck dimensions",
        )
        axis.tick_params(axis="x", rotation=50)
        axis.grid(axis="y", alpha=0.2)
        axis.legend(frameon=False, ncol=3)
        figure.savefig(figure_dir / "dimension_comparison.png", dpi=220)
        figure.savefig(figure_dir / "dimension_comparison.pdf")
        plt.close(figure)

    if runs and representatives:
        row_count = (len(representatives) + 1) // 2
        column_count = min(2, len(representatives))
        figure, axes = plt.subplots(
            row_count,
            column_count,
            figsize=(7.2, 2.9 * row_count),
            constrained_layout=True,
            squeeze=False,
        )
        for axis in axes.flat:
            axis.set_visible(False)
        for axis, slug in zip(axes.flat, representatives, strict=False):
            axis.set_visible(True)
            subset = [row for row in runs if row["slug"] == slug]
            for scale in sorted({int(row["q"]) for row in subset}):
                rows = sorted(
                    (row for row in subset if int(row["q"]) == scale),
                    key=lambda row: int(row["latent_dimension"]),
                )
                axis.plot(
                    [int(row["latent_dimension"]) for row in rows],
                    [float(row["final_nmse"]) for row in rows],
                    marker="o",
                    markersize=3,
                    label=f"q={scale}",
                )
            summary = next(row for row in summaries if row["slug"] == slug)
            axis.axhline(target_nmse, color="#333333", linestyle="--", linewidth=1)
            axis.axvline(
                float(summary["predicted_latent_dimension"]),
                color="#d55e00",
                linestyle=":",
                linewidth=1.2,
            )
            axis.set(
                xscale="log",
                yscale="log",
                xlabel="Latent scalars",
                ylabel="External-pool NMSE",
                title=slug,
            )
            axis.set_ylim(1e-4, 1.2)
            axis.xaxis.set_major_locator(LogLocator(base=10, numticks=4))
            axis.xaxis.set_minor_formatter(NullFormatter())
            axis.grid(alpha=0.2, which="both")
            axis.legend(frameon=False, fontsize=7)
        figure.savefig(figure_dir / "distortion_curves.png", dpi=220)
        figure.savefig(figure_dir / "distortion_curves.pdf")
        plt.close(figure)

    if repeats:
        order = [str(row["slug"]) for row in summaries]
        roles = {
            "below_boundary": ("#d55e00", "v", "one channel below"),
            "empirical_boundary": ("#009e73", "o", "empirical boundary"),
            "predicted": ("#0072b2", "D", "MS-SRD prediction"),
        }
        x = np.arange(len(order))
        figure, axis = plt.subplots(figsize=(7.4, 3.6), constrained_layout=True)
        for role, (color, marker, label) in roles.items():
            means, lower, upper = [], [], []
            for slug in order:
                values = np.asarray(
                    [
                        float(row["final_nmse"])
                        for row in repeats
                        if row["slug"] == slug and row["role"] == role
                    ]
                )
                means.append(float(values.mean()))
                lower.append(float(values.mean() - values.min()))
                upper.append(float(values.max() - values.mean()))
            axis.errorbar(
                x,
                means,
                yerr=np.vstack([lower, upper]),
                color=color,
                marker=marker,
                markersize=4.5,
                linewidth=1.0,
                capsize=2,
                label=label,
            )
        axis.axhline(
            target_nmse,
            color="#333333",
            linestyle="--",
            linewidth=1.1,
            label=f"{target_nmse:g} NMSE target",
        )
        axis.set(
            xticks=x,
            xticklabels=order,
            ylabel="External-pool NMSE",
            title="Boundary stability across two additional training seeds",
        )
        axis.tick_params(axis="x", rotation=45, labelsize=7.5)
        axis.grid(axis="y", alpha=0.2)
        axis.legend(frameon=False, ncol=2, fontsize=7.5)
        figure.savefig(figure_dir / "boundary_robustness.png", dpi=220)
        figure.savefig(figure_dir / "boundary_robustness.pdf")
        plt.close(figure)

    if unet_runs:
        order = [str(row["slug"]) for row in summaries]
        configurations = (
            (
                "full_skip_zero_bottleneck",
                "bottleneck disabled",
                "#cc79a7",
                "X",
            ),
            (
                "full_skip_one_channel",
                "one channel",
                "#009e73",
                "s",
            ),
        )
        figure, axis = plt.subplots(figsize=(7.5, 3.9), constrained_layout=True)
        x = np.arange(len(order))
        for role, label, color, marker in configurations:
            values = []
            for slug in order:
                row = next(
                    (item for item in unet_runs if item["slug"] == slug and item["role"] == role),
                    None,
                )
                values.append(float(row["final_nmse"]) if row else np.nan)
            axis.plot(
                x,
                values,
                color=color,
                marker=marker,
                markersize=4,
                linewidth=1,
                label=label,
            )
        axis.axhline(
            target_nmse,
            color="#333333",
            linestyle="--",
            linewidth=1,
            label=f"{target_nmse:g} target",
        )
        axis.set(
            yscale="log",
            xticks=x,
            xticklabels=order,
            ylabel="External-split NMSE",
            title="U-Net skip paths bypass the terminal bottleneck",
        )
        axis.set_ylim(1e-4, 1.2)
        axis.tick_params(axis="x", rotation=55, labelsize=7)
        axis.grid(axis="y", which="both", alpha=0.2)
        axis.legend(frameon=False, fontsize=7, ncol=2)
        figure.savefig(figure_dir / "unet_validation.png", dpi=220)
        figure.savefig(figure_dir / "unet_validation.pdf")
        plt.close(figure)

    if true_bottleneck_runs:
        order = [str(row["slug"]) for row in summaries]
        target_colors = {
            0.10: "#0072b2",
            0.05: "#009e73",
            0.02: "#e69f00",
            0.01: "#cc79a7",
        }
        figure, axes = plt.subplots(4, 3, figsize=(7.5, 8.8), constrained_layout=True)
        for axis in axes.flat:
            axis.set_visible(False)
        for axis, slug in zip(axes.flat, order, strict=False):
            axis.set_visible(True)
            rows = sorted(
                (row for row in true_bottleneck_runs if row["slug"] == slug),
                key=lambda row: int(row["channels"]),
            )
            axis.plot(
                [int(row["channels"]) for row in rows],
                [float(row["test_nmse"]) for row in rows],
                color="#333333",
                marker="o",
                markersize=2.5,
                linewidth=0.8,
            )
            selected_boundaries = [row for row in true_bottleneck_boundaries if row["slug"] == slug]
            for boundary in selected_boundaries:
                target = float(boundary["target_nmse"])
                color = target_colors.get(target, "#666666")
                is_primary = np.isclose(target, target_nmse)
                axis.axhline(
                    target,
                    color=color,
                    linewidth=0.9 if is_primary else 0.5,
                    alpha=0.9 if is_primary else 0.35,
                )
                if boundary["test_oracle_channels"] is not None:
                    axis.scatter(
                        [int(boundary["test_oracle_channels"])],
                        [float(boundary["test_oracle_nmse"])],
                        color=color,
                        marker="o",
                        s=18,
                        zorder=3,
                    )
                axis.scatter(
                    [int(boundary["predicted_channels"])],
                    [float(boundary["predicted_test_nmse"])],
                    facecolors="none",
                    edgecolors=color,
                    marker="s",
                    s=22,
                    linewidths=0.8,
                    zorder=3,
                )
            axis.set(yscale="log", xlabel="Terminal channels", ylabel="Test NMSE", title=slug)
            axis.set_ylim(5e-3, max(0.2, 1.15 * max(float(row["test_nmse"]) for row in rows)))
            axis.grid(alpha=0.18, which="both")
            axis.tick_params(labelsize=7)
        handles = [
            Line2D([], [], color=color, label=f"NMSE {target:g}")
            for target, color in target_colors.items()
        ]
        handles.extend(
            [
                Line2D(
                    [],
                    [],
                    marker="o",
                    linestyle="None",
                    color="#333333",
                    label="retrospective test boundary",
                ),
                Line2D(
                    [],
                    [],
                    marker="s",
                    linestyle="None",
                    markerfacecolor="none",
                    markeredgecolor="#333333",
                    label="MS-SRD width",
                ),
            ]
        )
        figure.legend(handles=handles, loc="outside lower center", ncol=3, frameon=False)
        figure.savefig(figure_dir / "true_bottleneck_curves.png", dpi=220)
        figure.savefig(figure_dir / "true_bottleneck_curves.pdf")
        plt.close(figure)

    complete_boundaries = [
        row for row in true_bottleneck_boundaries if row["test_oracle_channels"] is not None
    ]
    if complete_boundaries:
        target_colors = {
            0.10: "#0072b2",
            0.05: "#009e73",
            0.02: "#e69f00",
            0.01: "#cc79a7",
        }
        figure, (axis_scatter, axis_ratio) = plt.subplots(
            1, 2, figsize=(7.5, 3.5), constrained_layout=True
        )
        maximum = max(
            max(int(row["predicted_channels"]), int(row["test_oracle_channels"]))
            for row in complete_boundaries
        )
        for target in sorted(target_colors, reverse=True):
            rows = [row for row in complete_boundaries if float(row["target_nmse"]) == target]
            color = target_colors[target]
            is_primary = np.isclose(target, target_nmse)
            axis_scatter.scatter(
                [int(row["predicted_channels"]) for row in rows],
                [int(row["test_oracle_channels"]) for row in rows],
                color=color,
                s=36 if is_primary else 20,
                alpha=1.0 if is_primary else 0.5,
                edgecolors="#222222" if is_primary else "none",
                linewidths=0.5 if is_primary else 0.0,
                label=f"NMSE {target:g}",
            )
            axis_ratio.scatter(
                np.full(len(rows), target),
                [int(row["test_oracle_channels"]) / int(row["predicted_channels"]) for row in rows],
                color=color,
                s=32 if is_primary else 18,
                alpha=1.0 if is_primary else 0.45,
                edgecolors="#222222" if is_primary else "none",
                linewidths=0.5 if is_primary else 0.0,
            )
        median_targets = sorted(target_colors)
        median_ratios = [
            float(
                np.median(
                    [
                        int(row["test_oracle_channels"]) / int(row["predicted_channels"])
                        for row in complete_boundaries
                        if float(row["target_nmse"]) == target
                    ]
                )
            )
            for target in median_targets
        ]
        axis_ratio.plot(
            median_targets,
            median_ratios,
            color="#333333",
            marker="D",
            markersize=4,
            linewidth=1.2,
            label="median",
        )
        axis_scatter.plot(
            [1, maximum], [1, maximum], color="#555555", linestyle="--", linewidth=0.9
        )
        axis_scatter.set(
            xscale="log",
            yscale="log",
            xlabel="MS-SRD channels",
            ylabel="Empirical U-Net channels",
            title="True-bottleneck width",
        )
        axis_scatter.grid(alpha=0.2, which="both")
        axis_scatter.legend(frameon=False, fontsize=7)
        ordered_targets = sorted(target_colors, reverse=True)
        axis_ratio.set(
            xticks=ordered_targets,
            xticklabels=[f"{target:g}" for target in ordered_targets],
            xlabel="NMSE target",
            ylabel="Empirical / MS-SRD channels",
            title="Nonlinear width ratio",
        )
        axis_ratio.grid(axis="y", alpha=0.2)
        axis_ratio.legend(frameon=False, fontsize=7)
        figure.savefig(figure_dir / "unet_mssrd_relation.png", dpi=220)
        figure.savefig(figure_dir / "unet_mssrd_relation.pdf")
        plt.close(figure)


def reproduce_paper(
    *,
    data_dir: str | Path = "data",
    output_dir: str | Path = "paper-results",
    seed: int = 20260924,
    datasets: Sequence[str] | None = None,
    max_train: int = 20000,
    max_test: int = 5000,
    max_patch_samples: int = 250000,
    bootstrap_reps: int = 20,
    steps: int = 160,
    batch_size: int = 8192,
    unet_steps: int = 800,
    least_volume_steps: int = 800,
    unet_batch_size: int = 256,
    unet_targets: Sequence[float] = UNET_NMSE_TARGETS,
    target_nmse: float = PRIMARY_NMSE_TARGET,
    device: str = "auto",
    download: bool = True,
    spectral_only: bool = False,
    unet_only: bool = False,
    run_repeats: bool = True,
    run_unet_validation: bool = True,
    run_full_skip_validation: bool = True,
    run_true_bottleneck_validation: bool = True,
    run_geometry_stress_experiment: bool = True,
    run_deployable_baseline_experiment: bool = True,
) -> list[dict[str, Any]]:
    """Reproduce spectral predictions, neural validation, and paper figures.

    The default arguments match the manuscript. Results are cached per neural
    candidate so an interrupted run can be resumed by repeating the command.
    """

    output = Path(output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    if spectral_only and unet_only:
        raise ValueError("spectral_only and unet_only cannot both be enabled")
    selected = tuple(item[0] for item in PAPER_DATASETS) if datasets is None else tuple(datasets)
    known = {item[0] for item in PAPER_DATASETS}
    unknown = sorted(set(selected) - known)
    if unknown:
        raise ValueError(f"unknown dataset slugs: {unknown}")
    training_device = _device(device)
    print(f"Paper reproduction device: {training_device}", flush=True)
    summaries: list[dict[str, Any]] = []
    all_runs: list[dict[str, Any]] = []
    all_repeats: list[dict[str, Any]] = []
    all_unet_runs: list[dict[str, Any]] = []
    all_true_bottleneck_runs: list[dict[str, Any]] = []
    all_true_bottleneck_boundaries: list[dict[str, Any]] = []
    spectra: dict[str, dict[int, np.ndarray]] = {}
    for slug in selected:
        split = load_paper_dataset(
            slug,
            data_dir=data_dir,
            max_train=max_train,
            max_test=max_test,
            seed=seed,
            download=download,
        )
        train = _paper_grayscale(split.train, split.intensity_divisor)
        test = _paper_grayscale(split.test, split.intensity_divisor)
        scales = _paper_scales(train.shape[1], train.shape[2])
        print(
            f"[{slug}] train={len(train)} test={len(test)} shape={train.shape[1:]} scales={scales}",
            flush=True,
        )
        estimator = MSSRD(
            target_nmse=target_nmse,
            scales=scales,
            color_mode="grayscale",
            seed=seed,
            batch_size=512,
            compute_global=True,
        )
        result = estimator.fit(train)
        mean_image = estimator.mean_image_[..., 0].astype(np.float32)
        centered_train = np.ascontiguousarray(train - mean_image)
        centered_test = np.ascontiguousarray(test - mean_image)
        bootstrap = _bootstrap_geometry(
            train,
            scales=scales,
            retained_variance=1.0 - target_nmse,
            repetitions=bootstrap_reps,
            seed=seed + 77,
        )
        dataset_dir = output / slug
        dataset_dir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            dataset_dir / "spectra.npz",
            global_eigenvalues=result.global_eigenvalues,
            **{f"q{item.scale}_eigenvalues": item.eigenvalues for item in result.scales},
        )
        spectra[slug] = {item.scale: item.eigenvalues for item in result.scales}
        target_predictions = _spectral_predictions_for_targets(result, unet_targets)
        empirical: dict[str, Any] | None = None
        dataset_runs: list[dict[str, Any]] = []
        dataset_unet_runs: list[dict[str, Any]] = []
        dataset_true_bottleneck_runs: list[dict[str, Any]] = []
        dataset_true_bottleneck_boundaries: list[dict[str, Any]] = []
        if not spectral_only:
            if not unet_only:
                for estimate in result.scales:
                    dataset_runs.extend(
                        _validate_scale(
                            split=split,
                            centered_train=centered_train,
                            centered_test=centered_test,
                            estimate=estimate,
                            device=training_device,
                            dataset_dir=dataset_dir,
                            seed=seed,
                            steps=steps,
                            batch_size=batch_size,
                            target_nmse=target_nmse,
                            max_patch_samples=max_patch_samples,
                        )
                    )
                passing = [row for row in dataset_runs if bool(row["passed"])]
                if not passing:
                    raise RuntimeError(f"no tested bottleneck passed the target for {slug}")
                empirical = min(
                    passing,
                    key=lambda row: (
                        int(row["latent_dimension"]),
                        int(row["q"]),
                        int(row["channels"]),
                    ),
                )
                _write_csv(dataset_dir / "training_runs.csv", dataset_runs)
                all_runs.extend(dataset_runs)
                if run_repeats:
                    repeated = _repeat_boundaries(
                        split=split,
                        centered_train=centered_train,
                        centered_test=centered_test,
                        result=result,
                        empirical=empirical,
                        device=training_device,
                        seed=seed,
                        steps=steps,
                        batch_size=batch_size,
                        target_nmse=target_nmse,
                        max_patch_samples=max_patch_samples,
                    )
                    all_repeats.extend(repeated)
                    _write_csv(dataset_dir / "repeat_validation.csv", repeated)
            if run_unet_validation and run_full_skip_validation:
                dataset_unet_runs = _validate_unet(
                    split=split,
                    centered_train=centered_train,
                    centered_test=centered_test,
                    prediction=result.prediction,
                    device=training_device,
                    dataset_dir=dataset_dir,
                    seed=seed,
                    steps=unet_steps,
                    batch_size=unet_batch_size,
                    target_nmse=target_nmse,
                )
                all_unet_runs.extend(dataset_unet_runs)
                _write_csv(dataset_dir / "unet_runs.csv", dataset_unet_runs)
            if run_unet_validation and run_true_bottleneck_validation:
                (
                    dataset_true_bottleneck_runs,
                    dataset_true_bottleneck_boundaries,
                ) = _validate_true_bottleneck_unet(
                    split=split,
                    centered_train=centered_train,
                    centered_test=centered_test,
                    predictions=target_predictions,
                    scale_estimates=result.scales,
                    device=training_device,
                    dataset_dir=dataset_dir,
                    seed=seed,
                    steps=unet_steps,
                    batch_size=unet_batch_size,
                )
                all_true_bottleneck_runs.extend(dataset_true_bottleneck_runs)
                all_true_bottleneck_boundaries.extend(dataset_true_bottleneck_boundaries)
                _write_csv(
                    dataset_dir / "true_bottleneck_runs.csv",
                    dataset_true_bottleneck_runs,
                )
                _write_csv(
                    dataset_dir / "true_bottleneck_boundaries.csv",
                    dataset_true_bottleneck_boundaries,
                )
        summary = _summary_row(split=split, result=result, bootstrap=bootstrap, empirical=empirical)
        if dataset_unet_runs:
            summary["unet_validation"] = _unet_dataset_summary(
                dataset_unet_runs, result.prediction.latent_scalars
            )
        if dataset_true_bottleneck_boundaries:
            summary["true_bottleneck_unet"] = dataset_true_bottleneck_boundaries
        summaries.append(summary)
        _write_json(dataset_dir / "summary.json", summary)
        del train, test, centered_train, centered_test

    _write_json(output / "summary.json", summaries)
    _write_csv(output / "dataset_summary.csv", summaries)
    _write_csv(output / "training_runs.csv", all_runs)
    _write_csv(output / "repeat_validation.csv", all_repeats)
    _write_csv(output / "unet_runs.csv", all_unet_runs)
    _write_csv(output / "true_bottleneck_runs.csv", all_true_bottleneck_runs)
    _write_csv(output / "true_bottleneck_boundaries.csv", all_true_bottleneck_boundaries)
    reference_check = _reference_check(
        summaries,
        seed=seed,
        max_train=max_train,
        max_test=max_test,
        target_nmse=target_nmse,
    )
    _write_json(output / "reference_check.json", reference_check)
    metrics = _metrics(summaries, all_repeats)
    _write_json(output / "metrics.json", metrics)
    unet_metrics = _unet_metrics(all_unet_runs)
    _write_json(output / "unet_metrics.json", unet_metrics)
    true_bottleneck_metrics = _true_bottleneck_metrics(all_true_bottleneck_boundaries)
    _write_json(output / "true_bottleneck_metrics.json", true_bottleneck_metrics)
    _build_figures(
        summaries,
        all_runs,
        all_repeats,
        all_unet_runs,
        all_true_bottleneck_runs,
        all_true_bottleneck_boundaries,
        spectra,
        output,
        target_nmse,
    )
    if run_geometry_stress_experiment and not unet_only and "oxford_pets" in selected:
        from mssrd.paper.geometry import run_geometry_stress

        run_geometry_stress(
            data_dir=data_dir,
            output_dir=output / "geometry_stress",
            seed=seed,
            max_train=min(max_train, 3680),
            download=download,
            target_nmse=target_nmse,
        )
    if run_deployable_baseline_experiment and not spectral_only and not unet_only:
        from mssrd.paper.baselines import BASELINE_DATASETS, run_deployable_baselines

        baseline_datasets = tuple(slug for slug in BASELINE_DATASETS if slug in selected)
        if baseline_datasets:
            run_deployable_baselines(
                data_dir=data_dir,
                output_dir=output / "deployable_baselines",
                seed=seed,
                max_train=max_train,
                max_test=max_test,
                max_patch_samples=max_patch_samples,
                steps=steps,
                least_volume_steps=least_volume_steps,
                batch_size=batch_size,
                device=device,
                download=download,
                datasets=baseline_datasets,
                target_nmse=target_nmse,
            )
    print(f"Wrote paper results to {output}", flush=True)
    if reference_check["all_spectral_predictions_match"] is False:
        print("WARNING: spectral predictions differ from the committed paper reference", flush=True)
    return summaries

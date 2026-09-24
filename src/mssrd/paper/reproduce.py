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


def _paper_scales(height: int, width: int) -> tuple[int, ...]:
    divisors = [
        scale
        for scale in range(2, min(height, width) + 1)
        if height % scale == 0 and width % scale == 0
    ]
    preferred = tuple(scale for scale in divisors if scale in {2, 4, 7, 8})
    return preferred or tuple(divisors[:3])


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
        return cache[key]

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
                    "passed": float(metrics["final_nmse"]) <= 0.05,
                    **metrics,
                }
            )
            print(
                f"[{split.slug}] repeat {role} q={scale} c={channels} "
                f"seed={repeat_seed} NMSE={metrics['final_nmse']:.4f}",
                flush=True,
            )
    return rows


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
        "global_pca95_dimension": result.global_pca_dimension,
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
) -> dict[str, Any]:
    reference_path = files("mssrd.paper").joinpath("reference_results.json")
    reference = json.loads(reference_path.read_text(encoding="utf-8"))
    comparable = seed == 20260924 and max_train == 20000 and max_test == 5000
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
    spectra: dict[str, dict[int, np.ndarray]],
    output_dir: Path,
) -> None:
    figure_dir = output_dir / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9})

    representatives = [
        slug for slug in ("mnist", "fashion_mnist", "cifar10", "chestmnist") if slug in spectra
    ]
    if representatives:
        figure, axes = plt.subplots(2, 2, figsize=(7.2, 5.7), constrained_layout=True)
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
        figure, axis = plt.subplots(figsize=(7.3, 5.2), constrained_layout=True)
        handles: list[Line2D] = []
        for index, row in enumerate(summaries, start=1):
            x = float(row["empirical_latent_dimension"])
            y = float(row["predicted_latent_dimension"])
            color = palette[str(row["category"])]
            axis.scatter(x, y, s=95, color=color, edgecolor="black", linewidth=0.45)
            axis.text(x, y, str(index), ha="center", va="center", fontsize=7)
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
            [1, maximum * 1.1], [1, maximum * 1.1], color="#555555", linestyle="--", linewidth=1
        )
        axis.set(
            xscale="log",
            yscale="log",
            xlabel="Empirical minimum tested latent scalars",
            ylabel="Training-free predicted latent scalars",
            title="Predicted versus empirical 95% fidelity bottlenecks",
        )
        axis.grid(True, which="both", alpha=0.2)
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
            [float(row["global_pca95_dimension"]) for row in summaries],
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
        figure, axes = plt.subplots(2, 2, figsize=(7.2, 5.8), constrained_layout=True)
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
            axis.axhline(0.05, color="#333333", linestyle="--", linewidth=1)
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
                ylabel="Held-out NMSE",
                title=slug,
            )
            axis.set_ylim(8e-3, 1.2)
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
        axis.axhline(0.05, color="#333333", linestyle="--", linewidth=1.1, label="5% NMSE target")
        axis.set(
            xticks=x,
            xticklabels=order,
            ylabel="Held-out NMSE",
            title="Boundary stability across two additional training seeds",
        )
        axis.tick_params(axis="x", rotation=45, labelsize=7.5)
        axis.grid(axis="y", alpha=0.2)
        axis.legend(frameon=False, ncol=2, fontsize=7.5)
        figure.savefig(figure_dir / "boundary_robustness.png", dpi=220)
        figure.savefig(figure_dir / "boundary_robustness.pdf")
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
    device: str = "auto",
    download: bool = True,
    spectral_only: bool = False,
    run_repeats: bool = True,
) -> list[dict[str, Any]]:
    """Reproduce spectral predictions, neural validation, and paper figures.

    The default arguments match the manuscript. Results are cached per neural
    candidate so an interrupted run can be resumed by repeating the command.
    """

    output = Path(output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
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
            retained_variance=0.95,
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
            retained_variance=0.95,
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
        empirical: dict[str, Any] | None = None
        dataset_runs: list[dict[str, Any]] = []
        if not spectral_only:
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
                        target_nmse=0.05,
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
                    max_patch_samples=max_patch_samples,
                )
                all_repeats.extend(repeated)
                _write_csv(dataset_dir / "repeat_validation.csv", repeated)
        summary = _summary_row(split=split, result=result, bootstrap=bootstrap, empirical=empirical)
        summaries.append(summary)
        _write_json(dataset_dir / "summary.json", summary)
        del train, test, centered_train, centered_test

    _write_json(output / "summary.json", summaries)
    _write_csv(output / "dataset_summary.csv", summaries)
    _write_csv(output / "training_runs.csv", all_runs)
    _write_csv(output / "repeat_validation.csv", all_repeats)
    reference_check = _reference_check(summaries, seed=seed, max_train=max_train, max_test=max_test)
    _write_json(output / "reference_check.json", reference_check)
    metrics = _metrics(summaries, all_repeats)
    _write_json(output / "metrics.json", metrics)
    _build_figures(summaries, all_runs, all_repeats, spectra, output)
    print(f"Wrote paper results to {output}", flush=True)
    if reference_check["all_spectral_predictions_match"] is False:
        print("WARNING: spectral predictions differ from the committed paper reference", flush=True)
    return summaries

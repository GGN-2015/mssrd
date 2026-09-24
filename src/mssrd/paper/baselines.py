"""Deployable width-selection baselines used by the paper."""

from __future__ import annotations

import csv
import json
import random
from importlib.resources import files
from pathlib import Path
from typing import Any

import numpy as np
import torch

from mssrd.core import MSSRD
from mssrd.paper.datasets import load_paper_dataset
from mssrd.paper.model import (
    LeastVolumePatchAutoencoder,
    coarse_channels,
    extract_patches,
    train_deployable_candidate,
)

BASELINE_DATASETS = ("mnist", "cifar10", "oxford_pets", "eurosat")


def _grayscale(images: np.ndarray, divisor: float) -> np.ndarray:
    values = images.astype(np.float32) / divisor
    if values.ndim == 4:
        values = 0.299 * values[..., 0] + 0.587 * values[..., 1] + 0.114 * values[..., 2]
    return np.ascontiguousarray(values)


def _split_indices(size: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    validation_size = min(1000, max(1, size // 5))
    permutation = np.random.default_rng(seed).permutation(size)
    return np.sort(permutation[validation_size:]), np.sort(permutation[:validation_size])


def _sample_patches(patches: np.ndarray, maximum: int, seed: int) -> np.ndarray:
    if len(patches) <= maximum:
        return patches
    indices = np.sort(np.random.default_rng(seed).choice(len(patches), maximum, replace=False))
    return np.ascontiguousarray(patches[indices])


def _linear_nmse(patches: np.ndarray, basis: np.ndarray) -> float:
    reconstruction = (patches @ basis) @ basis.T
    denominator = float(np.mean(patches.astype(np.float64) ** 2))
    return float(np.mean((patches - reconstruction) ** 2) / denominator)


def _device(name: str) -> torch.device:
    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else "cpu"
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    return torch.device(name)


def _least_volume_latents(
    model: LeastVolumePatchAutoencoder,
    patches: np.ndarray,
    device: torch.device,
    batch_size: int = 32768,
) -> torch.Tensor:
    values = torch.from_numpy(patches.astype(np.float32, copy=False))
    encoded = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(values), batch_size):
            encoded.append(model.encode(values[start : start + batch_size].to(device)).cpu())
    return torch.cat(encoded)


def _least_volume_pruned_nmse(
    model: LeastVolumePatchAutoencoder,
    latent: torch.Tensor,
    targets: np.ndarray,
    latent_mean: torch.Tensor,
    order: torch.Tensor,
    channels: int,
    device: torch.device,
    batch_size: int = 32768,
) -> float:
    keep = order[:channels]
    denominator = float(np.mean(targets.astype(np.float64) ** 2))
    squared_error = 0.0
    element_count = 0
    target_tensor = torch.from_numpy(targets.astype(np.float32, copy=False))
    model.eval()
    with torch.no_grad():
        for start in range(0, len(latent), batch_size):
            stop = min(start + batch_size, len(latent))
            batch = latent_mean.expand(stop - start, -1).clone()
            batch[:, keep] = latent[start:stop, keep]
            prediction = model.decode(batch.to(device)).cpu()
            target = target_tensor[start:stop]
            squared_error += float(torch.sum((prediction - target) ** 2).item())
            element_count += int(target.numel())
    return float((squared_error / max(element_count, 1)) / denominator)


def _least_volume_select(
    train_patches: np.ndarray,
    validation_patches: np.ndarray,
    test_patches: np.ndarray,
    basis: np.ndarray,
    *,
    predicted_channels: int,
    target_nmse: float,
    device: torch.device,
    seed: int,
    steps: int,
    batch_size: int,
    volume_weight: float = 1e-3,
    volume_offset: float = 1.0,
) -> dict[str, Any]:
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    dimension = train_patches.shape[1]
    hidden = max(32, min(128, 2 * dimension))
    model = LeastVolumePatchAutoencoder(basis[:, :dimension], hidden).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.0)
    train_tensor = torch.from_numpy(train_patches.astype(np.float32, copy=False))
    generator = torch.Generator(device="cpu").manual_seed(seed)
    for _ in range(steps):
        indices = torch.randint(0, len(train_tensor), (batch_size,), generator=generator)
        batch = train_tensor[indices].to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        latent = model.encode(batch)
        reconstruction = model.decode(latent)
        reconstruction_loss = torch.mean((reconstruction - batch) ** 2)
        volume = torch.exp(torch.log(latent.std(dim=0) + volume_offset).mean())
        (reconstruction_loss + volume_weight * volume).backward()
        optimizer.step()

    train_latent = _least_volume_latents(model, train_patches, device)
    validation_latent = _least_volume_latents(model, validation_patches, device)
    test_latent = _least_volume_latents(model, test_patches, device)
    latent_mean = train_latent.mean(dim=0)
    order = torch.argsort(train_latent.std(dim=0), descending=True)
    evaluations: dict[int, tuple[float, float]] = {}

    def evaluate(channels: int) -> tuple[float, float]:
        if channels not in evaluations:
            evaluations[channels] = (
                _least_volume_pruned_nmse(
                    model,
                    validation_latent,
                    validation_patches,
                    latent_mean,
                    order,
                    channels,
                    device,
                ),
                _least_volume_pruned_nmse(
                    model,
                    test_latent,
                    test_patches,
                    latent_mean,
                    order,
                    channels,
                    device,
                ),
            )
        return evaluations[channels]

    coarse = coarse_channels(dimension, predicted_channels)
    for channels in coarse:
        evaluate(channels)
    passing = [channels for channels in coarse if evaluations[channels][0] <= target_nmse]
    first_pass = min(passing, default=dimension)
    lower = max(
        (
            channels
            for channels in coarse
            if channels < first_pass and evaluations[channels][0] > target_nmse
        ),
        default=0,
    )
    for channels in range(lower + 1, first_pass + 1):
        evaluate(channels)
    selected = min(
        (channels for channels, values in evaluations.items() if values[0] <= target_nmse),
        default=dimension,
    )
    validation_nmse, test_nmse = evaluate(selected)
    result = {
        "channels": selected,
        "validation_nmse": validation_nmse,
        "test_nmse": test_nmse,
        "training_runs": 1,
        "steps": steps,
        "volume_weight": volume_weight,
        "volume_offset": volume_offset,
    }
    del train_tensor, train_latent
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return result


def run_deployable_baselines(
    *,
    data_dir: str | Path,
    output_dir: str | Path,
    seed: int,
    max_train: int = 20000,
    max_test: int = 5000,
    max_patch_samples: int = 250000,
    steps: int = 160,
    least_volume_steps: int = 800,
    batch_size: int = 8192,
    device: str = "auto",
    download: bool = True,
    datasets: tuple[str, ...] = BASELINE_DATASETS,
) -> list[dict[str, Any]]:
    """Run training-free, grid-search, and Least-Volume width selectors."""

    target_nmse = 0.05
    training_device = _device(device)
    output = Path(output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    reference = json.loads(
        files("mssrd.paper").joinpath("reference_results.json").read_text(encoding="utf-8")
    )["datasets"]
    rows: list[dict[str, Any]] = []
    for slug in datasets:
        dataset_cache = output / f"{slug}.json"
        signature = {
            "protocol_version": 1,
            "seed": seed,
            "max_train": max_train,
            "max_test": max_test,
            "max_patch_samples": max_patch_samples,
            "steps": steps,
            "least_volume_steps": least_volume_steps,
            "batch_size": batch_size,
            "device_type": training_device.type,
            "target_nmse": target_nmse,
        }
        if dataset_cache.exists():
            cached = json.loads(dataset_cache.read_text(encoding="utf-8"))
            if cached.get("signature") == signature:
                rows.extend(cached["rows"])
                print(f"[baseline:{slug}] loaded cache", flush=True)
                continue
        split = load_paper_dataset(
            slug,
            data_dir=data_dir,
            max_train=max_train,
            max_test=max_test,
            seed=seed,
            download=download,
        )
        development = _grayscale(split.train, split.intensity_divisor)
        external = _grayscale(split.test, split.intensity_divisor)
        train_indices, validation_indices = _split_indices(len(development), seed + 401)
        train = development[train_indices]
        validation = development[validation_indices]
        mean_image = train.mean(axis=0, dtype=np.float64).astype(np.float32)
        train = np.ascontiguousarray(train - mean_image)
        validation = np.ascontiguousarray(validation - mean_image)
        external = np.ascontiguousarray(external - mean_image)
        q = int(reference[slug]["prediction"]["q"])
        estimator = MSSRD(target_nmse=target_nmse, scales=[q], seed=seed, batch_size=512)
        result = estimator.fit(train)
        estimate = result.prediction
        train_patches = _sample_patches(extract_patches(train, q), max_patch_samples, seed + 402)
        validation_patches = extract_patches(validation, q)
        test_patches = extract_patches(external, q)
        grid_runs: dict[int, dict[str, Any]] = {}

        def train_grid(
            channels: int,
            *,
            cache: dict[int, dict[str, Any]] = grid_runs,
            training_values: np.ndarray = train_patches,
            validation_values: np.ndarray = validation_patches,
            test_values: np.ndarray = test_patches,
            scale_basis: np.ndarray = estimate.basis,
        ) -> dict[str, Any]:
            if channels not in cache:
                cache[channels] = train_deployable_candidate(
                    training_values,
                    validation_values,
                    test_values,
                    scale_basis[:, :channels],
                    device=training_device,
                    seed=seed + 1000 + channels,
                    steps=steps,
                    batch_size=batch_size,
                )
            return cache[channels]

        coarse = coarse_channels(estimate.patch_dimension, estimate.channels)
        for channels in coarse:
            train_grid(channels)
        passing = [
            channels for channels in coarse if grid_runs[channels]["validation_nmse"] <= target_nmse
        ]
        first_pass = min(passing, default=estimate.patch_dimension)
        lower = max(
            (
                channels
                for channels in coarse
                if channels < first_pass and grid_runs[channels]["validation_nmse"] > target_nmse
            ),
            default=0,
        )
        for channels in range(lower + 1, first_pass + 1):
            train_grid(channels)
        grid_channels = min(
            (
                channels
                for channels, values in grid_runs.items()
                if values["validation_nmse"] <= target_nmse
            ),
            default=estimate.patch_dimension,
        )
        grid = grid_runs[grid_channels]
        predicted_run = train_grid(estimate.channels)
        least_volume = _least_volume_select(
            train_patches,
            validation_patches,
            test_patches,
            estimate.basis,
            predicted_channels=estimate.channels,
            target_nmse=target_nmse,
            device=training_device,
            seed=seed + 2000,
            steps=least_volume_steps,
            batch_size=batch_size,
        )
        oracle_channels = int(reference[slug]["empirical"]["channels"])
        grid_size = estimate.grid_height * estimate.grid_width
        method_values = (
            (
                "MS-SRD",
                estimate.channels,
                float(predicted_run["validation_nmse"]),
                float(predicted_run["test_nmse"]),
                0,
            ),
            (
                "Block PCA",
                estimate.channels,
                _linear_nmse(validation_patches, estimate.basis[:, : estimate.channels]),
                _linear_nmse(test_patches, estimate.basis[:, : estimate.channels]),
                0,
            ),
            (
                "Validation grid",
                grid_channels,
                float(grid["validation_nmse"]),
                float(grid["test_nmse"]),
                len(grid_runs),
            ),
            (
                "Least Volume",
                int(least_volume["channels"]),
                float(least_volume["validation_nmse"]),
                float(least_volume["test_nmse"]),
                1,
            ),
        )
        for method, channels, validation_nmse, test_nmse, training_runs in method_values:
            rows.append(
                {
                    "slug": slug,
                    "dataset": split.name,
                    "q": q,
                    "method": method,
                    "selected_channels": channels,
                    "selected_latent_scalars": grid_size * channels,
                    "external_oracle_channels": oracle_channels,
                    "channel_absolute_percentage_error": abs(channels - oracle_channels)
                    / oracle_channels,
                    "validation_nmse": validation_nmse,
                    "external_nmse": test_nmse,
                    "external_pass": test_nmse <= target_nmse,
                    "neural_training_runs": training_runs,
                }
            )
        dataset_rows = [row for row in rows if row["slug"] == slug]
        dataset_cache.write_text(
            json.dumps({"signature": signature, "rows": dataset_rows}, indent=2) + "\n",
            encoding="utf-8",
        )
        print(f"[baseline:{slug}] completed", flush=True)

    with (output / "deployable_baselines.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (output / "deployable_baselines.json").write_text(
        json.dumps(rows, indent=2) + "\n", encoding="utf-8"
    )
    metrics = {}
    for method in sorted({row["method"] for row in rows}):
        selected = [row for row in rows if row["method"] == method]
        metrics[method] = {
            "mape": float(np.mean([row["channel_absolute_percentage_error"] for row in selected])),
            "external_pass_fraction": float(np.mean([row["external_pass"] for row in selected])),
            "mean_external_nmse": float(np.mean([row["external_nmse"] for row in selected])),
            "neural_training_runs": int(sum(row["neural_training_runs"] for row in selected)),
        }
    (output / "deployable_baseline_metrics.json").write_text(
        json.dumps(metrics, indent=2) + "\n", encoding="utf-8"
    )
    return rows

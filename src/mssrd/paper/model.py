"""PCA-initialized nonlinear patch autoencoder used by the paper."""

from __future__ import annotations

import random
from typing import Any

import numpy as np
import torch
from torch import nn


def extract_patches(images: np.ndarray, scale: int) -> np.ndarray:
    n, height, width = images.shape
    grid_height = (height + scale - 1) // scale
    grid_width = (width + scale - 1) // scale
    padded = np.pad(
        images,
        ((0, 0), (0, grid_height * scale - height), (0, grid_width * scale - width)),
    )
    return np.ascontiguousarray(
        padded.reshape(n, grid_height, scale, grid_width, scale)
        .transpose(0, 1, 3, 2, 4)
        .reshape(-1, scale * scale)
    )


class NonlinearPatchAutoencoder(nn.Module):
    def __init__(self, basis: np.ndarray, hidden: int):
        super().__init__()
        dimension, channels = basis.shape
        self.encoder_linear = nn.Linear(dimension, channels, bias=False)
        self.encoder_nonlinear = nn.Sequential(
            nn.Linear(dimension, hidden), nn.GELU(), nn.Linear(hidden, channels, bias=False)
        )
        self.decoder_linear = nn.Linear(channels, dimension, bias=False)
        self.decoder_nonlinear = nn.Sequential(
            nn.Linear(channels, hidden), nn.GELU(), nn.Linear(hidden, dimension, bias=False)
        )
        basis_tensor = torch.from_numpy(basis.astype(np.float32))
        with torch.no_grad():
            self.encoder_linear.weight.copy_(basis_tensor.T)
            self.decoder_linear.weight.copy_(basis_tensor)
            nn.init.zeros_(self.encoder_nonlinear[-1].weight)
            nn.init.zeros_(self.decoder_nonlinear[-1].weight)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        latent = self.encoder_linear(values) + self.encoder_nonlinear(values)
        return self.decoder_linear(latent) + self.decoder_nonlinear(latent)


class LeastVolumePatchAutoencoder(NonlinearPatchAutoencoder):
    """Least-Volume adaptation of the shared nonlinear patch autoencoder."""

    def __init__(self, basis: np.ndarray, hidden: int):
        super().__init__(basis, hidden)
        nn.utils.parametrizations.spectral_norm(self.decoder_linear)
        for layer in self.decoder_nonlinear:
            if isinstance(layer, nn.Linear):
                if torch.count_nonzero(layer.weight).item() == 0:
                    nn.init.normal_(layer.weight, std=1e-3)
                nn.utils.parametrizations.spectral_norm(layer)

    def encode(self, values: torch.Tensor) -> torch.Tensor:
        return self.encoder_linear(values) + self.encoder_nonlinear(values)

    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        return self.decoder_linear(latent) + self.decoder_nonlinear(latent)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.decode(self.encode(values))


def _evaluate(
    model: nn.Module,
    values: torch.Tensor,
    denominator: float,
    batch_size: int = 32768,
) -> float:
    squared_error = 0.0
    element_count = 0
    model.eval()
    with torch.no_grad():
        for start in range(0, len(values), batch_size):
            batch = values[start : start + batch_size]
            prediction = model(batch)
            squared_error += float(torch.sum((prediction - batch) ** 2).item())
            element_count += int(batch.numel())
    return float((squared_error / max(element_count, 1)) / denominator)


def train_candidate(
    train_patches: np.ndarray,
    test_patches: np.ndarray,
    basis: np.ndarray,
    *,
    device: torch.device,
    seed: int,
    steps: int,
    batch_size: int,
) -> dict[str, Any]:
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    hidden = max(32, min(128, 2 * train_patches.shape[1]))
    model = NonlinearPatchAutoencoder(basis, hidden).to(device)
    test_tensor = torch.from_numpy(test_patches.astype(np.float32, copy=False)).to(device)
    denominator = float(np.mean(test_patches.astype(np.float64) ** 2))
    initial_nmse = _evaluate(model, test_tensor, denominator)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-6)
    train_tensor = torch.from_numpy(train_patches.astype(np.float32, copy=False))
    generator = torch.Generator(device="cpu").manual_seed(seed)
    best_nmse = initial_nmse
    best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    stale_checks = 0
    completed = 0
    for step in range(1, steps + 1):
        indices = torch.randint(0, len(train_tensor), (batch_size,), generator=generator)
        batch = train_tensor[indices].to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        loss = torch.mean((model(batch) - batch) ** 2)
        loss.backward()
        optimizer.step()
        completed = step
        if step % 40 == 0 or step == steps:
            nmse = _evaluate(model, test_tensor, denominator)
            if nmse < best_nmse - 1e-5:
                best_nmse = nmse
                best_state = {
                    key: value.detach().cpu().clone() for key, value in model.state_dict().items()
                }
                stale_checks = 0
            else:
                stale_checks += 1
            if stale_checks >= 4:
                break
    model.load_state_dict(best_state)
    final_nmse = _evaluate(model, test_tensor, denominator)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    del model, train_tensor, test_tensor
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return {
        "initial_nmse": initial_nmse,
        "final_nmse": final_nmse,
        "steps": completed,
        "parameter_count": parameter_count,
    }


def train_deployable_candidate(
    train_patches: np.ndarray,
    validation_patches: np.ndarray,
    test_patches: np.ndarray,
    basis: np.ndarray,
    *,
    device: torch.device,
    seed: int,
    steps: int,
    batch_size: int,
) -> dict[str, Any]:
    """Train one candidate with checkpointing on validation data only."""

    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    hidden = max(32, min(128, 2 * train_patches.shape[1]))
    model = NonlinearPatchAutoencoder(basis, hidden).to(device)
    validation_tensor = torch.from_numpy(validation_patches.astype(np.float32, copy=False)).to(
        device
    )
    test_tensor = torch.from_numpy(test_patches.astype(np.float32, copy=False)).to(device)
    validation_denominator = float(np.mean(validation_patches.astype(np.float64) ** 2))
    test_denominator = float(np.mean(test_patches.astype(np.float64) ** 2))
    best_validation = _evaluate(model, validation_tensor, validation_denominator)
    best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-6)
    train_tensor = torch.from_numpy(train_patches.astype(np.float32, copy=False))
    generator = torch.Generator(device="cpu").manual_seed(seed)
    completed = 0
    stale_checks = 0
    for step in range(1, steps + 1):
        indices = torch.randint(0, len(train_tensor), (batch_size,), generator=generator)
        batch = train_tensor[indices].to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        loss = torch.mean((model(batch) - batch) ** 2)
        loss.backward()
        optimizer.step()
        completed = step
        if step % 40 == 0 or step == steps:
            validation_nmse = _evaluate(model, validation_tensor, validation_denominator)
            if validation_nmse < best_validation - 1e-5:
                best_validation = validation_nmse
                best_state = {
                    key: value.detach().cpu().clone() for key, value in model.state_dict().items()
                }
                stale_checks = 0
            else:
                stale_checks += 1
            if stale_checks >= 4:
                break
    model.load_state_dict(best_state)
    result = {
        "validation_nmse": _evaluate(model, validation_tensor, validation_denominator),
        "test_nmse": _evaluate(model, test_tensor, test_denominator),
        "steps": completed,
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
    }
    del model, train_tensor, validation_tensor, test_tensor
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return result


def coarse_channels(dimension: int, predicted: int) -> list[int]:
    candidates = {1, dimension, predicted}
    power = 1
    while power < dimension:
        candidates.add(power)
        power *= 2
    for delta in (-2, -1, 1, 2):
        candidates.add(predicted + delta)
    return sorted(value for value in candidates if 1 <= value <= dimension)

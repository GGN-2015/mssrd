"""Controlled U-Net validation for the extended paper experiment."""

from __future__ import annotations

import math
import random
import time
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from mssrd.core import ScaleEstimate

UNET_PROTOCOL_VERSION = 1


@dataclass(frozen=True)
class UNetCandidate:
    """One U-Net skip-path and terminal-bottleneck configuration."""

    role: str
    channels: int
    grid_height: int
    grid_width: int
    latent_scalars: int
    skip_scalars: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class ConvBlock(nn.Module):
    def __init__(self, input_channels: int, output_channels: int) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(input_channels, output_channels, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(output_channels, output_channels, kernel_size=3, padding=1),
            nn.GELU(),
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.layers(values)


class UNetAutoencoder(nn.Module):
    """Three-level U-Net with controllable skip paths and terminal bottleneck."""

    widths = (16, 32, 64, 96)

    def __init__(
        self,
        *,
        image_shape: tuple[int, int],
        bottleneck_channels: int,
        expected_grid: tuple[int, int],
    ) -> None:
        super().__init__()
        internal_channels = max(1, bottleneck_channels)
        self.disable_bottleneck = bottleneck_channels == 0
        self.encoder0 = ConvBlock(1, self.widths[0])
        self.down1 = nn.Conv2d(self.widths[0], self.widths[1], 3, stride=2, padding=1)
        self.encoder1 = ConvBlock(self.widths[1], self.widths[1])
        self.down2 = nn.Conv2d(self.widths[1], self.widths[2], 3, stride=2, padding=1)
        self.encoder2 = ConvBlock(self.widths[2], self.widths[2])
        self.down3 = nn.Conv2d(self.widths[2], self.widths[3], 3, stride=2, padding=1)
        self.bridge = ConvBlock(self.widths[3], self.widths[3])
        self.to_latent = nn.Conv2d(self.widths[3], internal_channels, kernel_size=1, bias=False)
        self.from_latent = nn.Conv2d(internal_channels, self.widths[3], kernel_size=1, bias=False)
        self.decoder2 = ConvBlock(self.widths[3] + self.widths[2], self.widths[2])
        self.decoder1 = ConvBlock(self.widths[2] + self.widths[1], self.widths[1])
        self.decoder0 = ConvBlock(self.widths[1] + self.widths[0], self.widths[0])
        self.output = nn.Conv2d(self.widths[0], 1, kernel_size=1)

        height, width = image_shape
        actual_grid = (math.ceil(height / 8), math.ceil(width / 8))
        if actual_grid != expected_grid:
            raise ValueError(
                f"three-level U-Net produces grid {actual_grid}, expected {expected_grid}"
            )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        skip0 = self.encoder0(values)
        skip1 = self.encoder1(F.gelu(self.down1(skip0)))
        skip2 = self.encoder2(F.gelu(self.down2(skip1)))
        bridge = self.bridge(F.gelu(self.down3(skip2)))
        latent = self.to_latent(bridge)
        if self.disable_bottleneck:
            latent = torch.zeros_like(latent)
        decoded = self.from_latent(latent)
        decoded = F.interpolate(
            decoded, size=skip2.shape[-2:], mode="bilinear", align_corners=False
        )
        decoded = self.decoder2(torch.cat([decoded, skip2], dim=1))
        decoded = F.interpolate(
            decoded, size=skip1.shape[-2:], mode="bilinear", align_corners=False
        )
        decoded = self.decoder1(torch.cat([decoded, skip1], dim=1))
        decoded = F.interpolate(
            decoded, size=skip0.shape[-2:], mode="bilinear", align_corners=False
        )
        decoded = self.decoder0(torch.cat([decoded, skip0], dim=1))
        return self.output(decoded)


def _skip_scalar_count(height: int, width: int) -> int:
    sizes = (
        (height, width, UNetAutoencoder.widths[0]),
        (math.ceil(height / 2), math.ceil(width / 2), UNetAutoencoder.widths[1]),
        (math.ceil(height / 4), math.ceil(width / 4), UNetAutoencoder.widths[2]),
    )
    return sum(rows * columns * channels for rows, columns, channels in sizes)


def unet_candidates(
    prediction: ScaleEstimate, image_shape: tuple[int, int]
) -> tuple[UNetCandidate, ...]:
    """Build a U-Net skip ablation around the MS-SRD scalar prediction."""

    sites = prediction.grid_height * prediction.grid_width
    skip_scalars = _skip_scalar_count(*image_shape)
    candidates: list[UNetCandidate] = []
    for channels, role in (
        (0, "full_skip_zero_bottleneck"),
        (1, "full_skip_one_channel"),
        (prediction.channels, "full_skip_prediction"),
    ):
        candidates.append(
            UNetCandidate(
                role=role,
                channels=channels,
                grid_height=prediction.grid_height,
                grid_width=prediction.grid_width,
                latent_scalars=sites * channels,
                skip_scalars=skip_scalars,
            )
        )
    return tuple(candidates)


def build_unet_model(candidate: UNetCandidate, *, image_shape: tuple[int, int]) -> UNetAutoencoder:
    return UNetAutoencoder(
        image_shape=image_shape,
        bottleneck_channels=candidate.channels,
        expected_grid=(candidate.grid_height, candidate.grid_width),
    )


def _evaluate(
    model: nn.Module,
    values: torch.Tensor,
    denominator: float,
    *,
    batch_size: int = 512,
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


def train_unet_candidate(
    train_images: np.ndarray,
    test_images: np.ndarray,
    candidate: UNetCandidate,
    *,
    device: torch.device,
    seed: int,
    steps: int,
    batch_size: int,
) -> dict[str, Any]:
    """Train one controlled U-Net candidate on centered grayscale images."""

    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    image_shape = (int(train_images.shape[1]), int(train_images.shape[2]))
    model = build_unet_model(candidate, image_shape=image_shape).to(device)
    train_tensor = torch.from_numpy(train_images[:, np.newaxis].astype(np.float32, copy=False))
    test_tensor = torch.from_numpy(test_images[:, np.newaxis].astype(np.float32, copy=False)).to(
        device
    )
    denominator = float(np.mean(test_images.astype(np.float64) ** 2))
    initial_nmse = _evaluate(model, test_tensor, denominator)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-6)
    generator = torch.Generator(device="cpu").manual_seed(seed)
    best_nmse = initial_nmse
    best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    check_interval = max(20, steps // 4)
    stale_checks = 0
    completed = 0
    started = time.perf_counter()
    model.train()
    for step in range(1, steps + 1):
        indices = torch.randint(0, len(train_tensor), (batch_size,), generator=generator)
        batch = train_tensor[indices].to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        loss = torch.mean((model(batch) - batch) ** 2)
        loss.backward()
        optimizer.step()
        completed = step
        if step % check_interval == 0 or step == steps:
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
            model.train()
    elapsed_seconds = time.perf_counter() - started
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
        "elapsed_seconds": elapsed_seconds,
    }

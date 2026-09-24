"""Structural U-Net capacity accounting for MS-SRD predictions."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from mssrd.core import MSSRDResult


@dataclass(frozen=True)
class UNetCapacityAudit:
    """Compare a predicted terminal bottleneck with U-Net skip activations."""

    predicted_bottleneck_shape: tuple[int, int, int]
    predicted_bottleneck_scalars: int
    skip_shapes: tuple[tuple[int, ...], ...]
    skip_path_scalars: int
    total_transmitted_scalars: int
    skip_to_bottleneck_ratio: float
    terminal_is_global_information_bottleneck: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def audit_unet_capacity(
    result: MSSRDResult,
    skip_shapes: Sequence[Sequence[int]],
) -> UNetCapacityAudit:
    """Audit whether U-Net skip paths bypass the MS-SRD terminal bottleneck.

    ``skip_shapes`` contains the non-batch activation shape of every tensor
    transmitted from encoder to decoder. Scalar counts describe structural
    traffic, not statistical independence, entropy, or compressed bit rate.
    """

    normalized: list[tuple[int, ...]] = []
    skip_scalars = 0
    for index, shape in enumerate(skip_shapes):
        resolved = tuple(int(value) for value in shape)
        if not resolved or any(value < 1 for value in resolved):
            raise ValueError(f"skip shape {index} must contain positive dimensions")
        normalized.append(resolved)
        skip_scalars += int(np.prod(resolved))
    bottleneck = result.prediction
    bottleneck_scalars = bottleneck.latent_scalars
    return UNetCapacityAudit(
        predicted_bottleneck_shape=bottleneck.tensor_shape,
        predicted_bottleneck_scalars=bottleneck_scalars,
        skip_shapes=tuple(normalized),
        skip_path_scalars=skip_scalars,
        total_transmitted_scalars=bottleneck_scalars + skip_scalars,
        skip_to_bottleneck_ratio=skip_scalars / bottleneck_scalars,
        terminal_is_global_information_bottleneck=skip_scalars == 0,
    )

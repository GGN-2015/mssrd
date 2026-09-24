"""Resolution and aspect-ratio stress test for the paper."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt

from mssrd.core import MSSRD
from mssrd.paper.datasets import load_oxford_pets_geometry

GEOMETRIES = ((48, 64), (64, 64), (64, 96))
GEOMETRY_SCALES = (4, 8, 12, 16)


def _write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run_geometry_stress(
    *,
    data_dir: str | Path,
    output_dir: str | Path,
    seed: int,
    max_train: int = 3680,
    download: bool = True,
) -> list[dict[str, Any]]:
    """Measure the activation/parameter frontier under controlled image geometries."""

    output = Path(output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    for image_shape in GEOMETRIES:
        images = load_oxford_pets_geometry(
            data_dir=data_dir,
            image_shape=image_shape,
            max_train=max_train,
            seed=seed,
            download=download,
        )
        result = MSSRD(
            target_nmse=0.05,
            scales=GEOMETRY_SCALES,
            color_mode="grayscale",
            seed=seed,
            batch_size=128,
        ).fit(images)
        frontier_scales = {estimate.scale for estimate in result.pareto_frontier}
        for estimate in result.scales:
            rows.append(
                {
                    "height": image_shape[0],
                    "width": image_shape[1],
                    "q": estimate.scale,
                    "grid_height": estimate.grid_height,
                    "grid_width": estimate.grid_width,
                    "channels": estimate.channels,
                    "latent_scalars": estimate.latent_scalars,
                    "linear_parameter_count": estimate.linear_parameter_count,
                    "linear_nmse": estimate.linear_nmse,
                    "pareto": estimate.scale in frontier_scales,
                    "minimum_latent": estimate.scale == result.prediction.scale,
                }
            )

    _write_rows(output / "geometry_stress.csv", rows)
    (output / "geometry_stress.json").write_text(
        json.dumps(rows, indent=2) + "\n", encoding="utf-8"
    )

    figure, axes = plt.subplots(1, len(GEOMETRIES), figsize=(7.5, 2.8), constrained_layout=True)
    for axis, (height, width) in zip(axes, GEOMETRIES, strict=True):
        subset = [row for row in rows if row["height"] == height and row["width"] == width]
        axis.plot(
            [row["linear_parameter_count"] for row in subset],
            [row["latent_scalars"] for row in subset],
            color="#777777",
            linewidth=0.8,
            zorder=1,
        )
        for row in subset:
            color = "#0072b2" if row["pareto"] else "#aaaaaa"
            axis.scatter(
                row["linear_parameter_count"],
                row["latent_scalars"],
                color=color,
                s=28,
                zorder=2,
            )
            axis.annotate(
                f"q={row['q']}",
                (row["linear_parameter_count"], row["latent_scalars"]),
                xytext=(3, 3),
                textcoords="offset points",
                fontsize=7,
            )
        axis.set(
            xscale="log",
            yscale="log",
            xlabel="Shared linear parameters",
            title=f"{height} x {width}",
        )
        axis.grid(alpha=0.2, which="both")
    axes[0].set_ylabel("Latent scalars")
    figure.savefig(output / "geometry_stress.png", dpi=220)
    figure.savefig(output / "geometry_stress.pdf")
    plt.close(figure)
    return rows

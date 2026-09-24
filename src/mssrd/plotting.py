"""Optional plotting helpers."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from mssrd.core import MSSRDResult


def plot_spectra(result: MSSRDResult, path: str | Path) -> None:
    """Plot normalized covariance spectra for every fitted scale."""

    try:
        import matplotlib.pyplot as plt
    except ImportError as error:  # pragma: no cover - environment dependent
        raise RuntimeError(
            "matplotlib is required but is not available in this installation"
        ) from error
    figure, axis = plt.subplots(figsize=(7.0, 4.2), constrained_layout=True)
    for estimate in result.scales:
        values = estimate.eigenvalues
        normalized = values / max(float(values.sum()), np.finfo(float).eps)
        axis.plot(np.arange(1, len(values) + 1), normalized, label=f"q={estimate.scale}")
        axis.axvline(estimate.channels, color=axis.lines[-1].get_color(), alpha=0.25, linewidth=0.8)
    axis.set(
        yscale="log",
        xlabel="Patch eigenmode",
        ylabel="Variance fraction",
        title="MS-SRD multiscale patch spectra",
    )
    axis.grid(alpha=0.2)
    axis.legend(frameon=False)
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(destination, dpi=180)
    plt.close(figure)

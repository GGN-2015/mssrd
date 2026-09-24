"""Training-free multiscale spectral bottleneck prediction."""

from mssrd.core import (
    MSSRD,
    MSSRDResult,
    ScaleEstimate,
    predict_bottleneck,
)

__all__ = ["MSSRD", "MSSRDResult", "ScaleEstimate", "predict_bottleneck"]
__version__ = "0.1.0"

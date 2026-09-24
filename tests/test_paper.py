from __future__ import annotations

import json
from importlib.resources import files

import numpy as np

from mssrd.paper.reproduce import _metrics, _paper_grayscale, _paper_scales


def test_paper_scales_match_protocol() -> None:
    assert _paper_scales(28, 28) == (2, 4, 7)
    assert _paper_scales(32, 32) == (2, 4, 8)
    assert _paper_scales(8, 8) == (2, 4, 8)


def test_paper_grayscale_and_optdigits_scale() -> None:
    rgb = np.zeros((2, 4, 4, 3), dtype=np.uint8)
    rgb[..., 0] = 255
    grayscale = _paper_grayscale(rgb, 255.0)
    np.testing.assert_allclose(grayscale, 0.299, rtol=1e-6)
    digits = np.full((2, 8, 8), 16, dtype=np.uint8)
    np.testing.assert_allclose(_paper_grayscale(digits, 16.0), 1.0)


def test_reference_manifest_contains_twelve_datasets() -> None:
    reference = json.loads(
        files("mssrd.paper").joinpath("reference_results.json").read_text(encoding="utf-8")
    )
    assert len(reference["datasets"]) == 12
    assert reference["datasets"]["cifar10"]["prediction"] == {
        "q": 8,
        "channels": 13,
        "latent_dimension": 208,
    }


def test_single_dataset_metrics_omit_undefined_correlation() -> None:
    metrics = _metrics([{"predicted_latent_dimension": 28, "empirical_latent_dimension": 30}], [])
    assert metrics["dataset_count"] == 1
    assert metrics["log_dimension_pearson"] is None

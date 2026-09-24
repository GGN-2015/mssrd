"""Input and output helpers used by the MS-SRD command line interface."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

IMAGE_EXTENSIONS = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}


def _choose_npz_key(archive: np.lib.npyio.NpzFile, requested: str | None) -> str:
    if requested is not None:
        if requested not in archive.files:
            raise KeyError(f"array key {requested!r} not found; available keys: {archive.files}")
        return requested
    for candidate in ("images", "train_images", "x", "X", "arr_0"):
        if candidate in archive.files:
            return candidate
    if len(archive.files) == 1:
        return archive.files[0]
    raise ValueError(f"NPZ contains multiple arrays; choose one with --array-key: {archive.files}")


def _sample_paths(paths: list[Path], max_images: int | None, seed: int) -> list[Path]:
    if max_images is None or len(paths) <= max_images:
        return paths
    rng = np.random.default_rng(seed)
    indices = np.sort(rng.choice(len(paths), size=max_images, replace=False))
    return [paths[int(index)] for index in indices]


def _load_directory(path: Path, max_images: int | None, seed: int) -> np.ndarray:
    paths = sorted(
        item
        for item in path.rglob("*")
        if item.is_file() and item.suffix.lower() in IMAGE_EXTENSIONS
    )
    paths = _sample_paths(paths, max_images, seed)
    if not paths:
        raise ValueError(f"no supported image files found below {path}")
    with Image.open(paths[0]) as first:
        target_mode = "L" if first.mode in {"1", "L", "I", "I;16", "F"} else "RGB"
    arrays: list[np.ndarray] = []
    expected_shape: tuple[int, ...] | None = None
    for image_path in paths:
        with Image.open(image_path) as source:
            array = np.asarray(source.convert(target_mode))
        if expected_shape is None:
            expected_shape = array.shape
        elif array.shape != expected_shape:
            raise ValueError(
                f"all images must have one shape; {image_path} has {array.shape}, "
                f"expected {expected_shape}"
            )
        arrays.append(array)
    return np.stack(arrays)


def load_images(
    path: str | Path,
    *,
    array_key: str | None = None,
    max_images: int | None = None,
    seed: int = 0,
) -> np.ndarray:
    """Load an image dataset from a directory, NPY, NPZ, or one image file."""

    source = Path(path).expanduser().resolve()
    if source.is_dir():
        return _load_directory(source, max_images, seed)
    if not source.exists():
        raise FileNotFoundError(source)
    suffix = source.suffix.lower()
    if suffix == ".npy":
        images = np.load(source, mmap_mode="r")
    elif suffix == ".npz":
        with np.load(source) as archive:
            images = np.asarray(archive[_choose_npz_key(archive, array_key)])
    elif suffix in IMAGE_EXTENSIONS:
        with Image.open(source) as image:
            images = np.asarray(image.convert("RGB"))[np.newaxis, ...]
    else:
        raise ValueError("input must be an image directory, .npy, .npz, or a supported image file")
    if max_images is not None and len(images) > max_images:
        rng = np.random.default_rng(seed)
        indices = np.sort(rng.choice(len(images), size=max_images, replace=False))
        images = images[indices]
    return np.asarray(images)


def write_json(path: str | Path, payload: dict[str, Any]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

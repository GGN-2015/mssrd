from __future__ import annotations

import json

import numpy as np
from PIL import Image

from mssrd.cli import main
from mssrd.io import load_images


def test_load_npy_and_npz(tmp_path) -> None:
    images = np.arange(20 * 8 * 8, dtype=np.uint16).reshape(20, 8, 8)
    np.save(tmp_path / "images.npy", images)
    np.savez(tmp_path / "images.npz", train_images=images, labels=np.arange(20))
    assert load_images(tmp_path / "images.npy").shape == images.shape
    loaded = load_images(tmp_path / "images.npz", array_key="train_images")
    np.testing.assert_array_equal(loaded, images)


def test_load_image_directory(tmp_path) -> None:
    directory = tmp_path / "images"
    directory.mkdir()
    for index in range(5):
        array = np.full((8, 8), index * 20, dtype=np.uint8)
        Image.fromarray(array).save(directory / f"{index}.png")
    loaded = load_images(directory, max_images=3, seed=12)
    assert loaded.shape == (3, 8, 8)


def test_predict_cli_writes_json(tmp_path, capsys) -> None:
    rng = np.random.default_rng(1)
    images = rng.normal(size=(30, 8, 8)).astype(np.float32)
    input_path = tmp_path / "images.npy"
    output_path = tmp_path / "result.json"
    np.save(input_path, images)
    exit_code = main(
        [
            "predict",
            str(input_path),
            "--scales",
            "2,4",
            "--seed",
            "123",
            "--output",
            str(output_path),
        ]
    )
    assert exit_code == 0
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["seed"] == 123
    assert payload["prediction"]["scale"] in {2, 4}
    assert "MS-SRD prediction" in capsys.readouterr().out

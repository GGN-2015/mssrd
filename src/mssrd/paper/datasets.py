"""Dataset acquisition and loading for the eleven paper benchmarks."""

from __future__ import annotations

import gzip
import pickle
import struct
import tarfile
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import numpy as np

PAPER_DATASETS = (
    ("mnist", "MNIST", "handwriting"),
    ("kmnist", "KMNIST", "handwriting"),
    ("fashion_mnist", "Fashion-MNIST", "clothing"),
    ("cifar10", "CIFAR-10", "natural"),
    ("cifar100", "CIFAR-100", "natural"),
    ("chestmnist", "ChestMNIST", "medical"),
    ("pneumoniamnist", "PneumoniaMNIST", "medical"),
    ("breastmnist", "BreastMNIST", "medical"),
    ("organamnist", "OrganAMNIST", "medical"),
    ("retinamnist", "RetinaMNIST", "medical"),
    ("bloodmnist", "BloodMNIST", "medical"),
)


@dataclass(frozen=True)
class PaperDataset:
    slug: str
    name: str
    category: str
    train: np.ndarray
    test: np.ndarray
    original_shape: tuple[int, ...]
    train_available: int
    test_available: int
    intensity_divisor: float


def _read_idx_images(path: Path) -> np.ndarray:
    with gzip.open(path, "rb") as handle:
        magic, count, rows, columns = struct.unpack(">IIII", handle.read(16))
        if magic != 2051:
            raise ValueError(f"unexpected IDX image magic number in {path}")
        values = np.frombuffer(handle.read(), dtype=np.uint8)
    return values.reshape(count, rows, columns)


def _load_idx_directory(path: Path) -> tuple[np.ndarray, np.ndarray]:
    return (
        _read_idx_images(path / "train-images-idx3-ubyte.gz"),
        _read_idx_images(path / "t10k-images-idx3-ubyte.gz"),
    )


def _load_cifar_archive(path: Path, classes: int) -> tuple[np.ndarray, np.ndarray]:
    train_blocks: list[np.ndarray] = []
    test_blocks: list[np.ndarray] = []
    with tarfile.open(path, "r:gz") as archive:
        members = {member.name: member for member in archive.getmembers()}
        if classes == 10:
            train_names = [f"cifar-10-batches-py/data_batch_{index}" for index in range(1, 6)]
            test_names = ["cifar-10-batches-py/test_batch"]
        else:
            train_names = ["cifar-100-python/train"]
            test_names = ["cifar-100-python/test"]
        for destination, names in ((train_blocks, train_names), (test_blocks, test_names)):
            for name in names:
                extracted = archive.extractfile(members[name])
                if extracted is None:
                    raise ValueError(f"cannot read {name} from {path}")
                payload = pickle.load(extracted, encoding="bytes")  # noqa: S301 - trusted dataset
                destination.append(np.asarray(payload[b"data"], dtype=np.uint8))

    def reshape(blocks: list[np.ndarray]) -> np.ndarray:
        channel_first = np.concatenate(blocks).reshape(-1, 3, 32, 32)
        return np.ascontiguousarray(channel_first.transpose(0, 2, 3, 1))

    return reshape(train_blocks), reshape(test_blocks)


def _download(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".part")
    print(f"Downloading {url} -> {destination}", flush=True)
    urllib.request.urlretrieve(url, temporary)  # noqa: S310 - fixed HTTPS source
    temporary.replace(destination)


def _load_torchvision(slug: str, root: Path, download: bool) -> tuple[np.ndarray, np.ndarray]:
    from torchvision import datasets

    classes = {
        "mnist": datasets.MNIST,
        "kmnist": datasets.KMNIST,
        "fashion_mnist": datasets.FashionMNIST,
        "cifar10": datasets.CIFAR10,
        "cifar100": datasets.CIFAR100,
    }
    dataset_class = classes[slug]
    train_dataset = dataset_class(root=str(root), train=True, download=download)
    test_dataset = dataset_class(root=str(root), train=False, download=download)
    train = np.asarray(train_dataset.data)
    test = np.asarray(test_dataset.data)
    return train, test


def _load_medmnist(slug: str, root: Path, download: bool) -> tuple[np.ndarray, np.ndarray]:
    path = root / f"{slug}.npz"
    if not path.exists():
        if not download:
            raise FileNotFoundError(path)
        import medmnist

        info = medmnist.INFO[slug]
        dataset_class = getattr(medmnist, info["python_class"])
        dataset_class(root=str(root), split="train", download=True, size=28)
    with np.load(path) as archive:
        train = np.concatenate([archive["train_images"], archive["val_images"]])
        test = np.asarray(archive["test_images"])
    return train, test


def _deterministic_subset(images: np.ndarray, maximum: int, seed: int) -> np.ndarray:
    if len(images) <= maximum:
        return np.asarray(images)
    rng = np.random.default_rng(seed)
    indices = np.sort(rng.choice(len(images), size=maximum, replace=False))
    return np.asarray(images)[indices]


def load_paper_dataset(
    slug: str,
    *,
    data_dir: str | Path,
    max_train: int,
    max_test: int,
    seed: int,
    download: bool,
) -> PaperDataset:
    """Load one paper dataset using cached raw files or official downloaders."""

    metadata = {item[0]: item[1:] for item in PAPER_DATASETS}
    if slug not in metadata:
        raise ValueError(f"unknown paper dataset {slug!r}")
    name, category = metadata[slug]
    root = Path(data_dir).expanduser().resolve()
    if slug in {"mnist", "kmnist", "fashion_mnist"}:
        custom = root / slug
        if (custom / "train-images-idx3-ubyte.gz").exists():
            raw_train, raw_test = _load_idx_directory(custom)
        else:
            raw_train, raw_test = _load_torchvision(slug, root / "torchvision", download)
    elif slug in {"cifar10", "cifar100"}:
        classes = 10 if slug == "cifar10" else 100
        filename = "cifar-10-python.tar.gz" if classes == 10 else "cifar-100-python.tar.gz"
        archive = root / slug / filename
        if archive.exists():
            raw_train, raw_test = _load_cifar_archive(archive, classes)
        else:
            raw_train, raw_test = _load_torchvision(slug, root / "torchvision", download)
    else:
        raw_train, raw_test = _load_medmnist(slug, root / "medmnist", download)
    original_shape = tuple(int(value) for value in raw_train.shape[1:])
    train_available, test_available = len(raw_train), len(raw_test)
    train = _deterministic_subset(raw_train, max_train, seed)
    test = _deterministic_subset(raw_test, max_test, seed + 1)
    return PaperDataset(
        slug=slug,
        name=name,
        category=category,
        train=train,
        test=test,
        original_shape=original_shape,
        train_available=train_available,
        test_available=test_available,
        intensity_divisor=255.0,
    )

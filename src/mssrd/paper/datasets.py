"""Dataset acquisition and loading for the paper benchmarks."""

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
    ("oxford_pets", "Oxford-IIIT Pet", "natural"),
    ("eurosat", "EuroSAT", "remote sensing"),
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


def _load_resized_paths(
    paths: list[Path],
    maximum: int,
    seed: int,
    *,
    dataset_name: str,
    cache_path: Path,
    image_shape: tuple[int, int] = (64, 64),
) -> np.ndarray:
    from PIL import Image, ImageOps

    selected = _deterministic_subset(np.asarray(paths, dtype=object), maximum, seed)
    if cache_path.exists():
        cached = np.load(cache_path, allow_pickle=False)
        expected_shape = (len(selected), *image_shape, 3)
        if cached.shape == expected_shape and cached.dtype == np.uint8:
            print(f"[{dataset_name}] loaded resized cache {cache_path}", flush=True)
            return np.asarray(cached)
    print(f"[{dataset_name}] loading and resizing {len(selected):,} images", flush=True)
    images = np.empty((len(selected), *image_shape, 3), dtype=np.uint8)
    height, width = image_shape
    for index, path in enumerate(selected):
        with Image.open(Path(path)) as source:
            resized = ImageOps.fit(
                source.convert("RGB"),
                (width, height),
                method=Image.Resampling.LANCZOS,
                centering=(0.5, 0.5),
            )
            images[index] = np.asarray(resized, dtype=np.uint8)
        if (index + 1) % 1000 == 0 or index + 1 == len(selected):
            print(
                f"[{dataset_name}] prepared {index + 1:,}/{len(selected):,} images",
                flush=True,
            )
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = cache_path.with_suffix(cache_path.suffix + ".part")
    with temporary.open("wb") as handle:
        np.save(handle, images, allow_pickle=False)
    temporary.replace(cache_path)
    print(f"[{dataset_name}] wrote resized cache {cache_path}", flush=True)
    return images


def load_oxford_pets_geometry(
    *,
    data_dir: str | Path,
    image_shape: tuple[int, int],
    max_train: int,
    seed: int,
    download: bool,
) -> np.ndarray:
    """Load a deterministic Oxford-IIIT Pet training subset at a fixed geometry."""

    from torchvision import datasets

    if len(image_shape) != 2 or min(image_shape) < 1:
        raise ValueError("image_shape must contain two positive integers")
    root = Path(data_dir).expanduser().resolve() / "torchvision"
    dataset = datasets.OxfordIIITPet(root=str(root), split="trainval", download=download)
    height, width = image_shape
    return _load_resized_paths(
        list(dataset._images),
        max_train,
        seed,
        dataset_name=f"oxford_pets/{height}x{width}",
        cache_path=root
        / "mssrd_cache"
        / f"oxford_pets_train_{height}x{width}_n{max_train}_seed{seed}.npy",
        image_shape=image_shape,
    )


def _load_oxford_pets(
    root: Path,
    *,
    max_train: int,
    max_test: int,
    seed: int,
    download: bool,
) -> tuple[np.ndarray, np.ndarray, int, int]:
    from torchvision import datasets

    train_dataset = datasets.OxfordIIITPet(root=str(root), split="trainval", download=download)
    test_dataset = datasets.OxfordIIITPet(root=str(root), split="test", download=download)
    train_paths = list(train_dataset._images)
    test_paths = list(test_dataset._images)
    return (
        _load_resized_paths(
            train_paths,
            max_train,
            seed,
            dataset_name="oxford_pets/train",
            cache_path=root / "mssrd_cache" / f"oxford_pets_train_64_n{max_train}_seed{seed}.npy",
        ),
        _load_resized_paths(
            test_paths,
            max_test,
            seed + 1,
            dataset_name="oxford_pets/test",
            cache_path=root / "mssrd_cache" / f"oxford_pets_test_64_n{max_test}_seed{seed + 1}.npy",
        ),
        len(train_paths),
        len(test_paths),
    )


def _load_eurosat(
    root: Path,
    *,
    max_train: int,
    max_test: int,
    seed: int,
    download: bool,
) -> tuple[np.ndarray, np.ndarray, int, int]:
    from torchvision import datasets

    dataset = datasets.EuroSAT(root=str(root), download=download)
    paths = np.asarray([Path(path) for path, _ in dataset.samples], dtype=object)
    rng = np.random.default_rng(seed)
    permutation = rng.permutation(len(paths))
    split = int(0.8 * len(paths))
    train_paths = paths[permutation[:split]].tolist()
    test_paths = paths[permutation[split:]].tolist()
    return (
        _load_resized_paths(
            train_paths,
            max_train,
            seed + 11,
            dataset_name="eurosat/train",
            cache_path=root / "mssrd_cache" / f"eurosat_train_64_n{max_train}_seed{seed + 11}.npy",
        ),
        _load_resized_paths(
            test_paths,
            max_test,
            seed + 12,
            dataset_name="eurosat/test",
            cache_path=root / "mssrd_cache" / f"eurosat_test_64_n{max_test}_seed{seed + 12}.npy",
        ),
        len(train_paths),
        len(test_paths),
    )


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
    if slug == "oxford_pets":
        train, test, train_available, test_available = _load_oxford_pets(
            root / "torchvision",
            max_train=max_train,
            max_test=max_test,
            seed=seed,
            download=download,
        )
        original_shape = (64, 64, 3)
    elif slug == "eurosat":
        train, test, train_available, test_available = _load_eurosat(
            root / "torchvision",
            max_train=max_train,
            max_test=max_test,
            seed=seed,
            download=download,
        )
        original_shape = (64, 64, 3)
    elif slug in {"mnist", "kmnist", "fashion_mnist"}:
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
    if slug not in {"oxford_pets", "eurosat"}:
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

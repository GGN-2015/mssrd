"""Command line interface for MS-SRD."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from mssrd import __version__
from mssrd.core import MSSRD
from mssrd.io import load_images, write_json


def _parse_scales(value: str) -> tuple[int, ...] | None:
    if value.lower() == "auto":
        return None
    try:
        scales = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "scales must be 'auto' or comma-separated integers"
        ) from error
    if not scales:
        raise argparse.ArgumentTypeError("at least one scale is required")
    return scales


def _parse_nmse_targets(value: str) -> tuple[float, ...]:
    try:
        targets = tuple(float(item.strip()) for item in value.split(",") if item.strip())
    except ValueError as error:
        raise argparse.ArgumentTypeError("NMSE targets must be comma-separated numbers") from error
    if not targets or any(not 0.0 < target < 1.0 for target in targets):
        raise argparse.ArgumentTypeError("NMSE targets must lie in (0, 1)")
    return tuple(sorted(set(targets), reverse=True))


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="mssrd",
        description="Training-free multiscale spectral bottleneck prediction.",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    subparsers = parser.add_subparsers(dest="command", required=True)

    predict = subparsers.add_parser("predict", help="predict a bottleneck for an image dataset")
    predict.add_argument("input", type=Path, help="image directory, NPY, or NPZ dataset")
    predict.add_argument("--array-key", help="array key when reading NPZ")
    distortion = predict.add_mutually_exclusive_group()
    distortion.add_argument("--target-nmse", type=float, default=None)
    distortion.add_argument("--retained-variance", type=float, default=None)
    predict.add_argument("--scales", type=_parse_scales, default=None, metavar="auto|2,4,8")
    predict.add_argument("--max-patch-size", type=int, default=8)
    predict.add_argument("--color-mode", choices=("grayscale", "channels"), default="grayscale")
    predict.add_argument("--channel-axis", type=int)
    predict.add_argument("--max-images", type=int)
    predict.add_argument("--seed", type=int, default=0)
    predict.add_argument("--batch-size", type=int, default=512)
    predict.add_argument("--compute-global", action="store_true")
    predict.add_argument("--include-eigenvalues", action="store_true")
    predict.add_argument("--output", type=Path, help="write the complete result as JSON")
    predict.add_argument("--plot", type=Path, help="write a multiscale spectrum plot")

    reproduce = subparsers.add_parser(
        "reproduce-paper", help="download data and reproduce the paper experiment"
    )
    reproduce.add_argument("--data-dir", type=Path, default=Path("data"))
    reproduce.add_argument("--output-dir", type=Path, default=Path("paper-results"))
    reproduce.add_argument("--seed", type=int, default=20260924)
    reproduce.add_argument("--datasets", default="all", help="all or comma-separated dataset slugs")
    reproduce.add_argument("--max-train", type=int, default=20000)
    reproduce.add_argument("--max-test", type=int, default=5000)
    reproduce.add_argument("--max-patch-samples", type=int, default=250000)
    reproduce.add_argument("--bootstrap-reps", type=int, default=20)
    reproduce.add_argument("--steps", type=int, default=160)
    reproduce.add_argument("--batch-size", type=int, default=8192)
    reproduce.add_argument("--unet-steps", type=int, default=800)
    reproduce.add_argument("--unet-batch-size", type=int, default=256)
    reproduce.add_argument(
        "--unet-targets",
        type=_parse_nmse_targets,
        default=(0.10, 0.05, 0.02, 0.01),
        help="comma-separated NMSE targets for the true-bottleneck U-Net sweep",
    )
    reproduce.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    reproduce.add_argument("--no-download", action="store_true")
    reproduce.add_argument("--spectral-only", action="store_true")
    reproduce.add_argument("--unet-only", action="store_true")
    reproduce.add_argument("--skip-repeats", action="store_true")
    reproduce.add_argument("--skip-unet-validation", action="store_true")
    reproduce.add_argument("--skip-full-skip-validation", action="store_true")
    reproduce.add_argument("--skip-true-bottleneck-validation", action="store_true")
    reproduce.add_argument(
        "--quick",
        action="store_true",
        help="small smoke run: 512 train, 256 held out, 2 bootstraps, 40 steps",
    )
    return parser


def _predict(args: argparse.Namespace) -> int:
    images = load_images(
        args.input,
        array_key=args.array_key,
        max_images=args.max_images,
        seed=args.seed,
    )
    estimator = MSSRD(
        target_nmse=args.target_nmse,
        retained_variance=args.retained_variance,
        scales=args.scales,
        max_patch_size=args.max_patch_size,
        color_mode=args.color_mode,
        channel_axis=args.channel_axis,
        seed=args.seed,
        batch_size=args.batch_size,
        compute_global=args.compute_global,
    )
    result = estimator.fit(images)
    payload = result.to_dict(include_eigenvalues=args.include_eigenvalues)
    prediction = result.prediction
    print(
        "MS-SRD prediction: "
        f"{prediction.grid_height}x{prediction.grid_width}x{prediction.channels} "
        f"({prediction.latent_scalars} latent scalars, q={prediction.scale})"
    )
    print(f"Estimated linear NMSE: {prediction.linear_nmse:.6f}")
    print(
        f"Nonconstant input features: {result.nonconstant_input_features}; "
        f"constant features ignored: {result.constant_input_features}"
    )
    if args.output:
        write_json(args.output, payload)
        print(f"Wrote {args.output}")
    else:
        print(json.dumps(payload, indent=2))
    if args.plot:
        from mssrd.plotting import plot_spectra

        plot_spectra(result, args.plot)
        print(f"Wrote {args.plot}")
    return 0


def _reproduce(args: argparse.Namespace) -> int:
    try:
        from mssrd.paper.reproduce import reproduce_paper
    except ImportError as error:
        raise RuntimeError(
            "paper reproduction dependencies are missing; run `uv sync` to restore the "
            "project environment"
        ) from error
    if args.quick:
        args.max_train = min(args.max_train, 512)
        args.max_test = min(args.max_test, 256)
        args.max_patch_samples = min(args.max_patch_samples, 8192)
        args.bootstrap_reps = min(args.bootstrap_reps, 2)
        args.steps = min(args.steps, 40)
        args.unet_steps = min(args.unet_steps, 80)
        args.unet_batch_size = min(args.unet_batch_size, 128)
        args.unet_targets = tuple(target for target in args.unet_targets if target >= 0.05)
        args.skip_repeats = True
    selected = (
        None
        if args.datasets == "all"
        else tuple(item.strip() for item in args.datasets.split(",") if item.strip())
    )
    reproduce_paper(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        seed=args.seed,
        datasets=selected,
        max_train=args.max_train,
        max_test=args.max_test,
        max_patch_samples=args.max_patch_samples,
        bootstrap_reps=args.bootstrap_reps,
        steps=args.steps,
        batch_size=args.batch_size,
        unet_steps=args.unet_steps,
        unet_batch_size=args.unet_batch_size,
        unet_targets=args.unet_targets,
        device=args.device,
        download=not args.no_download,
        spectral_only=args.spectral_only,
        unet_only=args.unet_only,
        run_repeats=not args.skip_repeats,
        run_unet_validation=not args.skip_unet_validation,
        run_full_skip_validation=not args.skip_full_skip_validation,
        run_true_bottleneck_validation=not args.skip_true_bottleneck_validation,
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "predict":
            return _predict(args)
        if args.command == "reproduce-paper":
            return _reproduce(args)
        parser.error(f"unknown command {args.command}")
    except (FileNotFoundError, KeyError, RuntimeError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    return 2


if __name__ == "__main__":
    raise SystemExit(main())

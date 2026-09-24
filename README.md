# MS-SRD

MS-SRD (Multiscale Spectral Rate-Distortion) predicts a convolutional autoencoder
bottleneck tensor directly from an image dataset, before a neural network is trained.
It estimates covariance spectra of local image patches at several spatial scales,
finds the smallest channel count whose linear reconstruction meets a requested NMSE
budget, and selects the tensor with the fewest latent scalars.

The package provides:

- a Python API for arbitrary fixed-size image datasets;
- a CLI for image directories, NPY files, and NPZ archives;
- the linear forward and inverse mappings associated with the prediction;
- constant-feature exclusion and grayscale or joint color-channel analysis;
- a one-command reproduction of the twelve-dataset paper experiment;
- PCA-initialized nonlinear PyTorch validation, bootstraps, repeat seeds, CSV/JSON
  results, and publication figures;
- a true-bottleneck U-Net sweep, a full-skip control, and a structural capacity audit
  for custom U-Net layouts.

## Scope

MS-SRD takes an NMSE budget as a modeling choice. The paper reports 0.05 as its primary
operating point for continuity with variance-threshold dimension selection and tests
sensitivity at 0.10, 0.05, 0.02, and 0.01. No one threshold is universally correct:
the appropriate budget depends on the reconstruction loss, data scale, and downstream
use. The criterion concerns centered pixel MSE, not perceptual quality or task accuracy.

The exact theorem applies to a shared linear non-overlapping block-convolutional
autoencoder. The paper also tests a nonlinear U-shaped autoencoder in which every
sample-dependent encoder-to-decoder skip is closed, so all image content crosses one
terminal tensor. A conventional full-skip U-Net is a control: its skip paths bypass the
terminal tensor, which therefore is not a global bottleneck. Predictions for perceptual
losses or high-resolution color models remain architecture priors that require
validation.

## Installation

Python 3.12 and [uv](https://docs.astral.sh/uv/) are recommended. The project keeps
PyTorch, TorchVision, MedMNIST, and Matplotlib as required dependencies so every normal
installation can run the full paper reproduction.

```bash
git clone https://github.com/GGN-2015/mssrd.git
cd mssrd
uv python install 3.12
uv sync
```

To install the package into another environment:

```bash
pip install .
```

If downloads require the local proxy described in the experiment environment, set it
before running `uv sync` or the reproduction command.

PowerShell:

```powershell
$env:HTTP_PROXY = "http://127.0.0.1:10808"
$env:HTTPS_PROXY = "http://127.0.0.1:10808"
uv sync
```

Bash:

```bash
export HTTP_PROXY=http://127.0.0.1:10808
export HTTPS_PROXY=http://127.0.0.1:10808
uv sync
```

## CLI: predict a bottleneck

Analyze a directory recursively containing PNG, JPEG, TIFF, BMP, or WebP images:

```bash
uv run mssrd predict ./my-images \
  --seed 42 \
  --target-nmse 0.05 \
  --scales 2,4,8 \
  --output result.json \
  --plot spectra.png
```

Change `--target-nmse` to match the application's distortion requirement. The legacy
spelling `--retained-variance 0.95` is equivalent to `--target-nmse 0.05`.

Analyze a NumPy array with shape `(N,H,W)`, `(N,H,W,C)`, or `(N,C,H,W)`:

```bash
uv run mssrd predict images.npy --scales auto --max-patch-size 8
```

For an NPZ archive, choose the image array when it cannot be inferred:

```bash
uv run mssrd predict dataset.npz --array-key train_images --seed 7
```

RGB data are converted to BT.601 luma by default. To estimate modes that jointly mix
space and color channels:

```bash
uv run mssrd predict rgb.npy --color-mode channels --channel-axis -1
```

With `--scales auto`, MS-SRD uses every common image-dimension divisor from 2 through
`--max-patch-size`. Explicit scales must divide both the image height and width. Use
`--max-images` with a seed for a deterministic subset of a very large dataset.

## Python API

The estimator accepts any array-like object that NumPy can convert to a dense image
batch.

```python
import numpy as np

from mssrd import MSSRD, predict_bottleneck

images = np.load("images.npy")

result = predict_bottleneck(
    images,
    target_nmse=0.05,
    scales=[2, 4, 8],
    seed=42,
)

print(result.prediction.tensor_shape)
print(result.prediction.latent_scalars)
print(result.to_dict())
```

Use the estimator object when the analytic mapping and inverse mapping are needed:

```python
estimator = MSSRD(
    target_nmse=0.05,
    scales=[2, 4, 8],
    color_mode="grayscale",
    seed=42,
)

result = estimator.fit(images)
latent = estimator.transform(images)
linear_reconstruction = estimator.inverse_transform(latent)

print(latent.shape)
print(result.prediction.linear_nmse)
```

MS-SRD subtracts the training per-pixel mean. It does not divide every pixel by its
standard deviation, because per-pixel Z-normalization changes the raw-pixel MSE
objective. Constant pixels become zero-energy covariance directions and are excluded
automatically. A dataset in which every feature is constant raises an error.

Set `color_mode="channels"` to retain every input channel. Set `compute_global=True`
to include a global PCA dimension, but note that the full image covariance can be
expensive for large images.

For a U-Net, audit the capacity that bypasses the terminal bottleneck by supplying the
non-batch shape of every encoder feature tensor sent to the decoder:

```python
from mssrd import audit_unet_capacity

audit = audit_unet_capacity(
    result,
    skip_shapes=[(16, 28, 28), (32, 14, 14), (64, 7, 7)],
)
print(audit.skip_to_bottleneck_ratio)
print(audit.terminal_is_global_information_bottleneck)
```

These are raw activation scalar counts, not estimates of independent information,
entropy, or compressed bit rate. Any nonempty skip list means that the terminal tensor
is not the only information path into the decoder.

## Reproduce the paper

The following command downloads the public datasets when needed and runs the complete
experiment with the manuscript seed:

```bash
uv run mssrd reproduce-paper \
  --seed 20260924 \
  --data-dir data \
  --output-dir paper-results
```

The default run performs all of the following:

1. spectral prediction on twelve datasets;
2. twenty image-level bootstrap repetitions per dataset;
3. PCA-initialized nonlinear bottleneck searches at every paper scale;
4. empirical-boundary, one-channel-below, and predicted runs at two additional seeds;
5. a true-bottleneck U-Net width search at NMSE budgets 0.10, 0.05, 0.02, and 0.01,
   using a training/validation split for deployable selection and a separately labeled,
   retrospective test-set boundary for analysis;
6. full-skip U-Net controls with a zeroed terminal tensor, one terminal channel, and the
   MS-SRD-predicted channel count;
7. comparison against the committed paper reference predictions;
8. JSON, CSV, NumPy spectrum archives, and PDF/PNG figures.

The neural candidate cache is stored below each dataset result directory. Re-running the
same command resumes an interrupted experiment instead of retraining completed
candidates.

The patch experiment uses at most 20,000 training images, 5,000 held-out images,
250,000 training patches per scale, and 160 optimizer steps per candidate. The U-Net
experiment uses 800 image-level optimizer steps per candidate with a default batch size
of 256. CUDA is used automatically when available. Force a device with `--device cpu`
or `--device cuda`.

In `true_bottleneck_boundaries.csv`, `predicted_channels` is the training-free MS-SRD
output, `empirical_channels` is selected on the validation split, and
`test_oracle_channels` is a retrospective test-set crossing reported only for analysis.
Do not use the test oracle as a deployment-time selector.

Useful shorter runs:

```bash
# Recompute all training-free predictions without neural validation.
uv run mssrd reproduce-paper --spectral-only --skip-repeats

# Smoke-test one dataset with small subsets and 40 optimization steps.
uv run mssrd reproduce-paper --datasets optdigits --quick

# Reproduce only selected datasets.
uv run mssrd reproduce-paper --datasets mnist,cifar10,bloodmnist

# Run both U-Net experiments (plus the required spectral predictions).
uv run mssrd reproduce-paper --unet-only --skip-repeats --device cuda

# Run only the true-bottleneck sweep at selected distortion budgets.
uv run mssrd reproduce-paper --unet-only --skip-repeats \
  --skip-full-skip-validation --unet-targets 0.10,0.05,0.02,0.01 --device cuda
```

The default spectral predictions are:

| Dataset | Predicted tensor | Latent scalars |
|---|---:|---:|
| MNIST | 4 x 4 x 22 | 352 |
| KMNIST | 4 x 4 x 21 | 336 |
| UCI Optical Digits | 1 x 1 x 29 | 29 |
| Fashion-MNIST | 4 x 4 x 20 | 320 |
| CIFAR-10 | 4 x 4 x 13 | 208 |
| CIFAR-100 | 4 x 4 x 12 | 192 |
| ChestMNIST | 4 x 4 x 7 | 112 |
| PneumoniaMNIST | 4 x 4 x 10 | 160 |
| BreastMNIST | 4 x 4 x 9 | 144 |
| OrganAMNIST | 4 x 4 x 30 | 480 |
| RetinaMNIST | 4 x 4 x 6 | 96 |
| BloodMNIST | 4 x 4 x 13 | 208 |

`paper-results/reference_check.json` states whether a default spectral run exactly
matches these committed predictions. Small floating-point differences in nonlinear GPU
training are expected, but the reported channel boundary should remain stable.

## Reproduction outputs

The output directory contains:

```text
paper-results/
  dataset_summary.csv
  summary.json
  training_runs.csv
  repeat_validation.csv
  unet_runs.csv
  unet_metrics.json
  true_bottleneck_runs.csv
  true_bottleneck_boundaries.csv
  true_bottleneck_metrics.json
  metrics.json
  reference_check.json
  figures/
  <dataset>/
    spectra.npz
    summary.json
    training_cache.json
    training_runs.csv
    repeat_validation.csv
    unet_cache.json
    unet_runs.csv
    true_bottleneck_cache.json
    true_bottleneck_runs.csv
    true_bottleneck_boundaries.csv
```

The six MedMNIST subsets are downloaded through the official MedMNIST API. MNIST,
KMNIST, Fashion-MNIST, CIFAR-10, and CIFAR-100 use TorchVision downloaders. Optical
Digits is downloaded from the UCI Machine Learning Repository. Dataset labels are not
used by MS-SRD or by the reconstruction experiment.

## Development and tests

```bash
uv sync
uv run pytest
uv run ruff check .
uv build
```

The unit tests cover known-rank synthetic images, constant-feature handling, color
layouts, forward/inverse mappings, NPY/NPZ and directory input, CLI JSON output, and the
paper reference manifest. They also check U-Net candidate construction, true zeroing of
the terminal path, and skip-capacity accounting. The real-data spectral reproduction is
also exercised before a release by running the twelve datasets against
`reference_check.json`.

## Method summary

For a square patch scale `q`, let the pooled centered patch covariance have eigenvalues
`lambda_1 >= ... >= lambda_d`, and let `delta` be the requested NMSE budget. MS-SRD
chooses the smallest channel count `c_q` for which

```text
sum(lambda_(cq+1) ... lambda_d) / sum(lambda_1 ... lambda_d) <= delta.
```

There are `(H/q)(W/q)` spatial positions, so the candidate scalar count is

```text
M_q = (H/q)(W/q)c_q.
```

The prediction is the candidate with the smallest `M_q`. For a shared linear block
encoder and decoder under MSE, the unreconstructed population patch energy is exactly
the eigenvalue tail after `c_q`, making the rule optimal over the supplied scales.

## License and citation

The code is released under the MIT License. Dataset licenses remain with their original
publishers. Citation metadata are provided in `CITATION.cff`.

# Project spec: Restormer (repo)

## Summary

This repository provides **Restormer** inference and training code for multiple image restoration tasks (CVPR 2022).
The primary “product surface” in this repo is a **CLI inference pipeline** (`demo.py`) that runs pretrained weights on user images.

## Goals

- Provide a reliable, reproducible way to run **pretrained Restormer** models on local images.
- Support the six inference tasks exposed by `demo.py`:
  - `Motion_Deblurring`
  - `Single_Image_Defocus_Deblurring`
  - `Deraining`
  - `Real_Denoising`
  - `Gaussian_Gray_Denoising`
  - `Gaussian_Color_Denoising`
- Support both **single-file** and **directory batch** inputs.
- Support **tiled inference** for large images to reduce GPU memory usage.

## Non-goals (for this spec set)

- Redesigning the model architecture (Restormer is treated as given).
- Replacing the upstream training pipelines or datasets with new ones.
- Providing a GUI/web UI (the canonical interface is CLI).
- Guaranteeing deterministic output across all platforms and CUDA versions (best-effort reproducibility only).

## Users & use cases

- **Researchers / practitioners**: run pretrained models on custom photos, evaluate qualitative results, generate restored outputs for downstream tasks.
- **Engineers**: integrate the CLI inference step into batch scripts/pipelines.

## Constraints & assumptions

- Inference requires **task-specific pretrained weights** (`.pth`) in the expected per-task directories.
- Images are read via OpenCV; supported input formats are those discovered by extension in `demo.py`.
- Runtime supports CPU fallback, but GPU is expected for practical throughput.
- The stable CLI entry point is `python demo.py ...` (training entry points vary per task subfolder).

## Repository interfaces (stable)

- **CLI inference**: `demo.py`
  - Inputs: `--task`, `--input_dir`, `--result_dir`, optional `--tile`, `--tile_overlap`
  - Outputs: PNG images written into `<result_dir>/<task>/`
- **Documentation**: `README.md`, `INSTALL.md`, and this `docs/specs/` folder.

## Quality attributes

- **Correctness**: output dimensions match inputs; padding/unpadding correct; tiling covers full image.
- **Usability**: clear error messages for invalid tasks, missing inputs, missing weights.
- **Performance**: use CUDA when available; optional tiling for memory constraints.
- **Maintainability**: requirements and design remain traceable to the CLI behavior.

## Success criteria

- A new user can:
  - install dependencies (per `pyproject.toml` and/or `INSTALL.md`),
  - download weights,
  - run `demo.py` on a single file or directory,
  - obtain restored PNG outputs in the expected folder structure.


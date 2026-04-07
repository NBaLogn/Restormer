# CLI contract: `demo.py`

This document specifies the **expected user-facing behavior** of the inference CLI in `demo.py`.

## Command

Run from repo root:

```bash
python demo.py --task <Task_Name> --input_dir <path> --result_dir <path> [--tile <int>] [--tile_overlap <int>]
```

## Arguments

- `--task` (required)
  - **Allowed values**:
    - `Motion_Deblurring`
    - `Single_Image_Defocus_Deblurring`
    - `Deraining`
    - `Real_Denoising`
    - `Gaussian_Gray_Denoising`
    - `Gaussian_Color_Denoising`
  - **Behavior**:
    - Invalid/missing values cause the CLI to error and exit without running inference.

- `--input_dir` (optional; default `./demo/degraded/`)
  - Either:
    - path to a single image file, or
    - path to a directory containing images.
  - **Supported extensions** (case-sensitive, as implemented): `jpg`, `JPG`, `png`, `PNG`, `jpeg`, `JPEG`, `bmp`, `BMP`.
  - **Behavior**:
    - If `--input_dir` is a single supported file, process only that file.
    - If `--input_dir` is a directory, process all matching files in natural sort order.
    - If no files found, raise an exception and halt execution.

- `--result_dir` (optional; default `./demo/restored/`)
  - Root output directory.
  - Outputs are written to: `<result_dir>/<task>/`.

- `--tile` (optional; default `None`)
  - If omitted/`None`: run full-image inference in one forward pass.
  - If provided: run tiled inference.
  - **Constraints**:
    - Effective tile size is `min(tile, image_height, image_width)`.
    - Tile size must be a multiple of 8; otherwise raise an assertion error.

- `--tile_overlap` (optional; default `32`)
  - Number of pixels overlapped between adjacent tiles.

## Output conventions

- **Directory**: `<result_dir>/<task>/` (created if missing).
- **Format**: PNG (`.png`).
- **Filename**: input base name preserved; extension replaced with `.png`.
- **Channels**:
  - For `Gaussian_Gray_Denoising`: save single-channel grayscale PNG.
  - For all other tasks: save 3-channel RGB PNG.

## Runtime behavior

- **Device selection**:
  - Use CUDA if available; else CPU.
  - When CUDA is available, clear CUDA caches per image (`ipc_collect`, `empty_cache`) as implemented.

- **Progress reporting**:
  - Print task name and weights path before processing begins.
  - Show a progress bar for multiple images (`tqdm`).
  - Print the final output directory after all images finish.


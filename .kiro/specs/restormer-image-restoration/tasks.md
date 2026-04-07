# Implementation Plan: Restormer Image Restoration Pipeline

## Overview

Implement `demo.py` as a CLI-driven inference pipeline for the Restormer model. Tasks follow the linear execution flow: argument parsing → model configuration → image discovery → preprocessing → inference → output saving. Tests are co-located with each component.

## Tasks

- [-] 1. Set up test infrastructure and utility functions
  - Create `tests/test_demo.py` with imports for `pytest`, `hypothesis`, and `torch`
  - Implement `load_img`, `load_gray_img`, `save_img`, `save_gray_img` in `demo.py`
  - _Requirements: 3.1, 3.2, 6.5, 6.6_

  - [-] 1.1 Write unit tests for image load/save functions
    - Test `load_img` returns shape `(H, W, 3)` uint8 array
    - Test `load_gray_img` returns shape `(H, W, 1)` uint8 array
    - _Requirements: 3.1, 3.2_

- [~] 2. Implement `get_weights_and_parameters`
  - Write the pure function mapping each of the 6 tasks to weights path and parameter overrides
  - _Requirements: 1.2, 9.1, 9.2, 9.3_

  - [~] 2.1 Write unit tests for `get_weights_and_parameters`
    - Test correct weights path returned for all 6 tasks
    - Test `LayerNorm_type='BiasFree'` override for `Real_Denoising` and `Gaussian_Color_Denoising`
    - Test `inp_channels=1`, `out_channels=1`, `LayerNorm_type='BiasFree'` for `Gaussian_Gray_Denoising`
    - Test default parameters are not mutated for tasks with no overrides
    - _Requirements: 1.2, 9.1, 9.2, 9.3_

- [~] 3. Implement CLI argument parsing
  - Add `argparse` setup with `--task`, `--input_dir`, `--result_dir`, `--tile`, `--tile_overlap` arguments
  - Enforce `--task` choices to the 6 valid values; make `--task` required
  - _Requirements: 1.1, 1.3, 2.1, 5.2, 6.1_

  - [~] 3.1 Write unit tests for CLI argument parsing
    - Test that omitting `--task` causes `SystemExit`
    - Test that an invalid `--task` value causes `SystemExit`
    - Test default values for `--input_dir`, `--result_dir`, `--tile_overlap`
    - _Requirements: 1.1, 1.3_

- [~] 4. Implement file discovery logic
  - Write logic to detect single-file vs directory input and collect supported-extension files in natural sort order
  - Raise `Exception` when no supported files are found
  - _Requirements: 2.1, 2.2, 2.3, 2.4_

  - [~] 4.1 Write property test for file discovery (Property 4)
    - **Property 4: File discovery returns supported files in natural sort order**
    - **Validates: Requirements 2.2, 2.3**
    - Generate arbitrary sets of filenames mixing supported and unsupported extensions; assert result matches natsorted supported files only
    - `# Feature: restormer-image-restoration, Property 4: File discovery returns supported files in natural sort order`

  - [~] 4.2 Write unit tests for file discovery
    - Test single-file path with supported extension is returned as a one-element list
    - Test exception raised when directory contains no supported files
    - _Requirements: 2.2, 2.4_

- [~] 5. Implement image preprocessing (padding)
  - Write padding logic: compute `H'`, `W'` as next multiples of 8; apply `F.pad` with `'reflect'` mode
  - _Requirements: 3.3, 3.4_

  - [~] 5.1 Write property test for padding divisibility (Property 1)
    - **Property 1: Padding produces dimensions divisible by 8**
    - **Validates: Requirements 3.3**
    - Generate arbitrary `(H, W)` pairs; apply padding formula; assert `H' % 8 == 0` and `W' % 8 == 0`
    - `# Feature: restormer-image-restoration, Property 1: Padding produces dimensions divisible by 8`

  - [~] 5.2 Write property test for unpadding round-trip (Property 2)
    - **Property 2: Unpadding restores original dimensions**
    - **Validates: Requirements 4.3, 5.7**
    - Generate arbitrary `(H, W)`; pad then crop `[:, :, :H, :W]`; assert spatial dims equal `(H, W)`
    - `# Feature: restormer-image-restoration, Property 2: Unpadding restores original dimensions`

- [~] 6. Implement standard inference path
  - Write full-image forward pass (`restored = model(input_)`) when `--tile` is not provided
  - Apply `torch.clamp(restored, 0, 1)` and unpad to original dimensions
  - _Requirements: 4.1, 4.2, 4.3_

  - [~] 6.1 Write property test for output clamping (Property 3)
    - **Property 3: Output values are clamped to [0, 1]**
    - **Validates: Requirements 4.2, 5.6**
    - Generate arbitrary float tensors with values outside `[0, 1]`; apply `torch.clamp`; assert all values in `[0, 1]`
    - `# Feature: restormer-image-restoration, Property 3: Output values are clamped to [0, 1]`

- [~] 7. Implement tiled inference path
  - Write tile index generation using `stride = tile - tile_overlap`
  - Accumulate tile outputs into energy tensor `E` and weight tensor `W`; compute `E / W`
  - Enforce `assert tile % 8 == 0` and clamp tile size to `min(tile, h, w)`
  - Apply clamp and unpad after aggregation
  - _Requirements: 5.1, 5.2, 5.3, 5.4, 5.5, 5.6, 5.7_

  - [~] 7.1 Write property test for tile coverage (Property 5)
    - **Property 5: Tile coverage is complete**
    - **Validates: Requirements 5.1**
    - Generate arbitrary `(H, W, T, overlap)` where `T <= min(H, W)` and `T % 8 == 0`; assert every pixel is covered by at least one tile
    - `# Feature: restormer-image-restoration, Property 5: Tile coverage is complete`

  - [~] 7.2 Write property test for tile size clamping (Property 6)
    - **Property 6: Tile size is clamped to image dimensions**
    - **Validates: Requirements 5.4**
    - Generate arbitrary `T, H, W`; assert effective tile equals `min(T, H, W)`
    - `# Feature: restormer-image-restoration, Property 6: Tile size is clamped to image dimensions`

  - [~] 7.3 Write unit tests for tiled inference edge cases
    - Test `AssertionError` raised when tile size is not a multiple of 8
    - _Requirements: 5.3_

- [~] 8. Checkpoint — Ensure all tests pass
  - Ensure all tests pass, ask the user if questions arise.

- [~] 9. Implement output saving and directory creation
  - Create output directory `os.path.join(result_dir, task)` with `os.makedirs(..., exist_ok=True)`
  - Save restored images as PNG using base filename; dispatch to `save_gray_img` or `save_img` based on task
  - _Requirements: 6.1, 6.2, 6.3, 6.4, 6.5, 6.6_

  - [~] 9.1 Write property test for output path (Property 7)
    - **Property 7: Output path is task-scoped subdirectory**
    - **Validates: Requirements 6.2**
    - Generate arbitrary `result_dir` strings and task names; assert output dir equals `os.path.join(result_dir, task)`
    - `# Feature: restormer-image-restoration, Property 7: Output path is task-scoped subdirectory`

  - [~] 9.2 Write property test for output filename (Property 8)
    - **Property 8: Output filename preserves input base name with PNG extension**
    - **Validates: Requirements 6.4**
    - Generate arbitrary input filepaths with various extensions and directory structures; assert output filename equals `stem + '.png'`
    - `# Feature: restormer-image-restoration, Property 8: Output filename preserves input base name with PNG extension`

  - [~] 9.3 Write unit tests for output saving
    - Test `os.makedirs` called with `exist_ok=True`
    - Test `save_gray_img` called for `Gaussian_Gray_Denoising`, `save_img` for all others
    - _Requirements: 6.3, 6.5, 6.6_

- [~] 10. Implement model loading and device selection
  - Load Restormer architecture via `run_path`; instantiate with resolved parameters
  - Select device (`cuda` if available, else `cpu`); move model to device
  - Load checkpoint and apply `model.load_state_dict(checkpoint['params'])`; set `model.eval()`
  - Clear CUDA cache per image when on GPU
  - _Requirements: 1.2, 1.4, 7.1, 7.2, 7.3, 9.4_

  - [~] 10.1 Write unit tests for device selection and model loading
    - Test CUDA device selected when `torch.cuda.is_available()` returns `True` (mock)
    - Test CPU device selected when `torch.cuda.is_available()` returns `False` (mock)
    - Test `checkpoint['params']` key used for `load_state_dict`
    - Test CUDA cache cleared per image when on GPU (mock)
    - _Requirements: 7.1, 7.2, 7.3, 9.4_

- [~] 11. Implement progress reporting
  - Print task name and weights path before inference loop
  - Wrap file iteration with `tqdm` progress bar
  - Print output directory path after all images are processed
  - _Requirements: 8.1, 8.2, 8.3_

- [~] 12. Wire all components together in `demo.py`
  - Connect argument parsing → `get_weights_and_parameters` → model load → file discovery → preprocessing → inference (standard or tiled) → output saving → progress reporting in linear execution order
  - _Requirements: 1.1, 1.2, 2.1, 3.3, 4.1, 5.1, 6.1, 7.1, 8.1, 9.1_

- [~] 13. Final checkpoint — Ensure all tests pass
  - Ensure all tests pass, ask the user if questions arise.

## Notes

- Tasks marked with `*` are optional and can be skipped for faster MVP
- Each task references specific requirements for traceability
- Property tests use Hypothesis with `@settings(max_examples=100)` and include the property annotation comment
- Unit tests use pytest with mocking via `unittest.mock`

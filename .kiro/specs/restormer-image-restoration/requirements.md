# Requirements Document

## Introduction

Restormer is an image restoration system based on an Efficient Transformer architecture for high-resolution images. The system accepts degraded images as input and produces restored images using pretrained deep learning models. It supports six restoration tasks: Motion Deblurring, Single Image Defocus Deblurring, Deraining, Real Denoising, Gaussian Gray Denoising, and Gaussian Color Denoising. The system runs inference via a command-line interface and supports both single-image and batch processing, with optional tiled inference for large images.

## Glossary

- **System**: The Restormer image restoration pipeline as a whole
- **CLI**: The command-line interface exposed by `demo.py`
- **Restormer**: The transformer-based neural network architecture defined in `basicsr/models/archs/restormer_arch.py`
- **Task**: One of the six supported restoration operations (Motion_Deblurring, Single_Image_Defocus_Deblurring, Deraining, Real_Denoising, Gaussian_Gray_Denoising, Gaussian_Color_Denoising)
- **Input_Image**: A degraded image file provided by the user in JPG, PNG, JPEG, or BMP format
- **Restored_Image**: The output image produced by the Restormer after inference
- **Pretrained_Weights**: Task-specific `.pth` checkpoint files loaded before inference
- **Tile**: A rectangular sub-region of an image used during tiled inference
- **Tile_Overlap**: The number of pixels by which adjacent tiles overlap to avoid boundary artifacts
- **Device**: The compute device used for inference, either CUDA GPU or CPU

---

## Requirements

### Requirement 1: Task Selection and Model Loading

**User Story:** As a researcher, I want to select a specific image restoration task, so that the correct pretrained model is loaded for my use case.

#### Acceptance Criteria

1. THE CLI SHALL accept a `--task` argument with exactly one of the following values: `Motion_Deblurring`, `Single_Image_Defocus_Deblurring`, `Deraining`, `Real_Denoising`, `Gaussian_Gray_Denoising`, `Gaussian_Color_Denoising`.
2. WHEN a valid `--task` argument is provided, THE System SHALL load the Pretrained_Weights file corresponding to that Task before running inference.
3. WHEN the `--task` argument is omitted or set to an unsupported value, THE CLI SHALL display an error message and exit without running inference.
4. WHEN the Pretrained_Weights file for the selected Task does not exist at the expected path, THE System SHALL raise an error and halt execution.

---

### Requirement 2: Input Image Discovery

**User Story:** As a researcher, I want to provide either a single image path or a directory of images, so that I can process one or many images in a single run.

#### Acceptance Criteria

1. THE CLI SHALL accept an `--input_dir` argument that is either a path to a single image file or a path to a directory.
2. WHEN `--input_dir` points to a single file with a supported extension (jpg, JPG, png, PNG, jpeg, JPEG, bmp, BMP), THE System SHALL process that file only.
3. WHEN `--input_dir` points to a directory, THE System SHALL discover all files in that directory with supported extensions and process them in natural sort order.
4. IF no supported image files are found at the provided `--input_dir` path, THEN THE System SHALL raise an exception and halt execution.

---

### Requirement 3: Image Preprocessing

**User Story:** As a researcher, I want input images to be automatically prepared for the model, so that inference runs correctly regardless of image dimensions.

#### Acceptance Criteria

1. WHEN the selected Task is `Gaussian_Gray_Denoising`, THE System SHALL load Input_Images as single-channel grayscale arrays.
2. WHEN the selected Task is not `Gaussian_Gray_Denoising`, THE System SHALL load Input_Images as three-channel RGB arrays.
3. THE System SHALL pad Input_Images so that both height and width are multiples of 8 before passing them to the Restormer.
4. WHEN padding is applied, THE System SHALL use reflect padding to avoid introducing artificial edges.

---

### Requirement 4: Standard Inference

**User Story:** As a researcher, I want to run inference on full-resolution images, so that I get the highest quality restoration without tiling artifacts.

#### Acceptance Criteria

1. WHEN the `--tile` argument is not provided, THE System SHALL pass the full padded Input_Image tensor to the Restormer in a single forward pass.
2. WHEN inference is complete, THE System SHALL clamp the output tensor values to the range [0, 1].
3. WHEN inference is complete, THE System SHALL remove the padding added during preprocessing so the Restored_Image matches the original Input_Image dimensions.

---

### Requirement 5: Tiled Inference

**User Story:** As a researcher, I want to process large images using tiled inference, so that I can restore images that would otherwise exceed GPU memory limits.

#### Acceptance Criteria

1. WHEN the `--tile` argument is provided, THE System SHALL divide the padded input into overlapping Tiles of the specified size.
2. THE CLI SHALL accept a `--tile_overlap` argument specifying the number of overlapping pixels between adjacent Tiles, defaulting to 32.
3. WHEN `--tile` is provided, THE System SHALL require that the tile size is a multiple of 8, and SHALL raise an assertion error if it is not.
4. WHEN `--tile` is provided and the tile size exceeds the image height or width, THE System SHALL reduce the tile size to the smaller of the image height and width.
5. WHEN tiled inference is complete, THE System SHALL aggregate tile outputs by averaging overlapping regions to produce a seamless Restored_Image.
6. WHEN tiled inference is complete, THE System SHALL clamp the aggregated output tensor values to the range [0, 1].
7. WHEN tiled inference is complete, THE System SHALL remove the padding added during preprocessing so the Restored_Image matches the original Input_Image dimensions.

---

### Requirement 6: Output Saving

**User Story:** As a researcher, I want restored images saved to a structured output directory, so that I can easily find results organized by task.

#### Acceptance Criteria

1. THE CLI SHALL accept a `--result_dir` argument specifying the root directory for output.
2. THE System SHALL save Restored_Images to a subdirectory named after the Task within `--result_dir` (e.g., `<result_dir>/<task>/`).
3. WHEN the output directory does not exist, THE System SHALL create it before saving any images.
4. WHEN saving a Restored_Image, THE System SHALL use the same base filename as the Input_Image and save it in PNG format.
5. WHEN the selected Task is `Gaussian_Gray_Denoising`, THE System SHALL save the Restored_Image as a single-channel grayscale PNG.
6. WHEN the selected Task is not `Gaussian_Gray_Denoising`, THE System SHALL save the Restored_Image as a three-channel RGB PNG.

---

### Requirement 7: Device Selection

**User Story:** As a researcher, I want the system to automatically use a GPU when available, so that inference runs as fast as possible without manual configuration.

#### Acceptance Criteria

1. WHEN a CUDA-capable GPU is available, THE System SHALL run inference on the GPU.
2. WHEN no CUDA-capable GPU is available, THE System SHALL run inference on the CPU.
3. WHILE running inference on a CUDA GPU, THE System SHALL clear the CUDA memory cache before processing each image.

---

### Requirement 8: Progress Reporting

**User Story:** As a researcher, I want to see progress during batch processing, so that I know the system is running and can estimate completion time.

#### Acceptance Criteria

1. WHEN processing multiple images, THE System SHALL display a progress bar indicating the number of images processed and remaining.
2. WHEN inference begins, THE System SHALL print the selected Task name and the Pretrained_Weights path to standard output.
3. WHEN all images have been processed, THE System SHALL print the output directory path to standard output.

---

### Requirement 9: Model Architecture Configuration

**User Story:** As a researcher, I want the model architecture to be correctly configured per task, so that the loaded weights are compatible with the model structure.

#### Acceptance Criteria

1. THE System SHALL initialize the Restormer with default parameters: `inp_channels=3`, `out_channels=3`, `dim=48`, `num_blocks=[4,6,6,8]`, `num_refinement_blocks=4`, `heads=[1,2,4,8]`, `ffn_expansion_factor=2.66`, `bias=False`, `LayerNorm_type='WithBias'`, `dual_pixel_task=False`.
2. WHEN the selected Task is `Real_Denoising` or `Gaussian_Color_Denoising`, THE System SHALL override `LayerNorm_type` to `'BiasFree'`.
3. WHEN the selected Task is `Gaussian_Gray_Denoising`, THE System SHALL override `inp_channels` to `1`, `out_channels` to `1`, and `LayerNorm_type` to `'BiasFree'`.
4. WHEN loading Pretrained_Weights, THE System SHALL load the `params` key from the checkpoint dictionary into the model state dict.

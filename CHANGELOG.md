# Changelog

All notable changes to this project will be documented in this file.

## [0.1.1] - 2025-07-02

### Changed

* Updated processing time estimation coefficients for NVIDIA RTX 4070 and RTX 4060 Ti GPUs in `textify/media.py`.
* Simplified resource monitoring test in `tests/test_system.py` to use a shorter sampling interval (`0.1s`) and added a timeout to `thread.join()` to prevent hanging.
* Adjusted expectations in `tests/test_media.py` to reflect the new coefficients (`0.089 * duration + 15` for RTX 4070 and `0.134 * duration + 10.8` for RTX 4060 Ti).

## \[0.1.0] - 2025-06-27

### Added

* Initial release of **textify**.
* Batch processing of new audio and video files via the OpenAI Whisper.
* Document and image processing with OCR:
  * PDF handling first tries direct text extraction via PyMuPDF, then falls back to EasyOCR
  * Support for common image formats (jpg, png, bmp, tiff, etc.)
  * English / Japanese OCR reader pre‑loaded (no automatic language detection required)
* Automatic detection of unprocessed media files without corresponding transcript `.txt` files.
* Resource monitoring:

  * CPU utilization sampling using `psutil`.
  * GPU utilization, memory usage, power consumption, and temperature reporting via NVIDIA NVML (`pynvml`).
  * **Total GPU energy consumption (Wh) computed with trapezoidal integration.**
* Media duration estimation via `ffprobe` plus processing‑time models for NVIDIA GeForce RTX 4070 / 4060 Ti GPUs.
* Device selection with CUDA/CPU fallback and configurable GPU usage threshold (`--gpu-threshold`, `--ignore-gpu-threshold`).
* Detailed logging:

  * Timestamped console and file logging (single append‑mode log file per run).
  * Per-file dump files containing start/end timestamps, media duration, estimated and actual processing times, and full transcription output.
* Improved file handling:
  * Output files use `filename_extension.txt` format (e.g., `sample_mp3.txt` instead of `sample.txt`)
  * Corresponding dump files use `filename_extension_dump.txt` format
* Automatic file categorization:
  * Audio/video files processed with Whisper
  * Document/image files processed with OCR
* Comprehensive CLI options:

  * `--input-dir`, `--log-file`, `--monitoring-interval`, `--model`, `--language`, `--device`.
* Python package distribution via setuptools (`setup.py` and `pyproject.toml`) with `textify` console script entry point.
* Modular package architecture with clean separation of concerns:
  * `textify/core.py` - Main application logic and orchestration
  * `textify/utils.py` - Utility functions (time formatting, file discovery, categorization)
  * `textify/system.py` - System checks, GPU monitoring, resource monitoring
  * `textify/media.py` - Audio/video processing with Whisper
  * `textify/documents.py` - Document/image processing with OCR
  * `textify/cli.py` - Command-line argument parsing
  * `textify/__init__.py` - Package initialization with lazy imports
* Comprehensive test suite split by functionality:
  * `tests/test_core.py` - Tests for core functionality
  * `tests/test_utils.py` - Tests for utility functions
  * `tests/test_system.py` - Tests for system functionality
  * `tests/test_media.py` - Tests for media processing
  * `tests/test_documents.py` - Tests for document processing
  * `tests/test_cli.py` - Tests for CLI functionality
* Multiple usage methods:
  * Package import: `import textify; textify.main()`
  * Console script: `textify` (after installation)
* English and Japanese documentation (`README.md`, `README_jp.md`) with modular architecture details
* GitHub Actions CI workflow for running pytest on Python 3.10–3.13 (`.github/workflows/pytest.yml`).
* Development dependencies specified in `requirements-dev.txt` (pytest, pytest-mock, coverage, pytest-cov).
* Licensed under the MIT License.

### Changed

* `get_eligible_files` now skips textify-generated `.txt` files silently.
* Warnings for unsupported files are shown only when verbose mode is enabled.


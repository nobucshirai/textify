# textify

[![Version](https://img.shields.io/github/v/release/nobucshirai/textify?include_prereleases)](https://github.com/nobucshirai/textify/releases)
[![Tests](https://img.shields.io/github/actions/workflow/status/nobucshirai/textify/pytest.yml?label=pytest)](https://github.com/nobucshirai/textify/actions)
[![License](https://img.shields.io/github/license/nobucshirai/textify)](https://github.com/nobucshirai/textify/blob/main/LICENSE)

A modular Python package for batch processing audio, video, and document files. Uses [Whisper](https://github.com/openai/whisper) for audio/video transcription and [EasyOCR](https://github.com/JaidedAI/EasyOCR) for document/image text extraction. Features a clean modular architecture with specialized modules for different processing types, comprehensive logging of CPU/GPU usage, power consumption, processing time, and system information to help you optimize your workflows.

## Features

* **Modular Architecture**: Clean separation of concerns with specialized modules for media processing, document handling, system monitoring, and utilities
* **Batch Processing**: Processes individual files passed as arguments or automatically detects and transcribes new media files in a specified directory
* **Multi‑format Support**  
  * Transcribe audio & video with Whisper  
  * Extract text from PDFs & images with EasyOCR (automatic direct‑text extraction for PDFs, OCR fallback)
* **Resource Monitoring**: Records CPU/GPU utilisation *and total GPU energy consumption (Wh)*
* **Device Selection**: Automatically handles GPU (`cuda`) or CPU processing and falls back to CPU if CUDA is unavailable
* **Duration Estimation**: Estimates processing time for supported NVIDIA GPUs (RTX 4070, 4060 Ti)
* **Robust Logging**: Timestamped console & file logging (single append‑mode log per run) and per‑file dump files

## Getting Started

Quick installation and basic usage:

```bash
# Create and activate virtual environment
python -m venv textify-env
source textify-env/bin/activate

# Install textify
pip install git+https://github.com/nobucshirai/textify.git

# Process specific files
textify audio1.mp3 video1.mp4 document.pdf image.jpg

# Run with directory (uses GPU by default)
textify --input-dir /path/to/media_dir

# CPU-only operation with English language setting
textify --input-dir /path/to/media_dir --device cpu --language English
```

That's it! For more advanced options and configurations, see the detailed sections below.

## Prerequisites

* This program has been tested on Ubuntu 24.04 LTS.
* Python 3.8 or later
* `openai-whisper` Python package (installed via pip)
* `psutil` for CPU monitoring (optional but recommended)
* `nvidia-ml-py` (pynvml) for NVIDIA GPU monitoring (optional)
* `ffprobe` (from FFmpeg) for media duration estimation (optional)
* NVIDIA drivers and NVML library for GPU support (if using `--device cuda`)
* `easyocr` for document and image text extraction (optional)
* `PyMuPDF` for PDF processing (optional)

## Package Structure

The textify package uses a modular architecture for better maintainability and extensibility:

```text
textify/
├── __init__.py         # Package initialization and exports
├── core.py             # Core processing logic and main entry point
├── cli.py              # Command-line interface and argument parsing
├── utils.py            # Utility functions, formatting, file management
├── system.py           # System checks, monitoring, and hardware detection
├── media.py            # Audio/video processing with Whisper
└── documents.py        # Document/image processing with OCR
```

### Module Responsibilities

* **`core.py`**: Main application workflow, logging setup, and process orchestration
* **`cli.py`**: Command-line argument parsing and validation
* **`utils.py`**: Time formatting, file discovery, categorization, and logging utilities
* **`system.py`**: Hardware detection (CUDA, GPU, ffprobe), resource monitoring
* **`media.py`**: Audio/video duration extraction, Whisper model loading, and transcription
* **`documents.py`**: PDF and image processing with EasyOCR and PyMuPDF

## Installation

First, create and activate a virtual environment:

```bash
# Create a virtual environment
python -m venv textify-env

# Activate the virtual environment
source textify-env/bin/activate
```

Install via GitHub:

```bash
pip install git+https://github.com/nobucshirai/textify.git
```

## Usage

The textify package can be used in multiple ways after installation:

1. **Package command**: Use the installed `textify` command
2. **Python import**: Import and use the core functionality programmatically

### Package Command (Recommended)

After installation via pip, use the `textify` command:

The textify package supports two modes of operation:

1. **File mode**: Process specific files by passing them as arguments
2. **Directory mode**: Process all unprocessed files in a directory using `--input-dir`

```bash
# File mode
textify \
  audio1.mp3 video1.mp4 document.pdf \
  --model medium \
  --language English \
  --device cpu

# Directory mode
textify \
  --input-dir ./media \
  --log-file ./logs \
  --model medium \
  --language English \
  --device cpu
```

### Programmatic Usage

You can also import and use the core functionality in your own Python scripts:

```python
from textify.core import main

# Run with custom arguments
import sys
sys.argv = ['textify', '--input-dir', '/path/to/media', '--device', 'cpu']
main()

# Or use individual modules
from textify.media import process_audio_video_files
from textify.documents import process_document_files
from textify.system import initialize_system_checks
```

### Command-Line Arguments

* `files` (positional): Audio/video/document files to process (cannot be used with `--input-dir`)
* `--input-dir`: Directory containing media files to process (cannot be used with file arguments)
* `--log-file`: Path or directory for log output. If a directory is provided, a dated log file is created inside.
* `--monitoring-interval`: Seconds between resource usage samples (default: `10`).
* `--gpu-threshold`: GPU utilization (%) below which processing is allowed (default: `20`).
* `--ignore-gpu-threshold`: Process files regardless of current GPU usage.
* `--model`: Whisper model name (default: `large`).
* `--language`: Transcription language (default: `Japanese`).
* `--device`: Processing device, `cuda` or `cpu` (default: `cuda`; automatically falls back to `cpu` if CUDA is unavailable).
* `-w`, `--watch`: Watch `--input-dir` for new files and process them using `watchdog`.

**Note**: You must specify either `--input-dir` or provide files as arguments, but not both.

## Supported File Types

* **Audio**: mp3, wav, aac, flac, ogg, m4a, wma
* **Video**: mp4, mov, avi, wmv, flv, mkv, webm, m4v, mpg, mpeg, and more
* **Documents**: pdf
* **Images**: jpg, jpeg, png, bmp, tiff, tif, webp, gif, heic, heif

## Logging

* **Log File**: Contains timestamped entries for system info, resource usage, estimated and actual processing times.
* **Dump Files**: For each processed file, a `_dump.txt` file is generated in the same directory as the source file with:

  * Start & end timestamps
  * Media duration (for audio/video)
  * Estimated & actual processing time
  * Full Whisper or OCR output

## Output Files

For each processed file, textify creates:

* A text transcript file with the naming pattern `filename_extension.txt` (e.g., `lecture_mp3.txt`, `document_pdf.txt`)
* A dump file with the naming pattern `filename_extension_dump.txt` containing processing details and the full output

## Examples

* **Basic Japanese transcription on GPU (directory mode)**:

  ```bash
  textify --input-dir ./media
  ```

* **Process specific files with English transcription on CPU**:

  ```bash
  textify \
    audio1.mp3 video1.mp4 document.pdf \
    --language English \
    --device cpu
  ```

* **Custom log directory and lower sampling interval (directory mode)**:

  ```bash
  textify \
    --input-dir ./media \
    --log-file ./logs
  ```

* **Process files from different directories**:

  ```bash
  textify \
    /path/to/audio1.mp3 \
    /another/path/video1.mp4 \
    ./local/document.pdf
  ```

## Automated Processing with systemd

Textify includes a `--watch` option that monitors a directory for new files using the `watchdog` library. You can run this watcher manually or keep it running in the background with systemd.

If you simply want to monitor a directory without systemd, run:

```bash
textify --input-dir /path/to/media --watch
```

### Setup Steps

1. **Create a systemd service file** (e.g., `watch_textify.service`) in `~/.config/systemd/user/`:

```ini
[Unit]
Description=Watch Textify Directory
After=network.target

[Service]
Type=simple
ExecStart=/path/to/venv/bin/textify \
    --input-dir /path/to/media \
    --watch \
    --log-file /path/to/logs
Restart=always
Environment=PATH=/path/to/textify-env/bin:/usr/bin:/bin
WorkingDirectory=/path/to/working/directory

[Install]
WantedBy=default.target
```

2. **Enable and start the service**:

```bash
systemctl --user daemon-reload
systemctl --user enable watch_textify.service
systemctl --user start watch_textify.service
systemctl --user status watch_textify.service
```

With this setup, textify will continuously monitor the specified directory and process new files as they appear.

## Complementary Automation with Cron

While systemd with the `--watch` option provides real-time monitoring, it may occasionally miss files if multiple files are added simultaneously. For robust automation, you can complement it with a scheduled cron job that periodically checks for unprocessed files.

> **Note**: Cron-based automation can also be used independently without systemd if you prefer a simpler setup with scheduled checks rather than real-time monitoring.

### Setup Steps for Cron

1. **Create a processing script** (e.g., `process_directories.sh`):

```bash
#!/bin/bash

# Base directory containing media directories
BASE_DIR="/path/to/base/directory"
# Directory containing the textify script
SCRIPT_DIR="/path/to/textify"
# Activate your virtual environment
source /path/to/textify-env/bin/activate

# Process each directory that might contain media files
ls -dtr ${BASE_DIR}/* | grep -v "log" | while read dir; do
    if [[ -d ${dir} ]]; then
        # Process the directory with textify
        ${SCRIPT_DIR}/textify.py \
            --input-dir "${dir}" \
            --log-file "${BASE_DIR}/logs/"
        
        # Optional: wait a moment before processing the next directory
        sleep 1
    fi
done

# You can also add specific directories that should always be checked
${SCRIPT_DIR}/textify.py \
    --input-dir ${BASE_DIR}/special_media_dir \
    --log-file ${BASE_DIR}/logs/
```

2. **Make the script executable**:

```bash
chmod +x /path/to/process_directories.sh
```

3. **Add a cron job** (edit with `crontab -e`):

```bash
# Run every 15 minutes
*/15 * * * * /usr/bin/flock -n /path/to/lockfile /path/to/process_directories.sh 1> /path/to/logs/cron_textify.log 2> /path/to/logs/cron_textify.err
```

The `flock` command ensures that only one instance of the script runs at a time, preventing overlapping processes if the previous run hasn't completed.

### Why Use Both Systemd and Cron?

* **Systemd + `--watch`**: Provides immediate, event-driven processing when files are added but might miss events during high file system activity
* **Cron**: Acts as a safety net by periodically checking for any files that might have been missed by the event-based system

This complementary approach ensures maximum reliability:

1. Most files are processed immediately by the systemd service
2. Any missed files are caught during the next scheduled cron job
3. Using `flock` prevents resource conflicts if both systems try to process files simultaneously

For environments where continuous, reliable processing is critical, implementing both methods provides the most robust solution.

## Development Workflow

The modular architecture makes it easy to extend textify with new features. When adding functionality:

### Adding New Features

* **Audio/Video features**: Add to `textify/media.py`
* **Document/Image features**: Add to `textify/documents.py`
* **System monitoring**: Add to `textify/system.py`
* **Utility functions**: Add to `textify/utils.py`
* **CLI options**: Add to `textify/cli.py`
* **Integration logic**: Add to `textify/core.py`

### Testing

The modular structure enables isolated testing of individual components:

```bash
# Run all tests
python -m pytest tests/

# Test specific modules
python -m pytest tests/test_media.py
python -m pytest tests/test_documents.py
```

### Benefits of the Modular Structure

* **Single Responsibility**: Each module has a clear, focused purpose
* **Easier Maintenance**: Locate and modify specific functionality quickly
* **Better Testing**: Test individual components in isolation
* **Improved Readability**: Smaller, focused files are easier to understand
* **Reduced Coupling**: Modules interact through well-defined interfaces
* **Future Extensibility**: Easy to add new processing types or system monitors

## Author

[Nobu C. Shirai](https://github.com/nobucshirai) (Mie University, Japan)

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

## Acknowledgments

* [OpenAI](https://openai.com) for creating and open-sourcing the [Whisper](https://github.com/openai/whisper) speech recognition system that this project builds upon.
* [JaidedAI](https://github.com/JaidedAI) for developing and maintaining [EasyOCR](https://github.com/JaidedAI/EasyOCR), which is used for document and image text extraction in this project.

"""
Textify - Batch processing audio, video, and document files with Whisper and EasyOCR.
"""

__version__ = "0.1.1"

# Import key functions to make them available at package level
# Note: We import only essential functions to avoid circular imports

__all__ = [
    "main",
    "parse_arguments",
    "initialize_system_checks",
    "get_gpu_info",
    "format_time_for_display",
    "get_media_duration",
    "estimate_processing_time",
    "process_document_with_ocr",
]

# Import main function
def main():
    from .core import main as _main
    return _main()

# Import other functions on demand to avoid circular imports
def parse_arguments():
    from .cli import parse_arguments as _parse_arguments
    return _parse_arguments()

def initialize_system_checks(verbose=False):
    from .system import initialize_system_checks as _initialize_system_checks
    return _initialize_system_checks(verbose)

def get_gpu_info():
    from .system import get_gpu_info as _get_gpu_info
    return _get_gpu_info()

def format_time_for_display(seconds):
    from .utils import format_time_for_display as _format_time_for_display
    return _format_time_for_display(seconds)

def get_media_duration(path):
    from .media import get_media_duration as _get_media_duration
    return _get_media_duration(path)

def estimate_processing_time(duration_sec):
    from .media import estimate_processing_time as _estimate_processing_time
    return _estimate_processing_time(duration_sec)

def process_document_with_ocr(file_path):
    from .documents import process_document_with_ocr as _process_document_with_ocr
    return _process_document_with_ocr(file_path)

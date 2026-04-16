"""
Command-line interface module for textify.

This module handles argument parsing and CLI configuration.
"""

import argparse


WHISPER_EXTRA_OUTPUT_FORMATS = ["vtt", "srt", "tsv", "json"]


def parse_whisper_output_formats(value: str) -> list[str]:
    """
    Parse a comma-separated list of additional Whisper output formats.

    The plain text transcript is always written separately, so this option only
    controls extra output files such as VTT or JSON.
    """
    if value is None:
        return []

    raw_formats = [item.strip().lower() for item in value.split(",") if item.strip()]
    if not raw_formats:
        return []

    invalid_formats = [
        item for item in raw_formats
        if item not in WHISPER_EXTRA_OUTPUT_FORMATS and item != "all"
    ]
    if invalid_formats:
        valid = ", ".join(["all", *WHISPER_EXTRA_OUTPUT_FORMATS])
        raise argparse.ArgumentTypeError(
            f"Invalid Whisper output format(s): {', '.join(invalid_formats)}. "
            f"Valid values are: {valid}"
        )

    if "all" in raw_formats:
        return WHISPER_EXTRA_OUTPUT_FORMATS.copy()

    deduped_formats = []
    for item in raw_formats:
        if item not in deduped_formats:
            deduped_formats.append(item)

    return deduped_formats


def parse_arguments() -> argparse.Namespace:
    """
    Parses command-line arguments.

    Returns:
        argparse.Namespace: An object containing the parsed arguments.
    """
    parser = argparse.ArgumentParser(description='Batch transcribe media and extract text with resource monitoring')
    parser.add_argument('files', nargs='*', 
                        help='Audio/video files to process')
    parser.add_argument('--input-dir', type=str, 
                        help='Path to the directory containing MP3 (or other) audio files')
    parser.add_argument('--log-file', type=str, default='batch_process.log', 
                        help='Path to save the log file (if a directory is specified, \
                              the file name is generated using date)')
    parser.add_argument('--monitoring-interval', type=int, default=10, 
                        help='Interval in seconds to record resource usage')
    parser.add_argument('--gpu-threshold', type=int, default=20, 
                        help='GPU usage threshold percentage below which processing will proceed')
    parser.add_argument('--ignore-gpu-threshold', action='store_true',
                        help='Process files regardless of GPU usage (ignores threshold)')
    parser.add_argument('--model', type=str, default='large', 
                        help='Whisper model name (e.g., large, medium, etc.)')
    parser.add_argument('--language', type=str, default='Japanese', 
                        help='Language to use for transcription (e.g., Japanese, English)')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use for processing (cuda or cpu)')
    parser.add_argument(
        '--whisper-output-formats',
        type=parse_whisper_output_formats,
        default=[],
        help=(
            'Comma-separated additional Whisper output formats to write for '
            'audio/video files. txt is always written separately. '
            'Supported values: vtt,srt,tsv,json,all'
        ),
    )
    parser.add_argument('--verbose', '-v', action='store_true',
                        help='Enable verbose logging output')
    parser.add_argument('-w', '--watch', dest='watch', action='store_true',
                        help='Watch --input-dir for new files and process them')

    args = parser.parse_args()
    
    # Validate that either files or input-dir is provided
    if not args.files and not args.input_dir:
        parser.error("Either provide audio/video files as arguments or use --input-dir")
    
    if args.files and args.input_dir:
        parser.error("Cannot use both file arguments and --input-dir at the same time")

    if args.watch and not args.input_dir:
        parser.error("--watch requires --input-dir")

    return args

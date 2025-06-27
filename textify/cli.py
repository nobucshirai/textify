"""
Command-line interface module for textify.

This module handles argument parsing and CLI configuration.
"""

import argparse


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

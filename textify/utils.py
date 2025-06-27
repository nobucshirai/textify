"""
Utility functions module for textify.

This module contains helper functions for formatting, logging, and file management.
"""

import logging
import os
from typing import List


def has_handler_of_type(logger, handler_class):
    """Check if logger already has a handler of the specified type."""
    return any(isinstance(handler, handler_class) for handler in logger.handlers)


def format_time_for_display(seconds: float) -> str:
    """
    Format time in seconds to a human-readable string with appropriate units.
    
    Args:
        seconds (float): Time in seconds
        
    Returns:
        str: Formatted time string with appropriate units
    """
    if seconds < 60:
        return f"{seconds:.2f} seconds"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.2f} minutes ({seconds:.2f} seconds)"
    elif seconds < 86400:  # 24 hours
        hours = seconds / 3600
        return f"{hours:.2f} hours ({seconds:.2f} seconds)"
    else:
        days = seconds / 86400
        return f"{days:.2f} days ({seconds:.2f} seconds)"


def get_eligible_files(
    input_dir: str = None,
    file_list: list = None,
    verbose: bool = False,
) -> list:
    """
    Get a list of audio/video files that don't have corresponding .txt files 
    (i.e., files that need processing).
    
    Args:
        input_dir (str, optional): Path to the directory containing audio/video files.
        file_list (list, optional): List of specific files to check.
        verbose (bool, optional): Show warnings for unsupported files when True.
        
    Returns:
        list: List of eligible file paths that need processing.
    """
    # Supported audio/video file extensions
    audio_video_extensions = [
        '.mp3', '.wav', '.aac', '.flac', '.ogg', '.m4a', '.wma',
        '.mp4', '.mov', '.avi', '.wmv', '.flv', '.mkv', '.webm',
        '.m4v', '.mpg', '.mpeg', '.3gp', '.3g2', '.rm', '.rmvb',
        '.vob', '.ts', '.ogv', '.f4v', '.divx'
    ]
    
    # Supported document/image file extensions
    document_image_extensions = [
        '.pdf', '.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', 
        '.webp', '.gif', '.heic', '.heif'
    ]
    
    # Combined supported extensions
    supported_extensions = audio_video_extensions + document_image_extensions

    if input_dir:
        # Original directory-based logic
        files = [f for f in os.listdir(input_dir)
                 if os.path.splitext(f)[1].lower() in supported_extensions]
        
        eligible_files = []
        for file_name in files:
            base_name = os.path.splitext(file_name)[0]
            ext = os.path.splitext(file_name)[1].lower()[1:]  # Get extension without the dot
            txt_file = os.path.join(input_dir, f"{base_name}_{ext}.txt")
            if not os.path.exists(txt_file):
                eligible_files.append(os.path.join(input_dir, file_name))
        
        return eligible_files
    
    elif file_list:
        # File list-based logic
        eligible_files = []
        for file_path in file_list:
            # Check if file exists
            if not os.path.exists(file_path):
                logging.warning(f"File not found: {file_path}")
                continue

            # Check if file has supported extension
            ext = os.path.splitext(file_path)[1].lower()
            if ext not in supported_extensions:
                # Skip textify output files silently
                if ext == ".txt":
                    continue
                if verbose:
                    logging.warning(f"Unsupported file type: {file_path}")
                continue
            
            # Check if corresponding .txt file exists
            dir_name = os.path.dirname(file_path)
            base_name = os.path.splitext(os.path.basename(file_path))[0]
            ext = os.path.splitext(file_path)[1].lower()[1:]  # Get extension without the dot
            txt_file = os.path.join(dir_name, f"{base_name}_{ext}.txt")
            
            if not os.path.exists(txt_file):
                eligible_files.append(file_path)
            else:
                logging.info(f"Skipping {file_path} - already processed")
        
        return eligible_files
    
    return []


def categorize_files(files: List[str]) -> tuple:
    """
    Categorize files into audio/video files and document/image files.
    
    Args:
        files (list): List of file paths to categorize.
        
    Returns:
        tuple: (audio_video_files, document_image_files)
    """
    # Audio/video file extensions
    audio_video_extensions = [
        '.mp3', '.wav', '.aac', '.flac', '.ogg', '.m4a', '.wma',
        '.mp4', '.mov', '.avi', '.wmv', '.flv', '.mkv', '.webm',
        '.m4v', '.mpg', '.mpeg', '.3gp', '.3g2', '.rm', '.rmvb',
        '.vob', '.ts', '.ogv', '.f4v', '.divx'
    ]
    
    # Document/image file extensions
    document_image_extensions = [
        '.pdf', '.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', 
        '.webp', '.gif', '.heic', '.heif'
    ]
    
    audio_video_files = []
    document_image_files = []
    
    for file_path in files:
        ext = os.path.splitext(file_path)[1].lower()
        if ext in audio_video_extensions:
            audio_video_files.append(file_path)
        elif ext in document_image_extensions:
            document_image_files.append(file_path)
    
    return audio_video_files, document_image_files

"""
Document and image processing module using OCR.

This module handles:
- PDF text extraction using PyMuPDF and EasyOCR
- Image text extraction using EasyOCR
- Document file processing workflow
"""

import logging
import os
import time
import datetime
from typing import List

# Import system module to access globals
from . import system


def process_document_with_ocr(file_path: str) -> str:
    """
    Process a document (PDF, image) with OCR.
    
    Args:
        file_path (str): Path to the document file.
        
    Returns:
        str: Extracted text from the document, or empty string if OCR is not available.
    """
    if not system.easyocr_available:
        logging.warning("EasyOCR not available. Cannot process document.")
        return ""
    
    file_ext = os.path.splitext(file_path)[1].lower()
    extracted_text = ""
    
    try:
        # Initialize reader with the language
        logging.info("Initializing EasyOCR reader")
        reader = system.easyocr.Reader(['en', 'ja'])  # Support English and Japanese by default
        
        # Process PDF files
        if file_ext == '.pdf':
            try:
                import fitz  # PyMuPDF
                logging.info(f"Processing PDF file: {file_path}")
                
                # Open the PDF
                doc = fitz.open(file_path)
                all_text = []
                
                for page_num in range(len(doc)):
                    page = doc.load_page(page_num)
                    
                    # Try to extract text directly first
                    text = page.get_text()
                    if text.strip():
                        all_text.append(f"--- Page {page_num + 1} (direct text) ---\n{text}\n")
                    else:
                        # If no direct text, use OCR on the page image
                        logging.info(f"No direct text found on page {page_num + 1}, using OCR")
                        pix = page.get_pixmap()
                        img_data = pix.tobytes("png")
                        
                        # Use EasyOCR on the image data
                        results = reader.readtext(img_data)
                        ocr_text = "\n".join([result[1] for result in results])
                        all_text.append(f"--- Page {page_num + 1} (OCR) ---\n{ocr_text}\n")
                
                doc.close()
                extracted_text = "\n".join(all_text)
                
            except ImportError:
                logging.warning("PyMuPDF not available. Using EasyOCR only for PDF processing.")
                # Fallback to EasyOCR only (may not work well with PDFs)
                results = reader.readtext(file_path)
                extracted_text = "\n".join([result[1] for result in results])
                
        else:
            # Process image files
            logging.info(f"Processing image file: {file_path}")
            results = reader.readtext(file_path)
            extracted_text = "\n".join([result[1] for result in results])
            
        return extracted_text
        
    except Exception as e:
        logging.error(f"Error during OCR processing: {str(e)}")
        return f"ERROR during OCR processing: {str(e)}"


def process_document_files(files: List[str]) -> None:
    """
    Process document/image files with OCR.
    
    Args:
        files (list): List of document/image file paths to process.
    """
    if not files:
        logging.info("No document/image files to process.")
        return
        
    logging.info(f"Processing {len(files)} document/image files with OCR")
    
    from .utils import format_time_for_display
    
    for file_path in files:
        dir_name = os.path.dirname(file_path)
        base_name = os.path.splitext(os.path.basename(file_path))[0]
        file_ext = os.path.splitext(file_path)[1].lower()
        ext_without_dot = file_ext[1:]  # Remove the leading dot
        txt_file = os.path.join(dir_name, f"{base_name}_{ext_without_dot}.txt")
        dump_file = os.path.join(dir_name, f"{base_name}_{ext_without_dot}_dump.txt")
        
        logging.info(f"Starting OCR processing of {os.path.basename(file_path)}.")
        
        start_time = time.time()
        start_datetime = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        # Write header information to the dump file
        with open(dump_file, 'w', encoding='utf-8') as f:
            f.write(f"Start time: {start_datetime}\n")
            f.write(f"Document/image processing with OCR\n")
            f.write("\n--- Processing Output ---\n\n")
        
        try:
            # Use OCR for document/image files
            extracted_text = process_document_with_ocr(file_path)
            
            # Write the extracted text to the dump file
            with open(dump_file, 'a', encoding='utf-8') as f:
                f.write(extracted_text)
            
            # Also create the .txt file (which is the marker for processed files)
            with open(txt_file, 'w', encoding='utf-8') as f:
                f.write(extracted_text)
                
        except Exception as e:
            logging.error(f"Error processing {os.path.basename(file_path)}: {str(e)}")
            with open(dump_file, 'a', encoding='utf-8') as f:
                f.write(f"\nERROR: {str(e)}\n")
        
        # Calculate elapsed time and append to dump file
        elapsed_time = time.time() - start_time
        end_datetime = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        with open(dump_file, 'a', encoding='utf-8') as f:
            f.write("\n\n--- Processing Summary ---\n")
            f.write(f"End time: {end_datetime}\n")
            f.write(f"Actual processing time: {format_time_for_display(elapsed_time)}\n")
        
        logging.info(f"Processing time for {os.path.basename(file_path)}: {format_time_for_display(elapsed_time)}")

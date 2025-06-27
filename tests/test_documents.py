import unittest
from unittest.mock import patch, MagicMock, mock_open
import os
import sys
import tempfile

# Add the parent directory to the path so we can import textify
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from textify.documents import process_document_with_ocr, process_document_files


class TestDocuments(unittest.TestCase):
    
    def setUp(self):
        # Create a temporary directory for test files
        self.temp_dir = tempfile.mkdtemp()
        
    def tearDown(self):
        # Clean up temporary directory
        import shutil
        shutil.rmtree(self.temp_dir)

    @patch('textify.documents.system.easyocr_available', True)
    @patch('textify.documents.system.easyocr')
    def test_process_document_with_ocr(self, mock_easyocr):
        """Test the OCR document processing function"""
        # Setup mock reader instance
        mock_reader = MagicMock()
        mock_reader.readtext.return_value = [
            (None, "Sample OCR text 1", None),
            (None, "Sample OCR text 2", None)
        ]
        
        # Configure the mock Reader class
        mock_reader_class = MagicMock(return_value=mock_reader)
        mock_easyocr.Reader = mock_reader_class
        
        # Test image file
        result = process_document_with_ocr("test.jpg")
        mock_reader_class.assert_called_once_with(['en', 'ja'])
        mock_reader.readtext.assert_called_once_with("test.jpg")
        self.assertEqual(result, "Sample OCR text 1\nSample OCR text 2")
        
        # Test with PDF file using PyMuPDF
        mock_reader.reset_mock()
        mock_reader_class.reset_mock()
        
        # Mock PyMuPDF and other modules needed for PDF processing
        # These are imported locally inside the function, so we need to patch sys.modules
        mock_np = MagicMock()
        mock_image = MagicMock()
        mock_io = MagicMock()
        
        with patch.dict('sys.modules', {
                'fitz': MagicMock(),
                'numpy': mock_np,
                'PIL': MagicMock(),
                'PIL.Image': mock_image,
                'io': mock_io
            }):
            # Mock PIL.Image for the import "from PIL import Image"
            sys.modules['PIL'].Image = mock_image
            
            # Mock PDF document and page
            mock_pdf = MagicMock()
            mock_page = MagicMock()
            mock_pixmap = MagicMock()
            
            # Set up the mocks for PDF processing
            sys.modules['fitz'].open.return_value = mock_pdf
            mock_pdf.__len__.return_value = 1
            mock_pdf.load_page.return_value = mock_page
            mock_page.get_text.return_value = ""  # Empty text to force OCR
            mock_page.get_pixmap.return_value = mock_pixmap
            
            # Mock the image processing
            mock_image_instance = MagicMock()
            mock_image.open.return_value = mock_image_instance
                
            result = process_document_with_ocr("test.pdf")
            # Verify the PDF was processed correctly
            sys.modules['fitz'].open.assert_called_once_with("test.pdf")
            mock_pdf.load_page.assert_called_once_with(0)
            self.assertIn("Sample OCR text 1", result)
        
    @patch('textify.documents.system.easyocr_available', False)
    def test_process_document_with_ocr_unavailable(self):
        """Test with EasyOCR not available"""
        result = process_document_with_ocr("test.pdf")
        self.assertEqual(result, "")

    @patch('builtins.open', new_callable=mock_open)
    @patch('textify.documents.process_document_with_ocr')
    def test_process_document_files(self, mock_ocr, mock_file):
        """Test document file processing function"""
        test_files = [
            os.path.join(self.temp_dir, 'doc1.pdf'),
            os.path.join(self.temp_dir, 'image1.jpg')
        ]
        
        # Configure mock
        mock_ocr.return_value = "OCR test result"
        
        # Test with files
        process_document_files(test_files)
        
        # Verify OCR was called for each file
        self.assertEqual(mock_ocr.call_count, 2)
        
        # Should have opened files for writing (both txt and dump files)
        # Each file creates 2 files: filename_ext.txt and filename_ext_dump.txt
        self.assertGreater(mock_file.call_count, 0)
        
        # Test with no files
        mock_ocr.reset_mock()
        mock_file.reset_mock()
        
        with patch('textify.documents.logging') as mock_logging:
            process_document_files([])
            
            # Should log that no files to process  
            mock_logging.info.assert_called_with("No document/image files to process.")
            
            # Should not call OCR
            mock_ocr.assert_not_called()


if __name__ == '__main__':
    unittest.main()

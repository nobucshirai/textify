import unittest
from unittest.mock import patch, MagicMock
import os
import sys
import tempfile

# Add the parent directory to the path so we can import textify
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from textify.utils import format_time_for_display, get_eligible_files, categorize_files


class TestUtils(unittest.TestCase):
    
    def setUp(self):
        # Create a temporary directory for test files
        self.temp_dir = tempfile.mkdtemp()
        
    def tearDown(self):
        # Clean up temporary directory
        import shutil
        shutil.rmtree(self.temp_dir)

    def test_format_time_for_display(self):
        """Test time formatting function"""
        # Test seconds
        result = format_time_for_display(45.5)
        self.assertEqual(result, "45.50 seconds")
        
        # Test minutes
        result = format_time_for_display(125.75)
        self.assertEqual(result, "2.10 minutes (125.75 seconds)")
        
        # Test hours
        result = format_time_for_display(3725)
        self.assertEqual(result, "1.03 hours (3725.00 seconds)")
        
        # Test days
        result = format_time_for_display(90000)
        self.assertEqual(result, "1.04 days (90000.00 seconds)")

    @patch('textify.utils.os.listdir')
    @patch('textify.utils.os.path.exists')
    def test_get_eligible_files(self, mock_exists, mock_listdir):
        """Test file discovery functionality"""
        # Mock directory listing
        mock_listdir.return_value = [
            'audio.mp3', 'video.mp4', 'document.pdf', 'image.jpg',
            'text.txt', 'hidden.mp3', 'another.wav'
        ]
        
        # Mock exists to return False for txt files (so files are eligible)
        def side_effect(path):
            return not path.endswith('.txt')
            
        mock_exists.side_effect = side_effect
        
        # Test with input directory
        files = get_eligible_files(input_dir='/test/dir')
        
        # Should find all eligible files (excluding text.txt and any that have corresponding txt files)
        expected_files = [
            '/test/dir/audio.mp3', '/test/dir/video.mp4', 
            '/test/dir/document.pdf', '/test/dir/image.jpg',
            '/test/dir/hidden.mp3', '/test/dir/another.wav'
        ]
        self.assertEqual(sorted(files), sorted(expected_files))
        
        # Test with file list
        file_list = ['file1.mp3', 'file2.pdf']
        files = get_eligible_files(file_list=file_list)
        self.assertEqual(files, file_list)
        
        # Test with empty directory
        mock_listdir.return_value = []
        files = get_eligible_files(input_dir='/empty/dir')
        self.assertEqual(files, [])
        
        # Test with non-existent directory (though this isn't really tested in the function)
        mock_exists.return_value = False
        files = get_eligible_files(input_dir='/nonexistent/dir')
        self.assertEqual(files, [])

    @patch('textify.utils.os.path.exists', return_value=True)
    @patch('textify.utils.logging.warning')
    def test_skip_textify_outputs(self, mock_warn, mock_exists):
        """Ensure textify output .txt files are ignored without warnings."""
        files = ['sample_mp3.txt', 'sample_mp3_dump.txt']
        result = get_eligible_files(file_list=files)
        self.assertEqual(result, [])
        mock_warn.assert_not_called()

    @patch('textify.utils.os.path.exists', return_value=True)
    @patch('textify.utils.logging.warning')
    def test_unsupported_file_warning_verbose(self, mock_warn, mock_exists):
        """Unsupported files should trigger warnings when verbose."""
        files = ['sample.xyz']
        result = get_eligible_files(file_list=files, verbose=True)
        self.assertEqual(result, [])
        mock_warn.assert_called_once_with('Unsupported file type: sample.xyz')

    def test_categorize_files(self):
        """Test file categorization by type"""
        files = [
            'audio.mp3', 'video.mp4', 'sound.wav', 'movie.avi',
            'document.pdf', 'image.jpg', 'photo.png', 'text.txt'
        ]
        
        audio_video, documents = categorize_files(files)
        
        # Check audio/video files
        expected_audio_video = ['audio.mp3', 'video.mp4', 'sound.wav', 'movie.avi']
        self.assertEqual(sorted(audio_video), sorted(expected_audio_video))
        
        # Check document files
        expected_documents = ['document.pdf', 'image.jpg', 'photo.png']
        self.assertEqual(sorted(documents), sorted(expected_documents))
        
        # Test with empty list
        audio_video, documents = categorize_files([])
        self.assertEqual(audio_video, [])
        self.assertEqual(documents, [])
        
        # Test with only audio files
        audio_video, documents = categorize_files(['test.mp3', 'test.wav'])
        self.assertEqual(audio_video, ['test.mp3', 'test.wav'])
        self.assertEqual(documents, [])
        
        # Test with only document files
        audio_video, documents = categorize_files(['test.pdf', 'test.jpg'])
        self.assertEqual(audio_video, [])
        self.assertEqual(documents, ['test.pdf', 'test.jpg'])


if __name__ == '__main__':
    unittest.main()

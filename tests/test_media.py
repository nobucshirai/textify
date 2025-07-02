import unittest
from unittest.mock import patch, MagicMock, mock_open
import datetime
import os
import sys
import tempfile

# Add the parent directory to the path so we can import textify
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from textify.media import (
    get_media_duration, 
    estimate_processing_time, 
    load_whisper_model_with_warning_suppression,
    process_audio_video_files
)

import textify.system as sysmod

class TestMedia(unittest.TestCase):
    
    def setUp(self):
        # Create a temporary directory for test files
        self.temp_dir = tempfile.mkdtemp()
        
    def tearDown(self):
        # Clean up temporary directory
        import shutil
        shutil.rmtree(self.temp_dir)

    @patch('textify.media.subprocess.run')
    def test_get_media_duration(self, mock_run):
        """Test media duration extraction"""
        # Configure the mock for successful case
        process_mock = MagicMock()
        process_mock.returncode = 0
        process_mock.stdout = "120.5"
        mock_run.return_value = process_mock
        
        # Set ffprobe available flag
        sysmod.ffprobe_available = True
        
        # Call the function
        result = get_media_duration("test.mp3")
        
        # Verify the result
        self.assertEqual(result, 120.5)
        
        # Test with error case
        process_mock.returncode = 1
        process_mock.stderr = "Error processing file"
        mock_run.return_value = process_mock
        
        result = get_media_duration("test.mp3")
        self.assertEqual(result, 0.0)
        
        # Test with ffprobe unavailable
        sysmod.ffprobe_available = False
        result = get_media_duration("test.mp3")
        self.assertEqual(result, 0.0)

    def test_estimate_processing_time(self):
        """Test processing time estimation"""
        # Setup
        sysmod.gpu_available = True
        sysmod.pynvml_available = True
        
        # Test with RTX 4070
        with patch('textify.system.pynvml') as mock_pynvml:
            mock_handle = MagicMock()
            mock_pynvml.nvmlDeviceGetHandleByIndex.return_value = mock_handle
            mock_pynvml.nvmlDeviceGetName.return_value = "NVIDIA GeForce RTX 4070"
            
            result = estimate_processing_time(300.0)
            self.assertAlmostEqual(result, 0.089 * 300.0 + 15, places=4)
        
        # Test with RTX 4060 Ti
        with patch('textify.system.pynvml') as mock_pynvml:
            mock_handle = MagicMock()
            mock_pynvml.nvmlDeviceGetHandleByIndex.return_value = mock_handle
            mock_pynvml.nvmlDeviceGetName.return_value = "NVIDIA GeForce RTX 4060 Ti"
            
            result = estimate_processing_time(300.0)
            self.assertAlmostEqual(result, 0.134 * 300.0 + 10.8, places=4)
        
        # Test with unknown GPU model
        with patch('textify.system.pynvml') as mock_pynvml:
            mock_handle = MagicMock()
            mock_pynvml.nvmlDeviceGetHandleByIndex.return_value = mock_handle
            mock_pynvml.nvmlDeviceGetName.return_value = "NVIDIA Unknown Model"
            
            result = estimate_processing_time(300.0)
            self.assertEqual(result, 0.0)
        
        # Test with no GPU
        sysmod.gpu_available = False
        result = estimate_processing_time(300.0)
        self.assertEqual(result, 0.0)

    def test_load_whisper_model_with_warning_suppression(self):
        """Test Whisper model loading with warning suppression"""
        # Mock whisper module to avoid actual import
        mock_whisper = MagicMock()
        mock_model = MagicMock()
        mock_model.to.return_value = mock_model
        mock_whisper.load_model.return_value = mock_model
        
        with patch.dict('sys.modules', {'whisper': mock_whisper}):
            # Test with CUDA available
            sysmod.cuda_available = True
            result = load_whisper_model_with_warning_suppression("large", "cuda")
            self.assertEqual(result, mock_model)
            mock_whisper.load_model.assert_called_with("large")
            mock_model.to.assert_called_once_with("cuda")
            
            # Test with CUDA not available
            mock_whisper.load_model.reset_mock()
            mock_model.reset_mock()
            sysmod.cuda_available = False
            result = load_whisper_model_with_warning_suppression("large", "cuda")
            self.assertEqual(result, mock_model)
            mock_whisper.load_model.assert_called_with("large")
            # Should move to CPU since CUDA is not available
            mock_model.to.assert_called_once_with("cpu")

    def test_process_audio_video_files(self):
        """Test audio/video file processing wrapper function"""
        test_files = ["/path/to/audio.mp3", "/path/to/video.mp4"]
        mock_model = MagicMock()
        
        # Set up module globals
        sysmod.ffprobe_available = True
        sysmod.gpu_available     = True
        sysmod.pynvml_available  = True
        sysmod.cuda_available    = True
        
        # Mock dependencies
        class FixedDatetime(datetime.datetime):
            @classmethod
            def now(cls, tz=None):
                return cls(2024, 1, 1, 12, 0, 0, tzinfo=tz)

        with patch('textify.media.get_media_duration', return_value=120.0), \
             patch('textify.system.pynvml') as mock_pynvml, \
             patch('builtins.open', mock_open()), \
             patch('textify.media.time.time', return_value=1000.0), \
             patch('textify.media.datetime.datetime', FixedDatetime):
            
            # Setup GPU mock
            mock_handle = MagicMock()
            mock_pynvml.nvmlDeviceGetHandleByIndex.return_value = mock_handle
            mock_pynvml.nvmlDeviceGetName.return_value = "NVIDIA GeForce RTX 4070"
            
            # Setup model mock
            mock_model.transcribe.return_value = {"text": "Test transcription"}
            
            # Test with files
            process_audio_video_files(
                files=test_files,
                model=mock_model,
                language="English",
                gpu_threshold=20,
                device="cuda",
                ignore_gpu_threshold=False
            )
            
            # Verify model was called for each file
            self.assertEqual(mock_model.transcribe.call_count, 2)
        
        # Test with no files
        with patch('textify.media.logging') as mock_logging:
            process_audio_video_files(
                files=[],
                model=mock_model,
                language="English",
                gpu_threshold=20,
                device="cuda",
                ignore_gpu_threshold=False
            )
            
            # Should log that no files to process
            mock_logging.info.assert_called_with("No audio/video files to process.")


if __name__ == '__main__':
    unittest.main()

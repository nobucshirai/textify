import unittest
from unittest.mock import patch, MagicMock, mock_open
import os
import sys
import tempfile

# Add the parent directory to the path so we can import textify
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from textify.core import main, setup_logging


class TestCore(unittest.TestCase):

    def setUp(self):
        # Create a temporary directory for test files
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        # Clean up temporary directory
        import shutil

        shutil.rmtree(self.temp_dir)

    @patch("textify.system.initialize_system_checks")
    @patch("textify.core.parse_arguments")
    @patch("textify.core.categorize_files")
    @patch("textify.core.process_document_files")
    @patch("textify.core.process_audio_video_files")
    @patch("textify.core.get_eligible_files")
    @patch("textify.core.load_whisper_model_with_warning_suppression")
    @patch("textify.system.psutil")
    @patch("textify.system.pynvml")
    @patch("textify.system.monitor_resources")
    def test_main(
        self,
        mock_monitor_resources,
        mock_pynvml,
        mock_psutil,
        mock_load_model,
        mock_get_files,
        mock_process_audio,
        mock_process_docs,
        mock_categorize,
        mock_parse_args,
        mock_init_system,
    ):
        """Test the main function orchestrates file processing correctly"""

        # Mock system globals (live flags now live in textify.system)
        with patch("textify.system.cuda_available", True), patch(
            "textify.system.gpu_available", True
        ), patch("textify.system.ffprobe_available", True), patch(
            "textify.system.psutil_available", True
        ):

            # Mock psutil for CPU usage
            mock_psutil.cpu_percent.return_value = 25.0

            # Mock parse_arguments return value
            mock_args = MagicMock()
            mock_args.files = ["test1.mp3", "test2.pdf"]
            mock_args.input_dir = None
            mock_args.model = "tiny"
            mock_args.language = "Japanese"
            mock_args.device = "cpu"  # Use CPU to avoid CUDA issues
            mock_args.log_file = os.path.join(self.temp_dir, "test.log")
            mock_args.monitoring_interval = 10
            mock_args.gpu_threshold = 80
            mock_args.ignore_gpu_threshold = False
            mock_args.verbose = False
            mock_args.watch = False
            mock_parse_args.return_value = mock_args

            # Mock get_eligible_files return value
            mock_get_files.return_value = ["test1.mp3", "test2.pdf"]

            # Mock categorize_files return value
            mock_categorize.return_value = (["test1.mp3"], ["test2.pdf"])

            # Mock model loading
            mock_model = MagicMock()
            mock_load_model.return_value = mock_model

            # Test successful execution
            with patch("textify.core.os.path.isdir", return_value=False), patch(
                "textify.core.os.path.exists", return_value=False
            ), patch("textify.core.open", mock_open()), patch(
                "textify.core.threading.Thread"
            ) as mock_thread, patch(
                "textify.system.get_gpu_info", return_value={}
            ), patch(
                "textify.core.time.time", return_value=1000.0
            ), patch(
                "builtins.print"
            ):

                # Configure mock thread
                mock_thread_instance = MagicMock()
                mock_thread.return_value = mock_thread_instance

                # Run main function
                main()

                # Verify system initialization was called
                mock_init_system.assert_called_once()

                # Verify categorization was called
                mock_categorize.assert_called_once()

                # Verify process functions were called
                mock_process_audio.assert_called_once()
                mock_process_docs.assert_called_once()

                # Verify thread was started and joined
                mock_thread_instance.start.assert_called_once()
                mock_thread_instance.join.assert_called_once()

    @patch("textify.system.initialize_system_checks")
    @patch("textify.core.parse_arguments")
    @patch("textify.core.categorize_files")
    @patch("textify.core.process_document_files")
    @patch("textify.core.process_audio_video_files")
    @patch("textify.core.get_eligible_files")
    @patch("textify.core.load_whisper_model_with_warning_suppression")
    @patch("textify.system.psutil")
    @patch("textify.system.pynvml")
    @patch("textify.system.monitor_resources")
    def test_main_falls_back_to_cpu(
        self,
        mock_monitor_resources,
        mock_pynvml,
        mock_psutil,
        mock_load_model,
        mock_get_files,
        mock_process_audio,
        mock_process_docs,
        mock_categorize,
        mock_parse_args,
        mock_init_system,
    ):
        """CUDA request should fall back to CPU when unavailable"""

        with patch("textify.system.cuda_available", False), patch(
            "textify.system.gpu_available", False
        ), patch("textify.system.ffprobe_available", True), patch(
            "textify.system.psutil_available", True
        ):

            mock_psutil.cpu_percent.return_value = 0.0

            mock_args = MagicMock()
            mock_args.files = ["test1.mp3"]
            mock_args.input_dir = None
            mock_args.model = "tiny"
            mock_args.language = "English"
            mock_args.device = "cuda"  # request CUDA
            mock_args.log_file = os.path.join(self.temp_dir, "test.log")
            mock_args.monitoring_interval = 10
            mock_args.gpu_threshold = 20
            mock_args.ignore_gpu_threshold = False
            mock_args.verbose = False
            mock_args.watch = False
            mock_parse_args.return_value = mock_args

            mock_get_files.return_value = ["test1.mp3"]
            mock_categorize.return_value = (["test1.mp3"], [])

            mock_model = MagicMock()
            mock_load_model.return_value = mock_model

            with patch("textify.core.os.path.isdir", return_value=False), patch(
                "textify.core.os.path.exists", return_value=False
            ), patch("textify.core.open", mock_open()), patch(
                "textify.core.threading.Thread"
            ) as mock_thread, patch(
                "textify.system.get_gpu_info", return_value={}
            ), patch(
                "textify.core.time.time", return_value=1000.0
            ), patch(
                "builtins.print"
            ):

                mock_thread_instance = MagicMock()
                mock_thread.return_value = mock_thread_instance

                main()

                mock_load_model.assert_called_with("tiny", "cpu", False)

    def test_setup_logging(self):
        """Test that setup_logging configures logging correctly"""
        with patch("textify.core.logging.getLogger") as mock_get_logger, patch(
            "textify.core.logging.StreamHandler"
        ) as mock_stream_handler, patch(
            "textify.core.logging.Formatter"
        ) as mock_formatter, patch(
            "textify.core.has_handler_of_type", return_value=False
        ):

            mock_logger = MagicMock()
            mock_get_logger.return_value = mock_logger

            logger, log_format = setup_logging()

            # Verify that getLogger was called with no arguments (gets root logger)
            mock_get_logger.assert_called_once_with()

            # Verify handler and formatter were created
            mock_stream_handler.assert_called_once()
            mock_formatter.assert_called_once()

            # Verify the returned values
            self.assertEqual(logger, mock_logger)
            self.assertIsNotNone(log_format)


if __name__ == "__main__":
    unittest.main()

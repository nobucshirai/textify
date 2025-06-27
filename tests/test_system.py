import unittest
from unittest.mock import patch, MagicMock
import os
import sys
import tempfile
import threading
import time
import io
import logging

# Add the parent directory to the path so we can import textify
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from textify.system import get_gpu_info, monitor_resources


class TestSystem(unittest.TestCase):
    
    def setUp(self):
        # Create a temporary directory for test files
        self.temp_dir = tempfile.mkdtemp()
        
    def tearDown(self):
        # Clean up temporary directory
        import shutil
        shutil.rmtree(self.temp_dir)

    @patch('textify.system.pynvml')
    def test_get_gpu_info(self, mock_pynvml):
        """Test GPU information retrieval"""
        # Configure the mock
        mock_handle = MagicMock()
        mock_pynvml.nvmlDeviceGetCount.return_value = 1
        mock_pynvml.nvmlDeviceGetHandleByIndex.return_value = mock_handle
        mock_pynvml.nvmlDeviceGetName.return_value = 'NVIDIA GeForce RTX 4070'
        
        mem_info = MagicMock()
        mem_info.total = 8589934592  # 8 GB in bytes
        mem_info.used = 1073741824   # 1 GB in bytes
        mem_info.free = 7516192768   # 7 GB in bytes
        mock_pynvml.nvmlDeviceGetMemoryInfo.return_value = mem_info
        
        mock_pynvml.nvmlSystemGetDriverVersion.return_value = '535.104.05'
        
        util_info = MagicMock()
        util_info.gpu = 5
        util_info.memory = 10
        mock_pynvml.nvmlDeviceGetUtilizationRates.return_value = util_info
        
        mock_pynvml.nvmlDeviceGetPowerUsage.return_value = 30500  # 30.5 W in milliwatts
        mock_pynvml.nvmlDeviceGetTemperature.return_value = 45
        
        # Set global flags - need to access them through the module
        import textify.system as system_module
        system_module.pynvml_available = True
        system_module.gpu_available = True
        
        # Call the function
        result = get_gpu_info()
        
        # Verify the result
        self.assertEqual(result['device_count'], 1)
        self.assertEqual(result['name'], 'NVIDIA GeForce RTX 4070')
        self.assertAlmostEqual(result['memory_total'], 8192, delta=1)
        self.assertAlmostEqual(result['memory_used'], 1024, delta=1)
        self.assertAlmostEqual(result['memory_free'], 7168, delta=1)
        self.assertEqual(result['driver_version'], '535.104.05')
        self.assertEqual(result['gpu_util'], 5)
        self.assertEqual(result['memory_util'], 10)
        self.assertEqual(result['power_usage'], 30.5)
        self.assertEqual(result['temperature'], 45)

    @patch('textify.system.pynvml')
    @patch('textify.system.psutil')
    @patch('textify.system.time.time')
    def test_monitor_resources(self, mock_time, mock_psutil, mock_pynvml):
        """Test resource monitoring functionality"""
        # Setup logging capture
        log_stream = io.StringIO()
        handler = logging.StreamHandler(log_stream)
        logger = logging.getLogger()
        logger.setLevel(logging.INFO)  # Ensure INFO logs are captured
        logger.addHandler(handler)
        
        # Configure mocks
        time_sequence = [0, 5, 10]
        mock_time.side_effect = time_sequence
        
        mock_psutil.cpu_percent.return_value = 50
        
        mock_handle = MagicMock()
        mock_pynvml.nvmlDeviceGetHandleByIndex.return_value = mock_handle
        
        util_rates = MagicMock()
        util_rates.gpu = 30
        util_rates.memory = 15
        mock_pynvml.nvmlDeviceGetUtilizationRates.return_value = util_rates
        
        mock_pynvml.nvmlDeviceGetPowerUsage.return_value = 40000  # 40W
        
        # Set global flags
        import textify.system as system_module
        system_module.gpu_available = True
        system_module.pynvml_available = True
        system_module.psutil_available = True
        
        # Create a stop event
        stop_event = threading.Event()
        
        # Call function in a thread since it runs until stop_event is set
        def run_monitor():
            monitor_resources(stop_event, 1)
        
        monitor_thread = threading.Thread(target=run_monitor)
        monitor_thread.daemon = True
        monitor_thread.start()
        
        # Let it run for a moment
        time.sleep(0.1)
        stop_event.set()
        monitor_thread.join()
        
        # Check log output
        log_output = log_stream.getvalue()
        self.assertIn("Resource usage statistics", log_output)
        
        # Clean up
        logger.removeHandler(handler)


if __name__ == '__main__':
    unittest.main()

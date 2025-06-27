import unittest
from unittest.mock import patch
import os
import sys
import argparse

# Add the parent directory to the path so we can import textify
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from textify.cli import parse_arguments


class TestCLI(unittest.TestCase):

    def test_parse_arguments_validation(self):
        """Test that parse_arguments properly validates input combinations"""
        
        # Test with neither files nor input_dir
        with patch('sys.argv', ['textify']):
            with self.assertRaises(SystemExit):
                parse_arguments()
        
        # Test with both files and input_dir (should fail)
        with patch('sys.argv', ['textify', 'file1.mp3', '--input-dir', '/path']):
            with self.assertRaises(SystemExit):
                parse_arguments()

        # --watch without --input-dir should fail
        with patch('sys.argv', ['textify', '--watch', 'file1.mp3']):
            with self.assertRaises(SystemExit):
                parse_arguments()
        
        # Test with files only (should work)
        with patch('sys.argv', ['textify', 'file1.mp3', 'file2.wav']):
            args = parse_arguments()
            self.assertEqual(args.files, ['file1.mp3', 'file2.wav'])
            self.assertIsNone(args.input_dir)
        
        # Test with input_dir only (should work)
        with patch('sys.argv', ['textify', '--input-dir', '/path/to/dir']):
            args = parse_arguments()
            self.assertEqual(args.files, [])
            self.assertEqual(args.input_dir, '/path/to/dir')

    def test_parse_arguments_defaults(self):
        """Test default argument values"""
        with patch('sys.argv', ['textify', '--input-dir', '/test']):
            args = parse_arguments()
            
            # Check defaults
            self.assertEqual(args.model, 'large')
            self.assertEqual(args.language, 'Japanese')
            self.assertEqual(args.device, 'cuda')
            self.assertEqual(args.gpu_threshold, 20)
            self.assertEqual(args.monitoring_interval, 10)
            self.assertFalse(args.ignore_gpu_threshold)
            self.assertFalse(args.verbose)
            self.assertEqual(args.log_file, 'batch_process.log')
            self.assertFalse(args.watch)

    def test_parse_arguments_custom_values(self):
        """Test custom argument values"""
        with patch('sys.argv', [
            'textify',
            '--input-dir', '/test',
            '--model', 'tiny',
            '--language', 'English',
            '--device', 'cpu',
            '--gpu-threshold', '50',
            '--monitoring-interval', '10',
            '--ignore-gpu-threshold',
            '--verbose',
            '--log-file', 'test.log',
            '--watch'
        ]):
            args = parse_arguments()
            
            # Check custom values
            self.assertEqual(args.model, 'tiny')
            self.assertEqual(args.language, 'English')
            self.assertEqual(args.device, 'cpu')
            self.assertEqual(args.gpu_threshold, 50)
            self.assertEqual(args.monitoring_interval, 10)
            self.assertTrue(args.ignore_gpu_threshold)
            self.assertTrue(args.verbose)
            self.assertEqual(args.log_file, 'test.log')
            self.assertTrue(args.watch)


if __name__ == '__main__':
    unittest.main()

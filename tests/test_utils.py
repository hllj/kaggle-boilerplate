import os
import sys
import unittest
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils import set_seed, get_device, EarlyStopping


class TestUtils(unittest.TestCase):
    """Test cases for utility functions."""

    def test_set_seed(self):
        """Test set_seed function for reproducibility."""
        # Set a specific seed
        seed = 42
        set_seed(seed)
        
        # Generate random values
        np_rand1 = np.random.rand(5)
        torch_rand1 = torch.rand(5)
        
        # Set the same seed again
        set_seed(seed)
        
        # Generate random values again
        np_rand2 = np.random.rand(5)
        torch_rand2 = torch.rand(5)
        
        # Check that the random values are the same
        np.testing.assert_array_equal(np_rand1, np_rand2)
        torch.testing.assert_close(torch_rand1, torch_rand2)

    def test_get_device(self):
        """Test get_device function."""
        device = get_device()
        self.assertIsInstance(device, torch.device)
        
        # The device should be either 'cpu' or 'cuda'
        self.assertIn(device.type, ['cpu', 'cuda'])

    def test_early_stopping_min_mode(self):
        """Test EarlyStopping in 'min' mode."""
        # Create EarlyStopping object with min mode (default)
        early_stopping = EarlyStopping(patience=2, min_delta=0.1)
        
        # First call should always return False and set best_score
        self.assertFalse(early_stopping(1.0))
        self.assertEqual(early_stopping.best_score, 1.0)
        
        # Better score (lower in min mode)
        self.assertFalse(early_stopping(0.8))
        self.assertEqual(early_stopping.best_score, 0.8)
        self.assertEqual(early_stopping.counter, 0)
        
        # Worse score, counter should increase
        self.assertFalse(early_stopping(0.9))
        self.assertEqual(early_stopping.counter, 1)
        
        # Another worse score, counter should reach patience
        self.assertTrue(early_stopping(0.9))
        self.assertEqual(early_stopping.counter, 2)
        self.assertTrue(early_stopping.early_stop)

    def test_early_stopping_max_mode(self):
        """Test EarlyStopping in 'max' mode."""
        # Create EarlyStopping object with max mode
        early_stopping = EarlyStopping(patience=2, min_delta=0.1, mode='max')
        
        # First call should always return False and set best_score
        self.assertFalse(early_stopping(0.5))
        self.assertEqual(early_stopping.best_score, 0.5)
        
        # Better score (higher in max mode)
        self.assertFalse(early_stopping(0.7))
        self.assertEqual(early_stopping.best_score, 0.7)
        self.assertEqual(early_stopping.counter, 0)
        
        # Worse score, counter should increase
        self.assertFalse(early_stopping(0.6))
        self.assertEqual(early_stopping.counter, 1)
        
        # Another worse score, counter should reach patience
        self.assertTrue(early_stopping(0.6))
        self.assertEqual(early_stopping.counter, 2)
        self.assertTrue(early_stopping.early_stop)


if __name__ == "__main__":
    unittest.main() 
import os
import sys
import unittest
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import Config
from src.models.model import TabularMLP, ResNetClassifier, get_model


class TestModels(unittest.TestCase):
    """Test cases for model implementations."""

    def setUp(self):
        """Set up test fixtures."""
        # Create a test configuration
        self.config_dict = {
            "model": {
                "name": "tabular_mlp",
                "input_dim": 10,
                "hidden_dims": [64, 32],
                "dropout": 0.5,
                "num_classes": 1,
                "batch_norm": True,
                "activation": "relu",
                "backbone": "resnet18",
                "pretrained": True
            }
        }
        self.cfg = OmegaConf.create(self.config_dict)

    def test_tabular_mlp(self):
        """Test TabularMLP model."""
        # Initialize model
        model = TabularMLP(
            input_dim=10,
            hidden_dims=[64, 32],
            output_dim=1,
            dropout=0.5
        )

        # Check model structure
        self.assertIsInstance(model, torch.nn.Module)

        # Test forward pass
        x = torch.randn(32, 10)  # Batch size of 32, 10 features
        output = model(x)
        self.assertEqual(output.shape, (32, 1))

    def test_resnet_classifier(self):
        """Test ResNetClassifier model."""
        # Skip test if no GPU available and test is running in CI environment
        if not torch.cuda.is_available() and os.environ.get('CI') == 'true':
            self.skipTest("Skipping GPU test in CI environment without GPU")

        # Initialize model
        model = ResNetClassifier(
            num_classes=1,
            backbone="resnet18",
            pretrained=False  # Set to False for faster testing
        )

        # Check model structure
        self.assertIsInstance(model, torch.nn.Module)

        # Test forward pass
        x = torch.randn(2, 3, 224, 224)  # Batch size of 2, 3 channels, 224x224 images
        output = model(x)
        self.assertEqual(output.shape, (2, 1))

    def test_get_model(self):
        """Test get_model factory function."""
        # Test tabular model
        self.cfg.model.name = "tabular"
        model = get_model(self.cfg)
        self.assertIsInstance(model, TabularMLP)

        # Test that unavailable model raises error
        self.cfg.model.name = "invalid_model"
        with self.assertRaises(ValueError):
            get_model(self.cfg)


if __name__ == "__main__":
    unittest.main() 
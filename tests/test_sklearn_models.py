import os
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from omegaconf import OmegaConf

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.sklearn_models import (
    get_sklearn_model, 
    SklearnModel, 
    LinearModel, 
    TreeModel, 
    BoostingModel
)


class TestSklearnModels(unittest.TestCase):
    """Test cases for scikit-learn model implementations."""

    def setUp(self):
        """Set up test fixtures."""
        # Create sample data
        self.X = np.random.randn(100, 5)
        self.y_classification = (np.random.randn(100) > 0).astype(int)
        self.y_regression = np.random.randn(100)
        self.feature_names = [f"feature_{i}" for i in range(5)]
        
        # Set random seed for reproducibility
        np.random.seed(42)

    def test_linear_classification_model(self):
        """Test linear classification model."""
        config = {
            'name': 'logistic',
            'class': 'linear',
            'type': 'logistic',
            'C': 1.0,
            'penalty': 'l2',
            'solver': 'liblinear',
            'max_iter': 100,
            'random_state': 42
        }
        
        model = get_sklearn_model(config)
        
        # Check model type
        self.assertIsInstance(model, LinearModel)
        self.assertTrue(model.is_classifier)
        
        # Test fit method
        model.fit(self.X, self.y_classification, feature_names=self.feature_names)
        self.assertTrue(model.is_fitted)
        
        # Test predict method
        y_pred = model.predict(self.X)
        self.assertEqual(y_pred.shape, (100,))
        self.assertTrue(np.all((y_pred == 0) | (y_pred == 1)))
        
        # Test predict_proba method
        y_prob = model.predict_proba(self.X)
        self.assertEqual(y_prob.shape, (100, 2))
        self.assertTrue(np.all((y_prob >= 0) & (y_prob <= 1)))

    def test_tree_regression_model(self):
        """Test tree regression model."""
        config = {
            'name': 'rf_regressor',
            'class': 'tree',
            'type': 'random_forest_regressor',
            'n_estimators': 10,
            'max_depth': 3,
            'random_state': 42
        }
        
        model = get_sklearn_model(config)
        
        # Check model type
        self.assertIsInstance(model, TreeModel)
        self.assertFalse(model.is_classifier)
        
        # Test fit method
        model.fit(self.X, self.y_regression, feature_names=self.feature_names)
        self.assertTrue(model.is_fitted)
        
        # Test predict method
        y_pred = model.predict(self.X)
        self.assertEqual(y_pred.shape, (100,))
        
        # Test feature importance
        importance = model.get_feature_importance()
        self.assertIsNotNone(importance)
        self.assertEqual(len(importance), 5)
        
        # Ensure all feature names are in the importance dict
        for feature in self.feature_names:
            self.assertIn(feature, importance)

    def test_save_and_load(self):
        """Test saving and loading models."""
        config = {
            'name': 'rf_classifier',
            'class': 'tree',
            'type': 'random_forest_classifier',
            'n_estimators': 10,
            'max_depth': 3,
            'random_state': 42
        }
        
        model = get_sklearn_model(config)
        model.fit(self.X, self.y_classification, feature_names=self.feature_names)
        
        # Save model
        test_dir = Path(os.path.dirname(os.path.abspath(__file__)))
        model_path = test_dir / "test_model.pkl"
        model.save(str(model_path))
        
        # Load model
        loaded_model = SklearnModel.load(str(model_path))
        
        # Check loaded model
        self.assertIsInstance(loaded_model, TreeModel)
        self.assertTrue(loaded_model.is_classifier)
        self.assertTrue(loaded_model.is_fitted)
        
        # Test predictions
        y_pred_original = model.predict(self.X)
        y_pred_loaded = loaded_model.predict(self.X)
        np.testing.assert_array_equal(y_pred_original, y_pred_loaded)
        
        # Clean up
        if model_path.exists():
            os.remove(model_path)

    def test_ensemble_model(self):
        """Test ensemble model with multiple estimators."""
        config = {
            'name': 'voting_classifier',
            'class': 'ensemble',
            'type': 'voting_classifier',
            'voting': 'hard',
            'estimators': [
                {
                    'name': 'rf',
                    'type': 'tree',
                    'params': {
                        'type': 'random_forest_classifier',
                        'n_estimators': 10,
                        'max_depth': 3,
                        'random_state': 42
                    }
                },
                {
                    'name': 'lr',
                    'type': 'linear',
                    'params': {
                        'type': 'logistic',
                        'C': 1.0,
                        'random_state': 42
                    }
                }
            ],
            'n_jobs': 1
        }
        
        # This test might be skipped if dependencies are not installed
        try:
            model = get_sklearn_model(config)
            
            # Check model type
            self.assertIsInstance(model, SklearnModel)
            self.assertTrue(model.is_classifier)
            
            # Test fit method
            model.fit(self.X, self.y_classification)
            self.assertTrue(model.is_fitted)
            
            # Test predict method
            y_pred = model.predict(self.X)
            self.assertEqual(y_pred.shape, (100,))
            
        except ImportError:
            self.skipTest("Dependencies for ensemble models not installed")


if __name__ == "__main__":
    unittest.main() 
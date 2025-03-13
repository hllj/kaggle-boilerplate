"""
Inference script for generating predictions.
"""
import argparse
import logging
import os
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

import transformers
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from lightning.pytorch import LightningModule

from src.config import Config, get_config
from src.dataset.load_data import create_data_loaders, preprocess_data
from src.dataset.feature_engineering import feature_engineering_pipeline
from src.models.model import get_model
from src.utils import get_device, load_checkpoint, load_data, setup_logging
from src.models.sklearn_models import SklearnModel


def inference_pytorch(model: nn.Module, 
                    test_loader: DataLoader, 
                    device: torch.device,
                    cfg: Config) -> np.ndarray:
    """
    Generate predictions using a vanilla PyTorch model.
    
    Args:
        model: PyTorch model
        test_loader: Test data loader
        device: PyTorch device
        cfg: Configuration
        
    Returns:
        Numpy array of predictions
    """
    model.eval()
    all_preds = []
    
    # Use test-time augmentation if enabled
    n_tta = cfg.inference.get('n_tta', 1) if cfg.inference.tta else 1
    
    with torch.no_grad():
        for _ in range(n_tta):
            batch_preds = []
            for batch in tqdm(test_loader, desc="Inference"):
                features = batch['features'].to(device)
                outputs = model(features)
                
                # Handle different output types
                if isinstance(outputs, tuple):
                    outputs = outputs[0]
                
                # Convert outputs to predictions based on problem type
                if cfg.model.num_classes == 1:  # Binary classification or regression
                    # Check if regression or binary classification
                    if cfg.evaluation.metric.lower() in ['rmse', 'mae', 'mse']:
                        preds = outputs.cpu().numpy()  # Regression
                    else:
                        preds = torch.sigmoid(outputs).cpu().numpy()  # Binary classification
                else:  # Multi-class classification
                    preds = torch.softmax(outputs, dim=1).cpu().numpy()
                
                batch_preds.append(preds)
            
            # Concatenate batch predictions
            epoch_preds = np.concatenate(batch_preds, axis=0)
            all_preds.append(epoch_preds)
    
    # Average predictions from TTA
    final_preds = np.mean(all_preds, axis=0)
    
    return final_preds


def inference_pytorch_lightning(model: LightningModule, 
                              test_loader: DataLoader, 
                              device: torch.device,
                              cfg: Config) -> np.ndarray:
    """
    Generate predictions using a PyTorch Lightning model.
    
    Args:
        model: PyTorch Lightning model
        test_loader: Test data loader
        device: PyTorch device
        cfg: Configuration
        
    Returns:
        Numpy array of predictions
    """
    # Set model to evaluation mode
    model.eval()
    
    # Transfer to device
    model = model.to(device)
    
    # Use the base PyTorch inference function
    return inference_pytorch(model, test_loader, device, cfg)


def inference_huggingface(model_path: str, 
                        test_df: pd.DataFrame, 
                        cfg: Config,
                        device: torch.device) -> np.ndarray:
    """
    Generate predictions using a HuggingFace transformer model.
    
    Args:
        model_path: Path to saved model
        test_df: Test dataframe
        cfg: Configuration
        device: PyTorch device
        
    Returns:
        Numpy array of predictions
    """
    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    model = model.to(device)
    model.eval()
    
    # Get column names
    text_col = cfg.data.features.get('text_column', 'text')
    
    # Create test dataset
    batch_size = cfg.inference.batch_size
    
    # Process test data in batches
    all_preds = []
    
    for i in tqdm(range(0, len(test_df), batch_size), desc="Inference"):
        batch_df = test_df.iloc[i:i+batch_size]
        
        # Tokenize batch
        inputs = tokenizer(
            batch_df[text_col].tolist(),
            padding="max_length",
            truncation=True,
            max_length=cfg.model.get('max_length', 128),
            return_tensors="pt"
        )
        
        # Move inputs to device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # Generate predictions
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            
            # Convert logits to predictions based on problem type
            if model.config.problem_type == "regression":
                preds = logits.cpu().numpy()
            elif model.config.num_labels == 1:
                preds = torch.sigmoid(logits).cpu().numpy()
            else:
                preds = torch.softmax(logits, dim=1).cpu().numpy()
            
            all_preds.append(preds)
    
    # Concatenate batch predictions
    final_preds = np.concatenate(all_preds, axis=0)
    
    return final_preds


def ensemble_predictions(predictions: List[np.ndarray], 
                       weights: Optional[List[float]] = None) -> np.ndarray:
    """
    Ensemble multiple predictions.
    
    Args:
        predictions: List of predictions to ensemble
        weights: Optional weights for each prediction
        
    Returns:
        Ensembled predictions
    """
    if weights is None:
        weights = [1.0] * len(predictions)
    
    # Normalize weights
    weights = np.array(weights) / np.sum(weights)
    
    # Apply weights and sum
    weighted_preds = [pred * weight for pred, weight in zip(predictions, weights)]
    final_preds = np.sum(weighted_preds, axis=0)
    
    return final_preds


def load_model(cfg: Config, 
              checkpoint_path: str, 
              device: torch.device) -> nn.Module:
    """
    Load a trained model from checkpoint.
    
    Args:
        cfg: Configuration
        checkpoint_path: Path to checkpoint
        device: PyTorch device
        
    Returns:
        Loaded model
    """
    framework = cfg.training.framework.lower()
    
    if framework == 'pytorch':
        # Initialize model architecture
        model = get_model(cfg)
        
        # Load weights
        model, _ = load_checkpoint(checkpoint_path, model)
        
        # Move to device
        model = model.to(device)
        
    elif framework == 'lightning':
        # Import LightningModel class
        from src.models.train_pl import LightningModel
        
        # Initialize model architecture
        base_model = get_model(cfg)
        
        # Create lightning module
        model = LightningModel.load_from_checkpoint(
            checkpoint_path=checkpoint_path,
            model=base_model,
            cfg=cfg
        )
        
        # Move to device
        model = model.to(device)
        
    elif framework == 'huggingface':
        # For HuggingFace, we don't load the model here
        # It will be loaded separately in the inference function
        model = None
    
    else:
        raise ValueError(f"Framework {framework} not supported")
    
    return model


def inference_sklearn(model: SklearnModel, 
                    X_test: np.ndarray, 
                    cfg: Config) -> np.ndarray:
    """
    Generate predictions using a scikit-learn model.
    
    Args:
        model: Scikit-learn model
        X_test: Test features
        cfg: Configuration
        
    Returns:
        Array of predictions
    """
    # Get prediction type from config
    prediction_type = cfg.inference.get('prediction_type', 'proba')
    
    # Check if model is a classifier that supports probability predictions
    is_classifier = hasattr(model, 'is_classifier') and model.is_classifier
    supports_proba = is_classifier and hasattr(model, 'predict_proba')
    
    # Generate predictions
    if is_classifier and supports_proba and prediction_type == 'proba':
        # Get probability predictions
        predictions = model.predict_proba(X_test)
        
        # For binary classification, return probability of positive class
        if predictions.shape[1] == 2:
            predictions = predictions[:, 1]
    else:
        # Get class/value predictions
        predictions = model.predict(X_test)
    
    return predictions


def load_sklearn_model(cfg: Config, 
                     model_path: str) -> SklearnModel:
    """
    Load a scikit-learn model from a file.
    
    Args:
        cfg: Configuration
        model_path: Path to the model file
        
    Returns:
        Loaded scikit-learn model
    """
    model = SklearnModel.load(model_path)
    return model


def generate_predictions(cfg: Config, logger: logging.Logger) -> pd.DataFrame:
    """
    Generate predictions using a trained model.
    
    Args:
        cfg: Configuration
        logger: Logger
        
    Returns:
        DataFrame with predictions
    """
    # Get device
    device = get_device()
    logger.info(f"Using device: {device}")
    
    # Load data
    logger.info("Loading data...")
    train_df, test_df, sample_submission_df = load_data(cfg)
    
    # Determine model type
    model_type = cfg.training.framework
    
    if model_type == 'sklearn':
        # For scikit-learn models, load the model directly
        model_path = os.path.join(cfg.paths.model_dir, f"{cfg.model.name}_sklearn.pkl")
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found at {model_path}")
        
        # Load the model
        logger.info(f"Loading scikit-learn model from {model_path}")
        model = load_sklearn_model(cfg, model_path)
        
        # Preprocess data (without target for test set)
        X_train, _, X_test = preprocess_data(train_df, test_df, cfg)
        
        # Convert to numpy if DataFrame
        if hasattr(X_test, 'values'):
            X_test = X_test.values
        
        # Generate predictions
        logger.info("Generating predictions...")
        predictions = inference_sklearn(model, X_test, cfg)
    elif model_type == 'pytorch':
        # Feature engineering
        logger.info("Applying feature engineering...")
        train_df, test_df = feature_engineering_pipeline(train_df, test_df, cfg)
        
        # Preprocess data
        logger.info("Preprocessing data...")
        X_train, y_train, X_test = preprocess_data(train_df, test_df, cfg)
        
        # Create data loaders
        logger.info("Creating data loaders...")
        data_loaders = create_data_loaders(X_train, y_train, X_test, cfg, validation_split=False)
        
        # Determine checkpoint path
        if cfg.inference.checkpoint == 'best':
            checkpoint_path = os.path.join(cfg.paths.model_dir, "best_model.pt")
        else:
            checkpoint_path = os.path.join(cfg.paths.model_dir, "latest_model.pt")
        
        # Load model
        logger.info(f"Loading model from {checkpoint_path}...")
        model = load_model(cfg, checkpoint_path, device)
        
        # Generate predictions
        logger.info("Generating predictions...")
        predictions = inference_pytorch(model, data_loaders['test'], device, cfg)
    elif model_type == 'lightning':
        # Feature engineering
        logger.info("Applying feature engineering...")
        train_df, test_df = feature_engineering_pipeline(train_df, test_df, cfg)
        
        # Preprocess data
        logger.info("Preprocessing data...")
        X_train, y_train, X_test = preprocess_data(train_df, test_df, cfg)
        
        # Create data loaders
        logger.info("Creating data loaders...")
        data_loaders = create_data_loaders(X_train, y_train, X_test, cfg, validation_split=False)
        
        # Determine checkpoint path
        if cfg.inference.checkpoint == 'best':
            checkpoint_path = os.path.join(cfg.paths.model_dir, "best_model.pt")
        else:
            checkpoint_path = os.path.join(cfg.paths.model_dir, "latest_model.pt")
        
        # Load model
        logger.info(f"Loading model from {checkpoint_path}...")
        model = load_model(cfg, checkpoint_path, device)
        
        # Generate predictions
        logger.info("Generating predictions...")
        predictions = inference_pytorch_lightning(model, data_loaders['test'], device, cfg)
    elif model_type == 'huggingface':
        # Determine model path
        model_path = os.path.join(cfg.paths.model_dir, "best_model")
        
        # Generate predictions
        logger.info(f"Generating predictions using HuggingFace model from {model_path}...")
        predictions = inference_huggingface(model_path, test_df, cfg, device)
    else:
        raise ValueError(f"Unsupported framework: {model_type}")
    
    # Create submission DataFrame
    logger.info("Creating submission dataframe...")
    submission_df = create_submission_df(predictions, test_df, sample_submission_df, cfg)
    
    return submission_df


def create_submission_df(predictions: np.ndarray, 
                        test_df: pd.DataFrame, 
                        sample_submission_df: pd.DataFrame, 
                        cfg: Config) -> pd.DataFrame:
    """
    Create submission DataFrame.
    
    Args:
        predictions: Model predictions
        test_df: Test dataframe
        sample_submission_df: Sample submission dataframe
        cfg: Configuration
        
    Returns:
        Submission DataFrame
    """
    # Create copy of sample submission dataframe
    submission_df = sample_submission_df.copy()
    
    # Get ID column (first column)
    id_col = submission_df.columns[0]
    
    # Handle different prediction formats
    prediction_cols = submission_df.columns[1:]
    
    # For regression or binary classification with one output
    if len(predictions.shape) == 1 or predictions.shape[1] == 1:
        if len(predictions.shape) > 1:
            predictions = predictions.flatten()
        
        # Fill the first prediction column
        submission_df[prediction_cols[0]] = predictions
    
    # For multi-class classification
    elif len(prediction_cols) == predictions.shape[1]:
        for i, col in enumerate(prediction_cols):
            submission_df[col] = predictions[:, i]
    
    # Unexpected format
    else:
        raise ValueError(
            f"Mismatch between prediction shape {predictions.shape} and "
            f"submission columns {prediction_cols}"
        )
    
    return submission_df


def save_submission(submission_df: pd.DataFrame, cfg: Config) -> str:
    """
    Save submission dataframe to file.
    
    Args:
        submission_df: Submission dataframe
        cfg: Configuration
        
    Returns:
        Path to saved submission file
    """
    # Create submission directory if it doesn't exist
    os.makedirs(cfg.paths.submission_dir, exist_ok=True)
    
    # Create timestamp
    import datetime
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create submission filename
    submission_filepath = os.path.join(
        cfg.paths.submission_dir, 
        f"submission_{cfg.model.name}_{timestamp}.csv"
    )
    
    # Save submission
    submission_df.to_csv(submission_filepath, index=False)
    
    return submission_filepath


def main():
    """Main function."""
    # Parse arguments
    parser = argparse.ArgumentParser(description="Generate predictions")
    parser.add_argument("--config", type=str, default="default", help="Name of config file")
    args = parser.parse_args()
    
    # Load configuration
    cfg = get_config(args.config)
    
    # Set up logging
    logger = setup_logging(cfg)
    
    # Generate predictions
    submission_df = generate_predictions(cfg, logger)
    
    # Save submission
    submission_filepath = save_submission(submission_df, cfg)
    logger.info(f"Submission saved to {submission_filepath}")


if __name__ == "__main__":
    main() 
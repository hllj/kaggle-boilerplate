"""
Training script for scikit-learn models.
"""
import argparse
import logging
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score, KFold, StratifiedKFold
import joblib
import wandb
from omegaconf import DictConfig, OmegaConf

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.config import Config, get_config
from src.utils import set_seed, setup_logging, init_wandb
from src.dataset.load_data import preprocess_data
from src.models.sklearn_models import get_sklearn_model


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, 
                   y_prob: Optional[np.ndarray] = None,
                   is_classifier: bool = True) -> Dict[str, float]:
    """
    Compute evaluation metrics for the model.
    
    Args:
        y_true: True labels or target values
        y_pred: Predicted labels or values
        y_prob: Probability predictions for classification
        is_classifier: Whether the model is a classifier
        
    Returns:
        Dictionary of metric names and values
    """
    metrics = {}
    
    if is_classifier:
        # Classification metrics
        metrics['accuracy'] = accuracy_score(y_true, y_pred)
        
        if len(np.unique(y_true)) == 2:  # Binary classification
            metrics['precision'] = precision_score(y_true, y_pred, average='binary')
            metrics['recall'] = recall_score(y_true, y_pred, average='binary')
            metrics['f1'] = f1_score(y_true, y_pred, average='binary')
            
            if y_prob is not None:
                if y_prob.ndim > 1 and y_prob.shape[1] > 1:
                    # Get probabilities for the positive class
                    prob_pos = y_prob[:, 1]
                else:
                    prob_pos = y_prob
                
                try:
                    metrics['roc_auc'] = roc_auc_score(y_true, prob_pos)
                except:
                    # Handle cases where there's only one class in y_true
                    metrics['roc_auc'] = 0.5
        else:  # Multi-class classification
            metrics['precision_macro'] = precision_score(y_true, y_pred, average='macro')
            metrics['recall_macro'] = recall_score(y_true, y_pred, average='macro')
            metrics['f1_macro'] = f1_score(y_true, y_pred, average='macro')
            
            metrics['precision_weighted'] = precision_score(y_true, y_pred, average='weighted')
            metrics['recall_weighted'] = recall_score(y_true, y_pred, average='weighted')
            metrics['f1_weighted'] = f1_score(y_true, y_pred, average='weighted')
    else:
        # Regression metrics
        metrics['mae'] = mean_absolute_error(y_true, y_pred)
        metrics['mse'] = mean_squared_error(y_true, y_pred)
        metrics['rmse'] = np.sqrt(metrics['mse'])
        metrics['r2'] = r2_score(y_true, y_pred)
    
    return metrics


def run_cross_validation(model, X: np.ndarray, y: np.ndarray, 
                        cfg: Config, logger: logging.Logger) -> Dict[str, float]:
    """
    Run cross-validation for the model.
    
    Args:
        model: Scikit-learn model
        X: Input features
        y: Target values
        cfg: Configuration
        logger: Logger
        
    Returns:
        Dictionary of average metrics across folds
    """
    # Determine if this is a classification or regression problem
    is_classifier = hasattr(model, 'is_classifier') and model.is_classifier
    
    # Choose the right CV splitter
    if is_classifier and len(np.unique(y)) > 1:
        cv = StratifiedKFold(n_splits=cfg.data.num_folds, shuffle=True, random_state=cfg.seed)
    else:
        cv = KFold(n_splits=cfg.data.num_folds, shuffle=True, random_state=cfg.seed)
    
    # Select scoring metric
    if is_classifier:
        scoring = 'accuracy'
    else:
        scoring = 'neg_mean_squared_error'
    
    logger.info(f"Running {cfg.data.num_folds}-fold cross-validation...")
    
    # Store metrics for each fold
    cv_metrics = []
    feature_names = None
    
    # If X is a DataFrame, extract feature names
    if hasattr(X, 'columns'):
        feature_names = X.columns.tolist()
    
    # Manual cross-validation to compute all metrics
    for fold, (train_idx, val_idx) in enumerate(cv.split(X, y)):
        X_train_fold, X_val_fold = X[train_idx], X[val_idx]
        y_train_fold, y_val_fold = y[train_idx], y[val_idx]
        
        # Train model on this fold
        start_time = time.time()
        model.fit(X_train_fold, y_train_fold, feature_names=feature_names)
        train_time = time.time() - start_time
        
        # Make predictions
        y_pred = model.predict(X_val_fold)
        
        # Get probability predictions for classification
        y_prob = None
        if is_classifier and hasattr(model, 'predict_proba'):
            try:
                y_prob = model.predict_proba(X_val_fold)
            except:
                pass
        
        # Compute metrics
        fold_metrics = compute_metrics(y_val_fold, y_pred, y_prob, is_classifier)
        fold_metrics['train_time'] = train_time
        
        # Log metrics for this fold
        logger.info(f"Fold {fold+1}/{cfg.data.num_folds} - "
                   f"Time: {train_time:.2f}s - "
                   f"Metrics: {', '.join([f'{k}: {v:.4f}' for k, v in fold_metrics.items()])}")
        
        cv_metrics.append(fold_metrics)
    
    # Compute average metrics across folds
    avg_metrics = {}
    for metric in cv_metrics[0].keys():
        avg_metrics[metric] = np.mean([m[metric] for m in cv_metrics])
        logger.info(f"Average {metric}: {avg_metrics[metric]:.4f}")
    
    return avg_metrics


def train(cfg: Config, logger: logging.Logger) -> Any:
    """
    Train a scikit-learn model.
    
    Args:
        cfg: Configuration
        logger: Logger
        
    Returns:
        Trained model
    """
    # Set random seed for reproducibility
    set_seed(cfg.seed)
    
    # Log the configuration
    logger.info("Configuration:")
    logger.info(OmegaConf.to_yaml(cfg))
    
    # Initialize wandb if enabled
    if cfg.logging.wandb.enabled:
        run = init_wandb(cfg)
    
    # Load and preprocess data
    logger.info("Loading and preprocessing data...")
    train_df = pd.read_csv(cfg.data.train_file)
    test_df = pd.read_csv(cfg.data.test_file)
    
    # Extract categorical feature indices if specified in config
    categorical_features = cfg.data.features.get('categorical_indices', None)
    
    # Preprocess data
    X_train, y_train, X_test = preprocess_data(train_df, test_df, cfg)
    
    # Convert feature names if X_train is a DataFrame
    feature_names = None
    if hasattr(X_train, 'columns'):
        feature_names = X_train.columns.tolist()
        X_train = X_train.values
    if hasattr(X_test, 'columns'):
        X_test = X_test.values
    
    # Initialize model
    logger.info(f"Initializing {cfg.model.name} model...")
    model_config = OmegaConf.to_container(cfg.model, resolve=True)
    
    try:
        model = get_sklearn_model(model_config)
    except Exception as e:
        logger.error(f"Error initializing model: {str(e)}")
        raise
    
    # Determine if cross-validation should be used
    use_cv = cfg.training.get('use_cv', True)
    
    if use_cv:
        # Run cross-validation
        cv_metrics = run_cross_validation(model, X_train, y_train, cfg, logger)
        
        # Log metrics to wandb
        if cfg.logging.wandb.enabled:
            wandb.log(cv_metrics)
    
    # Train final model on all data
    logger.info("Training final model on all data...")
    start_time = time.time()
    model.fit(X_train, y_train, feature_names=feature_names, 
             categorical_features=categorical_features)
    train_time = time.time() - start_time
    logger.info(f"Training completed in {train_time:.2f} seconds")
    
    # Make predictions on training data
    y_pred_train = model.predict(X_train)
    
    # Get probability predictions for classification
    y_prob_train = None
    is_classifier = hasattr(model, 'is_classifier') and model.is_classifier
    
    if is_classifier and hasattr(model, 'predict_proba'):
        try:
            y_prob_train = model.predict_proba(X_train)
        except:
            pass
    
    # Compute metrics on training data
    train_metrics = compute_metrics(y_train, y_pred_train, y_prob_train, is_classifier)
    train_metrics = {f"train_{k}": v for k, v in train_metrics.items()}
    
    # Log training metrics
    logger.info(f"Training metrics: {', '.join([f'{k}: {v:.4f}' for k, v in train_metrics.items()])}")
    
    # Log metrics to wandb
    if cfg.logging.wandb.enabled:
        wandb.log(train_metrics)
    
    # Save model
    model_dir = Path(cfg.paths.model_dir)
    model_dir.mkdir(exist_ok=True)
    model_path = model_dir / f"{cfg.model.name}_sklearn.pkl"
    
    logger.info(f"Saving model to {model_path}")
    model.save(str(model_path))
    
    # Save feature importance if available
    feature_importance = model.get_feature_importance()
    if feature_importance:
        importance_df = pd.DataFrame({
            'feature': list(feature_importance.keys()),
            'importance': list(feature_importance.values())
        }).sort_values('importance', ascending=False)
        
        # Save feature importance to CSV
        importance_path = model_dir / f"{cfg.model.name}_feature_importance.csv"
        importance_df.to_csv(importance_path, index=False)
        logger.info(f"Feature importance saved to {importance_path}")
        
        # Log feature importance to wandb
        if cfg.logging.wandb.enabled:
            importance_table = wandb.Table(dataframe=importance_df)
            wandb.log({"feature_importance": importance_table})
            
            # Create feature importance plot
            wandb.log({
                "feature_importance_plot": wandb.plot.bar(
                    wandb.Table(dataframe=importance_df.head(20)),
                    "feature", "importance",
                    title="Feature Importance (Top 20)"
                )
            })
    
    # Finish wandb run
    if cfg.logging.wandb.enabled:
        wandb.finish()
    
    return model


def main():
    """Main function."""
    # Parse arguments
    parser = argparse.ArgumentParser(description="Train scikit-learn model")
    parser.add_argument("--config", type=str, default="default", help="Name of config file")
    args = parser.parse_args()
    
    # Load configuration
    cfg = get_config(args.config)
    
    # Set up logging
    logger = setup_logging(cfg)
    
    # Train model
    model = train(cfg, logger)


if __name__ == "__main__":
    main() 
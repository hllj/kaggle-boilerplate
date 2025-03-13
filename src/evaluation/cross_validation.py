"""
Cross-validation module for evaluating model performance.
"""
import logging
import os
import time
from typing import Dict, List, Optional, Tuple, Union, Any, Callable

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import KFold, StratifiedKFold
from torch.utils.data import DataLoader
from tqdm import tqdm

import wandb
from omegaconf import OmegaConf

from src.config import Config
from src.dataset.load_data import create_kfold_dataloaders, preprocess_data
from src.dataset.feature_engineering import feature_engineering_pipeline
from src.models.model import get_model
from src.models.train import (
    get_loss_fn, get_optimizer, get_scheduler, train_epoch, validate
)
from src.utils import (
    EarlyStopping, get_device, init_wandb, load_data, 
    save_checkpoint, set_seed, setup_logging
)
from src.evaluation.evaluate import evaluate_model


def cross_validate(cfg: Config, logger: logging.Logger) -> Dict[str, float]:
    """
    Perform cross-validation.
    
    Args:
        cfg: Configuration
        logger: Logger
        
    Returns:
        Dictionary with average metrics across folds
    """
    # Set random seed for reproducibility
    set_seed(cfg.seed)
    
    # Get device
    device = get_device()
    logger.info(f"Using device: {device}")
    
    # Load data
    logger.info("Loading data...")
    train_df, test_df, _ = load_data(cfg)
    
    # Feature engineering
    logger.info("Applying feature engineering...")
    train_df, test_df = feature_engineering_pipeline(train_df, test_df, cfg)
    
    # Preprocess data
    logger.info("Preprocessing data...")
    X_train, y_train, X_test = preprocess_data(train_df, test_df, cfg)
    
    # Create K-fold dataloaders
    logger.info(f"Creating {cfg.data.num_folds}-fold data loaders...")
    fold_data_loaders = create_kfold_dataloaders(X_train, y_train, X_test, cfg)
    
    # Initialize fold metrics
    fold_metrics = {}
    all_val_preds = []
    all_val_targets = []
    
    # Initialize W&B run if enabled
    if cfg.logging.wandb.enabled:
        run = init_wandb(cfg, run_name=f"cv_{cfg.model.name}")
    else:
        run = None
    
    # For each fold
    for fold_idx, data_loaders in enumerate(fold_data_loaders):
        logger.info(f"Training fold {fold_idx+1}/{cfg.data.num_folds}")
        
        # Initialize model
        model = get_model(cfg)
        model = model.to(device)
        
        # Initialize optimizer, scheduler, and loss function
        optimizer = get_optimizer(model, cfg)
        scheduler = get_scheduler(optimizer, cfg)
        loss_fn = get_loss_fn(cfg)
        
        # Initialize early stopping
        if cfg.training.early_stopping.enabled:
            early_stopping = EarlyStopping(
                patience=cfg.training.early_stopping.patience,
                min_delta=cfg.training.early_stopping.min_delta,
                mode='min' if cfg.evaluation.monitor == 'val_loss' else 'max'
            )
        
        # Training loop for this fold
        best_metric = float('inf') if cfg.evaluation.monitor == 'val_loss' else float('-inf')
        fold_start_time = time.time()
        
        # For each epoch
        for epoch in range(cfg.training.num_epochs):
            logger.info(f"Fold {fold_idx+1}/{cfg.data.num_folds} - Epoch {epoch+1}/{cfg.training.num_epochs}")
            
            # Train for one epoch
            train_metrics = train_epoch(
                model, 
                data_loaders['train'], 
                optimizer, 
                loss_fn, 
                device,
                mixed_precision=cfg.training.mixed_precision
            )
            
            # Validate
            val_metrics = validate(model, data_loaders['val'], loss_fn, device)
            
            # Update learning rate based on scheduler
            if scheduler is not None:
                if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    scheduler.step(val_metrics[cfg.evaluation.monitor])
                else:
                    scheduler.step()
            
            # Log metrics
            metrics = {**train_metrics, **val_metrics}
            metrics_str = " - ".join(f"{k}: {v:.4f}" for k, v in metrics.items())
            logger.info(f"Fold {fold_idx+1}/{cfg.data.num_folds} - Epoch {epoch+1}/{cfg.training.num_epochs} - {metrics_str}")
            
            # Log to Weights & Biases
            if run is not None:
                fold_metrics_dict = {f"fold_{fold_idx+1}/{k}": v for k, v in metrics.items()}
                wandb.log(fold_metrics_dict, step=epoch)
            
            # Check if model improved
            current_metric = val_metrics[cfg.evaluation.monitor]
            improved = (cfg.evaluation.monitor == 'val_loss' and current_metric < best_metric) or \
                      (cfg.evaluation.monitor != 'val_loss' and current_metric > best_metric)
            
            if improved:
                best_metric = current_metric
                logger.info(f"Fold {fold_idx+1}/{cfg.data.num_folds} - New best {cfg.evaluation.monitor}: {best_metric:.4f}")
                
                # Save best model for this fold
                save_checkpoint(
                    model=model,
                    optimizer=optimizer,
                    epoch=epoch,
                    loss=val_metrics['val_loss'],
                    metric=current_metric,
                    metric_name=cfg.evaluation.monitor,
                    filename=f"best_model_fold_{fold_idx+1}.pt",
                    cfg=cfg
                )
            
            # Early stopping
            if cfg.training.early_stopping.enabled:
                if early_stopping(current_metric):
                    logger.info(f"Fold {fold_idx+1}/{cfg.data.num_folds} - Early stopping triggered after {epoch+1} epochs")
                    break
        
        # Log fold training time
        fold_time = time.time() - fold_start_time
        logger.info(f"Fold {fold_idx+1}/{cfg.data.num_folds} - Training completed in {fold_time:.2f} seconds")
        
        # Get final validation predictions using the best model
        logger.info(f"Fold {fold_idx+1}/{cfg.data.num_folds} - Generating validation predictions...")
        
        # Load best model for this fold
        best_model_path = os.path.join(cfg.paths.model_dir, f"best_model_fold_{fold_idx+1}.pt")
        checkpoint = torch.load(best_model_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        
        # Generate validation predictions
        model.eval()
        val_preds = []
        val_targets = []
        
        with torch.no_grad():
            for batch in tqdm(data_loaders['val'], desc=f"Fold {fold_idx+1} Validation"):
                # Get data
                features = batch['features'].to(device)
                targets = batch['target'].to(device)
                
                # Forward pass
                outputs = model(features)
                
                # Convert outputs to predictions
                if cfg.model.num_classes == 1:  # Binary classification or regression
                    if cfg.evaluation.metric.lower() in ['rmse', 'mae', 'mse']:
                        # Regression
                        preds = outputs.cpu().numpy()
                    else:
                        # Binary classification
                        preds = torch.sigmoid(outputs).cpu().numpy()
                else:
                    # Multi-class classification
                    preds = torch.softmax(outputs, dim=1).cpu().numpy()
                
                # Collect predictions and targets
                val_preds.append(preds)
                val_targets.append(targets.cpu().numpy())
        
        # Concatenate predictions and targets
        val_preds = np.concatenate(val_preds, axis=0)
        val_targets = np.concatenate(val_targets, axis=0)
        
        # Store for later ensemble
        all_val_preds.append(val_preds)
        all_val_targets.append(val_targets)
        
        # Evaluate this fold
        logger.info(f"Fold {fold_idx+1}/{cfg.data.num_folds} - Evaluating performance...")
        
        # Determine task type
        if cfg.model.num_classes == 1:  # Binary classification or regression
            if cfg.evaluation.metric.lower() in ['rmse', 'mae', 'mse']:
                task_type = 'regression'
            else:
                task_type = 'classification'
                # Convert probabilities to class predictions for binary classification
                val_pred_classes = (val_preds > 0.5).astype(int)
        else:
            task_type = 'classification'
            # Convert probabilities to class predictions for multi-class
            val_pred_classes = np.argmax(val_preds, axis=1)
        
        # Evaluate
        if task_type == 'classification':
            fold_metrics[f"fold_{fold_idx+1}"] = evaluate_model(
                y_true=val_targets,
                y_pred=val_pred_classes,
                y_prob=val_preds,
                task_type=task_type,
                output_dir=None,  # Don't save plots for individual folds
                logger=None  # Don't log for individual folds
            )
        else:
            fold_metrics[f"fold_{fold_idx+1}"] = evaluate_model(
                y_true=val_targets,
                y_pred=val_preds,
                task_type=task_type,
                output_dir=None,
                logger=None
            )
        
        # Log fold metrics
        logger.info(f"Fold {fold_idx+1}/{cfg.data.num_folds} - Metrics:")
        for k, v in fold_metrics[f"fold_{fold_idx+1}"].items():
            logger.info(f"  {k}: {v:.4f}")
    
    # Calculate average metrics across folds
    average_metrics = {}
    
    for metric in fold_metrics[f"fold_0"].keys():
        values = [fold_metrics[f"fold_{i}"][metric] for i in range(cfg.data.num_folds)]
        
        # Calculate mean and standard deviation
        mean_value = np.mean(values)
        std_value = np.std(values)
        
        # Store in average metrics
        average_metrics[f"{metric}_mean"] = mean_value
        average_metrics[f"{metric}_std"] = std_value
    
    # Log average metrics
    logger.info("Average metrics across folds:")
    for k, v in average_metrics.items():
        if k.endswith("_mean"):
            metric_name = k[:-5]  # Remove "_mean"
            std_value = average_metrics[f"{metric_name}_std"]
            logger.info(f"  {metric_name}: {v:.4f} ± {std_value:.4f}")
    
    # Optionally, evaluate ensemble of all folds
    if len(all_val_preds) > 1:
        logger.info("Evaluating ensemble of all folds...")
        
        # Stack targets from all folds (they should be the same, but just in case)
        all_targets = np.concatenate(all_val_targets, axis=0)
        
        # Average predictions from all folds
        ensemble_preds = np.mean(all_val_preds, axis=0)
        
        # Evaluate ensemble
        if task_type == 'classification':
            if cfg.model.num_classes == 1:
                # Binary classification
                ensemble_pred_classes = (ensemble_preds > 0.5).astype(int)
            else:
                # Multi-class classification
                ensemble_pred_classes = np.argmax(ensemble_preds, axis=1)
            
            ensemble_metrics = evaluate_model(
                y_true=all_targets,
                y_pred=ensemble_pred_classes,
                y_prob=ensemble_preds,
                task_type=task_type,
                output_dir=os.path.join(cfg.paths.log_dir, "cv_ensemble"),
                logger=logger
            )
        else:
            # Regression
            ensemble_metrics = evaluate_model(
                y_true=all_targets,
                y_pred=ensemble_preds,
                task_type=task_type,
                output_dir=os.path.join(cfg.paths.log_dir, "cv_ensemble"),
                logger=logger
            )
        
        # Add ensemble metrics to average metrics
        for k, v in ensemble_metrics.items():
            average_metrics[f"ensemble_{k}"] = v
    
    # Finish W&B run if enabled
    if run is not None:
        wandb.finish()
    
    return average_metrics


def main():
    """Main function."""
    # Import required modules
    import argparse
    
    # Parse arguments
    parser = argparse.ArgumentParser(description="Cross-validate model")
    parser.add_argument("--config", type=str, default="default", help="Name of config file")
    args = parser.parse_args()
    
    # Load configuration
    from src.config import get_config
    cfg = get_config(args.config)
    
    # Set up logging
    logger = setup_logging(cfg)
    
    # Print configuration
    logger.info("Configuration:")
    logger.info(OmegaConf.to_yaml(cfg))
    
    # Perform cross-validation
    cross_validate(cfg, logger)


if __name__ == "__main__":
    main() 
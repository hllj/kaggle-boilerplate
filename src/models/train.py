"""
Vanilla PyTorch training script.
"""
import argparse
import logging
import os
import time
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

import wandb
from omegaconf import OmegaConf
from tqdm import tqdm

from src.config import Config, get_config
from src.data.load_data import create_data_loaders, preprocess_data
from src.data.feature_engineering import feature_engineering_pipeline
from src.models.model import get_model
from src.utils import (
    EarlyStopping, get_device, init_wandb, load_checkpoint, 
    load_data, save_checkpoint, set_seed, setup_logging
)


def get_optimizer(model: nn.Module, cfg: Config) -> torch.optim.Optimizer:
    """
    Get optimizer based on configuration.
    
    Args:
        model: PyTorch model
        cfg: Configuration
        
    Returns:
        PyTorch optimizer
    """
    optimizer_name = cfg.training.optimizer.name.lower()
    lr = cfg.training.optimizer.lr
    weight_decay = cfg.training.optimizer.weight_decay
    
    if optimizer_name == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer_name == 'adamw':
        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer_name == 'sgd':
        optimizer = optim.SGD(
            model.parameters(), 
            lr=lr, 
            momentum=0.9, 
            weight_decay=weight_decay
        )
    else:
        raise ValueError(f"Optimizer {optimizer_name} not supported")
    
    return optimizer


def get_scheduler(optimizer: torch.optim.Optimizer, cfg: Config) -> Optional[torch.optim.lr_scheduler._LRScheduler]:
    """
    Get learning rate scheduler based on configuration.
    
    Args:
        optimizer: PyTorch optimizer
        cfg: Configuration
        
    Returns:
        PyTorch learning rate scheduler or None
    """
    scheduler_name = cfg.training.scheduler.name.lower()
    
    if scheduler_name == 'none':
        return None
    elif scheduler_name == 'step':
        step_size = cfg.training.scheduler.step_size
        gamma = cfg.training.scheduler.factor
        return optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
    elif scheduler_name == 'cosine':
        return optim.lr_scheduler.CosineAnnealingLR(
            optimizer, 
            T_max=cfg.training.num_epochs
        )
    elif scheduler_name == 'plateau':
        patience = cfg.training.scheduler.patience
        factor = cfg.training.scheduler.factor
        return optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, 
            mode='min', 
            factor=factor, 
            patience=patience, 
            verbose=True
        )
    else:
        raise ValueError(f"Scheduler {scheduler_name} not supported")


def get_loss_fn(cfg: Config) -> nn.Module:
    """
    Get loss function based on configuration.
    
    Args:
        cfg: Configuration
        
    Returns:
        PyTorch loss function
    """
    num_classes = cfg.model.num_classes
    
    if num_classes == 1:  # Binary classification or regression
        # Check if regression or binary classification
        if cfg.evaluation.metric.lower() in ['rmse', 'mae', 'mse']:
            # Regression
            return nn.MSELoss()
        else:
            # Binary classification
            return nn.BCEWithLogitsLoss()
    else:  # Multi-class classification
        return nn.CrossEntropyLoss()


def train_epoch(model: nn.Module, 
               train_loader: DataLoader, 
               optimizer: torch.optim.Optimizer, 
               loss_fn: nn.Module, 
               device: torch.device,
               mixed_precision: bool = False) -> Dict[str, float]:
    """
    Train model for one epoch.
    
    Args:
        model: PyTorch model
        train_loader: Training data loader
        optimizer: PyTorch optimizer
        loss_fn: Loss function
        device: PyTorch device
        mixed_precision: Whether to use mixed precision training
        
    Returns:
        Dictionary with training metrics
    """
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    
    # Create scaler for mixed precision training
    scaler = torch.cuda.amp.GradScaler() if mixed_precision else None
    
    pbar = tqdm(train_loader, desc="Training")
    for batch in pbar:
        # Get data
        features = batch['features'].to(device)
        targets = batch['target'].to(device)
        
        # Zero gradients
        optimizer.zero_grad()
        
        # Forward pass with mixed precision if enabled
        if mixed_precision:
            with torch.cuda.amp.autocast():
                outputs = model(features)
                loss = loss_fn(outputs, targets)
            
            # Backward pass with scaler
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            # Regular forward and backward pass
            outputs = model(features)
            loss = loss_fn(outputs, targets)
            loss.backward()
            optimizer.step()
        
        # Update metrics
        total_loss += loss.item()
        
        # For classification tasks
        if loss_fn.__class__.__name__ in ['CrossEntropyLoss', 'BCEWithLogitsLoss']:
            if loss_fn.__class__.__name__ == 'BCEWithLogitsLoss':
                preds = (torch.sigmoid(outputs) > 0.5).float()
            else:
                _, preds = torch.max(outputs, 1)
            
            correct += (preds == targets).sum().item()
            total += targets.size(0)
            
            # Update progress bar
            pbar.set_postfix({
                'loss': total_loss / (pbar.n + 1),
                'acc': 100 * correct / total if total > 0 else 0.0
            })
        else:
            # For regression tasks
            pbar.set_postfix({
                'loss': total_loss / (pbar.n + 1)
            })
    
    # Calculate metrics
    metrics = {
        'train_loss': total_loss / len(train_loader)
    }
    
    # Add accuracy for classification tasks
    if loss_fn.__class__.__name__ in ['CrossEntropyLoss', 'BCEWithLogitsLoss']:
        metrics['train_acc'] = 100 * correct / total if total > 0 else 0.0
    
    return metrics


def validate(model: nn.Module, 
            val_loader: DataLoader, 
            loss_fn: nn.Module, 
            device: torch.device) -> Dict[str, float]:
    """
    Validate model on validation set.
    
    Args:
        model: PyTorch model
        val_loader: Validation data loader
        loss_fn: Loss function
        device: PyTorch device
        
    Returns:
        Dictionary with validation metrics
    """
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        pbar = tqdm(val_loader, desc="Validation")
        for batch in pbar:
            # Get data
            features = batch['features'].to(device)
            targets = batch['target'].to(device)
            
            # Forward pass
            outputs = model(features)
            loss = loss_fn(outputs, targets)
            
            # Update metrics
            total_loss += loss.item()
            
            # For classification tasks
            if loss_fn.__class__.__name__ in ['CrossEntropyLoss', 'BCEWithLogitsLoss']:
                if loss_fn.__class__.__name__ == 'BCEWithLogitsLoss':
                    preds = (torch.sigmoid(outputs) > 0.5).float()
                else:
                    _, preds = torch.max(outputs, 1)
                
                correct += (preds == targets).sum().item()
                total += targets.size(0)
                
                # Update progress bar
                pbar.set_postfix({
                    'loss': total_loss / (pbar.n + 1),
                    'acc': 100 * correct / total if total > 0 else 0.0
                })
            else:
                # For regression tasks
                pbar.set_postfix({
                    'loss': total_loss / (pbar.n + 1)
                })
    
    # Calculate metrics
    metrics = {
        'val_loss': total_loss / len(val_loader)
    }
    
    # Add accuracy for classification tasks
    if loss_fn.__class__.__name__ in ['CrossEntropyLoss', 'BCEWithLogitsLoss']:
        metrics['val_acc'] = 100 * correct / total if total > 0 else 0.0
    
    return metrics


def train(cfg: Config, logger: logging.Logger) -> nn.Module:
    """
    Train model based on configuration.
    
    Args:
        cfg: Configuration
        logger: Logger
        
    Returns:
        Trained PyTorch model
    """
    # Set random seed for reproducibility
    set_seed(cfg.seed)
    
    # Get device
    device = get_device()
    logger.info(f"Using device: {device}")
    
    # Load data
    logger.info("Loading data...")
    train_df, test_df, sample_submission_df = load_data(cfg)
    
    # Feature engineering
    logger.info("Applying feature engineering...")
    train_df, test_df = feature_engineering_pipeline(train_df, test_df, cfg)
    
    # Preprocess data
    logger.info("Preprocessing data...")
    X_train, y_train, X_test = preprocess_data(train_df, test_df, cfg)
    
    # Create data loaders
    logger.info("Creating data loaders...")
    data_loaders = create_data_loaders(X_train, y_train, X_test, cfg)
    
    # Initialize model
    logger.info(f"Initializing {cfg.model.name} model...")
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
    
    # Initialize Weights & Biases
    run = init_wandb(cfg)
    
    # Training loop
    logger.info("Starting training...")
    best_metric = float('inf') if cfg.evaluation.monitor == 'val_loss' else float('-inf')
    start_time = time.time()
    
    for epoch in range(cfg.training.num_epochs):
        logger.info(f"Epoch {epoch+1}/{cfg.training.num_epochs}")
        
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
            if isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(val_metrics[cfg.evaluation.monitor])
            else:
                scheduler.step()
        
        # Log metrics
        metrics = {**train_metrics, **val_metrics}
        
        # Print metrics
        metrics_str = " - ".join(f"{k}: {v:.4f}" for k, v in metrics.items())
        logger.info(f"Epoch {epoch+1}/{cfg.training.num_epochs} - {metrics_str}")
        
        # Log to Weights & Biases
        if run is not None:
            wandb.log(metrics, step=epoch)
        
        # Save best model
        current_metric = val_metrics[cfg.evaluation.monitor]
        improved = (cfg.evaluation.monitor == 'val_loss' and current_metric < best_metric) or \
                  (cfg.evaluation.monitor != 'val_loss' and current_metric > best_metric)
        
        if improved:
            best_metric = current_metric
            logger.info(f"New best {cfg.evaluation.monitor}: {best_metric:.4f}")
            
            # Save best model
            save_checkpoint(
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                loss=val_metrics['val_loss'],
                metric=current_metric,
                metric_name=cfg.evaluation.monitor,
                filename="best_model.pt",
                cfg=cfg
            )
        
        # Save latest model
        save_checkpoint(
            model=model,
            optimizer=optimizer,
            epoch=epoch,
            loss=val_metrics['val_loss'],
            metric=current_metric,
            metric_name=cfg.evaluation.monitor,
            filename="latest_model.pt",
            cfg=cfg
        )
        
        # Early stopping
        if cfg.training.early_stopping.enabled:
            if early_stopping(current_metric):
                logger.info(f"Early stopping triggered after {epoch+1} epochs")
                break
    
    # Log training time
    train_time = time.time() - start_time
    logger.info(f"Training completed in {train_time:.2f} seconds")
    
    # Load best model
    logger.info("Loading best model...")
    best_model, _ = load_checkpoint(
        checkpoint_path=os.path.join(cfg.paths.model_dir, "best_model.pt"),
        model=model
    )
    
    # Finish Weights & Biases run
    if run is not None:
        wandb.finish()
    
    return best_model


def main():
    """Main function."""
    # Parse arguments
    parser = argparse.ArgumentParser(description="Train PyTorch model")
    parser.add_argument("--config", type=str, default="default", help="Name of config file")
    args = parser.parse_args()
    
    # Load configuration
    cfg = get_config(args.config)
    
    # Set up logging
    logger = setup_logging(cfg)
    
    # Print configuration
    logger.info("Configuration:")
    logger.info(OmegaConf.to_yaml(cfg))
    
    # Train model
    train(cfg, logger)


if __name__ == "__main__":
    main() 
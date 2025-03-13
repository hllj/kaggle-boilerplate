"""
PyTorch Lightning training script.
"""
import argparse
import logging
import os
from typing import Dict, List, Optional, Tuple, Union, Any

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

import lightning as L
from lightning.pytorch.callbacks import (
    ModelCheckpoint, EarlyStopping, LearningRateMonitor, 
    TQDMProgressBar
)
from lightning.pytorch.loggers import WandbLogger, TensorBoardLogger
from lightning.pytorch.utilities.seed import seed_everything

import wandb
from omegaconf import OmegaConf

from src.config import Config, get_config
from src.data.load_data import create_data_loaders, preprocess_data
from src.data.feature_engineering import feature_engineering_pipeline
from src.models.model import get_model
from src.utils import get_device, load_data, setup_logging


class LightningModel(L.LightningModule):
    """
    PyTorch Lightning module wrapper for models.
    
    Attributes:
        model: PyTorch model
        cfg: Configuration
    """
    
    def __init__(self, model: nn.Module, cfg: Config):
        """
        Initialize LightningModel.
        
        Args:
            model: PyTorch model
            cfg: Configuration
        """
        super().__init__()
        self.model = model
        self.cfg = cfg
        self.save_hyperparameters(OmegaConf.to_container(cfg, resolve=True))
        
        # Initialize loss function
        self.loss_fn = self._get_loss_fn()
    
    def _get_loss_fn(self) -> nn.Module:
        """
        Get loss function based on configuration.
        
        Returns:
            PyTorch loss function
        """
        num_classes = self.cfg.model.num_classes
        
        if num_classes == 1:  # Binary classification or regression
            # Check if regression or binary classification
            if self.cfg.evaluation.metric.lower() in ['rmse', 'mae', 'mse']:
                # Regression
                return nn.MSELoss()
            else:
                # Binary classification
                return nn.BCEWithLogitsLoss()
        else:  # Multi-class classification
            return nn.CrossEntropyLoss()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor
            
        Returns:
            Output tensor
        """
        return self.model(x)
    
    def configure_optimizers(self) -> Dict[str, Any]:
        """
        Configure optimizers and schedulers.
        
        Returns:
            Dictionary with optimizer and scheduler configuration
        """
        # Initialize optimizer
        optimizer_name = self.cfg.training.optimizer.name.lower()
        lr = self.cfg.training.optimizer.lr
        weight_decay = self.cfg.training.optimizer.weight_decay
        
        if optimizer_name == 'adam':
            optimizer = optim.Adam(
                self.parameters(), 
                lr=lr, 
                weight_decay=weight_decay
            )
        elif optimizer_name == 'adamw':
            optimizer = optim.AdamW(
                self.parameters(), 
                lr=lr, 
                weight_decay=weight_decay
            )
        elif optimizer_name == 'sgd':
            optimizer = optim.SGD(
                self.parameters(), 
                lr=lr, 
                momentum=0.9, 
                weight_decay=weight_decay
            )
        else:
            raise ValueError(f"Optimizer {optimizer_name} not supported")
        
        # Initialize scheduler
        scheduler_name = self.cfg.training.scheduler.name.lower()
        
        if scheduler_name == 'none':
            return {'optimizer': optimizer}
        
        scheduler_config = {
            'optimizer': optimizer,
        }
        
        if scheduler_name == 'step':
            step_size = self.cfg.training.scheduler.step_size
            gamma = self.cfg.training.scheduler.factor
            scheduler = optim.lr_scheduler.StepLR(
                optimizer, 
                step_size=step_size, 
                gamma=gamma
            )
            scheduler_config['lr_scheduler'] = {
                'scheduler': scheduler,
                'interval': 'epoch',
                'frequency': 1
            }
        elif scheduler_name == 'cosine':
            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                optimizer, 
                T_max=self.cfg.training.num_epochs
            )
            scheduler_config['lr_scheduler'] = {
                'scheduler': scheduler,
                'interval': 'epoch',
                'frequency': 1
            }
        elif scheduler_name == 'plateau':
            patience = self.cfg.training.scheduler.patience
            factor = self.cfg.training.scheduler.factor
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, 
                mode='min', 
                factor=factor, 
                patience=patience, 
                verbose=True
            )
            scheduler_config['lr_scheduler'] = {
                'scheduler': scheduler,
                'interval': 'epoch',
                'frequency': 1,
                'monitor': self.cfg.evaluation.monitor
            }
        else:
            raise ValueError(f"Scheduler {scheduler_name} not supported")
        
        return scheduler_config
    
    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """
        Training step.
        
        Args:
            batch: Batch of data
            batch_idx: Batch index
            
        Returns:
            Loss tensor
        """
        features = batch['features']
        targets = batch['target']
        
        # Forward pass
        outputs = self(features)
        loss = self.loss_fn(outputs, targets)
        
        # Log metrics
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        
        # Calculate accuracy for classification tasks
        if self.loss_fn.__class__.__name__ in ['CrossEntropyLoss', 'BCEWithLogitsLoss']:
            if self.loss_fn.__class__.__name__ == 'BCEWithLogitsLoss':
                preds = (torch.sigmoid(outputs) > 0.5).float()
            else:
                _, preds = torch.max(outputs, 1)
            
            acc = (preds == targets).float().mean() * 100
            self.log('train_acc', acc, on_step=True, on_epoch=True, prog_bar=True)
        
        return loss
    
    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> Dict[str, torch.Tensor]:
        """
        Validation step.
        
        Args:
            batch: Batch of data
            batch_idx: Batch index
            
        Returns:
            Dictionary with loss and possibly other metrics
        """
        features = batch['features']
        targets = batch['target']
        
        # Forward pass
        outputs = self(features)
        loss = self.loss_fn(outputs, targets)
        
        # Log metrics
        self.log('val_loss', loss, on_epoch=True, prog_bar=True)
        
        # Calculate accuracy for classification tasks
        if self.loss_fn.__class__.__name__ in ['CrossEntropyLoss', 'BCEWithLogitsLoss']:
            if self.loss_fn.__class__.__name__ == 'BCEWithLogitsLoss':
                preds = (torch.sigmoid(outputs) > 0.5).float()
            else:
                _, preds = torch.max(outputs, 1)
            
            acc = (preds == targets).float().mean() * 100
            self.log('val_acc', acc, on_epoch=True, prog_bar=True)
        
        return {'val_loss': loss}
    
    def test_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> Dict[str, torch.Tensor]:
        """
        Test step.
        
        Args:
            batch: Batch of data
            batch_idx: Batch index
            
        Returns:
            Dictionary with loss and possibly other metrics
        """
        features = batch['features']
        targets = batch['target']
        
        # Forward pass
        outputs = self(features)
        loss = self.loss_fn(outputs, targets)
        
        # Log metrics
        self.log('test_loss', loss, on_epoch=True)
        
        # Calculate accuracy for classification tasks
        if self.loss_fn.__class__.__name__ in ['CrossEntropyLoss', 'BCEWithLogitsLoss']:
            if self.loss_fn.__class__.__name__ == 'BCEWithLogitsLoss':
                preds = (torch.sigmoid(outputs) > 0.5).float()
            else:
                _, preds = torch.max(outputs, 1)
            
            acc = (preds == targets).float().mean() * 100
            self.log('test_acc', acc, on_epoch=True)
        
        return {'test_loss': loss}


def train(cfg: Config, logger: logging.Logger) -> L.LightningModule:
    """
    Train model using PyTorch Lightning.
    
    Args:
        cfg: Configuration
        logger: Logger
        
    Returns:
        Trained PyTorch Lightning module
    """
    # Set random seed for reproducibility
    seed_everything(cfg.seed)
    
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
    
    # Create PyTorch Lightning module
    lightning_model = LightningModel(model, cfg)
    
    # Initialize loggers
    loggers = []
    
    # TensorBoard logger
    if cfg.logging.tensorboard.enabled:
        tb_logger = TensorBoardLogger(
            save_dir=cfg.paths.log_dir,
            name="tensorboard_logs"
        )
        loggers.append(tb_logger)
    
    # Weights & Biases logger
    if cfg.logging.wandb.enabled:
        wandb_logger = WandbLogger(
            project=cfg.logging.wandb.project,
            entity=cfg.logging.wandb.entity,
            name=f"{cfg.model.name}_{wandb.util.generate_id()}",
            log_model=True,
            tags=cfg.logging.wandb.tags
        )
        wandb_logger.log_hyperparams(OmegaConf.to_container(cfg, resolve=True))
        loggers.append(wandb_logger)
    
    # Initialize callbacks
    callbacks = []
    
    # Model checkpoint callback
    checkpoint_callback = ModelCheckpoint(
        dirpath=cfg.paths.model_dir,
        filename="best_model",
        monitor=cfg.evaluation.monitor,
        mode="min" if cfg.evaluation.monitor == "val_loss" else "max",
        save_top_k=1,
        save_last=True,
        verbose=True
    )
    callbacks.append(checkpoint_callback)
    
    # Early stopping callback
    if cfg.training.early_stopping.enabled:
        early_stopping_callback = EarlyStopping(
            monitor=cfg.evaluation.monitor,
            mode="min" if cfg.evaluation.monitor == "val_loss" else "max",
            patience=cfg.training.early_stopping.patience,
            min_delta=cfg.training.early_stopping.min_delta,
            verbose=True
        )
        callbacks.append(early_stopping_callback)
    
    # Learning rate monitor
    lr_monitor = LearningRateMonitor(logging_interval="epoch")
    callbacks.append(lr_monitor)
    
    # TQDM progress bar
    progress_bar = TQDMProgressBar(refresh_rate=10)
    callbacks.append(progress_bar)
    
    # Initialize trainer
    trainer = L.Trainer(
        max_epochs=cfg.training.num_epochs,
        callbacks=callbacks,
        logger=loggers,
        precision="16-mixed" if cfg.training.mixed_precision else "32",
        gradient_clip_val=cfg.training.gradient_clipping.max_norm if cfg.training.gradient_clipping.enabled else None,
        log_every_n_steps=10,
        enable_progress_bar=True,
        enable_model_summary=True,
        deterministic=True,
        devices=1,
        accelerator="auto"
    )
    
    # Train model
    logger.info("Starting training...")
    trainer.fit(
        model=lightning_model,
        train_dataloaders=data_loaders['train'],
        val_dataloaders=data_loaders['val']
    )
    
    # Log best metric
    best_metric = checkpoint_callback.best_model_score.item()
    logger.info(f"Best {cfg.evaluation.monitor}: {best_metric:.4f}")
    
    # Test model
    logger.info("Testing best model...")
    trainer.test(
        ckpt_path=checkpoint_callback.best_model_path,
        dataloaders=data_loaders['val']
    )
    
    # Close loggers
    if cfg.logging.wandb.enabled:
        wandb.finish()
    
    return lightning_model


def main():
    """Main function."""
    # Parse arguments
    parser = argparse.ArgumentParser(description="Train PyTorch Lightning model")
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
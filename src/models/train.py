"""
Training script for BirdCLEF competition.
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
from torch.cuda.amp import GradScaler, autocast
import wandb
from omegaconf import OmegaConf
from tqdm import tqdm
import numpy as np

from src.config import Config, get_config
from src.dataset.load_data import create_data_loaders
from src.models.model import get_model
from src.evaluation.metrics import calculate_competition_metrics, metrics_to_string
from src.utils import (
    EarlyStopping, get_device, init_wandb, load_checkpoint, 
    save_checkpoint, set_seed, setup_logging
)


class AverageMeter:
    """
    Computes and stores the average and current value.
    """
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def mixup(data, targets, alpha=0.5):
    """
    Perform mixup augmentation.
    
    Args:
        data: Batch data
        targets: Batch targets
        alpha: Mixup parameter
        
    Returns:
        Mixed data and targets
    """
    indices = torch.randperm(data.size(0))
    data2 = data[indices]
    targets2 = targets[indices]

    lam = torch.FloatTensor([np.random.beta(alpha, alpha)])
    data = data * lam + data2 * (1 - lam)
    targets = targets * lam + targets2 * (1 - lam)

    return data, targets


def train_one_epoch(model: nn.Module, 
                   loader: DataLoader, 
                   optimizer: optim.Optimizer,
                   criterion: nn.Module,
                   device: torch.device,
                   scaler: Optional[GradScaler] = None,
                   max_grad_norm: float = 10.0) -> Tuple[Dict[str, float], float]:
    """
    Train model for one epoch.
    
    Args:
        model: PyTorch model
        loader: Training data loader
        optimizer: PyTorch optimizer
        criterion: Loss function
        device: PyTorch device
        scaler: Gradient scaler for mixed precision training
        max_grad_norm: Maximum gradient norm for clipping
        
    Returns:
        Tuple of metrics dictionary and average loss
    """
    model.train()
    losses = AverageMeter()
    gt = []
    preds = []
    
    bar = tqdm(loader, total=len(loader))
    for batch in bar:
        optimizer.zero_grad()
        spec = batch['spec']
        target = batch['target']

        spec, target = mixup(spec, target, 0.5)

        spec = spec.to(device)
        target = target.to(device)

        if scaler is not None:
            with autocast():
                logits = model(spec)
                loss = criterion(logits, target)
            scaler.scale(loss).backward()
            grad_norm = nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(spec)
            loss = criterion(logits, target)
            loss.backward()
            grad_norm = nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)
            optimizer.step()

        losses.update(loss.item(), batch["spec"].size(0))
        bar.set_postfix(
            loss=losses.avg,
            grad=grad_norm.item(),
            lr=optimizer.param_groups[0]["lr"]
        )
        gt.append(target.cpu().detach().numpy())
        preds.append(logits.sigmoid().cpu().detach().numpy())
        
    gt = np.concatenate(gt)
    preds = np.concatenate(preds)
    scores = calculate_competition_metrics_no_map(gt, preds, target_columns)

    return scores, losses.avg


def validate(model: nn.Module, 
            loader: DataLoader, 
            criterion: nn.Module,
            device: torch.device) -> Tuple[Dict[str, float], float]:
    """
    Validate model.
    
    Args:
        model: PyTorch model
        loader: Validation data loader
        criterion: Loss function
        device: PyTorch device
        
    Returns:
        Tuple of metrics dictionary and average loss
    """
    model.eval()
    losses = AverageMeter()
    gt = []
    preds = []

    with torch.no_grad():
        bar = tqdm(loader, total=len(loader))
        for batch in bar:
            spec = batch['spec'].to(device)
            target = batch['target'].to(device)

            logits = model(spec)
            loss = criterion(logits, target)

            losses.update(loss.item(), batch["spec"].size(0))
            gt.append(target.cpu().detach().numpy())
            preds.append(logits.sigmoid().cpu().detach().numpy())
            bar.set_postfix(loss=losses.avg)

    gt = np.concatenate(gt)
    preds = np.concatenate(preds)
    scores = calculate_competition_metrics(gt, preds, target_columns)
    
    return scores, losses.avg


def train(cfg: Config, logger: logging.Logger) -> None:
    """
    Train model.
    
    Args:
        cfg: Configuration
        logger: Logger
    """
    # Set random seed
    set_seed(cfg.seed)
    
    # Get device
    device = get_device()
    logger.info(f"Using device: {device}")
    
    # Initialize wandb
    run = init_wandb(cfg)
    
    # Create output directories
    os.makedirs(cfg.paths.model_dir, exist_ok=True)
    
    # Create data loaders
    train_loader, val_loader = create_data_loaders(cfg, cfg.data.fold)
    
    # Create model
    model = get_model(cfg).to(device)
    
    # Create optimizer
    optimizer = optim.AdamW(
        model.parameters(), 
        lr=cfg.training.optimizer.lr,
        weight_decay=cfg.training.optimizer.weight_decay
    )
    
    # Create scheduler
    scheduler_cosine = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, 
        cfg.training.num_epochs - cfg.training.warmup_epochs
    )
    scheduler = GradualWarmupSchedulerV2(
        optimizer, 
        multiplier=10, 
        total_epoch=cfg.training.warmup_epochs,
        after_scheduler=scheduler_cosine
    )
    
    # Create criterion
    criterion = FocalLossBCE()
    
    # Create scaler for mixed precision training
    scaler = GradScaler() if cfg.training.mixed_precision else None
    
    # Create early stopping
    early_stopping = EarlyStopping(
        patience=cfg.training.early_stopping.patience,
        min_delta=cfg.training.early_stopping.min_delta,
        mode='max'
    )
    
    # Training loop
    best_score = 0.0
    for epoch in range(1, cfg.training.num_epochs + 1):
        logger.info(f"Epoch {epoch}/{cfg.training.num_epochs}")
        
        # Train
        train_scores, train_loss = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
            scaler=scaler,
            max_grad_norm=cfg.training.gradient_clipping.max_norm
        )
        
        # Log training metrics
        train_metrics_str = metrics_to_string(train_scores, "Train")
        logger.info(f"Epoch {epoch} - Train loss: {train_loss:.4f}, {train_metrics_str}")
        
        if run is not None:
            wandb.log({
                "train_loss": train_loss,
                **{f"train_{k}": v for k, v in train_scores.items()},
                "learning_rate": optimizer.param_groups[0]["lr"],
                "epoch": epoch
            })
        
        # Validate
        val_scores, val_loss = validate(
            model=model,
            loader=val_loader,
            criterion=criterion,
            device=device
        )
        
        # Log validation metrics
        val_metrics_str = metrics_to_string(val_scores, "Valid")
        logger.info(f"Epoch {epoch} - Valid loss: {val_loss:.4f}, {val_metrics_str}")
        
        if run is not None:
            wandb.log({
                "val_loss": val_loss,
                **{f"val_{k}": v for k, v in val_scores.items()},
                "epoch": epoch
            })
        
        # Update learning rate
        scheduler.step()
        
        # Save best model
        val_score = val_scores["ROC"]
        if val_score > best_score:
            best_score = val_score
            logger.info(f"Epoch {epoch} - Save Best Score: {best_score:.4f} Model")
            save_checkpoint(
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                loss=val_loss,
                metric=val_score,
                metric_name="ROC",
                filename=f"fold_{cfg.data.fold}.bin",
                cfg=cfg
            )
        
        # Early stopping
        if early_stopping(val_score):
            logger.info(f"Early stopping triggered after {epoch} epochs")
            break
    
    # Save final model
    save_checkpoint(
        model=model,
        optimizer=optimizer,
        epoch=epoch,
        loss=val_loss,
        metric=val_score,
        metric_name="ROC",
        filename=f"fold_{cfg.data.fold}_final.bin",
        cfg=cfg
    )
    
    if run is not None:
        wandb.finish()


def main():
    """Main function."""
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="default", help="Config name")
    args = parser.parse_args()
    
    # Load config
    cfg = get_config(args.config)
    
    # Set up logging
    logger = setup_logging(cfg)
    
    # Train model
    train(cfg, logger)


if __name__ == "__main__":
    main() 
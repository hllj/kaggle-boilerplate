"""
Utility functions for Kaggle competition.
"""
import logging
import os
import random
import warnings
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
import wandb
from omegaconf import DictConfig, OmegaConf

from src.config import Config


def set_seed(seed: int) -> None:
    """
    Set random seed for reproducibility.
    
    Args:
        seed: Random seed
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)


def get_device() -> torch.device:
    """
    Get device for PyTorch.
    
    Returns:
        PyTorch device
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")


def setup_logging(cfg: Config) -> logging.Logger:
    """
    Set up logging configuration.
    
    Args:
        cfg: Configuration
        
    Returns:
        Logger
    """
    log_level = getattr(logging, cfg.logging.console.level.upper())
    
    # Create logs directory if it doesn't exist
    os.makedirs(cfg.paths.log_dir, exist_ok=True)
    
    # Create timestamp for log file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(cfg.paths.log_dir, f"train_{timestamp}.log")
    
    # Configure logging
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    
    # Create logger
    logger = logging.getLogger(__name__)
    
    return logger


def init_wandb(cfg: Config, run_name: Optional[str] = None) -> Optional[wandb.run]:
    """
    Initialize Weights and Biases logging.
    
    Args:
        cfg: Configuration
        run_name: Name of the run
        
    Returns:
        Wandb run or None if disabled
    """
    if not cfg.logging.wandb.enabled:
        return None
    
    # Create timestamp for run name if not provided
    if run_name is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_name = f"{cfg.model.name}_{timestamp}"
    
    # Initialize wandb
    run = wandb.init(
        project=cfg.logging.wandb.project,
        entity=cfg.logging.wandb.entity,
        name=run_name,
        config=OmegaConf.to_container(cfg, resolve=True),
        tags=cfg.logging.wandb.tags,
    )
    
    return run


def load_data(cfg: Config) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load data from files specified in config.
    
    Args:
        cfg: Configuration
        
    Returns:
        Tuple of train, test, and sample submission dataframes
    """
    train_df = pd.read_csv(cfg.data.train_file)
    test_df = pd.read_csv(cfg.data.test_file)
    sample_submission_df = pd.read_csv(cfg.data.sample_submission_file)
    
    return train_df, test_df, sample_submission_df


def save_checkpoint(model: torch.nn.Module, 
                   optimizer: torch.optim.Optimizer, 
                   epoch: int, 
                   loss: float,
                   metric: float,
                   metric_name: str,
                   filename: str,
                   cfg: Config) -> None:
    """
    Save model checkpoint.
    
    Args:
        model: PyTorch model
        optimizer: PyTorch optimizer
        epoch: Current epoch
        loss: Current loss
        metric: Current metric value
        metric_name: Name of the metric
        filename: Name of the checkpoint file
        cfg: Configuration
    """
    # Create model directory if it doesn't exist
    os.makedirs(cfg.paths.model_dir, exist_ok=True)
    
    # Create checkpoint
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "loss": loss,
        metric_name: metric,
        "config": OmegaConf.to_container(cfg, resolve=True)
    }
    
    # Save checkpoint
    torch.save(checkpoint, os.path.join(cfg.paths.model_dir, filename))


def load_checkpoint(checkpoint_path: str, 
                   model: torch.nn.Module, 
                   optimizer: Optional[torch.optim.Optimizer] = None) -> Tuple[torch.nn.Module, Dict[str, Any]]:
    """
    Load model checkpoint.
    
    Args:
        checkpoint_path: Path to checkpoint file
        model: PyTorch model
        optimizer: PyTorch optimizer (optional)
        
    Returns:
        Tuple of model and checkpoint info
    """
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=get_device())
    
    # Load model state
    model.load_state_dict(checkpoint["model_state_dict"])
    
    # Load optimizer state if provided
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    
    # Extract checkpoint info
    checkpoint_info = {
        "epoch": checkpoint["epoch"],
        "loss": checkpoint["loss"]
    }
    
    # Add any additional metrics
    for key, value in checkpoint.items():
        if key not in ["epoch", "model_state_dict", "optimizer_state_dict", "loss", "config"]:
            checkpoint_info[key] = value
    
    return model, checkpoint_info


class EarlyStopping:
    """
    Early stopping to stop training when a monitored metric stops improving.
    
    Attributes:
        patience: Number of epochs with no improvement after which training will be stopped
        min_delta: Minimum change in the monitored metric to qualify as an improvement
        counter: Counter for epochs with no improvement
        best_score: Best score so far
        early_stop: Flag to indicate if early stopping should be triggered
    """
    
    def __init__(self, patience: int = 5, min_delta: float = 0.0, mode: str = "min"):
        """
        Initialize EarlyStopping.
        
        Args:
            patience: Number of epochs with no improvement after which training will be stopped
            min_delta: Minimum change in the monitored metric to qualify as an improvement
            mode: One of 'min' or 'max' to indicate whether we want to minimize or maximize the monitored metric
        """
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.mode = mode
        assert mode in ["min", "max"], "Mode must be one of 'min' or 'max'"
    
    def __call__(self, metric: float) -> bool:
        """
        Call function for the EarlyStopping object.
        
        Args:
            metric: Monitored metric
            
        Returns:
            True if early stopping should be triggered, False otherwise
        """
        if self.best_score is None:
            self.best_score = metric
            return False
        
        if self.mode == "min":
            if metric < self.best_score - self.min_delta:
                self.best_score = metric
                self.counter = 0
            else:
                self.counter += 1
                if self.counter >= self.patience:
                    self.early_stop = True
        else:  # mode == "max"
            if metric > self.best_score + self.min_delta:
                self.best_score = metric
                self.counter = 0
            else:
                self.counter += 1
                if self.counter >= self.patience:
                    self.early_stop = True
        
        return self.early_stop 
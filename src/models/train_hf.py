"""
HuggingFace Transformers training script.
"""
import argparse
import logging
import os
from typing import Dict, List, Optional, Tuple, Union, Any

import numpy as np
import torch
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, mean_squared_error

import transformers
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
    set_seed
)
from transformers.integrations import WandbCallback
from transformers.trainer_utils import get_last_checkpoint
from datasets import Dataset, DatasetDict

import wandb
from omegaconf import OmegaConf

from src.config import Config, get_config
from src.utils import get_device, load_data, setup_logging


def preprocess_text_data(train_df: pd.DataFrame, 
                        test_df: pd.DataFrame, 
                        cfg: Config) -> Tuple[DatasetDict, Any]:
    """
    Preprocess text data for HuggingFace Transformers.
    
    Args:
        train_df: Training dataframe
        test_df: Test dataframe
        cfg: Configuration
        
    Returns:
        Tuple of DatasetDict and tokenizer
    """
    # Get column names
    text_col = cfg.data.features.get('text_column', 'text')
    target_col = cfg.data.target_column
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        cfg.model.get('transformer_name', 'bert-base-uncased')
    )
    
    # Function to tokenize data
    def tokenize_function(examples):
        return tokenizer(
            examples[text_col], 
            padding="max_length",
            truncation=True,
            max_length=cfg.model.get('max_length', 128),
        )
    
    # Remove unnecessary columns
    train_cols = [text_col, target_col]
    test_cols = [text_col]
    
    # Ensure columns exist
    for col in train_cols:
        if col not in train_df.columns:
            raise ValueError(f"Column {col} not found in training data")
    
    for col in test_cols:
        if col not in test_df.columns:
            raise ValueError(f"Column {col} not found in test data")
    
    # Split train and validation
    if cfg.data.get('validation_file', None) is not None:
        # Load validation data from file
        val_df = pd.read_csv(cfg.data.validation_file)
        for col in train_cols:
            if col not in val_df.columns:
                raise ValueError(f"Column {col} not found in validation data")
    else:
        # Split train data
        train_df, val_df = train_test_split(
            train_df, 
            test_size=cfg.data.test_size,
            random_state=cfg.data.random_state
        )
    
    # Create Hugging Face datasets
    train_dataset = Dataset.from_pandas(train_df[train_cols])
    val_dataset = Dataset.from_pandas(val_df[train_cols])
    test_dataset = Dataset.from_pandas(test_df[test_cols])
    
    # Combine in a DatasetDict
    dataset_dict = DatasetDict({
        'train': train_dataset,
        'validation': val_dataset,
        'test': test_dataset
    })
    
    # Tokenize the data
    tokenized_datasets = dataset_dict.map(
        tokenize_function,
        batched=True,
        remove_columns=[text_col]
    )
    
    # Rename target column to "labels" (required by HuggingFace)
    tokenized_datasets = tokenized_datasets.rename_column(target_col, "labels")
    
    return tokenized_datasets, tokenizer


def compute_metrics(eval_pred: Tuple[np.ndarray, np.ndarray]) -> Dict[str, float]:
    """
    Compute metrics for evaluation.
    
    Args:
        eval_pred: Tuple of predictions and labels
        
    Returns:
        Dictionary with metrics
    """
    predictions, labels = eval_pred
    
    # For regression
    if len(labels.shape) == 1 or labels.shape[1] == 1:
        if len(predictions.shape) > 1 and predictions.shape[1] > 1:
            # Take the first column for regression
            predictions = predictions[:, 0]
        
        mse = mean_squared_error(labels, predictions)
        rmse = np.sqrt(mse)
        return {
            "mse": mse,
            "rmse": rmse,
        }
    
    # For binary classification
    elif len(np.unique(labels)) == 2:
        if len(predictions.shape) > 1 and predictions.shape[1] > 1:
            # Get predicted class 
            preds = np.argmax(predictions, axis=1)
            # Get probabilities for ROC AUC
            probs = predictions[:, 1]
        else:
            # For binary classification with one output neuron
            preds = (predictions > 0).astype(int)
            probs = 1 / (1 + np.exp(-predictions))  # Sigmoid
        
        accuracy = accuracy_score(labels, preds)
        f1 = f1_score(labels, preds)
        try:
            auc = roc_auc_score(labels, probs)
        except Exception:
            auc = 0.0
        
        return {
            "accuracy": accuracy,
            "f1": f1,
            "auc": auc,
        }
    
    # For multi-class classification
    else:
        preds = np.argmax(predictions, axis=1)
        accuracy = accuracy_score(labels, preds)
        f1_macro = f1_score(labels, preds, average="macro")
        f1_weighted = f1_score(labels, preds, average="weighted")
        
        return {
            "accuracy": accuracy,
            "f1_macro": f1_macro,
            "f1_weighted": f1_weighted,
        }


def train(cfg: Config, logger: logging.Logger) -> transformers.PreTrainedModel:
    """
    Train a HuggingFace transformer model.
    
    Args:
        cfg: Configuration
        logger: Logger
        
    Returns:
        Trained HuggingFace model
    """
    # Set random seed for reproducibility
    set_seed(cfg.seed)
    
    # Get device
    device = get_device()
    logger.info(f"Using device: {device}")
    
    # Load data
    logger.info("Loading data...")
    train_df, test_df, _ = load_data(cfg)
    
    # Preprocess data
    logger.info("Preprocessing data...")
    tokenized_datasets, tokenizer = preprocess_text_data(train_df, test_df, cfg)
    
    # Define model
    logger.info(f"Initializing model: {cfg.model.get('transformer_name', 'bert-base-uncased')}...")
    
    # Check problem type
    num_labels = cfg.model.num_classes
    problem_type = None
    
    if num_labels == 1:
        if cfg.evaluation.metric.lower() in ['rmse', 'mae', 'mse']:
            problem_type = "regression"
        else:
            problem_type = "single_label_classification"
    else:
        problem_type = "single_label_classification"
    
    # Initialize model
    model = AutoModelForSequenceClassification.from_pretrained(
        cfg.model.get('transformer_name', 'bert-base-uncased'),
        num_labels=num_labels,
        problem_type=problem_type
    )
    
    # Set up model directories
    os.makedirs(cfg.paths.model_dir, exist_ok=True)
    
    # Check if there's a checkpoint to resume from
    last_checkpoint = None
    if os.path.isdir(cfg.paths.model_dir):
        last_checkpoint = get_last_checkpoint(cfg.paths.model_dir)
        if last_checkpoint is not None:
            logger.info(f"Resuming from checkpoint: {last_checkpoint}")
    
    # Initialize wandb if enabled
    if cfg.logging.wandb.enabled:
        wandb.init(
            project=cfg.logging.wandb.project,
            entity=cfg.logging.wandb.entity,
            name=f"{cfg.model.name}_{wandb.util.generate_id()}",
            tags=cfg.logging.wandb.tags,
            config=OmegaConf.to_container(cfg, resolve=True)
        )
    
    # Define training arguments
    training_args = TrainingArguments(
        output_dir=cfg.paths.model_dir,
        num_train_epochs=cfg.training.num_epochs,
        per_device_train_batch_size=cfg.training.batch_size,
        per_device_eval_batch_size=cfg.inference.batch_size,
        learning_rate=cfg.training.optimizer.lr,
        weight_decay=cfg.training.optimizer.weight_decay,
        logging_dir=os.path.join(cfg.paths.log_dir, "hf_logs"),
        logging_steps=10,
        eval_steps=100,
        save_steps=100,
        evaluation_strategy="steps",
        save_strategy="steps",
        load_best_model_at_end=True,
        metric_for_best_model=cfg.evaluation.monitor.replace('val_', ''),
        greater_is_better=cfg.evaluation.monitor != 'val_loss',
        report_to="wandb" if cfg.logging.wandb.enabled else None,
        fp16=cfg.training.mixed_precision,
        gradient_accumulation_steps=1,
        gradient_checkpointing=False,
        dataloader_num_workers=cfg.resources.num_workers,
        dataloader_pin_memory=cfg.resources.pin_memory,
    )
    
    # Define callbacks
    callbacks = []
    
    # Add early stopping callback if enabled
    if cfg.training.early_stopping.enabled:
        early_stopping_callback = EarlyStoppingCallback(
            early_stopping_patience=cfg.training.early_stopping.patience,
            early_stopping_threshold=cfg.training.early_stopping.min_delta
        )
        callbacks.append(early_stopping_callback)
    
    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
        callbacks=callbacks
    )
    
    # Train model
    logger.info("Starting training...")
    trainer.train(resume_from_checkpoint=last_checkpoint)
    
    # Evaluate model
    logger.info("Evaluating model...")
    eval_results = trainer.evaluate()
    logger.info(f"Evaluation results: {eval_results}")
    
    # Save model
    logger.info("Saving model...")
    trainer.save_model(os.path.join(cfg.paths.model_dir, "best_model"))
    tokenizer.save_pretrained(os.path.join(cfg.paths.model_dir, "best_model"))
    
    # Finish wandb if enabled
    if cfg.logging.wandb.enabled:
        wandb.finish()
    
    return model


def main():
    """Main function."""
    # Import sklearn for train_test_split
    from sklearn.model_selection import train_test_split
    
    # Parse arguments
    parser = argparse.ArgumentParser(description="Train HuggingFace model")
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
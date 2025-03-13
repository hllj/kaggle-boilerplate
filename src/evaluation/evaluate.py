"""
Evaluation module for computing metrics.
"""
import logging
from typing import Dict, List, Optional, Tuple, Union, Any, Callable

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, log_loss, confusion_matrix,
    mean_squared_error, mean_absolute_error, r2_score,
    classification_report
)
import matplotlib.pyplot as plt
import seaborn as sns

from src.config import Config


def compute_classification_metrics(y_true: np.ndarray, 
                                 y_pred: np.ndarray, 
                                 y_prob: Optional[np.ndarray] = None) -> Dict[str, float]:
    """
    Compute classification metrics.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        y_prob: Predicted probabilities (for ROC AUC)
        
    Returns:
        Dictionary with metrics
    """
    metrics = {}
    
    # Basic metrics
    metrics['accuracy'] = accuracy_score(y_true, y_pred)
    
    # Check if binary or multiclass
    if len(np.unique(y_true)) == 2:  # Binary classification
        metrics['precision'] = precision_score(y_true, y_pred)
        metrics['recall'] = recall_score(y_true, y_pred)
        metrics['f1'] = f1_score(y_true, y_pred)
        
        # ROC AUC if probabilities provided
        if y_prob is not None:
            if len(y_prob.shape) > 1 and y_prob.shape[1] > 1:
                # Take positive class probability
                y_prob = y_prob[:, 1]
            metrics['roc_auc'] = roc_auc_score(y_true, y_prob)
            
            # Log loss
            metrics['log_loss'] = log_loss(y_true, y_prob)
    else:  # Multi-class classification
        # Macro-averaged metrics
        metrics['precision_macro'] = precision_score(y_true, y_pred, average='macro')
        metrics['recall_macro'] = recall_score(y_true, y_pred, average='macro')
        metrics['f1_macro'] = f1_score(y_true, y_pred, average='macro')
        
        # Weighted-averaged metrics
        metrics['precision_weighted'] = precision_score(y_true, y_pred, average='weighted')
        metrics['recall_weighted'] = recall_score(y_true, y_pred, average='weighted')
        metrics['f1_weighted'] = f1_score(y_true, y_pred, average='weighted')
        
        # Multi-class ROC AUC
        if y_prob is not None:
            try:
                metrics['roc_auc_ovr'] = roc_auc_score(y_true, y_prob, multi_class='ovr')
                metrics['roc_auc_ovo'] = roc_auc_score(y_true, y_prob, multi_class='ovo')
            except Exception:
                pass
            
            # Log loss
            metrics['log_loss'] = log_loss(y_true, y_prob)
    
    return metrics


def compute_regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Compute regression metrics.
    
    Args:
        y_true: True values
        y_pred: Predicted values
        
    Returns:
        Dictionary with metrics
    """
    metrics = {}
    
    # Mean squared error
    metrics['mse'] = mean_squared_error(y_true, y_pred)
    
    # Root mean squared error
    metrics['rmse'] = np.sqrt(metrics['mse'])
    
    # Mean absolute error
    metrics['mae'] = mean_absolute_error(y_true, y_pred)
    
    # R-squared
    metrics['r2'] = r2_score(y_true, y_pred)
    
    # Mean absolute percentage error (MAPE)
    # Avoid division by zero
    mask = y_true != 0
    if mask.sum() > 0:
        metrics['mape'] = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
    else:
        metrics['mape'] = np.nan
    
    return metrics


def plot_confusion_matrix(y_true: np.ndarray, 
                        y_pred: np.ndarray, 
                        class_names: Optional[List[str]] = None,
                        figsize: Tuple[int, int] = (10, 8),
                        cmap: str = 'Blues',
                        normalize: bool = False) -> plt.Figure:
    """
    Plot confusion matrix.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        class_names: Names of classes
        figsize: Figure size
        cmap: Color map
        normalize: Whether to normalize the confusion matrix
        
    Returns:
        Matplotlib figure
    """
    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # Normalize if requested
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot heatmap
    sns.heatmap(cm, annot=True, fmt='.2f' if normalize else 'd',
                cmap=cmap, cbar=True, square=True, ax=ax)
    
    # Set labels
    ax.set_xlabel('Predicted label')
    ax.set_ylabel('True label')
    
    # Set title
    title = 'Normalized Confusion Matrix' if normalize else 'Confusion Matrix'
    ax.set_title(title)
    
    # Set ticks
    if class_names is not None:
        ax.set_xticks(np.arange(len(class_names)) + 0.5)
        ax.set_yticks(np.arange(len(class_names)) + 0.5)
        ax.set_xticklabels(class_names)
        ax.set_yticklabels(class_names)
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    
    # Adjust layout
    fig.tight_layout()
    
    return fig


def plot_roc_curve(y_true: np.ndarray, 
                  y_prob: np.ndarray, 
                  class_names: Optional[List[str]] = None,
                  figsize: Tuple[int, int] = (10, 8)) -> plt.Figure:
    """
    Plot ROC curve.
    
    Args:
        y_true: True labels
        y_prob: Predicted probabilities
        class_names: Names of classes
        figsize: Figure size
        
    Returns:
        Matplotlib figure
    """
    from sklearn.metrics import roc_curve, auc
    from sklearn.preprocessing import label_binarize
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Check if binary or multiclass
    if len(np.unique(y_true)) == 2:  # Binary classification
        if len(y_prob.shape) > 1 and y_prob.shape[1] > 1:
            # Take positive class probability
            y_prob = y_prob[:, 1]
        
        # Compute ROC curve
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        roc_auc = auc(fpr, tpr)
        
        # Plot ROC curve
        ax.plot(fpr, tpr, lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
        
    else:  # Multi-class classification
        n_classes = len(np.unique(y_true))
        
        # Binarize the labels
        y_true_bin = label_binarize(y_true, classes=np.unique(y_true))
        
        # Compute ROC curve and ROC area for each class
        fpr = {}
        tpr = {}
        roc_auc = {}
        
        for i in range(n_classes):
            fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], y_prob[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])
            
            # Plot ROC curve for each class
            label = f'Class {i}' if class_names is None else class_names[i]
            ax.plot(fpr[i], tpr[i], lw=2,
                  label=f'ROC curve {label} (AUC = {roc_auc[i]:.2f})')
    
    # Plot diagonal
    ax.plot([0, 1], [0, 1], 'k--', lw=2)
    
    # Set axis limits
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    
    # Set labels
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('Receiver Operating Characteristic (ROC) Curve')
    
    # Add legend
    ax.legend(loc="lower right")
    
    # Adjust layout
    fig.tight_layout()
    
    return fig


def plot_precision_recall_curve(y_true: np.ndarray, 
                              y_prob: np.ndarray, 
                              class_names: Optional[List[str]] = None,
                              figsize: Tuple[int, int] = (10, 8)) -> plt.Figure:
    """
    Plot precision-recall curve.
    
    Args:
        y_true: True labels
        y_prob: Predicted probabilities
        class_names: Names of classes
        figsize: Figure size
        
    Returns:
        Matplotlib figure
    """
    from sklearn.metrics import precision_recall_curve, average_precision_score
    from sklearn.preprocessing import label_binarize
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Check if binary or multiclass
    if len(np.unique(y_true)) == 2:  # Binary classification
        if len(y_prob.shape) > 1 and y_prob.shape[1] > 1:
            # Take positive class probability
            y_prob = y_prob[:, 1]
        
        # Compute precision-recall curve
        precision, recall, _ = precision_recall_curve(y_true, y_prob)
        ap = average_precision_score(y_true, y_prob)
        
        # Plot precision-recall curve
        ax.plot(recall, precision, lw=2, label=f'PR curve (AP = {ap:.2f})')
        
    else:  # Multi-class classification
        n_classes = len(np.unique(y_true))
        
        # Binarize the labels
        y_true_bin = label_binarize(y_true, classes=np.unique(y_true))
        
        # Compute precision-recall curve and average precision for each class
        precision = {}
        recall = {}
        ap = {}
        
        for i in range(n_classes):
            precision[i], recall[i], _ = precision_recall_curve(y_true_bin[:, i], y_prob[:, i])
            ap[i] = average_precision_score(y_true_bin[:, i], y_prob[:, i])
            
            # Plot precision-recall curve for each class
            label = f'Class {i}' if class_names is None else class_names[i]
            ax.plot(recall[i], precision[i], lw=2,
                  label=f'PR curve {label} (AP = {ap[i]:.2f})')
    
    # Set axis limits
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    
    # Set labels
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_title('Precision-Recall Curve')
    
    # Add legend
    ax.legend(loc="lower left")
    
    # Adjust layout
    fig.tight_layout()
    
    return fig


def plot_residuals(y_true: np.ndarray, y_pred: np.ndarray, figsize: Tuple[int, int] = (10, 8)) -> plt.Figure:
    """
    Plot residuals for regression.
    
    Args:
        y_true: True values
        y_pred: Predicted values
        figsize: Figure size
        
    Returns:
        Matplotlib figure
    """
    # Compute residuals
    residuals = y_true - y_pred
    
    # Create figure with 2 subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    # Plot residuals vs predicted values
    ax1.scatter(y_pred, residuals, alpha=0.5)
    ax1.axhline(y=0, color='r', linestyle='-')
    ax1.set_xlabel('Predicted values')
    ax1.set_ylabel('Residuals')
    ax1.set_title('Residuals vs Predicted values')
    
    # Plot residuals histogram
    ax2.hist(residuals, bins=30, alpha=0.75)
    ax2.axvline(x=0, color='r', linestyle='-')
    ax2.set_xlabel('Residuals')
    ax2.set_ylabel('Frequency')
    ax2.set_title('Residuals Distribution')
    
    # Adjust layout
    fig.tight_layout()
    
    return fig


def evaluate_model(y_true: np.ndarray, 
                  y_pred: np.ndarray, 
                  y_prob: Optional[np.ndarray] = None,
                  task_type: str = 'classification',
                  class_names: Optional[List[str]] = None,
                  output_dir: Optional[str] = None,
                  logger: Optional[logging.Logger] = None) -> Dict[str, float]:
    """
    Evaluate model performance.
    
    Args:
        y_true: True labels/values
        y_pred: Predicted labels/values
        y_prob: Predicted probabilities (for classification)
        task_type: Type of task ('classification' or 'regression')
        class_names: Names of classes (for classification)
        output_dir: Directory to save plots
        logger: Logger
        
    Returns:
        Dictionary with metrics
    """
    # Compute metrics
    if task_type.lower() == 'classification':
        metrics = compute_classification_metrics(y_true, y_pred, y_prob)
        
        # Log classification report
        if logger is not None:
            report = classification_report(y_true, y_pred, target_names=class_names)
            logger.info(f"Classification Report:\n{report}")
        
        # Generate plots if output directory is provided
        if output_dir is not None:
            import os
            os.makedirs(output_dir, exist_ok=True)
            
            # Confusion matrix
            fig = plot_confusion_matrix(y_true, y_pred, class_names)
            fig.savefig(os.path.join(output_dir, "confusion_matrix.png"))
            plt.close(fig)
            
            # Normalized confusion matrix
            fig = plot_confusion_matrix(y_true, y_pred, class_names, normalize=True)
            fig.savefig(os.path.join(output_dir, "confusion_matrix_normalized.png"))
            plt.close(fig)
            
            # ROC curve (if probabilities provided)
            if y_prob is not None:
                # For binary classification or multi-class with probabilities
                if len(np.unique(y_true)) == 2 or (len(y_prob.shape) > 1 and y_prob.shape[1] > 1):
                    fig = plot_roc_curve(y_true, y_prob, class_names)
                    fig.savefig(os.path.join(output_dir, "roc_curve.png"))
                    plt.close(fig)
                    
                    # Precision-recall curve
                    fig = plot_precision_recall_curve(y_true, y_prob, class_names)
                    fig.savefig(os.path.join(output_dir, "precision_recall_curve.png"))
                    plt.close(fig)
    
    elif task_type.lower() == 'regression':
        metrics = compute_regression_metrics(y_true, y_pred)
        
        # Generate plots if output directory is provided
        if output_dir is not None:
            import os
            os.makedirs(output_dir, exist_ok=True)
            
            # Residuals plot
            fig = plot_residuals(y_true, y_pred)
            fig.savefig(os.path.join(output_dir, "residuals_plot.png"))
            plt.close(fig)
            
            # Actual vs Predicted
            fig, ax = plt.subplots(figsize=(10, 8))
            ax.scatter(y_true, y_pred, alpha=0.5)
            ax.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'k--', lw=2)
            ax.set_xlabel('Actual')
            ax.set_ylabel('Predicted')
            ax.set_title('Actual vs Predicted')
            fig.tight_layout()
            fig.savefig(os.path.join(output_dir, "actual_vs_predicted.png"))
            plt.close(fig)
    
    else:
        raise ValueError(f"Task type {task_type} not supported")
    
    # Log metrics
    if logger is not None:
        logger.info("Evaluation Metrics:")
        for k, v in metrics.items():
            logger.info(f"  {k}: {v:.4f}")
    
    return metrics


def evaluate_ensemble(predictions: List[np.ndarray], 
                     y_true: np.ndarray, 
                     weights: Optional[List[float]] = None,
                     task_type: str = 'classification',
                     class_names: Optional[List[str]] = None,
                     output_dir: Optional[str] = None,
                     logger: Optional[logging.Logger] = None) -> Dict[str, Dict[str, float]]:
    """
    Evaluate ensemble of models.
    
    Args:
        predictions: List of model predictions
        y_true: True labels/values
        weights: Optional weights for each model
        task_type: Type of task ('classification' or 'regression')
        class_names: Names of classes (for classification)
        output_dir: Directory to save plots
        logger: Logger
        
    Returns:
        Dictionary with metrics for each model and the ensemble
    """
    # Initialize results
    results = {}
    
    # Evaluate each individual model
    for i, preds in enumerate(predictions):
        if logger is not None:
            logger.info(f"Evaluating model {i+1}...")
        
        if task_type.lower() == 'classification':
            # For classification, we need to convert probabilities to class predictions
            if len(preds.shape) > 1 and preds.shape[1] > 1:
                # Multi-class
                y_pred = np.argmax(preds, axis=1)
                y_prob = preds
            else:
                # Binary
                y_pred = (preds > 0.5).astype(int)
                y_prob = preds
            
            metrics = evaluate_model(
                y_true=y_true,
                y_pred=y_pred,
                y_prob=y_prob,
                task_type=task_type,
                class_names=class_names,
                output_dir=None,  # Don't save plots for individual models
                logger=None  # Don't log for individual models
            )
        else:
            # For regression, predictions are already the values
            metrics = evaluate_model(
                y_true=y_true,
                y_pred=preds,
                task_type=task_type,
                output_dir=None,
                logger=None
            )
        
        results[f"model_{i+1}"] = metrics
    
    # Evaluate ensemble
    if logger is not None:
        logger.info("Evaluating ensemble...")
    
    # Combine predictions with weights
    from src.models.inference import ensemble_predictions
    ensemble_preds = ensemble_predictions(predictions, weights)
    
    if task_type.lower() == 'classification':
        # For classification, we need to convert probabilities to class predictions
        if len(ensemble_preds.shape) > 1 and ensemble_preds.shape[1] > 1:
            # Multi-class
            y_pred = np.argmax(ensemble_preds, axis=1)
            y_prob = ensemble_preds
        else:
            # Binary
            y_pred = (ensemble_preds > 0.5).astype(int)
            y_prob = ensemble_preds
        
        metrics = evaluate_model(
            y_true=y_true,
            y_pred=y_pred,
            y_prob=y_prob,
            task_type=task_type,
            class_names=class_names,
            output_dir=output_dir,
            logger=logger
        )
    else:
        # For regression, predictions are already the values
        metrics = evaluate_model(
            y_true=y_true,
            y_pred=ensemble_preds,
            task_type=task_type,
            output_dir=output_dir,
            logger=logger
        )
    
    results["ensemble"] = metrics
    
    return results 
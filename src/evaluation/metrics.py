"""
Competition metrics for BirdCLEF 2025.
"""
import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score


def calculate_competition_metrics(y_true: np.ndarray, y_pred: np.ndarray, target_columns: list) -> dict:
    """
    Calculate competition metrics for BirdCLEF.
    
    Args:
        y_true: True labels
        y_pred: Predicted probabilities
        target_columns: List of target column names
        
    Returns:
        Dictionary with metrics
    """
    # Calculate metrics
    metrics = {}
    
    # Calculate Mean Average Precision
    average_precision = []
    for i in range(len(target_columns)):
        if np.sum(y_true[:, i]) > 0:
            average_precision.append(
                average_precision_score(y_true[:, i], y_pred[:, i])
            )
    
    # Calculate mAP score
    metrics["mAP"] = np.mean(average_precision)
    
    # Calculate ROC-AUC
    roc_scores = []
    for i in range(len(target_columns)):
        if np.sum(y_true[:, i]) > 0 and np.sum(y_true[:, i]) < len(y_true[:, i]):
            roc_scores.append(
                roc_auc_score(y_true[:, i], y_pred[:, i])
            )
    
    metrics["ROC"] = np.mean(roc_scores)
    
    # Calculate cmAP@1 and cmAP@5
    # This implementation follows competition-specific logic for calculating cmAP at specific thresholds
    top_n_classes = [1, 5]
    for n in top_n_classes:
        # Get top n predictions for each sample
        top_n_indices = np.argsort(-y_pred, axis=1)[:, :n]
        
        # Calculate cmAP@n
        cmAP = 0
        valid_samples = 0
        
        for i in range(len(y_true)):
            # Get ground truth positives
            gt_pos = np.where(y_true[i] > 0)[0]
            
            if len(gt_pos) > 0:
                # Calculate precision at n
                intersect = np.intersect1d(gt_pos, top_n_indices[i])
                precision = len(intersect) / min(n, len(gt_pos))
                cmAP += precision
                valid_samples += 1
        
        if valid_samples > 0:
            metrics[f"cmAP_{n}"] = cmAP / valid_samples
        else:
            metrics[f"cmAP_{n}"] = 0.0
    
    return metrics


def calculate_competition_metrics_no_map(y_true: np.ndarray, y_pred: np.ndarray, target_columns: list) -> dict:
    """
    Calculate competition metrics without mAP (for training monitoring).
    
    Args:
        y_true: True labels
        y_pred: Predicted probabilities
        target_columns: List of target column names
        
    Returns:
        Dictionary with metrics
    """
    # Calculate metrics
    metrics = {}
    
    # Calculate cmAP@1 and cmAP@5
    top_n_classes = [1, 5]
    for n in top_n_classes:
        # Get top n predictions for each sample
        top_n_indices = np.argsort(-y_pred, axis=1)[:, :n]
        
        # Calculate cmAP@n
        cmAP = 0
        valid_samples = 0
        
        for i in range(len(y_true)):
            # Get ground truth positives
            gt_pos = np.where(y_true[i] > 0)[0]
            
            if len(gt_pos) > 0:
                # Calculate precision at n
                intersect = np.intersect1d(gt_pos, top_n_indices[i])
                precision = len(intersect) / min(n, len(gt_pos))
                cmAP += precision
                valid_samples += 1
        
        if valid_samples > 0:
            metrics[f"cmAP_{n}"] = cmAP / valid_samples
        else:
            metrics[f"cmAP_{n}"] = 0.0
    
    return metrics


def metrics_to_string(metrics: dict, prefix: str = "") -> str:
    """
    Convert metrics dictionary to string for logging.
    
    Args:
        metrics: Dictionary with metrics
        prefix: Prefix for metric names
        
    Returns:
        String representation of metrics
    """
    result = ""
    for k, v in metrics.items():
        if result:
            result += ", "
        result += f"{prefix} {k} : {v:.4f}"
    
    return result 
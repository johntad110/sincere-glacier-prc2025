"""
Evaluation metrics and reporting utilities.
"""
import numpy as np
from typing import List, Tuple
from sklearn.metrics import mean_squared_error


def calculate_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate Root Mean Squared Error.
    
    Args:
        y_true: True values
        y_pred: Predicted values
        
    Returns:
        RMSE value
    """
    return np.sqrt(mean_squared_error(y_true, y_pred))


def print_cv_results(scores: List[float], metric_name: str = "RMSE") -> None:
    """
    Print cross-validation results summary.
    
    Args:
        scores: List of scores from each fold
        metric_name: Name of the metric
    """
    print(f"\n{'='*60}")
    print(f"Cross-Validation {metric_name} Results")
    print(f"{'='*60}")
    
    for i, score in enumerate(scores, 1):
        print(f"Fold {i}: {score:.4f}")
    
    print(f"{'-'*60}")
    print(f"Mean {metric_name}: {np.mean(scores):.4f} (+/- {np.std(scores):.4f})")
    print(f"Min {metric_name}: {np.min(scores):.4f}")
    print(f"Max {metric_name}: {np.max(scores):.4f}")
    print(f"{'='*60}\n")


def get_fold_statistics(scores: List[float]) -> Tuple[float, float, float, float]:
    """
    Get statistical summary of fold scores.
    
    Args:
        scores: List of scores from each fold
        
    Returns:
        Tuple of (mean, std, min, max)
    """
    return (
        np.mean(scores),
        np.std(scores),
        np.min(scores),
        np.max(scores)
    )

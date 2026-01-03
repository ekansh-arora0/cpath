"""
Evaluation Metrics for CPath-Omni

Standard classification metrics for pathology evaluation.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from sklearn.metrics import (
    roc_auc_score, accuracy_score, precision_score, recall_score, 
    f1_score, confusion_matrix, roc_curve, precision_recall_curve,
    average_precision_score
)


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_score: Optional[np.ndarray] = None
) -> Dict[str, float]:
    """
    Compute standard classification metrics.
    
    Args:
        y_true: Ground truth labels (binary)
        y_pred: Predicted labels (binary)
        y_score: Prediction scores/probabilities (for AUC)
    
    Returns:
        Dictionary with metrics
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1_score": f1_score(y_true, y_pred, zero_division=0),
        "specificity": _compute_specificity(y_true, y_pred),
    }
    
    # AUC metrics (require scores)
    if y_score is not None:
        y_score = np.asarray(y_score)
        
        if len(np.unique(y_true)) > 1:
            metrics["auc_roc"] = roc_auc_score(y_true, y_score)
            metrics["auc_pr"] = average_precision_score(y_true, y_score)
        else:
            metrics["auc_roc"] = 0.5
            metrics["auc_pr"] = 0.5
    
    # Confusion matrix values
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    metrics["true_positives"] = int(tp)
    metrics["true_negatives"] = int(tn)
    metrics["false_positives"] = int(fp)
    metrics["false_negatives"] = int(fn)
    
    return metrics


def _compute_specificity(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute specificity (true negative rate)."""
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return tn / (tn + fp) if (tn + fp) > 0 else 0.0


def compute_confidence_interval(
    values: np.ndarray,
    confidence: float = 0.95,
    method: str = "percentile"
) -> Tuple[float, float, float]:
    """
    Compute confidence interval for a set of values.
    
    Args:
        values: Array of values (e.g., from bootstrap runs)
        confidence: Confidence level (default: 0.95 for 95% CI)
        method: 'percentile' or 'normal'
    
    Returns:
        Tuple of (mean, lower_bound, upper_bound)
    """
    values = np.asarray(values)
    mean = values.mean()
    
    if method == "percentile":
        alpha = (1 - confidence) / 2
        lower = np.percentile(values, alpha * 100)
        upper = np.percentile(values, (1 - alpha) * 100)
    else:  # normal approximation
        from scipy import stats
        sem = values.std() / np.sqrt(len(values))
        z = stats.norm.ppf((1 + confidence) / 2)
        lower = mean - z * sem
        upper = mean + z * sem
    
    return mean, lower, upper


def bootstrap_auc(
    y_true: np.ndarray,
    y_score: np.ndarray,
    n_bootstraps: int = 1000,
    confidence: float = 0.95,
    random_state: int = 42
) -> Dict[str, float]:
    """
    Compute AUC with bootstrap confidence interval.
    
    Args:
        y_true: Ground truth labels
        y_score: Prediction scores
        n_bootstraps: Number of bootstrap samples
        confidence: Confidence level
        random_state: Random seed
    
    Returns:
        Dictionary with mean AUC and confidence interval
    """
    rng = np.random.RandomState(random_state)
    n_samples = len(y_true)
    
    aucs = []
    for _ in range(n_bootstraps):
        # Bootstrap sample
        indices = rng.randint(0, n_samples, n_samples)
        y_true_boot = y_true[indices]
        y_score_boot = y_score[indices]
        
        # Skip if only one class in sample
        if len(np.unique(y_true_boot)) < 2:
            continue
        
        aucs.append(roc_auc_score(y_true_boot, y_score_boot))
    
    aucs = np.array(aucs)
    mean, lower, upper = compute_confidence_interval(aucs, confidence)
    
    return {
        "auc_mean": mean,
        "auc_std": aucs.std(),
        "auc_ci_lower": lower,
        "auc_ci_upper": upper,
        "confidence_level": confidence
    }


def compute_optimal_threshold(
    y_true: np.ndarray,
    y_score: np.ndarray,
    criterion: str = "youden"
) -> Tuple[float, Dict]:
    """
    Find optimal classification threshold.
    
    Args:
        y_true: Ground truth labels
        y_score: Prediction scores
        criterion: 'youden' (maximize TPR - FPR) or 'f1' (maximize F1)
    
    Returns:
        Tuple of (optimal_threshold, metrics_at_threshold)
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    
    if criterion == "youden":
        # Youden's J statistic
        j_scores = tpr - fpr
        optimal_idx = np.argmax(j_scores)
    elif criterion == "f1":
        # Maximize F1
        f1_scores = []
        for thresh in thresholds:
            y_pred = (y_score >= thresh).astype(int)
            f1_scores.append(f1_score(y_true, y_pred, zero_division=0))
        optimal_idx = np.argmax(f1_scores)
    else:
        raise ValueError(f"Unknown criterion: {criterion}")
    
    optimal_threshold = thresholds[optimal_idx]
    
    # Compute metrics at optimal threshold
    y_pred_optimal = (y_score >= optimal_threshold).astype(int)
    metrics = compute_metrics(y_true, y_pred_optimal, y_score)
    metrics["threshold"] = optimal_threshold
    
    return optimal_threshold, metrics


def compare_methods(
    results: Dict[str, Dict],
    metric: str = "auc_roc"
) -> Dict:
    """
    Compare multiple methods and compute improvement.
    
    Args:
        results: Dictionary mapping method names to their results
        metric: Metric to compare
    
    Returns:
        Dictionary with comparison statistics
    """
    comparison = {}
    
    methods = list(results.keys())
    values = {m: results[m].get(metric, 0) for m in methods}
    
    # Find best method
    best_method = max(values, key=values.get)
    baseline = methods[0] if len(methods) > 0 else None
    
    comparison["methods"] = methods
    comparison["values"] = values
    comparison["best_method"] = best_method
    comparison["best_value"] = values[best_method]
    
    # Compute improvements over baseline
    if baseline and baseline != best_method:
        improvement = values[best_method] - values[baseline]
        comparison["improvement_over_baseline"] = improvement
        comparison["relative_improvement"] = improvement / values[baseline] if values[baseline] > 0 else 0
    
    return comparison

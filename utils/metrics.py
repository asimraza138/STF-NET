"""
Evaluation metrics — Section 4.2 of the manuscript.

Implements: Accuracy, Precision, Recall, F1-Score, AUC.
All metrics operate on numpy arrays of predictions and ground-truth labels.
"""

import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix
)


def compute_metrics(y_true:     np.ndarray,
                    y_prob:     np.ndarray,
                    threshold:  float = 0.5) -> dict:
    """
    Compute all evaluation metrics reported in the manuscript.

    Args:
        y_true    : (N,) integer labels {0, 1}
        y_prob    : (N,) predicted probabilities ∈ [0, 1]
        threshold : decision threshold for binary predictions
    Returns:
        metrics dict with keys: accuracy, precision, recall, f1, auc, cm
    """
    y_pred = (y_prob >= threshold).astype(int)

    acc  = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec  = recall_score(y_true, y_pred, zero_division=0)
    f1   = f1_score(y_true, y_pred, zero_division=0)

    try:
        auc = roc_auc_score(y_true, y_prob)
    except ValueError:
        auc = float("nan")

    cm = confusion_matrix(y_true, y_pred)

    # False positive / negative rates
    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)
    fpr = fp / (fp + tn + 1e-8)
    fnr = fn / (fn + tp + 1e-8)

    return {
        "accuracy"   : float(acc),
        "precision"  : float(prec),
        "recall"     : float(rec),
        "f1"         : float(f1),
        "auc"        : float(auc),
        "fpr"        : float(fpr),
        "fnr"        : float(fnr),
        "cm"         : cm,
    }


def aggregate_seeds(results_per_seed: list[dict]) -> dict:
    """
    Aggregate metrics over multiple random seeds.

    Args:
        results_per_seed: list of metric dicts (one per seed)
    Returns:
        dict with mean and std for each scalar metric
    """
    keys = [k for k in results_per_seed[0] if k != "cm"]
    aggregated = {}
    for k in keys:
        vals = [r[k] for r in results_per_seed]
        aggregated[f"{k}_mean"] = float(np.mean(vals))
        aggregated[f"{k}_std"]  = float(np.std(vals, ddof=1))
    return aggregated


def format_results(metrics: dict) -> str:
    """Format a metrics dict as a readable string for logging."""
    lines = []
    for k, v in metrics.items():
        if k == "cm":
            lines.append(f"  Confusion matrix:\n{v}")
        elif isinstance(v, float):
            lines.append(f"  {k:12s}: {v:.4f}")
    return "\n".join(lines)

"""Evaluation metrics, calibration analysis, and threshold policy optimization."""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)

logger = logging.getLogger(__name__)


def compute_binary_metrics(
    y_true: np.ndarray,
    y_score: np.ndarray,
    threshold: float = 0.5,
) -> Dict[str, Any]:
    """Computes comprehensive classification metrics at a given decision threshold."""
    y_true = np.asarray(y_true, dtype=int)
    y_score = np.asarray(y_score, dtype=float)
    y_pred = (y_score >= threshold).astype(int)

    total_samples = len(y_true)
    positive_count = int(np.sum(y_true))
    negative_count = total_samples - positive_count
    prevalence = positive_count / total_samples if total_samples > 0 else 0.0

    # Handle edge case where there is only one class
    if positive_count == 0 or negative_count == 0:
        logger.warning("Single class in evaluation partition. Some metrics are undefined.")
        return {
            "sample_count": total_samples,
            "positive_count": positive_count,
            "prevalence": prevalence,
            "status": "INSUFFICIENT_DATA",
            "threshold": threshold,
        }

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    review_rate = (tp + fp) / total_samples if total_samples > 0 else 0.0

    # Ranking metrics
    ap = average_precision_score(y_true, y_score)
    roc_auc = roc_auc_score(y_true, y_score)
    brier = brier_score_loss(y_true, y_score)

    return {
        "sample_count": total_samples,
        "positive_count": positive_count,
        "negative_count": negative_count,
        "prevalence": float(prevalence),
        "threshold": float(threshold),
        "tp": int(tp),
        "fp": int(fp),
        "tn": int(tn),
        "fn": int(fn),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "fpr": float(fpr),
        "review_rate": float(review_rate),
        "average_precision": float(ap),
        "roc_auc": float(roc_auc),
        "brier_score": float(brier),
        "status": "SUCCESS",
    }


def optimize_threshold_fpr_budget(
    y_true: np.ndarray,
    y_score: np.ndarray,
    max_fpr: float = 0.01,
    min_precision: float = 0.10,
    default_threshold: float = 0.50,
) -> Tuple[float, Dict[str, Any]]:
    """Selects decision threshold to maximize recall subject to an FPR budget.

    Operates strictly on validation/tuning partitions.
    """
    y_true = np.asarray(y_true, dtype=int)
    y_score = np.asarray(y_score, dtype=float)

    # Use candidate thresholds from PR curve
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_score)

    best_threshold = default_threshold
    best_recall = -1.0
    best_metrics = None

    # Evaluate each threshold in sorted order
    # Sample a grid of thresholds for efficiency
    candidate_thresholds = np.unique(
        np.concatenate([np.linspace(0.01, 0.99, 100), thresholds])
    )

    for thresh in candidate_thresholds:
        metrics = compute_binary_metrics(y_true, y_score, threshold=thresh)
        if metrics.get("status") != "SUCCESS":
            continue

        fpr = metrics["fpr"]
        rec = metrics["recall"]
        prec = metrics["precision"]

        # Check constraint satisfaction
        if fpr <= max_fpr and prec >= min_precision:
            if rec > best_recall:
                best_recall = rec
                best_threshold = float(thresh)
                best_metrics = metrics

    # If no threshold meets strict constraint, select threshold closest to max_fpr
    if best_metrics is None:
        logger.warning(
            "No threshold met FPR <= %.3f with precision >= %.3f. Falling back to default %.2f",
            max_fpr,
            min_precision,
            default_threshold,
        )
        best_threshold = default_threshold
        best_metrics = compute_binary_metrics(y_true, y_score, threshold=default_threshold)

    return best_threshold, best_metrics


def generate_threshold_tradeoff_table(
    y_true: np.ndarray,
    y_score: np.ndarray,
    thresholds: Optional[List[float]] = None,
) -> List[Dict[str, Any]]:
    """Generates an evaluation table across multiple threshold points."""
    if thresholds is None:
        thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

    tradeoffs = []
    for thresh in thresholds:
        m = compute_binary_metrics(y_true, y_score, threshold=thresh)
        tradeoffs.append(m)

    return tradeoffs

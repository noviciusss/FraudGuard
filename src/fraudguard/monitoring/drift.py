"""Feature and prediction drift detection engine with statistical adapters."""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from scipy.spatial.distance import jensenshannon

logger = logging.getLogger(__name__)


def calculate_psi(
    expected: np.ndarray,
    actual: np.ndarray,
    num_bins: int = 10,
    epsilon: float = 1e-4,
) -> float:
    """Calculates the Population Stability Index (PSI) between reference and current samples."""
    expected = expected[~np.isnan(expected)]
    actual = actual[~np.isnan(actual)]

    if len(expected) < 10 or len(actual) < 10:
        return 0.0

    # Quantile binning on expected distribution
    percentiles = np.linspace(0, 100, num_bins + 1)
    bin_edges = np.percentile(expected, percentiles)
    bin_edges[0] = -np.inf
    bin_edges[-1] = np.inf
    bin_edges = np.unique(bin_edges)

    expected_counts, _ = np.histogram(expected, bins=bin_edges)
    actual_counts, _ = np.histogram(actual, bins=bin_edges)

    expected_pct = (expected_counts / len(expected)) + epsilon
    actual_pct = (actual_counts / len(actual)) + epsilon

    expected_pct /= expected_pct.sum()
    actual_pct /= actual_pct.sum()

    psi = np.sum((actual_pct - expected_pct) * np.log(actual_pct / expected_pct))
    return float(max(0.0, psi))


def calculate_categorical_divergence(
    expected_categories: pd.Series,
    actual_categories: pd.Series,
) -> float:
    """Calculates Jensen-Shannon distance for categorical feature distributions."""
    all_cats = list(set(expected_categories.dropna().unique()) | set(actual_categories.dropna().unique()))
    if not all_cats:
        return 0.0

    exp_counts = expected_categories.value_counts(normalize=True).to_dict()
    act_counts = actual_categories.value_counts(normalize=True).to_dict()

    p = np.array([exp_counts.get(cat, 1e-5) for cat in all_cats])
    q = np.array([act_counts.get(cat, 1e-5) for cat in all_cats])

    p /= p.sum()
    q /= q.sum()

    js_dist = jensenshannon(p, q)
    return float(js_dist)


class DriftMonitor:
    """Monitors covariate (feature) and prediction drift against a reference distribution."""

    def __init__(
        self,
        reference_df: pd.DataFrame,
        reference_version: str = "v1_dev_train",
        warning_threshold: float = 0.10,
        alert_threshold: float = 0.20,
        min_window_samples: int = 50,
    ):
        self.reference_df = reference_df.copy()
        self.reference_version = reference_version
        self.warning_threshold = warning_threshold
        self.alert_threshold = alert_threshold
        self.min_window_samples = min_window_samples

    def evaluate_window_drift(
        self,
        current_df: pd.DataFrame,
        features: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Evaluates drift on a window of current transactions."""
        sample_count = len(current_df)
        if sample_count < self.min_window_samples:
            return {
                "status": "INSUFFICIENT_DATA",
                "sample_count": sample_count,
                "message": f"Window sample count ({sample_count}) below minimum threshold ({self.min_window_samples})",
                "features": {},
                "max_drift_score": 0.0,
            }

        target_features = features or ["amount", "type"]
        feature_reports: Dict[str, Any] = {}
        drift_scores: List[float] = []

        for feat in target_features:
            if feat not in self.reference_df.columns or feat not in current_df.columns:
                continue

            ref_col = self.reference_df[feat]
            curr_col = current_df[feat]

            if pd.api.types.is_numeric_dtype(ref_col):
                drift_score = calculate_psi(ref_col.to_numpy(), curr_col.to_numpy())
                method = "psi"
            else:
                drift_score = calculate_categorical_divergence(ref_col, curr_col)
                method = "jensen_shannon"

            drift_scores.append(drift_score)
            feat_status = "NORMAL"
            if drift_score >= self.alert_threshold:
                feat_status = "ALERT"
            elif drift_score >= self.warning_threshold:
                feat_status = "WARNING"

            feature_reports[feat] = {
                "method": method,
                "score": round(drift_score, 4),
                "status": feat_status,
            }

        max_drift = max(drift_scores) if drift_scores else 0.0
        overall_status = "NORMAL"
        if max_drift >= self.alert_threshold:
            overall_status = "ALERT"
        elif max_drift >= self.warning_threshold:
            overall_status = "WARNING"

        return {
            "status": overall_status,
            "sample_count": sample_count,
            "reference_version": self.reference_version,
            "max_drift_score": round(max_drift, 4),
            "features": feature_reports,
        }

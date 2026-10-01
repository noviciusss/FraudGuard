"""Unit tests for Phase 1: Training routines, metric evaluations, and threshold optimization."""

import numpy as np

from fraudguard.data.prepare import generate_synthetic_paysim
from fraudguard.data.splits import split_by_complete_steps
from fraudguard.training.evaluate import (
    compute_binary_metrics,
    optimize_threshold_fpr_budget,
)
from fraudguard.training.train import (
    predict_pipeline_scores,
    train_baseline_model,
    train_xgboost_model,
)


def test_metric_computation():
    """Verify calculation of precision, recall, FPR, and AP."""
    y_true = np.array([0, 0, 0, 0, 1, 1])
    y_score = np.array([0.1, 0.2, 0.3, 0.4, 0.8, 0.9])

    metrics = compute_binary_metrics(y_true, y_score, threshold=0.5)
    assert metrics["status"] == "SUCCESS"
    assert metrics["tp"] == 2
    assert metrics["fp"] == 0
    assert metrics["tn"] == 4
    assert metrics["fn"] == 0
    assert np.isclose(metrics["precision"], 1.0)
    assert np.isclose(metrics["recall"], 1.0)
    assert np.isclose(metrics["fpr"], 0.0)
    assert np.isclose(metrics["average_precision"], 1.0)


def test_threshold_optimization_fpr_budget():
    """Verify threshold selector respects max FPR budget constraint."""
    y_true = np.array([0, 0, 0, 0, 0, 0, 0, 0, 1, 1])
    # Scores designed with one false positive at threshold 0.3
    y_score = np.array([0.1, 0.1, 0.1, 0.2, 0.2, 0.2, 0.3, 0.6, 0.7, 0.9])

    # If max_fpr = 0.0, threshold must eliminate the highest negative score (0.6)
    best_thresh, metrics = optimize_threshold_fpr_budget(
        y_true, y_score, max_fpr=0.0, min_precision=0.5
    )
    assert metrics["fpr"] == 0.0
    assert best_thresh > 0.6
    assert metrics["recall"] == 1.0


def test_training_pipelines_end_to_end():
    """Verify end-to-end training and inference for Baseline and XGBoost."""
    df = generate_synthetic_paysim(num_rows=2000, num_steps=20, random_seed=42)
    partitions = split_by_complete_steps(df, train_ratio=0.6, tune_ratio=0.2, replay_ratio=0.1, audit_ratio=0.1)

    # 1. Baseline Logistic Regression
    baseline_pipe = train_baseline_model(partitions.train, random_state=42)
    base_scores = predict_pipeline_scores(baseline_pipe, partitions.tune)
    assert len(base_scores) == len(partitions.tune)
    assert (base_scores >= 0.0).all() and (base_scores <= 1.0).all()

    # 2. XGBoost
    xgb_pipe = train_xgboost_model(partitions.train, n_estimators=20, max_depth=3, random_state=42)
    xgb_scores = predict_pipeline_scores(xgb_pipe, partitions.tune)
    assert len(xgb_scores) == len(partitions.tune)
    assert (xgb_scores >= 0.0).all() and (xgb_scores <= 1.0).all()

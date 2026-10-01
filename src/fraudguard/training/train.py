"""Model training pipeline comparing baseline Logistic Regression and XGBoost on PaySim temporal splits."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

from fraudguard.data.prepare import load_dataset
from fraudguard.data.splits import DatasetPartitions, split_by_complete_steps
from fraudguard.features.pipeline import build_fraud_pipeline
from fraudguard.training.evaluate import (
    compute_binary_metrics,
    generate_threshold_tradeoff_table,
    optimize_threshold_fpr_budget,
)

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")


def train_baseline_model(
    train_df: pd.DataFrame,
    target_col: str = "isFraud",
    step_col: str = "step",
    random_state: int = 42,
) -> Any:
    """Trains a LogisticRegression baseline pipeline on the training partition."""
    logger.info("Training baseline LogisticRegression pipeline...")
    clf = LogisticRegression(
        class_weight="balanced",
        max_iter=1000,
        random_state=random_state,
        solver="lbfgs",
    )
    pipeline = build_fraud_pipeline(model=clf, step_col=step_col)

    X_train = train_df.drop(columns=[target_col], errors="ignore")
    y_train = train_df[target_col].to_numpy()

    pipeline.fit(X_train, y_train)
    return pipeline


def train_xgboost_model(
    train_df: pd.DataFrame,
    target_col: str = "isFraud",
    step_col: str = "step",
    n_estimators: int = 150,
    max_depth: int = 5,
    learning_rate: float = 0.08,
    scale_pos_weight: Optional[float] = None,
    random_state: int = 42,
) -> Any:
    """Trains an XGBoost pipeline on the training partition."""
    logger.info("Training XGBoost pipeline...")
    y_train = train_df[target_col].to_numpy()

    # Calculate class weight if not specified
    if scale_pos_weight is None:
        neg_count = int(np.sum(y_train == 0))
        pos_count = int(np.sum(y_train == 1))
        scale_pos_weight = float(neg_count / max(1, pos_count))
        # Dampen extreme weights to prevent excessive false positives
        scale_pos_weight = min(scale_pos_weight, 50.0)

    clf = XGBClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        scale_pos_weight=scale_pos_weight,
        eval_metric="logloss",
        random_state=random_state,
        n_jobs=-1,
    )
    pipeline = build_fraud_pipeline(model=clf, step_col=step_col)

    X_train = train_df.drop(columns=[target_col], errors="ignore")
    pipeline.fit(X_train, y_train)
    return pipeline


def predict_pipeline_scores(pipeline: Any, df: pd.DataFrame) -> np.ndarray:
    """Extracts continuous fraud scores P(y=1) from pipeline."""
    # Ensure forbidden columns or target are not causing issues
    probs = pipeline.predict_proba(df)
    return probs[:, 1]


def run_training_experiment(
    data_path: Optional[str] = None,
    output_dir: str = "artifacts/phase1",
    max_fpr: float = 0.01,
    min_precision: float = 0.10,
    random_seed: int = 42,
) -> Dict[str, Any]:
    """Executes the full Phase 1 training, comparison, and evaluation pipeline."""
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load data
    df = load_dataset(path=data_path, fallback_generate=True, random_seed=random_seed)

    # 2. Strict step-based temporal partition
    partitions: DatasetPartitions = split_by_complete_steps(
        df=df,
        step_col="step",
        train_ratio=0.60,
        tune_ratio=0.15,
        replay_ratio=0.15,
        audit_ratio=0.10,
    )

    manifest_path = out_dir / "data_manifest.json"
    partitions.manifest.save(manifest_path)

    # 3. Train Baseline
    baseline_pipeline = train_baseline_model(partitions.train, random_state=random_seed)

    # 4. Train XGBoost
    xgb_pipeline = train_xgboost_model(partitions.train, random_state=random_seed)

    # 5. Optimize thresholds on validation/tune set ONLY
    tune_y = partitions.tune["isFraud"].to_numpy()
    tune_X = partitions.tune.drop(columns=["isFraud"])

    base_tune_scores = predict_pipeline_scores(baseline_pipeline, tune_X)
    xgb_tune_scores = predict_pipeline_scores(xgb_pipeline, tune_X)

    base_thresh, base_tune_metrics = optimize_threshold_fpr_budget(
        tune_y, base_tune_scores, max_fpr=max_fpr, min_precision=min_precision
    )
    xgb_thresh, xgb_tune_metrics = optimize_threshold_fpr_budget(
        tune_y, xgb_tune_scores, max_fpr=max_fpr, min_precision=min_precision
    )

    # 6. Evaluate on final audit set (Untouched until now)
    audit_y = partitions.audit["isFraud"].to_numpy()
    audit_X = partitions.audit.drop(columns=["isFraud"])

    base_audit_scores = predict_pipeline_scores(baseline_pipeline, audit_X)
    xgb_audit_scores = predict_pipeline_scores(xgb_pipeline, audit_X)

    base_audit_metrics = compute_binary_metrics(audit_y, base_audit_scores, threshold=base_thresh)
    xgb_audit_metrics = compute_binary_metrics(audit_y, xgb_audit_scores, threshold=xgb_thresh)

    # 7. Generate threshold trade-offs for XGBoost on Tune
    tradeoffs = generate_threshold_tradeoff_table(tune_y, xgb_tune_scores)

    # 8. Save artifacts
    xgb_model_path = out_dir / "xgb_fraud_pipeline.joblib"
    joblib.dump(xgb_pipeline, xgb_model_path)

    base_model_path = out_dir / "baseline_pipeline.joblib"
    joblib.dump(baseline_pipeline, base_model_path)

    summary = {
        "dataset_manifest": partitions.manifest.to_dict(),
        "baseline": {
            "validation_threshold": base_thresh,
            "validation_metrics": base_tune_metrics,
            "audit_metrics": base_audit_metrics,
        },
        "xgboost": {
            "validation_threshold": xgb_thresh,
            "validation_metrics": xgb_tune_metrics,
            "audit_metrics": xgb_audit_metrics,
            "threshold_tradeoffs": tradeoffs,
        },
    }

    summary_path = out_dir / "evaluation_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    logger.info("Evaluation Summary:")
    logger.info(
        "Baseline Tune AP: %.4f | Recall: %.4f | FPR: %.4f",
        base_tune_metrics.get("average_precision", 0.0),
        base_tune_metrics.get("recall", 0.0),
        base_tune_metrics.get("fpr", 0.0),
    )
    logger.info(
        "XGBoost  Tune AP: %.4f | Recall: %.4f | FPR: %.4f",
        xgb_tune_metrics.get("average_precision", 0.0),
        xgb_tune_metrics.get("recall", 0.0),
        xgb_tune_metrics.get("fpr", 0.0),
    )
    logger.info(
        "XGBoost Audit AP: %.4f | Recall: %.4f | FPR: %.4f",
        xgb_audit_metrics.get("average_precision", 0.0),
        xgb_audit_metrics.get("recall", 0.0),
        xgb_audit_metrics.get("fpr", 0.0),
    )

    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FraudGuard Model Training Pipeline")
    parser.add_argument("--data", type=str, default=None, help="Path to PaySim dataset (CSV or Parquet)")
    parser.add_argument("--output-dir", type=str, default="artifacts/phase1", help="Output directory for artifacts")
    parser.add_argument("--max-fpr", type=float, default=0.01, help="Max FPR budget on validation")
    args = parser.parse_args()

    run_training_experiment(
        data_path=args.data,
        output_dir=args.output_dir,
        max_fpr=args.max_fpr,
    )

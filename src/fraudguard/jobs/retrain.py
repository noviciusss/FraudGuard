"""Automated candidate model retraining, validation gating, and promotion execution."""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import joblib
import pandas as pd
from sqlalchemy.orm import Session

from fraudguard.data.splits import split_by_complete_steps
from fraudguard.persistence.models import RetrainRequestRecord
from fraudguard.training.evaluate import compute_binary_metrics, optimize_threshold_fpr_budget
from fraudguard.training.gates import GateResult, ModelPromotionGate
from fraudguard.training.train import predict_pipeline_scores, train_xgboost_model

logger = logging.getLogger(__name__)


def execute_retrain_cycle(
    db: Session,
    retrain_request_id: str,
    available_data: pd.DataFrame,
    simulated_cutoff: int,
    champion_pipeline: Any,
    champion_threshold: float,
    champion_version: str = "v1",
    output_dir: str = "artifacts/retrain",
) -> GateResult:
    """Executes the candidate retraining and gating workflow strictly using labels available at cutoff."""
    req: Optional[RetrainRequestRecord] = (
        db.query(RetrainRequestRecord).filter(RetrainRequestRecord.id == retrain_request_id).first()
    )
    if req:
        req.status = "RUNNING"
        req.started_at = datetime.now(timezone.utc)
        req.champion_version = champion_version
        db.commit()

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # 1. Select ONLY labels whose simulated release step <= cutoff
    # Ensure available_data only has rows where step/label_available_step <= cutoff
    cutoff_data = available_data[available_data["step"] <= simulated_cutoff].copy()

    # 2. Split into candidate train (60%), candidate tune (20%), candidate comparison (20%)
    partitions = split_by_complete_steps(
        df=cutoff_data,
        step_col="step",
        train_ratio=0.60,
        tune_ratio=0.20,
        replay_ratio=0.10,
        audit_ratio=0.10,
    )
    # Use replay+audit as the unseen comparison window
    comparison_df = pd.concat([partitions.replay, partitions.audit]).reset_index(drop=True)

    # 3. Fit candidate model on candidate train
    candidate_pipe = train_xgboost_model(partitions.train, n_estimators=100, max_depth=4)

    # 4. Select candidate threshold on candidate tune
    tune_y = partitions.tune["isFraud"].to_numpy()
    tune_scores = predict_pipeline_scores(candidate_pipe, partitions.tune)
    cand_threshold, _ = optimize_threshold_fpr_budget(tune_y, tune_scores, max_fpr=0.01, min_precision=0.10)

    # 5. Evaluate both candidate and champion on the identical comparison window
    comp_y = comparison_df["isFraud"].to_numpy()

    cand_comp_scores = predict_pipeline_scores(candidate_pipe, comparison_df)
    cand_metrics = compute_binary_metrics(comp_y, cand_comp_scores, threshold=cand_threshold)

    champ_comp_scores = predict_pipeline_scores(champion_pipeline, comparison_df)
    champ_metrics = compute_binary_metrics(comp_y, champ_comp_scores, threshold=champion_threshold)

    # 6. Apply Promotion Quality Gate
    gate = ModelPromotionGate(
        max_fpr=0.015,
        min_precision=0.10,
        min_recall_improvement=0.0,
        max_ap_drop=0.03,
        min_comparison_samples=30,
        min_comparison_positives=1,
    )
    gate_result = gate.evaluate_candidate(
        candidate_metrics=cand_metrics,
        champion_metrics=champ_metrics,
    )

    # Save candidate artifact
    cand_version = f"cand_cutoff_{simulated_cutoff}"
    cand_file = out_path / f"candidate_{cand_version}.joblib"
    joblib.dump(candidate_pipe, cand_file)

    # 7. Update database record with decision
    if req:
        req.finished_at = datetime.now(timezone.utc)
        req.candidate_version = cand_version
        req.evaluation_artifact_uri = str(cand_file)
        if gate_result.passed:
            req.status = "APPROVED"
            req.reason = gate_result.reason
        else:
            req.status = "REJECTED"
            req.reason = gate_result.reason
            req.failure_reason = gate_result.reason
        db.commit()

    logger.info(
        "Retrain cycle completed: Status=%s | Reason=%s",
        gate_result.status,
        gate_result.reason,
    )
    return gate_result

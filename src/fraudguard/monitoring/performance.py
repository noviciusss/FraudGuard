"""Production performance evaluation joining predictions with delayed outcome labels."""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

import numpy as np
from sqlalchemy.orm import Session

from fraudguard.persistence.models import (
    MonitoringReportRecord,
    OutcomeRecord,
    PredictionRecord,
    RetrainRequestRecord,
)
from fraudguard.training.evaluate import compute_binary_metrics

logger = logging.getLogger(__name__)


class PerformanceMonitor:
    """Evaluates realized classification metrics against delayed labels and triggers retraining."""

    def __init__(
        self,
        min_labeled_samples: int = 30,
        min_positive_labels: int = 2,
        alert_recall_drop_pct: float = 0.20,
    ):
        self.min_labeled_samples = min_labeled_samples
        self.min_positive_labels = min_positive_labels
        self.alert_recall_drop_pct = alert_recall_drop_pct

    def evaluate_window_performance(
        self,
        db: Session,
        replay_run_id: str,
        window_start: int,
        window_end: int,
        baseline_recall: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Queries predictions and joins available outcomes for the specified simulation step window."""
        # Query predictions with joined outcomes
        query = (
            db.query(
                PredictionRecord.id,
                PredictionRecord.fraud_score,
                PredictionRecord.decision,
                PredictionRecord.decision_threshold,
                PredictionRecord.model_version,
                OutcomeRecord.label,
            )
            .outerjoin(OutcomeRecord, PredictionRecord.id == OutcomeRecord.prediction_id)
            .filter(
                PredictionRecord.replay_run_id == replay_run_id,
                PredictionRecord.event_step >= window_start,
                PredictionRecord.event_step <= window_end,
            )
            .all()
        )

        total_predictions = len(query)
        if total_predictions == 0:
            return {
                "status": "NO_PREDICTIONS",
                "sample_count": 0,
                "labeled_count": 0,
                "positive_count": 0,
                "metrics": {},
            }

        # Filter only labeled rows
        labeled_rows = [row for row in query if row.label is not None]
        labeled_count = len(labeled_rows)
        positive_count = sum(1 for row in labeled_rows if row.label == 1)

        # Check label availability safeguards
        if labeled_count < self.min_labeled_samples or positive_count < self.min_positive_labels:
            logger.info(
                "Window steps %d..%d has %d predictions but only %d labeled (%d pos). Status: PERFORMANCE_UNAVAILABLE",
                window_start,
                window_end,
                total_predictions,
                labeled_count,
                positive_count,
            )
            return {
                "status": "PERFORMANCE_UNAVAILABLE",
                "sample_count": total_predictions,
                "labeled_count": labeled_count,
                "positive_count": positive_count,
                "coverage_pct": round(labeled_count / total_predictions, 4) if total_predictions > 0 else 0.0,
                "metrics": {},
                "message": "Insufficient delayed labels have matured for this window.",
            }

        # Compute metrics
        y_true = np.array([r.label for r in labeled_rows])
        y_scores = np.array([r.fraud_score for r in labeled_rows])
        threshold = labeled_rows[0].decision_threshold

        metrics = compute_binary_metrics(y_true, y_scores, threshold=threshold)

        status = "NORMAL"
        recall = metrics.get("recall", 0.0)
        if baseline_recall is not None and baseline_recall > 0:
            relative_drop = (baseline_recall - recall) / baseline_recall
            if relative_drop >= self.alert_recall_drop_pct:
                status = "ALERT"

        return {
            "status": status,
            "sample_count": total_predictions,
            "labeled_count": labeled_count,
            "positive_count": positive_count,
            "coverage_pct": round(labeled_count / total_predictions, 4),
            "metrics": metrics,
        }

    def record_report_and_check_retrain(
        self,
        db: Session,
        replay_run_id: str,
        window_start: int,
        window_end: int,
        report_kind: str,
        report_data: Dict[str, Any],
        reference_version: str = "v1_dev_train",
    ) -> Optional[RetrainRequestRecord]:
        """Saves a monitoring report and triggers a deduplicated retrain request if ALERT is sustained."""
        report = MonitoringReportRecord(
            replay_run_id=replay_run_id,
            window_start=window_start,
            window_end=window_end,
            reference_version=reference_version,
            sample_count=report_data.get("sample_count", 0),
            positive_label_count=report_data.get("positive_count"),
            report_kind=report_kind,
            metrics=report_data.get("metrics") or report_data.get("features") or {},
            status=report_data.get("status", "NORMAL"),
        )
        db.add(report)
        db.commit()

        # If ALERT condition, queue a deduplicated retraining request
        if report_data.get("status") == "ALERT":
            dedup_key = f"retrain-{replay_run_id}-step-{window_end}"
            existing = (
                db.query(RetrainRequestRecord)
                .filter(RetrainRequestRecord.deduplication_key == dedup_key)
                .first()
            )
            if not existing:
                retrain_req = RetrainRequestRecord(
                    deduplication_key=dedup_key,
                    triggering_report_id=report.id,
                    simulated_cutoff=window_end,
                    status="PENDING",
                    reason=f"Automated trigger from {report_kind} ALERT at window {window_start}..{window_end}",
                )
                db.add(retrain_req)
                db.commit()
                logger.warning("Created deduplicated retrain request: %s", dedup_key)
                return retrain_req

        return None

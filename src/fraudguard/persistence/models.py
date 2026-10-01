"""SQLAlchemy ORM models for FraudGuard persistence layer.

Defines schemas for predictions, delayed outcomes, monitoring reports,
retrain requests, and deployment events.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any, Dict

from sqlalchemy import (
    JSON,
    Boolean,
    Column,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    UniqueConstraint,
)
from sqlalchemy.orm import declarative_base, relationship

Base = declarative_base()


def generate_uuid() -> str:
    return str(uuid.uuid4())


class PredictionRecord(Base):
    """Stores every transaction prediction and serving audit metadata."""

    __tablename__ = "predictions"

    id = Column(String(36), primary_key=True, default=generate_uuid)
    transaction_id = Column(String(64), nullable=False)
    replay_run_id = Column(String(64), nullable=False, default="default")
    received_at = Column(
        DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(timezone.utc),
    )
    event_step = Column(Integer, nullable=False)
    features = Column(JSON, nullable=False)  # Allowlisted features only
    fraud_score = Column(Float, nullable=False)
    decision = Column(Boolean, nullable=False)
    decision_threshold = Column(Float, nullable=False)
    model_name = Column(String(64), nullable=False)
    model_version = Column(String(32), nullable=False)
    schema_version = Column(String(16), nullable=False, default="v1")
    inference_ms = Column(Float, nullable=False)
    scenario_id = Column(String(64), nullable=True)

    # Idempotency constraint: transaction_id must be unique within a replay_run_id
    __table_args__ = (
        UniqueConstraint("replay_run_id", "transaction_id", name="uq_replay_transaction"),
        Index("idx_predictions_event_step", "event_step"),
        Index("idx_predictions_run_step", "replay_run_id", "event_step"),
    )

    outcome = relationship("OutcomeRecord", back_populates="prediction", uselist=False)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "prediction_id": self.id,
            "transaction_id": self.transaction_id,
            "replay_run_id": self.replay_run_id,
            "received_at": self.received_at.isoformat() if self.received_at else None,
            "event_step": self.event_step,
            "features": self.features,
            "fraud_score": self.fraud_score,
            "decision": self.decision,
            "decision_threshold": self.decision_threshold,
            "model_version": self.model_version,
            "schema_version": self.schema_version,
            "inference_ms": self.inference_ms,
        }


class OutcomeRecord(Base):
    """Delayed ground truth labels released by simulated time clock."""

    __tablename__ = "outcomes"

    prediction_id = Column(
        String(36),
        ForeignKey("predictions.id", ondelete="CASCADE"),
        primary_key=True,
    )
    label = Column(Integer, nullable=False)  # 0 or 1
    label_available_step = Column(Integer, nullable=False)
    released_at = Column(
        DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(timezone.utc),
    )
    label_source = Column(String(32), nullable=False, default="synthetic_replay")

    prediction = relationship("PredictionRecord", back_populates="outcome")

    __table_args__ = (
        Index("idx_outcomes_step", "label_available_step"),
    )


class MonitoringReportRecord(Base):
    """Stores data drift, prediction drift, and labeled performance reports."""

    __tablename__ = "monitoring_reports"

    id = Column(String(36), primary_key=True, default=generate_uuid)
    replay_run_id = Column(String(64), nullable=False)
    window_start = Column(Integer, nullable=False)
    window_end = Column(Integer, nullable=False)
    reference_version = Column(String(64), nullable=False)
    sample_count = Column(Integer, nullable=False)
    positive_label_count = Column(Integer, nullable=True)
    report_kind = Column(String(32), nullable=False)  # data_drift, prediction_drift, labeled_performance
    metrics = Column(JSON, nullable=False)
    status = Column(String(32), nullable=False)  # NORMAL, WARNING, ALERT, PERFORMANCE_UNAVAILABLE
    artifact_uri = Column(String(255), nullable=True)
    created_at = Column(
        DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(timezone.utc),
    )


class RetrainRequestRecord(Base):
    """Deduplicated requests for automated candidate model retraining."""

    __tablename__ = "retrain_requests"

    id = Column(String(36), primary_key=True, default=generate_uuid)
    deduplication_key = Column(String(128), unique=True, nullable=False)
    triggering_report_id = Column(String(36), nullable=True)
    simulated_cutoff = Column(Integer, nullable=False)
    status = Column(String(32), nullable=False, default="PENDING")  # PENDING, RUNNING, APPROVED, REJECTED, FAILED
    reason = Column(String(255), nullable=False)
    started_at = Column(DateTime(timezone=True), nullable=True)
    finished_at = Column(DateTime(timezone=True), nullable=True)
    candidate_version = Column(String(32), nullable=True)
    champion_version = Column(String(32), nullable=True)
    data_manifest_uri = Column(String(255), nullable=True)
    evaluation_artifact_uri = Column(String(255), nullable=True)
    failure_reason = Column(String(255), nullable=True)


class DeploymentEventRecord(Base):
    """Lifecycle audit log of promotions, verifications, and rollbacks."""

    __tablename__ = "deployment_events"

    id = Column(String(36), primary_key=True, default=generate_uuid)
    candidate_version = Column(String(32), nullable=False)
    previous_version = Column(String(32), nullable=True)
    status = Column(String(32), nullable=False)  # SUCCESS, FAILED, ROLLED_BACK
    gate_summary = Column(JSON, nullable=False)
    deployment_revision = Column(String(64), nullable=True)
    started_at = Column(
        DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(timezone.utc),
    )
    verified_at = Column(DateTime(timezone=True), nullable=True)
    rollback_reason = Column(String(255), nullable=True)

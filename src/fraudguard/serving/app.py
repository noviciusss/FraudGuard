"""FastAPI inference service with fail-closed audit logging, immutable versioning, and idempotency."""

from __future__ import annotations

import json
import logging
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Optional

import pandas as pd
from fastapi import Depends, FastAPI, HTTPException, status
from sqlalchemy import text
from sqlalchemy.orm import Session

from fraudguard.persistence.database import get_db, init_db
from fraudguard.persistence.models import MonitoringReportRecord, PredictionRecord
from fraudguard.registry.client import ModelRegistryClient
from fraudguard.serving.schemas import (
    HealthLiveResponse,
    HealthReadyResponse,
    ModelInfoResponse,
    PredictRequest,
    PredictResponse,
)

logger = logging.getLogger(__name__)

# State container for loaded model and metadata
app_state: dict[str, Any] = {
    "pipeline": None,
    "model_name": "fraudguard-classifier",
    "model_version": "v1_init",
    "decision_threshold": 0.50,
    "registry_client": None,
}


def load_model_and_config() -> None:
    """Loads model pipeline and threshold metadata on application startup."""
    registry = ModelRegistryClient()
    app_state["registry_client"] = registry

    # Load threshold from evaluation summary if available
    summary_path = Path("artifacts/phase1/evaluation_summary.json")
    if summary_path.is_file():
        try:
            with open(summary_path, "r", encoding="utf-8") as f:
                summary = json.load(f)
                app_state["decision_threshold"] = summary["xgboost"]["validation_threshold"]
                logger.info("Loaded decision threshold %.4f from summary", app_state["decision_threshold"])
        except Exception as e:
            logger.warning("Could not load threshold from summary: %s", e)

    # Load model
    try:
        pipeline, version = registry.load_pipeline_by_version_or_path(
            fallback_path="artifacts/phase1/xgb_fraud_pipeline.joblib"
        )
        app_state["pipeline"] = pipeline
        app_state["model_version"] = version
        logger.info("Successfully loaded model version %s", version)
    except Exception as e:
        logger.error("Failed to load model pipeline: %s", e)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    logger.info("Starting FraudGuard Serving Engine...")
    init_db()
    load_model_and_config()
    yield
    # Shutdown
    logger.info("Shutting down FraudGuard Serving Engine...")


app = FastAPI(
    title="FraudGuard Serving Service",
    description="Drift-monitored, versioned fraud prediction service on PaySim transactions",
    version="0.1.0",
    lifespan=lifespan,
)


@app.get("/health/live", response_model=HealthLiveResponse, tags=["Health"])
def health_live():
    """Liveness probe: verifies the process is running."""
    return HealthLiveResponse(status="ok")


@app.get("/health/ready", response_model=HealthReadyResponse, tags=["Health"])
def health_ready(db: Session = Depends(get_db)):
    """Readiness probe: verifies model is loaded and database is reachable."""
    model_loaded = app_state["pipeline"] is not None
    db_connected = False
    try:
        db.execute(text("SELECT 1"))
        db_connected = True
    except Exception as e:
        logger.error("Database readiness check failed: %s", e)

    is_ready = model_loaded and db_connected
    resp = HealthReadyResponse(
        status="ready" if is_ready else "not_ready",
        model_loaded=model_loaded,
        model_name=app_state["model_name"],
        model_version=app_state["model_version"],
        database_connected=db_connected,
    )
    if not is_ready:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=resp.model_dump(),
        )
    return resp


@app.get("/model-info", response_model=ModelInfoResponse, tags=["Metadata"])
def get_model_info():
    """Returns immutable model identity, threshold, and feature policy."""
    return ModelInfoResponse(
        model_name=app_state["model_name"],
        model_version=app_state["model_version"],
        decision_threshold=app_state["decision_threshold"],
        features_allowlist=["type", "amount", "log1p_amount", "sin_hour", "cos_hour"],
        forbidden_columns=[
            "oldbalanceOrg",
            "newbalanceOrig",
            "oldbalanceDest",
            "newbalanceDest",
            "isFlaggedFraud",
            "nameOrig",
            "nameDest",
            "isFraud",
        ],
        schema_version="v1",
    )


@app.post("/predict", response_model=PredictResponse, tags=["Inference"])
def predict(request: PredictRequest, db: Session = Depends(get_db)):
    """Performs fraud score inference with fail-closed audit logging and idempotency."""
    pipeline = app_state["pipeline"]
    if pipeline is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model is not loaded or unavailable",
        )

    # 1. Idempotency Check: (replay_run_id, transaction_id)
    existing: Optional[PredictionRecord] = (
        db.query(PredictionRecord)
        .filter(
            PredictionRecord.replay_run_id == request.replay_run_id,
            PredictionRecord.transaction_id == request.transaction_id,
        )
        .first()
    )
    if existing:
        logger.debug(
            "Idempotency hit for transaction %s in run %s",
            request.transaction_id,
            request.replay_run_id,
        )
        return PredictResponse(
            prediction_id=existing.id,
            transaction_id=existing.transaction_id,
            fraud_score=existing.fraud_score,
            decision=existing.decision,
            decision_threshold=existing.decision_threshold,
            model_version=existing.model_version,
            schema_version=existing.schema_version,
            inference_ms=existing.inference_ms,
        )

    # 2. Measure inference latency
    t0 = time.perf_counter()

    # Build single-row DataFrame for pipeline input
    input_df = pd.DataFrame(
        [
            {
                "transaction_id": request.transaction_id,
                "step": request.event_step,
                "type": request.type.value,
                "amount": request.amount,
            }
        ]
    )

    try:
        probs = pipeline.predict_proba(input_df)
        fraud_score = float(probs[0, 1])
    except Exception as e:
        logger.error("Inference prediction error: %s", e)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Inference execution failed: {str(e)}",
        )

    inference_ms = round((time.perf_counter() - t0) * 1000.0, 3)
    threshold = float(app_state["decision_threshold"])
    decision = bool(fraud_score >= threshold)
    prediction_id = str(uuid.uuid4())

    # 3. Fail-Closed Audit Logging: write to DB before responding
    features_payload = {
        "type": request.type.value,
        "amount": request.amount,
    }

    record = PredictionRecord(
        id=prediction_id,
        transaction_id=request.transaction_id,
        replay_run_id=request.replay_run_id,
        event_step=request.event_step,
        features=features_payload,
        fraud_score=fraud_score,
        decision=decision,
        decision_threshold=threshold,
        model_name=app_state["model_name"],
        model_version=app_state["model_version"],
        schema_version=request.schema_version,
        inference_ms=inference_ms,
        scenario_id=request.scenario_id,
    )

    try:
        db.add(record)
        db.commit()
    except Exception as e:
        db.rollback()
        logger.error("Failed to persist prediction audit log: %s", e)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Audit persistence failure. Request failed closed per compliance policy.",
        )

    return PredictResponse(
        prediction_id=prediction_id,
        transaction_id=request.transaction_id,
        fraud_score=fraud_score,
        decision=decision,
        decision_threshold=threshold,
        model_version=app_state["model_version"],
        schema_version=request.schema_version,
        inference_ms=inference_ms,
    )


@app.get("/drift-status", tags=["Monitoring"])
def get_drift_status(db: Session = Depends(get_db)):
    """Returns recent monitoring reports and alerts."""
    reports = (
        db.query(MonitoringReportRecord)
        .order_by(MonitoringReportRecord.created_at.desc())
        .limit(10)
        .all()
    )
    return {
        "count": len(reports),
        "recent_reports": [
            {
                "id": r.id,
                "report_kind": r.report_kind,
                "window": f"{r.window_start}..{r.window_end}",
                "status": r.status,
                "sample_count": r.sample_count,
                "created_at": r.created_at.isoformat() if r.created_at else None,
            }
            for r in reports
        ],
    }

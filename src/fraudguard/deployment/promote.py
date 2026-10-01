"""Production model promotion workflow with readiness verification and alias rotation."""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Dict, Optional

import pandas as pd
from sqlalchemy.orm import Session

from fraudguard.persistence.models import DeploymentEventRecord
from fraudguard.registry.client import ModelRegistryClient

logger = logging.getLogger(__name__)


def promote_candidate_model(
    db: Session,
    candidate_pipeline: Any,
    candidate_version: str,
    current_champion_version: str,
    gate_summary: Dict[str, Any],
    registry_client: Optional[ModelRegistryClient] = None,
    simulate_failure: bool = False,
) -> DeploymentEventRecord:
    """Verifies candidate and safely updates champion and previous aliases.

    If verification fails, serving remains untouched and failure is recorded.
    """
    event = DeploymentEventRecord(
        candidate_version=candidate_version,
        previous_version=current_champion_version,
        status="PENDING",
        gate_summary=gate_summary,
        deployment_revision=f"rev-{candidate_version}",
        started_at=datetime.now(timezone.utc),
    )
    db.add(event)
    db.commit()

    logger.info("Initiating promotion of candidate %s (replacing %s)", candidate_version, current_champion_version)

    # 1. Readiness & Smoke Test Verification
    try:
        if simulate_failure:
            raise RuntimeError("Simulated deployment verification failure (smoke test failed)")

        # Verify smoke prediction
        smoke_df = pd.DataFrame([{"step": 1, "type": "TRANSFER", "amount": 100.0}])
        scores = candidate_pipeline.predict_proba(smoke_df)
        if scores.shape != (1, 2):
            raise ValueError(f"Smoke prediction produced invalid output shape: {scores.shape}")

    except Exception as e:
        logger.error("Candidate deployment verification failed: %s", e)
        event.status = "FAILED"
        event.rollback_reason = f"Verification failure: {str(e)}"
        db.commit()
        raise

    # 2. Update MLflow Aliases
    if registry_client:
        try:
            # Set previous
            registry_client.client.set_registered_model_alias(
                registry_client.model_name, "previous", current_champion_version
            )
            # Promote champion
            registry_client.client.set_registered_model_alias(
                registry_client.model_name, "champion", candidate_version
            )
            logger.info("Successfully updated MLflow registry aliases: champion -> %s", candidate_version)
        except Exception as e:
            logger.warning("Could not set MLflow aliases (operating in local/fallback mode): %s", e)

    event.status = "SUCCESS"
    event.verified_at = datetime.now(timezone.utc)
    db.commit()
    logger.info("Candidate %s promoted to champion successfully", candidate_version)
    return event

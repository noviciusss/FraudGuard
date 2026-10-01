"""Automated and emergency rollback workflows."""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Optional

from sqlalchemy.orm import Session

from fraudguard.persistence.models import DeploymentEventRecord
from fraudguard.registry.client import ModelRegistryClient

logger = logging.getLogger(__name__)


def execute_rollback(
    db: Session,
    failed_version: str,
    target_previous_version: str,
    reason: str,
    registry_client: Optional[ModelRegistryClient] = None,
) -> DeploymentEventRecord:
    """Restores the champion alias to the previous verified model version and logs audit trail."""
    logger.warning(
        "Executing emergency rollback: reverting failed version %s back to %s. Reason: %s",
        failed_version,
        target_previous_version,
        reason,
    )

    # 1. Update MLflow Alias
    if registry_client:
        try:
            registry_client.client.set_registered_model_alias(
                registry_client.model_name, "champion", target_previous_version
            )
            logger.info("Rolled back MLflow champion alias to version %s", target_previous_version)
        except Exception as e:
            logger.warning("Could not update MLflow alias during rollback: %s", e)

    # 2. Record rollback event
    event = DeploymentEventRecord(
        candidate_version=failed_version,
        previous_version=target_previous_version,
        status="ROLLED_BACK",
        gate_summary={"action": "rollback", "reverted_to": target_previous_version},
        deployment_revision=f"rollback-to-{target_previous_version}",
        started_at=datetime.now(timezone.utc),
        verified_at=datetime.now(timezone.utc),
        rollback_reason=reason,
    )
    db.add(event)
    db.commit()

    return event

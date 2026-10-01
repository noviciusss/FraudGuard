"""Delayed label simulation worker driven by simulation clock steps."""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import List

from sqlalchemy.orm import Session

from fraudguard.persistence.models import OutcomeRecord
from fraudguard.replay.stream import ReplayedTransaction

logger = logging.getLogger(__name__)


class PendingLabel:
    def __init__(self, prediction_id: str, label: int, label_available_step: int):
        self.prediction_id = prediction_id
        self.label = label
        self.label_available_step = label_available_step


class DelayedLabelWorker:
    """Manages the delayed release of fraud labels into the persistence layer.

    Guarantees labels become available strictly based on the simulation step clock,
    never prematurely or via naive wall-clock sleeps.
    """

    def __init__(self, delay_steps: int = 24):
        self.delay_steps = delay_steps
        self.pending_queue: List[PendingLabel] = []

    def enqueue_replayed_transactions(
        self,
        replayed: List[ReplayedTransaction],
    ) -> None:
        """Enqueues ground truth outcomes with scheduled availability step."""
        for item in replayed:
            available_step = item.event_step + self.delay_steps
            self.pending_queue.append(
                PendingLabel(
                    prediction_id=item.prediction_id,
                    label=item.ground_truth_label,
                    label_available_step=available_step,
                )
            )
        logger.info(
            "Enqueued %d labels with step delay +%d",
            len(replayed),
            self.delay_steps,
        )

    def release_eligible_labels(
        self,
        current_step: int,
        db: Session,
    ) -> int:
        """Releases only those labels whose label_available_step <= current_step into DB."""
        released_count = 0
        remaining: List[PendingLabel] = []

        now_utc = datetime.now(timezone.utc)

        for pending in self.pending_queue:
            if pending.label_available_step <= current_step:
                # Check if already present to prevent duplicate errors
                exists = (
                    db.query(OutcomeRecord)
                    .filter(OutcomeRecord.prediction_id == pending.prediction_id)
                    .first()
                )
                if not exists:
                    outcome = OutcomeRecord(
                        prediction_id=pending.prediction_id,
                        label=pending.label,
                        label_available_step=pending.label_available_step,
                        released_at=now_utc,
                        label_source="synthetic_replay",
                    )
                    db.add(outcome)
                    released_count += 1
            else:
                remaining.append(pending)

        db.commit()
        self.pending_queue = remaining

        logger.info(
            "Released %d eligible labels at simulation step %d (%d still deferred)",
            released_count,
            current_step,
            len(self.pending_queue),
        )
        return released_count

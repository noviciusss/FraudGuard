"""Transaction replay stream engine.

Replays event-time transactions sequentially against the inference API.
Guarantees that ground-truth labels are NEVER transmitted in inference requests.
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Dict, List, Optional

import pandas as pd
from pydantic import BaseModel

logger = logging.getLogger(__name__)


class ReplayedTransaction(BaseModel):
    transaction_id: str
    prediction_id: str
    event_step: int
    fraud_score: float
    decision: bool
    ground_truth_label: int


class ReplayStreamer:
    """Streams transactions sequentially to the prediction API."""

    def __init__(
        self,
        predict_fn: Callable[[Dict[str, Any]], Dict[str, Any]],
        replay_run_id: str = "run_001",
        scenario_id: Optional[str] = None,
    ):
        self.predict_fn = predict_fn
        self.replay_run_id = replay_run_id
        self.scenario_id = scenario_id

    def stream_dataframe(
        self,
        df: pd.DataFrame,
        limit: Optional[int] = None,
    ) -> List[ReplayedTransaction]:
        """Iterates through rows in chronological step order and dispatches inference requests.

        Extracts ground truth to local buffer and never exposes it to predict_fn.
        """
        # Ensure chronological step order
        df_sorted = df.sort_values(by=["step"]).reset_index(drop=True)
        if limit:
            df_sorted = df_sorted.head(limit)

        results: List[ReplayedTransaction] = []

        for idx, row in df_sorted.iterrows():
            step = int(row["step"])
            tx_type = str(row["type"])
            amount = float(row["amount"])
            gt_label = int(row["isFraud"]) if "isFraud" in row else 0

            # Generate stable transaction_id
            tx_id = f"tx-{self.replay_run_id}-{step:04d}-{idx:06d}"

            # Strict inference payload: NO LABELS, NO BALANCE FIELDS
            payload = {
                "transaction_id": tx_id,
                "event_step": step,
                "type": tx_type,
                "amount": amount,
                "schema_version": "v1",
                "replay_run_id": self.replay_run_id,
                "scenario_id": self.scenario_id,
            }

            resp = self.predict_fn(payload)

            results.append(
                ReplayedTransaction(
                    transaction_id=tx_id,
                    prediction_id=resp["prediction_id"],
                    event_step=step,
                    fraud_score=resp["fraud_score"],
                    decision=resp["decision"],
                    ground_truth_label=gt_label,
                )
            )

        logger.info(
            "Completed replay stream for run '%s': %d transactions sent",
            self.replay_run_id,
            len(results),
        )
        return results

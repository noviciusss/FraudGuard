"""Strict Pydantic schemas for FraudGuard API request and response contracts."""

from __future__ import annotations

from enum import Enum
from typing import List, Optional

from pydantic import BaseModel, ConfigDict, Field


class TransactionTypeEnum(str, Enum):
    CASH_IN = "CASH_IN"
    CASH_OUT = "CASH_OUT"
    DEBIT = "DEBIT"
    PAYMENT = "PAYMENT"
    TRANSFER = "TRANSFER"


class PredictRequest(BaseModel):
    """Transaction inference request schema.

    Strictly forbids unexpected fields, labels (isFraud), and balance columns
    to prevent inference data leakage.
    """

    model_config = ConfigDict(extra="forbid")

    transaction_id: str = Field(
        ...,
        min_length=1,
        max_length=64,
        description="Unique business identifier for the transaction",
        examples=["demo-000001"],
    )
    event_step: int = Field(
        ...,
        ge=0,
        description="Simulated event time step",
        examples=[100],
    )
    type: TransactionTypeEnum = Field(
        ...,
        description="Transaction type category",
        examples=["TRANSFER"],
    )
    amount: float = Field(
        ...,
        ge=0.0,
        description="Transaction amount in currency units (finite, nonnegative)",
        examples=[181.0],
    )
    schema_version: str = Field(
        default="v1",
        description="Feature schema version",
        examples=["v1"],
    )
    replay_run_id: str = Field(
        default="default",
        description="Simulation or replay run identifier for idempotency partition",
        examples=["run-001"],
    )
    scenario_id: Optional[str] = Field(
        default=None,
        description="Optional replay scenario identifier",
        examples=["stable_replay"],
    )


class PredictResponse(BaseModel):
    """Inference response returning continuous score, binary decision, and immutable model version."""

    prediction_id: str = Field(..., description="Unique generated UUID for audit logging")
    transaction_id: str = Field(..., description="Transaction business ID")
    fraud_score: float = Field(..., description="Estimated risk score (uncalibrated probability)")
    decision: bool = Field(..., description="Binary classification decision at active threshold")
    decision_threshold: float = Field(..., description="Decision threshold applied")
    model_version: str = Field(..., description="Immutable version string of the loaded model")
    schema_version: str = Field(default="v1", description="Schema version used")
    inference_ms: float = Field(..., description="Inference latency in milliseconds")


class HealthLiveResponse(BaseModel):
    status: str = "ok"


class HealthReadyResponse(BaseModel):
    status: str = "ready"
    model_loaded: bool
    model_name: str
    model_version: str
    database_connected: bool


class ModelInfoResponse(BaseModel):
    model_name: str
    model_version: str
    decision_threshold: float
    features_allowlist: List[str]
    forbidden_columns: List[str]
    schema_version: str

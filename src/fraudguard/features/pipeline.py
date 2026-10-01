"""Feature engineering and preprocessing pipeline for FraudGuard.

Implements strict feature allowlist policies, leakage-safe exclusions,
and cyclic temporal feature extraction.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Union

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

logger = logging.getLogger(__name__)

# Canonical categories in PaySim
CANONICAL_TYPES = ["CASH_IN", "CASH_OUT", "DEBIT", "PAYMENT", "TRANSFER"]

# Explicit allowlist of raw model inputs
ALLOWLIST_INPUTS = ["type", "amount"]

# Explicitly forbidden columns
FORBIDDEN_COLUMNS = {
    "oldbalanceOrg",
    "newbalanceOrig",
    "oldbalanceDest",
    "newbalanceDest",
    "isFlaggedFraud",
    "isFraud",
    "nameOrig",
    "nameDest",
}


class FeatureExtractor(BaseEstimator, TransformerMixin):
    """Extracts allowlisted and derived features while guarding against leakage.

    Accepts pandas DataFrame or dict input. Computes:
    - log1p_amount: log(1 + amount)
    - sin_hour, cos_hour: cyclic 24-hour encoding from step/event_step
    """

    def __init__(self, step_col: str = "step"):
        self.step_col = step_col
        self.feature_names_: List[str] = [
            "type",
            "amount",
            "log1p_amount",
            "sin_hour",
            "cos_hour",
        ]

    def fit(self, X: Any, y: Any = None) -> FeatureExtractor:
        return self

    def transform(self, X: Union[pd.DataFrame, Dict[str, Any], List[Dict[str, Any]]]) -> pd.DataFrame:
        if isinstance(X, dict):
            df = pd.DataFrame([X])
        elif isinstance(X, list):
            df = pd.DataFrame(X)
        elif isinstance(X, pd.DataFrame):
            df = X.copy()
        else:
            raise TypeError(f"Unsupported input type for FeatureExtractor: {type(X)}")

        # Enforce no forbidden columns in output
        found_forbidden = FORBIDDEN_COLUMNS.intersection(set(df.columns))
        if found_forbidden:
            logger.debug("Stripping forbidden columns from feature inputs: %s", found_forbidden)
            df = df.drop(columns=list(found_forbidden))

        # Check required columns
        if "amount" not in df.columns:
            raise ValueError("Input missing required feature 'amount'")
        if "type" not in df.columns:
            raise ValueError("Input missing required feature 'type'")

        # Ensure amount is numeric, non-negative, finite
        amount = pd.to_numeric(df["amount"], errors="coerce").fillna(0.0)
        amount = np.maximum(amount.to_numpy(), 0.0)
        log1p_amount = np.log1p(amount)

        # Extract step / event_step for cyclic hour
        step_val = 0
        if self.step_col in df.columns:
            step_val = pd.to_numeric(df[self.step_col], errors="coerce").fillna(0).to_numpy()
        elif "event_step" in df.columns:
            step_val = pd.to_numeric(df["event_step"], errors="coerce").fillna(0).to_numpy()
        else:
            step_val = np.zeros(len(df))

        # 1 step = 1 hour in PaySim simulator
        hour = (step_val % 24).astype(float)
        sin_hour = np.sin(2.0 * np.pi * hour / 24.0)
        cos_hour = np.cos(2.0 * np.pi * hour / 24.0)

        # Construct clean features dataframe
        features_df = pd.DataFrame(
            {
                "type": df["type"].astype(str),
                "amount": amount,
                "log1p_amount": log1p_amount,
                "sin_hour": sin_hour,
                "cos_hour": cos_hour,
            },
            index=df.index,
        )

        return features_df


def build_preprocessor() -> ColumnTransformer:
    """Builds the scikit-learn ColumnTransformer for categorical and numerical features."""
    categorical_features = ["type"]
    numerical_features = ["amount", "log1p_amount", "sin_hour", "cos_hour"]

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "cat",
                OneHotEncoder(
                    categories=[CANONICAL_TYPES],
                    handle_unknown="ignore",
                    sparse_output=False,
                ),
                categorical_features,
            ),
            (
                "num",
                StandardScaler(),
                numerical_features,
            ),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )
    return preprocessor


def build_fraud_pipeline(model: Any, step_col: str = "step") -> Pipeline:
    """Constructs the unified end-to-end inference pipeline:

    FeatureExtractor -> ColumnTransformer (OneHotEncoder + Scaler) -> Estimator.
    This entire object is serialized and deployed, preventing skew between training and serving.
    """
    pipeline = Pipeline(
        steps=[
            ("extractor", FeatureExtractor(step_col=step_col)),
            ("preprocessor", build_preprocessor()),
            ("model", model),
        ]
    )
    return pipeline

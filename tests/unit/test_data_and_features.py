"""Unit tests for Phase 1: Data preparation, leakage guards, splits, and feature pipeline."""

import numpy as np
import pandas as pd

from fraudguard.data.prepare import (
    FORBIDDEN_MODEL_COLUMNS,
    generate_synthetic_paysim,
    validate_raw_schema,
)
from fraudguard.data.splits import split_by_complete_steps
from fraudguard.features.pipeline import (
    FeatureExtractor,
    build_fraud_pipeline,
)


def test_synthetic_paysim_generation():
    """Verify synthetic PaySim matches schema, steps, and fraud rate."""
    df = generate_synthetic_paysim(num_rows=2000, num_steps=20, fraud_rate=0.01, random_seed=42)
    assert len(df) == 2000
    assert "step" in df.columns
    assert "isFraud" in df.columns
    assert df["isFraud"].sum() > 0

    valid, errors = validate_raw_schema(df)
    assert valid, f"Schema validation failed: {errors}"


def test_leakage_safe_step_splits():
    """Verify complete step disjunction: no intra-hour/step leakage across partitions."""
    df = generate_synthetic_paysim(num_rows=3000, num_steps=30, random_seed=42)
    partitions = split_by_complete_steps(
        df=df,
        step_col="step",
        train_ratio=0.60,
        tune_ratio=0.15,
        replay_ratio=0.15,
        audit_ratio=0.10,
    )

    train_steps = set(partitions.train["step"].unique())
    tune_steps = set(partitions.tune["step"].unique())
    replay_steps = set(partitions.replay["step"].unique())
    audit_steps = set(partitions.audit["step"].unique())

    # Ensure mutual exclusivity
    assert len(train_steps & tune_steps) == 0, "Train and Tune share simulation steps!"
    assert len(tune_steps & replay_steps) == 0, "Tune and Replay share simulation steps!"
    assert len(replay_steps & audit_steps) == 0, "Replay and Audit share simulation steps!"

    # Verify temporal ordering
    assert max(train_steps) < min(tune_steps)
    assert max(tune_steps) < min(replay_steps)
    assert max(replay_steps) < min(audit_steps)

    # Manifest checks
    assert partitions.manifest.splits["train"].fraud_count == partitions.train["isFraud"].sum()


def test_forbidden_columns_stripped():
    """Verify that forbidden balance columns, IDs, and labels are stripped by the feature pipeline."""
    extractor = FeatureExtractor(step_col="step")

    # Construct input with forbidden columns and leakage candidates
    input_data = pd.DataFrame(
        {
            "step": [1, 2],
            "type": ["TRANSFER", "CASH_OUT"],
            "amount": [5000.0, 100.0],
            "oldbalanceOrg": [10000.0, 200.0],
            "newbalanceOrig": [5000.0, 100.0],
            "oldbalanceDest": [0.0, 50.0],
            "newbalanceDest": [5000.0, 150.0],
            "isFlaggedFraud": [0, 0],
            "isFraud": [1, 0],
            "nameOrig": ["C12345", "C67890"],
            "nameDest": ["C99999", "M11111"],
        }
    )

    extracted = extractor.transform(input_data)

    for forbidden in FORBIDDEN_MODEL_COLUMNS:
        assert forbidden not in extracted.columns, f"Forbidden column {forbidden} leaked into features!"
    assert "isFraud" not in extracted.columns

    # Derived features present
    assert "log1p_amount" in extracted.columns
    assert "sin_hour" in extracted.columns
    assert "cos_hour" in extracted.columns
    assert np.isclose(extracted.loc[0, "log1p_amount"], np.log1p(5000.0))


def test_unknown_categorical_handling():
    """Verify that unseen transaction categories are safely ignored without raising exceptions."""
    from sklearn.dummy import DummyClassifier

    pipeline = build_fraud_pipeline(model=DummyClassifier(strategy="constant", constant=0))

    # Fit on canonical categories with binary classes [0, 1]
    fit_df = pd.DataFrame({"step": [1, 2], "type": ["TRANSFER", "PAYMENT"], "amount": [100.0, 50.0]})
    pipeline.fit(fit_df, [0, 1])

    # Transform/predict with an unknown category (e.g., 'CRYPTO_TRANSFER')
    test_df = pd.DataFrame({"step": [3], "type": ["UNKNOWN_METHOD"], "amount": [250.0]})
    # Should not raise exception
    scores = pipeline.predict_proba(test_df)
    assert scores.shape == (1, 2)

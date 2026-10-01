"""Data preparation, validation, and deterministic PaySim simulation generator."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Canonical PaySim columns
PAYSIM_RAW_COLUMNS = [
    "step",
    "type",
    "amount",
    "nameOrig",
    "oldbalanceOrg",
    "newbalanceOrig",
    "nameDest",
    "oldbalanceDest",
    "newbalanceDest",
    "isFraud",
    "isFlaggedFraud",
]

VALID_TRANSACTION_TYPES = ["CASH_IN", "CASH_OUT", "DEBIT", "PAYMENT", "TRANSFER"]

# Explicitly excluded columns from modeling per project design & PaySim warnings
FORBIDDEN_MODEL_COLUMNS = [
    "oldbalanceOrg",
    "newbalanceOrig",
    "oldbalanceDest",
    "newbalanceDest",
    "isFlaggedFraud",
    "nameOrig",
    "nameDest",
]


def generate_synthetic_paysim(
    num_rows: int = 50000,
    num_steps: int = 100,
    fraud_rate: float = 0.005,
    random_seed: int = 42,
) -> pd.DataFrame:
    """Generates a realistic synthetic PaySim-style dataset for development and testing.

    Models the step distribution, transaction types, fraud clustering on TRANSFER
    and CASH_OUT, and balance fields (which must later be discarded by feature policy).
    """
    rng = np.random.default_rng(random_seed)

    # 1. Simulate steps (chronological event time)
    steps = np.sort(rng.integers(1, num_steps + 1, size=num_rows))

    # 2. Simulate transaction types
    # Real PaySim distribution: PAYMENT ~34%, CASH_OUT ~35%, CASH_IN ~22%, TRANSFER ~8%, DEBIT ~1%
    type_probs = [0.22, 0.35, 0.01, 0.34, 0.08]
    types = rng.choice(VALID_TRANSACTION_TYPES, size=num_rows, p=type_probs)

    # 3. Simulate amounts (log-normal distribution)
    # Higher amounts on TRANSFER, smaller on PAYMENT
    amounts = np.zeros(num_rows, dtype=np.float64)
    for i, t in enumerate(types):
        if t == "PAYMENT":
            amounts[i] = float(np.round(rng.lognormal(mean=7.5, sigma=1.2), 2))
        elif t == "TRANSFER":
            amounts[i] = float(np.round(rng.lognormal(mean=11.5, sigma=1.5), 2))
        elif t == "CASH_OUT":
            amounts[i] = float(np.round(rng.lognormal(mean=10.5, sigma=1.4), 2))
        else:
            amounts[i] = float(np.round(rng.lognormal(mean=8.5, sigma=1.2), 2))

    # Ensure finite, nonnegative
    amounts = np.clip(amounts, 1.0, 10_000_000.0)

    # 4. Generate Account IDs
    name_orig = [f"C{rng.integers(10000000, 99999999)}" for _ in range(num_rows)]
    name_dest = [
        f"M{rng.integers(10000000, 99999999)}"
        if t == "PAYMENT"
        else f"C{rng.integers(10000000, 99999999)}"
        for t in types
    ]

    # 5. Balances (simulated for realism; PaySim warns not to use them)
    old_balance_orig = rng.lognormal(mean=9.0, sigma=2.0, size=num_rows).round(2)
    new_balance_orig = np.maximum(0.0, old_balance_orig - amounts).round(2)
    old_balance_dest = rng.lognormal(mean=9.5, sigma=2.0, size=num_rows).round(2)
    new_balance_dest = (old_balance_dest + amounts).round(2)

    # 6. Fraud labeling
    # Fraud only happens on TRANSFER or CASH_OUT in PaySim
    is_fraud = np.zeros(num_rows, dtype=np.int32)
    fraud_candidates = np.where(np.isin(types, ["TRANSFER", "CASH_OUT"]))[0]

    # Target number of frauds
    target_frauds = int(num_rows * fraud_rate)
    target_frauds = max(10, min(target_frauds, len(fraud_candidates)))

    # Select candidates with high amounts with higher fraud probability
    candidate_amounts = amounts[fraud_candidates]
    sampling_weights = candidate_amounts / candidate_amounts.sum()
    chosen_frauds = rng.choice(
        fraud_candidates, size=target_frauds, replace=False, p=sampling_weights
    )
    is_fraud[chosen_frauds] = 1

    # Simulated isFlaggedFraud rule (transfers > 200,000)
    is_flagged_fraud = np.where(
        (types == "TRANSFER") & (amounts > 200000.0) & (rng.random(num_rows) < 0.2),
        1,
        0,
    ).astype(np.int32)

    df = pd.DataFrame(
        {
            "step": steps,
            "type": types,
            "amount": amounts,
            "nameOrig": name_orig,
            "oldbalanceOrg": old_balance_orig,
            "newbalanceOrig": new_balance_orig,
            "nameDest": name_dest,
            "oldbalanceDest": old_balance_dest,
            "newbalanceDest": new_balance_dest,
            "isFraud": is_fraud,
            "isFlaggedFraud": is_flagged_fraud,
        }
    )

    logger.info(
        "Generated synthetic PaySim dataset: %d rows, %d frauds (%.3f%%), steps %d..%d",
        num_rows,
        df["isFraud"].sum(),
        100.0 * df["isFraud"].mean(),
        df["step"].min(),
        df["step"].max(),
    )
    return df


def validate_raw_schema(df: pd.DataFrame) -> Tuple[bool, List[str]]:
    """Checks whether the input DataFrame satisfies the expected raw PaySim schema."""
    errors = []
    missing_cols = [c for c in ["step", "type", "amount", "isFraud"] if c not in df.columns]
    if missing_cols:
        errors.append(f"Missing required columns: {missing_cols}")

    if "type" in df.columns:
        invalid_types = set(df["type"].dropna().unique()) - set(VALID_TRANSACTION_TYPES)
        if invalid_types:
            errors.append(f"Invalid transaction types detected: {invalid_types}")

    if "amount" in df.columns:
        if (df["amount"] < 0).any():
            errors.append("Negative transaction amounts found.")
        if df["amount"].isna().any():
            errors.append("NaN values found in amount column.")

    if "step" in df.columns:
        if (df["step"] < 0).any():
            errors.append("Negative step values found.")

    return len(errors) == 0, errors


def load_dataset(
    path: Optional[Path | str] = None,
    fallback_generate: bool = True,
    generate_rows: int = 50000,
    generate_steps: int = 120,
    random_seed: int = 42,
) -> pd.DataFrame:
    """Loads dataset from file if exists, or generates synthetic PaySim if allowed."""
    if path and Path(path).is_file():
        file_path = Path(path)
        logger.info("Loading dataset from %s", file_path)
        if file_path.suffix.lower() == ".parquet":
            df = pd.read_parquet(file_path)
        else:
            df = pd.read_csv(file_path)

        valid, errors = validate_raw_schema(df)
        if not valid:
            raise ValueError(f"Dataset at {path} failed schema validation: {errors}")
        return df

    if fallback_generate:
        logger.info("File not found or not specified. Generating synthetic PaySim data.")
        return generate_synthetic_paysim(
            num_rows=generate_rows,
            num_steps=generate_steps,
            random_seed=random_seed,
        )

    raise FileNotFoundError(f"Dataset path {path} not found and fallback_generate is False.")

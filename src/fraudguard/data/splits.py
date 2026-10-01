"""Strict temporal splitting by complete simulation steps."""

from __future__ import annotations

import logging
from typing import NamedTuple

import numpy as np
import pandas as pd

from fraudguard.data.manifests import DataManifest, SplitManifest, compute_dataframe_checksum

logger = logging.getLogger(__name__)


class DatasetPartitions(NamedTuple):
    train: pd.DataFrame
    tune: pd.DataFrame
    replay: pd.DataFrame
    audit: pd.DataFrame
    manifest: DataManifest


def split_by_complete_steps(
    df: pd.DataFrame,
    step_col: str = "step",
    train_ratio: float = 0.60,
    tune_ratio: float = 0.15,
    replay_ratio: float = 0.15,
    audit_ratio: float = 0.10,
    dataset_name: str = "PaySim_temporal",
) -> DatasetPartitions:
    """Splits a dataset chronologically by complete step units.

    Guarantees that no transactions from the same simulation step appear
    across multiple split partitions, preventing intra-hour data leakage.
    """
    if not np.isclose(train_ratio + tune_ratio + replay_ratio + audit_ratio, 1.0):
        raise ValueError("Split ratios must sum to 1.0")

    # Deterministic ordering: step ascending, preserving stable tie-breakers
    df_sorted = df.sort_values(by=[step_col], kind="mergesort").reset_index(drop=True)

    unique_steps = np.sort(df_sorted[step_col].unique())
    num_steps = len(unique_steps)
    if num_steps < 4:
        raise ValueError(f"Need at least 4 unique steps to create 4 disjoint splits, got {num_steps}")

    # Compute step cutoff indices
    n_train = max(1, int(round(num_steps * train_ratio)))
    n_tune = max(1, int(round(num_steps * tune_ratio)))
    n_replay = max(1, int(round(num_steps * replay_ratio)))

    # Ensure all step slices are valid and non-overlapping
    train_steps = unique_steps[:n_train]
    tune_steps = unique_steps[n_train : n_train + n_tune]
    replay_steps = unique_steps[n_train + n_tune : n_train + n_tune + n_replay]
    audit_steps = unique_steps[n_train + n_tune + n_replay :]

    if len(audit_steps) == 0:
        # Rebalance last step
        audit_steps = replay_steps[-1:]
        replay_steps = replay_steps[:-1]

    # Verify complete step disjunction
    assert len(set(train_steps) & set(tune_steps)) == 0, "Train and Tune steps overlap!"
    assert len(set(tune_steps) & set(replay_steps)) == 0, "Tune and Replay steps overlap!"
    assert len(set(replay_steps) & set(audit_steps)) == 0, "Replay and Audit steps overlap!"

    # Partition data
    train_df = df_sorted[df_sorted[step_col].isin(train_steps)].copy().reset_index(drop=True)
    tune_df = df_sorted[df_sorted[step_col].isin(tune_steps)].copy().reset_index(drop=True)
    replay_df = df_sorted[df_sorted[step_col].isin(replay_steps)].copy().reset_index(drop=True)
    audit_df = df_sorted[df_sorted[step_col].isin(audit_steps)].copy().reset_index(drop=True)

    # Build manifests
    manifest = DataManifest(dataset_name=dataset_name, total_rows=len(df_sorted))

    splits_map = {
        "train": (train_df, train_steps),
        "tune": (tune_df, tune_steps),
        "replay": (replay_df, replay_steps),
        "audit": (audit_df, audit_steps),
    }

    for name, (part_df, steps) in splits_map.items():
        fraud_count = int(part_df["isFraud"].sum()) if "isFraud" in part_df else 0
        total_rows = len(part_df)
        prevalence = fraud_count / total_rows if total_rows > 0 else 0.0
        checksum = compute_dataframe_checksum(part_df)

        split_meta = SplitManifest(
            name=name,
            start_step=int(steps[0]),
            end_step=int(steps[-1]),
            total_rows=total_rows,
            fraud_count=fraud_count,
            fraud_prevalence=prevalence,
            checksum=checksum,
        )
        manifest.add_split(split_meta)

        logger.info(
            "Split '%s': steps %d..%d | rows: %d | frauds: %d (%.4f%%)",
            name,
            steps[0],
            steps[-1],
            total_rows,
            fraud_count,
            100.0 * prevalence,
        )

    return DatasetPartitions(
        train=train_df,
        tune=tune_df,
        replay=replay_df,
        audit=audit_df,
        manifest=manifest,
    )

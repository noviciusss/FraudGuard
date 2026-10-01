"""Simulation scenario generators for stable and mix_shift evaluation windows."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np
import pandas as pd


@dataclass
class ScenarioManifest:
    scenario_id: str
    scenario_type: str  # "stable" or "mix_shift"
    random_seed: int
    row_count: int
    fraud_count: int
    description: str


class ScenarioGenerator:
    """Produces scenario streams with audit manifests."""

    def __init__(self, replay_df: pd.DataFrame, random_seed: int = 42):
        self.replay_df = replay_df.copy().sort_values(by=["step"]).reset_index(drop=True)
        self.random_seed = random_seed
        self.rng = np.random.default_rng(random_seed)

    def generate_stable_scenario(self, scenario_id: str = "stable_baseline") -> Tuple[pd.DataFrame, ScenarioManifest]:
        """Stable held-out replay: unaltered distribution of transactions and labels."""
        df = self.replay_df.copy()
        fraud_count = int(df["isFraud"].sum()) if "isFraud" in df.columns else 0

        manifest = ScenarioManifest(
            scenario_id=scenario_id,
            scenario_type="stable",
            random_seed=self.random_seed,
            row_count=len(df),
            fraud_count=fraud_count,
            description="Unmodified chronological replay partition representing standard traffic",
        )
        return df, manifest

    def generate_mix_shift_scenario(
        self,
        scenario_id: str = "mix_shift_transfer_heavy",
        transfer_weight_multiplier: float = 3.0,
    ) -> Tuple[pd.DataFrame, ScenarioManifest]:
        """Mix shift scenario: resamples complete rows to shift transaction-type mix

        toward high-value TRANSFER and CASH_OUT transactions, preserving original labels.
        """
        df = self.replay_df.copy()
        weights = np.ones(len(df), dtype=float)

        # Boost sampling weights for TRANSFER and CASH_OUT
        is_transfer = df["type"] == "TRANSFER"
        is_cash_out = df["type"] == "CASH_OUT"
        weights[is_transfer] *= transfer_weight_multiplier
        weights[is_cash_out] *= (transfer_weight_multiplier * 0.7)
        weights /= weights.sum()

        # Resample maintaining same overall size
        sampled_indices = self.rng.choice(len(df), size=len(df), replace=True, p=weights)
        shifted_df = df.iloc[sampled_indices].copy()
        # Sort by step to preserve event time ordering
        shifted_df = shifted_df.sort_values(by=["step"]).reset_index(drop=True)

        fraud_count = int(shifted_df["isFraud"].sum()) if "isFraud" in shifted_df.columns else 0

        manifest = ScenarioManifest(
            scenario_id=scenario_id,
            scenario_type="mix_shift",
            random_seed=self.random_seed,
            row_count=len(shifted_df),
            fraud_count=fraud_count,
            description=f"Resampled traffic with {transfer_weight_multiplier}x weighting on TRANSFER/CASH_OUT",
        )
        return shifted_df, manifest

"""Dataset and split manifest generation for reproducible auditing."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd


def compute_dataframe_checksum(df: pd.DataFrame) -> str:
    """Computes a deterministic SHA-256 hash of a DataFrame content."""
    # Convert dataframe to deterministic csv bytes
    csv_bytes = df.to_csv(index=False).encode("utf-8")
    return hashlib.sha256(csv_bytes).hexdigest()


def compute_file_checksum(file_path: Path | str) -> str:
    """Computes SHA-256 hash of a file."""
    path = Path(file_path)
    hasher = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()


class SplitManifest:
    """Records metadata for a single temporal partition."""

    def __init__(
        self,
        name: str,
        start_step: int,
        end_step: int,
        total_rows: int,
        fraud_count: int,
        fraud_prevalence: float,
        checksum: str,
    ):
        self.name = name
        self.start_step = start_step
        self.end_step = end_step
        self.total_rows = total_rows
        self.fraud_count = fraud_count
        self.fraud_prevalence = fraud_prevalence
        self.checksum = checksum

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "start_step": self.start_step,
            "end_step": self.end_step,
            "total_rows": self.total_rows,
            "fraud_count": self.fraud_count,
            "fraud_prevalence": round(self.fraud_prevalence, 6),
            "checksum": self.checksum,
        }


class DataManifest:
    """Comprehensive manifest recording data source and split governance."""

    def __init__(
        self,
        dataset_name: str,
        source_path: Optional[str] = None,
        source_checksum: Optional[str] = None,
        total_rows: int = 0,
        splits: Optional[Dict[str, SplitManifest]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        self.dataset_name = dataset_name
        self.source_path = source_path
        self.source_checksum = source_checksum
        self.total_rows = total_rows
        self.splits = splits or {}
        self.metadata = metadata or {}
        self.created_at = datetime.now(timezone.utc).isoformat()

    def add_split(self, split: SplitManifest) -> None:
        self.splits[split.name] = split

    def to_dict(self) -> Dict[str, Any]:
        return {
            "dataset_name": self.dataset_name,
            "source_path": self.source_path,
            "source_checksum": self.source_checksum,
            "total_rows": self.total_rows,
            "created_at": self.created_at,
            "metadata": self.metadata,
            "splits": {name: s.to_dict() for name, s in self.splits.items()},
        }

    def save(self, output_path: Path | str) -> None:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)

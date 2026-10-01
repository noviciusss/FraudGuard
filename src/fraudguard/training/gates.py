"""Validation and quality gates for candidate model promotion."""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass
from typing import Any, Dict

logger = logging.getLogger(__name__)


@dataclass
class GateResult:
    passed: bool
    status: str  # APPROVED, REJECTED, INSUFFICIENT_DATA
    reason: str
    candidate_metrics: Dict[str, Any]
    champion_metrics: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class ModelPromotionGate:
    """Enforces multi-metric quality criteria before authorizing model promotion.

    Guarantees:
    - Candidate recall must not degrade beyond non-inferiority bound.
    - False positive rate (FPR) must remain within the strict budget.
    - Precision must exceed a safety floor.
    - Comparison window must have sufficient sample and fraud volume.
    """

    def __init__(
        self,
        max_fpr: float = 0.01,
        min_precision: float = 0.15,
        min_recall_improvement: float = 0.0,
        max_ap_drop: float = 0.02,
        min_comparison_samples: int = 100,
        min_comparison_positives: int = 5,
    ):
        self.max_fpr = max_fpr
        self.min_precision = min_precision
        self.min_recall_improvement = min_recall_improvement
        self.max_ap_drop = max_ap_drop
        self.min_comparison_samples = min_comparison_samples
        self.min_comparison_positives = min_comparison_positives

    def evaluate_candidate(
        self,
        candidate_metrics: Dict[str, Any],
        champion_metrics: Dict[str, Any],
    ) -> GateResult:
        """Evaluates frozen candidate and champion on the identical comparison window."""
        cand_n = candidate_metrics.get("sample_count", 0)
        cand_pos = candidate_metrics.get("positive_count", 0)

        # 1. Sample volume check
        if cand_n < self.min_comparison_samples or cand_pos < self.min_comparison_positives:
            return GateResult(
                passed=False,
                status="INSUFFICIENT_DATA",
                reason=(
                    f"Comparison set insufficient: {cand_n} samples (min {self.min_comparison_samples}) "
                    f"or {cand_pos} positives (min {self.min_comparison_positives})"
                ),
                candidate_metrics=candidate_metrics,
                champion_metrics=champion_metrics,
            )

        # 2. FPR Budget check
        cand_fpr = candidate_metrics.get("fpr", 1.0)
        if cand_fpr > self.max_fpr:
            return GateResult(
                passed=False,
                status="REJECTED",
                reason=f"Candidate FPR ({cand_fpr:.4f}) breaches maximum budget ({self.max_fpr:.4f})",
                candidate_metrics=candidate_metrics,
                champion_metrics=champion_metrics,
            )

        # 3. Precision floor check
        cand_prec = candidate_metrics.get("precision", 0.0)
        if cand_prec < self.min_precision:
            return GateResult(
                passed=False,
                status="REJECTED",
                reason=f"Candidate precision ({cand_prec:.4f}) below safety floor ({self.min_precision:.4f})",
                candidate_metrics=candidate_metrics,
                champion_metrics=champion_metrics,
            )

        # 4. Recall comparison against champion
        champ_recall = champion_metrics.get("recall", 0.0)
        cand_recall = candidate_metrics.get("recall", 0.0)
        recall_diff = cand_recall - champ_recall

        if recall_diff < self.min_recall_improvement:
            return GateResult(
                passed=False,
                status="REJECTED",
                reason=(
                    f"Candidate recall ({cand_recall:.4f}) does not meet improvement requirement "
                    f"over champion ({champ_recall:.4f}, diff {recall_diff:+.4f} < {self.min_recall_improvement})"
                ),
                candidate_metrics=candidate_metrics,
                champion_metrics=champion_metrics,
            )

        # 5. Average Precision drop check
        champ_ap = champion_metrics.get("average_precision", 0.0)
        cand_ap = candidate_metrics.get("average_precision", 0.0)
        ap_drop = champ_ap - cand_ap

        if ap_drop > self.max_ap_drop:
            return GateResult(
                passed=False,
                status="REJECTED",
                reason=f"Candidate AP ({cand_ap:.4f}) degraded more than tolerance ({self.max_ap_drop}) vs champion ({champ_ap:.4f})",
                candidate_metrics=candidate_metrics,
                champion_metrics=champion_metrics,
            )

        # All criteria satisfied
        return GateResult(
            passed=True,
            status="APPROVED",
            reason=(
                f"Candidate passed all gates: recall {cand_recall:.4f} (champ {champ_recall:.4f}), "
                f"FPR {cand_fpr:.4f}, precision {cand_prec:.4f}, AP {cand_ap:.4f}"
            ),
            candidate_metrics=candidate_metrics,
            champion_metrics=champion_metrics,
        )

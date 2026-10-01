"""Lifecycle safety tests for Phase 5: Gating, rejection, deployment failure, and rollback."""

import pytest

from fraudguard.data.prepare import generate_synthetic_paysim
from fraudguard.deployment.promote import promote_candidate_model
from fraudguard.deployment.rollback import execute_rollback
from fraudguard.persistence.database import get_session, init_db
from fraudguard.persistence.models import DeploymentEventRecord
from fraudguard.training.gates import ModelPromotionGate
from fraudguard.training.train import train_xgboost_model


def test_promotion_gate_decisions():
    """Verify promotion gate strictly rejects bad candidates and approves qualifying ones."""
    gate = ModelPromotionGate(
        max_fpr=0.01,
        min_precision=0.15,
        min_recall_improvement=0.0,
        max_ap_drop=0.02,
        min_comparison_samples=100,
        min_comparison_positives=5,
    )

    champ_metrics = {
        "sample_count": 500,
        "positive_count": 20,
        "recall": 0.40,
        "fpr": 0.008,
        "precision": 0.30,
        "average_precision": 0.35,
    }

    # 1. Candidate with worse recall -> REJECTED
    bad_cand_metrics = {
        "sample_count": 500,
        "positive_count": 20,
        "recall": 0.30,  # Worse than 0.40
        "fpr": 0.008,
        "precision": 0.30,
        "average_precision": 0.35,
    }
    res_bad_recall = gate.evaluate_candidate(bad_cand_metrics, champ_metrics)
    assert not res_bad_recall.passed
    assert res_bad_recall.status == "REJECTED"
    assert "recall" in res_bad_recall.reason

    # 2. Candidate breaching FPR budget -> REJECTED
    bad_fpr_metrics = {
        "sample_count": 500,
        "positive_count": 20,
        "recall": 0.50,
        "fpr": 0.025,  # Exceeds 0.01
        "precision": 0.20,
        "average_precision": 0.38,
    }
    res_bad_fpr = gate.evaluate_candidate(bad_fpr_metrics, champ_metrics)
    assert not res_bad_fpr.passed
    assert res_bad_fpr.status == "REJECTED"
    assert "FPR" in res_bad_fpr.reason

    # 3. Candidate with insufficient samples -> INSUFFICIENT_DATA
    small_metrics = {
        "sample_count": 50,
        "positive_count": 2,
        "recall": 0.50,
        "fpr": 0.005,
        "precision": 0.40,
        "average_precision": 0.45,
    }
    res_small = gate.evaluate_candidate(small_metrics, champ_metrics)
    assert not res_small.passed
    assert res_small.status == "INSUFFICIENT_DATA"

    # 4. Qualifying candidate -> APPROVED
    good_cand_metrics = {
        "sample_count": 500,
        "positive_count": 20,
        "recall": 0.45,  # Better than 0.40
        "fpr": 0.007,  # Within 0.01
        "precision": 0.32,  # > 0.15
        "average_precision": 0.37,  # > 0.35
    }
    res_good = gate.evaluate_candidate(good_cand_metrics, champ_metrics)
    assert res_good.passed
    assert res_good.status == "APPROVED"


def test_deployment_failure_and_rollback_safety(tmp_path):
    """Verify deployment failure leaves champion intact and rollback restores previous state."""
    db_file = tmp_path / "test_lifecycle.db"
    db_url = f"sqlite:///{db_file}"
    init_db(db_url)
    session = get_session(db_url)

    # Train a dummy pipeline
    train_df = generate_synthetic_paysim(num_rows=200, num_steps=5, random_seed=42)
    pipeline = train_xgboost_model(train_df, n_estimators=5, max_depth=2)

    # 1. Test simulated deployment failure
    with pytest.raises(RuntimeError):
        promote_candidate_model(
            db=session,
            candidate_pipeline=pipeline,
            candidate_version="v_faulty",
            current_champion_version="v1_initial",
            gate_summary={"status": "APPROVED"},
            simulate_failure=True,
        )

    failed_event = (
        session.query(DeploymentEventRecord)
        .filter(DeploymentEventRecord.candidate_version == "v_faulty")
        .first()
    )
    assert failed_event is not None
    assert failed_event.status == "FAILED"
    assert "Verification failure" in failed_event.rollback_reason

    # 2. Test successful promotion
    success_event = promote_candidate_model(
        db=session,
        candidate_pipeline=pipeline,
        candidate_version="v2_champion",
        current_champion_version="v1_initial",
        gate_summary={"status": "APPROVED"},
        simulate_failure=False,
    )
    assert success_event.status == "SUCCESS"

    # 3. Test rollback
    rollback_event = execute_rollback(
        db=session,
        failed_version="v2_champion",
        target_previous_version="v1_initial",
        reason="Post-deployment anomaly detected",
    )
    assert rollback_event.status == "ROLLED_BACK"
    assert rollback_event.previous_version == "v1_initial"

    session.close()

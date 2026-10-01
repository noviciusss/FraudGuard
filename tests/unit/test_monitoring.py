"""Unit tests for Phase 4: Drift detection, performance evaluation, and deduplicated retrain triggers."""

import numpy as np

from fraudguard.data.prepare import generate_synthetic_paysim
from fraudguard.monitoring.drift import DriftMonitor, calculate_psi
from fraudguard.monitoring.performance import PerformanceMonitor
from fraudguard.persistence.database import get_session, init_db
from fraudguard.persistence.models import (
    OutcomeRecord,
    PredictionRecord,
    RetrainRequestRecord,
)


def test_psi_and_drift_calculation():
    """Verify PSI behaves correctly under stable vs shifted conditions."""
    rng = np.random.default_rng(42)
    reference = rng.normal(loc=100.0, scale=20.0, size=2000)

    # 1. Stable sample: same distribution
    stable_sample = rng.normal(loc=100.0, scale=20.0, size=1000)
    psi_stable = calculate_psi(reference, stable_sample)
    assert psi_stable < 0.10, f"Expected low PSI for stable sample, got {psi_stable}"

    # 2. Shifted sample: altered mean
    shifted_sample = rng.normal(loc=150.0, scale=40.0, size=1000)
    psi_shifted = calculate_psi(reference, shifted_sample)
    assert psi_shifted > 0.20, f"Expected high PSI for shifted sample, got {psi_shifted}"


def test_drift_monitor_window_evaluation():
    """Verify DriftMonitor classifies windows into NORMAL, ALERT, and INSUFFICIENT_DATA."""
    df_ref = generate_synthetic_paysim(num_rows=1500, num_steps=20, random_seed=42)
    monitor = DriftMonitor(reference_df=df_ref, min_window_samples=50)

    # Insufficient samples window
    small_df = df_ref.head(20)
    report_small = monitor.evaluate_window_drift(small_df)
    assert report_small["status"] == "INSUFFICIENT_DATA"

    # Stable window
    stable_window = generate_synthetic_paysim(num_rows=200, num_steps=5, random_seed=43)
    report_stable = monitor.evaluate_window_drift(stable_window)
    assert report_stable["status"] in ["NORMAL", "WARNING"]

    # Shifted window (amounts multiplied)
    shifted_window = stable_window.copy()
    shifted_window["amount"] *= 10.0
    report_alert = monitor.evaluate_window_drift(shifted_window)
    assert report_alert["status"] == "ALERT"


def test_performance_monitor_delayed_labels_and_deduplication(tmp_path):
    """Verify PerformanceMonitor returns PERFORMANCE_UNAVAILABLE when labels are missing

    and creates deduplicated retrain requests when alerts occur.
    """
    db_file = tmp_path / "test_monitor.db"
    db_url = f"sqlite:///{db_file}"
    init_db(db_url)
    session = get_session(db_url)

    run_id = "test_perf_run"

    # Insert predictions without outcomes
    for i in range(50):
        pred = PredictionRecord(
            id=f"pred-mon-{i}",
            transaction_id=f"tx-mon-{i}",
            replay_run_id=run_id,
            event_step=100,
            features={"type": "TRANSFER", "amount": 1000.0},
            fraud_score=0.80 if i < 10 else 0.10,
            decision=True if i < 10 else False,
            decision_threshold=0.50,
            model_name="test_model",
            model_version="v1",
            schema_version="v1",
            inference_ms=1.0,
        )
        session.add(pred)
    session.commit()

    perf_monitor = PerformanceMonitor(min_labeled_samples=20, min_positive_labels=2)

    # 1. Evaluate with 0 outcomes -> must be PERFORMANCE_UNAVAILABLE
    res_unlabeled = perf_monitor.evaluate_window_performance(
        db=session, replay_run_id=run_id, window_start=95, window_end=105
    )
    assert res_unlabeled["status"] == "PERFORMANCE_UNAVAILABLE"
    assert res_unlabeled["labeled_count"] == 0

    # 2. Add delayed outcomes
    for i in range(40):
        outcome = OutcomeRecord(
            prediction_id=f"pred-mon-{i}",
            label=1 if i < 5 else 0,
            label_available_step=100,
            label_source="synthetic_replay",
        )
        session.add(outcome)
    session.commit()

    # 3. Re-evaluate with outcomes present
    res_labeled = perf_monitor.evaluate_window_performance(
        db=session, replay_run_id=run_id, window_start=95, window_end=105
    )
    assert res_labeled["status"] == "NORMAL"
    assert res_labeled["labeled_count"] == 40
    assert res_labeled["positive_count"] == 5
    assert res_labeled["metrics"]["recall"] > 0.0

    # 4. Trigger retrain and verify deduplication
    alert_report_data = {
        "status": "ALERT",
        "sample_count": 50,
        "positive_count": 5,
        "metrics": {"recall": 0.10},
    }
    req1 = perf_monitor.record_report_and_check_retrain(
        db=session,
        replay_run_id=run_id,
        window_start=95,
        window_end=105,
        report_kind="labeled_performance",
        report_data=alert_report_data,
    )
    assert req1 is not None

    # Call again for same window -> must deduplicate (return None, no duplicate DB record)
    req2 = perf_monitor.record_report_and_check_retrain(
        db=session,
        replay_run_id=run_id,
        window_start=95,
        window_end=105,
        report_kind="labeled_performance",
        report_data=alert_report_data,
    )
    assert req2 is None

    total_requests = (
        session.query(RetrainRequestRecord)
        .filter(RetrainRequestRecord.simulated_cutoff == 105)
        .count()
    )
    assert total_requests == 1

    session.close()

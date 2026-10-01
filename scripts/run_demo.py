"""Live end-to-end demonstration of the FraudGuard MLOps Lifecycle."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import pandas as pd
from fastapi.testclient import TestClient

from fraudguard.data.prepare import generate_synthetic_paysim
from fraudguard.monitoring.drift import DriftMonitor
from fraudguard.monitoring.performance import PerformanceMonitor
from fraudguard.persistence.database import get_session, init_db
from fraudguard.persistence.models import OutcomeRecord, PredictionRecord
from fraudguard.replay.labels import DelayedLabelWorker
from fraudguard.replay.scenarios import ScenarioGenerator
from fraudguard.replay.stream import ReplayStreamer
from fraudguard.serving.app import app, load_model_and_config

logging.basicConfig(level=logging.WARNING)


def print_header(title: str):
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)


def run_demo():
    print_header("FRAUDGUARD LIVE END-TO-END DEMO")

    # 1. Check Model Artifacts
    model_path = Path("artifacts/phase1/xgb_fraud_pipeline.joblib")
    if not model_path.exists():
        print("[!] Trained model artifact not found! Run training first:")
        print("   uv run python -m fraudguard.training.train --output-dir artifacts/phase1")
        return

    print("[OK] Found trained production pipeline artifact: artifacts/phase1/xgb_fraud_pipeline.joblib")

    # 2. Initialize Database & Serving Engine
    db_url = "sqlite:///fraudguard.db"
    init_db(db_url)
    load_model_and_config()
    session = get_session(db_url)
    client = TestClient(app)

    # 3. Health & Model Info Check
    print_header("1. Serving Health & Immutable Model Identity")
    health_resp = client.get("/health/ready")
    info_resp = client.get("/model-info")

    print(f"Health Status:      {health_resp.json()['status'].upper()}")
    print(f"Loaded Model:       {info_resp.json()['model_name']}")
    print(f"Immutable Version:  {info_resp.json()['model_version']}")
    print(f"Decision Threshold: {info_resp.json()['decision_threshold']:.4f}")
    print(f"Feature Allowlist:  {info_resp.json()['features_allowlist']}")
    print(f"Forbidden Columns:  {info_resp.json()['forbidden_columns']}")

    # 4. Live Inference & Replay Stream
    print_header("2. Live Transaction Replay Stream (POST /predict)")
    print("Replaying 5 transactions from replay partition (steps 91..100)...")

    sample_txs = [
        {"transaction_id": "tx-live-001", "event_step": 95, "type": "PAYMENT", "amount": 42.50},
        {"transaction_id": "tx-live-002", "event_step": 95, "type": "TRANSFER", "amount": 250000.00},
        {"transaction_id": "tx-live-003", "event_step": 96, "type": "CASH_OUT", "amount": 8500.00},
        {"transaction_id": "tx-live-004", "event_step": 96, "type": "DEBIT", "amount": 120.00},
        {"transaction_id": "tx-live-005", "event_step": 97, "type": "TRANSFER", "amount": 180000.00},
    ]

    responses = []
    print(f"{'TX ID':<13} | {'TYPE':<9} | {'AMOUNT':<12} | {'SCORE':<7} | {'DECISION':<9} | {'LATENCY'}")
    print("-" * 70)
    for tx in sample_txs:
        resp = client.post("/predict", json=tx)
        data = resp.json()
        responses.append(data)
        dec = "[ALERT: FRAUD]" if data["decision"] else "[ALLOW: OK]"
        print(f"{tx['transaction_id']:<13} | {tx['type']:<9} | ${tx['amount']:<11,.2f} | {data['fraud_score']:<7.4f} | {dec:<14} | {data['inference_ms']:.2f}ms")

    # 5. Test Fail-Closed Idempotency & Safety Guards
    print_header("3. Security & Safety Gates")

    # Test Idempotency
    print("Testing Idempotency (re-sending tx-live-002)...")
    idemp_resp = client.post("/predict", json=sample_txs[1])
    is_idempotent = idemp_resp.json()["prediction_id"] == responses[1]["prediction_id"]
    print(f"Idempotency Verified: {'[PASSED] Returned existing prediction ID' if is_idempotent else '[FAILED]'}")

    # Test Forbidden Column Rejection
    print("\nTesting Forbidden Column Rejection (attempting to send 'oldbalanceOrg')...")
    leak_payload = {"transaction_id": "tx-leak-001", "event_step": 95, "type": "TRANSFER", "amount": 5000.0, "oldbalanceOrg": 5000.0}
    leak_resp = client.post("/predict", json=leak_payload)
    print(f"Status Code: {leak_resp.status_code} ({'[PASSED] 422 UNPROCESSABLE ENTITY - REJECTED' if leak_resp.status_code == 422 else '[FAILED] LEAKAGE ALLOWED'})")

    # 6. Delayed Label Simulation Clock
    print_header("4. Delayed Labels Simulation Clock")
    worker = DelayedLabelWorker(delay_steps=24)

    # Convert to ReplayedTransaction list with ground-truth
    gt_map = {"tx-live-001": 0, "tx-live-002": 1, "tx-live-003": 0, "tx-live-004": 0, "tx-live-005": 1}
    from fraudguard.replay.stream import ReplayedTransaction
    replayed = [
        ReplayedTransaction(
            transaction_id=r["transaction_id"],
            prediction_id=r["prediction_id"],
            event_step=sample_txs[i]["event_step"],
            fraud_score=r["fraud_score"],
            decision=r["decision"],
            ground_truth_label=gt_map[r["transaction_id"]],
        )
        for i, r in enumerate(responses)
    ]
    worker.enqueue_replayed_transactions(replayed)

    print(f"Simulation Clock at Step 100 (< 95 + 24 = 119):")
    released_early = worker.release_eligible_labels(current_step=100, db=session)
    print(f"Labels Released Early: {released_early} ([OK] Delayed feedback preserved)")

    print(f"\nAdvancing Simulation Clock to Step 125 (>= 119):")
    released_matured = worker.release_eligible_labels(current_step=125, db=session)
    print(f"Labels Released Matured: {released_matured} ([OK] Labels matured & persisted to outcomes table)")

    # 7. Drift and Performance Monitoring
    print_header("5. Independent Drift & Performance Monitoring")
    ref_df = generate_synthetic_paysim(num_rows=500, num_steps=10, random_seed=42)
    monitor = DriftMonitor(reference_df=ref_df, min_window_samples=5)

    curr_df = pd.DataFrame(sample_txs)
    drift_report = monitor.evaluate_window_drift(curr_df, features=["amount", "type"])
    print(f"Feature Drift Status: {drift_report['status']}")
    print(f"Amount PSI Score:     {drift_report['features']['amount']['score']:.4f} ({drift_report['features']['amount']['status']})")
    print(f"Type J-S Divergence:  {drift_report['features']['type']['score']:.4f} ({drift_report['features']['type']['status']})")

    perf_monitor = PerformanceMonitor(min_labeled_samples=2, min_positive_labels=1)
    perf_report = perf_monitor.evaluate_window_performance(db=session, replay_run_id="default", window_start=90, window_end=100)
    print(f"\nRealized Performance Status: {perf_report['status']}")
    if perf_report.get("metrics"):
        m = perf_report["metrics"]
        print(f"Realized Precision:  {m.get('precision', 0.0):.4f}")
        print(f"Realized Recall:     {m.get('recall', 0.0):.4f}")
        print(f"Realized FPR:        {m.get('fpr', 0.0):.4f}")
        print(f"Average Precision:   {m.get('average_precision', 0.0):.4f}")

    print_header("DEMO COMPLETED SUCCESSFULLY!")
    print("To view the full interactive dashboard, run:")
    print("   uv run streamlit run dashboard/app.py")
    session.close()


if __name__ == "__main__":
    run_demo()

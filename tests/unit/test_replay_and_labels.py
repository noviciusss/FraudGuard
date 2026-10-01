"""Unit tests for Phase 3: Scenario generation, transaction streaming, and delayed label release."""

import pandas as pd

from fraudguard.data.prepare import generate_synthetic_paysim
from fraudguard.persistence.database import get_session, init_db
from fraudguard.persistence.models import OutcomeRecord, PredictionRecord
from fraudguard.replay.labels import DelayedLabelWorker
from fraudguard.replay.scenarios import ScenarioGenerator
from fraudguard.replay.stream import ReplayStreamer


def test_scenario_generation():
    """Verify stable and mix_shift scenario properties."""
    df = generate_synthetic_paysim(num_rows=1000, num_steps=20, random_seed=42)
    gen = ScenarioGenerator(df, random_seed=42)

    stable_df, stable_manifest = gen.generate_stable_scenario("stable_test")
    assert len(stable_df) == 1000
    assert stable_manifest.scenario_type == "stable"

    shifted_df, shifted_manifest = gen.generate_mix_shift_scenario("shift_test", transfer_weight_multiplier=5.0)
    assert len(shifted_df) == 1000
    assert shifted_manifest.scenario_type == "mix_shift"

    # Verify shift towards TRANSFER
    transfer_pct_original = (df["type"] == "TRANSFER").mean()
    transfer_pct_shifted = (shifted_df["type"] == "TRANSFER").mean()
    assert transfer_pct_shifted > transfer_pct_original


def test_streaming_and_delayed_label_release(tmp_path):
    """Verify transaction streamer omits labels and label worker releases only at simulated step."""
    db_file = tmp_path / "test_replay.db"
    db_url = f"sqlite:///{db_file}"
    init_db(db_url)
    session = get_session(db_url)

    # Mock predict function that records to DB
    captured_payloads = []

    def mock_predict(payload):
        captured_payloads.append(payload)
        pred_id = f"pred-{payload['transaction_id']}"
        rec = PredictionRecord(
            id=pred_id,
            transaction_id=payload["transaction_id"],
            replay_run_id=payload["replay_run_id"],
            event_step=payload["event_step"],
            features={"type": payload["type"], "amount": payload["amount"]},
            fraud_score=0.85,
            decision=True,
            decision_threshold=0.50,
            model_name="mock_model",
            model_version="v1",
            schema_version="v1",
            inference_ms=1.5,
        )
        session.add(rec)
        session.commit()
        return {
            "prediction_id": pred_id,
            "fraud_score": 0.85,
            "decision": True,
            "decision_threshold": 0.50,
        }

    # Generate small sample at step 100
    sample_df = pd.DataFrame(
        {
            "step": [100, 100, 100],
            "type": ["TRANSFER", "CASH_OUT", "PAYMENT"],
            "amount": [50000.0, 10000.0, 50.0],
            "isFraud": [1, 0, 0],
        }
    )

    streamer = ReplayStreamer(predict_fn=mock_predict, replay_run_id="run_delayed_test")
    replayed = streamer.stream_dataframe(sample_df)

    # 1. Verify inference requests strictly lack labels or balance fields
    assert len(captured_payloads) == 3
    for p in captured_payloads:
        assert "isFraud" not in p
        assert "oldbalanceOrg" not in p

    # 2. Delayed label release worker with 24-step delay (available at step 124)
    worker = DelayedLabelWorker(delay_steps=24)
    worker.enqueue_replayed_transactions(replayed)

    # Check at simulation step 110 (before delay)
    released_at_110 = worker.release_eligible_labels(current_step=110, db=session)
    assert released_at_110 == 0

    outcomes_count_110 = session.query(OutcomeRecord).count()
    assert outcomes_count_110 == 0  # No labels released prematurely

    # Check at simulation step 125 (after delay)
    released_at_125 = worker.release_eligible_labels(current_step=125, db=session)
    assert released_at_125 == 3

    outcomes_count_125 = session.query(OutcomeRecord).count()
    assert outcomes_count_125 == 3

    # Check stored values
    positive_outcome = session.query(OutcomeRecord).filter(OutcomeRecord.label == 1).first()
    assert positive_outcome is not None
    assert positive_outcome.label_available_step == 124

    session.close()

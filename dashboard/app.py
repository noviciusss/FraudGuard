"""FraudGuard Read-Only Operations and Observability Dashboard (Streamlit)."""

from __future__ import annotations

import os

import pandas as pd
import streamlit as st
from sqlalchemy import func

from fraudguard.persistence.database import get_session, init_db
from fraudguard.persistence.models import (
    DeploymentEventRecord,
    MonitoringReportRecord,
    OutcomeRecord,
    PredictionRecord,
    RetrainRequestRecord,
)

st.set_page_config(
    page_title="FraudGuard — MLOps Observability",
    page_icon="🛡️",
    layout="wide",
)

st.title("🛡️ FraudGuard — Drift-Monitored Fraud Lifecycle Dashboard")
st.markdown(
    "**Operational Observability**: Real-time serving status, delayed label coverage, "
    "covariate drift monitoring, and candidate lifecycle audits."
)

db_url = os.getenv("DATABASE_URL", "sqlite:///fraudguard.db")
init_db(db_url)
session = get_session(db_url)

# Top KPI Metric Cards
col1, col2, col3, col4 = st.columns(4)

total_predictions = session.query(func.count(PredictionRecord.id)).scalar() or 0
total_outcomes = session.query(func.count(OutcomeRecord.prediction_id)).scalar() or 0
total_frauds = (
    session.query(func.count(OutcomeRecord.prediction_id))
    .filter(OutcomeRecord.label == 1)
    .scalar()
    or 0
)
label_coverage = (total_outcomes / total_predictions * 100.0) if total_predictions > 0 else 0.0

latest_pred = (
    session.query(PredictionRecord)
    .order_by(PredictionRecord.received_at.desc())
    .first()
)
active_version = latest_pred.model_version if latest_pred else "v1_init"
active_thresh = latest_pred.decision_threshold if latest_pred else 0.50

with col1:
    st.metric("Total Predictions Logged", f"{total_predictions:,}")
with col2:
    st.metric("Delayed Labels Matured", f"{total_outcomes:,}", f"{label_coverage:.1f}% coverage")
with col3:
    st.metric("Confirmed Frauds", f"{total_frauds:,}", f"{(total_frauds/max(1, total_outcomes)*100.0):.2f}% prevalence")
with col4:
    st.metric("Active Model Version", active_version, f"Threshold: {active_thresh:.4f}")

st.divider()

# Tab Layout
tab1, tab2, tab3, tab4 = st.tabs(
    ["📊 Drift & Performance", "🔍 Recent Predictions", "⚙️ Retraining Queue", "📜 Deployment Events"]
)

with tab1:
    st.subheader("Monitoring Reports & Drift Alerts")
    reports = (
        session.query(MonitoringReportRecord)
        .order_by(MonitoringReportRecord.created_at.desc())
        .limit(20)
        .all()
    )
    if reports:
        report_data = []
        for r in reports:
            report_data.append(
                {
                    "ID": r.id[:8],
                    "Kind": r.report_kind,
                    "Step Window": f"{r.window_start}..{r.window_end}",
                    "Status": r.status,
                    "Sample Count": r.sample_count,
                    "Positive Labels": r.positive_label_count or 0,
                    "Created At": r.created_at.strftime("%Y-%m-%d %H:%M:%S") if r.created_at else "",
                }
            )
        st.dataframe(pd.DataFrame(report_data), use_container_width=True)
    else:
        st.info("No monitoring reports generated yet. Run the replay or monitoring worker.")

with tab2:
    st.subheader("Audited Prediction Logs (Fail-Closed)")
    preds = (
        session.query(PredictionRecord)
        .order_by(PredictionRecord.received_at.desc())
        .limit(50)
        .all()
    )
    if preds:
        pred_data = []
        for p in preds:
            pred_data.append(
                {
                    "Transaction ID": p.transaction_id,
                    "Step": p.event_step,
                    "Type": p.features.get("type", "N/A"),
                    "Amount": f"${p.features.get('amount', 0.0):,.2f}",
                    "Fraud Score": f"{p.fraud_score:.4f}",
                    "Decision": "🚨 ALERT" if p.decision else "✅ ALLOW",
                    "Latency (ms)": f"{p.inference_ms:.2f}",
                    "Model Version": p.model_version,
                }
            )
        st.dataframe(pd.DataFrame(pred_data), use_container_width=True)
    else:
        st.info("No predictions recorded yet.")

with tab3:
    st.subheader("Retraining Requests & Automated Gates")
    retrains = (
        session.query(RetrainRequestRecord)
        .order_by(RetrainRequestRecord.id.desc())
        .all()
    )
    if retrains:
        retrain_data = []
        for r in retrains:
            retrain_data.append(
                {
                    "Dedup Key": r.deduplication_key,
                    "Cutoff Step": r.simulated_cutoff,
                    "Status": r.status,
                    "Candidate Version": r.candidate_version or "N/A",
                    "Reason": r.reason,
                    "Failure Reason": r.failure_reason or "None",
                }
            )
        st.dataframe(pd.DataFrame(retrain_data), use_container_width=True)
    else:
        st.info("No retrain requests generated.")

with tab4:
    st.subheader("Deployment & Rollback History")
    deployments = (
        session.query(DeploymentEventRecord)
        .order_by(DeploymentEventRecord.started_at.desc())
        .all()
    )
    if deployments:
        dep_data = []
        for d in deployments:
            dep_data.append(
                {
                    "Candidate": d.candidate_version,
                    "Previous": d.previous_version,
                    "Status": d.status,
                    "Revision": d.deployment_revision,
                    "Started At": d.started_at.strftime("%Y-%m-%d %H:%M:%S") if d.started_at else "",
                    "Verified At": d.verified_at.strftime("%Y-%m-%d %H:%M:%S") if d.verified_at else "N/A",
                    "Rollback Reason": d.rollback_reason or "N/A",
                }
            )
        st.dataframe(pd.DataFrame(dep_data), use_container_width=True)
    else:
        st.info("No deployment events recorded.")

session.close()

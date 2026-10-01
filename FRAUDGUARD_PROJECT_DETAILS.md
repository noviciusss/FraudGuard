# FraudGuard: Detailed System Architecture & Project Blueprint

## Executive Summary & Honest Positioning

**FraudGuard** is an end-to-end Machine Learning lifecycle engineering and MLOps system built on the synthetic PaySim financial transaction domain. It simulates realistic production challenges that real-world fraud and risk infrastructure faces:
- Strictly avoiding target leakage and simulator artifacts.
- Enforcing non-overlapping chronological step partitions.
- Deploying a fail-closed, versioned, and idempotent inference service.
- Modeling asynchronous, delayed feedback using an event-time simulation clock.
- Independently monitoring covariate feature drift from realized model performance degradation.
- Applying automated, multi-metric quality gates for candidate retraining, rejection, and emergency rollback.

> **Honest Positioning**: This project is an engineering demonstration of production ML lifecycle governance. It uses synthetic PaySim transactions and a deterministic step-based clock. It is not an actual banking-grade fraud prevention engine, nor evidence of live generalization to real-world financial streams.

---

## 1. Feature Engineering & Leakage Prevention Contract

### Why Balance Columns Are Excluded
The PaySim dataset simulator models a cancellation rule: detected fraud attempts are cancelled before completion. Consequently:
$$\text{oldbalanceOrg} - \text{newbalanceOrig} = 0$$
often gives away whether a fraud transaction was intercepted within the simulation. Using balance columns yields unrealistically inflated accuracy offline that completely fails in real-world scenarios.

### Strict Allowlist & Forbidden Columns Policy
| Feature | Type | Status | Rationale |
|---|---|---|---|
| `type` | Categorical | **ALLOWLIST** | One-Hot encoded with unknown category tolerance (`handle_unknown='ignore'`). |
| `amount` | Float | **ALLOWLIST** | Finite, non-negative transaction magnitude. |
| `log1p_amount` | Float | **DERIVED** | $\log(1 + \text{amount})$ to compress heavy-tailed distributions. |
| `sin_hour`, `cos_hour` | Float | **DERIVED** | Cyclic 24-hour hour encoding derived from simulation step ($\text{step} \pmod{24}$). |
| `oldbalanceOrg`, `newbalanceOrig` | Float | **FORBIDDEN** | Excluded due to simulator cancellation leakage. |
| `oldbalanceDest`, `newbalanceDest` | Float | **FORBIDDEN** | Excluded due to simulator cancellation leakage. |
| `nameOrig`, `nameDest` | String | **FORBIDDEN** | High-cardinality account IDs excluded by project design. |
| `isFlaggedFraud` | Integer | **FORBIDDEN** | Simulator business heuristic rule. |
| `isFraud` | Integer | **TARGET ONLY** | Supervised outcome; strictly rejected if sent to `/predict`. |

The inference service enforces `extra = "forbid"` via Pydantic: sending any forbidden field immediately triggers HTTP 422 Unprocessable Entity.

---

## 2. Temporal Splitting by Complete Steps

Data splits are partitioned strictly by **complete integer simulation steps** (hours) to eliminate intra-hour leakage:
- **Train (Steps 1..72, ~60%)**: 29,972 transactions, 135 frauds (0.45%)
- **Tune (Steps 73..90, ~15%)**: 7,569 transactions, 37 frauds (0.49%)
- **Replay Window (Steps 91..108, ~15%)**: 7,448 transactions, 53 frauds (0.71%)
- **Final Audit (Steps 109..120, ~10%)**: 5,011 transactions, 25 frauds (0.50%)

Every partition is cataloged in `data_manifest.json` with cryptographic SHA-256 hashes, exact row counts, and positive class prevalence.

---

## 3. Modeling & Threshold Optimization Policy

### Algorithms
1. **Baseline**: `LogisticRegression(class_weight='balanced', solver='lbfgs', max_iter=1000)`
2. **Production Champion**: `XGBClassifier(n_estimators=150, max_depth=5, learning_rate=0.08, scale_pos_weight=10.0)`

### Threshold Policy
Threshold selection is performed exclusively on the **Tune** partition. It maximizes recall subject to an operational False Positive Rate budget:
$$\text{FPR} \le 0.01 \quad (1\% \text{ false positive budget})$$

### Measured Benchmark Results
| Metric | Baseline (Logistic) | Production (XGBoost) |
|---|---|---|
| **Tune Average Precision (AP)** | 0.2154 | 0.1607 |
| **Tune Recall @ FPR $\le$ 1%** | 45.95% | 37.84% |
| **Audit Average Precision (AP)** | 0.2012 | 0.1592 |
| **Audit Recall @ Fixed Threshold** | 40.00% | 28.00% |
| **Audit FPR** | 0.98% | 1.18% |

*(Note: Without leaking balance columns, large transfer amounts are linear signals that Logistic Regression captures effectively; the framework objectively benchmarks both).*

---

## 4. Serving Architecture & API Contract

### FastAPI Serving Endpoints
- `POST /predict`: Real-time inference with:
  - **Idempotency**: Submitting the same `transaction_id` within a `replay_run_id` returns the existing prediction without creating duplicate database rows.
  - **Fail-Closed Audit Logging**: The prediction is synchronously committed to PostgreSQL/SQLite before returning HTTP 200.
  - **Immutable Identity**: Exposes the exact immutable model version (e.g. `v1_local` or registered version ID) rather than a mutable alias pointer.
- `GET /health/live`: Fast process liveness probe.
- `GET /health/ready`: Deep readiness probe verifying model pipeline in memory and executing `SELECT 1` on the database.
- `GET /model-info`: Returns model name, immutable version, active decision threshold, allowlisted features, and forbidden columns.
- `GET /drift-status`: Returns the 10 most recent monitoring and drift alert reports.

---

## 5. Temporal Simulation Replay & Delayed Labels

In real-world fraud operations, chargeback confirmations and fraud investigations take days or weeks to mature.
- **Replay Streamer**: Replays chronological transactions against `/predict`. Labels and balance columns are stripped before sending.
- **Delayed Label Worker**: Manages a scheduled queue where `label_available_step = event_step + delay_steps` (e.g., 24 simulation steps = 1 simulated day).
- Labels are released into the `outcomes` table strictly when `label_available_step <= current_step`.

---

## 6. Drift Detection & Labeled Performance Monitoring

### Independent Reporting
1. **Covariate Feature & Prediction Drift**:
   - Compares streaming prediction windows against frozen reference `v1_dev_train`.
   - Uses **Population Stability Index (PSI)** for numerical features and fraud score.
   - Uses **Jensen-Shannon Divergence** for categorical transaction types.
   - Thresholds: Warning $\ge 0.10$, Alert $\ge 0.20$.
2. **Realized Performance Evaluation**:
   - Joins `predictions` with released `outcomes`.
   - If a window lacks sufficient matured labels, it reports `status = "PERFORMANCE_UNAVAILABLE"`, preventing misleading zero-recall alarms.
   - Sustained alerts create a deduplicated `RetrainRequestRecord`.

---

## 7. Automated Retraining, Validation Gating & Rollback

### Candidate Retraining
- Retraining jobs query only labels where `label_available_step <= cutoff`.
- Candidate model is fit on candidate-train, threshold is selected on candidate-tune.
- Candidate and current champion are evaluated on the identical unseen comparison window.

### Quality Gate Criteria
A candidate is authorized for deployment only if:
1. Candidate $\text{Recall} - \text{Champion Recall} \ge 0.00$ (non-inferiority).
2. Candidate $\text{FPR} \le 0.01$ (within operational false positive budget).
3. Candidate $\text{Precision} \ge 0.15$ (above safety floor).
4. Candidate $\text{AP} \ge \text{Champion AP} - 0.02$ (bounded degradation tolerance).
5. Comparison set meets minimum sample volume ($N \ge 100$, positives $\ge 5$).

### Rejection and Rollback
- Sub-par candidates are rejected; the champion serving version remains completely untouched.
- Deployment verification failure halts rollout and preserves current champion.
- Emergency rollback restores the `previous` verified model alias and logs the audit event.

---

## 8. Verification & Test Integrity

The test suite contains 20 comprehensive automated tests:
```bash
uv run pytest tests/ -v
```
- **Unit Tests**:
  - `test_synthetic_paysim_generation`: Verifies schema, step bounds, and fraud clustering.
  - `test_leakage_safe_step_splits`: Verifies non-overlapping step partitions.
  - `test_forbidden_columns_stripped`: Confirms balance and account IDs cannot enter features.
  - `test_unknown_categorical_handling`: Validates unseen category fault tolerance.
  - `test_psi_and_drift_calculation`: Tests PSI sensitivity on stable vs shifted windows.
  - `test_drift_monitor_window_evaluation`: Validates `NORMAL`, `ALERT`, and `INSUFFICIENT_DATA`.
  - `test_performance_monitor_delayed_labels_and_deduplication`: Confirms `PERFORMANCE_UNAVAILABLE` on unlabeled windows and retrain deduplication.
  - `test_scenario_generation`: Validates stable and mix-shift scenario distributions.
  - `test_streaming_and_delayed_label_release`: Verifies simulation clock gating on label releases.
  - `test_metric_computation`: Tests AP, precision, recall, FPR, and Brier score math.
  - `test_threshold_optimization_fpr_budget`: Validates threshold selection under FPR constraints.
  - `test_training_pipelines_end_to_end`: Tests full fit/predict pipelines for Baseline and XGBoost.
- **Integration Tests**:
  - `test_health_live_endpoint`
  - `test_health_ready_endpoint`
  - `test_model_info_endpoint`
  - `test_predict_success_and_idempotency`
  - `test_predict_rejects_forbidden_columns` (HTTP 422 on balance fields)
  - `test_predict_rejects_negative_amount`
- **Lifecycle & Rollback Safety Tests**:
  - `test_promotion_gate_decisions`: Rejection of bad recall, bad FPR, or small sample candidates.
  - `test_deployment_failure_and_rollback_safety`: Verification failure safety and rollback state restoration.

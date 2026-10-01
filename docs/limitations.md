# FraudGuard Scope, Limitations & Honest Positioning

## 1. What This Project Is
FraudGuard is an engineering demonstration of end-to-end Machine Learning lifecycle governance, addressing:
- Reproducible temporal model training without future or intra-hour leakage.
- Strict schema validation and fail-closed persistence auditing.
- Delayed feedback simulation using a step-based clock rather than unrealistic real-time assumptions.
- Separation of covariate drift alerts from realized classification degradation.
- Multi-metric promotion gating with verified rejection and rollback mechanics.

## 2. What This Project Is NOT
- **Not a Banking-Grade Fraud System**: PaySim is a synthetic multi-agent simulator with simplified behavior rules. Results cannot be directly transferred to real financial transaction networks.
- **Not Evidence of Live Stream Generalization**: Replay simulates event-time progression deterministically; real production traffic exhibits concurrent race conditions, network partition drops, and active adversarial fraud adaptation.
- **Not Calibrated Posterior Probabilities**: Boosted trees with class weighting yield risk scores that can be used for ranking transactions, but raw score outputs are not calibrated probabilities.

## 3. Operational Performance Characteristics
- **Fail-Closed Availability Trade-Off**: Every prediction request synchronously writes an audit record to the persistence store before returning HTTP 200. If the database is unreachable, requests fail with HTTP 503 rather than serving unaudited inferences. In high-throughput banking systems, asynchronous message queuing (e.g. Kafka buffer) would be employed.
- **Inference Latency**: Warm API inference latency is typically 1.5ms - 5ms locally (excluding network transport and synchronous SQLite/Postgres commit latency).

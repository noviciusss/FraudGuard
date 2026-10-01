# FraudGuard — Drift-Monitored Fraud Detection and MLOps Pipeline

## 1. Purpose and honest positioning

Build a supervised ML service on synthetic PaySim transactions, with reproducible training, versioned serving, delayed-label simulation, drift monitoring, and validation-gated model updates.

This is a portfolio demonstration of ML lifecycle engineering. It is not a banking-grade fraud system, not evidence of generalization to real financial data, and not an actual live transaction stream.

Placement positioning: use alongside DoCopilot and the Transformer Fine-Tuning Suite for broader ML/MLOps applications. Replace Argus in that resume variant only after this project is completed. Keep applying while building.

### Success criteria

- A baseline and boosted-tree model are compared on leakage-aware temporal splits.
- The same preprocessing artifact is used in training and serving.
- Predictions are linked to delayed outcome labels without exposing labels to inference.
- Drift and labeled performance are reported separately.
- A candidate is trained only from labels available at the simulated cutoff.
- Candidate and champion are evaluated on the same unseen comparison window.
- Rejection, deployment failure, and rollback work in tests.
- The README clearly separates measured behavior, synthetic scenarios, and unimplemented features.

## 2. Corrections to the original plan

| Change | Implementation consequence |
|---|---|
| Replace “watches accuracy degrade” pitch | Drift is a proxy; performance requires labels |
| Exclude warned-against balance columns | Smaller but more defensible feature set |
| Remove raw account IDs and isFlaggedFraud from initial model | Conservative feature policy, not a claim these are universally leakage |
| Add delayed labels | Separate outcomes table and label-release worker |
| Split by complete time steps | No same-hour overlap between temporal partitions |
| Separate tuning, comparison, and final audit data | Prevent repeated test-set tuning |
| Gate precision/recall and operational behavior | Recall alone cannot authorize promotion |
| Use MLflow aliases | champion, candidate, previous |
| Resolve aliases to immutable versions | Audit serving identity and prevent silent changes |
| Keep training triggers private | No public endpoint dispatching privileged workflows |
| Version scenario and data manifests | Reproducible replay and evaluations |
| Add rollback and failure tests | Prove lifecycle safety, not only happy path |
| Drop promises of guaranteed improvement | Valid retraining may correctly reject a candidate |

## 3. Stack to use

Choose one tool per function. Do not add tools for resume keywords.

| Layer | Chosen technology | Usage |
|---|---|---|
| Language | Python 3.11 | Common runtime; verify dependency compatibility before locking |
| Dependencies | uv + pyproject.toml + uv.lock | Pin resolved dependencies and reproduce environments |
| Data | Pandas, NumPy, PyArrow/Parquet | Typed data preparation and compact partitions |
| Baseline | scikit-learn LogisticRegression | Simple, explainable comparison |
| Main model | XGBoost | One boosted-tree implementation; no need for LightGBM too |
| Preprocessing | scikit-learn Pipeline/ColumnTransformer | Shared transformations and categorical handling |
| Experiment tracking | MLflow | Runs, metrics, artifacts, model signatures and versions |
| API | FastAPI, Pydantic, Uvicorn | Strict schemas, inference, health endpoints |
| Persistence | PostgreSQL, SQLAlchemy, Alembic, psycopg | Prediction logs, delayed labels, jobs, migrations |
| Monitoring | Evidently + scikit-learn metrics | Feature/prediction drift and label-based performance |
| Dashboard | Streamlit | Read-only operations view; build last |
| Tests | pytest, HTTPX | Unit, API, lifecycle and integration tests |
| Quality | Ruff | Lint and formatting |
| Containers | Docker, Docker Compose | Local API, DB, MLflow and worker services |
| CI/CD | GitHub Actions | Tests, build, controlled training and deployment workflows |
| Cloud | Azure Container Apps, ACR, Blob Storage | Serving, images and durable artifacts |
| Cloud database | Azure Database for PostgreSQL or another reachable managed PostgreSQL | Persistent application metadata; cost-dependent |
| Credentials | Azure managed identity where supported; GitHub OIDC for Azure | Avoid long-lived deployment credentials |
| Optional alert | Email or Slack webhook | One integration after persisted alerts work |

For local MLflow, use PostgreSQL as its metadata backend with a separate database/user, and a persistent artifact volume. For cloud, use durable artifact storage and an authenticated/private MLflow endpoint. A local SQLite file inside an ephemeral cloud container is not a persistence plan.

Do not add Kafka, Spark, Kubernetes, Airflow, Redis, Celery, a feature store, or a frontend framework to the MVP.

## 4. Learn these topics in order

### A. Before training

- Binary classification, logistic regression and gradient boosting.
- Imbalanced classification: precision, recall, false-positive rate, average precision, PR curves and confusion matrices.
- Threshold selection versus model fitting.
- Leakage, temporal validation, and fitting transformations only on training data.
- Feature availability at the actual decision time.

Deliverable: explain why predicting every transaction as legitimate can have high accuracy and still fail the task.

### B. While building training

- scikit-learn pipelines and categorical encoding.
- XGBoost regularization, early stopping, and class weighting.
- Calibration: a weighted classifier's score is not automatically a calibrated fraud probability.
- MLflow parameters, artifacts, signatures, registered versions and aliases.
- Data checksums, Git commits, random seeds, dependency locks and run manifests.

Deliverable: reproduce a run within documented numerical tolerance. Do not promise bitwise equality across different hardware.

### C. Before monitoring

- Covariate drift: changes in P(X).
- Label/prevalence shift: changes in P(y).
- Concept drift: changes in P(y|X).
- Why changing fraud prevalence alone is not automatically concept drift.
- Delayed feedback, missing labels, minimum sample requirements and detection false alarms.
- PSI or another distribution comparison: sensitivity to bins, sample size and thresholds.

Deliverable: demonstrate drift without assuming performance has declined.

### D. Before automation/deployment

- Idempotency, retry policies, unique constraints and job locks.
- Migration management and database connectivity.
- Model version pinning, health/readiness checks, deployment verification and rollback.
- GitHub Actions workflow permissions, secrets and Azure OIDC.
- Container lifecycle and durable versus ephemeral storage.

Deliverable: explain what happens if training succeeds but deployment fails.

## 5. Dataset and feature contract

### Dataset

Use the PaySim dataset from its original public dataset page. Record source, downloaded file checksum, license, row count, column types, minimum/maximum step and fraud counts. Do not commit the raw CSV to Git.

PaySim is synthetic. Its dataset card explicitly warns against using oldbalanceOrg, newbalanceOrig, oldbalanceDest and newbalanceDest for fraud detection because detected fraud transactions are cancelled.

### Initial feature allowlist

- type
- amount
- Derived log1p(amount)
- Derived cyclic hour-of-simulation feature from step, if supported by the task definition

Keep step as event-time metadata for ordering. Raw step can encode simulator chronology and does not naturally generalize indefinitely; start without it as a predictor and report any later ablation explicitly.

Drop all four balance fields, raw account identifiers, isFlaggedFraud, and isFraud from model inputs. isFraud is the supervised target only.

IDs and isFlaggedFraud are exclusions by project design. Do not describe them as definitively leaking labels in all conceivable systems.

### Shared transformations

Fit categorical encoding, scaling and any imputation on the training partition only. Serialize the whole preprocessing-plus-model pipeline. Use explicit unknown-category handling and test it. No separate handwritten serving transformations.

### Validation

- amount: finite, nonnegative.
- type: documented enum; reject unsupported types initially.
- transaction_id: nonempty bounded string; not a model feature.
- event_step: nonnegative integer, used as simulation metadata.
- schema_version: supported value.
- Reject NaN, infinity, unexpected fields, labels and forbidden balance columns.

## 6. Splits and data governance

### Initial development split

Sort by step, preserving deterministic row order within each step. Choose boundaries by complete steps, not exact row percentages. Approximate proportions are acceptable; publish exact resulting counts.

- Earliest approximately 60%: development training.
- Next approximately 15%: hyperparameter/threshold tuning.
- Next approximately 15%: development replay and lifecycle demonstration.
- Latest approximately 10%: untouched final audit.

Inspect positive and negative counts in every partition before training. A split with too few fraud examples is not a reliable evaluation set. Adjust boundaries transparently before tuning, never based on desired scores.

Use small deterministic development samples only for plumbing. Report headline metrics on a defined evaluation partition without silently undersampling its natural fraud prevalence.

### Candidate lifecycle windows

At simulated cutoff T:

1. Select only labels whose label_available_at <= T.
2. Construct disjoint chronological candidate-train, candidate-tune and candidate-comparison windows from available outcomes.
3. Fit on candidate-train, select threshold on candidate-tune.
4. Evaluate frozen candidate and champion on the same candidate-comparison examples.
5. Ensure the comparison rows were not used to fit either model. Track cumulative training manifests.
6. Advance windows for later cycles. Do not repeatedly optimize against one tiny fixed comparison set.
7. Keep final audit data outside lifecycle training and promotion decisions.

Record identifiers and checksum for every split. If insufficient positive labels exist, record INSUFFICIENT_DATA and skip retraining or evaluation.

## 7. Modeling and evaluation

### Models

1. Dummy baseline.
2. Logistic regression pipeline.
3. XGBoost pipeline with a modest, predefined search budget.

Start without oversampling. If class weighting is used, tune it on development data and record the choice. Any sampling must be confined to fitting partitions.

### Metrics

- Average precision (AP), named explicitly rather than ambiguously called PR-AUC.
- Precision, recall, F1, FPR, TP, FP, TN, FN at the chosen threshold.
- ROC-AUC as a secondary metric.
- Alert/review rate and class prevalence.
- Calibration plot and Brier score if claiming meaningful probabilities.

For MVP, call the output fraud_score unless calibration has been evaluated. Do not promise a calibrated probability merely because predict_proba returns a number.

### Threshold policy

Choose a maximum FPR or review-rate budget on validation data. Optimize recall subject to that constraint. If using an illustrative cost function, document that costs are hypothetical, not supplied by a bank.

Never select thresholds on the final audit set.

### Artifacts

Confusion matrix, PR curve, threshold trade-off table, error analysis by transaction type and amount bucket, model card, feature policy, and a run manifest.

## 8. Architecture

```text
Historical synthetic dataset
  -> preparation + split/scenario manifests
  -> replay worker -> POST /predict -> predictions table
  -> private label-release worker -> outcomes table

predictions + reference snapshot -> drift monitor
predictions JOIN available outcomes -> performance monitor
monitors -> alerts + deduplicated retrain_requests

private scheduled workflow/worker
  -> available-label snapshot
  -> candidate train/tune/evaluate
  -> validation gate
  -> registry candidate version
  -> pinned deployment + readiness verification
  -> champion alias updated after successful deployment
  -> previous version retained for rollback

dashboard -> read-only queries of reports, jobs and serving identity
```

A registry alias is a mutable reference. The deployed API must expose the immutable version it actually loaded, not assume the alias guarantees all running replicas updated.

## 9. Database schema

Use UTC wall-clock timestamps and a separate simulated event-time/cutoff field. Do not mix accelerated replay time with real elapsed time.

### predictions

- id UUID primary key
- transaction_id unique within replay_run_id
- replay_run_id
- received_at
- event_step
- features JSONB, containing allowlisted values only
- fraud_score
- decision
- decision_threshold
- model_name, model_version
- schema_version
- inference_ms
- scenario_id

### outcomes

- prediction_id unique foreign key
- label (0 or 1)
- label_available_at / label_available_step
- released_at
- label_source (synthetic_replay)

### monitoring_reports

- id, replay_run_id, window_start, window_end
- reference_version
- sample_count, positive_label_count
- report_kind (data_drift, prediction_drift, labeled_performance)
- metrics JSONB
- status, artifact_uri, created_at

### retrain_requests

- id, unique deduplication_key
- triggering_report_id
- simulated_cutoff
- status, reason
- started_at, finished_at
- candidate_version, champion_version
- data_manifest_uri, evaluation_artifact_uri
- failure_reason

### deployment_events

- id, candidate_version, previous_version
- status, gate_summary, deployment_revision
- started_at, verified_at, rollback_reason

Use Alembic migrations and role separation. MLflow metadata should not share application table namespaces casually.

## 10. API contract

### POST /predict

```json
{
  "transaction_id": "demo-000001",
  "event_step": 100,
  "type": "TRANSFER",
  "amount": 181.0,
  "schema_version": "v1"
}
```

```json
{
  "prediction_id": "generated-uuid",
  "fraud_score": 0.87,
  "decision": true,
  "decision_threshold": 0.42,
  "model_version": "3",
  "schema_version": "v1"
}
```

The values above are examples, not measured results. Define retry/idempotency behavior: duplicate transaction IDs within a replay run return the original prediction instead of creating duplicate records.

### Other endpoints

- GET /health/live: process is alive; no heavyweight checks.
- GET /health/ready: model loaded and dependencies required for the chosen logging policy are available.
- GET /model-info: actual loaded immutable version, threshold, feature schema and build SHA.
- GET /drift-status: protected/read-only summary.

Labels, promotions, replay control and retraining are private worker operations, not unauthenticated public endpoints. Limit public request rate and size. Restrict CORS. Keep the dashboard read-only.

Logging policy for MVP: prediction is successful only after its audit record persists. If DB logging fails, return a retriable service error; document the availability trade-off. Add buffering only as a later improvement.

## 11. Replay and delayed labels

Replay rows in event-time order at configurable speed. Keep labels in the private replay/label process; never send them to /predict.

Release labels after a configurable number of simulated steps. The simulation clock, not a naive real-time sleep, determines label eligibility.

Modes:

- stable: unmodified held-out transactions.
- mix_shift: resample complete rows to alter transaction-type or amount mix while preserving original labels.
- synthetic_stress: explicitly modify features and/or generate labels using a documented synthetic rule.

Do not multiply amounts while retaining original labels and then claim authentic fraud-performance degradation. Such transformations change feature-label validity. Treat modified-label scenarios as artificial pipeline stress tests, not real fraud benchmarks.

For MVP, stable and mix_shift are enough. A concept-shift scenario is optional and must define a changed conditional label mechanism, not merely a higher fraud prevalence.

Store scenario configuration, random seed, original row references, transformation rules, and generated labels in a scenario manifest. Keep synthetic stress metrics separate from the untouched PaySim evaluation.

## 12. Monitoring and triggers

### Reference

Freeze a versioned reference snapshot from development training. Do not silently update it to eliminate alerts. Any reference update is an explicit logged operation.

### Drift

Use Evidently behind an internal adapter so its pinned API does not leak throughout the codebase. Choose feature-specific methods, record thresholds and explain them.

Calibrate trigger behavior by replaying stable windows first. There is no universal PSI threshold that proves model failure.

Configurable safeguards:

- minimum window size
- minimum labeled positives/negatives for performance metrics
- persistence across multiple eligible windows
- cooldown
- one active training job at a time
- deduplication key
- maximum automated attempts per replay run

Numeric values belong in configuration and must be justified by available event volume and label counts, not copied blindly.

### Performance

Join predictions to available outcomes and calculate metrics by actual model version. Show label coverage and sample count. An unlabeled dashboard window must say PERFORMANCE_UNAVAILABLE, not show zero recall.

Drift can create a retraining request, but it cannot directly promote a model. Insufficient labels produce an alert and deferred request.

## 13. Retraining and quality gates

Choose one executor for MVP:

- Local: private scheduled worker reading the database queue.
- Cloud: private scheduled job or GitHub Actions scheduled/manual workflow reading pending requests securely.

No public API holds a privileged GitHub dispatch token. Scheduled Actions is suitable for a demo, not a guarantee of immediate execution; expose time from alert to job start honestly.

Job steps:

1. Acquire lock and freeze a data manifest at cutoff.
2. Validate schema, label availability and sample sufficiency.
3. Train candidate and choose threshold on its tuning window.
4. Register immutable candidate version and assign candidate alias.
5. Evaluate candidate/champion on the same eligible comparison set.
6. Store gate decision with metrics and sample counts.
7. If rejected, leave serving unchanged.
8. If approved, deploy pinned version to a new revision.
9. Verify readiness, schema compatibility and smoke predictions.
10. Switch traffic in the controlled deployment process.
11. After successful verification, set previous and champion aliases.
12. On failure, keep/restore old serving version and record the event.

Gate policy:

- Recall must improve by a predefined minimum or pass a documented noninferiority policy.
- FPR/review budget must be met.
- Precision and AP must meet predefined floors/tolerances.
- Latency and schema tests must pass.
- Enough positive/negative labels must exist.
- Prefer paired uncertainty estimates when sample size permits; tiny differences are not proven wins.

Do not guarantee every drift event produces a better model. A rejected candidate is evidence that the gate works.

For an affordable MVP, package the approved immutable artifact into the serving image or download a pinned artifact at startup. The API need not contact the registry on every request. Record artifact checksum and model version in its manifest.

## 14. Deployment and cost control

Local Compose services: postgres, mlflow, api, worker, optional dashboard. Use health checks, migrations, named volumes and an explicit bootstrap command. Downloading the licensed dataset and configuring credentials are documented prerequisites, not magically solved by docker compose up.

Cloud:

- API in Azure Container Apps.
- Images in ACR.
- Artifacts/manifests in Blob Storage.
- Application and MLflow metadata in persistent PostgreSQL.
- MLflow privately reachable/authenticated if hosted.
- Training/monitoring job on a network that can reach required private resources.
- Dashboard optional; do not expose database/MLflow publicly for convenience.

If full cloud infrastructure is too expensive, deploy only the inference demonstration and run the complete lifecycle locally. State this limitation honestly; do not claim a fully cloud-hosted lifecycle.

Set a budget alert, request quotas, limited retention and a teardown checklist. Cold-start time is measured separately from warm inference latency.

## 15. Repository layout

```text
fraudguard/
  pyproject.toml
  uv.lock
  Dockerfile
  compose.yaml
  .env.example
  config/
    model.yaml
    monitoring.yaml
    promotion.yaml
  src/fraudguard/
    data/prepare.py
    data/splits.py
    data/manifests.py
    features/pipeline.py
    training/train.py
    training/evaluate.py
    training/gates.py
    registry/client.py
    serving/app.py
    serving/schemas.py
    persistence/models.py
    replay/stream.py
    replay/labels.py
    replay/scenarios.py
    monitoring/drift.py
    monitoring/performance.py
    jobs/retrain.py
    deployment/promote.py
    deployment/rollback.py
  migrations/
  tests/unit/
  tests/integration/
  tests/lifecycle/
  dashboard/app.py
  scripts/
  .github/workflows/
  docs/
    data_card.md
    model_card.md
    architecture.md
    evaluation.md
    runbook.md
    limitations.md
    demo.md
```

Keep data paths configurable. Use CLI modules with explicit config paths, not notebook-only training. Notebooks are optional exploration artifacts.

## 16. Build schedule and acceptance gates

Planning estimate: approximately 2–3 weeks of focused work, longer if sharing time with placements/internship. This is an estimate, not a promise.

### Phase 1: Data and baseline

Prepare features, temporal partitions, manifests, baseline, XGBoost and evaluation.

Acceptance: forbidden features absent; transformations fit only on training; positive counts published; threshold chosen without test leakage.

### Phase 2: Tracking and serving

MLflow runs/registry, full pipeline artifact, FastAPI, PostgreSQL migrations and containers.

Acceptance: fresh local setup works with documented prerequisites; predictions log exact version; duplicate requests do not duplicate rows.

### Phase 3: Replay and labels

Implement simulation clock, stable replay and private label release.

Acceptance: no labels in inference payload; labels cannot arrive before scheduled cutoff; performance reflects only eligible outcomes.

### Phase 4: Monitoring

Stable-window calibration, drift reports, label-based performance, persisted alerts and job deduplication.

Acceptance: stable traffic behavior documented; drift alert produced; unlabeled periods display unavailable performance.

### Phase 5: Candidate lifecycle

Train/evaluate candidate, gate, deploy pinned version and rollback.

Acceptance: rejection works; failure leaves champion intact; rollback works. Demonstrate promotion only if a candidate truly qualifies. Use controlled test fixtures to test branches without inventing benchmark wins.

### Phase 6: Cloud and evidence

Deploy within budget, add read-only dashboard, run smoke/load tests, capture demo and publish docs.

Acceptance: reproduce measured p50/p95 warm end-to-end latency under stated concurrency, hardware and DB logging policy; report cold starts separately.

## 17. Tests required

- Forbidden/target columns cannot enter feature pipeline.
- Split boundaries do not overlap at the step level.
- Fit uses no comparison/audit examples.
- Unknown categories and malformed numeric values behave as specified.
- Offline and API predictions match for a golden fixture.
- Duplicate requests are idempotent.
- Delayed labels respect simulated cutoff.
- Missing labels do not become negatives.
- Drift monitor handles empty and too-small windows.
- Zero positive/negative cases are marked insufficient where appropriate.
- Retraining requests deduplicate and respect locks/cooldowns.
- Failed gate never changes champion.
- Failed deployment never claims success.
- Rollback restores serving version and records alias/serving identity.
- Secrets and dataset files are excluded from Git.

## 18. Definition of done

- [ ] Public README accurately describes synthetic data and replay.
- [ ] Baseline and XGBoost evaluated with exact splits and label counts.
- [ ] Shared preprocessing is serialized and tested.
- [ ] MLflow records code SHA, data checksum, params and artifacts.
- [ ] Serving reports immutable model identity and threshold.
- [ ] Delayed labels and performance monitoring work.
- [ ] Drift alert fires on a reproducible scenario.
- [ ] Candidate training uses only eligible labels.
- [ ] Quality gate, rejection, deployment failure and rollback are tested.
- [ ] Actual promotion demonstrated only if metrics support it.
- [ ] Dashboard shows label coverage and distinguishes drift/performance.
- [ ] Local lifecycle reproducible; cloud scope clearly stated.
- [ ] Measured latency has conditions and percentiles.
- [ ] Setup, teardown and failure runbooks written.

## 19. Resume and interview evidence

Do not use the following until implemented:

FraudGuard — Drift-Monitored ML Pipeline | XGBoost, MLflow, FastAPI, PostgreSQL, Docker, Azure, Evidently

- Compared logistic regression and XGBoost on synthetic PaySim transactions using a conservative feature policy, temporal splits and validation-selected decision thresholds; reported [measured AP/recall/FPR].
- Versioned training artifacts in MLflow and deployed a Dockerized FastAPI service with immutable model identity, schema validation and prediction logging.
- Replayed historical transactions with delayed labels, monitored drift and labeled performance, and implemented validation-gated candidate updates with rejection and rollback tests.

Interview questions to answer:

1. Why exclude balance fields?
2. Why can drift exist without performance decline?
3. What labels were available at retraining time?
4. How did you avoid evaluating on training data?
5. Why choose that threshold and false-positive budget?
6. Is the output calibrated?
7. What prevents two retraining jobs racing?
8. What happens if deployment fails after registration?
9. How do you know which model served each prediction?
10. What is synthetic and what generalization claims are unsupported?

## 20. Reference documentation

- Original PaySim data card: https://www.kaggle.com/datasets/ealaxi/paysim1
- MLflow registry and aliases: https://mlflow.org/docs/latest/ml/model-registry/
- MLflow registry workflows/stage migration: https://mlflow.org/docs/latest/ml/model-registry/workflow/
- Evidently data drift: https://www.evidentlyai.com/ml-in-production/data-drift
- Evidently concept drift: https://www.evidentlyai.com/ml-in-production/concept-drift
- scikit-learn leakage guidance: https://scikit-learn.org/stable/common_pitfalls.html
- scikit-learn temporal cross-validation: https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.TimeSeriesSplit.html

Pin dependency versions after checking compatibility, and follow documentation for those exact versions. This plan deliberately does not invent benchmark results or financial business requirements.

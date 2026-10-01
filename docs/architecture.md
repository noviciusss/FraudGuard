# FraudGuard System Architecture

## Overview
FraudGuard is an end-to-end MLOps pipeline and supervised classification service built to demonstrate the full lifecycle of machine learning systems under realistic production constraints:
- **Strict feature contracts and leakage prevention**
- **Disjoint temporal step-based partitioning**
- **Fail-closed inference auditing and idempotency**
- **Delayed outcome simulation via an event-time simulation clock**
- **Independent covariate drift and labeled performance monitoring**
- **Validation-gated candidate retraining, promotion, and automated rollback**

## High-Level Data Flow

```mermaid
flowchart TD
    subgraph DataEngine[1. Data & Feature Governance]
        RawData[Raw PaySim / Sample Data] --> Prep[Data Preparation & Validation]
        Prep --> Split[Strict Temporal Step-Based Splits]
        Split --> DevTrain[Dev Train ~60%]
        Split --> DevTune[Dev Tune ~15%]
        Split --> DevReplay[Replay Window ~15%]
        Split --> Audit[Final Audit ~10%]
    end

    subgraph Pipeline[2. Preprocessing & Modeling]
        DevTrain --> FeatPipe[Scikit-learn Pipeline / ColumnTransformer]
        FeatPipe --> BaseMdl[Logistic Regression Baseline]
        FeatPipe --> XGBMdl[XGBoost Pipeline]
        DevTune --> ThreshPolicy[Validation Threshold Optimizer]
        XGBMdl --> MLflowReg[MLflow Artifact & Model Registry]
    end

    subgraph Serving[3. Serving & Persistence]
        MLflowReg --> FastAPI[FastAPI Serving Engine]
        FastAPI --> Postgres[(PostgreSQL DB)]
        Postgres --> PredLog[predictions table: immutable version logged]
    end

    subgraph ReplayModule[4. Simulation Replay & Delayed Labels]
        DevReplay --> StreamWorker[Replay Stream Worker: POST /predict]
        DevReplay --> LabelWorker[Delayed Label Worker: Simulated Step Clock]
        LabelWorker --> Outcomes[(outcomes table)]
    end

    subgraph MonitoringModule[5. Drift & Performance Engine]
        PredLog --> DriftEng[Evidently Adapter: Data & Prediction Drift]
        PredLog --> PerfEng[Performance Monitor: JOIN predictions + outcomes]
        Outcomes --> PerfEng
        DriftEng --> Alerts[(monitoring_reports / alerts)]
        PerfEng --> Alerts
    end

    subgraph LifecycleGate[6. Automated Retrain & Promotion Gating]
        Alerts --> DedupQueue[Deduplicated retrain_requests]
        DedupQueue --> RetrainWorker[Cutoff Retrain Job]
        RetrainWorker --> CandidateFit[Train Candidate on Eligible Labels]
        CandidateFit --> CandEval[Evaluate Candidate vs Champion on Joint Comparison Set]
        CandEval --> GateDecision{Validation Gate Passed?}
        GateDecision -- Yes --> DeployVerification[Readiness Verification & Rollout]
        DeployVerification --> UpdateChampion[Update champion alias / set previous]
        GateDecision -- No --> RejectCandidate[Reject & Record Decision]
        DeployVerification -- Failed --> RollbackAction[Execute Rollback]
    end
```

## Database Tables
1. `predictions`: Primary inference audit table storing timestamp, simulated event step, allowlisted features, continuous risk score, threshold, decision, and immutable model version.
2. `outcomes`: Delayed ground truth labels associated via foreign key with the prediction.
3. `monitoring_reports`: Periodic window evaluations of feature PSI, Jensen-Shannon categorical divergence, and realized precision/recall/FPR.
4. `retrain_requests`: Deduplicated queue of automated retraining workflows triggered by sustained alerts.
5. `deployment_events`: Immutable audit trail recording every promotion attempt, readiness verification status, and rollback action.

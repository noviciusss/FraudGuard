# FraudGuard Operations & Troubleshooting Runbook

## 1. Quickstart & Local Setup

### Prerequisites
- Python 3.12+
- `uv` package manager (`curl -LsSf https://astral.sh/uv/install.sh | sh` or `winget install astral-sh.uv`)
- Optional: Docker & Docker Compose

### Fast Bootstrap
```bash
# Clone and enter repo
cd d:/mlops

# Install all dependencies with uv
uv sync

# Run complete test suite (Unit, Integration, Lifecycle)
uv run pytest -v

# Run Phase 1 training to generate model artifacts
uv run python -m fraudguard.training.train --output-dir artifacts/phase1

# Launch the FastAPI serving engine
uv run uvicorn fraudguard.serving.app:app --host 0.0.0.0 --port 8000
```

### Launch Observability Dashboard
```bash
uv run streamlit run dashboard/app.py
```

### Docker Compose
```bash
docker compose up -d
```
Services available:
- FastAPI Serving: `http://localhost:8000/docs`
- Postgres Database: `localhost:5432` (db: `fraudguard`, user: `fraudguard`)
- MLflow Tracking: `http://localhost:5000`

---

## 2. API Health & Verification

```bash
# Check Liveness
curl http://localhost:8000/health/live

# Check Readiness (verifies model loaded and database ping)
curl http://localhost:8000/health/ready

# Check Loaded Model Info & Active Threshold
curl http://localhost:8000/model-info
```

---

## 3. Incident Management: Emergency Rollback Procedure

If a candidate model is promoted but exhibits performance degradation or operational anomalies:
```python
from fraudguard.persistence.database import get_session
from fraudguard.deployment.rollback import execute_rollback
from fraudguard.registry.client import ModelRegistryClient

session = get_session()
registry = ModelRegistryClient()

# Execute instant rollback to previous version
execute_rollback(
    db=session,
    failed_version="cand_cutoff_108",
    target_previous_version="v1_init",
    reason="Post-promotion false-positive spike observed",
    registry_client=registry,
)
```

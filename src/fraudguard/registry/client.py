"""MLflow Model Registry client with immutable version resolution and alias management."""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import joblib
import mlflow
from mlflow.tracking import MlflowClient

logger = logging.getLogger(__name__)

DEFAULT_EXPERIMENT_NAME = "FraudGuard"
DEFAULT_MODEL_NAME = "fraudguard-classifier"


class ModelRegistryClient:
    """Manages model tracking, registration, immutable version resolution, and aliases."""

    def __init__(
        self,
        tracking_uri: Optional[str] = None,
        model_name: str = DEFAULT_MODEL_NAME,
        local_artifacts_dir: str = "artifacts/phase1",
    ):
        self.tracking_uri = tracking_uri or os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")
        self.model_name = model_name
        self.local_artifacts_dir = Path(local_artifacts_dir)
        mlflow.set_tracking_uri(self.tracking_uri)
        self.client = MlflowClient(tracking_uri=self.tracking_uri)

    def log_and_register_pipeline(
        self,
        pipeline: Any,
        params: Dict[str, Any],
        metrics: Dict[str, Any],
        run_name: str = "training_run",
        set_champion: bool = False,
    ) -> Tuple[str, str]:
        """Logs pipeline artifact to MLflow and registers a new immutable model version."""
        mlflow.set_experiment(DEFAULT_EXPERIMENT_NAME)

        with mlflow.start_run(run_name=run_name) as run:
            run_id = run.info.run_id
            mlflow.log_params(params)
            mlflow.log_metrics(metrics)

            # Save pipeline to temporary path and log artifact
            tmp_model_path = Path("artifacts") / "model.joblib"
            tmp_model_path.parent.mkdir(parents=True, exist_ok=True)
            joblib.dump(pipeline, tmp_model_path)
            mlflow.log_artifact(str(tmp_model_path), artifact_path="model")

            # Create or register model in MLflow model registry
            artifact_uri = f"runs:/{run_id}/model"
            try:
                model_version_info = mlflow.register_model(
                    model_uri=artifact_uri,
                    name=self.model_name,
                )
                version = str(model_version_info.version)
            except Exception as e:
                logger.warning("Could not register model in MLflow registry: %s. Using run_id.", e)
                version = run_id[:8]

            if set_champion and version:
                try:
                    self.client.set_registered_model_alias(self.model_name, "champion", version)
                    logger.info("Assigned alias 'champion' -> version %s", version)
                except Exception as e:
                    logger.warning("Failed to set alias: %s", e)

            return run_id, version

    def resolve_alias_version(self, alias: str = "champion") -> Optional[str]:
        """Resolves a mutable registry alias (e.g., 'champion') to its immutable version string."""
        try:
            model_info = self.client.get_model_version_by_alias(self.model_name, alias)
            return str(model_info.version)
        except Exception as e:
            logger.debug("Failed to resolve alias '%s' via MLflow: %s", alias, e)
            return None

    def load_pipeline_by_version_or_path(
        self,
        version: Optional[str] = None,
        alias: str = "champion",
        fallback_path: Optional[str] = None,
    ) -> Tuple[Any, str]:
        """Loads the immutable pipeline artifact and returns (pipeline, immutable_version)."""
        resolved_version = version or self.resolve_alias_version(alias)

        if resolved_version:
            try:
                model_uri = f"models:/{self.model_name}/{resolved_version}"
                # If logged as pyfunc or artifact
                loaded = mlflow.sklearn.load_model(model_uri)
                logger.info("Loaded pipeline version %s from MLflow registry", resolved_version)
                return loaded, resolved_version
            except Exception as e:
                logger.warning("Could not load from MLflow URI models:/%s/%s: %s", self.model_name, resolved_version, e)

        # Fallback to local joblib artifact
        local_candidate = (
            Path(fallback_path)
            if fallback_path
            else self.local_artifacts_dir / "xgb_fraud_pipeline.joblib"
        )
        if local_candidate.is_file():
            logger.info("Loaded pipeline artifact from local path: %s", local_candidate)
            pipeline = joblib.load(local_candidate)
            return pipeline, "v1_local"

        raise FileNotFoundError(
            f"Unable to load model pipeline: neither MLflow registry nor local file {local_candidate} is available."
        )

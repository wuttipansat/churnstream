from typing import Any
from pathlib import Path

import joblib
import json

import mlflow
import mlflow.sklearn
from mlflow import MlflowClient
import pandas as pd
from mlflow.models import infer_signature
from sklearn.base import BaseEstimator
from sklearn.pipeline import Pipeline

from churnstream.core.config import get_settings
from churnstream.features.config import FeatureConfig

def register_best_model(
        model_uri: str,
) -> str:
    
    settings = get_settings()

    model_version = mlflow.register_model(
        model_uri=model_uri,
        name=settings.mlflow_registered_model_name,
    )

    MlflowClient().set_registered_model_alias(
        name=settings.mlflow_registered_model_name,
        alias=settings.mlflow_model_alias,
        version=model_version.version,
    )

    return f"models:/{settings.mlflow_registered_model_name}@{settings.mlflow_model_alias}"

def export_champion_model() -> Path:
    settings = get_settings()

    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    mlflow.set_registry_uri(settings.mlflow_tracking_uri)

    model_uri = f"models:/{settings.mlflow_registered_model_name}@{settings.mlflow_model_alias}"

    print(f"Loading model from: {model_uri}")

    model = mlflow.sklearn.load_model(model_uri)

    output_path = Path(settings.model_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    joblib.dump(model, output_path)

    client = MlflowClient()

    model_version = client.get_model_version_by_alias(
        name=settings.mlflow_registered_model_name,
        alias=settings.mlflow_model_alias,
    )

    metadata = {
        "registered_model_name": settings.mlflow_registered_model_name,
        "alias": settings.mlflow_model_alias,
        "version": str(model_version.version),
        "source_uri": model_uri,
        "run_id": model_version.run_id,
    }

    metadata_path = output_path.parent / "model_metadata.json"

    metadata_path.write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )

    print(f"Model exported to: {output_path}")
    print(f"Metadata exported to: {metadata_path}")
    print(f"Model version: {model_version.version}")

    return output_path

def configure_mlflow() -> None:
    settings = get_settings()

    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    mlflow.set_experiment(settings.mlflow_experiment_name)

def get_model_params(
        model: BaseEstimator,
) -> dict[str, Any]:
    return {
        f"model__{name}": value
        for name, value in model.get_params(
            deep=False
        ).items()
    }

def log_benchmark_run(
        *,
        model_name: str,
        model: BaseEstimator,
        metrics: dict[str, float],
        config: FeatureConfig,
        train_row: int,
        cv_folds: int,
) -> str:
    with mlflow.start_run(
        run_name=f"benchmark-{model_name}",
        tags={
            "stage": "benchmark",
            "model_name": model_name,
        },
    ) as run:
        mlflow.log_params(
            {
                **get_model_params(model),
                "train_rows": train_row,
                "cv_folds": cv_folds,
                "selection_metric": "pr_auc"
            }
        )

        mlflow.log_metrics(metrics)
        
        mlflow.log_dict(
            {
                "input_features": list(config.input_features),
                "model_numeric": list(config.model_numeric),
                "model_categorical": list(config.model_categorical),
                "target": config.target,     
            },
            "feature_config.json",
        )

    return run.info.run_id
    
def log_final_run(
        *,
        model_name: str,
        pipeline: Pipeline,
        model: BaseEstimator,
        metrics: dict[str, float],
        config: FeatureConfig,
        input_example: pd.DataFrame,
        train_rows: int,
        test_rows: int,
        benchmark_results: pd.DataFrame,
) -> tuple[str, str]:
    predictions = pipeline.predict(input_example)

    signature = infer_signature(
        input_example,
        predictions,
    )

    with mlflow.start_run(
        run_name=f"final-{model_name}",
        tags={
            "stage": "final",
            "model_name": model_name,

        },
    ) as run:
        mlflow.log_params(
            {
                **get_model_params(model),
                "train_rows": train_rows,
                "test_rows": test_rows,
            }
        )

        mlflow.log_metrics(metrics)
        
        mlflow.log_dict(
            {
                "input_features": list(config.input_features),
                "model_numeric": list(config.model_numeric),
                "model_categorical": list(config.model_categorical),
                "target": config.target,     
            },
            "feature_config.json",
        )

        mlflow.log_text(
            benchmark_results.to_csv(index=False),
            "benchmark_results.csv"
        )

        model_info = mlflow.sklearn.log_model(
            sk_model=pipeline,
            name="model",
            signature=signature,
            input_example=input_example,
            serialization_format="cloudpickle",
        )

        return run.info.run_id, model_info.model_uri
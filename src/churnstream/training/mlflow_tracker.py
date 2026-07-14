from typing import Any

import mlflow
import mlflow.sklearn
import pandas as pd
from mlflow.models import infer_signature
from sklearn.base import BaseEstimator
from sklearn.pipeline import Pipeline

from churnstream.core.config import get_settings
from churnstream.features.config import FeatureConfig

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
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from sklearn.model_selection import (
    StratifiedKFold,
    cross_validate,
    train_test_split,
)

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer

from churnstream.data.loader import load_data
from churnstream.data.validator import validate_data
from churnstream.features.config import FeatureConfig, build_feature_config
from churnstream.features.engineering import engineer_features
from churnstream.preprocess.preprocessing import build_preprocessor
from churnstream.training.mlflow_tracker import (
    configure_mlflow,
    log_benchmark_run,
    log_final_run,
    register_best_model,
)

from churnstream.training.models import RANDOM_STATE, get_models

TEST_SIZE = 0.2
CV_FOLDS = 5

def build_pipeline(
        config: FeatureConfig,
        model: BaseEstimator,
) -> Pipeline:
    return Pipeline(
        steps=[
            (
                "feature_engineering",
                FunctionTransformer(
                    engineer_features,
                    validate=False,
                ),
            ),
            (
                "preprocessing",
                build_preprocessor(config),
            ),
            (
                "model",
                model,
            ),
        ]
    )

def benchmark_models(
        X_train: pd.DataFrame,
        y_train: pd.Series,
        config: FeatureConfig,
) -> pd.DataFrame:
    cv = StratifiedKFold(
        n_splits=CV_FOLDS,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    results = []

    for model_name, model in get_models().items():
        print(f"Benchmarking: {model_name}")

        scores = cross_validate(
            estimator=build_pipeline(config, model),
            X=X_train,
            y=y_train,
            cv=cv,
            scoring={
                "roc_auc": "roc_auc",
                "pr_auc": "average_precision",
                "precision": "precision",
                "recall": "recall",
                "f1": "f1"
            },
            n_jobs=-1,
        )

        metrics = {
            "cv_roc_auc": float(scores["test_roc_auc"].mean()),
            "cv_pr_auc": float(scores["test_pr_auc"].mean()),
            "cv_precision": float(scores["test_precision"].mean()),
            "cv_recall": float(scores["test_recall"].mean()),
            "cv_fit_time": float(scores["fit_time"].mean())
        }

        log_benchmark_run(
            model_name=model_name,
            model=model,
            metrics=metrics,
            config=config,
            train_row=len(X_train),
            cv_folds=CV_FOLDS,
        )

        results.append(
            {
                "model": model_name,
                **metrics,
            }
        )

    return (
        pd.DataFrame(results)
        .sort_values("cv_pr_auc", ascending=False)
        .reset_index(drop=True)
    )

def train() -> None:
    configure_mlflow()

    dataframe = load_data()

    validation = validate_data(dataframe)

    if not validation.is_valid:
        raise ValueError(
            "Training data validation failed: "
            + "; ".join(validation.errors)
        )
    
    config = build_feature_config()

    dataframe = dataframe.dropna(
        subset=[config.target]
    )

    X = dataframe.loc[:, list(config.input_features),]
    y = dataframe[config.target].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        stratify=y,
        random_state=RANDOM_STATE
    )

    benchmark_results = benchmark_models(
        X_train,
        y_train,
        config,
    )

    print("\nBenchmark results:")
    print(benchmark_results.to_string(index=False))

    best_model_name = str(
        benchmark_results.loc[
            benchmark_results["model"] != "dummy",
            "model",
        ].iloc[0]
    )

    best_model = get_models()[best_model_name]
    pipeline = build_pipeline(config, best_model)

    pipeline.fit(X_train, y_train)

    predictions = pipeline.predict(X_test)
    probabilities = pipeline.predict_proba(X_test)[:, 1]

    test_metrics = {
        "test_accuracy": float(accuracy_score(y_test, predictions)),
        "test_roc_auc": float(roc_auc_score(y_test, probabilities)),
        "test_pr_auc": float(average_precision_score(y_test, probabilities)),
        "test_precision": float(precision_score(y_test, predictions, zero_division=0)),
        "test_recall": float(recall_score(y_test, predictions, zero_division=0)),
        "test_f1": float(f1_score(y_test, predictions, zero_division=0)),
    }

    run_id, model_uri = log_final_run(
        model_name=best_model_name,
        pipeline=pipeline,
        model=best_model,
        metrics=test_metrics,
        config=config,
        input_example=X_test.head(5),
        train_rows=len(X_train),
        test_rows=len(X_test),
        benchmark_results=benchmark_results,
    )

    register_best_model(model_uri)

    print(f"\n Best model: {best_model_name}")

    for metric, value in test_metrics.items():
        print(f"{metric}: {value:.4f}")

    print(f"\n MLflow run ID: {run_id}")
    print(f"Model URI: {model_uri}")


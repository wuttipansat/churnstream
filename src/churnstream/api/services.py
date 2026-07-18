import os
from functools import lru_cache
from pathlib import Path
import json

import joblib
import pandas as pd
from sklearn.pipeline import Pipeline

from churnstream.schemas.data_schemas import PredictionInput, PredictionOutput
from churnstream.core.config import get_settings

settings = get_settings()

MODEL_PATH = Path(settings.model_path)
MODEL_METADATA_PATH = Path(settings.model_metadata_path)

@lru_cache
def load_model() -> Pipeline:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model file was not found: {MODEL_PATH.resolve()}"
        )

    model = joblib.load(MODEL_PATH)

    if not hasattr(model, "predict"):
        raise TypeError(
            "Loaded artifact does not support predict()."
        )
    
    return model

def get_model_info() -> dict[str, str]:
    if not MODEL_METADATA_PATH.exists():
        return {
            "model": "model.pkl",
            "metadata": "not available",
        }
    
    return json.loads(MODEL_METADATA_PATH.read_text(encoding="utf-8"))

def get_prediction(
        input: PredictionInput,
) -> PredictionOutput:
    model = load_model()

    input_data = pd.DataFrame([input.model_dump()])

    prediction = int(model.predict(input_data)[0])
    probability = float(
        model.predict_proba(input_data)[0, 1]
    )

    return PredictionOutput(
        prediction=prediction,
        probability=probability,
    )
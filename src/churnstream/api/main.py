from contextlib import asynccontextmanager

from fastapi import FastAPI

from churnstream.schemas.data_schemas import PredictionInput, PredictionOutput
from churnstream.api.services import load_model, get_prediction, get_model_info

@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.model = load_model()

    yield
    
    load_model.cache_clear()

app = FastAPI(
    title="ChurnStream API",
    description="Customer churn prediction API",
    version="0.1.0",
    lifespan=lifespan
)

@app.get("/")
def root() -> dict[str, str]:
    return {
        "message": "ChurnStream API",
        "docs": "/docs",
    }

@app.get("/health")
def health() -> dict[str, str]:
    return {
        "status": "ok",
        "model": "loaded",
    }

@app.get("/model-info")
def model_info() -> dict[str, str]:
    return get_model_info()

@app.post("/predict", response_model=PredictionOutput)
def predict(input: PredictionInput) -> PredictionOutput:
    return get_prediction(input)
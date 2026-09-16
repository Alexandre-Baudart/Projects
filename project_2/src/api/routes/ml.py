from fastapi import APIRouter, Depends

import pandas as pd

from src.api.schemas import (
    PredictRequest,
    PredictResponse,
    PredictProbaResponse
)
from src.ml.libs.session import MLSession
from src.api.dependencies import get_session

router = APIRouter(
    prefix="/ml", # prefix -> GET /ml/...
    tags=["ML"]
)


@router.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest, session: MLSession = Depends(get_session)):
    X = pd.DataFrame(request.data)

    preds = session.predict(X)
    p_pos = session.predict(X, return_probs=True)
    p_pos = [float(p) for p in p_pos]

    probabilities = [
        [1 - p, p]
        for p in p_pos
    ]

    return {
        "predictions": preds.tolist(),
        "probabilities": probabilities
    }

@router.post("/predict_proba", response_model=PredictProbaResponse)
def predict_proba(request: PredictRequest, session: MLSession = Depends(get_session)):
    X = pd.DataFrame(request.data)

    p_pos = session.predict(X, return_probs=True)
    p_pos = [float(p) for p in p_pos]

    probabilities = [
        [1 - p, p]
        for p in p_pos
    ]

    return {
        "probabilities": probabilities
    }

@router.get("/health")
def health():
    return {"status": "ok"}


from typing import Any
from pydantic import BaseModel

class PredictRequest(BaseModel):
    data: list[dict[str, Any]]

class PredictResponse(BaseModel):
    predictions: list[int]
    probabilities: list[list[float]]

class PredictProbaResponse(BaseModel):
    probabilities: list[list[float]]
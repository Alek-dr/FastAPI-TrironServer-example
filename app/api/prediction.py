import logging
from typing import List

import numpy as np
from fastapi import APIRouter

from app.model.pydantic import ClassificationResult, Features
from app.prediction_models import sklearn_model, t_model, xgb_model

router = APIRouter()

log = logging.getLogger(__name__)


@router.post("/predict_torch", response_model=List[ClassificationResult])
async def predict(features: List[Features]):
    """Классификация с помощью pytorch модели"""
    x = np.array([[f.x1, f.x2] for f in features]).astype(np.float32)
    out = t_model(x)
    response = [{"label": int(label)} for label in out]
    return response


@router.post("/predict_xgboost", response_model=List[ClassificationResult])
async def predict(features: List[Features]):
    """Классификация с помощью xgboost модели"""
    x = np.array([[f.x1, f.x2] for f in features]).astype(np.float32)
    out = xgb_model(x)
    response = [{"label": int(label)} for label in out]
    return response


@router.post("/predict_sklearn", response_model=List[ClassificationResult])
async def predict(features: List[Features]):
    """Классификация с помощью sklearn модели"""
    x = np.array([[f.x1, f.x2] for f in features]).astype(np.float32)
    out = sklearn_model(x)
    response = [{"label": int(label)} for label in out]
    return response

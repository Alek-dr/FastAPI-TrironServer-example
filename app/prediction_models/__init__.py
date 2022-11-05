import os
import pickle

from .client import client
from .sklearn_model import SklearnModel
from .torch_model import TorchModel
from .xgboost_model import XGBOOSTModel

with open("app/prediction_models/data/scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

t_model = TorchModel(
    client, os.getenv("TORCH_MODEL_NAME"), os.getenv("TORCH_MODEL_VERSION"), scaler
)
xgb_model = XGBOOSTModel(
    client, os.getenv("XGBOOST_MODEL_NAME"), os.getenv("XGBOOST_MODEL_VERSION"), scaler
)
sklearn_model = SklearnModel(
    client, os.getenv("SKLEARN_MODEL_NAME"), os.getenv("SKLEARN_MODEL_VERSION"), scaler
)

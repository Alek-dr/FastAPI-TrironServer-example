import logging

import numpy as np

from app.prediction_models.model_client import ModelGRPCClient

log = logging.getLogger(__name__)


class XGBOOSTModel(ModelGRPCClient):
    def __init__(
            self,
            triton_client,
            model_name,
            model_version,
            scaler,
    ):
        super().__init__(triton_client, model_name, model_version)
        self.scaler = scaler

    def postprocess(self, data):
        return np.rint(data)

    def preprocess(self, data):
        return self.scaler.transform(data)

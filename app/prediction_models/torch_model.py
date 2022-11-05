import logging

import numpy as np

from app.prediction_models.model_client import ModelGRPCClient

log = logging.getLogger(__name__)


class TorchModel(ModelGRPCClient):
    def __init__(
        self,
        triton_client,
        model_name,
        model_version,
        scaler,
    ):
        super().__init__(triton_client, model_name, model_version)
        self.scaler = scaler

    def __call__(self, data):
        data = self.preprocess(data)
        raw_result = self.send_via_grpc(data)
        return self.postprocess(raw_result)

    def postprocess(self, data):
        data = np.rint(data).astype(np.bool)
        # у pytorch модели получились инвертированные классы
        data = np.invert(data)
        return data.astype(np.int32)

    def preprocess(self, data):
        return self.scaler.transform(data)

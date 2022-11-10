import logging
from abc import abstractmethod

import tritonclient.grpc as grpcclient

log = logging.getLogger(__name__)


class ModelGRPCClient:
    """
    Класс для общения с triton inference server по протоколу gRPC
    """

    def __init__(self, triton_client, model_name, model_version):
        self.triton_client = triton_client
        self.model_name = model_name
        self.model_version = model_version
        metadata = self.triton_client.get_model_metadata(
            model_name=model_name, model_version=model_version
        )
        config = self.triton_client.get_model_config(
            model_name=model_name, model_version=model_version
        ).config
        self.grpcclient = grpcclient

        input_metadata = metadata.inputs[0]
        input_config = config.input[0]

        self.input_config = dict(
            name=input_config.name,
            dtype=input_metadata.datatype,
            n_features=input_config.dims[0],
        )
        output_metadata = metadata.outputs[0]
        self.output_name = output_metadata.name

    @abstractmethod
    def preprocess(self, data):
        pass

    @abstractmethod
    def postprocess(self, *args, **kwargs):
        pass

    def __call__(self, data):
        data = self.preprocess(data)
        raw_result = self.send_via_grpc(data)
        return self.postprocess(raw_result)

    def send_via_grpc(self, data):
        input_shape = [len(data), self.input_config.get("n_features")]
        inputs = [
            self.grpcclient.InferInput(
                self.input_config.get("name"),
                input_shape,
                self.input_config.get("dtype"),
            )
        ]
        inputs[0].set_data_from_numpy(data)
        outputs = [self.grpcclient.InferRequestedOutput(self.output_name)]
        model_res = self.triton_client.infer(
            self.model_name,
            inputs,
            model_version=self.model_version,
            outputs=outputs,
        )
        output_array = model_res.as_numpy(self.output_name)
        return output_array

import os

import tritonclient.grpc as grpcclient
from dotenv import load_dotenv

load_dotenv()

url = os.getenv("TRITON_URL", "localhost:8001")
client = grpcclient.InferenceServerClient(url)

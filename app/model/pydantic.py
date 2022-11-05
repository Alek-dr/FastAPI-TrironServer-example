from pydantic import BaseModel


class Features(BaseModel):
    x1: float
    x2: float


class ClassificationResult(BaseModel):
    label: int

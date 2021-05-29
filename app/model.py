from typing import List, Optional
from pydantic import BaseModel, Field


class AvailableModels(BaseModel):
    models: List[str]


class RecordDataRequest(BaseModel):
    age: int
    sex: int        # 0 or 1
    cp: int         # 0, 1, 2 or 3
    trestbps: int
    chol: int
    fbs: int        # 0 or 1
    restecg: int    # 0 or 1
    thalach: int
    exang: int      # 0 or 1
    oldpeak: float
    slope: int      # 0, 1 or 2
    ca: int         # 0 1 or 2
    thal: int       # 0, 1, 2 or 3


class RecordRequest(BaseModel):

    """Record identifier number."""
    record_id: str
    """Name of model to be used for prediction."""
    model_name: str
    """Mapping of feature column names and values."""
    data: RecordDataRequest


class RecordsRequest(BaseModel):
    values: List[RecordRequest]


class RecordDataResponse(BaseModel):

    """Name of Machine Learning model responsible for prediction result."""
    model_name: str

    """Confidence score by model (%)."""
    confidence_score: float

    """Whether a patient has heart disease or not. 0 for no heart disease.
    Any positive integer (usually 1) represents presence of heart disease."""
    has_heart_disease: bool


class Message(BaseModel):
    message: str


class RecordResponse(BaseModel):
    record_id: str
    data: RecordDataResponse
    errors: Optional[List[Message]]
    warnings: Optional[List[Message]]


class RecordsResponse(BaseModel):
    values: List[RecordResponse]


class Metadata(BaseModel):
    version: str
    name: str

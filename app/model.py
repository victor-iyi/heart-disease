from typing import List, Optional
from pydantic import BaseModel


class RecordDataRequest(BaseModel):
    feature_1: str
    feature_2: str
    feature_3: str


class RecordRequest(BaseModel):
    record_id: str
    data: RecordDataRequest


class RecordsRequest(BaseModel):
    values: List[RecordRequest]


class RecordDataResponse(BaseModel):
    """Name of Machine Learning model responsible for prediction result."""
    model_name: str

    """Confidence score by model (%)."""
    confidence_score: float

    """Whether a patient has heart disease or not. 0 for no heart disease.
    Any positive integer represents presence of heart disease."""
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

# Copyright 2021 Victor I. Afolabi
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import List, Optional
from pydantic import BaseModel  # , Field


class AvailableModels(BaseModel):
    models: List[str]


class Features(BaseModel):
    age: int
    sex: int        # Literal[0, 1]  - 0 or 1
    cp: int         # Literal[0, 1, 2, 3]  - 0, 1, 2 or 3
    trestbps: int
    chol: int
    fbs: int        # Literal[0, 1]  - 0 or 1
    restecg: int    # Literal[0, 1]  - 0 or 1
    thalach: int
    exang: int      # Literal[0, 1]  - 0 or 1
    oldpeak: float
    slope: int      # Literal[0, 1, 2]  - 0, 1 or 2
    ca: int         # Literal[0, 1, 2]  - 0 1 or 2
    thal: int       # Literal[0, 1, 2, 3]  - 0, 1, 2 or 3


class PredictionRequest(BaseModel):
    """Request model for a single prediction."""

    """Name of model to be used for prediction."""
    model_name: str

    """Mapping of feature column names and values."""
    data: Features


class RecordRequest(BaseModel):
    """Request model for single record."""

    """Record identifier number."""
    record_id: str

    """Mapping of feature column names and values."""
    data: Features


class BatchPredictionRequest(BaseModel):
    """Request model for batch prediction."""

    """Name of mdoel to be used for batch prediction."""
    model_name: str

    """List of multiple requests."""
    values: List[RecordRequest]


class PredictionResponse(BaseModel):

    """Name of Machine Learning model responsible for prediction result."""
    model_name: str

    """Confidence score by model (%)."""
    confidence_score: Optional[float]

    """Whether a patient has heart disease or not. 0 for no heart disease.
    Any positive integer (usually 1) represents presence of heart disease."""
    has_heart_disease: bool


class Message(BaseModel):
    message: str


class RecordResponse(BaseModel):
    record_id: str
    data: PredictionResponse
    errors: Optional[List[Message]]
    warnings: Optional[List[Message]]


class BatchResponse(BaseModel):
    values: List[RecordResponse]


class Metadata(BaseModel):
    name: str
    version: str
    author: str
    license: str

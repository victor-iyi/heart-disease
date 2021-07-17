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
from pydantic import BaseModel, Field


class AvailableModels(BaseModel):
    """List of trained models available to make necessary predictions."""

    models: List[str]


class Features(BaseModel):
    """Features to the model (excluding target)."""

    age: int = Field(
        ..., gt=0, title='Age',
        description='Age of the patient in years'
    )
    sex: int = Field(
        ..., gt=0, lt=1,
        title='Sex',
        description='Male/Female. 0 for Male 1 for Female',
    )
    cp: int = Field(
        ..., gt=0, lt=3,
        title='Chest Pain Type',
        description='Typical Angina, Atypical Angina, Non-Anginal, Asymptomatic'
    )
    trestbps: int = Field(
        ..., gt=0,
        title='Resting Blood Pressure',
        description='Resting blood pressure (in mm Hg on admission to the hospital'
    )
    chol: int = Field(
        ..., gt=0,
        title='Cholesterol',
        description='Serum Cholesterol in mg/dl'
    )
    fbs: int = Field(
        ..., gt=0, lt=1,
        title='Fasting blood sugar',
        description='If fasting blood sugar > 120 mg/dl'
    )
    restecg: int = Field(
        ..., gt=0, lt=2,
        title='Resting electrocardiographic results',
        description='Values: [normal, stt abnormality, Iv hypertrophy]'
    )
    thalach: int = Field(
        ..., gt=0,
        title='Maximum heart rate',
        description='Maximum heart rate achieved'
    )
    exang: int = Field(
        ..., gt=0, lt=1,
        title='Exercise-induced angina',
        description='Exercise-induced anagina (True/False)',
    )
    oldpeak: float = Field(
        ..., gt=0,
        title='Old peak',
        description='ST depression induced by exercise relative to rest'
    )
    slope: int = Field(
        ..., gt=0, lt=2,
        title='Slope',
        description='The slope of the peak exercise ST segment'
    )
    ca: int = Field(
        ..., gt=0, lt=4,
        title='Major vessels',
        description='Number of major vessels (0-3) colored by fluoroscopy'
    )
    thal: int = Field(
        ..., gt=0, lt=3,
        title='Normal; fixed defect; reversible defect'
    )

    class Config:
        orm_mode = True


class PredictionRequest(BaseModel):
    """Request model for a single prediction."""

    """Name of model to be used for prediction."""
    model_name: Optional[str] = None

    """Mapping of feature column names and values."""
    data: Features

    class Config:
        orm_mode = True


class RecordRequest(BaseModel):
    """Request model for single record."""

    """Record identifier number."""
    id: int

    """Mapping of feature column names and values."""
    data: Features


class PredictionResponse(BaseModel):

    """Name of Machine Learning model responsible for prediction result."""
    model_name: str

    """Confidence score by model (%)."""
    confidence_score: Optional[float]

    """Whether a patient has heart disease or not. 0 for no heart disease.
    Any positive integer (usually 1) represents presence of heart disease."""
    has_heart_disease: bool

    class Config:
        orm_mode = True


class Message(BaseModel):

    """Error/warning messages."""
    message: str


class RecordResponse(BaseModel):

    """If everything went well, here's the response data."""
    data: PredictionResponse

    """Error(s) occurred"""
    errors: Optional[List[Message]]

    """Any warning(s) information/log"""
    warnings: Optional[List[Message]]


class Metadata(BaseModel):
    """Information about the models"""

    name: str = Field(
        'heart-disease',
        title='Name',
    )
    version: str = Field(
        'v1',
        title='Version',
    )
    author: str = Field(
        'Victor I. Afolabi',
        title='Author',
    )
    license: str = Field(
        'MIT or Apache',
        title='License',
    )

    class Config:
        orm_mode = True

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

    models: List[str] = Field(
        ...,
        title='List of trained models.',
        description='List of trained models available to make predictions.'
    )


class Features(BaseModel):
    """Features to the model (excluding target)."""

    age: int = Field(
        ..., ge=0, title='Age',
        description='Age of the patient in years'
    )
    sex: int = Field(
        ..., ge=0, le=1,
        title='Sex',
        description='Male/Female. 0 for Male 1 for Female',
    )
    cp: int = Field(
        ..., ge=0, le=3,
        title='Chest Pain Type',
        description='Typical Angina, Atypical Angina, Non-Anginal, Asymptomatic'
    )
    trestbps: int = Field(
        ..., ge=0,
        title='Resting Blood Pressure',
        description='Resting blood pressure (in mm Hg on admission to the hospital'
    )
    chol: int = Field(
        ..., ge=0,
        title='Cholesterol',
        description='Serum Cholesterol in mg/dl'
    )
    fbs: int = Field(
        ..., ge=0, le=1,
        title='Fasting blood sugar',
        description='If fasting blood sugar > 120 mg/dl'
    )
    restecg: int = Field(
        ..., ge=0, le=2,
        title='Resting electrocardiographic results',
        description='Values: [normal, stt abnormality, Iv hypertrophy]'
    )
    thalach: int = Field(
        ..., ge=0,
        title='Maximum heart rate',
        description='Maximum heart rate achieved'
    )
    exang: int = Field(
        ..., ge=0, le=1,
        title='Exercise-induced angina',
        description='Exercise-induced anagina (True/False)',
    )
    oldpeak: float = Field(
        ..., ge=0,
        title='Old peak',
        description='ST depression induced by exercise relative to rest'
    )
    slope: int = Field(
        ..., ge=0, le=2,
        title='Slope',
        description='The slope of the peak exercise ST segment'
    )
    ca: int = Field(
        ..., ge=0, le=4,
        title='Major vessels',
        description='Number of major vessels (0-3) colored by fluoroscopy'
    )
    thal: int = Field(
        ..., ge=0, le=3,
        title='Normal; fixed defect; reversible defect'
    )

    # target: Optional[int] = Field(
    #     None, ge=0, le=1,
    #     title='Target/Label. Has heart disease or not.',
    #     description='0 for no heart disease & 1 for heart disease.'
    # )

    class Config:
        orm_mode = True


class PredictionRequest(BaseModel):
    """Request model for a single prediction."""

    data: Features = Field(
        ...,
        title='List of features',
        description='Mapping of feature column names and values.'
    )

    model_name: Optional[str] = Field(
        None,
        title='Model name',
        description='Name of model to be used for prediction.'
    )

    class Config:
        orm_mode = True


class PredictionResponse(BaseModel):

    model_name: str = Field(
        ...,
        title='Model name',
        description='Name of ML model responsible for prediction result.',
    )

    has_heart_disease: bool = Field(
        ...,
        title='Has heart disease',
        description='Whether a patient has heart disease or not. \
            0 for no heart disease, 1 for presence of heart disease',
    )

    confidence_score: Optional[float] = Field(
        None,
        title='Confidence score',
        description='Confidence score by model (%)'
    )

    class Config:
        orm_mode = True


class Message(BaseModel):

    message: str = Field(
        ...,
        title='Error/warning messages'
    )


class Response(BaseModel):

    data: PredictionResponse = Field(
        ...,
        title='Prediction response',
        description='If everything went well, here\'s the response data',
    )

    errors: Optional[List[Message]] = Field(
        None,
        title='Error(s) occurred'
    )

    warnings: Optional[List[Message]] = Field(
        None,
        title='Warning(s) occurred',
        description='Any warnings(s) information/log'
    )


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

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

import os
from typing import Dict, List

import srsly

from fastapi import Body, APIRouter
from dotenv import find_dotenv, load_dotenv

from app.backend.inference import SavedModel
from app.schemas.model import AvailableModels, Metadata
from app.schemas.model import BatchPredictionRequest, BatchResponse
from app.schemas.model import PredictionRequest, PredictionResponse
from heart_disease.config.consts import FS


router = APIRouter()

# Local .env or env files.
load_dotenv(find_dotenv())

# Request example.
single_example = srsly.read_json('app/data/single_request_sample.json')
batch_example = srsly.read_json('app/data/batch_request_sample.json')

# Path to `saved_model.pb`.
MODEL_DIR: str = os.getenv('MODEL_DIR', FS.SAVED_MODELS)

# Loaded saved model object.
saved_model = SavedModel(model_dir=MODEL_DIR)

# Request example.
single_example = srsly.read_json('app/sample/single_request_sample.json')
batch_example = srsly.read_json('app/sample/batch_request_sample.json')


@router.get(
    '/models',
    response_model=AvailableModels,
    response_description='List of available models',
    summary='Return available models',
    tags=['heart-disease']
)
async def available_models() -> List[str]:
    """Returns the list of models that are supported by API."""

    return saved_model.list_available_models()


@router.post(
    '/predict',
    response_model=PredictionResponse,
    response_model_exclude=False,
    response_description='Presence of heart disease or not',
    summary='Make prediction',
    tags=['heart-disease']
)
async def predict(
        body: PredictionRequest = Body(..., example=single_example)
) -> Dict[str, str]:
    """Make a prediction given a model name and list of features.

    - **record_id**: Unique identifier for each set of records to be
        predicted.
    - **model_name**: Name of model. Must be one of [Suport Vector Machine,
        Decision Tree, Naive Bayes and K-Nearest Neighbors].
    - **data**: List of features are:
        age, sex, cp, trestbps, chol, fbs, restecg,
        thalach, exang, oldpeak, slope, ca, thal
    """
    res = {}
    # Extract data in correct order.
    data_dict = body.data.dict()
    print(data_dict)

    # Load saved model.
    # Get the model via body.model_name

    # Make prediction.
    return res


@router.post(
    '/batch-predict',
    response_model=BatchResponse,
    response_model_exclude=False,
    response_description='Presence of heart disease or not',
    summary='Make batch prediction',
    tags=['heart-disease']
)
async def batch_predict(
        body: BatchPredictionRequest = Body(..., example=batch_example)
) -> List[Dict[str, str]]:
    """Perform a batch prediction over mutliple patients with a given
        model name and features for each patients.

    - **values**: List of `RecordRequest`.

    `RecordRequest` has the following info:

    - **record_id**: Unique identifier for each set of records to be
        predicted.
    - **model_name**: Name of model. Must be one of [Suport Vector Machine,
        Decision Tree, Naive Bayes and K-Nearest Neighbors].
    - **data**: List of features are:
        age, sex, cp, trestbps, chol, fbs, restecg,
        thalach, exang, oldpeak, slope, ca, thal
    """

    for req in body.values:
        [f for _, f in req.data.dict()]

    return []


@router.get('/metadata', response_model=Metadata)
async def metadata() -> Dict[str, str]:
    """Returns important metadata about current API."""

    return {
        'name': 'heart_disease',
        'version': '1.0.0',
        'author': 'Victor I. Afaolbi',
        'license': 'MIT or Apache',
    }

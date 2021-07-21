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

import json
from typing import Any, Dict, List

from fastapi import APIRouter, Body

from app.backend.inference import SavedModel
from app.schemas import model


# HTTP: /predict
router = APIRouter(
    prefix='/predict',
    tags=['models', 'predict'],
)

# Request sample.
request_sample: Dict[str, str] = json.load(
    open('app/sample/predict_heart_disease.json')
)


@router.get(
    '/',
    response_model=List[model.PredictionResponse],
    tags=['predict'],
)
async def predict_heart_disease(
    body: model.PredictionRequest = Body(..., example=request_sample),
) -> List[Dict[str, Any]]:
    """Make prediction given features.

    Note:
        Uses model with the best accuracy. To use a specific model; see
        endpoint `GET: /predict/{model_name}`.

    Args:
        body (PredictionRequest, optional): Prediction body.
            Defaults to `Body(..., example=request_sample)`
    """
    # Make predictions with all models.
    saved_model = SavedModel()
    data = SavedModel.data_to_array(body.data)
    results = saved_model.predict_all(data)

    return results


@router.get(
    '/{model_name}',
    response_model=model.PredictionResponse,
    tags=['predict'],
)
async def predict_with_model(
    model_name: str,
    body: model.PredictionRequest = Body(..., example=request_sample)
) -> Dict[str, Any]:
    """Use a `model_name` to make prediction given model features.

    Args:
        model_name (str): Model name. See `heart_disease.models.MODELS`
        body (PredictionRequest, optional): Prediction body.
            Defaults to Body(..., example=request_sample).
    """
    # Make a prediction with a given model name.
    saved_model = SavedModel()

    data = SavedModel.data_to_array(body.data)
    result = saved_model.predict(data, name=model_name)

    return result

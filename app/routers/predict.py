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

from typing import Any, Dict, List, overload, Union

from fastapi import APIRouter, Body
from pydantic.errors import EnumMemberError
import srsly

from app.backend.inference import SavedModel
from app.schemas import model


# HTTP: /predict
router = APIRouter(
    prefix='/predict',
    tags=['models', 'predict'],
)

# Request example.
request_sample: Dict[str, Any] = srsly.read_json(
    'app/sample/predict_heart_disease.json'
)


@router.post(
    '/',
    response_model=Union[model.PredictionResponse,
                         model.PredictionResponse],
    tags=['predict'],
)
async def predict_heart_disease(
    body: model.PredictionRequest = Body(
        ..., embed=True, example=request_sample
    ),
) -> Union[Dict[str, Any], List[Dict[str, Any]]]:
    """Make prediction given features.

    Args:
        body (model.PredictionRequest, optional): Prediction body.
            Defaults to Body( ..., embed=True, example=request_sample ).

    Returns:
        Union[Dict[str, Any], List[Dict[str, Any]]]:
            Returns a result for each available model or
            a single response if `model_name` is specified.
    """
    # Make predictions with all models.
    saved_model = SavedModel()
    data = SavedModel.data_to_array(body.data)

    if body.model_name:
        # Return a response for given `model_name`.
        result = saved_model.predict(data, name=body.model_name)
    else:
        # Return a response for each available model.
        result = saved_model.predict_all(data)

    return result


@router.post(
    '/{model_name}',
    response_model=model.PredictionResponse,
    tags=['predict'],
)
async def predict_with_model(
    model_name: str,
    body: model.PredictionRequest = Body(
        ..., embed=True, example=request_sample
    )
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

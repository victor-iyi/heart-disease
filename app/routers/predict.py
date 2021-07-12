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

from app.schemas.model import PredictionRequest
import srsly

from fastapi import APIRouter, Body, Depends

from app.api import get_db


router = APIRouter(
    prefix='predict',
    tags=['model', 'predict'],
    dependencies=[Depends(get_db)]
)

# Request sample.
request_sample = srsly.read_json('app/sample/single_request_sample.json')


@router.get('/')
async def predict_heart_disease(
    body: PredictionRequest = Body(..., example=request_sample)
) -> None:
    """Make prediction given features.

    Note:
        Uses model with the best accuracy. To use a specific model; see
        endpoint `GET: predict/{model_name}`.

    Args:
        body (PredictionRequest, optional): Prediction body.
            Defaults to `Body(..., example=request_sample)`
    """
    pass


@router.get('/{model_name}')
async def predict_with_model(
    model_name: str,
    body: PredictionRequest = Body(..., example=request_sample)
) -> None:
    """Use a `model_name` to make prediction given model features.

    Args:
        model_name (str): Model name. See `heart_disease.models.MODELS`
        body (PredictionRequest, optional): Prediction body.
            Defaults to Body(..., example=request_sample).
    """
    pass


@router.post('/')
async def add_data():
    """Add prediction data to the Database."""
    pass

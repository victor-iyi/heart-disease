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
import pytest
from httpx import AsyncClient

from app.api import app
from app.backend.inference import SavedModel


@pytest.mark.asyncio
async def test_predict_heart_disease() -> None:
    """Test out a simple API request for prediction."""
    request_data = json.load('app/sample/predict_heart_disease.json')

    async with AsyncClient(app=app) as client:
        response = await client.get(
            '/predict', json=request_data
        )
    assert response.status_code == 200

    record = response.json()
    assert record['errors'] is None
    assert record['warnings'] is None

    assert record['data']['model_name'] == 'Decision Tree'
    assert record['data']['confidence_score'] == 90.2
    assert record['data']['has_heart_disease'] is True


@pytest.mark.asyncio
async def test_predict_with_model() -> None:
    """Test model prediction given a model name."""

    saved_model = SavedModel()
    request_data = json.load('app/sample/predict-heart-disease.json')
    model_name = saved_model.get_best_model().name

    async with AsyncClient(app=app) as client:
        response = await client.get(
            '/predict/{model_name}',
            json=request_data
        )
    assert response.status_code == 200

    record = response.json()
    assert record['errors'] is None
    assert record['warnings'] is None

    assert record['data']['model_name'] == model_name

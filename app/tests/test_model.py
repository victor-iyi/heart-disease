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

import pytest
from httpx import AsyncClient

from app.api import app


@pytest.mark.asyncio
async def test_available_model() -> None:
    """Test available models."""
    async with AsyncClient(app=app) as client:
        response = await client.get('/models')

    response.status_code == 200

    record = response.json()
    assert len(record) == 4

    assert 'Decision Tree' in record
    assert 'Naive Bayes' in record
    assert 'Support Vector Machine' in record
    assert 'K-Nearest Neighbors' in record


@pytest.mark.asyncio
async def test_metadata() -> None:
    """Test models metadata."""
    async with AsyncClient(app=app) as client:
        response = await client.get('/models/metadata')

    response.status_code == 200

    record = response.json()
    assert record['name'] == 'heart-disease'
    assert record['version'] == 'v1'
    assert record['license'] == 'MIT or Apache'

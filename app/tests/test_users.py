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
async def test_read_user() -> None:
    """Test loading a single user."""
    async with AsyncClient(app=app) as client:
        user_id = 1
        response = await client.get(f'/users/{user_id}')

    assert response.status_code == 200


@pytest.mark.asyncio
async def test_read_patient() -> None:
    """Test loading a single patient."""
    patient_id = 2

    async with AsyncClient(app=app) as client:
        response = await client.get(f'/users/{patient_id}')

    assert response.status_code == 200


@pytest.mark.asyncio
async def test_read_practitioner() -> None:
    """Test loading a single practitioner."""
    practitioner_id = 3

    async with AsyncClient(app=app) as client:
        response = await client.get(f'/users/{practitioner_id}')

    assert response.status_code == 200


@pytest.mark.asyncio
async def test_create_user() -> None:
    """Test registering a user."""
    user = {
        'email': 'john@doe.com',
        'password': 'Pa55w0rd',
        'first_name': 'John',
        'last_name': 'Doe',
    }

    async with AsyncClient(app=app) as client:
        response = await client.post('/users', data=user)

    assert response.status_code == 200


@pytest.mark.asyncio
async def test_create_patient() -> None:
    """Test registering a patient."""
    patient = {
        'email': 'jane@doe.com',
        'password': 'Pa55w0rd',
        'first_name': 'Jane',
        'last_name': 'Doe',
        'category': 'Patient',
    }

    async with AsyncClient(app=app) as client:
        response = await client.post('/users/patient', data=patient)

    assert response.status_code == 200


@pytest.mark.asyncio
async def test_create_practitioner() -> None:
    """Test registering a medical practitioner."""
    practitioner = {
        'email': 'john@doe.com',
        'password': 'Pa55w0rd',
        'first_name': 'John',
        'last_name': 'Doe',
        'category': 'Medical Practitioner',
    }

    async with AsyncClient(app=app) as client:
        response = await client.post('/users/practitioner', data=practitioner)

    assert response.status_code == 200

    async with AsyncClient(app=app) as client:
        response = await client.post('/users/practitioner', data=practitioner)

    assert response.status_code == 200

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

from fastapi import APIRouter, Depends, HTTPException

from app.api import get_db
from app.schemas import users


router = APIRouter(
    prefix='users',
    tags=['users'],
    dependencies=[Depends(get_db)],
    responses={404: 'User cannot be found!'}
)


# @router.get('/')
# async def read_patients() -> None:
#     pass


@router.get(
    '/{patient_id}',
    response_model=users.Patient
)
async def read_patient(patient_id: int) -> None:
    pass


@router.post('/patient', response_model=users.Patient)
def create_patient(patient: users.Patient) -> None:
    pass


@router.post('/', response_model=users.UserRegister)
async def create_user(user: users.UserRegister) -> None:
    pass


@router.get(
    '/{practitioner_id}',
    response_model=users.Practitioner,
    responses={403: 'Operation Forbidden.'}
)
async def read_practitioner(
    practitioner_id: int,
) -> None:
    pass


@router.post(
    '/practitioner',
    response_model=users.Practitioner,
    responses={403: 'Operation Forbidden.'}
)
def create_practitioner(practitioner_id: int) -> None:
    pass

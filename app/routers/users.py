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
from sqlalchemy.orm.session import Session

from app.api import get_db
from app.database.query import User, Patient, Practitioner
from app.schemas import users


router = APIRouter(
    prefix='users',
    dependencies=[Depends(get_db)],
    responses={
        400: 'User cannot be found!'
    },
    tags=['users', 'patient', 'practitioner'],
)


@router.get(
    '/{user_id}',
    response_model=users.User,
    responses={400: 'User cannot be found!'},
    tags=['users'],
)
async def read_user(
    user_id: int,
    db: Session = Depends(get_db)
) -> users.User:
    """Get user by `user_id`.

    Args:
        user_id (int): User id.
        db (Session, optional): Database session. Defaults to Depends(get_db).

    Returns:
        users.User: User schema.
    """
    # Get user by id.
    user = await User.get_user(db, user_id)
    return user


@router.get(
    '/{patient_id}',
    response_model=users.Patient,
    responses={400: 'Patient cannot be found!'},
    tags=['patient'],
)
async def read_patient(
    patient_id: int, db: Session = Depends(get_db)
) -> users.Patient:
    """Get patient by `patient_id`.

    Args:
        patient_id (int): Patient id.
        db (Session, optional): Database session. Defaults to Depends(get_db).

    Returns:
        users.Patient: Patient schema.
    """
    # Get patient by id.
    return Patient.get_patient(db, patient_id)


@router.get(
    '/{practitioner_id}',
    response_model=users.Practitioner,
    responses={
        400: 'Practitioner cannot be found.',
        403: 'Operation Forbidden!'
    },
    tags=['practitioner'],
)
async def read_practitioner(
    practitioner_id: int,
    db: Session = Depends(get_db)
) -> users.Practitioner:
    """Get practitioner by `practitioner_id`.

    Args:
        practitioner_id (int): Practitioner id.
        db (Session, optional): Database session. Defaults to Depends(get_db).

    Returns:
        users.Practitioner: Practitioner schema.
    """
    # Get practitioner by id.
    practitioner = await Practitioner.get_practitioner(db, practitioner_id)
    return practitioner


@router.post(
    '/',
    response_model=users.User,
    tags=['users'],
)
async def create_user(
    user: users.User, db: Session = Depends(get_db)
) -> users.User:
    """Create a new (unique) user.

    Args:
        user (users.User): User registration info.
        db (Session, optional): Database session. Defaults to Depends(get_db).

    Raises:
        HTTPException: 400 - User already exist.

    Returns:
        users.User: Created user info.
    """
    # Get user by email.
    db_user = await User.get_user_by_email(db, user.email)

    # User already exist.
    if db_user:
        raise HTTPException(status_code=400, detail='User already exists.')

    # Create new user.
    return User.add_user(db, user)


@router.post(
    '/patient',
    response_model=users.Patient,
    tags=['patient'],
)
async def create_patient(
    patient: users.Patient, db: Session = Depends(get_db)
) -> users.Patient:
    """Create a new (unique) patient.

    Args:
        patient (users.Patient): Patient registration details.
        db (Session, optional): Database session. Defaults to Depends(get_db).

    Raises:
        HTTPException: 400 - Patient already exist.

    Returns:
        users.Patient: Created patient info.
    """
    # Get patient by email.
    db_patient = await Patient.get_user_by_email(db, patient.email)

    # Patient already exist.
    if db_patient:
        raise HTTPException(status_code=400, detail='Patient already exists.')

    # Create new patient.
    return Patient.add_patient(db, patient)


@router.post(
    '/practitioner',
    response_model=users.Practitioner,
    tags=['practitioner'],
    responses={
        400: 'Practitioner already exists.',
        403: 'Operation Forbidden!',
    },
)
async def create_practitioner(
    practitioner: users.Practitioner, db: Session = Depends(get_db)
) -> users.Practitioner:
    """Create a new (unique) practitioner.

    Args:
        practitioner (users.Practitioner): Practitioner registration details.
        db (Session, optional): Database session. Defaults to Depends(get_db).

    Raises:
        HTTPException: 400 - Practitioner already exists.
                       403 - Operation Forbidden!


    Returns:
        users.Practitioner: Created practitioner info.
    """
    # Get practitioner by email.
    db_practitioner = await Practitioner.get_user_by_email(db, practitioner.email)

    # Practitioner already exist.
    if db_practitioner:
        raise HTTPException(
            status_code=400, detail='Practitioner already exists.'
        )

    # Create a new practitioner.
    return Practitioner.add_practitioner(db, practitioner)

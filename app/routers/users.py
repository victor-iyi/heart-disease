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

from app.database.query import Patient, User
from app.dependencies import get_db
from app.schemas import users


router = APIRouter(
    prefix='/users',
    dependencies=[Depends(get_db)],
    tags=['users', 'patient', 'practitioner'],
)


@router.get(
    '/{user_id}',
    response_model=users.UserInfo,
    tags=['users', 'patient', 'practitioner'],
)
async def read_user(
    user_id: int,
    db: Session = Depends(get_db)
) -> users.UserInfo:
    """Get user by `user_id`.

    Args:
        user_id (int): User id.
        db (Session, optional): Database session. Defaults to Depends(get_db).

    Returns:
        users.User: User schema.
    """
    # Get user by id.
    return User.get_user(db, user_id)


@router.get(
    '/{patient_id}',
    response_model=users.PatientInfo,
    tags=['patient'],
)
async def read_patient(
    patient_id: int, db: Session = Depends(get_db)
) -> users.PatientInfo:
    """Get patient by `patient_id`.

    Args:
        patient_id (int): Patient id.
        db (Session, optional): Database session. Defaults to Depends(get_db).

    Returns:
        users.PatientInfo: Patient schema.
    """
    # Get patient by id.
    return Patient.get_patient(db, patient_id)


@router.post(
    '/',
    response_model=users.User,
    tags=['users', 'patient', 'practitioner'],
)
async def register_user(
    user: users.UserInfo, db: Session = Depends(get_db)
) -> users.UserInfo:
    """Create a new (unique) user.

    Args:
        user (users.User): User registration info.
        db (Session, optional): Database session. Defaults to Depends(get_db).

    Raises:
        HTTPException: 400 - User already exist.

    Returns:
        users.UserInfo: Created user info.
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
    response_model=users.PatientInfo,
    tags=['patient'],
)
async def add_patient_info(
    patient: users.PatientInfo, db: Session = Depends(get_db)
) -> users.PatientInfo:
    """Add patient info to a given user.

    Args:
        patient (users.PatientInfo): Patient registration details.
        db (Session, optional): Database session. Defaults to Depends(get_db).

    Raises:
        HTTPException: 400 - Patient already exist.

    Returns:
        users.PatientInfo: Created patient info.
    """
    # Get patient by email.
    db_patient = await Patient.get_user_by_email(db, patient.email)

    # Patient already exist.
    if db_patient:
        raise HTTPException(status_code=400, detail='Patient already exists.')

    # Create new patient.
    return Patient.add_patient(db, patient)

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

from datetime import datetime
from enum import Enum
from typing import  Optional

from pydantic import BaseModel, EmailStr, Field

class User(BaseModel):
    email: EmailStr = Field(
        ...,
        title='User\'s email address',
        description='Email must be a validated email'
    )
    password: str = Field(
        ...,
        title='User raw password',
        description='Password must be more than 4 characters but less than 20',
        min_length=4,
        max_length=20,
    )

    class Config:
        orm_mode = True


class Category(str, Enum):
    """User categories: Patient & Medical Practitioner"""

    patient: str = 'patient'
    practitioner: str = 'practitioner'


class UserInfo(User):
    """User registration details."""

    first_name: Optional[str] = Field(
        None,
        title='First name',
        description='First name must be less than 32 characters.',
        max_length=32,
    )
    last_name: Optional[str] = Field(
        None,
        title='Last name',
        description='Last name must be less than 32 characters.',
        max_length=32,
    )
    category: Optional[Category] = Field(
        None,
        title='Category',
        description='Category must be either a Patient or a Medical Practitioner',
    )

    class Config:
        orm_mode = True


class PatientInfo(User):
    """Patient registration details."""

    age: Optional[int] = Field(
        None,
        title='Patient\'s age',
        description='Patient must be 120 years or younger.',
        lt=120,
        gt=0,
    )
    last_name: Optional[str] = Field(
        None,
        title='Patient\'s last name',
        description='Last name must be less than 32 characters.',
        max_length=32,
    )
    contact: Optional[str] = Field(
        None,
        title='Personal contact',
        description='Phone number could contain country area code e.g +1',
        min_len=6,
        max_length=15
    )
    history: Optional[str] = Field(
        None,
        title='History of present illness',
        description='Full details of patient\'s medical history should be provided.',
    )
    aliment: Optional[str] = Field (
        None,
        title='Underlying medical aliment',
    )
    last_visit_diagnosis: Optional[datetime] = Field(
        None,
        title='Last visit date for diagnosis',
        description='Last diagnosis visit.',
    )
    guardian_fullname: Optional[str] = Field(
        None,
        title='Guardian\'s fullname',
        description='Must be less than 64 characters in total',
        max_length=128,
    )
    guardian_email: Optional[EmailStr] = Field(
        None,
        title='Guardian\'s email address',
    )
    guardian_phone: Optional[str] = Field(
        None,
        title='Guardian\'s phone number',
        description='Phone number could contain country\'s area code e.g +1 or +234',
        min_length=6,
        max_length=15,
    )
    occurences_of_illness: Optional[str] = Field(
        None,
        title='Recent occurrence of illness',
        description='Full description of recent occurence of illness'
    )
    last_treatment: Optional[datetime] = Field(
        None,
        title='Last treatment given',
        description='Last treatment.',
    )

    class Config:
        orm_mode = True

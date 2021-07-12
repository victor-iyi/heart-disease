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

from sqlalchemy import Column, DateTime, Enum, ForeignKey
from sqlalchemy import Integer, Numeric, String, Text
from sqlalchemy.orm import relationship

from app.database import Base


class Category(Enum):
    patient = 'Patient'
    practitioner = 'Medical Practitioner'


class User(Base):
    __tablename__ = 'users'

    id = Column(Integer, primary_key=True, index=True)
    email = Column(String, unique=True, index=True)
    hashed_password = Column(String)
    category = Column(Enum(Category))
    # category = Column(String(10))

    __mapper_args__ = {
        'polymorphic_identity': 'user',
        'polymorphic_on': category,
    }


class Patient(User):
    __tablename__ = 'patient'

    id = Column(Integer, ForeignKey('user.id'), primary_key=True, index=True)
    age = Column(Integer)
    last_name = Column(String(32), index=True)
    contact = Column(String(15), index=True)
    history = Column(Text)
    aliment = Column(Text)
    last_visit_diagnosis = Column(DateTime, server_default='NOW()')
    guardian_fullname = Column(String(64))
    guardian_email = Column(String)
    guardian_phone = Column(String(15))
    occurences_of_illness = Column(Text)
    last_treatment = Column(DateTime, server_default='NOW()')

    user_id = Column(Integer, ForeignKey('user.id'))
    relationship(User, foreign_keys=user_id)

    __mapper_args__ = {
        'polymorphic_identity': 'patient',
        'inherit_condition': id == User.id
    }


class Practitioner(User):
    __tablename__ = 'practitioner'

    id = Column(Integer, ForeignKey('user.id'), primary_key=True, index=True)

    user_id = Column(Integer, ForeignKey('user.id'))
    relationship(User, foreign_keys=user_id)

    __mapper_args__ = {
        'polymorphic_identity': 'practitioner',
        'inherit_condition': id == User.id,
    }


class Feature(Base):
    __tablename__ = 'features'
    # Primary key.
    id = Column(Integer, primary_key=True, index=True)

    # Features.
    age = Column(Integer, nullable=False)
    sex = Column(Integer, nullable=False)
    cp = Column(Integer, nullable=False)
    trestbps = Column(Integer, nullable=False)
    chol = Column(Integer, nullable=False)
    fbs = Column(Integer, nullable=False)
    restecg = Column(Integer, nullable=False)
    thalach = Column(Integer, nullable=False)
    exang = Column(Integer, nullable=False)
    oldpeak = Column(Numeric, nullable=False)
    slope = Column(Integer, nullable=False)
    ca = Column(Integer, nullable=False)
    thal = Column(Integer, nullable=False)

    # Target.
    target = Column(Integer, nullable=True)


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

from passlib.context import CryptContext
from sqlalchemy import Column, DateTime, Enum
from sqlalchemy import Integer, Numeric, String, Text

from app.database import Base


class Category(Enum):
    patient = 'Patient'
    practitioner = 'Medical Practitioner'


class User(Base):
    __tablename__ = 'user'

    # User ID column.
    id = Column(Integer, primary_key=True, index=True)

    email = Column(String, unique=True, index=True)
    password_hash = Column(String(64), nullable=False)

    first_name = Column(String(32), index=True)
    last_name = Column(String(32), index=True)

    category = Column(Category, index=True,
                      nullable=False,
                      default=Category.patient)

    __mapper_args__ = {
        'polymorphic_identity': 'user',
        'polymorphic_on': category,
    }

    # Password context.
    pwd_context = CryptContext(schemes=['bcrypt'], deprecated='auto')

    def __repr__(self) -> str:
        return f'User({self.email}, {self.category})'

    @staticmethod
    def hash_password(password: str) -> str:
        return User.pwd_context.hash(password)

    @staticmethod
    def verify_password(password: str, hash_password: str) -> bool:
        return User.pwd_context.verify(password, hash_password)


class Patient(User):
    # Patient info.
    age = Column(Integer)
    contact = Column(String(15), index=True)
    history = Column(Text)
    aliment = Column(Text)
    last_visit_diagnosis = Column(DateTime)
    guardian_fullname = Column(String(64))
    guardian_email = Column(String)
    guardian_phone = Column(String(15))
    occurences_of_illness = Column(Text)
    last_treatment = Column(DateTime)

    __mapper_args__ = {
        'polymorphic_identity': 'patient',
        'inherit_condition': User.category == Category.patient
    }

    def __repr__(self) -> str:
        return f'Patient({self.email})'


class Practitoner(User):
    practitioner_data = Column(String)

    __mapper_args__ = {
        'polymorphic_identity': 'practitioner',
        'inherit_condition': User.category == Category.practitioner
    }

    def __repr__(self) -> str:
        return f'Practitioner({self.email})'


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

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

from sqlalchemy.orm import Session

from app.schemas import users, model
from app.database import tables


class User:

    @staticmethod
    def get_user(
        db: Session, user_id: int
    ) -> tables.User:
        return db.query(tables.User)\
                .filter(tables.User.id == user_id)\
                .first()

    @staticmethod
    def get_user_by_email(
        db: Session, email: str
    ) -> tables.User:
        return db.query(tables.User)\
                .filter(tables.User.email == email)\
                .first()

    @staticmethod
    def add_user(
        db: Session, user: users.Users
    ) -> tables.User:
        pass


class Patient(User):

    @staticmethod
    def get_patient(
        db: Session, patient_id: int
    ) -> tables.Patient:
        return db.query(tables.Patient)\
                .filter(tables.Patient.id == patient_id)\
                .first()

    @staticmethod
    def add_patient(
        db: Session, patient: users.Patient
    ) -> tables.Patient:
        pass

class Practitioner(User):
    @staticmethod
    def get_practitioner(
        db: Session,
        practitioner_id: int
    ) -> tables.Practitioner:
        return db.query(tables.Practitioner)\
                .filter(tables.Practitioner.id == practitioner_id)\
                .first()

    @staticmethod
    def add_practitioner(
        db: Session, patient: users.Practitioner
    ):
        pass

class Model:

    @staticmethod
    def add_features(
        db: Session, features: model.Features
    ) -> tables.Feature:
        # Create features from schema.
        db_feature = tables.Feature(**features.dict())

        # Add features to db.
        db.add(db_feature)
        db.commit()

        # Update the db with the new feature.
        db.refresh(db_feature)

        return db_feature

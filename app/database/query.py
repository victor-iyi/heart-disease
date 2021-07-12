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
    def get_user(db: Session, user_id: int) -> tables.User:
        return db.query(tables.User).filter(tables.User.id == user_id).first()

    @staticmethod
    def get_user_by_email(db: Session, email: str) -> tables.User:
        return db.query(tables.User).filter(tables.User.email == email).first()

    @staticmethod
    def add_user(db: Session, user: users.Users) -> None:
        pass


class Patient(User):

    @staticmethod
    def add_patient(db: Session, patient: users.Patient) -> None:
        pass


class Model:

    @staticmethod
    def add_features(db: Session, features: model.Features):
        pass

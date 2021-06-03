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

from heart_diesase.data import Data
from heart_disease.models import MODELS


def train_all(filename: str, test_size: float = 0.2) -> None:
    data = Data(filename)

    (X_train, y_train), _ = data.train_test_split(test_size=0.2)

    for name, Model in MODELS.items():
        model = Model()
        model.fit(X_train, y_train)
        model.save_model(f'data/{name}.joblib')


def train_model(model_name: str, filename: str,
                test_size: float = 0.2) -> None:
    data = Data(filename)
    (X_train, y_train), _ = data.train_test_split(test_size=0.2)

    Model = MODELS[model_name]
    model = Model()
    model.fit(X_train, y_train)
    model.save_model(f'data/{model_name}.joblib')

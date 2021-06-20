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

import os
import concurrent.futures

from typing import TypeVar
from functools import partial

from heart_disease.data import Data
from heart_disease.models import MODELS
from heart_disease.config.consts import FS


# Generic `X` & `y` train data.
T = TypeVar('T')


def train_all(filename: str, test_size: float = 0.2) -> None:
    """Train models defined in `models.MODELS`.

    Args:
        filename (str): Filename of CSV data to train models on.
        test_size (float, optional): Test split size. Defaults to 0.2.
    """
    # Load data & split into train/test set.
    data = Data(filename)
    (X_train, y_train), _ = data.train_test_split(test_size=test_size)

    # Create a partial function where data is passed in.
    func = partial(_train_and_save_model, X_train, y_train)

    # Train and save models concurrently.
    with concurrent.futures.ProcessPoolExecutor() as executor:
        executor.map(func, MODELS.keys())


def train_model(model_name: str, filename: str,
                test_size: float = 0.2) -> None:
    """Train a single model given it's model_name in `models.MODELS`.

    Args:
        model_name (str): Name of the model. Avaiable models are found in `models.MODELS`.
        filename (str): Filename of CSV data to train model on.
        test_size (float, optional): Test split size. Defaults to 0.2.
    """
    # Load data & split into train/test set.
    data = Data(filename)
    (X_train, y_train), _ = data.train_test_split(test_size=test_size)

    # Train and save model on multiple process.
    with concurrent.futures.ProcessPoolExecutor() as executor:
        executor.submit(_train_and_save_model, X_train, y_train, model_name)


def _train_and_save_model(X_train: T, y_train: T, model_name: str) -> None:
    """Train and save model process function.

    Args:
        X_train (T): Train features.
        y_train (T): Train labels/target.
        model_name (str): Name of the model.
    """
    # Load model via `model_name` & train it.
    model = MODELS[model_name]()
    model.fit(X_train, y_train)

    # Save model.
    path = os.path.join(FS.SAVED_MODELS, f'{model_name}.joblib')
    model.save_model(path)


if __name__ == '__main__':
    train_all('data/heart.csv')
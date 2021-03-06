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

from typing import Type, TypeVar, Union
from functools import partial

from heart_disease.data import Data
from heart_disease.base import Model
from heart_disease.models import Models


# Generic `X` & `y` train data.
_T = TypeVar('_T')


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
    func = partial(_train_and_save, X_train, y_train)

    # Train and save models concurrently.
    with concurrent.futures.ProcessPoolExecutor() as executor:
        executor.map(func, Models.types())


def train_model(model_name: Union[str, Models], filename: str,
                test_size: float = 0.2) -> None:
    """Train a single model given it's model_name in `models.MODELS`.

    Args:
        model_name (str): Name of the model. Avaiable models are found in
            `models.MODELS`.
        filename (str): Filename of CSV data to train model on.
        test_size (float, optional): Test split size. Defaults to 0.2.
    """
    # Load data & split into train/test set.
    data = Data(filename)
    (X_train, y_train), _ = data.train_test_split(test_size=test_size)

    # Train and save model on multiple process.
    _train_and_save(X_train, y_train, model=Models.get_type(model_name))


def _train_and_save(X_train: _T, y_train: _T, model: Type[Model]) -> None:
    """Train and save model process function.

    Args:
        X_train (_T): Train features.
        y_train (_T): Train labels/target.
        model (Type[Model]): Model to be used.
    """
    # Load & train model.
    _model = model()
    _model.train(X_train, y_train)

    # Save model.
    model.save_model()


if __name__ == '__main__':
    train_all('data/heart.csv')

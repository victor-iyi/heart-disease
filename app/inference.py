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

from typing import Dict, ForwardRef, Iterable, List
from typing import Mapping, TypeVar, Union

import numpy as np

from heart_disease import base
from heart_disease import models
from heart_disease.config import FS

_T = TypeVar('_T', int, float)
_Array = Union[np.ndarray, List[_T]]
_NestedArray = Union[
    _Array,
    Iterable[ForwardRef('_NestedArray')],
    Mapping[str, ForwardRef('_NestedArray')],
]


class SavedModel:
    """Loads saved models and makes prediction."""

    def __init__(self, model_dir: str = FS.SAVED_MODELS) -> None:
        if not os.path.isdir(model_dir):
            raise FileNotFoundError(f'{model_dir} was not found.')

        self.model_dir = model_dir
        self._models: Dict[str, base.Model] = {}

        for model_file in os.listdir(self.model_dir):
            # Saved model name and full path.
            model_name = model_file.removesuffix('.joblib')
            path = os.path.join(model_dir, model_file)

            # Get model via it's saved name.
            _model: base.Model = models.MODELS[model_name]()
            _model.load_model(path)

            self._models[model_name] = _model

    def __getitem__(self, item: str) -> base.Model:
        """Get model by it's name.

        Args:
            item (str): Name of model.

        Returns:
            base.Model: Corresponding (trained) loaded model class.
        """
        return self._models[item]

    def list_available_models(self) -> List[str]:
        """List all the available (trained) models.

        Returns:
            List[str]: List of (trained) loaded models.
        """
        return list(self._models.keys())

    def predict(self, inputs: _Array) -> _NestedArray:
        """Makes prediction from a saved moel (model_dir) given
            unknown features.

        Args:
            inputs (_Array): Array-like of shape (n_samples, n_features).
                Unknown features to be predicted.

        Returns:
            np.ndarray: Output of the saved model.
        """
        results = self._model(inputs)

        prediction = results['prediction']
        confidence = results['confidence']

        results.update({
            # Has heart disease or not (true/false).
            'has_heart_disease': np.cast[bool](prediction),
            # Update `confidence` to (%) and as `has_heart_disease`.
            'confidence': confidence * 100 if confidence is not None else None,
        })

        return results

    @property
    def models(self) -> Dict[str, base.Model]:
        return self._models


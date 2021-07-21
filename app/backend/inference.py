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

import concurrent.futures
import os

from functools import partial
from typing import Any, Dict, List, Optional, TypeVar, Union

import numpy as np

from app.schemas import model
from heart_disease import base
from heart_disease import models
from heart_disease.config import FS, Log


# Typing info.
_T = TypeVar('_T', int, float)
_Array = Union[np.ndarray, List[_T]]


class SavedModel:
    """Loads saved models and makes prediction."""

    def __init__(self, model_dir: Optional[str] = None) -> None:
        self.model_dir = model_dir or FS.SAVED_MODELS

        if not os.path.isdir(model_dir):
            raise FileNotFoundError(f'{self.model_dir} was not found.')

        # Name & model mapping.
        self._models: Dict[str, base.Model] = {}

        for model_file in os.listdir(self.model_dir):
            # Saved model name and full path.
            path = os.path.join(self.model_dir, model_file)
            Log.info(f'Loading {path}...')

            # Get model via it's saved name.
            model_name = model_file.removesuffix('.joblib')
            _model: base.Model = models.MODELS[model_name]()

            _model.load_model(path)
            Log.info(f'Loaded {_model.name} from {path}')

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

    def predict(
        self,
        inputs: _Array, *,
        name: Optional[str] = None,
        model: Optional[base.Model] = None
    ) -> Dict[str, Any]:
        """Makes prediction from saved model given unknown features.

        Args:
            inputs (_Array): Array-like of shape (n_samples, n_features).
                Unknown features to be predicted.
            name (Optional[str], optional): Name of the model to use.
                Defaults to None.
            model (Optional[base.Model], optional): Use model if name is not given.
                Defaults to None.

        Returns:
            Dict[str, Any]: Output of the saved model.
            ```
            {
                model_name: string
                has_heart_disease: boolean
                confidence_score: float
            }
            ```
        """
        if not any(name, model):
            raise ValueError('One of `name` or `model` must be provided.')

        # Use given model or model name.
        try:
            model: base.Model = model or self._models[name]
        except KeyError:
            raise KeyError('Model name not found. See `list_available_models`')

        # Get model prediction.
        result = model(inputs)
        prediction: np.ndarray = np.cast[bool](result['prediction'])

        # Convert confidence to (%) or None if no confidence score.
        confidence = result['confidence']
        if confidence is not None:
            confidence: float = max(confidence) * 100.0

        return {
            # Name of the model used.
            'model_name': model.name,

            # Has heart disease or not (true/false).
            'has_heart_disease': prediction.tolist(),

            # Update `confidence` to (%).
            'confidence_score': confidence,
        }

        return result

    def predict_all(self, inputs: _Array) -> List[Dict[str, Any]]:
        """Make prediction for all saved models.

        Args:
            inputs (_Array): Array-lie of shape (n_samples, n_features).
                Unknown features to be predicted.

        Returns:
            List[Dict[str, Any]]: List of formatted outputs for each model.
        """

        # Returned results.
        results = []

        # Create a partial input function passing the input features.
        func = partial(self.predict, inputs=inputs)

        with concurrent.futures.ProcessPoolExecutor() as executor:
            futures = [
                executor.submit(func, model=model)
                for model in self._models.values()
            ]

            for future in concurrent.futures.as_completed(futures):
                try:
                    results.append(future.result())
                except Exception as e:
                    Log.exception(e)

        # for name, model in self._models.items():
        #     try:
        #         result = self.predict(
        #             inputs=inputs, model=model, name=name
        #         )

        #         results.append(result)
        #     except Exception as e:
        #         Log.exception(e)

        return results

    @staticmethod
    def data_to_array(data: model.Features) -> np.ndarray:
        """Convert `Features` schema to numpy array.

        Args:
            data (model.Features): Features schema.

        Returns:
            np.ndarray: Array-like.
        """
        return np.array(data.dict().values())

    @property
    def models(self) -> Dict[str, base.Model]:
        """Mapping of model names and model classifier object.

        Returns:
            Dict[str, base.Model]: Model mapping.
        """
        return self._models

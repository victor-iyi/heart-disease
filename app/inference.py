from typing import ForwardRef, Iterable, List
from typing import Mapping, TypeVar, Union

import numpy as np

from heart_disease.models import MODELS

_T = TypeVar('_T', int, float)
_Array = Union[np.ndarray, List[_T]]
_NestedArray = Union[
        _Array,
        Iterable[ForwardRef('_NestedArray')],
        Mapping[str, ForwardRef('_NestedArray')],
]


class SavedModel:
    """Loads saved models and makes prediction."""

    def __init__(self, model_dir: str) -> None:
        self.model_dir = model_dir
        self._model = None

    def load_model(self) -> None:
        pass

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

        return results

    def list_available_models(self) -> List[str]:
        models = MODELS.keys()
        return list(models)

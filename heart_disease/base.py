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

"""Module defines abstract base classes for the different ML model classes.

## Methods

- train(self, X: Array, y: Array,
        sample_weight: Optional[Array] = None) -> Model

- test(self, X: Array, y: Array,
       sample_weight: Optional[Array] = None) -> float

- predict(self, inputs: Array) -> Array

- metrics(y_true: Array, y_pred: Array, labels: Optional[Array] = None,
          sample_weight: Optional[Array] = None,
          normalize: Literal['true', 'pred', 'all']):

## Properties

- model: object
"""
from __future__ import annotations

import os

from abc import ABCMeta
from typing import Any, Dict, ForwardRef, Iterable

try:
    from typing import Literal
except ImportError:
    # PEP 586
    from typing_extensions import Literal
from typing import Optional, TypeVar, Union

import joblib
import numpy as np
from sklearn.metrics import confusion_matrix, plot_confusion_matrix
from sklearn.base import ClassifierMixin

from heart_disease.config import Log, FS

# Array-like types.
_T = TypeVar('_T', np.ndarray, int, float)
_Array = Union[_T, Iterable[_T]]


class Model(metaclass=ABCMeta):
    """Base class for classifier models."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        # Various (classifier) model objects. (SVM, NB, DT, KNN).
        self._model: ClassifierMixin = None

        # Name of models (used as their save path).
        self._name: str = kwargs.get('name', self._model.__class__.__name__)

        # Path to save model.
        self._path: str = kwargs.get(
            'path', os.path.join(FS.SAVED_MODELS,
                                 f'{self._name}.joblib')
        )

    def __repr__(self) -> str:
        return repr(self._model)

    def __str__(self) -> str:
        return str(self._model)

    def __call__(self, inputs: _Array) -> Dict[str, _Array]:
        """Makes predictions as well as report the confidence level for each
           class labels.

        Args:
            inputs (_Array): Prediction data. New (unseen) data to be
                predicted.

        Returns:
            Dict[str, _Array]:
                `prediction` with the predicted class index and;

                `confidence` with a floating value for each classes
                representing the `confidence` that a particular class label is
                the right class.
                Some algorithms cannot return the confidence level in
                which case `confidence` is `None`.

            Example:
            ```python
                >>> {
                ...  'prediction': [1., 0., 1.],
                ...  'confidence': [[0.2, 0.8],
                ...                 [1.0, 0.0],
                ...                 [0.3, 0.7]]
                ...  }

            ```
        """

        prediction = self.predict(inputs)
        confidence = self.predict_probability(inputs)

        return {
            'prediction': prediction,
            'confidence': confidence,
        }

    def train(
            self, X: _Array, y: _Array,
    ) -> ForwardRef('Model'):
        """Train model according to X, y.

        Arguments:
            X (_Array): array-like of shape (n_samples, n_features)
                Training vectors, where n_samples is the number of samples
                and n_features is the number of features.
            y (_Array): array-like of shape (n_samples,)
                Target values.

        Raises:
            TypeError - If `self._model` is not defined.

        Returns:
            self (Mode): Base model object.
        """

        if self._model is not None:
            self._model.fit(X, y)
        else:
            raise TypeError('self._model not defined')

        return self

    def test(
            self, X: _Array, y: _Array, *,
            sample_weight: Optional[_Array] = None,
    ) -> float:
        """Return the mean accuracy on the given test data and labels.

        In multi-label classification, this is the subset accuracy
        which is a harsh metric since you require for each sample that each
        label set be correctly predicted.

        Arguments:
            X (_Array): array-like of shape (n_samples, n_features)
                Test samples.
            y (_Array): array-like of shape (n_samples,) or
                (n_samples, n_outputs) - True labels for `X`.
            sample_weight (_Array): array-like of shape (n_samples,).
                Sample weihgt. Defaults to None.

        Raises:
            TypeError - If `self._model` is not set.

        Returns:
            score (float): Mean accuracy of `self.predict(X)` wrt. `y`.
        """
        if self._model is not None:
            return self._model.score(X, y, sample_weight=sample_weight)
        else:
            raise TypeError('self._model not defined')

    def predict(self, inputs: _Array) -> _Array:
        """Perform classification on an array of test vectors X.

        Arguments:
            inputs (_Array): array-like of shape (n_samples, n_features)

        Raises:
            TypeError - If  self._model is not defined.

        Returns:
            C: ndarray of shape (n_samples,) - Predicted target values for X.
        """
        if self._model is not None:
            return self._model.predict(inputs)
        else:
            raise TypeError('self.model is not defined')

    def predict_probability(self, inputs: _Array) -> _Array:
        """Compute probabilities of possible outcomes for samples in X.

        Arguments:
            inputs (Array): array-like of shape (n_samples, n_features) or
                (n_samples_test, n_samples_train)

        Returns:
            T: ndarray of shape (n_samples, n_classes)
                Returns the probability of the sample for each class in the
                model. The columns correspond to the classes in sorted order,
                as they appear in the attribute.
        """
        if self._model is not None:
            try:
                return self._model.predict_proba(inputs)
            except AttributeError as e:
                Log.exception(e)
        else:
            raise TypeError('self.model is not defined')

    def metrics(self, X: _Array, y: _Array, *,
                plot: Optional[bool] = True,
                labels: Optional[_Array] = None,
                sample_weight: Optional[_Array] = None,
                normalize: Optional[Literal['true', 'pred', 'all']] = None
                ) -> _Array:
        """Compute confusion matrix to evaluate the accuracy of a
        classification.

        Arguments:
            X (_Array): array-like of shape (n_samples, n_features)
                - Test samples.
            y (_Array): array-like of shape (n_samples,) or
                (n_samples, n_outputs)
                - Ture lables for `X`.
            labels (_Array): array-like of shape (n_classes).
                List of labels to index the matrix. This may be used to reorder
                or select a subset of labels.
                If `None` is given, those that appear at least once in `y_true`
                or `y_pred` are used in sorted order. Defaults to None.
            sample_weight (_Array): array-like of shape (n_samples,).
                Sample weight. Defaults to None.
            normalize: {'true', 'pred', 'all'}. Normalizes confusion matrix
                over the true (rows), predicted (columns) conditions or all
                population. If None, confusion matrix will not be normalized.

        See Also:
            ``self.confusion_matrix(y_true, y_pred, ...)``.

        Returns:
            C : ndarray of shape (n_classes, n_classes)
                Confusion matrix whose i-th row and j-th column entry
                indicates the number of samples with true lable being i-th
                class and predicted label being j-th class.
        """
        # Ground truth (correct) target values. (n_samples,)
        y_true = y

        # Estimated targets as returned by a classifier. (n_samples,)
        y_pred = self.predict(X)

        if plot:
            # Plot confusion matrix.
            plot_confusion_matrix(self._model, X, y_true,
                                  labels=labels,
                                  sample_weight=sample_weight,
                                  normalize=normalize)

        return confusion_matrix(y_true, y_pred,
                                labels=labels,
                                sample_weight=sample_weight,
                                normalize=normalize)

    def confusion_matrix(
            self, y_true: _Array, y_pred: _Array, *,
            labels: Optional[_Array] = None,
            sample_weight: Optional[_Array] = None,
            normalize: Optional[Literal['true', 'pred', 'all']] = None
    ) -> _Array:
        """Compute confusion matrix to evaluate the accuracy of a
        classification.

        Arguments:
            y_true (_Array): array-like of shape (n_samples,)
                Ground truth (correct) taget values.
            y_pred (_Array): array-like of shape (n_samples,)
                Estimated targets as returned by a classifier.
            labels (_Array): array-like of shape (n_classes).
                List of labels to index the matrix. This may be used to reorder
                or select a subset of labels.
                If `None` is given, those that appear at least once in `y_true`
                or `y_pred` are used in sorted order. Defaults to None.
            sample_weight (_Array): array-like of shape (n_samples,).
                Sample weight. Defaults to None.
            normalize: {'true', 'pred', 'all'}. Normalizes confusion matrix
                over the true (rows), predicted (columns) conditions or all
                population. If None, confusion matrix will not be normalized.

        See Also:
            ``self.metrix(X, y, ...)``.

        Returns:
            C : ndarray of shape (n_classes, n_classes)
                Confusion matrix whose i-th row and j-th column entry
                indicates the number of samples with true lable being i-th
                class and predicted label being j-th class.
        """
        return confusion_matrix(y_true, y_pred, labels=labels,
                                sample_weight=sample_weight,
                                normalize=normalize)

    def save_model(self) -> None:
        """Save classifiers into `path`.

        Arguments:
            path (str): Path to a joblib saved path.
                Preferred extension: `path/to/file.joblib`.

        Example:
            ```python
            >>> from sklearn.linear_model import LogisticRegression
            >>>
            >>> clf = LogisticRegression(solver='lbfgs')
            >>> clf.fit(X_train, y_train)
            LogisticRegression(C=1.0, class_weight=None, dual=False,
                      fit_intercept=True, intercept_scaling=1, max_iter=100,
                      multi_class='warn', n_jobs=None, penalty='l2',
                      random_state=None, solver='lbfgs', tol=0.0001,
                      verbose=0, warm_start=False)
            >>> clf.score(X_test, y_test)
            0.95
            >>> import joblib
            >>> joblib.dump(clf, 'model_1.joblib')
            ['model_1.joblib']
            >>> del clf
            >>> clf = joblib.load('model_1.joblib')
            >>> clf.score(X_test, y_test)
            0.95
            ```
        """
        # Create directory if it doesn't exist.
        os.makedirs(os.path.dirname(self._path), exist_ok=True)

        joblib.dump(self._model, self._path)
        Log.info(f'Model saved to {self._path}')

    def load_model(self, path: str = None) -> None:
        """Load saved classifier from `path`.

        Arguments:
            path (str): Path to a `joblib` saved model.
                Preferred extension: `path/to/file.joblib`.
                Defaults to `{FS.SAVED_MODELS}/{self.name}.joblib`

        Raises:
            FileNotFoundError - If `path` does not exist or isn't a file.
        """
        path = path or self._path

        if os.path.isfile(path):
            raise FileNotFoundError(f'{path} could not be found.')

        self._model = joblib.load(path)
        Log.info(f'Model loaded from {path}')

    save = save_model

    load = load_model

    @property
    def model(self) -> object:
        """Model object defined as a `scikit-learn` model."""
        return self._model

    @model.setter
    def model(self, _model: object) -> None:
        self._model = _model

    @property
    def name(self) -> str:
        """Model name"""
        return self._name

    @property
    def path(self) -> str:
        """Path to model's trained object."""
        return self._path

    @path.setter
    def path(self, path: str) -> None:
        self._path = path

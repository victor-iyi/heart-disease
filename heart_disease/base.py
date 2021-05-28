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

- models: object
"""
from __future__ import annotations

from abc import ABCMeta
from typing import Any, ForwardRef, Iterable, Literal
from typing import Mapping, Optional, TypeVar, Union

import numpy as np
from sklearn.metrics import confusion_matrix


# Array-like types.
_T = TypeVar('_T', int, float)
_Array = Union[np.ndarray, Iterable[_T]]
_NestedArray = Union[
        _Array,
        Iterable[ForwardRef('_NestedArray')],
        Mapping[str, ForwardRef('_NestedArray')],
]


class Model(metaclass=ABCMeta):
    """Base class for classifier models."""

    def __init__(self, *args: Any, **kwargs) -> None:
        # Various model objects. (SVM, NB, DT, KNN).
        self._model: object = None

    def __repr__(self) -> str:
        return repr(self._model)

    def __str__(self) -> str:
        return str(self._model)

    def __call__(self, inputs: _Array) -> _Array:
        """See ``self.predict(X)``."""
        self.predict(inputs)

    def train(
            self, X: _Array, y: _Array,
            sample_weight: Optional[_Array] = None,
    ) -> ForwardRef('Model'):
        """Train model according to X, y.

        Arguments:
            X (_Tensor): array-like of shape (n_samples, n_features)
                Training vectors, where n_samples is the number of samples
                and n_features is the number of features.
            y (_Tensor): array-like of shape (n_samples,)
                Target values.
            sample_weight (_Tensor): array-like of shape (n_samples,).
                Sample weight. Defaults to None.

        Raises:
            TypeError - If `self._model` is not defined.

        Returns:
            self (Mode): Base model object.
        """

        if self._model is not None:
            self._model.fit(X, y, sample_weight=sample_weight)
            return self
        else:
            raise TypeError('self._model not defined')

    def test(
            self, X: _Array, y: _Array,
            sample_weight: Optional[_Array] = None,
    ) -> float:
        """Return the mean accuracy on the given test data and labels.

        In multi-label classification, this is the subset accuracy
        which is a harsh metric since you require for each sample that each
        label set be correctly predicted.

        Arguments:
            X (_Tensor): array-like of shape (n_samples, n_features)
                Test samples.
            y (_Tensor): array-like of shape (n_samples,) or
                (n_samples, n_outputs) - True labels for `X`.
            sample_weight (_Tensor): array-like of shape (n_samples,).
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
            inputs (_Tensor): array-like of shape (n_samples, n_features)

        Raises:
            TypeError - If  self._model is not defined.

        Returns:
            C: ndarray of shape (n_samples,) - Predicted target values for X.
        """
        if self._model is not None:
            return self._model.predict(inputs)
        else:
            raise TypeError('self.model is not defined')

    @staticmethod
    def metrics(
            y_true: _Array, y_pred: _Array, *,
            labels: Optional[_Array] = None,
            sample_weight: Optional[_Array] = None,
            normalize: Optional[Literal['true', 'pred', 'all']] = None
    ) -> _Array:
        """Compute confusion matrix to evaluate the accuracy of a
        classification.

        Arguments:
            y_true (_Tensor): array-like of shape (n_samples,)
                Ground truth (correct) taget values.
            y_pred (_Tenseor): array-like of shape (n_samples,)
                Estimated targets as returned by a classifier.
            labels (_Tensor): array-like of shape (n_classes).
                List of labels to index the matrix. This may be used to reorder
                or select a subset of labels.
                If `None` is given, those that appear at least once in `y_true`
                or `y_pred` are used in sorted order. Defaults to None.
            sample_weight (_Tensor): array-like of shape (n_samples,).
                Sample weight. Defaults to None.
            normalize: {'true', 'pred', 'all'}. Normalizes confusion matrix
                over the true (rows), predicted (columns) conditions or all
                population. If None, confusion matrix will not be normalized.

        Returns:
            C (_Tenosr): ndarray of shape (n_classes, n_classes)
            Confusion matrix whose i-th row and j-th column entry indicates the
            number of samples with true lable being i-th class and predicted
            label being j-th class.
        """

        return confusion_matrix(y_true, y_pred,
                                labels=labels,
                                sample_weight=sample_weight,
                                normalize=normalize)

    @property
    def model(self) -> object:
        return self._model

    @model.setter
    def model(self, _model: object) -> None:
        self._model = _model

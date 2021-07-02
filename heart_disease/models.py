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

from enum import Enum
from typing import Any, Dict

import sklearn.svm as svm
import sklearn.naive_bayes as naive_bayes
import sklearn.neighbors as neighbors
import sklearn.tree as tree

from heart_disease import base


__all__ = [
    'SupportVectorMachine',
    'KNearestNeighbors',
    'NaiveBayes',
    'DecisionTree',
    'Models',
    'MODELS',
]


class SupportVectorMachine(base.Model):
    """C-support vector Classificiation."""

    def __init__(self, **kwargs: Any) -> None:
        super(SupportVectorMachine, self).__init__()

        self._model = svm.SVC(**kwargs)


class KNearestNeighbors(base.Model):
    """Classifier implementing the k-nearest neighbors vote."""

    def __init__(self, k: int = 5, **kwargs: Any) -> None:
        super(KNearestNeighbors, self).__init__()
        self.k = k

        self._model = neighbors.KNeighborsClassifier(
            n_neighbors=self.k,
            n_jobs=-1,
            **kwargs
        )


class NaiveBayes(base.Model):
    """Gaussian Naive Bayes implementation of Naive Bayes."""

    def __init__(self, **kwargs: Any) -> None:
        super(NaiveBayes, self).__init__()

        self._model = naive_bayes.GaussianNB(**kwargs)


class DecisionTree(base.Model):
    """A decision tree classifier."""

    def __init__(self, **kwargs: Any) -> None:
        super(DecisionTree, self).__init__()

        self._model = tree.DecisionTreeClassifier(**kwargs)


class Models(Enum):
    """Classifier enumeration mapping."""
    SVM = SupportVectorMachine
    KNN = KNearestNeighbors
    NB = NaiveBayes
    DT = DecisionTree


MODELS: Dict[str, base.Model] = {
    'Support Vector Machine': SupportVectorMachine,
    'K-Nearest Neighbors': KNearestNeighbors,
    'Naive Bayes': NaiveBayes,
    'Decision Tree': DecisionTree,
}

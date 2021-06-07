from enum import Enum
from typing import Any, Dict

import sklearn.svm as svm
import sklearn.naive_bayes as naive_bayes
import sklearn.tree as tree
import sklearn.neighbors as neighbors

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

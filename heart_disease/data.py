from __future__ import annotations

from typing import Iterable, List, Optional, Tuple
try:
    from typing import Literal
except ImportError:
    # PEP 586
    from typing_extensions import Literal

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split


__all__ = [
    'Data',
]

# Features (n_samples, n_features).
_Features = np.ndarray
# Target (n_samples,)
_Target = np.ndarray

# Train & Test data type-hints.
_TrainData = Tuple[_Features, _Target]
_TestData = Tuple[_Features, _Target]


class Data:
    def __init__(self, filename: str) -> None:
        self.filename = filename
        # Dataframe object.
        self._df: pd.DataFrame = pd.read_csv(filename)

        # Used in `self.__next__`
        self.__current_id = 0

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(filename={self.filename})'

    def __str__(self) -> str:
        return f'{self.__class__.__name__}({self.filename})'

    def __len__(self) -> int:
        return len(self._df)

    def __iter__(self) -> Iterable[Tuple[_Features, _Target]]:
        return self

    def __next__(self) -> Tuple[_Features, _Target]:
        if self.__current_id == len(self._df):
            raise StopIteration

        # Increase the current index.
        self.__current_id += 1

        features = np.array(
                self._df[self.feature_names].loc[self.__current_id],
                dtype=np.float32
        )
        target = np.array(
                self._df[self.target_name].loc[self.__current_id],
                dtype=np.float32
        )

        return features, target

    def get_class_name(self, target: Literal[0, 1]) -> str:
        """Returns what numeric class names represents.

        Args:
            target (Literal[0, 1]): Either a 0 or 1.

        Returns:
            str: Returns corresponding class name given target.
        """
        self.class_names[target]

    def train_test_split(
            self,
            test_size: float = 0.2,
            random_state: Optional[int] = None,
            shuffle: bool = True,
    ) -> Tuple[_TrainData, _TestData]:
        """Split features and labels into random train and test subsets.

        Arguments:
            test_size (float): A number between 0.0 and 1.0 that represents
              the proportion of the dataset to include in the test split.
              Defaults to 0.2 (2% of the data).
            random_state (int, optional): Controls the shuffling applied to
              the data before applying the split. Pass int for reproducable
              output accross multiple function calls. Defaults to None.
            shuffle (bool): Whether or not to shuffle the data before
              splitting. Defaults to True.

        Returns:
          Tuple[TrainData, TestData]: Containing train-test split of
            inputs.
        """
        X_train, X_test, y_train, y_test = train_test_split(
           self.features, self.target, test_size=test_size,
           random_state=random_state, shuffle=shuffle
        )

        # Train data                Test data
        return (X_train, y_train), (X_test, y_test)

    @property
    def df(self) -> pd.DataFrame:
        """Dataframe object."""
        return self._df

    @property
    def columns(self) -> List[str]:
        """Column names."""
        return self._df.columns.tolist()

    @property
    def feature_names(self) -> List[str]:
        """List of feature names."""
        return self.columns[:-1]

    @property
    def target_name(self) -> str:
        """Target (label) name."""
        return self.columns[-1]

    @property
    def features(self) -> _Features:
        """Features as an array-like (n_samples, n_features)."""
        return np.array(self._df[self.feature_names],
                        dtype=np.float32)

    @property
    def target(self) -> _Target:
        """Target (labels) as an array-like (n_samples,)."""
        return np.array(self._df[self.target_name],
                        dtype=np.float32)

    @property
    def n_classes(self) -> int:
        """Number of classes."""
        return len(self._df[self.target_name].unique())

    @property
    def n_samples(self) -> int:
        """Number of data samples."""
        return len(self._df)

    @property
    def class_names(self) -> List[str]:
        return ['No Heart disease', 'Has Heart disease']

<!--
 Copyright 2021 Victor I. Afolabi

 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

     http://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
-->

# Machine Learning Model

The `heart_disease` module provides both `data` and `models` to train different machine learning models with the `data.Data` API. The `models` API is also straight forward to use and intuitive.

## Models

Machine learning models built to diagnose heart diseases include:

- Support Vector Machines (SVM)
- Naive Bayes (NB)
- Descision Trees (DT)
- K-Nearest Neighbors (KNN)

In `heart_disease/base.py` contains the base class for all models (`Model`) which contain the API in which all subclasses are called and implemented.

To create a new model, all you need to do is to create a new `sklearn` model. For example:

```python
import sklearn.ensemble as ensemble
from heart_disease import base


class Ensemble(base.Model):
    """An Adaboost ensemble model."""

    def __init__(self, **kwargs: Any) -> None:
        super(Ensemble, self).__init__()
        self._model = ensemble.AdaBoostClassifier(**kwargs)
```

For access to all models defined in `models.py` two data structures have been created:

- `Models` - `Enum`
- `MODELS` - `Dict[str, base.Model]`

## Data

The `Data` class provides a simple and elegant API for working with CSV data [heart.csv](../data/heart.csv) provided in the [`data/`](../data/) folder.

The data contains the following features:

- Features:

```txt
    age: int
    sex: int        # 0 or 1
    cp: int         # 0, 1, 2 or 3
    trestbps: int
    chol: int
    fbs: int        # 0 or 1
    restecg: int    # 0 or 1
    thalach: int
    exang: int      # 0 or 1
    oldpeak: float
    slope: int      # 0, 1 or 2
    ca: int         # 0 1 or 2
    thal: int       # 0, 1, 2 or 3
```

- Target:

```txt
    target: int     # 0 or 1
```

Below is a sample use of the `Data` class to print out the feature and target names:

```python
>>> from heart_disease.data import Data
>>>
>>> data = Data(filename='data/heart.csv')
>>> data.feature_names
['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal']
>>> data.target_name
'target'
```

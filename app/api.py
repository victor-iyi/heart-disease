import os

from typing import List, Union

import srsly

from dotenv import load_dotenv, find_dotenv
from fastapi import FastAPI, Body
from starlette.responses import RedirectResponse

from app.inference import SavedModel
from app.model import AvailableModels, RecordRequest
from app.model import RecordsRequest, RecordsResponse


load_dotenv(find_dotenv())
PREFIX = os.getenv('CLUSTER_ROUTE_PREFIX', '').rstrip('/')

# Path to `saved_model.pb`
MODEL_DIR = os.getenv('MODEL_DIR', 'res/trained_model/')

# App object.
app = FastAPI(
    title='heart-disease',
    version='1.0',
    description='Predict heart disease with different ML algorithms.',
    openapi_prefix=PREFIX,
)

example_request = srsly.read_json('app/data/example_request.json')
# Loaded saved model object.
model = SavedModel(model_dir=MODEL_DIR)


@app.get('/', include_in_schema=False)
def docs_redirect():
    return RedirectResponse(f'{PREFIX}/docs')


@app.get('/models',
         response_model=AvailableModels,
         response_description='List of available models',
         summary='Return available models',
         tags=['available-models'])
def models():
    """Returns the list of models that are supported by API.

    Response:
        ```json
        {
            "models": [
                "Support Vector Machine",
                "Naive Bayes",
                "Decision Tree",
                "K-Nearest Neighbors"
            ]
        }
        ```
    """
    return model.list_available_models()


@app.post('/predict',
          response_model=RecordRequest,
          response_model_exclude=False,
          response_description='Presence of heart disease or not',
          summary='Make prediction',
          tags=["prediction"])
def predict(
        body: RecordsRequest = Body(..., example=example_request)
):
    """Make a prediction given a model name and list of features.

    - **record_id**: Unique identifier for each set of records to be
        predicted.
    - **model_name**: Name of model. Must be one of [Suport Vector Machine,
        Decision Tree, Naive Bayes and K-Nearest Neighbors].
    - **data**: List of features are:
        age, sex, cp, trestbps, chol, fbs, restecg,
        thalach, exang, oldpeak, slope, ca, thal

    Example:
        ```json
        {
            "record_id": ...,
            "model_name": ...,
            "data": [
                age: ...,
                sex: ...,
                ...,
                thal: ...
            ]
        }
        ```
    """
    res = []
    features: List[Union[int, float]] = []

    for val in body.values:
        features.append(body.values.data.values)

    return res


@app.post('/batch-predict',
          response_model=RecordsRequest,
          response_model_exclude=False,
          response_description='Presence of heart disease or not',
          summary='Make batch prediction',
          tags=['batch', 'prediction'])
def batch_predict(
        body: RecordsRequest = Body(..., example=example_request)
):
    """Perform a batch prediction over mutliple patients with a given
        model name and features for each patients.

    - **values**: List of `RecordRequest`.

    `RecordRequest` has the following info:

    - **record_id**: Unique identifier for each set of records to be
        predicted.
    - **model_name**: Name of model. Must be one of [Suport Vector Machine,
        Decision Tree, Naive Bayes and K-Nearest Neighbors].
    - **data**: List of features are:
        age, sex, cp, trestbps, chol, fbs, restecg,
        thalach, exang, oldpeak, slope, ca, thal

    Example:
        ```json
        [
            {
                "record_id": ...,
                "model_name": ...,
                "data": [
                    "age": ...
                    "sex": ...,
                    ...,
                    "thal": ...
                ]
            },
            {
                "record_id": ...,
                "model_name": ...,
                "data': [
                    ...
                ]
            }
        ]
        ```
    """
    res = []
    return res


@app.get('/metadata')
def metadata():
    """Returns important metadata about current API."""
    return {
        'version': '1.0.0',
        'name': 'heart_disease',
    }

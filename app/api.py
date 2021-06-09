import os
from typing import Dict, List, Union

import srsly

from dotenv import find_dotenv, load_dotenv
from fastapi import Body, FastAPI
from starlette.responses import RedirectResponse

from app.inference import SavedModel
from app.model import AvailableModels, Metadata
from app.model import RecordsRequest, RecordRequest
from app.model import RecordResponse, RecordsResponse
from heart_disease.config.consts import FS


load_dotenv(find_dotenv())
PREFIX: str = os.getenv('CLUSTER_ROUTE_PREFIX', '').rstrip('/')

# Path to `saved_model.pb`
MODEL_DIR: str = os.getenv('MODEL_DIR', FS.SAVED_MODELS)

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
async def docs_redirect():
    return RedirectResponse(f'{PREFIX}/docs')


@app.get('/models',
         response_model=AvailableModels,
         response_description='List of available models',
         summary='Return available models',
         tags=['available-models'])
async def models() -> List[str]:
    """Returns the list of models that are supported by API."""

    return model.list_available_models()


@app.post('/predict',
          response_model=RecordResponse,
          response_model_exclude=False,
          response_description='Presence of heart disease or not',
          summary='Make prediction',
          tags=["prediction"])
async def predict(
        body: RecordRequest = Body(..., example=example_request)
   ) -> Dict[str, str]:
    """Make a prediction given a model name and list of features.

    - **record_id**: Unique identifier for each set of records to be
        predicted.
    - **model_name**: Name of model. Must be one of [Suport Vector Machine,
        Decision Tree, Naive Bayes and K-Nearest Neighbors].
    - **data**: List of features are:
        age, sex, cp, trestbps, chol, fbs, restecg,
        thalach, exang, oldpeak, slope, ca, thal
    """
    res = {}
    # Extract data in correct order.
    data_dict = body.data.dict()
    print(data_dict)

    # Load saved model.
    # Get the model via body.model_name

    # Make prediction.
    return res


@app.post('/batch-predict',
          response_model=RecordsResponse,
          response_model_exclude=False,
          response_description='Presence of heart disease or not',
          summary='Make batch prediction',
          tags=['batch', 'prediction'])
async def batch_predict(
        body: RecordsRequest = Body(..., example=example_request)
   ) -> List[Dict[str, str]]:
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
    """

    for req in body.values:
        [f for _, f in req.data.dict()]

    return []


@app.get('/metadata', response_model=Metadata)
async def metadata() -> Metadata:
    """Returns important metadata about current API."""

    return {
        'version': '1.0.0',
        'name': 'heart_disease',
    }

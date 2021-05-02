import os

import srsly

from dotenv import load_dotenv, find_dotenv
from fastapi import FastAPI
from starlette.responses import RedirectResponse

from app.inference import SavedModel

load_dotenv(find_dotenv())
PREFIX = os.getenv('CLUSTER_ROUTE_PREFIX', '').rstrip('/')

# Path to `saved_model.pb`
MODEL_DIR = os.getenv('MODEL_DIR', 'res/trained_model/model')

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


@app.get('/models')
def models():
    """Returns the list of models that are supported by API."""
    return [
        'Support Vector Machine',
        'Naive Bayes',
    ]


@app.get('/metadata')
def metadata():
    """Returns important metadata about current API."""
    return {
        'version': '1.0.0',
        'name': 'heart_disease',
    }

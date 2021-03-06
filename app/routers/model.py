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

import os
from typing import Any, Dict, List

import srsly

from dotenv import find_dotenv, load_dotenv
from fastapi import APIRouter, Body, Depends
from sqlalchemy.orm.session import Session

from app.backend.inference import SavedModel
from app.database.query import Model
from app.dependencies import get_db
from app.schemas import model

from heart_disease.config.consts import FS


router = APIRouter(
    prefix='/models',
    dependencies=[Depends(get_db)],
    tags=['models'],
)

# Local .env or env files.
load_dotenv(find_dotenv())


# Path to `saved_model.pb`.
MODEL_DIR: str = os.getenv('MODEL_DIR', FS.SAVED_MODELS)


@router.get(
    '/',
    response_model=List[model.AvailableModels],
    response_description='List of available models',
    summary='Return available models',
    tags=['models']
)
async def available_models() -> List[str]:
    """Returns the list of models that are supported by API."""

    # Loaded saved model object.
    saved_model = SavedModel(model_dir=MODEL_DIR)

    return saved_model.list_available_models()


@router.get(
    '/metadata',
    response_model=model.Metadata,
    tags=['models'],
)
async def metadata() -> Dict[str, str]:
    """Returns important metadata about current API."""

    return {
        'name': 'heart-disease',
        'version': 'v1',
        'author': 'Victor I. Afaolbi',
        'author-email': 'javafolabi@gmail.com',
        'license': 'MIT or Apache',
    }


@router.post(
    '/',
    response_model=model.Features,
    dependencies=[Depends(get_db)],
    tags=['models'],
)
async def add_features(
    features: model.Features = Body(
        ..., embed=True,
        example=srsly.read_json('app/sample/models_feature.json')
    ), db: Session = Depends(get_db)
) -> Dict[str, Any]:
    """Add prediction data to the database.

    Args:
        features (model.Features): New features to be added to db.
        db (Session, optional): Database session. Defaults to Depends(get_db).

    Returns:
        model.Features: Added features.
    """

    return Model.add_features(db, features)

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

from fastapi import Depends, FastAPI
from starlette.responses import RedirectResponse

from app.database import Base, engine
from app.dependencies import get_db
from app.routers import model, predict, users


# Create the database.
Base.metadata.create_all(bind=engine)

# App object.
app = FastAPI(
    title='heart-disease',
    version='1.0',
    description='Predict heart disease with different ML algorithms.',
    dependencies=[Depends(get_db)],
)

# Add routers.
app.include_router(model.router)
app.include_router(predict.router)
app.include_router(users.router)


# @app.middleware("http")
# async def db_session_middleware(request: Request, call_next: Callable[[Request], Response]) -> Response:
#     response = Response('Internal server error', status_code=500)
#     try:
#         request.state.db = SessionLocal()
#         response = await call_next(request)
#     finally:
#         request.state.db.close()
#     return response


@app.get('/', include_in_schema=False)
async def docs_redirect() -> RedirectResponse:
    return RedirectResponse(f'/docs')

# @app.on_event('startup')
# async def startup():
#     await database.connect()

# @app.on_event('shutdown')
# async def shutdown():
#     await database.disconnect()

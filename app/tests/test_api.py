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

import pytest
from httpx import AsyncClient

from app.api import app


@pytest.mark.asyncio
async def test_doc_redirect():
    """Redirect docs from `https://domain.com/docs`
    to `https://domain.com/<prefix>/docs`
    """
    async with AsyncClient(app) as client:
        response = await client.get('/')

    assert response.history[0].status_code == 302
    assert response.status_code == 200
    assert response.url == 'http://testserver/docs'

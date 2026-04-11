import asyncio 
from fastapi import FastAPI
from fastapi.testclient import TestClient
from unittest.mock import MagicMock

from app import app, lifespan
from core.config import settings


def test_lifespan_loads_models(monkeypatch):
    mock_load = MagicMock()
    monkeypatch.setattr('app.load_model', mock_load, raising=False)

    async def run_lifespan():
        async with lifespan(FastAPI()):
            pass

    asyncio.run(run_lifespan())

    assert mock_load.call_count == 2
    assert mock_load.call_args_list[0][0][0] == settings.model1_path
    assert mock_load.call_args_list[1][0][0] == settings.model2_path


def test_app_title_and_root_endpoint(monkeypatch):
    mock_load = MagicMock()
    monkeypatch.setattr('app.load_model', mock_load, raising=False)

    with TestClient(app) as client:
        response = client.get('/')

    assert response.status_code == 200
    assert response.json()['app_title'] == settings.api_title
    assert response.json()['version'] == settings.api_version

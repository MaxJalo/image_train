import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import HTTPException
from PIL import Image

from routes import debug

from routes.debug import model_status, run_test_photo


class TestDebugRoutes:
    def test_model_status_returns_state_and_versions(self):
        fake_state = SimpleNamespace(model_status={'model1': 'ok'})
        fake_app = SimpleNamespace(state=fake_state)
        fake_request = SimpleNamespace(app=fake_app)

        result = asyncio.run(model_status(fake_request))

        assert result['models'] == {'model1': 'ok'}
        assert 'python' in result['versions']
        assert 'model_paths' in result

    def test_test_photo_rejects_missing_photo_path(self):
        with pytest.raises(HTTPException) as excinfo:
            asyncio.run(run_test_photo(''))

        assert excinfo.value.status_code == 400

    def test_test_photo_rejects_nonexistent_photo_path(self):
        with pytest.raises(HTTPException) as excinfo:
            asyncio.run(run_test_photo('nonexistent_file.jpg'))

        assert excinfo.value.status_code == 404

    def test_model_status_route_returns_versions(self, test_client):
        response = test_client.get('/debug/model-status')

        assert response.status_code == 200
        payload = response.json()
        assert 'models' in payload
        assert 'model_paths' in payload
        assert 'python' in payload['versions']

    def test_run_test_photo_success(self, tmp_path, monkeypatch):
        photo = tmp_path / 'photo.jpg'
        Image.new('RGB', (16, 16)).save(photo)

        fake_model = MagicMock()
        fake_model.return_value = 'output'

        monkeypatch.setattr('routes.debug.model_loader.get_cached_models', lambda: {})
        monkeypatch.setattr('routes.debug.model_loader.load_model', lambda path: fake_model)

        result = asyncio.run(run_test_photo(str(photo)))

        assert result['model1']['loaded'] is True
        assert result['model2']['loaded'] is True
        assert result['model1']['error'] is None

    def test_test_model1_rejects_missing_photo_path(self):
        with pytest.raises(HTTPException) as excinfo:
            asyncio.run(debug.test_model1(''))

        assert excinfo.value.status_code == 400

    def test_test_model1_rejects_nonexistent_photo_path(self):
        with pytest.raises(HTTPException) as excinfo:
            asyncio.run(debug.test_model1('nonexistent.jpg'))

        assert excinfo.value.status_code == 404

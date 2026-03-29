# (conftest)
import asyncio
import sys
from io import BytesIO
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest
from PIL import Image

root_path = Path(__file__).parent.parent
sys.path.insert(0, str(root_path))


@pytest.fixture
def sample_image():
    img = Image.new("RGB", (224, 224), color="red")
    return img


@pytest.fixture
def sample_image_as_bytes():
    img = Image.new("RGB", (224, 224), color="blue")
    img_bytes = BytesIO()
    img.save(img_bytes, format="PNG")
    img_bytes.seek(0)
    return img_bytes


@pytest.fixture
def sample_grayscale_image():
    img = Image.new("L", (224, 224), color=128)
    return img


@pytest.fixture
def sample_rgba_image():
    img = Image.new("RGBA", (224, 224), color=(255, 0, 0, 255))
    return img


@pytest.fixture
def temp_storage(tmp_path, monkeypatch):
    temp_dir = tmp_path / "wagon_uploads"
    temp_dir.mkdir(parents=True)
    monkeypatch.setattr("services.upload_handler.UploadHandler.TEMP_BASE_DIR", temp_dir, raising=False)
    monkeypatch.setattr("services.upload_handler.TEMP_BASE_DIR", temp_dir, raising=False)
    return temp_dir


@pytest.fixture
def temp_aggregate_dir(tmp_path):
    agg_dir = tmp_path / "photo_aggregate"
    agg_dir.mkdir(parents=True)
    return agg_dir


@pytest.fixture
def mock_mongodb(monkeypatch):
    mock_client = MagicMock()
    mock_db = MagicMock()
    mock_client.__getitem__.return_value = mock_db
    mock_db.list_collection_names = AsyncMock(return_value=[])
    mock_db.__getitem__.return_value = mock_db
    monkeypatch.setattr("db.repository.AsyncIOMotorClient", MagicMock(return_value=mock_client), raising=False)
    return mock_client


@pytest.fixture
def mock_models():
    return {
        "model1": MagicMock(name="model1"),
        "model2": MagicMock(name="model2")
    }


@pytest.fixture
def test_client(monkeypatch):
    from fastapi.testclient import TestClient
    from app import app as fastapi_app
    from services import model_loader

    fake_model = MagicMock()
    fake_model.parameters.return_value = [MagicMock(device="cpu")]
    fake_model.eval = MagicMock()
    fake_model.return_value = MagicMock()

    monkeypatch.setattr("app.load_model", MagicMock(return_value=fake_model), raising=False)
    monkeypatch.setattr(model_loader, "load_model", MagicMock(return_value=fake_model), raising=False)

    with TestClient(fastapi_app) as client:
        yield client


def pytest_configure(config):
    config.addinivalue_line("markers", "unit: unit tests")
    config.addinivalue_line("markers", "integration: integration tests")
    config.addinivalue_line("markers", "slow: slow running tests")

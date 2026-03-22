# (conftest)
import pytest
from pathlib import Path
from unittest.mock import MagicMock
from io import BytesIO
from PIL import Image
import sys

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
def mock_settings():
    mock = MagicMock()
    mock.model1_path = "NN_models/model-1.pt"
    mock.model2_path = "NN_models/model-2.pt"
    mock.mongodb_url = "mongodb://localhost:27017"
    mock.db_name = "wagon_classifier"
    mock.upload_dir = "uploads"
    mock.job_timeout = 3600
    return mock


@pytest.fixture
def temp_upload_dir(tmp_path):
    upload_dir = tmp_path / "uploads"
    upload_dir.mkdir()
    return upload_dir


@pytest.fixture
def temp_aggregate_dir(tmp_path):
    agg_dir = tmp_path / "photo_aggregate"
    agg_dir.mkdir()
    return agg_dir


@pytest.fixture
def mock_model1():
    mock_model = MagicMock()
    mock_model.eval = MagicMock()
    mock_model.to = MagicMock(return_value=mock_model)
    return mock_model


@pytest.fixture
def mock_model2():
    mock_model = MagicMock()
    mock_model.predict = MagicMock()
    mock_model.eval = MagicMock()
    return mock_model


@pytest.fixture
def mock_mongodb_client():
    mock_client = MagicMock()
    mock_db = MagicMock()
    mock_client.__getitem__ = MagicMock(return_value=mock_db)
    return mock_client


@pytest.fixture
def mock_mongodb_document():
    return {
        "_id": "job_12345",
        "batch_id": "batch_12345",
        "status": "completed",
        "total_files": 10,
        "processed_files": 10,
    }


@pytest.fixture
def mock_upload_file():
    mock_file = MagicMock()
    mock_file.filename = "test.jpg"
    mock_file.content_type = "image/jpeg"
    mock_file.size = 10240
    return mock_file


def pytest_configure(config):
    config.addinivalue_line("markers", "unit: unit tests")
    config.addinivalue_line("markers", "integration: integration tests")
    config.addinivalue_line("markers", "slow: slow running tests")

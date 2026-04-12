import zipfile
from io import BytesIO
from unittest.mock import AsyncMock


from services.job_manager import JobManager


def make_image_bytes():
    from PIL import Image

    buffer = BytesIO()
    Image.new("RGB", (16, 16), color="red").save(buffer, format="JPEG")
    buffer.seek(0)
    return buffer


def make_zip_bytes(file_map):
    buffer = BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        for name, data in file_map.items():
            archive.writestr(name, data.getvalue())
    buffer.seek(0)
    return buffer


class TestWagonRoutes:
    def test_upload_zip_endpoint_with_valid_zip(self, test_client, monkeypatch, temp_storage):
        zip_bytes = make_zip_bytes({"images/photo1.jpg": make_image_bytes()})
        mock_batch = {
            "batch_id": "batch_123",
            "status": "success",
            "data": {"batch_id": "batch_123"},
        }
        monkeypatch.setattr(
            "routes.wagon.aggregator.get_batch_results", AsyncMock(return_value=mock_batch)
        )
        monkeypatch.setattr("routes.wagon.process_job", lambda *args, **kwargs: None)

        response = test_client.post(
            "/api/ml/upload/zip",
            files={"file": ("test.zip", zip_bytes.getvalue(), "application/zip")},
        )

        assert response.status_code == 200
        assert response.json()["status"] == "success"
        assert response.json()["data"]["batch_id"] == "batch_123"

    def test_upload_zip_endpoint_rejects_invalid_file(self, test_client):
        response = test_client.post(
            "/api/ml/upload/zip", files={"file": ("bad.txt", b"hello world", "text/plain")}
        )

        assert response.status_code == 400
        assert "ZIP" in response.json()["detail"]

    def test_upload_single_file_success(self, test_client, monkeypatch, tmp_path):
        image = make_image_bytes()
        monkeypatch.setattr(
            "routes.wagon.UploadHandler.save_single_file",
            AsyncMock(return_value=(True, tmp_path / "job", None)),
        )
        monkeypatch.setattr("routes.wagon.process_job", lambda *args, **kwargs: None)

        response = test_client.post(
            "/api/ml/upload/single?camera_id=1&wagon_id=wagon_1",
            files={"file": ("photo.jpg", image.getvalue(), "image/jpeg")},
        )

        assert response.status_code == 200
        assert response.json()["status"] == "success"
        assert response.json()["wagon_id"] == "wagon_1"

    def test_upload_multiple_files_success(self, test_client, monkeypatch, tmp_path):
        image1 = make_image_bytes()
        image2 = make_image_bytes()
        monkeypatch.setattr(
            "routes.wagon.UploadHandler.save_multiple_files",
            AsyncMock(return_value=(True, tmp_path / "job", None, 2)),
        )
        monkeypatch.setattr("routes.wagon.process_job", lambda *args, **kwargs: None)

        response = test_client.post(
            "/api/ml/upload/multiple?camera_id=2",
            files=[
                ("files", ("image1.jpg", image1.getvalue(), "image/jpeg")),
                ("files", ("image2.jpg", image2.getvalue(), "image/jpeg")),
            ],
        )

        assert response.status_code == 200
        assert response.json()["status"] == "success"
        assert response.json()["files_received"] == 2

    def test_upload_multiple_files_failure(self, test_client, monkeypatch, tmp_path):
        image = make_image_bytes()
        monkeypatch.setattr(
            "routes.wagon.UploadHandler.save_multiple_files",
            AsyncMock(return_value=(False, tmp_path / "job", "save failed", 0)),
        )

        response = test_client.post(
            "/api/ml/upload/multiple?camera_id=2",
            files=[("files", ("image1.jpg", image.getvalue(), "image/jpeg"))],
        )

        assert response.status_code == 400
        assert "save failed" in response.json()["detail"]

    def test_get_job_status_returns_not_found(self, test_client):
        response = test_client.get("/api/ml/job/does_not_exist")
        assert response.status_code == 404
        assert "Задание не найдено" in response.json()["detail"]

    def test_get_job_status_returns_pending_job(self, test_client):
        job_id = "job_test_1"
        JobManager.create_job(job_id, total_files=1)

        response = test_client.get(f"/api/ml/job/{job_id}")

        assert response.status_code == 200
        assert response.json()["job_id"] == job_id
        assert response.json()["job_status"] == "pending"

    def test_get_job_status_simple_returns_message(self, test_client):
        job_id = "job_test_2"
        JobManager.create_job(job_id, total_files=1)

        response = test_client.get(f"/api/ml/job/{job_id}/status")

        assert response.status_code == 200
        assert response.json()["status"] == "pending"
        assert "Задание в очереди" in response.json()["message"]

    def test_get_batch_status_returns_data(self, test_client, monkeypatch):
        monkeypatch.setattr(
            "routes.wagon.aggregator.get_batch_status",
            AsyncMock(return_value={"batch_id": "batch_1", "status": "completed"}),
        )

        response = test_client.get("/api/ml/batch-status/batch_1")

        assert response.status_code == 200
        assert response.json()["data"]["status"] == "completed"

    def test_get_batch_results_returns_data(self, test_client, monkeypatch):
        monkeypatch.setattr(
            "routes.wagon.aggregator.get_batch_results",
            AsyncMock(return_value={"batch_id": "batch_1", "status": "completed", "results": {}}),
        )

        response = test_client.get("/api/ml/batch-results/batch_1")

        assert response.status_code == 200
        assert response.json()["data"]["batch_id"] == "batch_1"

import asyncio
from pathlib import Path
from unittest.mock import AsyncMock

from services import aggregator, classifier, detector, job_manager, upload_handler
from services.background_process import process_job
from services.job_manager import JobStatus


class TestBackgroundProcess:
    def test_process_job_completes_successfully(self, tmp_path, monkeypatch):
        job_id = "job_bg"
        job_manager.create_job(job_id, total_files=1)

        temp_dir = tmp_path / "upload"
        temp_dir.mkdir()
        image_file = temp_dir / "image.jpg"
        image_file.write_bytes(b"JPEGDATA")

        monkeypatch.setattr(
            upload_handler.UploadHandler, "get_job_files", lambda _job_id: [image_file]
        )
        monkeypatch.setattr(
            classifier,
            "classify_and_group_wagons",
            AsyncMock(return_value={"wagon_1": [(image_file, 0)]}),
        )
        monkeypatch.setattr(
            detector,
            "detect_wagon_sides",
            AsyncMock(
                return_value={
                    "wagon_1": {
                        "total_photos": 1,
                        "processed_photos": 1,
                        "left_count": 1,
                        "right_count": 0,
                        "final_side": "left",
                        "cameras": [0],
                    }
                }
            ),
        )
        monkeypatch.setattr(
            aggregator, "process_and_save_batch", AsyncMock(return_value="batch_123")
        )
        monkeypatch.setattr(upload_handler.UploadHandler, "cleanup_job_files", lambda _job_id: True)

        asyncio.run(process_job(job_id, str(temp_dir)))

        stored = job_manager.get_job(job_id)
        assert stored.status == JobStatus.COMPLETED
        assert stored.result["batch_id"].startswith("batch_")
        assert stored.result["processed_files"] == 1

    def test_process_job_fails_for_missing_folder(self):
        job_id = "job_missing"
        job_manager.create_job(job_id, total_files=1)

        asyncio.run(process_job(job_id, str(Path("does_not_exist"))))

        stored = job_manager.get_job(job_id)
        assert stored.status == JobStatus.FAILED
        assert "Папка не найдена" in stored.error

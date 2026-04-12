import asyncio 
import zipfile
from io import BytesIO
from pathlib import Path

from fastapi import UploadFile
from PIL import Image

from services.upload_handler import UploadHandler, ALLOWED_EXTENSIONS, MAX_FILE_SIZE


class TestUploadHandler:
    def make_image_bytes(self, format="JPEG"):
        buffer = BytesIO()
        Image.new("RGB", (16, 16), color="green").save(buffer, format=format)
        buffer.seek(0)
        return buffer

    def make_zip_bytes(self, files):
        buffer = BytesIO()
        with zipfile.ZipFile(buffer, "w") as archive:
            for name, data in files.items():
                archive.writestr(name, data.getvalue())
        buffer.seek(0)
        return buffer

    def test_extract_and_save_zip_with_valid_zip(self, temp_storage, tmp_path):
        image_bytes = self.make_image_bytes()
        zip_bytes = self.make_zip_bytes({"nested/camera_1/photo.jpg": image_bytes})
        upload = UploadFile(filename="test.zip", file=BytesIO(zip_bytes.read()))

        success, job_dir, warning, extracted_files = asyncio.run(
            UploadHandler.extract_and_save_zip(upload, "job1")
        )

        assert success is True
        assert job_dir.name == "job1"
        assert extracted_files
        assert warning is None
        assert (job_dir / "extracted" / "nested" / "camera_1" / "photo.jpg").exists()

    def test_extract_and_save_zip_with_corrupted_zip(self, temp_storage):
        upload = UploadFile(filename="broken.zip", file=BytesIO(b"not a zip"))

        success, job_dir, error, extracted_files = asyncio.run(
            UploadHandler.extract_and_save_zip(upload, "job2")
        )

        assert success is False
        assert "ZIP" in error
        assert extracted_files == []

    def test_validate_image_file_rejects_unsupported_extension(self):
        upload = UploadFile(filename="photo.txt", file=BytesIO(b"dummy"))

        valid, error = asyncio.run(
            UploadHandler.validate_image_file(upload)
        )

        assert valid is False
        assert "Тип файла не поддерживается" in error

    def test_validate_image_file_rejects_large_file(self):
        data = BytesIO(b"0" * (MAX_FILE_SIZE + 1))
        upload = UploadFile(filename="large.jpg", file=data)

        valid, error = asyncio.run(
            UploadHandler.validate_image_file(upload)
        )

        assert valid is False
        assert "Размер файла превышает лимит" in error

    def test_validate_image_file_accepts_valid_image(self):
        buffer = BytesIO()
        Image.new("RGB", (16, 16)).save(buffer, format="PNG")
        buffer.seek(0)
        upload = UploadFile(filename="photo.png", file=buffer)

        valid, error = asyncio.run(
            UploadHandler.validate_image_file(upload)
        )

        assert valid is True
        assert error is None

    def test_save_single_file_saves_valid_image(self, temp_storage):
        buffer = BytesIO()
        Image.new("RGB", (16, 16)).save(buffer, format="PNG")
        buffer.seek(0)
        upload = UploadFile(filename="photo.png", file=buffer)

        success, job_dir, error = asyncio.run(
            UploadHandler.save_single_file(upload, "job4", camera_id=123)
        )

        assert success is True
        assert error is None
        saved_file = job_dir / "camera_123" / "photo.png"
        assert saved_file.exists()
        assert saved_file.read_bytes()[:4] == b"\x89PNG"

    def test_save_multiple_files_handles_partial_failure(self, temp_storage):
        valid_buffer = BytesIO()
        Image.new("RGB", (16, 16)).save(valid_buffer, format="PNG")
        valid_buffer.seek(0)

        valid_file = UploadFile(filename="good.png", file=BytesIO(valid_buffer.read()))
        invalid_file = UploadFile(filename="bad.txt", file=BytesIO(b"dummy"))

        success, job_dir, error, saved_count = asyncio.run(
            UploadHandler.save_multiple_files([valid_file, invalid_file], "job5", camera_id=1)
        )

        assert success is True
        assert saved_count == 1
        assert "Тип файла не поддерживается" in error
        assert (job_dir / "camera_1" / "good.png").exists()

    def test_get_job_files_returns_saved_images(self, temp_storage):
        job_id = "job6"
        job_dir = temp_storage / job_id
        job_dir.mkdir(parents=True)
        (job_dir / "photo.jpg").write_bytes(b"JPEG")

        files = UploadHandler.get_job_files(job_id)

        assert len(files) == 1
        assert files[0].name == "photo.jpg"

    def test_extract_metadata_from_path_parses_nested_structure(self, tmp_path):
        extract_root = tmp_path / "extracted"
        nested_dir = extract_root / "trainhash" / "123"
        nested_dir.mkdir(parents=True)
        image_path = nested_dir / "photo.jpg"
        image_path.write_bytes(b"dummy")

        camera_id, train_hash, depth = UploadHandler._extract_metadata_from_path(image_path, extract_root)

        assert camera_id == "123"
        assert train_hash == "trainhash"
        assert depth == 2

    def test_cleanup_job_files_removes_job_directory(self, temp_storage):
        job_dir = temp_storage / "job3"
        (job_dir / "extracted").mkdir(parents=True)
        (job_dir / "extracted" / "image.png").write_bytes(b"abc")

        assert UploadHandler.cleanup_job_files("job3") is True
        assert not job_dir.exists()

import os 
import time
from pathlib import Path

import pytest

from services import storage


@pytest.fixture(autouse=True)
def isolate_aggregate_dir(tmp_path, monkeypatch):
    aggregate_dir = tmp_path / "photo_aggregate"
    monkeypatch.setattr(storage, "get_aggregate_dir", lambda: aggregate_dir)
    yield


class TestStorageService:
    def test_copy_photo_to_wagon_with_real_file(self, tmp_path):
        source_file = tmp_path / "sample.jpg"
        source_file.write_bytes(b"JPEGDATA")

        result = storage.copy_photo_to_wagon(str(source_file), "wagon_1")

        assert result is True
        target_file = (tmp_path / "photo_aggregate" / "wagon_1" / "sample.jpg")
        assert target_file.exists()
        assert target_file.read_bytes() == b"JPEGDATA"

    def test_delete_wagon_removes_directory(self, tmp_path):
        wagon_dir = tmp_path / "photo_aggregate" / "wagon_2"
        wagon_dir.mkdir(parents=True)
        (wagon_dir / "image.jpg").write_bytes(b"123")

        assert storage.delete_wagon("wagon_2") is True
        assert not wagon_dir.exists()

    def test_cleanup_old_wagons_with_different_timestamps(self, tmp_path):
        old_wagon = tmp_path / "photo_aggregate" / "old_wagon"
        recent_wagon = tmp_path / "photo_aggregate" / "recent_wagon"
        old_wagon.mkdir(parents=True)
        recent_wagon.mkdir(parents=True)

        old_time = time.time() - (60 * 60 * 24 * 10)
        recent_time = time.time()
        os.utime(old_wagon, (old_time, old_time))
        os.utime(recent_wagon, (recent_time, recent_time))

        result = storage.cleanup_old_wagons(days=5)

        assert "old_wagon" in result["deleted"]
        assert "recent_wagon" not in result["deleted"]
        assert (tmp_path / "photo_aggregate" / "recent_wagon").exists()

    def test_get_wagon_photos_returns_correct_paths(self, tmp_path):
        wagon_dir = tmp_path / "photo_aggregate" / "wagon_3"
        wagon_dir.mkdir(parents=True)
        photo = wagon_dir / "door.jpg"
        photo.write_bytes(b"JPEGDATA")

        photos = storage.get_wagon_photos("wagon_3")

        assert len(photos) == 1
        assert photos[0]["filename"] == "door.jpg"
        assert photos[0]["relative_path"] == "wagon_3/door.jpg"
        assert Path(photos[0]["path"]).exists()

    def test_list_wagons_ignores_nested_directories(self, tmp_path):
        root = tmp_path / "photo_aggregate"
        (root / "wagon_a").mkdir(parents=True)
        (root / "wagon_b").mkdir(parents=True)
        (root / "wagon_b" / "nested").mkdir(parents=True)

        wagons = storage.list_wagons()

        assert wagons == ["wagon_a", "wagon_b"]

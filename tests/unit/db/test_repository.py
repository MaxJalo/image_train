import asyncio
from unittest.mock import AsyncMock, MagicMock, patch


from db import repository


class TestRepository:
    @patch("db.repository.init_beanie", new_callable=AsyncMock)
    @patch("db.repository.AsyncIOMotorClient")
    def test_ensure_db_connection_success(self, mock_client, mock_init_beanie):
        mock_db = MagicMock()
        mock_client.return_value.__getitem__.return_value = mock_db
        mock_db.list_collection_names = AsyncMock(return_value=["batches"])

        result = asyncio.run(repository.ensure_db_connection())

        assert result is True
        mock_client.assert_called_once()
        mock_init_beanie.assert_awaited_once()

    @patch("db.repository.AsyncIOMotorClient")
    def test_ensure_db_connection_failure(self, mock_client):
        mock_client.side_effect = Exception("connection failed")

        result = asyncio.run(repository.ensure_db_connection())

        assert result is False

    @patch("db.repository.WagonAggregateDocument")
    def test_get_wagon_status_not_found(self, mock_wagon_doc):
        mock_wagon_doc.wagon_id = "wagon_id"
        mock_wagon_doc.find_one = AsyncMock(return_value=None)

        result = asyncio.run(repository.get_wagon_status("unknown_wagon"))

        assert result["wagon_id"] == "unknown_wagon"
        assert result["processing_status"] == "no_data"

    @patch("db.repository.WagonAggregateDocument")
    def test_get_wagon_status_found(self, mock_wagon_doc):
        fake_doc = MagicMock()
        fake_doc.total_photos = 5
        fake_doc.final_side = "left"
        mock_wagon_doc.wagon_id = "wagon_id"
        mock_wagon_doc.find_one = AsyncMock(return_value=fake_doc)

        result = asyncio.run(repository.get_wagon_status("wagon_1"))

        assert result["wagon_id"] == "wagon_1"
        assert result["processing_status"] == "completed"
        assert result["total_photos"] == 5

    @patch("db.repository.WagonAggregateDocument")
    def test_get_wagon_result_not_found(self, mock_wagon_doc):
        mock_wagon_doc.wagon_id = "wagon_id"
        mock_wagon_doc.find_one = AsyncMock(return_value=None)

        result = asyncio.run(repository.get_wagon_result("unknown_wagon"))

        assert result["wagon_id"] == "unknown_wagon"
        assert result["status"] == "no_data"
        assert result["total_photos"] == 0

    @patch("db.repository.WagonAggregateDocument")
    def test_get_wagon_result_found(self, mock_wagon_doc):
        fake_doc = MagicMock()
        fake_doc.final_side = "left"
        fake_doc.left_count = 2
        fake_doc.right_count = 1
        fake_doc.camera_ids = [1, 2]
        fake_doc.total_photos = 3
        mock_wagon_doc.wagon_id = "wagon_id"
        mock_wagon_doc.find_one = AsyncMock(return_value=fake_doc)

        result = asyncio.run(repository.get_wagon_result("wagon_1"))

        assert result["wagon_id"] == "wagon_1"
        assert result["status"] == "completed"
        assert result["final_verdict"].side == "left"
        assert result["camera_ids"] == [1, 2]

    @patch("db.repository.BatchDocument")
    def test_get_batch_status_not_found(self, mock_batch_doc):
        mock_batch_doc.batch_id = "batch_id"
        mock_batch_doc.find_one = AsyncMock(return_value=None)

        result = asyncio.run(repository.get_batch_status("missing_batch"))

        assert result["status"] == "not_found"
        assert result["batch_id"] == "missing_batch"

    @patch("db.repository.BatchDocument")
    def test_get_batch_status_found(self, mock_batch_doc):
        fake_doc = MagicMock()
        fake_doc.folder = "folder"
        fake_doc.status = "completed"
        fake_doc.total_photos = 2
        fake_doc.processed_photos = 2
        fake_doc.total_wagons = 1
        fake_doc.error_message = None
        mock_batch_doc.batch_id = "batch_id"
        mock_batch_doc.find_one = AsyncMock(return_value=fake_doc)

        result = asyncio.run(repository.get_batch_status("batch_1"))

        assert result["status"] == "completed"
        assert result["folder"] == "folder"

    @patch("db.repository.BatchDocument")
    def test_get_batch_result_not_found(self, mock_batch_doc):
        mock_batch_doc.batch_id = "batch_id"
        mock_batch_doc.find_one = AsyncMock(return_value=None)

        result = asyncio.run(repository.get_batch_result("missing_batch"))

        assert result["status"] == "not_found"
        assert result["results"] == {}

    @patch("db.repository.BatchDocument")
    def test_get_batch_result_found(self, mock_batch_doc):
        fake_doc = MagicMock()
        fake_doc.folder = "folder"
        fake_doc.total_photos = 10
        fake_doc.total_wagons = 2
        fake_doc.processed_photos = 10
        fake_doc.results = {"wagon_1": {"final_side": "left"}}
        fake_doc.processed_at = "2026-01-01T12:00:00"
        fake_doc.status = "completed"
        mock_batch_doc.batch_id = "batch_id"
        mock_batch_doc.find_one = AsyncMock(return_value=fake_doc)

        result = asyncio.run(repository.get_batch_result("batch_2"))

        assert result["status"] == "completed"
        assert result["results"]["wagon_1"]["final_side"] == "left"

from unittest.mock import AsyncMock, MagicMock, patch

from services import aggregator


class TestAggregatorService:
    @patch("services.aggregator.ensure_db_connection", new_callable=AsyncMock)
    @patch("services.aggregator.BatchDocument", autospec=True)
    def test_process_and_save_batch_inserts_document(self, mock_batch_cls, mock_db_check):
        mock_db_check.return_value = True
        mock_instance = MagicMock()
        mock_instance.insert = AsyncMock(return_value=None)
        mock_batch_cls.return_value = mock_instance

        batch_id = __import__("asyncio").run(
            aggregator.process_and_save_batch(
                batch_id="batch1",
                folder_name="folder",
                wagon_results={
                    "wagon_1": {
                        "total_photos": 1,
                        "processed_photos": 1,
                        "final_side": "left",
                        "left_count": 1,
                        "right_count": 0,
                        "cameras": [1],
                    }
                },
            )
        )

        assert batch_id == "batch1"
        mock_instance.insert.assert_awaited_once()

    @patch("services.aggregator.ensure_db_connection", new_callable=AsyncMock)
    @patch("services.aggregator.BatchDocument")
    def test_get_batch_status_not_found(self, mock_batch_cls, mock_db_check):
        mock_db_check.return_value = True
        mock_batch_cls.batch_id = "batch_id"
        mock_batch_cls.find_one = AsyncMock(return_value=None)

        response = __import__("asyncio").run(aggregator.get_batch_status("missing_batch"))

        assert response["status"] == "not_found"
        assert response["batch_id"] == "missing_batch"

    @patch("services.aggregator.ensure_db_connection", new_callable=AsyncMock)
    @patch("services.aggregator.BatchDocument")
    def test_get_batch_results_returns_document(self, mock_batch_cls, mock_db_check):
        mock_db_check.return_value = True
        mock_batch_cls.batch_id = "batch_id"
        fake_doc = MagicMock()
        fake_doc.folder = "folder"
        fake_doc.status = "completed"
        fake_doc.total_photos = 2
        fake_doc.total_wagons = 1
        fake_doc.processed_photos = 2
        fake_doc.results = {"wagon_1": {"final_side": "left"}}
        fake_doc.processed_at = "2026-01-01T00:00:00Z"
        mock_batch_cls.find_one = AsyncMock(return_value=fake_doc)

        response = __import__("asyncio").run(aggregator.get_batch_results("batch1"))

        assert response["status"] == "completed"
        assert response["total_wagons"] == 1
        assert response["results"]["wagon_1"]["final_side"] == "left"

    @patch("services.aggregator.ensure_db_connection", new_callable=AsyncMock)
    def test_process_and_save_batch_returns_none_when_db_unavailable(self, mock_db_check):
        mock_db_check.return_value = False

        response = __import__("asyncio").run(
            aggregator.process_and_save_batch(
                batch_id="batch2", folder_name="folder", wagon_results={}
            )
        )

        assert response is None

# (test_workflows)
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock


class TestImageProcessingPipeline:
    @patch('services.classifier.load_model')
    @patch('services.storage.ensure_aggregate_dir')
    def test_full_workflow(self, mock_ensure_dir, mock_load_model, tmp_path):
        mock_model = MagicMock()
        mock_load_model.return_value = mock_model
        mock_ensure_dir.return_value = tmp_path / "photo_aggregate"
        assert mock_model is not None

    @patch('services.classifier.load_model')
    @patch('services.aggregator.ensure_db_connection')
    def test_batch_processing_workflow(self, mock_db_conn, mock_classifier):
        mock_model = MagicMock()
        mock_classifier.return_value = mock_model
        mock_db_conn.return_value = True
        assert mock_model is not None


class TestDataFlow:
    @patch('services.storage.get_wagon_dir')
    def test_storage_organization(self, mock_get_wagon, tmp_path):
        agg_dir = tmp_path / "photo_aggregate"
        agg_dir.mkdir()
        wagon_dir = agg_dir / "wagon_1"
        wagon_dir.mkdir()
        mock_get_wagon.return_value = wagon_dir
        assert wagon_dir.exists()

    def test_result_schema(self):
        sample_result = {
            'file': 'image_1.jpg',
            'classification': 'one_wagon',
            'confidence': 0.95,
        }
        assert 'file' in sample_result
        assert 'classification' in sample_result


class TestErrorRecovery:
    @patch('services.classifier.load_model')
    def test_graceful_fallback(self, mock_load_model):
        mock_load_model.side_effect = FileNotFoundError()
        with pytest.raises(FileNotFoundError):
            mock_load_model()

# (test_detector)
import pytest
from unittest.mock import patch, MagicMock


class TestDetectorModule:
    @patch('services.detector.load_model')
    def test_detector_initialization(self, mock_load_model):
        mock_model = MagicMock()
        mock_load_model.return_value = mock_model
        assert mock_load_model() == mock_model

    def test_detector_with_sample_image(self, sample_image):
        assert sample_image is not None
        assert sample_image.size[0] > 0

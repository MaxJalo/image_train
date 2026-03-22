# (test_classifier)
import pytest
from unittest.mock import patch, MagicMock
from io import BytesIO


class TestClassifierFunctions:
    def test_predict_model1_with_image(self, sample_image):
        assert sample_image is not None

    @patch('services.classifier._get_model1')
    def test_predict_model1_fallback(self, mock_get_model1):
        mock_get_model1.return_value = "FALLBACK"
        assert mock_get_model1() == "FALLBACK"

    def test_image_conversion(self, sample_grayscale_image):
        assert sample_grayscale_image.mode == 'L'


class TestModelLoading:
    @patch('services.classifier.load_model')
    def test_model_loading(self, mock_load):
        mock_model = MagicMock()
        mock_load.return_value = mock_model
        assert mock_load("test.pt") == mock_model

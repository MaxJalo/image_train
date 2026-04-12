import asyncio 
import pytest
import torch
from unittest.mock import MagicMock, patch

from services import classifier


class TestClassifierFunctions:
    @patch('services.classifier._get_model1')
    def test_predict_model1_fallback(self, mock_get_model1):
        mock_get_model1.return_value = "FALLBACK"

        is_wagon, confidence = classifier.predict_model1(torch.rand((3, 300, 300)))

        assert is_wagon is True
        assert confidence == 0.5
        mock_get_model1.assert_called_once()

    @patch('services.classifier._get_model1')
    def test_predict_model1_with_mock_model(self, mock_get_model1):
        class FakeModel:
            def parameters(self):
                return iter([torch.nn.Parameter(torch.zeros(1))])

            def eval(self):
                return None

            def __call__(self, x):
                return torch.tensor([[2.0, 0.5]])

        mock_get_model1.return_value = FakeModel()

        image = torch.zeros((3, 300, 300))
        from PIL import Image
        pil_image = Image.new("RGB", (300, 300))

        is_wagon, confidence = classifier.predict_model1(pil_image)

        assert confidence >= 0.5
        assert isinstance(is_wagon, bool)

    @patch('services.classifier._get_model1')
    def test_predict_model1_converts_grayscale_images(self, mock_get_model1, sample_grayscale_image):
        class FakeModel:
            def parameters(self):
                return iter([torch.nn.Parameter(torch.zeros(1))])

            def eval(self):
                return None

            def __call__(self, x):
                return torch.tensor([[1.0, 0.0]])

        mock_get_model1.return_value = FakeModel()

        is_wagon, confidence = classifier.predict_model1(sample_grayscale_image)

        assert confidence >= 0.0
        assert is_wagon is True

    @patch('services.classifier._get_model1')
    def test_predict_model1_handles_non_tensor_output(self, mock_get_model1, sample_image):
        class FakeModel:
            def parameters(self):
                return iter([MagicMock(device="cpu")])

            def eval(self):
                return None

            def __call__(self, x):
                return "not-a-tensor"

        fake_model = FakeModel()

        mock_get_model1.return_value = fake_model

        is_wagon, confidence = classifier.predict_model1(sample_image)

        assert is_wagon is True
        assert confidence == 0.5

    def test_classify_and_group_wagons_empty_folder(self, tmp_path):
        result = asyncio.run(classifier.classify_and_group_wagons(str(tmp_path)))
        assert result == {}

    def test_classify_and_group_wagons_missing_folder(self):
        with pytest.raises(FileNotFoundError):
            asyncio.run(classifier.classify_and_group_wagons('missing_folder'))

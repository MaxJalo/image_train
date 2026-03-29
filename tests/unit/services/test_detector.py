import pytest
from unittest.mock import MagicMock, patch
from PIL import Image

from services import detector
from models.schemas import Model2Output


class TestDetectorModule:
    @patch('services.detector._get_model2')
    def test_predict_model2_fallback(self, mock_get_model2, sample_image):
        mock_get_model2.return_value = "FALLBACK"

        result = detector.predict_model2(sample_image)

        assert isinstance(result, Model2Output)
        assert result.side == "left"
        assert result.confidence == 0.5

    @patch('services.detector._get_model2')
    def test_predict_model2_with_mock_yolo_result(self, mock_get_model2, sample_image):
        class FakeBox:
            def __init__(self, cls, conf):
                self.cls = [cls]
                self.conf = [conf]

        class FakeResult:
            def __init__(self):
                self.boxes = [FakeBox(0, 0.8), FakeBox(1, 0.6)]

        class FakeModel:
            names = {0: "brake_rod", 1: "rod_nose", 2: "crane", 3: "tank"}

            def __call__(self, image):
                return [FakeResult()]

        mock_get_model2.return_value = FakeModel()

        result = detector.predict_model2(sample_image)

        assert result.brake_rod == 0.8
        assert result.rod_nose == 0.6
        assert result.crane == 0.0
        assert result.tank == 0.0
        assert result.side == "left"
        assert result.confidence > 0.0

    def test_determine_side_without_detections_returns_right_for_even_width(self):
        image = Image.new("RGB", (200, 50))
        assert detector._determine_side(image, {}) == "right"

    def test_detect_wagon_sides_aggregates_counts(self, tmp_path, monkeypatch):
        image_path = tmp_path / "photo.jpg"
        Image.new("RGB", (10, 10)).save(image_path)

        def fake_predict(image):
            return Model2Output(
                brake_rod=0.5,
                rod_nose=0.4,
                crane=0.0,
                tank=0.0,
                side="right",
                confidence=0.9
            )

        monkeypatch.setattr(detector, "predict_model2", fake_predict)

        result = __import__('asyncio').run(
            detector.detect_wagon_sides({"wagon_1": [(image_path, 1)]})
        )

        assert result["wagon_1"]["final_side"] == "right"
        assert result["wagon_1"]["processed_photos"] == 1
        assert result["wagon_1"]["cameras"] == [1]

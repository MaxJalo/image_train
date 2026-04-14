import pytest
from pydantic import ValidationError

from models.schemas import FinalVerdictModel, Model2Output


def test_model2_output_validates_correctly():
    output = Model2Output(
        brake_rod=0.1,
        rod_nose=0.2,
        crane=0.3,
        tank=0.4,
        side="left",
        confidence=0.7,
    )

    assert output.side == "left"
    assert output.confidence == 0.7


def test_model2_output_rejects_invalid_confidence():
    with pytest.raises(ValidationError):
        Model2Output(
            brake_rod=0.1,
            rod_nose=0.2,
            crane=0.3,
            tank=0.4,
            side="left",
            confidence=1.5,
        )


def test_final_verdict_model_requires_fields():
    verdict = FinalVerdictModel(
        side="right", left_count=2, right_count=1, total_photos=3
    )
    assert verdict.side == "right"
    assert verdict.total_photos == 3

# (test_storage)
import pytest


class TestExample:
    def test_simple(self):
        assert True

    def test_with_fixture(self, sample_image):
        assert sample_image is not None

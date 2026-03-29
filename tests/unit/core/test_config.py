# (test_config)
from pathlib import Path

import pytest

from core import config


class TestConfigSettings:
    def test_validate_model_paths_with_existing_files(self, tmp_path, monkeypatch):
        model1 = tmp_path / 'model-1.pt'
        model2 = tmp_path / 'model-2.pt'
        model1.write_text('dummy')
        model2.write_text('dummy')

        monkeypatch.setattr(config.settings, 'model1_path', str(model1))
        monkeypatch.setattr(config.settings, 'model2_path', str(model2))

        result = config.settings.validate_model_paths()

        assert result['valid'] is True
        assert result['model1']['exists'] is True
        assert result['model1']['readable'] is True
        assert result['model2']['exists'] is True
        assert result['model2']['readable'] is True

    def test_validate_model_paths_with_missing_files(self, tmp_path, monkeypatch):
        missing1 = tmp_path / 'missing-1.pt'
        missing2 = tmp_path / 'missing-2.pt'

        monkeypatch.setattr(config.settings, 'model1_path', str(missing1))
        monkeypatch.setattr(config.settings, 'model2_path', str(missing2))

        result = config.settings.validate_model_paths()

        assert result['valid'] is False
        assert result['model1']['exists'] is False
        assert result['model1']['readable'] is False
        assert result['model2']['exists'] is False
        assert result['model2']['readable'] is False

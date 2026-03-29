import os
import pickle
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from services.model_loader import load_model, _load_pickle_model


class TestModelLoader:
    def test_load_model_pt_uses_internal_loader(self, tmp_path):
        model_path = tmp_path / 'test_model.pt'
        model_path.write_bytes(b'fake')

        with patch('services.model_loader.Path.exists', return_value=True), \
             patch('services.model_loader._load_pytorch_model', return_value=MagicMock()) as mock_loader:
            model = load_model(str(model_path))

        assert model is not None
        mock_loader.assert_called_once()

    def test_load_model_pkl_reads_pickle_file(self, tmp_path):
        model_path = tmp_path / 'test_model.pkl'
        value = {'hello': 'world'}
        with open(model_path, 'wb') as f:
            pickle.dump(value, f)

        model = load_model(str(model_path))

        assert model == value

    def test_load_model_unsupported_extension(self, tmp_path):
        model_path = tmp_path / 'test_model.onnx'
        model_path.write_text('dummy')

        with patch('services.model_loader.Path.exists', return_value=True):
            with pytest.raises(ValueError):
                load_model(str(model_path))

    def test_load_model_file_not_found(self):
        with pytest.raises(FileNotFoundError):
            load_model('missing_model.pt')

    def test_load_pickle_model_raises_on_invalid_pickle(self, tmp_path):
        invalid_path = tmp_path / 'invalid.pkl'
        invalid_path.write_bytes(b'not a pickle')

        with pytest.raises(Exception):
            _load_pickle_model(str(invalid_path))

    def test_load_model_caches_models(self, tmp_path, monkeypatch):
        model_path = tmp_path / 'test_model.pt'
        model_path.write_bytes(b'fake')

        with patch('services.model_loader.Path.exists', return_value=True), \
             patch('services.model_loader._load_pytorch_model', return_value=MagicMock()) as mock_loader:
            model = load_model(str(model_path))

        assert model is not None
        assert load_model(str(model_path)) is model
        mock_loader.assert_called_once()

    def test_clear_cache_and_get_cached_models(self):
        from services import model_loader

        model_loader._model_cache.clear()
        model_loader._model_cache['cached'] = MagicMock()

        cached = model_loader.get_cached_models()
        assert 'cached' in cached

        model_loader.clear_cache()
        assert model_loader.get_cached_models() == {}


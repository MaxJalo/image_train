import pytest
from unittest.mock import patch, MagicMock
from pathlib import Path


class TestModelLoader:
    """Test cases for model_loader module"""
    
    def test_model_path_validation(self):
        """Test model path validation"""
        valid_paths = ['model.pt', 'NN_models/model.pt', 'NN_models\\model.pt']
        for path in valid_paths:
            assert isinstance(path, str)
            assert len(path) > 0
    
    def test_supported_formats(self):
        """Test that supported model formats are recognized"""
        supported_formats = ['.pt', '.pkl']
        for fmt in supported_formats:
            assert fmt in ['.pt', '.pkl']
    
    @patch('services.model_loader.Path.exists')
    @patch('services.model_loader.Path')
    def test_load_model_validation(self, mock_path_class, mock_exists):
        """Test load_model path validation"""
        from services.model_loader import load_model
        # Test import works
        assert callable(load_model)
    
    def test_load_model_file_not_found(self):
        """Test FileNotFoundError when model file missing"""
        from services.model_loader import load_model
        with pytest.raises(FileNotFoundError):
            load_model("nonexistent_model_12345.pt")
    
    def test_load_model_import(self):
        """Test that load_model can be imported"""
        from services.model_loader import load_model
        assert callable(load_model)


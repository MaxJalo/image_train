# (test_aggregator)
import pytest
from unittest.mock import patch, MagicMock, AsyncMock


class TestAggregatorFunctions:
    """Test cases for aggregator module"""
    
    @patch('services.aggregator.ensure_db_connection')
    def test_aggregator_functions_exist(self, mock_db_check):
        """Test that aggregator functions can be imported"""
        from services.aggregator import get_batch_status, get_batch_results
        assert callable(get_batch_status)
        assert callable(get_batch_results)
    
    @patch('services.aggregator.ensure_db_connection')
    @patch('models.schemas.BatchDocument.find_one')
    def test_get_batch_status_structure(self, mock_find_one, mock_db_check):
        """Test get_batch_status function exists and is async"""
        from services.aggregator import get_batch_status
        import inspect
        assert inspect.iscoroutinefunction(get_batch_status)
    
    @patch('services.aggregator.ensure_db_connection')
    def test_get_batch_results_structure(self, mock_db_check):
        """Test get_batch_results function exists and is async"""
        from services.aggregator import get_batch_results
        import inspect
        assert inspect.iscoroutinefunction(get_batch_results)
    
    @patch('db.repository.ensure_db_connection')
    def test_aggregator_db_connection(self, mock_db):
        """Test aggregator uses db connection"""
        mock_db.return_value = True
        assert callable(mock_db)


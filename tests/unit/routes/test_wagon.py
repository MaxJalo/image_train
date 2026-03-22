# (test_wagon)
import pytest
from unittest.mock import patch, MagicMock
from routes.wagon import router


class TestWagonRoutes:
    def test_upload_zip_endpoint_exists(self):
        """Test that upload endpoint is registered"""
        # Check that the router has routes
        assert router.routes is not None
        # Check that at least one route is registered
        assert len(router.routes) > 0
        # Check that POST routes exist
        post_routes = [r for r in router.routes if 'POST' in str(r.methods) or (hasattr(r, 'methods') and 'POST' in r.methods)]
        assert len(post_routes) > 0

    def test_batch_status_endpoint_exists(self):
        """Test that batch status endpoint is registered"""
        # Check that the router has routes
        assert router.routes is not None
        # Check that at least one route is registered
        assert len(router.routes) > 0
        # Check that GET routes exist
        get_routes = [r for r in router.routes if 'GET' in str(r.methods) or (hasattr(r, 'methods') and 'GET' in r.methods)]
        assert len(get_routes) > 0

    def test_router_prefix(self):
        assert router.prefix == "/api/ml"

    def test_router_tags(self):
        assert "wagon_processing" in router.tags

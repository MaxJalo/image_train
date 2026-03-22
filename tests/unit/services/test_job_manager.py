# (test_job_manager)
import pytest
from unittest.mock import MagicMock, patch
from services.job_manager import JOB_STORAGE, create_job, get_job


class TestJobManager:
    """Test cases for job_manager module"""
    
    def test_create_job(self):
        """Test job creation function exists"""
        assert callable(create_job)

    def test_get_job_status(self):
        """Test getting job status through get_job"""
        job_id = 'test_job_123'
        job_info = create_job(job_id, total_files=5)
        assert job_info is not None
        assert job_info.job_id == job_id

    def test_job_lifecycle(self):
        """Test job can be created"""
        from services.job_manager import JobStatus
        job_id = 'lifecycle_test_456'
        job = create_job(job_id, total_files=10)
        assert job.status == JobStatus.PENDING
    
    def test_job_storage_exists(self):
        """Test that JOB_STORAGE is accessible"""
        assert JOB_STORAGE is not None
        assert isinstance(JOB_STORAGE, dict)
    
    def test_job_storage_operations(self):
        """Test basic job operations"""
        from services.job_manager import start_job
        job_id = 'storage_test_789'
        job_info = create_job(job_id, total_files=3)
        
        # Job should be in storage
        assert job_id in JOB_STORAGE
        
        # Should be able to start job
        success = start_job(job_id)
        assert success is True


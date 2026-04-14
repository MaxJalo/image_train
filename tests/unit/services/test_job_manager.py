from datetime import datetime, timedelta

import pytest

from services.job_manager import (
    JOB_STORAGE,
    JobInfo,
    JobManager,
    JobStatus,
    complete_job,
    create_job,
    fail_job,
    get_job,
    start_job,
    update_progress,
)


@pytest.fixture(autouse=True)
def clear_jobs_store():
    JOB_STORAGE.clear()
    yield
    JOB_STORAGE.clear()


class TestJobManager:
    def test_full_job_lifecycle(self):
        job_id = "job_lifecycle"

        job_info = create_job(job_id, total_files=5)
        assert job_info.job_id == job_id
        assert job_info.status == JobStatus.PENDING

        assert start_job(job_id) is True
        assert JOB_STORAGE[job_id]["status"] == JobStatus.PROCESSING.value

        assert update_progress(job_id, processed_files=2) is True
        assert JOB_STORAGE[job_id]["processed_files"] == 2
        assert 0 < JOB_STORAGE[job_id]["progress"] < 100

        assert complete_job(job_id, {"result": "ok"}) is True
        assert JOB_STORAGE[job_id]["status"] == JobStatus.COMPLETED.value
        assert JOB_STORAGE[job_id]["progress"] == 100
        assert JOB_STORAGE[job_id]["result"] == {"result": "ok"}

        restored = get_job(job_id)
        assert isinstance(restored, JobInfo)
        assert restored.status == JobStatus.COMPLETED

    def test_parallel_job_updates(self):
        first = create_job("job_a", total_files=2)
        second = create_job("job_b", total_files=3)

        assert start_job("job_a") is True
        assert start_job("job_b") is True

        assert update_progress("job_a", processed_files=1, progress=50) is True
        assert update_progress("job_b", processed_files=2) is True

        assert JOB_STORAGE["job_a"]["progress"] == 50
        assert JOB_STORAGE["job_b"]["processed_files"] == 2

    def test_recovery_after_failure(self):
        job_id = "job_error"
        create_job(job_id, total_files=1)
        assert fail_job(job_id, "Something went wrong") is True

        stored = get_job(job_id)
        assert stored.status == JobStatus.FAILED
        assert stored.error == "Something went wrong"

    def test_cleanup_old_jobs_removes_stale_entries(self):
        job_id = "job_old"
        create_job(job_id, total_files=1)
        old_time = datetime.now() - timedelta(hours=25)
        JOB_STORAGE[job_id]["created_at"] = old_time.isoformat()

        JobManager.cleanup_old_jobs(max_age_hours=24)

        assert job_id not in JOB_STORAGE

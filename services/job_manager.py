# (job_manager)

import logging
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

# Хранилище заданий (в реальном приложении использовать Redis или БД)
_jobs_store: Dict[str, Dict[str, Any]] = {}

# Экспортируемое имя для совместимости с тестами
JOB_STORAGE = _jobs_store


class JobStatus(str, Enum):
    """Статусы заданий"""

    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class JobInfo:
    """Информация о задании"""

    job_id: str
    status: JobStatus
    progress: int
    total_files: int
    processed_files: int
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    created_at: datetime = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None


class JobManager:
    """Менеджер для управления заданиями обработки"""

    @staticmethod
    def create_job(job_id: str, total_files: int = 1) -> JobInfo:
        """Создать новое задание"""
        logger.info(f"📝 Создание задания: {job_id}, файлов: {total_files}")

        job_info = JobInfo(
            job_id=job_id,
            status=JobStatus.PENDING,
            progress=0,
            total_files=total_files,
            processed_files=0,
            created_at=datetime.now(),
        )

        _jobs_store[job_id] = {
            "job_id": job_id,
            "status": JobStatus.PENDING.value,
            "progress": 0,
            "total_files": total_files,
            "processed_files": 0,
            "result": None,
            "error": None,
            "created_at": datetime.now().isoformat(),
            "started_at": None,
            "completed_at": None,
        }

        logger.debug(f"✅ Задание создано: {job_id}")
        return job_info

    @staticmethod
    def get_job(job_id: str) -> Optional[JobInfo]:
        """Получить информацию о задании"""
        job_data = _jobs_store.get(job_id)

        if not job_data:
            logger.debug(f"⚠️ Задание не найдено: {job_id}")
            return None

        return JobInfo(
            job_id=job_data["job_id"],
            status=JobStatus(job_data["status"]),
            progress=job_data["progress"],
            total_files=job_data["total_files"],
            processed_files=job_data["processed_files"],
            result=job_data.get("result"),
            error=job_data.get("error"),
            created_at=datetime.fromisoformat(job_data["created_at"]),
            started_at=(
                datetime.fromisoformat(job_data["started_at"])
                if job_data.get("started_at")
                else None
            ),
            completed_at=(
                datetime.fromisoformat(job_data["completed_at"])
                if job_data.get("completed_at")
                else None
            ),
        )

    @staticmethod
    def start_job(job_id: str) -> bool:
        """Начать обработку задания"""
        if job_id not in _jobs_store:
            logger.error(f"❌ Задание не найдено: {job_id}")
            return False

        logger.info(f"▶️ Начало обработки задания: {job_id}")
        _jobs_store[job_id]["status"] = JobStatus.PROCESSING.value
        _jobs_store[job_id]["started_at"] = datetime.now().isoformat()
        _jobs_store[job_id]["progress"] = 5  # Начальный прогресс

        return True

    @staticmethod
    def update_progress(job_id: str, processed_files: int, progress: int = None) -> bool:
        """Обновить прогресс обработки"""
        if job_id not in _jobs_store:
            logger.error(f"❌ Задание не найдено: {job_id}")
            return False

        job_data = _jobs_store[job_id]
        job_data["processed_files"] = processed_files

        if progress is not None:
            job_data["progress"] = min(progress, 99)  # Максимум 99% до завершения
        else:
            # Автоматический расчет прогресса
            if job_data["total_files"] > 0:
                job_data["progress"] = int((processed_files / job_data["total_files"]) * 90) + 5

        logger.debug(f"📊 Прогресс {job_id}:\
             {job_data['progress']}% ({processed_files}/{job_data['total_files']})")
        return True

    @staticmethod
    def complete_job(job_id: str, result: Dict[str, Any]) -> bool:
        """Завершить задание успешно"""
        if job_id not in _jobs_store:
            logger.error(f"❌ Задание не найдено: {job_id}")
            return False

        logger.info(f"✅ Завершение задания: {job_id}")
        _jobs_store[job_id]["status"] = JobStatus.COMPLETED.value
        _jobs_store[job_id]["progress"] = 100
        _jobs_store[job_id]["result"] = result
        _jobs_store[job_id]["completed_at"] = datetime.now().isoformat()

        return True

    @staticmethod
    def fail_job(job_id: str, error: str) -> bool:
        """Завершить задание с ошибкой"""
        if job_id not in _jobs_store:
            logger.error(f"❌ Задание не найдено: {job_id}")
            return False

        logger.error(f"❌ Ошибка в задании {job_id}: {error}")
        _jobs_store[job_id]["status"] = JobStatus.FAILED.value
        _jobs_store[job_id]["error"] = error
        _jobs_store[job_id]["completed_at"] = datetime.now().isoformat()

        return True

    @staticmethod
    def cleanup_old_jobs(max_age_hours: int = 24):
        """Очистить старые задания старше max_age_hours часов"""
        from datetime import timedelta

        now = datetime.now()
        jobs_to_delete = []

        for job_id, job_data in _jobs_store.items():
            created_at = datetime.fromisoformat(job_data["created_at"])
            age = now - created_at

            if age > timedelta(hours=max_age_hours):
                jobs_to_delete.append(job_id)

        for job_id in jobs_to_delete:
            del _jobs_store[job_id]
            logger.debug(f"🗑️ Удалено старое задание: {job_id}")

        if jobs_to_delete:
            logger.info(f"🗑️ Очищено {len(jobs_to_delete)} старых заданий")


# Функции-хелперы для быстрого доступа
def create_job(job_id: str, total_files: int = 1) -> JobInfo:
    """Создать новое задание"""
    return JobManager.create_job(job_id, total_files)


def get_job(job_id: str) -> Optional[JobInfo]:
    """Получить статус задания"""
    return JobManager.get_job(job_id)


def start_job(job_id: str) -> bool:
    """Начать обработку"""
    return JobManager.start_job(job_id)


def update_progress(job_id: str, processed_files: int, progress: int = None) -> bool:
    """Обновить прогресс"""
    return JobManager.update_progress(job_id, processed_files, progress)


def complete_job(job_id: str, result: Dict[str, Any]) -> bool:
    """Завершить успешно"""
    return JobManager.complete_job(job_id, result)


def fail_job(job_id: str, error: str) -> bool:
    """Завершить с ошибкой"""
    return JobManager.fail_job(job_id, error)

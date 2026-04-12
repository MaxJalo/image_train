# (wagon)
import logging
import uuid
from typing import List, Optional

from fastapi import APIRouter, BackgroundTasks, File, HTTPException, Query, UploadFile

from services import aggregator
from services.background_process import process_job
from services.job_manager import JobManager, JobStatus
from services.upload_handler import UploadHandler

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/ml", tags=["wagon_processing"])


@router.post("/upload/zip")
async def upload_zip_file(
    file: UploadFile = File(..., description="ZIP архив с изображениями"),
    background_tasks: BackgroundTasks = BackgroundTasks(),
):

    logger.info(f"🚀 POST /upload/zip: {file.filename}")

    batch_id = f"batch_{uuid.uuid4().hex[:12]}"
    job_id = f"job_{uuid.uuid4().hex[:12]}"
    logger.info(f"📝 Batch ID: {batch_id}")

    try:
        # Проверка расширения
        if not file.filename.lower().endswith(".zip"):
            error = "Файл должен быть ZIP архивом (.zip)"
            logger.error(f"❌ {error}")
            raise HTTPException(status_code=400, detail=error)

        # Создать задание (будет обновлено после распаковки)
        JobManager.create_job(job_id, total_files=1)

        # Распаковать и сохранить
        success, job_dir, error, extracted_files = await UploadHandler.extract_and_save_zip(
            file=file, job_id=job_id
        )

        if not success:
            logger.error(f"❌ Ошибка обработки ZIP: {error}")
            JobManager.fail_job(job_id, error)
            raise HTTPException(status_code=400, detail=error)

        extracted_count = len(extracted_files)
        logger.info(f"✅ Извлечено {extracted_count} изображений из ZIP в {job_dir}")

        # Обновить информацию о задании
        JobManager.create_job(job_id, total_files=extracted_count)

        # Запустить фоновую обработку
        background_tasks.add_task(
            process_job, job_id=job_id, folder_path=str(job_dir / "extracted"), wagon_id=None
        )

        logger.info(f"✅ Задание {job_id} отправлено на обработку")

        return await aggregator.get_batch_results(batch_id)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Ошибка обработки ZIP: {type(e).__name__}: {str(e)}")
        JobManager.fail_job(job_id, str(e))
        raise HTTPException(status_code=500, detail=f"Ошибка при обработке ZIP: {str(e)}")


# ==================== BATCH INFORMATION ====================


@router.get("/batch-status/{batch_id}")
async def get_batch_status(batch_id: str):
    """
    Получить статус обработки батча из MongoDB.
    """
    logger.debug(f"GET /batch-status/{batch_id}")

    try:
        status = await aggregator.get_batch_status(batch_id)
        return {"status": "success" if status.get("status") != "error" else "error", "data": status}
    except Exception as e:
        logger.error(f"❌ Ошибка получения статуса: {type(e).__name__}: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/batch-results/{batch_id}")
async def get_batch_results(batch_id: str):
    """
    Получить полные результаты батча из MongoDB.
    """
    logger.debug(f"GET /batch-results/{batch_id}")

    try:
        result = await aggregator.get_batch_results(batch_id)
        return {"status": "success" if result.get("status") != "error" else "error", "data": result}
    except Exception as e:
        logger.error(f"❌ Ошибка получения результатов: {type(e).__name__}: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


# ==================== FILE UPLOAD ENDPOINTS ====================


@router.post("/upload/single")
async def upload_single_file(
    file: UploadFile = File(..., description="Один файл изображения"),
    camera_id: Optional[int] = Query(None, description="ID камеры"),
    wagon_id: Optional[str] = Query(None, description="ID вагона (если известен)"),
    background_tasks: BackgroundTasks = BackgroundTasks(),
):
    logger.info(f"🚀 POST /upload/single: {file.filename}")

    job_id = f"job_{uuid.uuid4().hex[:12]}"
    logger.info(f"📝 Job ID: {job_id}")

    try:
        # Создать задание
        JobManager.create_job(job_id, total_files=1)

        # Сохранить файл
        success, job_dir, error = await UploadHandler.save_single_file(
            file=file, job_id=job_id, camera_id=camera_id
        )

        if not success:
            logger.error(f"❌ Ошибка сохранения: {error}")
            JobManager.fail_job(job_id, error)
            raise HTTPException(status_code=400, detail=error)

        logger.info(f"✅ Файл сохранен: {job_dir}")

        # Запустить фоновую обработку
        background_tasks.add_task(
            process_job,
            job_id=job_id,
            folder_path=str(job_dir),
            camera_id=camera_id,
            wagon_id=wagon_id,
        )

        logger.info(f"✅ Задание {job_id} отправлено на обработку")

        return {
            "status": "success",
            "job_id": job_id,
            "message": "Файл принят и добавлен в очередь обработки",
            "file": file.filename,
            "camera_id": camera_id,
            "wagon_id": wagon_id,
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Ошибка загрузки: {type(e).__name__}: {str(e)}")
        JobManager.fail_job(job_id, str(e))
        raise HTTPException(status_code=500, detail=f"Ошибка при загрузке: {str(e)}")


@router.post("/upload/multiple")
async def upload_multiple_files(
    files: List[UploadFile] = File(..., description="Несколько файлов изображений"),
    camera_id: Optional[int] = Query(None, description="ID камеры"),
    background_tasks: BackgroundTasks = BackgroundTasks(),
):
    logger.info(f"🚀 POST /upload/multiple: {len(files)} файлов")

    job_id = f"job_{uuid.uuid4().hex[:12]}"
    logger.info(f"📝 Job ID: {job_id}")

    try:
        # Проверка что файлы есть
        if not files or len(files) == 0:
            error = "Не указаны файлы для загрузки"
            logger.error(f"❌ {error}")
            raise HTTPException(status_code=400, detail=error)

        # Создать задание
        JobManager.create_job(job_id, total_files=len(files))

        # Сохранить файлы
        success, job_dir, error, saved_count = await UploadHandler.save_multiple_files(
            files=files, job_id=job_id, camera_id=camera_id
        )

        if not success:
            logger.error(f"❌ Ошибка сохранения: {error}")
            JobManager.fail_job(job_id, error)
            raise HTTPException(status_code=400, detail=error)

        logger.info(f"✅ Сохранено {saved_count} файлов в {job_dir}")

        # Запустить фоновую обработку
        background_tasks.add_task(
            process_job, job_id=job_id, folder_path=str(job_dir), camera_id=camera_id, wagon_id=None
        )

        logger.info(f"✅ Задание {job_id} отправлено на обработку")

        return {
            "status": "success",
            "job_id": job_id,
            "message": f"Принято {saved_count} файлов\
             из {len(files)}. Задание добавлено в очередь обработки",
            "files_received": len(files),
            "files_saved": saved_count,
            "camera_id": camera_id,
            "warning": error if error else None,
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Ошибка загрузки: {type(e).__name__}: {str(e)}")
        JobManager.fail_job(job_id, str(e))
        raise HTTPException(status_code=500, detail=f"Ошибка при загрузке: {str(e)}")


# ==================== JOB STATUS ENDPOINTS ====================


@router.get("/job/{job_id}")
async def get_job_status(job_id: str):

    logger.debug(f"GET /job/{job_id}")

    job_info = JobManager.get_job(job_id)

    if not job_info:
        logger.warning(f"⚠️ Задание не найдено: {job_id}")
        raise HTTPException(status_code=404, detail=f"Задание не найдено: {job_id}")

    return {
        "status": "success",
        "job_id": job_id,
        "job_status": job_info.status.value,
        "progress": job_info.progress,
        "processed_files": job_info.processed_files,
        "total_files": job_info.total_files,
        "result": job_info.result,
        "error": job_info.error,
        "created_at": job_info.created_at.isoformat() if job_info.created_at else None,
        "started_at": job_info.started_at.isoformat() if job_info.started_at else None,
        "completed_at": job_info.completed_at.isoformat() if job_info.completed_at else None,
    }


@router.get("/job/{job_id}/status")
async def get_job_status_simple(job_id: str):
    logger.debug(f"GET /job/{job_id}/status")

    job_info = JobManager.get_job(job_id)

    if not job_info:
        logger.warning(f"⚠️ Задание не найдено: {job_id}")
        raise HTTPException(status_code=404, detail=f"Задание не найдено: {job_id}")

    # Определить сообщение о статусе
    status_messages = {
        JobStatus.PENDING: "Задание в очереди ожидания",
        JobStatus.PROCESSING: f"Обработка...\
         {job_info.processed_files}/{job_info.total_files} файлов",
        JobStatus.COMPLETED: "Задание завершено успешно",
        JobStatus.FAILED: f"Ошибка: {job_info.error}",
    }

    return {
        "status": job_info.status.value,
        "progress": job_info.progress,
        "message": status_messages.get(job_info.status, "Неизвестный статус"),
        "job_id": job_id,
    }

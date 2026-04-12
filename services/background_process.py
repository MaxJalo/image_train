# (background_process)
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

from services import aggregator, classifier, detector
from services.job_manager import JobManager
from services.upload_handler import UploadHandler

logger = logging.getLogger(__name__)


async def process_job(
    job_id: str, folder_path: str, camera_id: Optional[int] = None, wagon_id: Optional[str] = None
) -> None:

    logger.info(f"🚀 Начало обработки задания {job_id}")
    logger.info(f"   📂 Папка: {folder_path}")
    logger.info(f"   📷 Camera ID: {camera_id}")
    logger.info(f"   🚂 Wagon ID: {wagon_id}")

    # Обновить статус на PROCESSING
    JobManager.start_job(job_id)

    temp_dir = None
    try:
        # Получить папку с файлами
        temp_dir = Path(folder_path)

        if not temp_dir.exists():
            error = f"Папка не найдена: {folder_path}"
            logger.error(f"❌ {error}")
            JobManager.fail_job(job_id, error)
            return

        # Получить список файлов
        image_files = UploadHandler.get_job_files(job_id)

        if not image_files:
            error = "Не найдено изображений для обработки"
            logger.error(f"❌ {error}")
            JobManager.fail_job(job_id, error)
            return

        logger.info(f"📸 Найдено файлов для обработки: {len(image_files)}")
        JobManager.update_progress(job_id, 0, 10)

        # ============ STEP 1: Классификация (Model-1) ============
        logger.info(f"⏳ Шаг 1: Классификация (Model-1) и группировка по вагонам...")
        try:
            wagon_groups = await classifier.classify_and_group_wagons(str(temp_dir))

            if not wagon_groups:
                error = "Не найдено вагонов для обработки"
                logger.warning(f"⚠️ {error}")
                JobManager.fail_job(job_id, error)
                return

            logger.info(f"✅ Выделено вагонов: {len(wagon_groups)}")
            JobManager.update_progress(job_id, len(image_files) // 3, 35)

        except Exception as e:
            error = f"Ошибка классификации (Model-1): {str(e)}"
            logger.error(f"❌ {error}")
            JobManager.fail_job(job_id, error)
            return

        # ============ STEP 2: Детекция (Model-2) ============
        logger.info(f"⏳ Шаг 2: Детекция (Model-2) и определение сторон...")
        try:
            wagon_results = await detector.detect_wagon_sides(wagon_groups)

            if not wagon_results:
                error = "Ошибка при детекции вагонов"
                logger.error(f"❌ {error}")
                JobManager.fail_job(job_id, error)
                return

            logger.info(f"✅ Обработано вагонов: {len(wagon_results)}")
            JobManager.update_progress(job_id, len(image_files) * 2 // 3, 70)

        except Exception as e:
            error = f"Ошибка детекции (Model-2): {str(e)}"
            logger.error(f"❌ {error}")
            JobManager.fail_job(job_id, error)
            return

        # ============ STEP 3: Агрегация и сохранение в MongoDB ============
        logger.info(f"⏳ Шаг 3: Сохранение результатов в MongoDB...")
        try:
            # Создать batch_id на основе job_id
            batch_id = f"batch_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            folder_name = Path(folder_path).name

            saved_batch_id = await aggregator.process_and_save_batch(
                batch_id=batch_id, folder_name=folder_name, wagon_results=wagon_results
            )

            if not saved_batch_id:
                error = "Ошибка при сохранении результатов в БД"
                logger.error(f"❌ {error}")
                JobManager.fail_job(job_id, error)
                return

            logger.info(f"✅ Результаты сохранены в MongoDB")
            JobManager.update_progress(job_id, len(image_files), 95)

        except Exception as e:
            error = f"Ошибка сохранения результатов: {str(e)}"
            logger.error(f"❌ {error}")
            JobManager.fail_job(job_id, error)
            return

        # ============ STEP 4: Подготовить результаты ============
        result = {
            "status": "success",
            "batch_id": batch_id,
            "folder": folder_name,
            "wagons_count": len(wagon_results),
            "processed_files": len(image_files),
            "results": wagon_results,
            "camera_id": camera_id,
            "wagon_id": wagon_id,
        }

        logger.info(f"✅ Результаты готовы:")
        logger.info(f"   Batch ID: {batch_id}")
        logger.info(f"   Вагонов: {len(wagon_results)}")
        logger.info(f"   Файлов: {len(image_files)}")

        # ============ STEP 5: Очистка временных файлов ============
        logger.info(f"🗑️ Очистка временных файлов...")
        UploadHandler.cleanup_job_files(job_id)

        # ============ STEP 6: Завершить задание ============
        JobManager.complete_job(job_id, result)
        logger.info(f"✅ Задание {job_id} завершено успешно")

    except Exception as e:
        error = f"Неожиданная ошибка при обработке: {type(e).__name__}: {str(e)}"
        logger.error(f"❌ {error}")
        logger.exception(f"Traceback: {e}")
        JobManager.fail_job(job_id, error)

        # Попытка очистки
        try:
            if temp_dir:
                UploadHandler.cleanup_job_files(job_id)
        except:
            pass

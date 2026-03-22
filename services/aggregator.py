# (aggregator)
"""
Микросервис для агрегации и управления результатами в MongoDB

Отвечает за:
- Сохранение итоговых результатов в BatchDocument
- Запросы к базе данных для получения результатов
- Управление батчами обработки
"""

import logging
from typing import Dict, Any, Optional, List
from datetime import datetime

from models.schemas import BatchDocument
from db.repository import ensure_db_connection

logger = logging.getLogger(__name__)


async def process_and_save_batch(
    batch_id: str,
    folder_name: str,
    wagon_results: Dict[str, Dict[str, Any]]
) -> Optional[str]:
    logger.info(f"💾 Сохранение полных результатов батча {batch_id} в MongoDB")
    
    db_ok = await ensure_db_connection()
    if not db_ok:
        logger.error(f"❌ Не удалось подключиться к MongoDB")
        return None
    
    try:
        # Рассчитать общую статистику
        total_photos = sum(r["total_photos"] for r in wagon_results.values())
        total_wagons = len(wagon_results)
        processed_photos = sum(r["processed_photos"] for r in wagon_results.values())
        
        logger.info(f"   📊 Статистика:")
        logger.info(f"      Вагонов: {total_wagons}")
        logger.info(f"      Всего фото: {total_photos}")
        logger.info(f"      Обработано: {processed_photos}")
        
        # Создать документ батча
        batch_doc = BatchDocument(
            batch_id=batch_id,
            folder=folder_name,
            results=wagon_results,  # Сохран все результаты прямо в поле results
            total_photos=total_photos,
            total_wagons=total_wagons,
            processed_photos=processed_photos,
            status="completed",
            error_message=None,
            processed_at=datetime.now(),
            created_at=datetime.now()
        )
        
        await batch_doc.insert()
        logger.info(f"✅ BatchDocument сохранен в MongoDB: {batch_id}")
        return batch_id
        
    except Exception as e:
        logger.error(f"❌ Ошибка сохранения BatchDocument: {type(e).__name__}: {str(e)}")
        return None


async def get_batch_status(batch_id: str) -> Dict[str, Any]:
    logger.debug(f"🔍 Получение статуса батча {batch_id}")
    
    db_ok = await ensure_db_connection()
    if not db_ok:
        return {
            "batch_id": batch_id,
            "status": "db_unavailable"
        }
    
    try:
        batch_doc = await BatchDocument.find_one(
            BatchDocument.batch_id == batch_id
        )
        
        if not batch_doc:
            return {
                "batch_id": batch_id,
                "status": "not_found"
            }
        
        return {
            "batch_id": batch_id,
            "folder": batch_doc.folder,
            "status": batch_doc.status,
            "total_photos": batch_doc.total_photos,
            "processed_photos": batch_doc.processed_photos,
            "total_wagons": batch_doc.total_wagons,
            "error_message": batch_doc.error_message
        }
        
    except Exception as e:
        logger.error(f"❌ Ошибка при получении статуса батча {batch_id}: {type(e).__name__}: {str(e)}")
        return {
            "batch_id": batch_id,
            "status": "error"
        }


async def get_batch_results(batch_id: str) -> Dict[str, Any]:

    logger.debug(f"🔍 Получение результатов батча {batch_id}")
    
    db_ok = await ensure_db_connection()
    if not db_ok:
        return {
            "batch_id": batch_id,
            "results": {},
            "status": "db_unavailable"
        }
    
    try:
        batch_doc = await BatchDocument.find_one(
            BatchDocument.batch_id == batch_id
        )
        
        if not batch_doc:
            return {
                "batch_id": batch_id,
                "results": {},
                "status": "not_found"
            }
        
        return {
            "batch_id": batch_id,
            "folder": batch_doc.folder,
            "total_photos": batch_doc.total_photos,
            "total_wagons": batch_doc.total_wagons,
            "processed_photos": batch_doc.processed_photos,
            "results": batch_doc.results,
            "status": batch_doc.status,
            "processed_at": batch_doc.processed_at
        }
        
    except Exception as e:
        logger.error(f"❌ Ошибка при получении результатов батча {batch_id}: {type(e).__name__}: {str(e)}")
        return {
            "batch_id": batch_id,
            "results": {},
            "status": "error"
        }


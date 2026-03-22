import logging
from datetime import datetime
from typing import Dict, Any
from core.config import settings
from models.schemas import (
    PhotoDocument,
    WagonAggregateDocument,
    FinalVerdictModel,
    BatchDocument
)
from beanie import init_beanie
from motor.motor_asyncio import AsyncIOMotorClient

logger = logging.getLogger(__name__)

async def ensure_db_connection() -> bool:
    try:
        logger.debug(f"🔍 Проверка подключения MongoDB...")
        
        # Пытались подключиться и проверь наличие коллекций
        client = AsyncIOMotorClient(settings.mongodb_url)
        db = client[settings.database_name]
        
        # Проверь можно ли мы листить коллекции (простой тест подключения)
        collections = await db.list_collection_names()
        logger.debug(f"   ✅ MongoDB нодключен, найдено {len(collections)} коллекций")
        
        # Инициализируем Beanie если еще не сделано
        try:
            await init_beanie(
                database=db,
                document_models=[PhotoDocument, WagonAggregateDocument, BatchDocument]
            )
            logger.info(f"✅ MongoDB инициализирована с Beanie")
        except RuntimeError as e:
            # Beanie уже инициализирована
            if "init_beanie" in str(e) or "already" in str(e):
                logger.debug(f"   ℹ️ Beanie уже инициализирована")
            else:
                raise
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Ошибка при проверке подключения БД: {type(e).__name__}: {str(e)}")
        return False

async def get_wagon_status(wagon_id: str) -> Dict[str, Any]:
    wagon_agg = await WagonAggregateDocument.find_one(
        WagonAggregateDocument.wagon_id == wagon_id
    )
    
    if not wagon_agg:
        return {
            "wagon_id": wagon_id,
            "total_photos": 0,
            "accepted_photos": 0,
            "rejected_photos": 0,
            "processing_status": "no_data"
        }
    
    return {
        "wagon_id": wagon_id,
        "total_photos": wagon_agg.total_photos,
        "accepted_photos": wagon_agg.total_photos,
        "rejected_photos": 0,
        "processing_status": "completed" if wagon_agg.final_side else "in_progress"
    }


async def get_wagon_result(wagon_id: str) -> Dict[str, Any]:
    wagon_agg = await WagonAggregateDocument.find_one(
        WagonAggregateDocument.wagon_id == wagon_id
    )
    
    if not wagon_agg:
        return {
            "wagon_id": wagon_id,
            "final_verdict": None,
            "camera_ids": [],
            "total_photos": 0,
            "status": "no_data"
        }
    
    verdict = None
    if wagon_agg.final_side:
        verdict = FinalVerdictModel(
            side=wagon_agg.final_side,
            left_count=wagon_agg.left_count,
            right_count=wagon_agg.right_count,
            total_photos=wagon_agg.total_photos
        )
    
    return {
        "wagon_id": wagon_id,
        "final_verdict": verdict,
        "camera_ids": wagon_agg.camera_ids,
        "total_photos": wagon_agg.total_photos,
        "status": "completed" if verdict else "processing"
    }


# ==================== Операции запросов батча ====================

async def get_batch_status(batch_id: str) -> Dict[str, Any]:
    batch_doc = await BatchDocument.find_one(BatchDocument.batch_id == batch_id)
    
    if not batch_doc:
        return {
            "batch_id": batch_id,
            "folder": "",
            "status": "not_found",
            "total_photos": 0,
            "processed_photos": 0,
            "total_wagons": 0,
            "message": f"Batch {batch_id} not found"
        }
    
    return {
        "batch_id": batch_id,
        "folder": batch_doc.folder,
        "status": batch_doc.status,
        "total_photos": batch_doc.total_photos,
        "processed_photos": batch_doc.processed_photos,
        "total_wagons": batch_doc.total_wagons,
        "message": batch_doc.error_message or f"Batch {'completed' if batch_doc.status == 'completed' else 'processing'}"
    }


async def get_batch_result(batch_id: str) -> Dict[str, Any]:
    batch_doc = await BatchDocument.find_one(BatchDocument.batch_id == batch_id)
    
    if not batch_doc:
        return {
            "batch_id": batch_id,
            "folder": "",
            "total_wagons": 0,
            "total_photos": 0,
            "results": {},
            "processed_at": datetime.now(),
            "status": "not_found"
        }
    
    return {
        "batch_id": batch_id,
        "folder": batch_doc.folder,
        "total_wagons": batch_doc.total_wagons,
        "total_photos": batch_doc.total_photos,
        "results": batch_doc.results,
        "processed_at": batch_doc.processed_at,
        "status": batch_doc.status
    }

"""
Микросервис для управления хранилищем фотографий (photo_aggregate)

Отвечает за:
- Организацию структуры папок для фотографий
- Поиск и получение фотографий по вагонам
- Отслеживание метаданных файлов
- Уборку старых файлов
"""

import logging
import shutil
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime

from core.constants import AGGREGATE_OUTPUT_DIR

logger = logging.getLogger(__name__)


def get_aggregate_dir() -> Path:
    """Получить путь к директории photo_aggregate"""
    return Path(AGGREGATE_OUTPUT_DIR)


def ensure_aggregate_dir() -> Path:
    """
    Убедиться, что директория photo_aggregate существует.
    
    Returns:
        Path к директории photo_aggregate
    """
    agg_dir = get_aggregate_dir()
    try:
        agg_dir.mkdir(parents=True, exist_ok=True)
        logger.debug(f"✅ Директория photo_aggregate готова: {agg_dir}")
        return agg_dir
    except Exception as e:
        logger.error(f"❌ Ошибка создания photo_aggregate: {type(e).__name__}: {str(e)}")
        raise


def get_wagon_dir(wagon_id: str) -> Path:
    """
    Получить путь к директории конкретного вагона.
    
    Args:
        wagon_id: ID вагона
        
    Returns:
        Path к директории вагона
    """
    agg_dir = ensure_aggregate_dir()
    return agg_dir / wagon_id


def list_wagons() -> List[str]:
    """
    Получить список всех вагонов в photo_aggregate.
    
    Returns:
        List wagon_ids
    """
    logger.debug(f"🔍 Поиск вагонов в photo_aggregate")
    
    try:
        agg_dir = get_aggregate_dir()
        if not agg_dir.exists():
            logger.debug(f"⚠️ photo_aggregate не существует")
            return []
        
        wagons = [d.name for d in agg_dir.iterdir() if d.is_dir()]
        logger.info(f"✅ Найдено {len(wagons)} вагонов")
        return sorted(wagons)
        
    except Exception as e:
        logger.error(f"❌ Ошибка при поиске вагонов: {type(e).__name__}: {str(e)}")
        return []


def get_wagon_photos(wagon_id: str) -> List[Dict[str, Any]]:
    """
    Получить список фотографий для конкретного вагона.
    
    Args:
        wagon_id: ID вагона
        
    Returns:
        List с информацией о фотографиях
    """
    logger.debug(f"🔍 Получение фотографий для {wagon_id}")
    
    try:
        wagon_dir = get_wagon_dir(wagon_id)
        
        if not wagon_dir.exists():
            logger.warning(f"⚠️ Директория вагона не найдена: {wagon_dir}")
            return []
        
        photos = []
        for photo_path in sorted(wagon_dir.glob("*.[jp][pn]g")):
            stat = photo_path.stat()
            photos.append({
                "filename": photo_path.name,
                "path": str(photo_path),
                "size_bytes": stat.st_size,
                "modified_time": datetime.fromtimestamp(stat.st_mtime),
                "relative_path": f"{wagon_id}/{photo_path.name}"
            })
        
        logger.info(f"✅ Найдено {len(photos)} фотографий в {wagon_id}")
        return photos
        
    except Exception as e:
        logger.error(f"❌ Ошибка при получении фотографий {wagon_id}: {type(e).__name__}: {str(e)}")
        return []


def get_wagon_stats(wagon_id: str) -> Dict[str, Any]:
    """
    Получить статистику по фотографиям вагона.
    
    Args:
        wagon_id: ID вагона
        
    Returns:
        Dict со статистикой
    """
    logger.debug(f"📊 Получение статистики для {wagon_id}")
    
    try:
        wagon_dir = get_wagon_dir(wagon_id)
        
        if not wagon_dir.exists():
            return {
                "wagon_id": wagon_id,
                "exists": False,
                "photo_count": 0,
                "total_size_mb": 0.0
            }
        
        photos = list(wagon_dir.glob("*.[jp][pn]g"))
        total_size = sum(p.stat().st_size for p in photos)
        
        return {
            "wagon_id": wagon_id,
            "exists": True,
            "photo_count": len(photos),
            "total_size_mb": round(total_size / (1024 * 1024), 2),
            "directory": str(wagon_dir)
        }
        
    except Exception as e:
        logger.error(f"❌ Ошибка при получении статистики {wagon_id}: {type(e).__name__}: {str(e)}")
        return {
            "wagon_id": wagon_id,
            "error": str(e)
        }


def get_all_stats() -> Dict[str, Any]:
    """
    Получить общую статистику по photo_aggregate.
    
    Returns:
        Dict с общей статистикой
    """
    logger.debug(f"📊 Получение общей статистики photo_aggregate")
    
    try:
        wagons = list_wagons()
        
        total_photos = 0
        total_size = 0
        wagon_stats = []
        
        for wagon_id in wagons:
            stats = get_wagon_stats(wagon_id)
            if stats.get("exists"):
                total_photos += stats.get("photo_count", 0)
                total_size += stats.get("total_size_mb", 0) * (1024 * 1024)
                wagon_stats.append(stats)
        
        return {
            "total_wagons": len(wagons),
            "total_photos": total_photos,
            "total_size_mb": round(total_size / (1024 * 1024), 2),
            "wagons": wagon_stats
        }
        
    except Exception as e:
        logger.error(f"❌ Ошибка при получении общей статистики: {type(e).__name__}: {str(e)}")
        return {
            "error": str(e)
        }


def copy_photo_to_wagon(
    source_path: str,
    wagon_id: str,
    filename: Optional[str] = None
) -> bool:
    """
    Скопировать фотографию в директорию вагона.
    
    Args:
        source_path: Полный путь к исходному файлу
        wagon_id: ID вагона назначения
        filename: Новое имя файла (опционально)
        
    Returns:
        bool: True если успешно, False если ошибка
    """
    logger.debug(f"📋 Копирование фотографии в {wagon_id}")
    
    try:
        source = Path(source_path)
        if not source.exists():
            logger.error(f"❌ Исходный файл не найден: {source_path}")
            return False
        
        wagon_dir = get_wagon_dir(wagon_id)
        wagon_dir.mkdir(parents=True, exist_ok=True)
        
        # Использовать переданное имя или исходное имя файла
        target_filename = filename or source.name
        target_path = wagon_dir / target_filename
        
        shutil.copy2(source, target_path)
        logger.info(f"   ✅ Скопировано в {wagon_dir / target_filename}")
        return True
        
    except Exception as e:
        logger.error(f"❌ Ошибка копирования файла: {type(e).__name__}: {str(e)}")
        return False


def delete_wagon(wagon_id: str) -> bool:
    """
    Удалить директорию вагона со всеми фотографиями.
    
    Args:
        wagon_id: ID вагона
        
    Returns:
        bool: True если успешно, False если ошибка
    """
    logger.warning(f"🗑️ Удаление вагона {wagon_id}")
    
    try:
        wagon_dir = get_wagon_dir(wagon_id)
        
        if not wagon_dir.exists():
            logger.warning(f"⚠️ Директория вагона не найдена: {wagon_dir}")
            return False
        
        shutil.rmtree(wagon_dir)
        logger.info(f"✅ Удален вагон {wagon_id}")
        return True
        
    except Exception as e:
        logger.error(f"❌ Ошибка при удалении вагона {wagon_id}: {type(e).__name__}: {str(e)}")
        return False


def cleanup_old_wagons(days: int = 30) -> Dict[str, Any]:
    """
    Очистить старые вагоны (не обновлялись дольше чем days дней).
    
    Args:
        days: Количество дней для определения "старых" файлов
        
    Returns:
        Dict с информацией об удаленных вагонах
    """
    logger.warning(f"🧹 Очистка wagons старше {days} дней")
    
    from datetime import timedelta
    import time
    
    try:
        agg_dir = get_aggregate_dir()
        if not agg_dir.exists():
            return {"deleted": [], "total_wagons": 0}
        
        current_time = datetime.now().timestamp()
        threshold = timedelta(days=days).total_seconds()
        
        deleted_wagons = []
        
        for wagon_dir in agg_dir.iterdir():
            if not wagon_dir.is_dir():
                continue
            
            # Получить время последнего изменения директории
            dir_mtime = wagon_dir.stat().st_mtime
            age = current_time - dir_mtime
            
            if age > threshold:
                wagon_id = wagon_dir.name
                shutil.rmtree(wagon_dir)
                deleted_wagons.append(wagon_id)
                logger.info(f"   ✅ Удален: {wagon_id} (возраст: {age/86400:.1f} дней)")
        
        logger.info(f"✅ Удалено {len(deleted_wagons)} старых wagons")
        
        return {
            "deleted": deleted_wagons,
            "total_deleted": len(deleted_wagons)
        }
        
    except Exception as e:
        logger.error(f"❌ Ошибка при очистке: {type(e).__name__}: {str(e)}")
        return {
            "error": str(e)
        }


def get_photo_path(wagon_id: str, filename: str) -> Optional[Path]:
    """
    Получить полный путь к фотографии.
    
    Args:
        wagon_id: ID вагона
        filename: Имя файла
        
    Returns:
        Path если файл существует, None если нет
    """
    logger.debug(f"🔍 Получение пути для {wagon_id}/{filename}")
    
    try:
        wagon_dir = get_wagon_dir(wagon_id)
        photo_path = wagon_dir / filename
        
        if photo_path.exists():
            return photo_path
        
        logger.warning(f"⚠️ Фотография не найдена: {photo_path}")
        return None
        
    except Exception as e:
        logger.error(f"❌ Ошибка при получении пути: {type(e).__name__}: {str(e)}")
        return None

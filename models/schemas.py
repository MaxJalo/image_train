# (schemas)
from datetime import datetime
from typing import Any, Dict, List, Literal, Optional

from beanie import Document
from pydantic import BaseModel, Field

# ==================== ML Model Output Schemas ====================


class Model1Output(BaseModel):
    """Результат модели фильтрации (Model-1)"""

    is_valid: bool = Field(..., description="Валидна ли фотография для классификации")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Оценка доверия (0-1)")


class Model2Output(BaseModel):
    """Результат модели классификации (Model-2)"""

    brake_rod: float = Field(
        ..., ge=0.0, le=1.0, description="Уверенность обнаружения тормозной штанги"
    )
    rod_nose: float = Field(
        ..., ge=0.0, le=1.0, description="Уверенность обнаружения носика штока"
    )
    crane: float = Field(
        ..., ge=0.0, le=1.0, description="Уверенность обнаружения крана"
    )
    tank: float = Field(..., ge=0.0, le=1.0, description="Уверенность обнаружения бака")
    side: Literal["left", "right"] = Field(
        ..., description="Сторона вагона (левая или правая)"
    )
    confidence: float = Field(..., ge=0.0, le=1.0, description="Общая оценка доверия")


# ==================== Pydantic Request/Response Models ====================


# Single Photo Upload (legacy)
class PhotoUploadRequest(BaseModel):
    """Модель запроса для загрузки фотографии"""

    wagon_id: str = Field(..., description="Unique wagon identifier")
    camera_id: int = Field(..., description="Camera identifier")


class PhotoUploadResponse(BaseModel):
    """Модель ответа при загрузке фотографии"""

    photo_id: str
    wagon_id: str
    camera_id: int
    status: str  # "accepted" or "rejected"
    message: str
    processed_at: datetime


class WagonStatusResponse(BaseModel):
    """Модель ответа для статуса обработки вагона"""

    wagon_id: str
    total_photos: int
    accepted_photos: int
    rejected_photos: int
    processing_status: str  # "in_progress" или "completed"


class FinalVerdictModel(BaseModel):
    """Финальное агрегированное решение для вагона"""

    side: str  # "left" or "right"
    left_count: int
    right_count: int
    total_photos: int


class WagonResultResponse(BaseModel):
    """Модель ответа для финального результата вагона"""

    wagon_id: str
    final_verdict: Optional[FinalVerdictModel] = None
    camera_ids: List[int]
    total_photos: int
    status: str  # "no_data", "processing", "completed"


# Batch Folder Upload
class FolderUploadRequest(BaseModel):
    """Модель запроса для загрузки папки c картографированием файлов в вагоны"""

    folder_path: str = Field(..., description="Путь к папке c фотографиями")
    mapping: Optional[Dict[str, Dict[str, str]]] = Field(
        None, description="Картографирование camera_id -> {file_hash: wagon_id}"
    )


class SinglePhotoUploadRequest(BaseModel):
    """Модель запроса для загрузки одной фотографии"""

    wagon_id: str = Field(..., description="Unique wagon identifier")
    camera_id: int = Field(..., description="Camera identifier")


class WagonBatchResult(BaseModel):
    """Результат для одного вагона в батче"""

    final_side: str
    left_count: int
    right_count: int
    cameras: List[int]
    photos_processed: int


class BatchResultResponse(BaseModel):
    """Модель ответа для результата пакетной обработки"""

    batch_id: str
    folder: str
    total_wagons: int
    total_photos: int
    results: Dict[str, WagonBatchResult]
    processed_at: datetime
    status: str  # "completed", "in_progress", "error"
    session_dir: Optional[str] = None  # Путь к папке session с принятыми фото
    summary: Optional[Dict[str, Any]] = None  # Статистика обработки и отбраковки


class BatchStatusResponse(BaseModel):
    """Модель ответа для статуса пакетной обработки"""

    batch_id: str
    folder: str
    status: str  # "processing", "completed", "error"
    total_photos: int
    processed_photos: int
    total_wagons: int
    message: str


# ==================== MongoDB Beanie Models ====================


class PhotoDocument(Document):
    """Документ для хранения данных обработанной фотографии"""

    wagon_id: str
    file_hash: str  # Исходный путь файла (например, из имени файла или содержимого)
    camera_id: int
    side: str  # "left" или "right"
    confidence: float
    features: Dict[str, Any]  # brake_rod, rod_nose, crane, tank и т.д.
    processed_at: datetime
    batch_id: Optional[str] = None  # Ссылка на батч

    # Результаты Model-1 (фильтрация)
    model1_result: Optional[bool] = (
        None  # True если прошла фильтр, False если отбракована
    )
    model1_confidence: Optional[float] = None  # Уверенность Model-1

    # Поля финального вердикта (заполняются после агрегации)
    final_verdict: Optional[FinalVerdictModel] = None

    class Settings:
        name = "photos"
        indexes = [
            "wagon_id",
            "camera_id",
            "batch_id",
            "file_hash",
            ("wagon_id", "camera_id"),
        ]


class WagonAggregateDocument(Document):
    """Документ для хранения агрегированных данных вагона"""

    wagon_id: str
    photos: List[str]  # Список ID фотодокументов
    left_count: int = 0
    right_count: int = 0
    total_photos: int = 0
    final_side: Optional[str] = None  # "left", "right" или None
    camera_ids: List[int] = Field(default_factory=list)
    batch_id: Optional[str] = None  # Ссылка на батч
    created_at: datetime
    updated_at: datetime

    class Settings:
        name = "wagon_aggregates"
        indexes = ["wagon_id", "batch_id"]


class BatchDocument(Document):
    """Документ для хранения результатов обработки батча папок"""

    batch_id: str
    folder: str
    results: Dict[
        str, Dict[str, Any]
    ]  # wagon_id -> {final_side, left_count, right_count, cameras, photos_processed}
    total_photos: int = 0
    total_wagons: int = 0
    processed_photos: int = 0
    status: str  # "processing", "completed", "error"
    error_message: Optional[str] = None
    processed_at: datetime
    created_at: datetime

    class Settings:
        name = "batches"
        indexes = ["batch_id", "folder"]

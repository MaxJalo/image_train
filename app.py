
import logging
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

from core.config import Settings
from routes import wagon, health, debug
from services.model_loader import load_model

# Инициализация логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Инициализация конфигурации
settings = Settings()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Управление жизненным циклом приложения"""
    # Startup
    logger.info("🚀 Запуск приложения...")
    try:
        load_model(settings.model1_path)
        load_model(settings.model2_path)
        logger.info("✅ Модели загружены успешно")
    except Exception as e:
        logger.error(f"❌ Ошибка при загрузке моделей: {str(e)}")
        raise
    
    yield
    
    # Shutdown
    logger.info("🛑 Завершение приложения...")


# Создание приложения
app = FastAPI(
    title=settings.api_title,
    version=settings.api_version,
    description="API для обработки фотографий вагонов с использованием ML моделей",
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Подключение маршрутов
app.include_router(health.router)
app.include_router(wagon.router)
app.include_router(debug.router)


@app.get("/")
async def root():
    """Корневой эндпоинт"""
    return {
        "app_title": settings.api_title,
        "version": settings.api_version,
        "status": "running",
        "documentation": "/docs"
    }


if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info"
    )

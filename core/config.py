from pydantic_settings import BaseSettings, SettingsConfigDict
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

# Find .env file in project root or microservise directory
_config_dir = Path(__file__).parent.parent  # microservise directory
_project_root = _config_dir.parent  # ml_project directory
_env_files = [
    _config_dir / ".env",  # /home/max_jalo/ml_project/microservise/.env (check first)
    _project_root / ".env",  # /home/max_jalo/ml_project/.env
    Path.cwd() / ".env"  # Current working directory
]
_env_file = next((f for f in _env_files if f.exists()), _env_files[0])


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=str(_env_file),
        env_file_encoding='utf-8',
        case_sensitive=False,
        protected_namespaces=('settings_',),
    )
    
    # MongoDB — с значениями по умолчанию
    mongodb_url: str = "mongodb://root:example@mongodb:27017"
    database_name: str = "wagon_db" 
    
    # API
    api_title: str = "Wagon ML API" 
    api_version: str = "1.0.0" 
    debug: bool = False
    
    # Загрузка файлов
    max_upload_size: int = 52428800
    
    # ML-модели — можно сделать необязательными с заглушками
    model1_path: str = "./NN_models/model-1.pt"
    model2_path: str = "./NN_models/model-2.pt" 
    model_timeout: int = 30 

    # Сохранение фотографий от Model-1
    output_model1_path: str = "./result"
    
    # Логирование
    log_level: str = "INFO"
    
    def validate_model_paths(self) -> dict:
        results = {
            "valid": True,
            "model1": {"path": self.model1_path, "exists": False, "readable": False},
            "model2": {"path": self.model2_path, "exists": False, "readable": False}
        }
        
        for model_name, model_path in [("model1", self.model1_path), ("model2", self.model2_path)]:
            path = Path(model_path)
            logger.debug(f"Проверка пути {model_name}: {path.resolve()}")
            
            if path.exists():
                results[model_name]["exists"] = True
                try:
                    with open(path, 'r') as f:
                        pass
                    results[model_name]["readable"] = True
                    logger.info(f"✅ {model_name} найден: {path.resolve()} ({path.stat().st_size} bytes)")
                except PermissionError:
                    logger.warning(f"⚠️ {model_name} существует, но недоступен: {path.resolve()}")
                    results["valid"] = False
            else:
                logger.warning(f"⚠️ {model_name} не найден: {path.resolve()}")
                results["valid"] = False
        
        return results


settings = Settings()

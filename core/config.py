from pydantic import Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict
from pathlib import Path
from typing import Literal
import logging

logger = logging.getLogger(__name__)

class Settings(BaseSettings):    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding='utf-8',
        case_sensitive=False,
        protected_namespaces=('settings_',),
        extra='ignore',
    )
    

    app_env: Literal["dev", "staging", "production"] = Field(
        default="dev", 
        env="APP_ENV"
    )
    

    mongodb_user: str = Field(default="root", env="MONGODB_USER")
    mongodb_password: SecretStr = Field(..., env="MONGODB_PASSWORD")
    mongodb_host: str = Field(default="localhost", env="MONGODB_HOST")
    mongodb_port: int = Field(default=27017, env="MONGODB_PORT")
    database_name: str = Field(default="wagon_db", env="DATABASE_NAME")
    

    api_title: str = Field(default="Wagon ML API", env="API_TITLE")
    api_version: str = Field(default="1.0.0", env="API_VERSION")
    debug: bool = Field(default=False, env="DEBUG")
    

    max_upload_size: int = Field(
        default=52428800, 
        le=104857600,
        env="MAX_UPLOAD_SIZE"
    )
    allowed_origins: list[str] = Field(
        default=["http://localhost:3000"],
        env="ALLOWED_ORIGINS"
    )
    

    model1_path: Path = Field(default="./NN_models/model-1.pt", env="MODEL1_PATH")
    model2_path: Path = Field(default="./NN_models/model-2.pt", env="MODEL2_PATH")
    model_timeout: int = Field(default=30, ge=5, le=300, env="MODEL_TIMEOUT")
    

    output_model1_path: Path = Field(default="./result", env="OUTPUT_MODEL1_PATH")
    

    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = Field(
        default="INFO", 
        env="LOG_LEVEL"
    )
    
    @property
    def mongodb_url(self) -> str:
        password = self.mongodb_password.get_secret_value()
        logger.debug(f"Connecting to MongoDB at {self.mongodb_host}:{self.mongodb_port}")
        return f"mongodb://{self.mongodb_user}:{password}@{self.mongodb_host}:{self.mongodb_port}"
    
    @field_validator("model1_path", "model2_path", mode="before")
    @classmethod
    def resolve_paths(cls, v: str | Path) -> Path:
        path = Path(v)
        if not path.is_absolute():
            project_root = Path(__file__).parent.parent
            path = project_root / path
        return path.resolve()
    
    @field_validator("allowed_origins", mode="before")
    @classmethod
    def parse_allowed_origins(cls, v: str | list) -> list:
        if isinstance(v, str):
            return [origin.strip() for origin in v.split(",")]
        return v
    
    def model_post_init(self, __context):
        for model_path in [self.model1_path, self.model2_path]:
            if not model_path.exists():
                logger.warning(f"⚠️ Model not found: {model_path}")
            else:
                logger.info(f"✅ Model found: {model_path} ({model_path.stat().st_size} bytes)")
        
        if self.app_env == "production" and self.debug:
            raise ValueError("DEBUG mode cannot be True in production!")
        
        if self.app_env == "production" and Path(".env").exists():
            logger.warning("⚠️ .env file exists in production - not recommended!")

try:
    settings = Settings()
    logger.info(f"✅ Config loaded successfully (env: {settings.app_env})")
except Exception as e:
    logger.error(f"❌ Failed to load config: {e}")
    raise

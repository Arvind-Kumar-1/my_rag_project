# app/config.py
from pydantic_settings import BaseSettings, SettingsConfigDict
from pathlib import Path
import os

# Get the path to the .env file
env_path = Path(__file__).parent.parent / ".env"

class Settings(BaseSettings):
    # Authentication
    AUTH_TOKEN: str = "default-token-for-development"
    
    # Model Configuration
    EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"
    LLM_MODEL: str = "llama3"
    
    # Processing Configuration
    CHUNK_SIZE: int = 512
    CHUNK_OVERLAP: int = 50
    MAX_CHUNKS_PER_QUERY: int = 5
    
    # Performance Settings
    REQUEST_TIMEOUT: int = 30
    MAX_FILE_SIZE_MB: int = 50
    
    # LLM Settings
    LLM_TEMPERATURE: float = 0.1
    LLM_TOP_P: float = 0.9
    LLM_MAX_TOKENS: int = 1000
    
    # Ollama Configuration
    OLLAMA_BASE_URL: str = "http://localhost:11434"
    
    # Vector Search Configuration
    FAISS_INDEX_TYPE: str = "IndexFlatL2"  # or "IndexHNSWFlat" for larger datasets
    
    # Logging Configuration
    LOG_LEVEL: str = "INFO"
    LOG_FORMAT: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    # Development/Production flags
    DEBUG: bool = False
    ENVIRONMENT: str = "development"
    
    # CORS settings
    CORS_ORIGINS: list = ["*"]
    
    model_config = SettingsConfigDict(
        env_file=env_path,
        env_file_encoding='utf-8',
        case_sensitive=True
    )
    
    @property
    def is_development(self) -> bool:
        return self.ENVIRONMENT.lower() in ["development", "dev", "local"]
    
    @property
    def is_production(self) -> bool:
        return self.ENVIRONMENT.lower() in ["production", "prod"]

# Global settings instance
settings = Settings()

# Logging configuration
import logging

def setup_logging():
    """Setup logging configuration"""
    logging.basicConfig(
        level=getattr(logging, settings.LOG_LEVEL.upper()),
        format=settings.LOG_FORMAT
    )
    
    # Suppress some verbose libraries in production
    if settings.is_production:
        logging.getLogger("transformers").setLevel(logging.WARNING)
        logging.getLogger("sentence_transformers").setLevel(logging.WARNING)
        logging.getLogger("torch").setLevel(logging.WARNING)

# Initialize logging
setup_logging()
logger = logging.getLogger(__name__)
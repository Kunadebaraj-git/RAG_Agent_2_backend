"""DocMind AI - Configuration"""

from pydantic_settings import BaseSettings
from functools import lru_cache


class Settings(BaseSettings):
    # Database
    DATABASE_URL: str = ""

    # Auth
    JWT_SECRET: str = ""
    SECRET_KEY: str = ""
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 60

    # LLM
    GROQ_API_KEY: str = ""
    HF_API_TOKEN: str = ""

    # Chunking
    CHUNK_SIZE: int = 800
    CHUNK_OVERLAP: int = 150

    # Retrieval
    RETRIEVAL_TOP_K: int = 5
    SIMILARITY_THRESHOLD: float = 0.3

    # Embedding
    HF_EMBEDDING_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"
    HF_EMBEDDING_DIM: int = 384

    # CORS
    FRONTEND_URL: str = "http://localhost:5173"

    model_config = {"env_file": ".env", "extra": "ignore"}

    def validate_required_settings(self) -> None:
        """Ensure all required environment variables are non-empty at startup."""
        # Sync JWT_SECRET and SECRET_KEY
        if self.JWT_SECRET and not self.SECRET_KEY:
            self.SECRET_KEY = self.JWT_SECRET
        elif self.SECRET_KEY and not self.JWT_SECRET:
            self.JWT_SECRET = self.SECRET_KEY

        required_map = {
            "DATABASE_URL": self.DATABASE_URL,
            "GROQ_API_KEY": self.GROQ_API_KEY,
            "HF_API_TOKEN": self.HF_API_TOKEN,
            "JWT_SECRET": self.JWT_SECRET or self.SECRET_KEY,
        }

        for name, value in required_map.items():
            if not value or not str(value).strip():
                raise RuntimeError(
                    f"Configuration Error: Missing required environment variable '{name}'. "
                    f"Please run 'python scripts/setup_env.py' to set up your environment."
                )


@lru_cache
def get_settings() -> Settings:
    settings = Settings()
    settings.validate_required_settings()
    return settings


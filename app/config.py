from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field, HttpUrl, SecretStr, ValidationError
from typing import Optional, Literal
import logging

logger = logging.getLogger(__name__)


class CriticalConfigError(Exception):
    """Custom exception for critical configuration failures."""
    pass


class AppSettings(BaseSettings):
    # spaCy
    SPACY_MODEL: str = Field("en_core_web_sm", env="SPACY_MODEL")

    # -- LLM provider selection --
    LLM_PROVIDER: Literal[
        'openai', 'pinecone', 'ollama', 'openrouter', 'groq'] = Field(
        default='ollama',
        env='LLM_PROVIDER',
        description="Which LLM provider to use: openai, ollama, openrouter, or groq."
    )
    LLM_TIMEOUT: int = Field(120, env="LLM_TIMEOUT")

    # -- Embedding provider selection --
    EMBEDDING_PROVIDER: Literal['openai', 'pinecone', 'ollama'] = Field(
        default='pinecone',
        env='EMBEDDING_PROVIDER',
        description="Which embedding provider to use: openai, pinecone, or ollama."
    )
    EMBEDDING_TIMEOUT: int = Field(30, env="EMBEDDING_TIMEOUT")

    # Contextual Retrieval
    ENABLE_CONTEXTUAL_RETRIEVAL: Optional[bool] = Field(default=True, description="Enable Contextual Retrieval.")
    OLLAMA_CONTEXTUAL_LLM_BASE_URL: Optional[HttpUrl] = Field(default='http://localhost:11434',
                                                   description="Ollama LLM base URL for contextual retrieval.")
    OLLAMA_CONTEXTUAL_LLM_MODEL: Optional[str] = Field(default=None, description="Ollama LLM model for contextual retrieval.")

    # Groq
    GROQ_API_KEY: Optional[SecretStr] = Field(default=None, description="Your Groq API Key.")
    GROQ_LLM_MODEL: Optional[str] = Field(default="llama3-70b-8192", description="The Groq LLM model to be used.")

    # Ollama
    OLLAMA_LLM_BASE_URL: Optional[HttpUrl] = Field(default='http://localhost:11434',
                                                   description="Ollama LLM base URL if self-hosted and not using the default.")
    OLLAMA_LLM_MODEL: Optional[str] = Field(default=None, description="Ollama LLM model.")

    OLLAMA_EMBEDDING_BASE_URL: str = Field(default='http://localhost:11434', env='EMBEDDING_BASE_URL', description='Ollama Embedding base URL if self-hosted and not using the default.')
    OLLAMA_EMBEDDING_MODEL: Optional[str] = Field(default="mxbai-embed-large",
                                                  description="The Ollama embedding model to be used.")

    # Vector store (Chroma)
    CHROMA_PERSIST_DIR: str = Field("./data/vector_db", env="CHROMA_PERSIST_DIR")
    CHROMA_COLLECTION_NAME: str = Field("documents", env="CHROMA_COLLECTION_NAME")

    # BM25 Index Storage
    BM25_PERSIST_DIR: str = Field("./data/bm25_index", env="BM25_PERSIST_DIR",
                                  description="Directory to store BM25 index files")
    BM25_INDEX_NAME: str = Field("documents_bm25", env="BM25_INDEX_NAME",
                                 description="Name for the BM25 index")

    # Database (SQLite) for metadata persistence
    # We assume SQLAlchemy Async with aiosqlite; repository adapters will use this URL.
    DATABASE_URL: Optional[str] = Field("sqlite+aiosqlite:///./data/app.db", env="DATABASE_URL")

    UPLOADS_DIR: Optional[str] = Field(default="data/uploads", env="UPLOADS_DIR", description="The directory to upload to.")
    UPLOAD_MAX_PART_SIZE: int = 50 * 1024 * 1024
    UPLOAD_MAX_FIELDS: Optional[int] = None
    UPLOAD_MAX_FILE_SIZE: Optional[int] = None

    # FastAPI settings
    API_HOST: str = Field("0.0.0.0", env="API_HOST")
    API_PORT: int = Field(8000, env="API_PORT")
    DEBUG: bool = Field(False, env="DEBUG")

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


# Instantiate settings. This will load, validate, and expose the settings.
# Pydantic will raise a ValidationError if required fields are missing or types are wrong.
try:
    settings = AppSettings()
except ValidationError as e:
    # Log the detailed validation error
    error_messages = []
    for error in e.errors():
        field = ".".join(str(loc) for loc in error['loc'])
        message = error['msg']
        error_messages.append(f"  - Field '{field}': {message}")

    full_error_message = "😱 Environment variable validation failed!\n" + "\n".join(error_messages) + \
                         "\nPlease check the logs and your .env file or environment settings."
    logger.error(full_error_message)

    # Re-raise a custom error
    raise CriticalConfigError(full_error_message) from e

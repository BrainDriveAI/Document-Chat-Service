from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field, HttpUrl, SecretStr, ValidationError
from typing import Optional, Literal
import logging

logger = logging.getLogger(__name__)


class CriticalConfigError(Exception):
    """Custom exception for critical configuration failures."""
    pass


class AppSettings(BaseSettings):
    # Pydantic model configuration
    model_config = {
        'extra': 'ignore',
        'env_file': '.env',
        'env_file_encoding': 'utf-8',
        'validate_default': True,
    }

    # spaCy
    SPACY_MODEL: str = Field("en_core_web_sm", env="SPACY_MODEL")

    # -- LLM provider selection --
    LLM_PROVIDER: Literal[
        'openai', 'pinecone', 'ollama', 'openrouter', 'groq'] = Field(
        default='ollama',
        env='LLM_PROVIDER',
        description="Which LLM provider to use: openai, ollama, openrouter, or groq."
    )
    LLM_TIMEOUT: Optional[int] = Field(
        default=300,
        env="LLM_TIMEOUT",
        description="LLM API Requests Timeout Config."
    )

    # -- Embedding provider selection --
    EMBEDDING_PROVIDER: Literal['openai', 'pinecone', 'ollama'] = Field(
        default='ollama',
        env='EMBEDDING_PROVIDER',
        description="Which embedding provider to use: openai, pinecone, or ollama."
    )

    # OpenAI (for Evaluation)
    OPENAI_API_KEY: Optional[SecretStr] = Field(
        default=None,
        env="OPENAI_API_KEY",
        description="OpenAI API key for evaluation judge service."
    )
    OPENAI_EVALUATION_MODEL: str = Field(
        default="gpt-5-mini",
        env="OPENAI_EVALUATION_MODEL",
        description="OpenAI model to use as judge for evaluation."
    )
    OPENAI_EVALUATION_TIMEOUT: int = Field(
        default=60,
        env="OPENAI_EVALUATION_TIMEOUT",
        description="Timeout for evaluation judge API calls in seconds."
    )

    # Evaluation Settings
    INITIALIZE_EVALUATION: bool = Field(
        default=True,
        env="INITIALIZE_EVALUATION",
        description="Whether to initialize evaluation system on startup."
    )
    EVALUATION_TEST_COLLECTION_ID: str = Field(
        default="eval-test-collection-00000000-0000-0000-0000-000000000001",
        env="EVALUATION_TEST_COLLECTION_ID",
        description="Fixed ID for the evaluation test collection."
    )
    EVALUATION_TEST_COLLECTION_NAME: str = Field(
        default="evaluation_test_collection",
        env="EVALUATION_TEST_COLLECTION_NAME",
        description="Name of the evaluation test collection."
    )
    EVALUATION_TEST_DOCS_DIR: str = Field(
        default="evaluation_test_docs",
        env="EVALUATION_TEST_DOCS_DIR",
        description="Directory containing evaluation test documents."
    )

    # EMBEDDING OPTIMIZATIONS
    EMBEDDING_BATCH_SIZE: Optional[int] = Field(
        default=4,  # Start even smaller - 4 instead of 8
        env="EMBEDDING_BATCH_SIZE",
        description="Number of texts to embed in one API call. Lower = less memory."
    )
    
    EMBEDDING_CONCURRENCY: Optional[int] = Field(
        default=1,  # Process one batch at a time to avoid memory spikes
        env="EMBEDDING_CONCURRENCY",
        description="Number of concurrent batch requests. 1 = sequential processing.",
    )
    
    EMBEDDING_TIMEOUT: Optional[int] = Field(
        default=120,  # Reduced from 300 - fail faster
        env="EMBEDDING_TIMEOUT",
        description="Timeout per embedding request (seconds)."
    )
    
    # Add retry configuration for embeddings
    EMBEDDING_MAX_RETRIES: Optional[int] = Field(
        default=3,
        env="EMBEDDING_MAX_RETRIES",
        description="Max retries for failed embedding requests."
    )
    
    EMBEDDING_RETRY_DELAY: Optional[float] = Field(
        default=2.0,
        env="EMBEDDING_RETRY_DELAY", 
        description="Delay between embedding retries (seconds)."
    )

    # Contextual Retrieval
    ENABLE_CONTEXTUAL_RETRIEVAL: Optional[bool] = Field(
        default=True,
        env='ENABLE_CONTEXTUAL_RETRIEVAL',
        description="Enable Contextual Retrieval."
    )
    OLLAMA_CONTEXTUAL_LLM_BASE_URL: Optional[HttpUrl] = Field(
        default='http://localhost:11434',
        env='OLLAMA_CONTEXTUAL_LLM_BASE_URL',
        description="Ollama LLM base URL for contextual retrieval."
    )
    OLLAMA_CONTEXTUAL_LLM_MODEL: Optional[str] = Field(
        default=None,
        env='OLLAMA_CONTEXTUAL_LLM_MODEL',
        description="Ollama LLM model for contextual retrieval."
    )

    # CONTEXTUAL RETRIEVAL OPTIMIZATIONS
    CONTEXTUAL_BATCH_SIZE: Optional[int] = Field(
        default=2,
        description="Batch size for contextual processing."
    )
    CONTEXTUAL_CHUNK_TIMEOUT: Optional[int] = Field(
        default=90,
        description="Timeout in seconds for contextual batch processing."
    )
    CONTEXTUAL_DOC_MAX_LENGTH: Optional[int] = Field(
        default=4000,
        description="Contextual processing prompt length."
    )

    # Groq
    GROQ_API_KEY: Optional[SecretStr] = Field(default=None, description="Your Groq API Key.")
    GROQ_LLM_MODEL: Optional[str] = Field(default="llama3-70b-8192", description="The Groq LLM model to be used.")

    # Ollama
    OLLAMA_LLM_BASE_URL: Optional[HttpUrl] = Field(
        default='http://localhost:11434',
        env='OLLAMA_LLM_BASE_URL',
        description="Ollama LLM base URL."
    )
    OLLAMA_LLM_MODEL: Optional[str] = Field(
        default=None,
        env='OLLAMA_LLM_MODEL',
        description="Ollama LLM model."
    )

    OLLAMA_EMBEDDING_BASE_URL: Optional[HttpUrl] = Field(
        default='http://localhost:11434',
        env='OLLAMA_EMBEDDING_BASE_URL',
        description='Ollama Embedding base URL.'
    )
    OLLAMA_EMBEDDING_MODEL: Optional[str] = Field(
        default="mxbai-embed-large",
        env='OLLAMA_EMBEDDING_MODEL',
        description="The Ollama embedding model to be used."
    )
    EMBEDDING_BATCH_SIZE: Optional[int] = Field(
        default=8,
        env="EMBEDDING_BATCH_SIZE",
        description="Embedding batch size."
    )

    EMBEDDING_CONCURRENCY: Optional[int] = Field(
        default=2,
        env="EMBEDDING_CONCURRENCY",
        description="Embedding concurrency.",
    )

    # Vector store (Chroma)
    CHROMA_PERSIST_DIR: str = Field("./data/vector_db", env="CHROMA_PERSIST_DIR")
    CHROMA_COLLECTION_NAME: str = Field("documents", env="CHROMA_COLLECTION_NAME")
    ANONYMIZED_TELEMETRY: Optional[bool] = Field(False, env="ANONYMIZED_TELEMETRY")

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

    # Document Processor API Configuration
    DOCUMENT_PROCESSOR_API_URL: Optional[HttpUrl] = Field(
        default='http://host.docker.internal:8080/documents/',
        env='DOCUMENT_PROCESSOR_API_URL',
        description="Document processor API URL"
    )
    DOCUMENT_PROCESSOR_API_KEY: Optional[SecretStr] = Field(
        default=None,
        env='DOCUMENT_PROCESSOR_API_KEY',
        description="Your Document Processing API Key."
    )
    DOCUMENT_PROCESSOR_TIMEOUT: Optional[int] = Field(
        default=300,
        env='DOCUMENT_PROCESSOR_TIMEOUT',
        description="Document Processing Timeout Config."
    ) 
    DOCUMENT_PROCESSOR_MAX_RETRIES: Optional[int] = Field(
        default=3,
        env='DOCUMENT_PROCESSOR_MAX_RETRIES',
        description="Document Processing Max Retry Config."
    )

    # FastAPI settings
    API_HOST: str = Field("0.0.0.0", env="API_HOST")
    API_PORT: int = Field(8000, env="API_PORT")
    DEBUG: bool = Field(False, env="DEBUG")


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

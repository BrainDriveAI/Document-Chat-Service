# Use official Python runtime as base image
FROM python:3.12-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PIP_DISABLE_PIP_VERSION_CHECK=1

# Set work directory
WORKDIR /app

# Install system dependencies (minimal approach)
RUN apt-get update && apt-get install -y \
    gcc \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install core dependencies first (most stable, cached layer)
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --no-cache-dir \
    fastapi==0.115.13 \
    uvicorn[standard]==0.34.3 \
    sqlalchemy==2.0.41 \
    aiosqlite==0.21.0 \
    python-multipart==0.0.20 \
    jinja2==3.1.6 \
    aiofiles==24.1.0 \
    prometheus-client==0.22.1 \
    python-dotenv==1.1.0 \
    pydantic_settings==2.9.1


# Install ML/NLP dependencies (heavier packages)
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --no-cache-dir \
    pandas==2.3.0 \
    httpx==0.28.1 \
    tiktoken==0.9.0 \
    aiohttp==3.12.13 \
    psycopg2-binary==2.9.10

# Install spacy and related packages
# RUN pip install --no-cache-dir \
#     spacy==3.8.7 \
#     spacy-layout==0.0.12

# Install search and vector store packages
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --no-cache-dir \
    chromadb==1.0.13 \
    rank-bm25==0.2.2 \
    langchain-text-splitters==0.3.8

# Copy project files
COPY . .

# Create necessary directories
RUN mkdir -p /app/data/uploads /app/data/bm25_index /app/data/vector_db /app/logs

# Make sure the data directory is writable
RUN chmod -R 755 /app/data /app/logs

# Expose port
EXPOSE 8000

# Health check
# HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
#     CMD curl -f http://localhost:8000/health || exit 1

# Run the application
CMD ["python", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]

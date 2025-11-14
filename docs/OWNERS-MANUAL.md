# Chat with Documents - Owner's Manual

**Version:** 1.0
**Last Updated:** November 2025

A comprehensive guide to operating, maintaining, and extending your RAG-powered document chat application.

---

## Table of Contents

1. [Quick Start](#quick-start)
2. [What Is This Application?](#what-is-this-application)
3. [Core Features](#core-features)
4. [Getting Started](#getting-started)
5. [Daily Operations](#daily-operations)
6. [Advanced Configuration](#advanced-configuration)
7. [Troubleshooting](#troubleshooting)
8. [Performance Tuning](#performance-tuning)
9. [Security & Privacy](#security--privacy)
10. [Extending the System](#extending-the-system)
11. [Maintenance & Updates](#maintenance--updates)
12. [FAQ](#faq)

---

## Quick Start

**Get running in 3 steps:**

### Option 1: Local Development (Recommended)
```bash
# 1. Setup environment
cp .env.example .env
poetry install && poetry shell

# 2. Start server
uvicorn app.main:app --reload

# 3. Open browser
http://localhost:8000
```

### Option 2: Docker
```bash
# 1. Configure
cp .env.example .env

# 2. Run
docker-compose up --build

# 3. Access
http://localhost:8000
```

**First Steps After Installation:**
1. Create a collection at `/collections`
2. Upload a document (PDF, DOCX, TXT)
3. Start chatting at `/chat`

---

## What Is This Application?

### In Simple Terms
Chat with your documents using AI. Upload PDFs, Word docs, or text files, then ask questions about their content in natural language.

### How It Works
1. **Upload** → Documents processed into searchable chunks
2. **Search** → Your question finds relevant chunks (hybrid vector + keyword search)
3. **Chat** → AI answers using only retrieved context (no hallucinations)

### What Makes It Special
- **Local-first:** Runs on your infrastructure (Ollama models)
- **Clean Architecture:** Swappable components (Ollama → OpenAI, ChromaDB → Qdrant)
- **Advanced Retrieval:** Hybrid search, query transformation, intent classification
- **Privacy:** Your data stays on your servers

---

## Core Features

### 1. Document Management
- **Upload:** PDF, DOCX, TXT, MD files
- **Process:** spaCy Layout-based structure extraction
- **Chunk:** Multiple strategies (hierarchical, semantic, optimized)
- **Store:** ChromaDB vector store + BM25 keyword index

### 2. Collections
- **Organize:** Group related documents
- **Isolate:** Each collection is a separate search space
- **Summarize:** Get overview of collection contents

### 3. Intelligent Search
- **Hybrid Search:** Vector similarity + BM25 keyword matching
- **Query Transformation:**
  - Contextualize with chat history
  - Multi-query expansion
  - HyDE (hypothetical document embeddings)
- **Intent Classification:** Automatically detects what you're asking for

### 4. Chat Interface
- **Context-aware:** Remembers conversation history
- **Factual:** Only answers from document content
- **Transparent:** Shows source chunks for every answer

### 5. Evaluation System
- **Automated Testing:** Measure RAG accuracy
- **LLM Judge:** OpenAI evaluates factual correctness
- **Metrics:** Track retrieval quality over time

### 6. Advanced Features
- **Contextual Retrieval:** Enhanced chunk context for better search
- **Dynamic Context Window:** Auto-detects model capabilities
- **Batch Processing:** Optimized for large documents
- **Background Tasks:** Non-blocking document processing

---

## Getting Started

### Prerequisites

**Required:**
- Python 3.12+
- Ollama server (local or remote)
- Document Processor API (BrainDrive-Document-AI)

**Optional:**
- Docker & Docker Compose (for containerized deployment)
- OpenAI API key (for evaluation system)

### Installation

#### Method 1: Local Development

1. **Clone repository:**
```bash
git clone https://github.com/BrainDriveAI/chat-with-your-documents.git
cd chat-with-your-documents
```

2. **Install dependencies:**
```bash
poetry install
```

3. **Configure environment:**
```bash
cp .env.example .env
```

Edit `.env` with your settings:
```bash
# LLM Configuration
LLM_PROVIDER=ollama
OLLAMA_LLM_BASE_URL=http://localhost:11434
OLLAMA_LLM_MODEL=llama3.2:8b

# Embedding Configuration
EMBEDDING_PROVIDER=ollama
OLLAMA_EMBEDDING_BASE_URL=http://localhost:11434
OLLAMA_EMBEDDING_MODEL=mxbai-embed-large

# Document Processing
DOCUMENT_PROCESSOR_API_URL=http://localhost:8080/documents/
DOCUMENT_PROCESSOR_API_KEY=your_api_key

# Performance
EMBEDDING_BATCH_SIZE=8
EMBEDDING_CONCURRENCY=2
```

4. **Start application:**
```bash
poetry shell
uvicorn app.main:app --reload
```

5. **Verify startup:**
Check logs for successful Ollama connectivity:
```
INFO: Embedding service OK (mxbai-embed-large)
INFO: LLM service OK (llama3.2:8b)
INFO: Startup complete
```

#### Method 2: Docker

1. **Setup:**
```bash
git clone https://github.com/BrainDriveAI/chat-with-your-documents.git
cd chat-with-your-documents
cp .env.example .env
```

2. **Run:**
```bash
docker-compose up --build
```

3. **Production deployment:**
```bash
docker-compose -f docker-compose.prod.yml up --build -d
```

### First Use

1. **Access web UI:** http://localhost:8000
2. **API docs:** http://localhost:8000/docs
3. **Create collection:**
   - Navigate to "Collections"
   - Click "Create New Collection"
   - Name it (e.g., "Product Documentation")
4. **Upload document:**
   - Select collection
   - Click "Upload Document"
   - Choose file (PDF/DOCX/TXT)
   - Wait for processing
5. **Start chatting:**
   - Go to "Chat"
   - Select collection
   - Ask a question about your documents

---

## Daily Operations

### Managing Collections

**Create Collection:**
```bash
# Via Web UI
http://localhost:8000/collections → "Create"

# Via API
curl -X POST http://localhost:8000/api/collections \
  -H "Content-Type: application/json" \
  -d '{"name": "My Collection", "description": "Product docs"}'
```

**List Collections:**
```bash
# Web UI: /collections
# API: GET /api/collections
```

**Delete Collection:**
```bash
# Deletes collection + all documents + embeddings
DELETE /api/collections/{collection_id}
```

### Uploading Documents

**Via Web UI:**
1. Select collection
2. Click "Upload"
3. Choose file
4. Wait for background processing
5. Check "Documents" tab for status

**Via API:**
```bash
curl -X POST http://localhost:8000/api/documents/ \
  -F "file=@document.pdf" \
  -F "collection_id=<collection_id>"
```

**Supported formats:**
- PDF (`.pdf`)
- Word (`.docx`)
- Text (`.txt`)
- Markdown (`.md`)

**Processing time:**
- Small doc (<10 pages): 10-30 seconds
- Medium doc (10-50 pages): 30-120 seconds
- Large doc (>50 pages): 2-10 minutes

### Chatting with Documents

**Start Chat:**
1. Navigate to `/chat`
2. Select collection
3. Type question
4. View answer + source chunks

**Chat Features:**
- **Context-aware:** Remembers conversation
- **Source attribution:** Shows which chunks were used
- **Follow-ups:** Ask clarifying questions
- **Multi-document:** Searches across all docs in collection

**Best Practices:**
- Be specific in questions
- Reference document sections if known
- Use follow-up questions to refine answers
- Check source chunks for verification

### Monitoring & Metrics

**Health Check:**
```bash
curl http://localhost:8000/health
```

**Prometheus Metrics:**
```bash
http://localhost:8000/metrics
```

**Key Metrics:**
- `http_requests_total`: API request count
- `http_request_duration_seconds`: Latency
- `documents_processed_total`: Processing count
- `embedding_generation_duration_seconds`: Embedding time

**Logs:**
```bash
# Local: stdout
# Docker: docker-compose logs -f

# Log levels: DEBUG, INFO, WARNING, ERROR
# Configure via DEBUG=true in .env
```

---

## Advanced Configuration

### Environment Variables

#### LLM & Embedding Configuration

```bash
# Provider selection
LLM_PROVIDER=ollama                    # ollama | openai
EMBEDDING_PROVIDER=ollama              # ollama | openai

# Ollama LLM
OLLAMA_LLM_BASE_URL=http://localhost:11434
OLLAMA_LLM_MODEL=llama3.2:8b
OLLAMA_LLM_TEMPERATURE=0.7
OLLAMA_LLM_CONTEXT_WINDOW=4096

# Ollama Embeddings
OLLAMA_EMBEDDING_BASE_URL=http://localhost:11434
OLLAMA_EMBEDDING_MODEL=mxbai-embed-large
EMBEDDING_DIMENSION=1024
```

#### Contextual Retrieval (Advanced)

```bash
# Enable context generation for chunks
ENABLE_CONTEXTUAL_RETRIEVAL=true

# Smaller model for context generation
OLLAMA_CONTEXTUAL_LLM_BASE_URL=http://localhost:11434
OLLAMA_CONTEXTUAL_LLM_MODEL=llama3.2:3b
CONTEXTUAL_BATCH_SIZE=3
CONTEXTUAL_CHUNK_TIMEOUT=60
```

**What it does:** Generates contextual description for each chunk using smaller LLM, improving retrieval accuracy.

**When to enable:**
- ✅ Need high retrieval accuracy
- ✅ Have computational resources
- ❌ Processing speed is critical (doubles processing time)

#### Performance Tuning

```bash
# Embedding generation
EMBEDDING_BATCH_SIZE=8                 # Chunks per batch (4-16)
EMBEDDING_CONCURRENCY=2                # Parallel requests (1-4)
EMBEDDING_TIMEOUT=120                  # Seconds
EMBEDDING_MAX_RETRIES=3

# Document processing
DOCUMENT_PROCESSOR_TIMEOUT=300         # Seconds
DOCUMENT_PROCESSOR_MAX_RETRIES=3
CONTEXTUAL_DOC_MAX_LENGTH=4000         # Chars

# Search
DEFAULT_TOP_K=5                        # Results to retrieve
HYBRID_SEARCH_ALPHA=0.5                # Vector vs BM25 weight (0-1)
```

**Tuning guide:**
- **16GB RAM:** `EMBEDDING_BATCH_SIZE=8`, `EMBEDDING_CONCURRENCY=2`
- **32GB RAM:** `EMBEDDING_BATCH_SIZE=16`, `EMBEDDING_CONCURRENCY=4`
- **8GB RAM:** `EMBEDDING_BATCH_SIZE=4`, `EMBEDDING_CONCURRENCY=1`

#### Storage Configuration

```bash
# Directories
UPLOADS_DIR=./data/uploads
CHROMA_PERSIST_DIR=./data/vector_db
BM25_PERSIST_DIR=./data/bm25_index
DATABASE_URL=sqlite+aiosqlite:///./data/app.db

# Telemetry (disable for privacy)
CHROMA_TELEMETRY=0
SCARF_NO_ANALYTICS=true
DO_NOT_TRACK=true
ANONYMIZED_TELEMETRY=false
```

### Chunking Strategies

**Available strategies:**

1. **OptimizedHierarchicalChunking** (Default)
   - Preserves document structure
   - Uses headings as boundaries
   - Best for: Structured documents, technical docs

2. **HierarchicalChunking**
   - Standard hierarchical splitting
   - Best for: General documents

3. **SemanticChunking**
   - Groups semantically similar content
   - Best for: Narrative content, essays

4. **RecursiveChunking**
   - Recursive text splitting
   - Best for: Unstructured text

5. **FixedSizeChunking**
   - Simple fixed-size chunks
   - Best for: Testing, simple docs

**Change strategy:**
Edit `app/main.py`:
```python
from app.adapters.document_processing.chunking_strategies import SemanticChunkingStrategy

# In on_startup()
chunking_strategy = SemanticChunkingStrategy(
    chunk_size=512,
    chunk_overlap=50
)
```

### Search Configuration

**Hybrid search parameters:**
```python
# app/config.py
HYBRID_SEARCH_ALPHA=0.5  # 0 = BM25 only, 1 = Vector only
```

**Query transformation:**
```python
# Enable/disable in use case
methods = [
    QueryTransformationMethod.CONTEXTUALIZE,  # Use chat history
    QueryTransformationMethod.MULTI_QUERY,     # Generate variations
    QueryTransformationMethod.HYDE             # Hypothetical docs
]
```

---

## Troubleshooting

### Common Issues

#### 1. Server Won't Start

**Symptom:** `uvicorn app.main:app` fails

**Solutions:**
```bash
# Check Python version
python --version  # Must be 3.12+

# Check dependencies
poetry install

# Check .env file exists
ls .env

# Check Ollama is running
curl http://localhost:11434/api/tags
```

#### 2. Document Upload Fails

**Symptom:** Upload returns error or hangs

**Check:**
```bash
# 1. Document Processor is running
curl http://localhost:8080/health

# 2. File size reasonable (<100MB)
ls -lh document.pdf

# 3. Check logs
docker-compose logs -f  # or stdout in local dev
```

**Solutions:**
- Increase `DOCUMENT_PROCESSOR_TIMEOUT` in `.env`
- Check Document Processor API key
- Verify file format (PDF, DOCX, TXT, MD)

#### 3. Embeddings Fail / OOM

**Symptom:** Embedding generation crashes or times out

**Solutions:**
```bash
# Reduce batch size
EMBEDDING_BATCH_SIZE=4  # Down from 8

# Reduce concurrency
EMBEDDING_CONCURRENCY=1  # Down from 2

# Increase timeout
EMBEDDING_TIMEOUT=180  # Up from 120
```

#### 4. Search Returns No Results

**Check:**
```bash
# 1. Documents processed successfully
GET /api/documents?collection_id=<id>

# 2. ChromaDB has data
ls -la data/vector_db/

# 3. BM25 index exists
ls -la data/bm25_index/
```

**Solutions:**
- Re-upload documents
- Check collection ID matches
- Verify embeddings generated (check logs)

#### 5. Chat Responses Slow

**Tuning:**
```bash
# Use smaller LLM
OLLAMA_LLM_MODEL=llama3.2:3b  # Faster than 8b

# Reduce context window
OLLAMA_LLM_CONTEXT_WINDOW=2048  # Down from 4096

# Reduce retrieved chunks
DEFAULT_TOP_K=3  # Down from 5
```

#### 6. Unicode Errors on Windows

**Symptom:** Logging errors with emojis (🔍, ✅)

**Solution:** Cosmetic only - app still works. To fix:
```python
# app/infrastructure/logging.py
# Remove emojis from log messages
```

### Error Codes

| Code | Meaning | Solution |
|------|---------|----------|
| 400 | Bad request | Check request format |
| 404 | Not found | Verify collection/document ID |
| 413 | File too large | Reduce file size or increase limit |
| 500 | Server error | Check logs, verify services running |
| 503 | Service unavailable | Check Ollama/Document Processor |

### Debug Mode

**Enable verbose logging:**
```bash
# .env
DEBUG=true
LOG_LEVEL=DEBUG
```

**Check specific components:**
```python
# app/main.py - startup logs show:
# - Ollama connectivity
# - Model verification
# - Service initialization
# - Index loading
```

---

## Performance Tuning

### For Different Hardware

#### 8GB RAM System
```bash
EMBEDDING_BATCH_SIZE=4
EMBEDDING_CONCURRENCY=1
CONTEXTUAL_BATCH_SIZE=2
ENABLE_CONTEXTUAL_RETRIEVAL=false
```

#### 16GB RAM System (Recommended)
```bash
EMBEDDING_BATCH_SIZE=8
EMBEDDING_CONCURRENCY=2
CONTEXTUAL_BATCH_SIZE=3
ENABLE_CONTEXTUAL_RETRIEVAL=true
```

#### 32GB+ RAM System
```bash
EMBEDDING_BATCH_SIZE=16
EMBEDDING_CONCURRENCY=4
CONTEXTUAL_BATCH_SIZE=5
ENABLE_CONTEXTUAL_RETRIEVAL=true
```

### Ollama Performance

**Use remote Ollama instances:**
```bash
# Dedicated embedding server
OLLAMA_EMBEDDING_BASE_URL=https://ollama-embed.example.com

# Dedicated LLM server
OLLAMA_LLM_BASE_URL=https://ollama-llm.example.com
```

**Model selection:**
```bash
# Faster (lower quality)
OLLAMA_LLM_MODEL=llama3.2:3b
OLLAMA_EMBEDDING_MODEL=nomic-embed-text

# Balanced (recommended)
OLLAMA_LLM_MODEL=llama3.2:8b
OLLAMA_EMBEDDING_MODEL=mxbai-embed-large

# Better (slower)
OLLAMA_LLM_MODEL=llama3.1:70b
OLLAMA_EMBEDDING_MODEL=mxbai-embed-large
```

### Database Optimization

**For large deployments:**
```bash
# Use PostgreSQL instead of SQLite
DATABASE_URL=postgresql+asyncpg://user:pass@localhost/chatdocs

# Connection pooling
DB_POOL_SIZE=20
DB_MAX_OVERFLOW=10
```

### Caching

**Model info cache:**
- Automatically caches Ollama model metadata
- Reduces startup time
- Refresh: Restart application

---

## Security & Privacy

### Data Privacy

**What stays local:**
- ✅ All documents
- ✅ All embeddings
- ✅ All chat history
- ✅ All metadata

**What goes to external services:**
- ❌ Nothing (if using local Ollama)
- ⚠️ Evaluation queries (if using OpenAI judge)

### API Security

**Production checklist:**
```bash
# 1. Add authentication
# app/api/middleware/auth.py

# 2. Enable HTTPS
# Use Nginx reverse proxy (docker-compose.prod.yml)

# 3. Restrict CORS
CORS_ORIGINS=https://yourdomain.com

# 4. Rate limiting
# app/api/middleware/rate_limit.py

# 5. API keys for Document Processor
DOCUMENT_PROCESSOR_API_KEY=<strong_key>
```

### File Upload Security

**Built-in protections:**
- File type validation
- Path traversal prevention
- Size limits
- Temp file cleanup

**Configure limits:**
```python
# app/config.py
MAX_UPLOAD_SIZE=100_000_000  # 100MB
ALLOWED_EXTENSIONS=[".pdf", ".docx", ".txt", ".md"]
```

### Sensitive Data

**Do NOT commit:**
- `.env` file (use `.env.example` as template)
- API keys
- Database files
- Uploaded documents

**Use secrets management:**
```bash
# Example: AWS Secrets Manager, HashiCorp Vault
# Retrieve at runtime, not in .env
```

---

## Extending the System

### Adding New LLM Provider

1. **Create adapter:**
```python
# app/adapters/llm/new_provider_llm.py
from app.core.ports.llm_service import LLMService

class NewProviderLLMService(LLMService):
    async def generate(self, prompt: str, **kwargs) -> str:
        # Implementation
        pass
```

2. **Update config:**
```python
# app/config.py
class Settings(BaseSettings):
    llm_provider: str = "new_provider"
    new_provider_api_key: SecretStr
```

3. **Register in startup:**
```python
# app/main.py
if settings.llm_provider == "new_provider":
    app.state.llm_service = NewProviderLLMService(...)
```

### Adding New Vector Store

1. **Implement port:**
```python
# app/adapters/vector_store/new_store.py
from app.core.ports.vector_store import VectorStore

class NewVectorStoreAdapter(VectorStore):
    async def add_chunks(self, chunks: List[DocumentChunk]) -> None:
        # Implementation
        pass
```

2. **Update startup:**
```python
# app/main.py
app.state.vector_store = NewVectorStoreAdapter(...)
```

### Adding Custom Chunking Strategy

```python
# app/adapters/document_processing/chunking_strategies/custom_chunking.py
from app.core.ports.document_processor import ChunkingStrategy

class CustomChunkingStrategy(ChunkingStrategy):
    def chunk(self, elements: List[StructuredElement]) -> List[DocumentChunk]:
        # Your chunking logic
        return chunks
```

### Adding New API Endpoint

```python
# app/api/routes/my_feature.py
from fastapi import APIRouter, Depends

router = APIRouter(prefix="/my-feature", tags=["my-feature"])

@router.get("/")
async def my_endpoint():
    return {"message": "Hello"}

# Register in app/main.py
from app.api.routes import my_feature
app.include_router(my_feature.router, prefix="/api")
```

---

## Maintenance & Updates

### Backup & Restore

**What to backup:**
```bash
# 1. Database
cp data/app.db data/backups/app_$(date +%Y%m%d).db

# 2. Vector store
tar -czf vector_db_backup.tar.gz data/vector_db/

# 3. BM25 index
tar -czf bm25_backup.tar.gz data/bm25_index/

# 4. Uploaded documents
tar -czf uploads_backup.tar.gz data/uploads/

# 5. Configuration
cp .env backups/.env.backup
```

**Restore:**
```bash
# Stop application
docker-compose down

# Restore files
cp backups/app_20251114.db data/app.db
tar -xzf vector_db_backup.tar.gz -C data/
tar -xzf bm25_backup.tar.gz -C data/
tar -xzf uploads_backup.tar.gz -C data/

# Restart
docker-compose up -d
```

### Updating Dependencies

```bash
# Update all dependencies
poetry update

# Update specific package
poetry update fastapi

# Check for security issues
poetry check
```

### Database Migrations

**Alembic setup (future):**
```bash
# Initialize
alembic init migrations

# Create migration
alembic revision --autogenerate -m "Add new table"

# Apply migration
alembic upgrade head
```

### Monitoring Production

**Setup checklist:**
- [ ] Prometheus scraping `/metrics`
- [ ] Log aggregation (ELK, Loki)
- [ ] Uptime monitoring
- [ ] Disk space alerts
- [ ] Error rate alerts

**Key alerts:**
- Error rate > 1%
- Response time > 5s
- Disk usage > 80%
- Document processing failures

---

## FAQ

### General

**Q: What file formats are supported?**
A: PDF, DOCX, TXT, MD. Others require Document Processor API extension.

**Q: Can I use OpenAI instead of Ollama?**
A: Yes. Set `LLM_PROVIDER=openai` and provide API key. See extending section.

**Q: Is my data private?**
A: Yes (with local Ollama). All processing happens locally. Only evaluation system uses external API (OpenAI judge).

**Q: How many documents can I upload?**
A: No hard limit. Depends on disk space and RAM. Tested with 1000+ documents.

**Q: Does it work offline?**
A: Yes, with local Ollama. Document Processor must be accessible (can also run locally).

### Technical

**Q: Why Clean Architecture?**
A: Swappable components. Change LLM, vector store, chunking without touching business logic.

**Q: Why Ollama?**
A: Local-first, privacy, free, good quality models. Can swap to OpenAI anytime.

**Q: What's contextual retrieval?**
A: Each chunk gets AI-generated context, improving search accuracy. Optional feature.

**Q: How does hybrid search work?**
A: Combines vector similarity (semantic) + BM25 (keyword) with rank fusion. Best of both.

**Q: Can I customize prompts?**
A: Yes. Edit prompts in respective use cases (`app/core/use_cases/`).

### Performance

**Q: Why is document processing slow?**
A: Remote Document Processor API + embedding generation. Use contextual retrieval=false for 2x speedup.

**Q: Why are embeddings slow?**
A: Large batch size or slow Ollama. Reduce `EMBEDDING_BATCH_SIZE` and `EMBEDDING_CONCURRENCY`.

**Q: Can I use GPU acceleration?**
A: Yes. Ollama auto-detects GPU. Verify with `ollama ps`.

**Q: How to make chat faster?**
A: Smaller LLM (`llama3.2:3b`), reduce `DEFAULT_TOP_K`, lower `CONTEXT_WINDOW`.

### Troubleshooting

**Q: "Connection refused" to Ollama**
A: Ollama not running or wrong URL. Check `OLLAMA_LLM_BASE_URL` and `ollama serve`.

**Q: Document upload stuck**
A: Document Processor timeout or crash. Check logs, increase `DOCUMENT_PROCESSOR_TIMEOUT`.

**Q: No search results**
A: Documents not processed. Check `/api/documents`, verify embeddings generated.

**Q: Unicode errors on Windows**
A: Cosmetic logging issue (emojis). App works fine. Remove emojis from code if annoying.

---

## Quick Reference

### Important URLs

- **Web UI:** http://localhost:8000
- **API Docs:** http://localhost:8000/docs
- **Health:** http://localhost:8000/health
- **Metrics:** http://localhost:8000/metrics

### Key Files

- **Config:** `app/config.py`
- **Startup:** `app/main.py`
- **Routes:** `app/api/routes/`
- **Use Cases:** `app/core/use_cases/`
- **Adapters:** `app/adapters/`

### Environment Variables

| Variable | Purpose | Default |
|----------|---------|---------|
| `LLM_PROVIDER` | LLM service | `ollama` |
| `OLLAMA_LLM_MODEL` | Chat model | `llama3.2:8b` |
| `OLLAMA_EMBEDDING_MODEL` | Embedding model | `mxbai-embed-large` |
| `EMBEDDING_BATCH_SIZE` | Chunks per batch | `8` |
| `ENABLE_CONTEXTUAL_RETRIEVAL` | Context generation | `false` |
| `DEFAULT_TOP_K` | Search results | `5` |

### Support

- **Issues:** https://github.com/BrainDriveAI/chat-with-your-documents/issues
- **Docs:** `/docs` directory
- **Architecture:** `docs/braindrive_rag_system.md`
- **Evaluation:** `docs/evaluation-system.md`

---

**Need help?** Check troubleshooting section or open an issue on GitHub.

**Want to contribute?** Read `CLAUDE.md` for development guidelines.

**Ready to extend?** See "Extending the System" section and Clean Architecture guides.

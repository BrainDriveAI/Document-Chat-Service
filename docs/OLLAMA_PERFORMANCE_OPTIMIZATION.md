# Ollama Performance Optimization Guide

This guide helps you optimize Ollama performance for the entire RAG application based on your hardware configuration.

## Overview

This application uses Ollama for multiple operations:
- 📄 **Document Processing**: Generating embeddings for document chunks (batch operations)
- 💬 **Chat Interactions**: LLM inference for answering questions
- 🔍 **Search/Retrieval**: Embedding generation for query vectors
- 📊 **Evaluation**: Intensive context retrieval for multiple questions (most demanding)

All operations share the same Ollama instance, so optimizing Ollama configuration improves performance across the entire application.

---

## Quick Start

### 1. Check Your System Resources

**Windows (PowerShell):**
```powershell
# Check total RAM
systeminfo | findstr /C:"Total Physical Memory"

# Check GPU (if NVIDIA)
nvidia-smi

# Check CPU
wmic cpu get name
```

**Linux/Mac:**
```bash
# Check total RAM
free -h

# Check GPU
nvidia-smi  # For NVIDIA
lspci | grep -i vga  # For any GPU

# Check CPU
lscpu | grep "Model name"
```

### 2. Configure Ollama for Your System

Based on your hardware profile (see sections below), set these environment variables:

**Windows (PowerShell):**
```powershell
# Temporary (current session only)
$env:OLLAMA_NUM_PARALLEL = "2"
$env:OLLAMA_MAX_LOADED_MODELS = "1"

# Permanent (restart PowerShell to take effect)
[System.Environment]::SetEnvironmentVariable('OLLAMA_NUM_PARALLEL', '2', 'User')
[System.Environment]::SetEnvironmentVariable('OLLAMA_MAX_LOADED_MODELS', '1', 'User')

# Restart Ollama service
Restart-Service Ollama
# Or restart from Services app if service not found
```

**Linux:**
```bash
# Add to ~/.bashrc or ~/.zshrc
export OLLAMA_NUM_PARALLEL=2
export OLLAMA_MAX_LOADED_MODELS=1
export OLLAMA_MAX_QUEUE=512

# Reload shell
source ~/.bashrc

# Restart Ollama
systemctl restart ollama
```

**Mac:**
```bash
# Add to ~/.zshrc or ~/.bash_profile
export OLLAMA_NUM_PARALLEL=2
export OLLAMA_MAX_LOADED_MODELS=1

# Reload shell
source ~/.zshrc

# Restart Ollama.app
pkill Ollama && open -a Ollama
```

### 3. Configure Application

Create/update `.env` file in project root:
```bash
# Evaluation-specific concurrency
EVALUATION_CONCURRENCY=2

# Embedding optimization (affects document processing & search)
EMBEDDING_BATCH_SIZE=8
EMBEDDING_CONCURRENCY=2

# See .env.example for all options
```

### 4. Restart Application

```bash
# Stop application
# Ctrl+C if running in terminal

# Restart
uvicorn app.main:app --reload
```

---

## Configuration by Hardware Profile

### 💻 Basic Laptop / Entry-Level PC
**Specs:** 8-12GB RAM, integrated GPU or no GPU, CPU-only inference

#### Ollama Configuration
```bash
# Windows PowerShell
$env:OLLAMA_NUM_PARALLEL = "1"
$env:OLLAMA_MAX_LOADED_MODELS = "1"

# Linux/Mac bash/zsh
export OLLAMA_NUM_PARALLEL=1
export OLLAMA_MAX_LOADED_MODELS=1
```

#### Application Configuration (`.env`)
```bash
# Evaluation
EVALUATION_CONCURRENCY=1

# Embeddings
EMBEDDING_BATCH_SIZE=5
EMBEDDING_CONCURRENCY=1
EMBEDDING_TIMEOUT=180

# Document processing
CONTEXTUAL_BATCH_SIZE=2
ENABLE_CONTEXTUAL_RETRIEVAL=false  # Disable for speed
```

#### Expected Performance
| Operation | Speed | Notes |
|-----------|-------|-------|
| Document ingestion | ~5-10 docs/min | Small documents (1-2 pages) |
| Chat response | 10-20 tokens/sec | Noticeable delay |
| Search query | 2-4 seconds | Single query |
| Evaluation (50 questions) | 20-25 minutes | Most intensive |

#### Recommended Models
- **LLM**: `llama3.2:3b` or `phi3:mini` (small, fast)
- **Embeddings**: `nomic-embed-text:latest` (137M params, fast)

#### Optimization Tips
- Close browser tabs and other apps during heavy operations
- Use quantized models (Q4, Q5 GGUF formats)
- Process documents in smaller batches (5-10 at a time)
- Reduce `top_k` in search from 5 to 3
- Disable contextual retrieval
- Avoid running evaluations with >30 questions at once

---

### 🖥️ Mid-Range PC
**Specs:** 16GB RAM, integrated or entry-level dedicated GPU (GTX 1650, AMD RX 6500, Intel Arc A380)

#### Ollama Configuration
```bash
# Windows PowerShell
$env:OLLAMA_NUM_PARALLEL = "2"
$env:OLLAMA_MAX_LOADED_MODELS = "1"

# Linux/Mac
export OLLAMA_NUM_PARALLEL=2
export OLLAMA_MAX_LOADED_MODELS=1
```

#### Application Configuration (`.env`)
```bash
# Evaluation
EVALUATION_CONCURRENCY=2

# Embeddings
EMBEDDING_BATCH_SIZE=8
EMBEDDING_CONCURRENCY=2
EMBEDDING_TIMEOUT=120

# Document processing
CONTEXTUAL_BATCH_SIZE=3
ENABLE_CONTEXTUAL_RETRIEVAL=false  # Optional, slows processing 2x
```

#### Expected Performance
| Operation | Speed | Notes |
|-----------|-------|-------|
| Document ingestion | 15-25 docs/min | Standard documents |
| Chat response | 25-40 tokens/sec | Responsive |
| Search query | 1-2 seconds | Fast |
| Evaluation (50 questions) | 8-12 minutes | Reasonable |

#### Recommended Models
- **LLM**: `llama3.2:8b` or `mistral:7b` (balanced quality/speed)
- **Embeddings**: `mxbai-embed-large` (335M params, good quality)

#### Optimization Tips
- Monitor RAM usage during first operations
- If RAM >90%, reduce concurrency to 1
- If RAM <70%, try concurrency=3 for evaluation
- GPU should handle embedding generation efficiently
- Can enable contextual retrieval for better quality (doubles processing time)

---

### 🚀 Enthusiast / Gaming PC
**Specs:** 32GB RAM, mid-to-high end GPU (RTX 3060/4060, AMD RX 6700 XT, RTX 3080)

#### Ollama Configuration
```bash
# Windows PowerShell
$env:OLLAMA_NUM_PARALLEL = "4"
$env:OLLAMA_MAX_LOADED_MODELS = "2"

# Linux/Mac
export OLLAMA_NUM_PARALLEL=4
export OLLAMA_MAX_LOADED_MODELS=2
```

#### Application Configuration (`.env`)
```bash
# Evaluation
EVALUATION_CONCURRENCY=4

# Embeddings
EMBEDDING_BATCH_SIZE=12
EMBEDDING_CONCURRENCY=3
EMBEDDING_TIMEOUT=120

# Document processing
CONTEXTUAL_BATCH_SIZE=5
ENABLE_CONTEXTUAL_RETRIEVAL=true  # Enable for best quality
```

#### Expected Performance
| Operation | Speed | Notes |
|-----------|-------|-------|
| Document ingestion | 40-60 docs/min | Fast bulk ingestion |
| Chat response | 50-80 tokens/sec | Very responsive |
| Search query | <1 second | Nearly instant |
| Evaluation (50 questions) | 4-6 minutes | Efficient |

#### Recommended Models
- **LLM**: `llama3.2:8b`, `mistral:7b`, or `llama3:13b` (larger models)
- **Embeddings**: `mxbai-embed-large` or `nomic-embed-text`

#### Optimization Tips
- Can handle multiple concurrent operations (chat while processing docs)
- Enable all advanced features (contextual retrieval, hybrid search)
- Increase `top_k` to 7 for better context quality
- Can run evaluations with 100+ questions
- GPU VRAM allows larger batch sizes

---

### 🏢 Workstation / Server
**Specs:** 64GB+ RAM, high-end GPU (RTX 4090, A6000, A100) or multi-GPU setup

#### Ollama Configuration
```bash
# Windows PowerShell
$env:OLLAMA_NUM_PARALLEL = "8"
$env:OLLAMA_MAX_LOADED_MODELS = "3"
$env:OLLAMA_MAX_QUEUE = "1024"

# Linux/Mac
export OLLAMA_NUM_PARALLEL=8
export OLLAMA_MAX_LOADED_MODELS=3
export OLLAMA_MAX_QUEUE=1024
```

#### Application Configuration (`.env`)
```bash
# Evaluation
EVALUATION_CONCURRENCY=8

# Embeddings
EMBEDDING_BATCH_SIZE=16
EMBEDDING_CONCURRENCY=4
EMBEDDING_TIMEOUT=90

# Document processing
CONTEXTUAL_BATCH_SIZE=8
ENABLE_CONTEXTUAL_RETRIEVAL=true
```

#### Expected Performance
| Operation | Speed | Notes |
|-----------|-------|-------|
| Document ingestion | 100+ docs/min | Bulk processing |
| Chat response | 100+ tokens/sec | Real-time streaming |
| Search query | <500ms | Instant |
| Evaluation (50 questions) | 2-3 minutes | Very fast |

#### Recommended Models
- **LLM**: Any model including `llama3:70b`, `mixtral:8x7b`, `codellama:70b`
- **Embeddings**: `mxbai-embed-large` or specialized domain embeddings

#### Optimization Tips
- Consider vLLM for even better performance (10-20x throughput)
- Enable all advanced features
- Can run multiple evaluation pipelines simultaneously
- For multi-GPU: Configure tensor parallelism (see Advanced section)
- Suitable for production multi-user deployments

---

## Performance by Operation Type

### 📄 Document Processing

**What happens:**
1. Upload document → Process structure → Chunk text
2. **Generate embeddings** for each chunk (batch operation)
3. Optionally generate contextual summaries (if enabled)
4. Store in vector database

**Performance factors:**
- `EMBEDDING_BATCH_SIZE`: How many chunks processed together
- `EMBEDDING_CONCURRENCY`: Parallel embedding requests
- `ENABLE_CONTEXTUAL_RETRIEVAL`: Doubles processing time but improves quality

**Optimization:**
```bash
# Fast ingestion (basic quality)
EMBEDDING_BATCH_SIZE=12
EMBEDDING_CONCURRENCY=2
ENABLE_CONTEXTUAL_RETRIEVAL=false

# Best quality (slower)
EMBEDDING_BATCH_SIZE=8
EMBEDDING_CONCURRENCY=2
ENABLE_CONTEXTUAL_RETRIEVAL=true
CONTEXTUAL_BATCH_SIZE=3
```

### 💬 Chat Interactions

**What happens:**
1. Receive user question
2. **Generate embedding** for query
3. Search vector database (fast)
4. **LLM generates answer** using retrieved context

**Performance factors:**
- LLM model size (larger = slower but better)
- Context window size
- Streaming vs non-streaming
- `OLLAMA_NUM_PARALLEL`: Handles concurrent chat users

**Optimization:**
```bash
# For responsiveness, use smaller models
OLLAMA_LLM_MODEL=llama3.2:8b  # Instead of 70b

# Enable streaming for better UX
# (handled automatically in code)
```

### 🔍 Search/Retrieval

**What happens:**
1. Receive search query
2. **Generate embedding** for query
3. Hybrid search: Vector similarity + BM25 keyword matching
4. Rank fusion and return results

**Performance factors:**
- Collection size (more documents = slower)
- Hybrid vs vector-only search
- `top_k` results requested

**Optimization:**
```bash
# Fast search (may miss some results)
# Use vector-only, reduce top_k
top_k=3

# Balanced (recommended)
# Use hybrid search, default top_k
top_k=5

# Comprehensive (slower)
# Use hybrid search, more results
top_k=10
```

### 📊 Evaluation (Most Intensive)

**What happens:**
1. Receive N questions (e.g., 50)
2. For EACH question:
   - Generate query embedding
   - Retrieve context (hybrid search)
3. Return all test data with retrieved context

**Performance factors:**
- Number of questions
- `EVALUATION_CONCURRENCY`: Parallel context retrieval
- Collection size
- `OLLAMA_NUM_PARALLEL`: Ollama's ability to handle parallel requests

**Optimization:**
```bash
# This is where hardware matters most!
# Configure based on your hardware profile above

# Safe default for most systems
EVALUATION_CONCURRENCY=2

# High-end systems
EVALUATION_CONCURRENCY=4-8
```

**Example timing for 50 questions:**
- Basic laptop (concurrency=1): 20-25 minutes
- Mid-range PC (concurrency=2): 8-12 minutes
- Gaming PC (concurrency=4): 4-6 minutes
- Workstation (concurrency=8): 2-3 minutes

---

## Performance Testing & Tuning

### Step 1: Baseline Test

Test each operation type to establish baseline performance.

#### Test Document Processing
```bash
# Upload a test document (2-3 pages)
# Monitor time and resource usage
curl -X POST http://localhost:8000/documents/upload \
  -F "file=@test_document.pdf" \
  -F "collection_id=test-collection"
```

#### Test Chat
```bash
# Send a chat query
curl -X POST http://localhost:8000/chat/sessions/test-session/messages \
  -H "Content-Type: application/json" \
  -d '{
    "content": "What is RAG?",
    "collection_id": "test-collection"
  }'
```

#### Test Search
```bash
# Perform search
curl -X POST http://localhost:8000/search \
  -H "Content-Type: application/json" \
  -d '{
    "query": "machine learning",
    "collection_id": "test-collection",
    "top_k": 5
  }'
```

#### Test Evaluation
```bash
# Run small evaluation (10 questions)
curl -X POST http://localhost:8000/evaluation/plugin/start-with-questions \
  -H "Content-Type: application/json" \
  -d '{
    "collection_id": "test-collection",
    "questions": ["Q1?", "Q2?", "Q3?", "Q4?", "Q5?", "Q6?", "Q7?", "Q8?", "Q9?", "Q10?"],
    "llm_model": "llama3.2:8b"
  }'
```

### Step 2: Monitor Resources

**Windows (PowerShell):**
```powershell
# Monitor in real-time
while ($true) {
    Clear-Host
    Write-Host "=== System Resources ===" -ForegroundColor Cyan

    # RAM
    $mem = Get-WmiObject Win32_OperatingSystem
    $totalRAM = [math]::Round($mem.TotalVisibleMemorySize / 1MB, 2)
    $freeRAM = [math]::Round($mem.FreePhysicalMemory / 1MB, 2)
    $usedRAM = $totalRAM - $freeRAM
    $ramPercent = [math]::Round(($usedRAM / $totalRAM) * 100, 1)
    Write-Host "RAM: ${usedRAM}GB / ${totalRAM}GB (${ramPercent}%)"

    # CPU
    $cpu = Get-Counter '\Processor(_Total)\% Processor Time' | Select-Object -ExpandProperty CounterSamples | Select-Object -ExpandProperty CookedValue
    $cpu = [math]::Round($cpu, 1)
    Write-Host "CPU: ${cpu}%"

    # GPU (if NVIDIA)
    try {
        nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv,noheader,nounits 2>$null
    } catch {}

    Start-Sleep -Seconds 2
}
```

**Linux:**
```bash
# Use htop or custom script
watch -n 2 'free -h; nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader'
```

### Step 3: Tuning Guidelines

**Scenario: Out of Memory Errors**

**Symptoms:**
- Application crashes during document processing
- Ollama errors: "context size exceeded"
- System freezes/swapping

**Solution:**
```bash
# Reduce parallel requests
OLLAMA_NUM_PARALLEL=1
EVALUATION_CONCURRENCY=1
EMBEDDING_CONCURRENCY=1

# Reduce batch sizes
EMBEDDING_BATCH_SIZE=5
CONTEXTUAL_BATCH_SIZE=2
```

**Scenario: Low Resource Usage (<70% RAM/GPU)**

**Symptoms:**
- Slow performance
- RAM usage consistently below 70%
- GPU underutilized

**Solution:**
```bash
# Increase parallelism
OLLAMA_NUM_PARALLEL=4
EVALUATION_CONCURRENCY=4
EMBEDDING_CONCURRENCY=3

# Increase batch sizes
EMBEDDING_BATCH_SIZE=12
```

**Scenario: Slow Despite Good Hardware**

**Symptoms:**
- Modern hardware but slow performance
- Low CPU/GPU usage
- Long response times

**Checklist:**
1. ✓ Verify Ollama is using GPU: `ollama ps` should show GPU
2. ✓ Check Ollama logs: Look for errors or warnings
3. ✓ Verify models are loaded: `ollama list`
4. ✓ Check network: Ollama should be localhost (no network latency)
5. ✓ Check disk I/O: Slow disk can bottleneck model loading
6. ✓ Other apps using GPU: Close games, video editors, etc.

**Solution:**
```bash
# Force GPU usage (if using NVIDIA)
# Windows PowerShell
$env:CUDA_VISIBLE_DEVICES = "0"

# Linux/Mac
export CUDA_VISIBLE_DEVICES=0

# Preload models to avoid loading delays
ollama run llama3.2:8b ""
ollama run mxbai-embed-large ""
```

---

## Advanced Optimizations

### Multi-GPU Setup (Ollama)

If you have multiple GPUs, Ollama automatically detects and uses them:

```bash
# Check GPU detection
ollama ps

# Should show multiple GPUs
# GPU 0: RTX 4090 (24GB)
# GPU 1: RTX 4090 (24GB)
```

**For large models across multiple GPUs:**
```bash
# Ollama automatically uses tensor parallelism
# No configuration needed for models that fit in single GPU

# For 70B+ models that need multiple GPUs:
# Ollama will automatically shard across available GPUs
ollama run llama3:70b
```

### vLLM for Production (High Performance)

For production deployments or high-throughput requirements, consider vLLM.

**Performance Comparison:**
| Metric | Ollama | vLLM |
|--------|--------|------|
| Peak Throughput | 41 TPS | 793 TPS |
| P99 Latency | 673ms | 80ms |
| Multi-user | Good | Excellent |
| Setup | Easy | Moderate |
| Best For | Development | Production |

**Docker vLLM Setup:**

**Single GPU:**
```bash
docker run --gpus all \
  --ipc=host \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  -p 8001:8000 \
  vllm/vllm-openai:latest \
  --model mistralai/Mistral-7B-Instruct-v0.2 \
  --tensor-parallel-size 1
```

**Multi-GPU (4 GPUs):**
```bash
docker run --gpus all \
  --ipc=host \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  -p 8001:8000 \
  vllm/vllm-openai:latest \
  --model mistralai/Mistral-7B-Instruct-v0.2 \
  --tensor-parallel-size 4
```

**Configure Application to Use vLLM:**
```bash
# In .env
LLM_PROVIDER=openai  # vLLM has OpenAI-compatible API
OLLAMA_LLM_BASE_URL=http://localhost:8001/v1
OLLAMA_LLM_MODEL=mistralai/Mistral-7B-Instruct-v0.2

# Embedding still uses Ollama (vLLM focuses on inference)
EMBEDDING_PROVIDER=ollama
OLLAMA_EMBEDDING_BASE_URL=http://localhost:11434
```

**When to Use vLLM:**
- Multi-user production deployments
- Need >100 TPS throughput
- Running evaluations continuously
- Enterprise environments

**When to Stick with Ollama:**
- Single-user or small team
- Development/prototyping
- Simpler setup preferred
- Resource-constrained environments

### Docker Model Runner (Apple Silicon)

For Mac users with Apple Silicon (M1/M2/M3):

**Advantages:**
- Optimized for Apple Silicon (Metal acceleration)
- Better performance than Ollama in Docker VM
- Part of Docker Desktop (easy setup)

**Setup:**
```bash
# Requires Docker Desktop 4.40+ on macOS
docker model run llama3.2:8b

# Configure application
# In .env
OLLAMA_LLM_BASE_URL=http://host.docker.internal:8080
OLLAMA_LLM_MODEL=llama3.2:8b
```

**Note:** Currently macOS only, expanding to other platforms.

---

## Troubleshooting

### Common Issues & Solutions

#### Issue: "Connection refused" to Ollama

**Symptoms:**
```
Error: Failed to connect to Ollama API at http://localhost:11434
```

**Solution:**
```bash
# 1. Check if Ollama is running
curl http://localhost:11434/api/tags

# 2. Check Ollama service status
# Windows: Services app -> Check "Ollama" service
# Linux: systemctl status ollama
# Mac: Check if Ollama.app is running

# 3. Restart Ollama
# Windows: Restart service or restart Ollama app
# Linux: systemctl restart ollama
# Mac: killall Ollama && open -a Ollama

# 4. Check firewall
# Ensure localhost/127.0.0.1 traffic is allowed
```

#### Issue: "Model not found"

**Symptoms:**
```
Error: model 'llama3.2:8b' not found
```

**Solution:**
```bash
# Pull the model
ollama pull llama3.2:8b
ollama pull mxbai-embed-large

# Verify models are available
ollama list

# If model exists but still errors, try running once
ollama run llama3.2:8b "test"
```

#### Issue: "Context length exceeded"

**Symptoms:**
```
Error: context size exceeded: 8192 > 4096
```

**Cause:** Parallel requests multiply context size

**Solution:**
```bash
# Option 1: Reduce parallel requests
OLLAMA_NUM_PARALLEL=2  # Instead of 4
EVALUATION_CONCURRENCY=2

# Option 2: Use model with larger context
# Switch from 4k to 8k or 32k context model
OLLAMA_LLM_MODEL=llama3.2:8b:32k  # If available
```

#### Issue: Slow Embedding Generation

**Symptoms:**
- Document processing takes very long
- Search queries slow (>5 seconds)
- Evaluation stuck on context retrieval

**Solutions:**
```bash
# 1. Verify embedding model is loaded
ollama list | grep embed

# 2. Preload embedding model
ollama run mxbai-embed-large ""

# 3. Reduce batch size
EMBEDDING_BATCH_SIZE=5  # From default 8

# 4. Check if GPU is being used
ollama ps  # Should show GPU info

# 5. Try smaller embedding model
OLLAMA_EMBEDDING_MODEL=nomic-embed-text  # Smaller, faster
```

#### Issue: Inconsistent Performance

**Symptoms:**
- First operation fast, subsequent ones slow
- Performance degrades over time
- Random slowdowns

**Causes & Solutions:**

**Model swapping:**
```bash
# Keep models loaded
OLLAMA_MAX_LOADED_MODELS=2  # Increase from 1

# Or preload models
ollama run llama3.2:8b ""
ollama run mxbai-embed-large ""
```

**Memory pressure:**
```bash
# Monitor memory usage
# If swapping to disk occurs, reduce concurrency
OLLAMA_NUM_PARALLEL=1
```

**Thermal throttling:**
```bash
# Check CPU/GPU temperatures
# Ensure adequate cooling
# Clean dust from fans/heatsinks
```

#### Issue: Application Hangs During Evaluation

**Symptoms:**
- Evaluation starts but never completes
- No error messages
- Application unresponsive

**Solution:**
```bash
# 1. Check Ollama queue status
curl http://localhost:11434/api/ps

# 2. If queue is full, increase limit
OLLAMA_MAX_QUEUE=1024  # From default 512

# 3. Reduce concurrent requests
EVALUATION_CONCURRENCY=1

# 4. Add timeout configuration
EMBEDDING_TIMEOUT=180  # Increase from 120
```

---

## Best Practices

### 1. Development vs Production

**Development Environment:**
- Use smaller, faster models for quick iteration
- Lower concurrency for stability (1-2)
- Enable debug logging: `DEBUG=true` in `.env`
- Test with small datasets (5-10 documents, 10-20 questions)

**Production Environment:**
- Optimize concurrency for your hardware (2-8)
- Use quality models (7B-13B range)
- Monitor performance metrics (`/metrics` endpoint)
- Consider vLLM for high throughput
- Regular performance testing

### 2. Resource Planning

**Estimate operation times:**

**Document Processing:**
```
Time = (Num documents × Avg pages) / (Processing speed)

Example: 100 docs × 3 pages / 20 pages/min = 15 minutes
```

**Evaluation:**
```
Time = Number of questions / Questions per minute

Example: 50 questions / 5 q/min = 10 minutes (mid-range PC)
```

**Plan accordingly:**
- Schedule large operations during off-hours
- Use state persistence API for long-running evaluations
- Process in batches if needed (e.g., 1000 docs → 10 batches of 100)

### 3. Model Selection Strategy

**Development (Speed Priority):**
- LLM: `llama3.2:3b` or `phi3:mini` (3-4B params)
- Embeddings: `nomic-embed-text` (137M params)
- Trade-off: Lower quality but 2-3x faster

**Production (Balanced):**
- LLM: `llama3.2:8b` or `mistral:7b` (7-8B params)
- Embeddings: `mxbai-embed-large` (335M params)
- Trade-off: Good quality, acceptable speed

**Enterprise (Quality Priority):**
- LLM: `llama3:70b` or `mixtral:8x7b` (70B params)
- Embeddings: `mxbai-embed-large` or domain-specific
- Trade-off: Best quality, requires powerful hardware

### 4. Monitoring & Observability

**Check Ollama Status:**
```bash
# Running models
ollama ps

# Available models
ollama list

# Server logs (Linux)
journalctl -u ollama -f

# Server logs (Mac)
tail -f ~/Library/Logs/Ollama/server.log
```

**Check Application Metrics:**
```bash
# Prometheus metrics endpoint
curl http://localhost:8000/metrics

# Health check
curl http://localhost:8000/health
```

**Monitor Performance:**
- Track document processing time
- Monitor chat response latency
- Log evaluation completion time
- Watch for error rate increases

---

## Environment Variable Reference

### Ollama Environment Variables

**Core Settings:**
```bash
# Maximum parallel requests per model (default: auto 1 or 4)
OLLAMA_NUM_PARALLEL=2

# Maximum loaded models (default: 3×GPU count or 3)
OLLAMA_MAX_LOADED_MODELS=1

# Maximum queued requests (default: 512)
OLLAMA_MAX_QUEUE=512

# Server host (default: 127.0.0.1:11434)
OLLAMA_HOST=0.0.0.0:11434  # For external access
```

**GPU Configuration:**
```bash
# Limit GPU memory usage (default: auto)
OLLAMA_GPU_MEMORY_FRACTION=0.9  # Use 90% of VRAM

# Force specific GPU (multi-GPU systems)
CUDA_VISIBLE_DEVICES=0  # NVIDIA
HIP_VISIBLE_DEVICES=0   # AMD
```

**Advanced:**
```bash
# Keep models loaded (default: unload after 5 min)
OLLAMA_KEEP_ALIVE=24h  # Keep for 24 hours

# Thread count for CPU inference (default: auto)
OLLAMA_NUM_THREADS=8
```

### Application Environment Variables

**LLM & Embeddings:**
```bash
# Provider selection
LLM_PROVIDER=ollama
EMBEDDING_PROVIDER=ollama

# Ollama configuration
OLLAMA_LLM_BASE_URL=http://localhost:11434
OLLAMA_LLM_MODEL=llama3.2:8b
OLLAMA_EMBEDDING_BASE_URL=http://localhost:11434
OLLAMA_EMBEDDING_MODEL=mxbai-embed-large
```

**Performance Optimization:**
```bash
# Evaluation
EVALUATION_CONCURRENCY=2  # Parallel context retrieval

# Embedding optimization
EMBEDDING_BATCH_SIZE=8
EMBEDDING_CONCURRENCY=2
EMBEDDING_TIMEOUT=120
EMBEDDING_MAX_RETRIES=3

# Contextual retrieval (optional)
ENABLE_CONTEXTUAL_RETRIEVAL=false
CONTEXTUAL_BATCH_SIZE=3
CONTEXTUAL_CHUNK_TIMEOUT=60
```

---

## Performance Benchmarks

### Reference Benchmarks (50-question evaluation)

| Hardware | Concurrency | Time | Questions/min |
|----------|-------------|------|---------------|
| Laptop (8GB, CPU) | 1 | 23 min | 2.2 |
| Laptop (16GB, iGPU) | 1 | 18 min | 2.8 |
| PC (16GB, GTX 1650) | 2 | 10 min | 5.0 |
| PC (32GB, RTX 3060) | 4 | 5 min | 10.0 |
| Workstation (64GB, RTX 4090) | 8 | 2.5 min | 20.0 |

**Note:** Times vary based on collection size, document complexity, and model choice.

### Document Processing Benchmarks (100 documents, 3 pages avg)

| Hardware | Batch Size | Contextual | Time |
|----------|------------|------------|------|
| Laptop (8GB) | 5 | No | 45 min |
| PC (16GB) | 8 | No | 20 min |
| PC (32GB) | 12 | Yes | 25 min |
| Workstation (64GB) | 16 | Yes | 10 min |

---

## Summary & Quick Reference

### Default Safe Configuration (Most Users)

**Ollama:**
```bash
OLLAMA_NUM_PARALLEL=2
OLLAMA_MAX_LOADED_MODELS=1
```

**Application (`.env`):**
```bash
EVALUATION_CONCURRENCY=2
EMBEDDING_BATCH_SIZE=8
EMBEDDING_CONCURRENCY=2
ENABLE_CONTEXTUAL_RETRIEVAL=false
```

**Models:**
```bash
OLLAMA_LLM_MODEL=llama3.2:8b
OLLAMA_EMBEDDING_MODEL=mxbai-embed-large
```

### When to Tune

**Increase concurrency if:**
- ✅ RAM usage < 70%
- ✅ Operations feel slow
- ✅ GPU underutilized
- ✅ Have high-end hardware

**Decrease concurrency if:**
- ❌ Out of memory errors
- ❌ System freezing/swapping
- ❌ Ollama timeouts
- ❌ Context length errors

### Getting Help

**Check logs first:**
```bash
# Application logs
tail -f logs/app.log

# Ollama logs
# Linux: journalctl -u ollama -f
# Mac: tail -f ~/Library/Logs/Ollama/server.log
# Windows: Check Event Viewer
```

**Common log locations:**
- Application: `./data/logs/` or console output
- Ollama: Varies by OS (see above)
- Docker: `docker logs <container_name>`

**Still having issues?**
- Check [GitHub Issues](https://github.com/YOUR_REPO/issues)
- Review Ollama documentation: https://docs.ollama.com
- Open a new issue with logs and system details

---

For questions or contributions to this guide, please open an issue or PR on GitHub.

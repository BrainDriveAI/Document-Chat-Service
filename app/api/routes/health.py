from fastapi import APIRouter, HTTPException
import os
import httpx
from ...config import settings

router = APIRouter()


@router.get("/")
async def health_check():
    # 1. Check writable paths
    try:
        test_path = os.path.join(settings.CHROMA_PERSIST_DIR, ".healthcheck")
        with open(test_path, "w") as f:
            f.write("ok")
        os.remove(test_path)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Filesystem check failed: {e}")

    # 2. Check database connectivity: try a simple query
    from ...adapters.persistence.sqlite_repository import SQLiteCollectionRepository
    repo = SQLiteCollectionRepository(settings.DATABASE_URL)
    try:
        await repo.init_models()  # ensures tables can be accessed/created
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Database check failed: {e}")

    # 3. Check Ollama embedding endpoint
    # Optionally, send a lightweight request to embedding or LLM service
    # Warning: this may slow health check; you can skip or make asynchronous timeout.
    try:
        async with httpx.AsyncClient(timeout=2) as client:
            url = settings.EMBEDDING_BASE_URL.rstrip("/") + "/api/embeddings"
            # Do not send real payload; instead, HEAD or GET if endpoint supports; if not, skip or use low-weight call
            # For now, skip actual call or do a lightweight ping if available.
            # response = await client.get(settings.EMBEDDING_BASE_URL + "/ping")
            # if response.status_code != 200:
            #     raise Exception("Embedding service ping failed")
            pass
    except Exception:
        # Log warning but do not fail health for external services if you prefer
        # raise HTTPException(status_code=500, detail=f"Embedding service unreachable: {e}")
        ...

    return {"status": "ok"}

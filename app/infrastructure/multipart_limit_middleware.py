import logging
from starlette.middleware.base import BaseHTTPMiddleware
from fastapi import Request
from starlette.responses import Response


class MultipartLimitMiddleware(BaseHTTPMiddleware):
    """
    Middleware to pre-parse multipart/form-data with larger limits.
    This calls request.form(max_part_size=...) so that FastAPI
    reuses the parsed form (cached) instead of using the default 1 MiB limit.
    """

    def __init__(self, app, max_part_size: int = 50 * 1024 * 1024, max_fields: int = None, max_file_size: int = None):
        """
        Args:
            app: ASGI app
            max_part_size: maximum size of a single part (in bytes). Default: 50 MiB.
            max_fields: optional maximum number of form fields; if None, leave default.
            max_file_size: optional maximum size to keep in memory before spooling to disk; if None, leave default.
        """
        super().__init__(app)
        self.max_part_size = max_part_size
        self.max_fields = max_fields
        self.max_file_size = max_file_size
        self.logger = logging.getLogger("app.multipart")

    async def dispatch(self, request: Request, call_next):
        content_type = request.headers.get("content-type", "")
        # Only intercept multipart/form-data requests
        if request.method in ("POST", "PUT", "PATCH") and "multipart/form-data" in content_type:
            try:
                # Pre-parse the form with desired limits.
                # Subsequent accesses to request.form() or dependencies (UploadFile=File(...)) will reuse this parsed form.
                # Pass max_part_size; optionally max_fields, max_file_size
                kwargs = {"max_part_size": self.max_part_size}
                if self.max_fields is not None:
                    kwargs["max_fields"] = self.max_fields
                if self.max_file_size is not None:
                    kwargs["max_file_size"] = self.max_file_size
                # This triggers parsing under the hood:
                await request.form(**kwargs)
                self.logger.debug(f"Parsed multipart form with increased limits: max_part_size={self.max_part_size}")
            except Exception as e:
                # If parsing fails (e.g. still too large), log and re-raise so FastAPI returns 400
                self.logger.warning(f"Multipart parsing failed under increased limits: {e}")
                # Let the exception propagate; FastAPI will return 400 with detail.
                raise

        response: Response = await call_next(request)
        return response

import logging
import sys
from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import Response
import time


def setup_logging():
    """
    Configure root logger for the application.
    Logs to stdout in a simple format with timestamp, level, logger name, and message.
    """
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)

    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(logging.INFO)

    formatter = logging.Formatter(
        fmt='%(asctime)s %(levelname)s %(name)s %(message)s',
        datefmt='%Y-%m-%dT%H:%M:%S%z'
    )
    handler.setFormatter(formatter)

    # Avoid adding multiple handlers in reload mode
    if not any(isinstance(h, logging.StreamHandler) for h in root_logger.handlers):
        root_logger.addHandler(handler)


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """
    Middleware to log incoming requests and responses with timing.
    Uses a custom logger (\"app.request\") to avoid conflicts with Uvicorn's access logger formatting.
    """

    async def dispatch(self, request: Request, call_next):
        logger = logging.getLogger("app.request")

        start_time = time.time()
        # Log request start
        logger.info(f"Request start: method={request.method} path={request.url.path}")
        try:
            response: Response = await call_next(request)
        except Exception as e:
            # Log exception
            logger.exception(f"Exception handling request: method={request.method} path={request.url.path} error={e}")
            raise

        process_time = (time.time() - start_time) * 1000
        # Log response status and time
        logger.info(f"Request end: method={request.method} path={request.url.path} status_code={response.status_code} time_ms={process_time:.2f}")
        # Optionally add a header
        response.headers["X-Process-Time-Ms"] = f"{process_time:.2f}"
        return response

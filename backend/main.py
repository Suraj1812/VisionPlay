from __future__ import annotations

from contextlib import asynccontextmanager
import logging
from pathlib import Path
import time
import uuid

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from starlette.middleware.trustedhost import TrustedHostMiddleware
from sqlalchemy import text

from backend.api.routes import router
from backend.database import Base
from backend.database.session import engine
from backend.services.processing_service import processing_service
from backend.utils.config import settings
from backend.utils.logging import configure_logging
from backend.utils.rate_limit import SlidingWindowRateLimiter
from backend.utils.request_context import reset_request_id, set_request_id


configure_logging()
settings.ensure_directories()
logger = logging.getLogger(__name__)
rate_limiter = SlidingWindowRateLimiter()


@asynccontextmanager
async def lifespan(_: FastAPI):
    Base.metadata.create_all(bind=engine)
    processing_service.start()
    yield
    processing_service.stop()


app = FastAPI(
    title=settings.app_name,
    version="1.0.0",
    debug=settings.debug,
    lifespan=lifespan,
)

app.add_middleware(TrustedHostMiddleware, allowed_hosts=settings.trusted_hosts or ["*"])
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.add_middleware(GZipMiddleware, minimum_size=settings.gzip_minimum_size_bytes)

app.mount(settings.media_mount_path, StaticFiles(directory=settings.media_root_path), name="media")
app.include_router(router)


@app.middleware("http")
async def observability_middleware(request: Request, call_next):
    request_id = request.headers.get(settings.request_id_header_name) or uuid.uuid4().hex
    request.state.request_id = request_id
    token = set_request_id(request_id)
    started_at = time.perf_counter()
    response = None

    try:
        response = await call_next(request)
    except Exception:
        logger.exception("Unhandled request failure for %s %s", request.method, request.url.path)
        raise
    finally:
        duration_ms = round((time.perf_counter() - started_at) * 1000, 2)
        if duration_ms >= settings.slow_request_threshold_ms:
            logger.warning(
                "Slow request %s %s completed in %.2fms",
                request.method,
                request.url.path,
                duration_ms,
            )
        reset_request_id(token)

    if response is None:
        return JSONResponse(status_code=500, content={"detail": "Internal server error"})
    response.headers[settings.request_id_header_name] = request_id
    return response


@app.middleware("http")
async def rate_limit_middleware(request: Request, call_next):
    if request.url.path in {"/health", "/ready"} or request.url.path.startswith(settings.media_mount_path):
        return await call_next(request)

    client_host = request.client.host if request.client is not None else "unknown"
    bucket = "global"
    limit = settings.global_rate_limit_per_minute

    if request.url.path == "/upload-video":
        bucket = "upload"
        limit = settings.upload_rate_limit_per_minute
    elif request.url.path == "/detect-live-frame":
        bucket = "live"
        limit = settings.live_rate_limit_per_minute
    elif request.url.path.startswith("/status/"):
        bucket = "status"
        limit = settings.status_rate_limit_per_minute

    allowed, retry_after = rate_limiter.allow(f"{bucket}:{client_host}", limit)
    if not allowed:
        request_id = request.headers.get(settings.request_id_header_name) or uuid.uuid4().hex
        return JSONResponse(
            status_code=429,
            content={
                "detail": "Rate limit exceeded. Please retry shortly.",
                "retry_after_seconds": retry_after,
            },
            headers={
                "Retry-After": str(retry_after),
                settings.request_id_header_name: request_id,
            },
        )

    return await call_next(request)


def _dependency_status() -> tuple[str, str]:
    database_status = "ok"
    storage_status = "ok"

    try:
        with engine.connect() as connection:
            connection.execute(text("SELECT 1"))
    except Exception:
        database_status = "error"

    required_directories = [
        settings.media_root_path,
        settings.upload_input_dir,
        settings.upload_output_dir,
    ]
    if not all(Path(directory).exists() for directory in required_directories):
        storage_status = "error"

    return database_status, storage_status


@app.get("/health")
def health_check() -> JSONResponse:
    worker_status = "ok" if processing_service.worker_started else "degraded"
    return JSONResponse(
        status_code=200,
        content={
            "status": "ok",
            "worker": worker_status,
            "environment": settings.environment,
            "version": "1.0.0",
        },
    )


@app.get("/ready")
def readiness_check() -> JSONResponse:
    database_status, storage_status = _dependency_status()
    worker_status = "ok" if processing_service.worker_started else "error"
    status_value = (
        "ok"
        if database_status == "ok" and storage_status == "ok" and worker_status == "ok"
        else "degraded"
    )
    status_code = 200 if status_value == "ok" else 503

    return JSONResponse(
        status_code=status_code,
        content={
            "status": status_value,
            "database": database_status,
            "storage": storage_status,
            "worker": worker_status,
            "environment": settings.environment,
            "version": "1.0.0",
        },
    )

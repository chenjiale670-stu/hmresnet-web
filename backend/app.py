from __future__ import annotations

import asyncio
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel, Field

from . import __version__
from .config import settings
from .fasta import FastaError, parse_fasta
from .model import ModelService


LOGGER = logging.getLogger("hmresnet")


class PredictRequest(BaseModel):
    fasta: str = Field(min_length=1, max_length=settings.max_request_bytes)
    threshold: float = Field(default=0.5, ge=0.01, le=0.99)
    top_k: int = Field(default=5, ge=1, le=23)


@asynccontextmanager
async def lifespan(app: FastAPI):
    service = ModelService(settings.model_dir, settings.device, settings.batch_size)
    app.state.model = service
    app.state.model_error = None
    app.state.prediction_slots = asyncio.Semaphore(max(1, settings.max_concurrent_predictions))
    try:
        service.load()
    except Exception as exc:
        LOGGER.exception("Failed to load HMResNet model")
        app.state.model_error = str(exc)
    yield


app = FastAPI(
    title="HMResNet Web API",
    description="Metal-resistance protein multi-label prediction",
    version=__version__,
    docs_url="/api/docs",
    redoc_url=None,
    openapi_url="/api/openapi.json",
    lifespan=lifespan,
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=list(settings.cors_origins),
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)


@app.middleware("http")
async def production_headers(request: Request, call_next):
    content_length = request.headers.get("content-length", "")
    if content_length.isdigit() and int(content_length) > settings.max_request_bytes:
        return JSONResponse(status_code=413, content={"detail": "请求体超过在线服务限制"})
    response = await call_next(request)
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
    response.headers["Permissions-Policy"] = "camera=(), microphone=(), geolocation=()"
    if request.url.path.startswith("/api/"):
        response.headers["Cache-Control"] = "no-store"
    return response


@app.get("/", include_in_schema=False)
def home() -> HTMLResponse:
    html_path = settings.frontend_dir / "index.html"
    if html_path.is_file():
        return HTMLResponse(html_path.read_text(encoding="utf-8"))
    return HTMLResponse("<html><body><h1>HMResNet</h1><p>Web UI is not configured.</p></body></html>", status_code=200)


@app.get("/api/health")
async def health(request: Request) -> dict:
    service: ModelService = request.app.state.model
    error = request.app.state.model_error
    return {
        "status": "ok" if service.model is not None else "degraded",
        "version": __version__,
        "model_loaded": service.model is not None,
        "device": str(service.device),
        "error": error,
    }


@app.get("/api/model")
async def model_information(request: Request) -> dict:
    service: ModelService = request.app.state.model
    return service.description()


@app.post("/api/predict")
async def predict(payload: PredictRequest, request: Request) -> dict:
    service: ModelService = request.app.state.model
    if service.model is None:
        raise HTTPException(status_code=503, detail="模型暂不可用，请检查 /api/health")
    try:
        records = parse_fasta(payload.fasta, max_records=settings.max_records)
    except FastaError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    async with request.app.state.prediction_slots:
        try:
            return await run_in_threadpool(
                service.predict, records, threshold=payload.threshold, top_k=payload.top_k
            )
        except Exception as exc:
            LOGGER.exception("Prediction failed")
            raise HTTPException(status_code=500, detail="模型推理失败") from exc

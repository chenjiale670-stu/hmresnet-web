from __future__ import annotations

import os
from pathlib import Path

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, Response


STATIC_DIR = Path(__file__).resolve().parents[2] / "frontend"
API_BASE = os.environ.get("HMRESNET_API", "http://127.0.0.1:8011").rstrip("/")
HOP_BY_HOP = {"connection", "keep-alive", "proxy-authenticate", "proxy-authorization", "te", "trailers", "transfer-encoding", "upgrade"}

app = FastAPI(title="HMResNet Gateway")


@app.on_event("startup")
async def startup() -> None:
    app.state.http = httpx.AsyncClient(timeout=700, follow_redirects=False)


@app.on_event("shutdown")
async def shutdown() -> None:
    await app.state.http.aclose()


@app.get("/")
async def home() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.api_route("/api/{path:path}", methods=["GET", "POST", "OPTIONS"])
async def api_proxy(request: Request, path: str) -> Response:
    upstream = await request.app.state.http.request(
        request.method,
        f"{API_BASE}/api/{path}",
        params=request.query_params,
        content=await request.body(),
        headers={
            name: value
            for name, value in request.headers.items()
            if name.lower() in {"accept", "content-type", "user-agent", "x-forwarded-for", "x-real-ip"}
        },
    )
    headers = {name: value for name, value in upstream.headers.items() if name.lower() not in HOP_BY_HOP}
    return Response(upstream.content, status_code=upstream.status_code, headers=headers)

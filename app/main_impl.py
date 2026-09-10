from __future__ import annotations

import logging

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from app import __version__
from app.config import Settings
from app.errors import APIError
from app.routes import create_router
from app.services import EngineService, LTRService, RenderService
from app.storage import FileStore


settings = Settings.load()
store = FileStore(settings)
engines = EngineService(settings)
renders = RenderService(settings)
ltr = LTRService(settings, engines)

app = FastAPI(
    title="STP 零件相似度检索 API",
    description="多零件库管理、上传解析、六视图渲染、三路索引、相似检索和 LTR 服务",
    version=__version__,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=list(settings.cors_origins),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/media/renders", StaticFiles(directory=settings.render_dir), name="renders")
app.include_router(create_router(settings, store, engines, renders, ltr))


@app.get("/", tags=["0. 通用说明"])
async def root() -> dict:
    return {
        "name": "STP 零件相似度检索 API",
        "version": __version__,
        "port": settings.port,
        "docs": "/docs",
        "status": "/api/status",
    }


@app.exception_handler(APIError)
async def api_error_handler(request: Request, exc: APIError) -> JSONResponse:
    return JSONResponse(
        status_code=exc.status_code,
        content={"success": False, "detail": exc.detail},
    )


@app.exception_handler(RequestValidationError)
async def validation_error_handler(
    request: Request, exc: RequestValidationError
) -> JSONResponse:
    return JSONResponse(
        status_code=422,
        content={"success": False, "detail": "请求参数校验失败", "errors": exc.errors()},
    )


@app.exception_handler(Exception)
async def unexpected_error_handler(request: Request, exc: Exception) -> JSONResponse:
    logging.exception("Unhandled API error")
    return JSONResponse(
        status_code=500,
        content={"success": False, "detail": f"服务器内部错误: {type(exc).__name__}"},
    )

"""ASGI application entrypoint."""

from fastapi import Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from app.main_impl import app, engines, ltr, renders, settings, store


@app.exception_handler(RequestValidationError)
async def safe_validation_error_handler(
    request: Request, exc: RequestValidationError
) -> JSONResponse:
    errors = [
        {
            "loc": list(error.get("loc", ())),
            "msg": error.get("msg", "参数无效"),
            "type": error.get("type", "value_error"),
        }
        for error in exc.errors()
    ]
    return JSONResponse(
        status_code=422,
        content={"success": False, "detail": "请求参数校验失败", "errors": errors},
    )


__all__ = ["app"]

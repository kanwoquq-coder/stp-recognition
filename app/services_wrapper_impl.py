"""Service exports with a strict JSON-normalization layer."""

from __future__ import annotations

import math
from typing import Any

from app import services_impl


def to_builtin(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): to_builtin(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [to_builtin(item) for item in value]
    if hasattr(value, "tolist"):
        try:
            return to_builtin(value.tolist())
        except (TypeError, ValueError):
            pass
    if hasattr(value, "item"):
        try:
            return to_builtin(value.item())
        except (TypeError, ValueError):
            pass
    if isinstance(value, float) and not math.isfinite(value):
        return 0.0
    if isinstance(value, PathLike):
        return str(value)
    return value


try:
    from pathlib import Path as PathLike
except ImportError:  # pragma: no cover
    PathLike = ()  # type: ignore[assignment]


services_impl.to_builtin = to_builtin

EngineService = services_impl.EngineService
RenderService = services_impl.RenderService
LTRService = services_impl.LTRService
STP_EXTENSIONS = services_impl.STP_EXTENSIONS

__all__ = [
    "EngineService",
    "RenderService",
    "LTRService",
    "STP_EXTENSIONS",
    "to_builtin",
]

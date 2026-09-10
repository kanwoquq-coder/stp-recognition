from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parents[1]


def _load_dotenv() -> None:
    env_file = PROJECT_DIR / ".env"
    if not env_file.exists():
        return
    for raw_line in env_file.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def _as_bool(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _path_env(name: str, default: Path) -> Path:
    value = os.getenv(name)
    return Path(value).expanduser().resolve() if value else default.resolve()


def _choice_env(name: str, default: str, choices: set[str]) -> str:
    value = os.getenv(name, default).strip().lower()
    if value not in choices:
        allowed = ", ".join(sorted(choices))
        raise ValueError(f"{name} 必须是以下值之一: {allowed}")
    return value


@dataclass(frozen=True)
class Settings:
    host: str
    port: int
    api_key: str
    base_url: str
    llm_api_key: str
    llm_base_url: str
    llm_model: str
    llm_timeout: float
    llm_max_retries: int
    llm_response_format: str
    llm_thinking: str
    embedding_model: str
    runtime_dir: Path
    upload_dir: Path
    library_dir: Path
    render_dir: Path
    report_dir: Path
    model_dir: Path
    db_path: Path
    max_upload_bytes: int
    allow_local_paths: bool
    cors_origins: tuple[str, ...]

    @classmethod
    def load(cls) -> "Settings":
        _load_dotenv()
        runtime_dir = _path_env("RUNTIME_DIR", PROJECT_DIR / "runtime")
        origins = tuple(
            item.strip()
            for item in os.getenv(
                "CORS_ORIGINS", "http://localhost:3000,http://localhost:5173"
            ).split(",")
            if item.strip()
        )
        api_key = os.getenv("API_KEY", "")
        base_url = os.getenv("BASE_URL", "")
        settings = cls(
            host=os.getenv("API_HOST", "0.0.0.0"),
            port=int(os.getenv("API_PORT", "8001")),
            api_key=api_key,
            base_url=base_url,
            llm_api_key=os.getenv("LLM_API_KEY", api_key),
            llm_base_url=os.getenv("LLM_BASE_URL", base_url),
            llm_model=os.getenv("MODEL", "gpt-4o"),
            llm_timeout=float(os.getenv("LLM_TIMEOUT_SECONDS", "180")),
            llm_max_retries=int(os.getenv("LLM_MAX_RETRIES", "1")),
            llm_response_format=_choice_env(
                "LLM_RESPONSE_FORMAT", "auto", {"auto", "json", "none"}
            ),
            llm_thinking=_choice_env(
                "LLM_THINKING", "auto", {"auto", "on", "off"}
            ),
            embedding_model=os.getenv("EMBEDDING_MODEL", "text-embedding-3-large"),
            runtime_dir=runtime_dir,
            upload_dir=_path_env("UPLOAD_DIR", runtime_dir / "uploads"),
            library_dir=_path_env("LIBRARY_DIR", runtime_dir / "library"),
            render_dir=_path_env("RENDER_DIR", runtime_dir / "renders"),
            report_dir=_path_env("REPORT_DIR", runtime_dir / "reports"),
            model_dir=_path_env("MODEL_DIR", runtime_dir / "models"),
            db_path=_path_env("DB_PATH", runtime_dir / "chroma_parts"),
            max_upload_bytes=int(os.getenv("MAX_UPLOAD_MB", "200")) * 1024 * 1024,
            allow_local_paths=_as_bool("ALLOW_LOCAL_PATHS", False),
            cors_origins=origins,
        )
        if settings.llm_timeout <= 0:
            raise ValueError("LLM_TIMEOUT_SECONDS 必须大于 0")
        if settings.llm_max_retries < 0:
            raise ValueError("LLM_MAX_RETRIES 不能小于 0")
        settings.ensure_directories()
        return settings

    def ensure_directories(self) -> None:
        for path in (
            self.runtime_dir,
            self.upload_dir,
            self.library_dir,
            self.render_dir,
            self.report_dir,
            self.model_dir,
            self.db_path,
        ):
            path.mkdir(parents=True, exist_ok=True)

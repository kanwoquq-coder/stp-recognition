import uvicorn

from app.config import Settings


if __name__ == "__main__":
    settings = Settings.load()
    uvicorn.run(
        "app.main:app",
        host=settings.host,
        port=settings.port,
        reload=False,
    )

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse

from app.api.routes import router
from app.config import get_settings

app = FastAPI(
    title="AI Resume Screening API",
    version="1.0.0",
    description="Semantic resume screening against job descriptions using embeddings and FAISS.",
)

app.include_router(router)

settings = get_settings()


@app.exception_handler(HTTPException)
def http_exception_handler(_: object, exc: HTTPException) -> JSONResponse:
    return JSONResponse(status_code=exc.status_code, content={"detail": exc.detail})


@app.get("/")
def root() -> dict[str, str]:
    return {"message": "Resume screening service is running"}

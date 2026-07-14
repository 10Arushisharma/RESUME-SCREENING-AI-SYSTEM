from __future__ import annotations

import logging
from datetime import datetime
from hashlib import sha256
from pathlib import Path
from typing import Any

import numpy as np

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse

from app.config import get_settings
from app.models.resume import JobDescription, ParsedResume
from app.services.embedding_service import EmbeddingService
from app.services.parser_service import ParserService
from app.services.pdf_service import PDFService
from app.services.ranking_service import RankingService
from app.vectorstore.faiss_manager import FAISSManager

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG, format="%(asctime)s %(levelname)s %(name)s %(message)s")
router = APIRouter()
settings = get_settings()
parser_service = ParserService()
pdf_service = PDFService()
embedding_service = EmbeddingService()
ranking_service = RankingService()
faiss_manager = FAISSManager(settings.embedding_dimensions)

resumes: list[ParsedResume] = []
current_job_description: JobDescription | None = None
latest_ranked_results: list[dict[str, Any]] = []
latest_ranked_at: str | None = None


@router.get("/health")
def health() -> dict[str, Any]:
    return {"status": "ok", "service": "resume-screening-ai"}


@router.post("/upload-job-description")
def upload_job_description(job_title: str = Form(...), job_description: str = Form(...)) -> dict[str, Any]:
    global current_job_description, latest_ranked_results, latest_ranked_at
    logger.info("Received job description upload: %s", job_title)
    parsed_job = parser_service.parse_job_description(job_title, job_description)
    parsed_job.embedding = embedding_service.embed_text(f"{job_title} {job_description}").tolist()
    current_job_description = parsed_job
    latest_ranked_results = []
    latest_ranked_at = None
    logger.info("Parsed job description skills=%s education=%s experience=%s", parsed_job.skills, parsed_job.education, parsed_job.experience)
    logger.info("Job description embedding length=%d", len(parsed_job.embedding))
    return {
        "job_title": parsed_job.title,
        "description": parsed_job.description,
        "skills": parsed_job.skills,
        "education": parsed_job.education,
        "experience": parsed_job.experience,
    }


@router.post("/upload-resumes")
def upload_resumes(files: list[UploadFile] | None = File(None)) -> dict[str, Any]:
    global resumes, latest_ranked_results, latest_ranked_at
    uploaded: list[ParsedResume] = []
    if not files:
        raise HTTPException(status_code=400, detail="No files provided")

    resumes.clear()
    faiss_manager.reset()
    latest_ranked_results = []
    latest_ranked_at = None

    seen_hashes: set[str] = set()
    resume_names: set[str] = set()

    for upload in files:
        if not upload.filename:
            raise HTTPException(status_code=400, detail="One or more uploaded files are missing a name")

        extension = Path(upload.filename).suffix.lower()
        if extension not in settings.allowed_extensions_list:
            raise HTTPException(status_code=400, detail=f"Unsupported file type for {upload.filename}")

        if upload.size and upload.size > settings.max_upload_size_mb * 1024 * 1024:
            raise HTTPException(status_code=413, detail="File too large")

        content = upload.file.read()
        content_hash = sha256(content).hexdigest()
        if content_hash in seen_hashes or upload.filename in resume_names:
            logger.warning("Skipping duplicate resume %s", upload.filename)
            continue
        seen_hashes.add(content_hash)
        resume_names.add(upload.filename)

        if len(content) == 0:
            raise HTTPException(status_code=400, detail=f"Uploaded file {upload.filename} is empty")
        if len(content) > settings.max_upload_size_mb * 1024 * 1024:
            raise HTTPException(status_code=413, detail=f"File {upload.filename} exceeds maximum upload size")

        logger.info("Extracting text from uploaded resume %s", upload.filename)
        text = pdf_service.extract_text_from_bytes(content, upload.filename)
        logger.info("Extracted resume %s text length=%d", upload.filename, len(text))
        logger.debug("Extracted resume %s text preview=%s", upload.filename, text[:400].replace("\n", " "))
        if not text:
            logger.warning("No text extracted from %s", upload.filename)

        logger.info("======== Resume Processing ========")
        logger.info("Resume: %s", upload.filename)
        logger.info("Extracted text length: %d", len(text))
        if not text:
            logger.warning("No text extracted from %s", upload.filename)
        resume = parser_service.parse_resume(text, upload.filename, candidate_name=upload.filename)
        logger.info("Parsed skills: %s", resume.skills)
        logger.info("Parsed education: %s", resume.education)
        logger.info("Parsed experience: %s", resume.experience)

        resume.file_path = str(Path("app/uploads") / upload.filename)
        resume.embedding = embedding_service.embed_text(text).tolist()
        logger.info("Resume %s parsed skills=%s experience=%s education=%s", resume.candidate_name, resume.skills, resume.experience, resume.education)
        logger.info("Resume %s embedding length=%d", resume.candidate_name, len(resume.embedding))
        resume.status = "parsed"
        uploaded.append(resume)
        resumes.append(resume)
        faiss_manager.insert(np.asarray(resume.embedding, dtype="float32"), resume.candidate_name)

    if not uploaded:
        raise HTTPException(status_code=400, detail="No valid resumes were uploaded")

    logger.info("FAISS index total vectors: %d", faiss_manager.index.ntotal)
    return {
        "message": "Resumes uploaded",
        "uploaded_count": len(uploaded),
        "resumes": [resume.candidate_name for resume in uploaded],
        "status": "uploaded",
    }


@router.post("/rank")
def rank_candidates() -> dict[str, Any]:
    global latest_ranked_results, latest_ranked_at
    if current_job_description is None:
        raise HTTPException(status_code=400, detail="Upload a job description before ranking")
    if not resumes:
        raise HTTPException(status_code=404, detail="No resumes available")

    logger.info("Ranking %d resumes against job description %s", len(resumes), current_job_description.title)
    ranked_results = ranking_service.rank_candidates(resumes, current_job_description)
    latest_ranked_results = ranked_results
    latest_ranked_at = datetime.utcnow().isoformat()
    logger.info("Ranking complete: %d candidates", len(ranked_results))
    return {"results": ranked_results, "generated_at": latest_ranked_at, "total_candidates": len(ranked_results)}


@router.get("/results")
def get_results() -> dict[str, Any]:
    global latest_ranked_results, latest_ranked_at
    if not resumes or current_job_description is None:
        logger.info("Results requested but no current job description or resumes available")
        return {"results": []}
    if latest_ranked_results:
        logger.info("Returning latest cached ranking computed at %s", latest_ranked_at)
        return {"results": latest_ranked_results, "generated_at": latest_ranked_at, "total_candidates": len(latest_ranked_results)}
    logger.info("No cached ranking found; computing fresh ranking from /results")
    latest_ranked_results = ranking_service.rank_candidates(resumes, current_job_description)
    latest_ranked_at = datetime.utcnow().isoformat()
    return {"results": latest_ranked_results, "generated_at": latest_ranked_at, "total_candidates": len(latest_ranked_results)}



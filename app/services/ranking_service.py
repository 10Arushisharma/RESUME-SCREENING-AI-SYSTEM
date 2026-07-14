from __future__ import annotations

import logging
import numpy as np

from app.config import get_settings
from app.models.resume import JobDescription, ParsedResume
from app.services.embedding_service import EmbeddingService
from app.services.explanation_service import ExplanationService
from app.services.similarity_service import SimilarityService

logger = logging.getLogger(__name__)


class RankingService:
    """Service responsible for computing candidate rankings."""

    def __init__(self) -> None:
        self.settings = get_settings()
        self.embedding_service = EmbeddingService()
        self.similarity_service = SimilarityService()
        self.explanation_service = ExplanationService()

    def rank_candidates(self, resumes: list[ParsedResume], job_description: JobDescription) -> list[dict]:
        if not resumes:
            return []

        job_text = self._build_text(job_description)
        logger.info("Building job description text length=%d skills=%s experience=%s education=%s", len(job_text), job_description.skills, job_description.experience, job_description.education)
        job_embedding = self.embedding_service.embed_text(job_text)
        results = []
        for resume in resumes:
            resume_text = self._build_text(resume)
            logger.info("Ranking resume=%s full_text_length=%d raw_text_length=%d skills=%s education=%s experience=%s", resume.candidate_name, len(resume_text), len(resume.raw_text or ""), resume.skills, resume.education, resume.experience)
            if resume.embedding is None or np.linalg.norm(np.asarray(resume.embedding, dtype=np.float32)) == 0:
                logger.info("Resume %s embedding missing or empty; generating from resume text", resume.candidate_name)
                resume.embedding = self.embedding_service.embed_text(resume_text).tolist()
            similarity = self.similarity_service.cosine_similarity(resume.embedding, job_embedding)
            logger.info("Resume %s semantic similarity=%s", resume.candidate_name, similarity)
            skills_score = self._score_overlap(resume.skills, job_description.skills)
            experience_score = self._score_overlap(resume.experience, job_description.experience)
            education_score = self._score_overlap(resume.education, job_description.education)
            projects_score = self._score_overlap(resume.projects, job_description.projects)
            certifications_score = self._score_overlap(resume.certifications, job_description.certifications)
            overall_score = self._calculate_weighted_score(
                semantic_similarity=similarity,
                skills_score=skills_score,
                experience_score=experience_score,
                education_score=education_score,
                projects_score=projects_score,
                certifications_score=certifications_score,
            )

            explanation = self.explanation_service.build_explanation(
                resume=resume,
                job_description=job_description,
                semantic_similarity=similarity,
                skills_score=skills_score,
                experience_score=experience_score,
                education_score=education_score,
                projects_score=projects_score,
                certifications_score=certifications_score,
                overall_score=overall_score,
            )
            logger.info("Resume %s scores semantic=%s skills=%s experience=%s education=%s overall=%s", resume.candidate_name, similarity, skills_score, experience_score, education_score, overall_score)
            results.append(explanation)

        results.sort(key=lambda item: item["overall_score"], reverse=True)
        return results

    def _score_overlap(self, candidate_items: list[str], job_items: list[str]) -> float:
        if not job_items:
            return 0.0
        if not candidate_items:
            return 0.0
        normalized_candidate = {item.lower().strip() for item in candidate_items if item}
        normalized_job = {item.lower().strip() for item in job_items if item}
        overlap = normalized_candidate.intersection(normalized_job)
        return round((len(overlap) / len(normalized_job)) * 100.0, 2)

    def _calculate_weighted_score(self, *, semantic_similarity: float, skills_score: float, experience_score: float, education_score: float, projects_score: float, certifications_score: float) -> float:
        weighted = (
            semantic_similarity * self.settings.semantic_weight
            + skills_score / 100.0 * self.settings.skills_weight
            + experience_score / 100.0 * self.settings.experience_weight
            + education_score / 100.0 * self.settings.education_weight
            + projects_score / 100.0 * self.settings.projects_weight
            + certifications_score / 100.0 * self.settings.certifications_weight
        )
        return round(weighted * 100.0, 2)

    def _build_text(self, value: ParsedResume | JobDescription) -> str:
        parts = []
        if hasattr(value, "raw_text") and getattr(value, "raw_text"):
            parts.append(getattr(value, "raw_text"))
        if hasattr(value, "title"):
            parts.append(getattr(value, "title") or "")
        if hasattr(value, "description"):
            parts.append(getattr(value, "description") or "")
        parts.extend(getattr(value, "skills", []))
        parts.extend(getattr(value, "education", []))
        parts.extend(getattr(value, "experience", []))
        parts.extend(getattr(value, "projects", []))
        parts.extend(getattr(value, "certifications", []))
        parts.extend(getattr(value, "technologies", []))
        joined = " ".join([part for part in parts if part])
        logger.debug("Built text length=%d for %s", len(joined), getattr(value, "candidate_name", getattr(value, "title", "unknown")))
        return joined

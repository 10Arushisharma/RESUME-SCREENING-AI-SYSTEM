from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field


class ParsedResumeSchema(BaseModel):
    candidate_name: str
    file_name: str
    file_path: str
    raw_text: str
    skills: list[str] = Field(default_factory=list)
    education: list[str] = Field(default_factory=list)
    experience: list[str] = Field(default_factory=list)
    projects: list[str] = Field(default_factory=list)
    certifications: list[str] = Field(default_factory=list)
    technologies: list[str] = Field(default_factory=list)
    years_experience: int = 0
    status: str = "parsed"
    error: str | None = None


class JobDescriptionSchema(BaseModel):
    title: str
    description: str
    skills: list[str] = Field(default_factory=list)
    education: list[str] = Field(default_factory=list)
    experience: list[str] = Field(default_factory=list)
    projects: list[str] = Field(default_factory=list)
    certifications: list[str] = Field(default_factory=list)
    technologies: list[str] = Field(default_factory=list)
    years_experience: int = 0


class RankingResultSchema(BaseModel):
    candidate_name: str
    overall_score: float
    semantic_similarity: float
    skills_score: float
    experience_score: float
    education_score: float
    projects_score: float
    certifications_score: float
    strengths: list[str] = Field(default_factory=list)
    missing_skills: list[str] = Field(default_factory=list)
    matching_technologies: list[str] = Field(default_factory=list)
    relevant_experience: list[str] = Field(default_factory=list)
    education_match: list[str] = Field(default_factory=list)
    experience_match: list[str] = Field(default_factory=list)
    project_match: list[str] = Field(default_factory=list)
    certifications_match: list[str] = Field(default_factory=list)
    strength_summary: str
    weakness_summary: str
    recommendations: list[str] = Field(default_factory=list)


class RankingResponseSchema(BaseModel):
    results: list[RankingResultSchema] = Field(default_factory=list)
    generated_at: datetime
    total_candidates: int


class UploadResponseSchema(BaseModel):
    message: str
    uploaded_count: int
    resumes: list[dict[str, Any]]

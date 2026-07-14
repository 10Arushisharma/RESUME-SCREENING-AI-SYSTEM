from __future__ import annotations

import logging
import re
from typing import Iterable

from app.models.resume import JobDescription, ParsedResume

logger = logging.getLogger(__name__)


class ParserService:
    """Service responsible for parsing resumes and job descriptions into structured data."""

    KNOWN_SKILLS = [
        "python",
        "fastapi",
        "django",
        "flask",
        "java",
        "c#",
        "c++",
        "javascript",
        "typescript",
        "react",
        "node",
        "sql",
        "postgresql",
        "mysql",
        "mongodb",
        "redis",
        "docker",
        "kubernetes",
        "aws",
        "azure",
        "gcp",
        "linux",
        "git",
        "api",
        "microservices",
        "pytest",
        "pandas",
        "numpy",
        "pytorch",
        "tensorflow",
        "opencv",
        "nlp",
        "machine learning",
        "deep learning",
        "data engineering",
    ]

    EDUCATION_TERMS = ["bachelor", "master", "phd", "degree", "university", "college", "engineering", "computer science"]
    PROJECT_TERMS = ["project", "developed", "built", "implemented", "designed", "launched"]
    CERTIFICATION_TERMS = ["certified", "certificate", "certification", "aws", "azure", "cka", "ckad"]

    def parse_resume(self, text: str, file_name: str, candidate_name: str | None = None) -> ParsedResume:
        logger.debug("Parsing resume %s text length=%d", file_name, len(text))
        normalized = self._normalize_text(text)
        skills = self._extract_skills(normalized)
        education = self._extract_education(normalized)
        experience_items = self._extract_experience(normalized)
        projects = self._extract_projects(normalized)
        certifications = self._extract_certifications(normalized)
        technologies = [skill for skill in skills if skill.lower() in {"python", "fastapi", "docker", "aws", "azure", "react", "sql", "kubernetes", "postgresql"}]
        years_experience = self._extract_years_experience(normalized)

        return ParsedResume(
            candidate_name=candidate_name or file_name.replace(".pdf", ""),
            file_name=file_name,
            file_path="",
            raw_text=text,
            skills=skills,
            education=education,
            experience=experience_items,
            projects=projects,
            certifications=certifications,
            technologies=technologies,
            years_experience=years_experience,
        )

    def parse_job_description(self, title: str, description: str, years_experience: int = 0) -> JobDescription:
        normalized = self._normalize_text(description)
        skills = self._extract_skills(normalized)
        education = self._extract_education(normalized)
        experience = self._extract_experience(normalized)
        projects = self._extract_projects(normalized)
        certifications = self._extract_certifications(normalized)
        technologies = [skill for skill in skills if skill.lower() in {"python", "fastapi", "docker", "aws", "azure", "react", "sql", "kubernetes", "postgresql"}]

        return JobDescription(
            title=title,
            description=description,
            skills=skills,
            education=education,
            experience=experience,
            projects=projects,
            certifications=certifications,
            technologies=technologies,
            years_experience=years_experience,
        )

    def _normalize_text(self, text: str) -> str:
        return re.sub(r"\s+", " ", text or "").strip().lower()

    def _extract_skills(self, text: str) -> list[str]:
        matches = []
        for skill in self.KNOWN_SKILLS:
            if re.search(rf"\b{re.escape(skill)}\b", text):
                matches.append(skill)
        return sorted(set(matches))

    def _extract_education(self, text: str) -> list[str]:
        matches = []
        for term in self.EDUCATION_TERMS:
            if re.search(rf"\b{re.escape(term)}\b", text):
                matches.append(term)
        return sorted(set(matches))

    def _extract_experience(self, text: str) -> list[str]:
        matches = []
        for match in re.finditer(r"\b\d+\+?\s*years?\b", text):
            matches.append(match.group(0))
        if not matches:
            return []
        return sorted(set(matches))

    def _extract_projects(self, text: str) -> list[str]:
        text_segments = [segment.strip() for segment in re.split(r"[.;]", text) if segment.strip()]
        projects = [segment for segment in text_segments if any(term in segment for term in self.PROJECT_TERMS)]
        return projects[:5]

    def _extract_certifications(self, text: str) -> list[str]:
        matches = []
        for term in self.CERTIFICATION_TERMS:
            if re.search(rf"\b{re.escape(term)}\b", text):
                matches.append(term)
        return sorted(set(matches))

    def _extract_years_experience(self, text: str) -> int:
        years_matches = re.findall(r"\b(\d+)\+?\s*years?\b", text)
        if not years_matches:
            return 0
        return max(int(year) for year in years_matches)

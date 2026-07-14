from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class ParsedResume:
    candidate_name: str
    file_name: str
    file_path: str
    raw_text: str
    skills: list[str] = field(default_factory=list)
    education: list[str] = field(default_factory=list)
    experience: list[str] = field(default_factory=list)
    projects: list[str] = field(default_factory=list)
    certifications: list[str] = field(default_factory=list)
    technologies: list[str] = field(default_factory=list)
    years_experience: int = 0
    status: str = "parsed"
    error: Optional[str] = None
    embedding: Optional[list[float]] = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class JobDescription:
    title: str
    description: str
    skills: list[str] = field(default_factory=list)
    education: list[str] = field(default_factory=list)
    experience: list[str] = field(default_factory=list)
    projects: list[str] = field(default_factory=list)
    certifications: list[str] = field(default_factory=list)
    technologies: list[str] = field(default_factory=list)
    years_experience: int = 0
    embedding: Optional[list[float]] = None

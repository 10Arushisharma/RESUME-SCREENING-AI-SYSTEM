from __future__ import annotations

from app.models.resume import JobDescription, ParsedResume


class ExplanationService:
    """Service responsible for translating scores into human-readable explanations."""

    def build_explanation(
        self,
        *,
        resume: ParsedResume,
        job_description: JobDescription,
        semantic_similarity: float,
        skills_score: float,
        experience_score: float,
        education_score: float,
        projects_score: float,
        certifications_score: float,
        overall_score: float,
    ) -> dict:
        matching_skills = [skill for skill in resume.skills if skill.lower() in {item.lower() for item in job_description.skills}]
        missing_skills = [skill for skill in job_description.skills if skill.lower() not in {item.lower() for item in resume.skills}]
        matching_technologies = [tech for tech in resume.technologies if tech.lower() in {item.lower() for item in job_description.technologies}]
        relevant_experience = resume.experience[:3]
        education_match = resume.education[:3] if resume.education else []
        experience_match = resume.experience[:3] if resume.experience else []
        project_match = resume.projects[:3] if resume.projects else []
        certifications_match = resume.certifications[:3] if resume.certifications else []

        strengths = []
        if matching_skills:
            strengths.append(f"Strong match on skills: {', '.join(matching_skills)}")
        if matching_technologies:
            strengths.append(f"Relevant technologies: {', '.join(matching_technologies)}")
        if relevant_experience:
            strengths.append("Relevant experience is present")
        if semantic_similarity >= 0.5:
            strengths.append("The resume shows strong semantic alignment with the role.")

        weakness_summary = "Resume would benefit from closer alignment with the job description and additional evidence of the requested technologies."
        if missing_skills and skills_score < 50.0:
            weakness_summary = f"The main gap is the absence of: {', '.join(missing_skills[:5])}."

        return {
            "candidate_name": resume.candidate_name,
            "overall_score": round(overall_score, 2),
            "semantic_similarity": round(semantic_similarity * 100.0, 2),
            "skills_score": round(skills_score, 2),
            "experience_score": round(experience_score, 2),
            "education_score": round(education_score, 2),
            "projects_score": round(projects_score, 2),
            "certifications_score": round(certifications_score, 2),
            "strengths": strengths,
            "missing_skills": missing_skills,
            "matching_technologies": matching_technologies,
            "relevant_experience": relevant_experience,
            "education_match": education_match,
            "experience_match": experience_match,
            "project_match": project_match,
            "certifications_match": certifications_match,
            "strength_summary": "The candidate demonstrates a meaningful match to the role based on their experience and technical profile." if strengths else "The candidate profile is too limited to establish a strong fit yet.",
            "weakness_summary": weakness_summary,
            "recommendations": [
                "Tailor the resume around the requested tools and responsibilities.",
                "Highlight measurable achievements tied to the role's core requirements.",
            ],
        }

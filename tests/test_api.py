from fastapi.testclient import TestClient

from app.main import app
from app.services.pdf_service import PDFService


client = TestClient(app)


def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_upload_job_description_and_rank_without_resumes():
    response = client.post(
        "/upload-job-description",
        data={"job_title": "Python Backend Engineer", "job_description": "Build APIs with FastAPI and Python."},
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["job_title"] == "Python Backend Engineer"
    assert payload["description"] == "Build APIs with FastAPI and Python."


def test_results_endpoint_empty():
    response = client.get("/results")
    assert response.status_code == 200
    assert response.json()["results"] == []


def test_extract_text_from_simple_text_file():
    service = PDFService()
    text = service.extract_text_from_bytes(b"Jane Doe\nPython backend engineer\nFastAPI and Docker", file_name="resume.txt")
    assert "Jane Doe" in text
    assert "FastAPI" in text


def test_upload_resume_and_rank_candidate():
    jd_response = client.post(
        "/upload-job-description",
        data={"job_title": "Python Backend Engineer", "job_description": "Build APIs with FastAPI and Python."},
    )
    assert jd_response.status_code == 200

    upload_response = client.post(
        "/upload-resumes",
        files=[
            (
                "files",
                (
                    "resume.txt",
                    b"Jane Doe\nExperienced Python backend engineer skilled in FastAPI, Docker, and API design.",
                    "text/plain",
                ),
            )
        ],
    )
    assert upload_response.status_code == 200
    assert upload_response.json()["uploaded_count"] == 1

    rank_response = client.post("/rank")
    assert rank_response.status_code == 200
    results = rank_response.json()["results"]
    assert len(results) == 1
    assert results[0]["candidate_name"] == "resume.txt"
    assert results[0]["overall_score"] >= 0
    assert results[0]["semantic_similarity"] >= 0

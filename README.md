# AI Resume Screening System

A production-ready FastAPI service for semantically screening resumes against a job description using embeddings, FAISS, and explainable ranking.

## Features

- Upload multiple PDF resumes
- Upload a job description
- Extract text with PyMuPDF
- Parse skills, education, experience, projects, and certifications
- Generate embeddings with Sentence Transformers
- Store vectors in FAISS
- Rank candidates with weighted scoring
- Return explainable results with strengths, gaps, and recommendations

## Architecture

- `app/api` exposes the REST endpoints
- `app/services` contains parsing, embedding, ranking, and explanation logic
- `app/vectorstore` manages FAISS vector storage
- `app/models` and `app/schemas` define the domain models and response payloads

## Installation

```bash
pip install -r requirements.txt
```

## Running locally

```bash
uvicorn main:app --reload
```

## Docker

```bash
docker compose up --build
```

## AI Flow

1. Resume ingestion:
   - Upload one or more resumes via `POST /upload-resumes`
   - Extract text from PDF or document files using PyMuPDF
2. Text processing:
   - Parse resume content into skills, experience, education, and project sections
   - Normalize and clean extracted text
3. Embedding generation:
   - Create semantic embeddings from resume text and job description text
   - Use `sentence-transformers/all-MiniLM-L6-v2` to capture meaning beyond exact keywords
4. Vector storage:
   - Store resume embeddings in a FAISS index for fast similarity search
5. Ranking and scoring:
   - Compare job description embedding against candidate embeddings
   - Compute cosine similarity and weighted scores including skill overlap and semantic fit
   - Return explainable ranking results with strengths and gaps

## AI Tools Used

- `Sentence Transformers` for semantic embeddings
- `FAISS` for vector similarity search
- `PyMuPDF` for document text extraction
- `OpenAI Codex` for assistance in code generation and logic refinement
- `Amazon Q` for inspiration in semantic search and question-answering design patterns

## API Endpoints

- `POST /upload-resumes`
- `POST /upload-job-description`
- `POST /rank`
- `GET /results`
- `GET /health`
- `GET /docs`

## API Documentation (Swagger)

After starting the server, open:

http://127.0.0.1:8000/docs

Interactive Swagger UI is available here.

Alternative OpenAPI schema:

http://127.0.0.1:8000/openapi.json

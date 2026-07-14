# AI Workflow

## Purpose

This document describes the AI workflow for the AI Resume Screening project, including the prompts provided to the assistant, the difficulties encountered, and the complete processing flow.

## Initial Prompts

The main prompts provided during this session included:

- "add readme here and AI flow in which add AI tool that i used like codex and amazon Q"
- "make one new file name as AI workflow included what propmpts i gave to u initially what were the difficulties faced and full flow"

Earlier work context also included requests to:

- fix AI Resume Screening ranking so semantic meaning matches count
- eliminate candidate accumulation and zero scores
- verify extraction, embeddings, cosine similarity, FAISS, and scoring
- improve semantic matching beyond exact keyword overlap

## Difficulties Faced

- Candidate accumulation across multiple resume uploads: the app was retaining old resumes and required a reset of in-memory state and FAISS index.
- Zero or stale ranking scores: resumes could receive zero similarity/score when embeddings or raw text were missing or when the fallback logic was incomplete.
- Semantic ranking mismatch: strong resumes were not always ranked correctly if they used different wording from the job description, so the scoring needed better semantic similarity handling.
- Debugging the pipeline: the ranking flow depended on extraction, parsing, embedding generation, vector storage, and similarity computations, which made it harder to isolate where the problem occurred.
- Integration of explanation logic: ensuring the returned results were not only ranked correctly but also explainable with strengths, gaps, and recommendations.

## Full AI Flow

1. Upload resumes
   - Users upload PDF or document files through `POST /upload-resumes`.
   - The app extracts raw text from resume files, using PyMuPDF for PDF extraction.

2. Resume text processing
   - The extracted text is normalized and cleaned.
   - The parser extracts structured data like skills, experience, education, and projects from resume text.

3. Job description input
   - Users submit a job description via `POST /upload-job-description`.
   - The job description text is stored and used as the target for ranking.

4. Embedding generation
   - The system creates semantic embeddings for resumes and the job description.
   - Uses `sentence-transformers/all-MiniLM-L6-v2` to capture semantic meaning beyond exact word matching.

5. Vector storage
   - Resume embeddings are stored in a FAISS index for fast similarity search.
   - The FAISS index is reset on new uploads to prevent stale candidates.

6. Ranking and scoring
   - The job description embedding is compared against resume embeddings.
   - Cosine similarity is computed to measure semantic fit.
   - A weighted score is produced using semantic similarity plus overlap of parsed skills and other structured fields.
   - The system provides ranking results that favor meaning-based matches.

7. Explanation
   - Ranked output includes why a candidate scored well or poorly.
   - The explanation layer summarizes strengths, gaps, and areas that matter for the role.

## AI Tools Used

- `Sentence Transformers` for semantic embedding generation
- `FAISS` for vector similarity search and fast retrieval
- `PyMuPDF` for PDF text extraction
- `OpenAI Codex` as an AI assistant for code generation and logic refinement
- `Amazon Q` as an inspiration for semantic search and QA design patterns

## Files Updated

- `README.md` was updated to include the AI flow and AI tools used.
- `AI_WORKFLOW.md` was created to document prompts, challenges, and the full flow.

## Notes

This document is intended to help future developers understand the AI pipeline, the assistant prompts that drove the changes, and the challenges addressed during development.
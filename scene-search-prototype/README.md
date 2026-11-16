# Group 8 — Scene Search Prototype 0.1

**Purpose:** Prove natural-language scene retrieval using FastAPI + PostgreSQL/pgvector before implementing video ingestion, CLIP, or the VLM.

**This is a demonstration, not a completed CCTV system.** Seeded footage names and timestamps are fictional. There is no actual CCTV video processing or identity verification at this stage.

## Prerequisites

- Docker container `group8-postgres` already running, with `group8_fyp` and `vector` extension enabled.
- Your existing Group 8 `.venv` contains FastAPI, Psycopg 3, pgvector, Sentence Transformers, Uvicorn and python-dotenv.
- PostgreSQL exposed locally on port `15432`.

## Run on Windows CMD

1. Unzip **this folder** into `C:\Users\hp\Documents\group8-visual-asset-retrieval\scene-search-prototype`.
2. From a CMD window in the project root:

   ```bat
   cd /d "%USERPROFILE%\Documents\group8-visual-asset-retrieval"
   .venv\Scripts\activate
   cd scene-search-prototype
   copy .env.example .env
   notepad .env
   ```

3. Replace the placeholder `DB_PASSWORD` value with your **real** local DB password, save, and close Notepad. Never share `.env` or commit it to Git.
4. Confirm Docker is running: `docker ps` (the `group8-postgres` container should be Up).
5. Seed demo data:

   ```bat
   python seed.py
   ```

   The first run downloads `sentence-transformers/all-MiniLM-L6-v2` (~90 MB model), so internet is required once. The model runs on CPU to preserve GPU resources. Subsequent runs use the local cache. Seed script safely skips existing identical samples.
6. Start the API:

   ```bat
   python -m uvicorn app:app --reload --host 127.0.0.1 --port 8000
   ```

7. Open `http://127.0.0.1:8000/docs` and try:
   - **GET /health** — confirm DB + pgvector connection
   - **GET /search** — `q=black bag on a chair`, `limit=3`
   - **POST /scenes** — use a JSON example to add your own description and timestamps

## Example POST /scenes body

```json
{
  "video_id": "demo_04.mp4",
  "camera_id": "CAM-04",
  "start_seconds": 40,
  "end_seconds": 47,
  "description": "A blue backpack is next to a white bottle on a table."
}
```

## Design notes

- All descriptions and queries are encoded using **the same** 384-dimensional Sentence Transformers model.
- PostgreSQL `vector(384)` stores them; `<=>` performs cosine-distance search.
- Scenes are returned with IDs, camera IDs, start/end seconds, descriptions and similarity scores.
- All SQL parameters are bound safely. Database connection is created per request for this simple local prototype; a connection pool belongs in the full application.
- This prototype uses **exact** ranking without an HNSW index because it only has a few records. Indexing and migrations belong in the production-oriented phase.
- The API has **no authentication**; keep it on `127.0.0.1` only. Do not expose it publicly or deploy it as production code.
- This sample does not compare incompatible CLIP and text-encoder embedding spaces.

## Next milestone

Add real video-file ingestion using OpenCV/FFmpeg and associate actual clip paths with each indexed scene. Follow with VLM scene captions, change-aware indexing, CLIP baseline, and evaluation.

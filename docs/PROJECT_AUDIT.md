# Group 8 project audit — 16 November 2026

## Scope and evidence

Inspected the existing Python source, tests, requirement snapshots, local reports,
tracking labels, Git state, video metadata and running databases. PostgreSQL was
queried with `default_transaction_read_only=on`; no seed, ingestion or indexing
command was run against it. Credentials were read locally through dotenv without
displaying their values. This is an engineering audit, not an exhaustive security
scan or a new model-accuracy benchmark.

The supplied brief is the current product scope. The root README and
`Group8_Antigravity_Handoff.md` describe an older project direction. In particular,
the existing `src/verification.py` and its CLI remain historical work; the new
application should not depend on that removed module or require cross-camera
physical identity matching.

## Actual structure and implementation

| Area | Verified state | Qualification |
| --- | --- | --- |
| `src/` | YOLO/BoT-SORT ingestion, SQLite sightings, track summaries, OpenCLIP crop/context embeddings, Chroma retrieval and CLI | A separate older application, not integrated with PostgreSQL/FastAPI |
| `scene-search-prototype/app.py` | `/health`, `POST /scenes`, `GET /search`; cached CPU MiniLM encoder | No upload, job, playback, camera or visual-search API |
| Stage 2 | Constant-FPS sequential decoding, change/guard sampling, frame manifest and PostgreSQL metadata | Defaults to five minutes maximum; a dry run still writes images and manifest |
| Stages 3–4 | Individual-frame VLM captioning, YOLO crops, local comparison reports | Stage 3 `--index` inserts predictions into search without a per-caption review gate |
| Stages 5–7 | Multi-frame prompts, overlapping temporal windows, explicit Stage 6 window approval, model comparisons | Reports demonstrate hallucinations; review CSVs currently have empty review fields |
| Stage 8 | Annotated tracking video and 223 label files in `scene-search-prototype/runs/detect/backpack_tracking/` | No saved tracker settings/run manifest alongside those files; settings in the brief are historical claims |
| Tests | 30 root unittest tests and 19 prototype pytest tests passed before changes | Root suite includes real local CLIP/Chroma tests and the historical verification module; fixtures are not accuracy measurements |
| Frontend | No React/TypeScript app or package manifest found | Not implemented |
| Deployment/streaming | Two local pgvector containers are running; FFmpeg/ffprobe installed | No versioned Docker Compose, Alembic migration tree, MediaMTX config, or RTSP ingestion service found in this repo |

The nested `scene-search-prototype/scene-search-prototype/` contains Stage 5/6
READMEs and six helper tests, not another complete application. Preserve it until
a separately tested documentation/test relocation. No `AGENTS.md` was found.

## Executed verification

- Python **3.13.17**. `pip check`: **No broken requirements found**.
- GPU: RTX 4060 Laptop, **8,188 MiB**, driver **621.14**. No GPU memory was in use
  at the initial check. This updates the older handoff's driver information.
- `test01.mp4`: H.264, **640 × 360**, **24 FPS**, **240 frames**, **10 seconds**
  according to ffprobe. CFR timing is suitable for this initial import; arbitrary
  variable-frame-rate recordings need decoder presentation timestamps later.
- FastAPI TestClient: `/health` **200**, pgvector **0.8.7**; `/search` **200** using
  the cached MiniLM model with network access disabled. The brief's movement query
  ranked scene **15**, **0–4 s**, first at cosine similarity **0.9132**.
- The root suite's real OpenCLIP/Chroma search assertions passed. Therefore CLIP
  retrieval already exists locally in the older stack; Stage 10 should adapt it
  to the current recording model and pgvector, not repeat an experiment from zero.
- Saved Stage 5 report: **3491.8 ms**, **5643.7 MiB**; saved Qwen short-prompt
  report: **4159 ms**, **4183.6 MiB**. These are historical report measurements,
  not fresh GPU benchmarks, throughput claims, or total system memory usage.
- Visually inspected saved frames at **4 s** and **9 s**: the backpack is on the
  desk in both. `STAGE7_GROUND_TRUTH.json` supplies the earlier human-reviewed
  sequence. A new frame-level tracking ground truth has not yet been created.

## Database: inspected live, not assumed

Correct target: `group8-postgres`, PostgreSQL **17.12**, pgvector **0.8.7**,
loopback port **15432**. Another container, `group8-pgvector`, runs PostgreSQL 18
on **5433**; do not confuse the two. Both observed port bindings are loopback only.

| Table | Rows | Important columns and relationships |
| --- | ---: | --- |
| `scenes` | 15 | id, video/camera IDs, numeric start/end, description, embedding_model, **vector(384)**, created_at; unique video/range/description, nonnegative start and end > start |
| `video_recordings` | 1 | video_id PK, local_video_path, camera_id, fps, duration_seconds, created_at |
| `frame_candidates` | 9 | video FK, numeric timestamp/clip bounds, frame_path, change_fraction, reason, nullable description; unique video/timestamp |
| `temporal_window_index` | 1 | video FK, scene FK, model_id, prompt_version, window bounds, JSONB candidate IDs/times, caption, inference/memory data, report_path |

All 15 scene embeddings use `sentence-transformers/all-MiniLM-L6-v2`, 384
dimensions. Only the 0–4 s temporal window is indexed, as scene 15. There is no
`ai_frame_captions` table in this database. The scene table also contains seeded
fictional recording references: ordinary `/search` can return these alongside
real footage. `scenes.video_id` has no recording FK; do not add one blindly before
classifying those seed records. JSONB temporal candidate references have no
individual FK. Only primary/unique B-tree indexes exist; no vector ANN index.
At this scale an exact scan is appropriate.

The older `outputs/sightings.db` has **1 ingestion run**, **386 sightings**
(158 laptop, 228 person), not backpack observations. Its SQLite authority and
Chroma index are separate from the database above.

## Dependencies

Installed versions include torch **2.14.0+cu130**, torchvision **0.29.0+cu130**,
Ultralytics **8.4.202**, OpenCV **5.0.0.93**, NumPy **2.5.4**, OpenCLIP **3.3.0**,
Chroma **1.5.9**, FastAPI **0.162.0**, psycopg **3.3.8**, pgvector Python client
**0.5.1**, sentence-transformers **6.3.0**, transformers **5.23.0**, accelerate
**1.15.0**, SQLAlchemy **2.1.8**, Alembic **1.20.1**, pytest **9.2.0**.

Alembic/SQLAlchemy being installed does not mean migrations or ORM models exist.
The root pinned requirements describe the older application and omit several
prototype dependencies; prototype requirements are unpinned and omit CV/VLM
requirements. The full `requirements-working.txt` snapshot is useful evidence,
not a portable CUDA installation procedure. Reconcile manifests in a later
reproducibility milestone; no upgrades or installs are needed for Stage 9.

## Weaknesses, bugs and risks to address

1. **Two data authorities and conflicting scope documentation.** Choose
   PostgreSQL for the integrated product; preserve SQLite/Chroma as a baseline.
   Do not route new work through the legacy verification module.
2. **Destructive legacy reingestion.** `src/ingest.py` clears previous evidence
   JPGs before processing; `SightingStore.create_ingestion_run` deletes prior runs
   by default, cascading sightings. A failure can lose the preceding successful
   result. New jobs should be immutable runs and switch active results only after
   successful completion. Do not run old ingestion on existing outputs casually.
3. **Unsafe source naming if exposed through an API.** Legacy evidence paths use
   caller-supplied camera IDs and video stems, and deletion uses those paths. Use
   generated storage IDs and resolved containment checks before web exposure.
4. **Invented timing fallback.** Legacy `inspect_video_file` substitutes 25 FPS
   for missing metadata. That produces unsupported timestamps; reject missing FPS
   or use decoded timestamps instead. Its ingestion also retains all sightings and
   several full-frame buffers per track, unsuitable for long recordings.
5. **Unreviewed text can enter search.** Stage 3 `--index` and public `POST /scenes`
   have no durable review identity/status. General search does not expose source
   provenance and mixes seed/manual/reviewed-AI rows. Preserve Stage 6's explicit
   approval behavior and add provenance-aware search before automated indexing.
6. **Runtime DDL.** Stage scripts create tables opportunistically. No schema
   baseline/migration history exists; add introspection-tested Alembic migrations
   and backup instructions before schema evolution, without truncating data.
7. **Prototype access limits.** API has no authentication/authorization or request
   quotas. Bound SQL parameters are used, and DB failures return generic errors,
   which are good. Keep current API on loopback until access control is implemented.
   No upload/RTSP endpoints exist yet, so their safeguards still need implementing.
8. **Blocking/scaling behavior.** CPU embedding and one DB connection per request
   are acceptable for the demo; use pooling and move indexing out of handlers.
   Chroma search currently fetches every matching candidate and can prune on a
   search request; this is not a long-footage retrieval architecture.
9. **Temporal import validation is partial.** Stage 6 validates recording/camera,
   range and candidate membership, but does not re-derive each candidate timestamp
   and its membership in the stated window. Saved reports are trusted local input.
10. **Export ambiguity and append behavior.** YOLO text rows have optional IDs;
    missing files do not establish object absence or completed processing. The
    installed exporter appends to existing TXT files, so repeated `exist_ok=True`
    exports can duplicate/conflict. Stage 9 must reject duplicates and retain
    export gaps instead of manufacturing clean continuity.
11. **Tests and academic claims.** Prototype tests are split across nested folders;
    a crop test globally replaces `auto_caption_frames` in `sys.modules`. Root live
    tests depend on exact local outputs and can initialize/mutate legacy stores.
    Keep them as baselines but later isolate immutable fixtures. Uncalibrated
    heuristics and blank review CSVs do not establish detection, event or retrieval
    accuracy. No multi-recording evaluation dataset was found.
12. **Repository hygiene.** Existing modified/untracked source predates this task.
    The prototype's `.env` is ignored, but its frames/reports are not ignored by
    its current rules and the whole prototype is untracked. Never `git add .`
    without reviewing footage, secrets and generated artifacts. No footage or
    credentials were staged in this task.

## Preserve and next milestone

Preserve all original source, model files, `.env`, both databases, Chroma data,
Stage 2 manifests, manual labels, Stage 3–7 raw reports, Stage 7 ground truth and
Stage 8 video/labels. Preserve the older encoder and identity scoping as reusable
ideas, while separating the removed verification module from the new scope.

The best immediate milestone is a **read-only Stage 8 label importer plus
deterministic image-coordinate trajectory analysis**. It can produce a runnable
Stage 9 result without GPU loading, database migration, speculative identity
merging or a new frontend. See `ARCHITECTURE.md` for the incremental product plan
and `STAGE9.md` for commands, results and limitations.

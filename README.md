# Group 8 — Evidence-Guided Multi-Camera Visual Asset Retrieval System

**Milestone 3: OpenCLIP + ChromaDB Description-Based Visual Asset Retrieval & Synchronization-Aware Indexing**

## Overview & Problem Statement

Reviewing CCTV footage across multiple cameras to locate a misplaced physical asset (such as a laptop or bag) is slow and error-prone when the exact time or camera is unknown. Rather than requiring manual video scrubbing or over-claiming object identity across cameras, this system:
1. Ingests prerecorded videos from multiple cameras with independent local tracking namespaces and deterministic source content fingerprints (`sha256`).
2. Aligns cross-camera timelines **only** when explicit recording start datetimes (`--start-time`) or relative synchronization offsets (`--time-offset`) are provided — never inventing CCTV timestamps.
3. Separates **raw detector confidence** from **heuristic within-camera track stability**, distinguishes **latest raw class detections** from **last strongly supported sightings**, and returns **`INCONCLUSIVE`** whenever cross-camera synchronization, tracking loss, simultaneous multi-camera sightings, or unlinked camera views prevent a verified final location.
4. Supports **natural-language visual search** (e.g., `"black laptop"`, `"person holding a black laptop"`, `"black desk telephone"`) using **OpenCLIP** (`ViT-B-32`, `laion2b_s34b_b79k`) and **ChromaDB** over both **tight object crops (`crop`)** and **context-preserving regions (`context`)**, while keeping SQLite authoritative and never conflating uncalibrated CLIP cosine similarity with physical object identity.

---

## Architecture

- [`src/models.py`](src/models.py):
  - Data schemas: [`VideoMetadata`](src/models.py#L166-L220), [`SightingRecord`](src/models.py#L224-L330), [`TrackSummary`](src/models.py#L334-L416), [`CameraRetrievalSummary`](src/models.py#L420-L454), and [`RetrievalReport`](src/models.py#L458-L564).
  - Identity isolation: [`SightingRecord.isolation_key`](src/models.py#L256-L260) and [`SightingRecord.scoped_track_key`](src/models.py#L262-L270) isolate every track by `run<run_id>:<camera_id>:<source_video>:local_track_<id>` (and per-frame keys for untracked detections).
- [`src/storage.py`](src/storage.py):
  - [`SightingStore`](src/storage.py#L21-L454) persists `ingestion_runs` and `sightings` in **SQLite** (`outputs/sightings.db`) and **JSONL** (`outputs/sightings.jsonl`). SQLite is the authoritative source of truth for all detections and evidence metadata.
- [`src/ingest.py`](src/ingest.py):
  - [`ingest_video`](src/ingest.py#L171-L327) and [`ingest_multiple_videos`](src/ingest.py#L330-L380) run **YOLO11m + BoT-SORT** (`stream=True`) on CUDA device `0` when available, saving annotated evidence frames and crops under `outputs/evidence/<camera_id>/<video_stem>/`.
- [`src/retrieval.py`](src/retrieval.py):
  - [`determine_synchronization_mode`](src/retrieval.py#L145-L187) and [`build_retrieval_report`](src/retrieval.py#L420-L781) compute per-camera summaries, cross-camera conflict/departure observations, and calibrated conclusion statuses (`STRONGLY_SUPPORTED`, `UNCERTAIN_CANDIDATE`, `INCONCLUSIVE`).
- [`src/visual_search.py`](src/visual_search.py):
  - [`OpenCLIPEmbedder`](src/visual_search.py#L75-L169) loads OpenCLIP (`ViT-B-32`, `laion2b_s34b_b79k`, 512-dim) once onto CUDA (`cuda:0`) when available and computes L2-normalized image and text embeddings without gradients.
  - [`extract_context_region_pil`](src/visual_search.py#L188-L217) and [`VideoFrameContextReader`](src/visual_search.py#L220-L256) extract expanded context-preserving regions around detected objects from clean source video frames (falling back to saved evidence frames).
  - [`ChromaVisualIndex`](src/visual_search.py#L413-L796) manages the persistent ChromaDB vector index at `outputs/chroma_db` (`sighting_visual_embeddings`), stores deterministic IDs (`s<sighting_id>_<variant>_<model_id>`), automatically prunes stale embeddings when SQLite runs are replaced, and resolves natural-language queries against SQLite sightings grouped by isolated candidate tracks.
- [`src/comparison.py`](src/comparison.py):
  - [`analyze_tracking_gap_and_candidates`](src/comparison.py#L87-L390) investigates within-camera tracking gaps, concurrent person carrier trajectories, and pre-arrival Frame 0 background stationarity (`NCC`) vs. HSV histogram similarity.
- [`src/cli.py`](src/cli.py):
  - CLI providing `ingest`, `ingest-batch`, `query`, `compare-tracks`, `index-embeddings`, `search`, and `index-stats`.
- [`tests/test_indexer_and_retrieval.py`](tests/test_indexer_and_retrieval.py):
  - 24 automated unit and live-verification tests covering single-camera indexing, multi-camera evaluation scenarios, and OpenCLIP + ChromaDB visual retrieval safeguards.

---

## Environment, Pretrained Models & Data Storage Locations

Verified local environment (`.venv`, Windows 11, NVIDIA GeForce RTX 4060 Laptop GPU 8GB VRAM):
- Python `3.13.17`, PyTorch `2.14.0+cu130`, TorchVision `0.29.0+cu130`
- Ultralytics `8.4.202` (`yolo11m.pt` + `botsort.yaml`)
- OpenCLIP (`open_clip_torch`) `3.3.0` — Pretrained model: `ViT-B-32` (`laion2b_s34b_b79k`, 512-dim L2-normalized embeddings, cached under `~/.cache/huggingface/hub/`)
- ChromaDB `1.5.9`, FastAPI `0.162.0`, Uvicorn `0.54.0`, OpenCV `5.0.0.93`

### Storage Locations
- **Authoritative SQLite Database**: `outputs/sightings.db`
- **JSONL Sighting Export**: `outputs/sightings.jsonl`
- **Saved Evidence Frames & Crops**: `outputs/evidence/<camera_id>/<video_stem>/{frames,crops}/`
- **Annotated Tracking Videos**: `outputs/annotated_videos/<camera_id>_<video_stem>/`
- **Persistent ChromaDB Vector Index**: `outputs/chroma_db/` (collection: `sighting_visual_embeddings`)

See [`requirements.txt`](requirements.txt) for pinned package versions.

---

## CLI Usage & Reproducing the Laptop Visual Search Experiment

Run all commands from `C:\Users\hp\Documents\group8-visual-asset-retrieval` using `.\.venv\Scripts\python.exe`.

### 1. Build / Rebuild the OpenCLIP + ChromaDB Visual Index

```powershell
# Build (or rebuild from scratch) crop + context embeddings for all SQLite sightings with saved evidence
.\.venv\Scripts\python.exe -m src.cli index-embeddings --rebuild

# Inspect persisted embedding statistics and SQLite synchronization status
.\.venv\Scripts\python.exe -m src.cli index-stats
```

### 2. Run Natural-Language Description Searches (`laptop_test.mp4`)

```powershell
# Search 1: "black laptop"
.\.venv\Scripts\python.exe -m src.cli search --query "black laptop" --top-k 10

# Search 2: "person holding a black laptop"
.\.venv\Scripts\python.exe -m src.cli search --query "person holding a black laptop" --top-k 10

# Search 3: "black desk telephone" (investigates the stationary telephone false-positive track)
.\.venv\Scripts\python.exe -m src.cli search --query "black desk telephone" --top-k 10

# Optional filters: restrict by camera, detected class, embedding variant ('crop' or 'context'), or JSON output
.\.venv\Scripts\python.exe -m src.cli search --query "laptop sitting on a wooden desk" --camera-id office_cam_01 --class-name laptop --variant context --top-k 5
.\.venv\Scripts\python.exe -m src.cli search --query "black laptop" --top-k 5 --json
```

### 3. Class-Based Multi-Camera Ingestion, Retrieval & Gap Comparison

```powershell
# Class-based retrieval report across all indexed cameras
.\.venv\Scripts\python.exe -m src.cli query --class-name laptop --show-timeline

# Within-camera tracking gap & candidate visual comparison on laptop_test.mp4
.\.venv\Scripts\python.exe -m src.cli compare-tracks --camera-id office_cam_01 --video laptop_test.mp4 --class-name laptop
```

### 4. Run Automated Test Suite (24 Tests)

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

---

## Known Limitations & Boundaries

1. **CLIP Similarity Is Not Calibrated Probability or Physical Identity**: OpenCLIP cosine similarity (typically `0.18–0.34` for `ViT-B-32` image-text pairs) is an uncalibrated relative ranking score. It measures visual/semantic resemblance to the text description, **not** ground-truth verification that two sightings show the exact same physical object.
2. **Pretrained Detector & Small/Occluded Object Challenges**: OpenCLIP only indexes sightings where the object detector (`YOLO11m` at `conf >= 0.15`) recorded a candidate bounding box and saved evidence. During complete detector blind gaps (such as Frames `111..177` in `laptop_test.mp4` when the laptop is occluded by the employee's torso), no laptop bounding boxes exist to embed, though concurrent `person` tracks (`local_track_2`) remain searchable.
3. **False-Positive Candidate Disambiguation Requires Multi-Factor Verification**: In `laptop_test.mp4`, `local_track_7` (Frames `178..208`) was triggered on a stationary black desk telephone. While OpenCLIP ranks `local_track_1` highest for `"black laptop"` (`0.3300` vs. `0.2259` for `local_track_7`) and ranks `local_track_7` highest for `"black desk telephone"` (`0.2869`), combining visual embeddings with pre-arrival background stationarity (`NCC`), track stability, and spatiotemporal reasoning remains essential.
4. **Synchronization Requires Explicit Metadata**: Cross-camera chronological ordering is only performed when `recording_start_utc` (`--start-time`) or `time_offset_seconds` (`--time-offset`) is provided for every compared recording.

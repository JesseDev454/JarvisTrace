> Current project: **Text-Guided Surveillance Video Retrieval for Portable Objects**. Start with the [project audit](docs/PROJECT_AUDIT.md), [architecture](docs/ARCHITECTURE.md), [Stage 9 review guide](docs/STAGE9_REVIEW.md), and [Stage 10 PostgreSQL visual search](docs/STAGE10.md). The legacy CLI and verification experiments below are preserved historical work; the removed verification module is not part of the current application plan.

# Group 8 — Evidence-Guided Multi-Camera Visual Asset Retrieval System

**Milestone 4: Reference Asset Anchoring, Multi-Factor Candidate Verification & OpenCLIP + ChromaDB Retrieval**

## Overview & Problem Statement

Reviewing CCTV footage across multiple cameras to locate a misplaced physical asset (such as a laptop or bag) is slow and error-prone when the exact time or camera is unknown. Rather than requiring manual video scrubbing or over-claiming object identity across cameras, this system:
1. Ingests prerecorded videos from multiple cameras with independent local tracking namespaces and deterministic source content fingerprints (`sha256`).
2. Aligns cross-camera timelines **only** when explicit recording start datetimes (`--start-time`) or relative synchronization offsets (`--time-offset`) are provided — never inventing CCTV timestamps.
3. Separates **raw detector confidence** from **heuristic within-camera track stability**, distinguishes **latest raw class detections** from **last strongly supported sightings**, and returns **`INCONCLUSIVE`** whenever cross-camera synchronization, tracking loss, simultaneous multi-camera sightings, or unlinked camera views prevent a verified final location.
4. Supports **natural-language visual search** (e.g., `"black laptop"`, `"person holding a black laptop"`, `"black desk telephone"`) using **OpenCLIP** (`ViT-B-32`, `laion2b_s34b_b79k`) and **ChromaDB** over both **tight object crops (`crop`)** and **context-preserving regions (`context`)**, while keeping SQLite authoritative.
5. Anchors a **reference asset track** across multiple representative appearance states (e.g., open on desk, peak confidence, closed/carried) and evaluates candidate tracks across 6 independent evidence factors, producing calibrated outcomes (`SUPPORTED_CANDIDATE`, `CONTRADICTED_CANDIDATE`, or `INCONCLUSIVE`) without ever fabricating probabilities or silently merging tracks.

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
  - [`OpenCLIPEmbedder`](src/visual_search.py#L87-L182) loads OpenCLIP (`ViT-B-32`, `laion2b_s34b_b79k`, 512-dim) once onto CUDA (`cuda:0`) when available and computes L2-normalized image and text embeddings without gradients.
  - [`ChromaVisualIndex`](src/visual_search.py#L437-L855) manages the persistent ChromaDB vector index at `outputs/chroma_db` (`sighting_visual_embeddings`), provides explicit OS file-handle cleanup via [`ChromaVisualIndex.close`](src/visual_search.py#L465-L485), automatically prunes stale embeddings when SQLite runs are replaced, and resolves natural-language queries against SQLite sightings.
- [`src/verification.py`](src/verification.py):
  - [`build_reference_asset_anchor`](src/verification.py#L345-L505) anchors a reference track/sighting with its multi-appearance states (`first_appearance_state`, `peak_confidence_state`, `last_strongly_supported_state`, `track_fadeout_last_state`), bounding-box geometries, BGR patches, and OpenCLIP crop/context embeddings.
  - [`verify_candidates_against_reference`](src/verification.py#L508-L654) evaluates candidate tracks independently across 6 separate evidence factors: (1) OpenCLIP crop/context similarity, (2) HSV color histogram & patch correlation, (3) temporal ordering & camera synchronization, (4) detector confidence & heuristic track stability, (5) pre-arrival Frame 0 background stationarity (`NCC`), and (6) spatial/temporal track conflicts.
- [`src/comparison.py`](src/comparison.py):
  - [`analyze_tracking_gap_and_candidates`](src/comparison.py#L87-L390) investigates within-camera tracking gaps and concurrent person carrier trajectories.
- [`src/cli.py`](src/cli.py):
  - CLI providing `ingest`, `ingest-batch`, `query`, `compare-tracks`, `index-embeddings`, `search`, `index-stats`, and `verify-candidates`.
- [`tests/test_indexer_and_retrieval.py`](tests/test_indexer_and_retrieval.py):
  - 30 automated unit and live-verification tests covering single-camera indexing, multi-camera evaluation scenarios, OpenCLIP + ChromaDB visual retrieval, and reference-anchored candidate verification.

---

## CLI Usage & Reproducing Experiments

Run all commands from `C:\Users\hp\Documents\group8-visual-asset-retrieval` using `.\.venv\Scripts\python.exe`.

### 1. Anchor Reference Track & Verify Candidate Tracks (`laptop_test.mp4`)

```powershell
# Anchor local_track_1 in laptop_test.mp4 (office_cam_01) and verify candidate tracks (local_track_7 and local_track_12)
.\.venv\Scripts\python.exe -m src.cli verify-candidates `
  --ref-camera-id office_cam_01 `
  --ref-video laptop_test.mp4 `
  --ref-track-id 1 `
  --class-name laptop

# Emit full structured JSON verification report (with all 6 separate evidence factor dictionaries)
.\.venv\Scripts\python.exe -m src.cli verify-candidates `
  --ref-camera-id office_cam_01 `
  --ref-video laptop_test.mp4 `
  --ref-track-id 1 `
  --json
```

**Observed Results on `laptop_test.mp4` (`office_cam_01`):**
- **Reference Anchor (`run1:office_cam_01:laptop_test.mp4:local_track_1`):** `STRONGLY_SUPPORTED` (`stability=0.95`, frames `0..110`, last strongly supported at Frame `107` / `+4.458s`). Captures 4 representative states from open on foreground desk (`210.3x203.6 px` at Frame `0`) to closed/carried (`108.9x44.5 px` at Frame `107`).
- **Candidate 1 (`run1:office_cam_01:laptop_test.mp4:local_track_7`, frames `178..208`):** **`CONTRADICTED_CANDIDATE`** — Although OpenCLIP crop similarity against the thin closed laptop reaches `0.7955`, Frame 0 pre-arrival background `NCC = 0.9851` (`>= 0.90`) and HSV color correlation (`max = 0.1152`) prove the tight bounding box encloses the pre-existing stationary black desk telephone.
- **Candidate 2 (`run1:office_cam_01:laptop_test.mp4:local_track_12`, frames `200..215`):** **`INCONCLUSIVE`** — Supporting evidence includes OpenCLIP crop similarity (`0.7721` against closed state), continuous person carrier trajectory (`local_track_2`), and lower-left sub-region divergence from Frame 0 (`placed_subregion_ncc = 0.7706`). However, low detector confidence (`mean = 0.26`, `0/16` frames `>= 0.50`), partial background conflation with the adjacent desk phone (`full-box Frame 0 NCC = 0.8279`), and spatial nesting with `local_track_7` require returning `INCONCLUSIVE` without merging.

### 2. Build / Rebuild OpenCLIP + ChromaDB Index & Run Description Search

```powershell
.\.venv\Scripts\python.exe -m src.cli index-embeddings --rebuild
.\.venv\Scripts\python.exe -m src.cli index-stats
.\.venv\Scripts\python.exe -m src.cli search --query "black laptop" --top-k 10
.\.venv\Scripts\python.exe -m src.cli search --query "person holding a black laptop" --top-k 10
.\.venv\Scripts\python.exe -m src.cli search --query "black desk telephone" --top-k 10
```

### 3. Run Automated Test Suite (30 Tests)

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

---

## Requirements for Proper Real-World Two-Camera Evaluation

Synthetic fixtures validate software logic and safeguard rules, **not** real-world cross-camera re-identification accuracy. A credible two-camera experimental evaluation requires:
1. **Authorized Dual-Camera Recording Setup**: Two physical cameras (e.g., `office_cam_01` and `corridor_cam_02`) recording at `>= 720p` (`1280x720`) and `>= 15–24 FPS` with documented camera placement (non-overlapping or partially overlapping fields of view).
2. **Synchronized Wall-Clock or Offset Metadata**: Verified NTP wall-clock start times (`--start-time`) or a visual/audio sync clap at recording start (`--time-offset`) so cross-camera transition gaps are measured accurately.
3. **Pre-Arrival Clean Background Frames**: At least `2–3 seconds` of unobstructed background footage at the start of each recording (`Frame 0`) before the target asset enters the destination area, enabling pre-arrival stationarity (`NCC`) checks.
4. **Controlled & Distractor Object Protocols**:
   - Positive transitions: A person carries the target asset from Camera 1 into Camera 2 and places it down.
   - Distractor objects: A second similar-looking laptop or dark office fixture (e.g., desk phone, binder) already present in Camera 2 or carried simultaneously by another person.
   - Negative departures: The asset leaves Camera 1 and enters an unmonitored room (never appearing in Camera 2) to test false last-location suppression.
5. **Independent Ground-Truth Annotations**: Manually logged frame numbers, timestamps, and camera IDs for every true appearance, disappearance, and placement event prior to running retrieval and verification.

---

## Known Limitations & Boundaries

1. **CLIP Similarity Is Not Physical Identity**: Even a stationary desk telephone (`local_track_7`) can have a `0.7955` image-to-image OpenCLIP cosine similarity with a thin closed black laptop (`local_track_1` at Frame `108–110`) because both appear as small dark rectangular slabs at low resolution. Multi-factor verification (combining Frame 0 stationarity `NCC`, HSV color correlation, and track stability) is mandatory to reject false positives.
2. **Selective Keyframe Sampling (`keyframe_interval=12`)**: During ingestion, evidence images are saved for track first, peak, last strongly supported, last overall, every 12th frame, and all low-confidence frames (`< 0.50`). While this preserves boundary and transition states (`20/111` frames for `local_track_1`) without exhausting disk space, rapid sub-second pose transitions between 12-frame intervals are only captured if they coincide with peak/last milestones or drop below `0.50` confidence.
3. **Synchronization Requirement**: Cross-camera transition verification returns `INCONCLUSIVE` whenever `recording_start_utc` or `time_offset_seconds` is missing on either recording.

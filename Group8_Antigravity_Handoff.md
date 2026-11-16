# GROUP 8 — ANTIGRAVITY DEVELOPMENT HANDOFF

**Prepared:** November 4, 2026
**Status:** Working local computer-vision environment; baseline detection and tracking experiments completed. Next milestone: evidence/sighting extraction and retrieval.
**Project:** Design and Implementation of an Evidence-Guided Multi-Camera Visual Asset Retrieval System

> **How to use this file:** Give this document to the Antigravity coding agent as background context. It is a *structured export of our ChatGPT work and decisions*, **not** a verbatim chat transcript. Distinguish verified local test outputs from inferred observations, and inspect the actual workspace before editing files.

## 1. What we are building

We are Group 8, a six-member final-year Software Engineering project team. We propose a software-first system that searches multiple CCTV recordings to help locate missing or misplaced physical assets (e.g., laptops, bags, office equipment) and returns the **last sighting supported by video evidence**, rather than requiring someone to scrub through footage manually.

**The central user problem:** Reviewing video from several cameras to find a missing item is slow and cumbersome, particularly when the precise time or camera is unknown. Do **not** describe the project mainly as a “lost-and-found app” or open with the niche problem of identical laptops. First explain the practical footage-search problem.

**Example user question:** “Where was the black laptop last seen?”

**Desired response (when supported):** Camera/location, local video timestamp, relevant frame/video clip, detection/matching evidence, and calibrated confidence or uncertainty. If the final location cannot be verified, **say that explicitly**, and differentiate confirmed from possible sightings.

**Important capabilities to develop, incrementally:**
- Ingest prerecorded CCTV videos from one or more known camera sources, with camera IDs and time metadata.
- Detect candidate assets and people per frame and track objects *within* each camera feed.
- Store frame-level sightings (object label, track ID, time, coordinates, confidence, evidence reference, camera ID).
- Search for objects using labels initially; later support natural language and image/text embeddings.
- Rank or link plausible sightings across time and cameras using evidence and **without falsely asserting identity**.
- Return evidence clips/frames and explicitly describe unresolved identity, occlusion, and blind spots.
- Support integration with video from CCTV/IP cameras eventually, but **no hardware or live camera connectivity is needed for the first prototype**.

The aim is a working, demonstrable system, not just an annotated YOLO video. Use licensed/authorized footage; synthetic footage is acceptable for prototyping but should not be presented as validation on real CCTV data.

## 2. Local development machine and workspace

- **OS:** Windows, PowerShell terminal; IDE: **Antigravity** (already in use, do not recommend VS Code or PyCharm as a prerequisite).
- **Laptop:** HP Victus 16, Intel Core i7 H-series-class CPU.
- **GPU:** NVIDIA GeForce RTX 4060 Laptop GPU, approximately **8 GB VRAM**.
- **RAM:** 16 GB DDR5.
- **SSD:** Samsung 990 EVO Plus, **2 TB** (recently installed after cloning from a 512 GB SSD).
- **NVIDIA driver (confirmed via `nvidia-smi`):** **592.82**; driver reports **CUDA Version 13.1**. This is a *driver capability*, not proof of the standalone CUDA Toolkit being installed.
- **Project path:** `C:\Users\hp\Documents\group8-visual-asset-retrieval`
- **Python virtual environment:** `.venv` inside project. Use `\.venv\Scripts\python.exe` relative to project root for all installs/scripts.
- `python` and `python -m pip` now work from a fresh PowerShell after PATH repair. The `py` launcher previously did not resolve; avoid requiring it.

### Confirmed installed package versions

| Component | Local state |
|---|---|
| Python | **3.13.17** |
| pip | **26.2.2** |
| PyTorch | **2.14.0+cu130** |
| TorchVision | **0.29.0+cu130** |
| NumPy | **2.5.4** |
| OpenCV (`cv2`) | **5.0.0** (`opencv-python` 5.0.0.93 installation log) |
| Ultralytics | **8.4.202** |
| Pillow | **11.3.0**, adjusted by dependency installation |
| `lap` | **0.5.14**, auto-installed by Ultralytics |
| YOLO model files | `yolo11n.pt` and later `yolo11m.pt` downloaded/used |

Other Python dependencies installed transitively. **Do not assume that OpenCLIP, ChromaDB, FastAPI, FFmpeg, Node, Git, Postman, etc. have been installed**—they were discussed/planned, but not all explicitly verified. Inspect the environment before installing.

### Verified GPU tests

These succeeded on November 4, 2026:

```powershell
.\.venv\Scripts\python.exe -c "import torch; print('CUDA:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'Not detected')"
# CUDA: True
# GPU: NVIDIA GeForce RTX 4060 Laptop GPU

.\.venv\Scripts\python.exe -c "import cv2; print('OpenCV:', cv2.__version__)"
# OpenCV: 5.0.0

.\.venv\Scripts\python.exe -c "from ultralytics import YOLO; print('YOLO import successful')"
# YOLO import successful
```

**Prior Windows troubleshooting (resolved):** Initial Python 3.13.17 installer failed with `0x80070570`; initial download SHA-256 differed from the official checksum. Redownloaded installer hash matched, repaired Python, added its location to user PATH, then verified `python --version` and pip. Do not re-troubleshoot unless a new issue appears.

## 3. Experimental work already completed

### Experiment A — Simple image object detection: PASSED

Ran:

```powershell
.\.venv\Scripts\python.exe -c "from ultralytics import YOLO; model = YOLO('yolo11n.pt'); model.predict(source='https://ultralytics.com/images/bus.jpg', device=0, save=True)"
```

- GPU inference completed successfully.
- Detected 4 people and 1 bus in sample image.
- Reported around 48.3 ms inference for that sample; one run, **not a general benchmark**.
- Saved to `runs/detect/predict`.

### Experiment B — Pedestrian video tracking: PASSED as a pipeline test, not an accuracy validation

- Used a 62-frame video saved as `test.mp4`.
- Ran YOLO11n + BoT-SORT on CUDA device 0.
- Roughly 17–22 people detected per frame; occasional backpack detection.
- Reported approximately 20 ms inference per frame in that particular short run.
- Saved outputs to `runs/detect/track`, then `runs/detect/track-2`.
- `lap` was installed as a tracker dependency.
- Ultralytics warned about memory buildup without streaming, so long video processing should use **`stream=True`** and consume results incrementally.
- Video inspection showed tracking IDs can appear consistent for some people but also false classifications/ID changes. **No tracking accuracy metric was computed.**

Example:

```powershell
.\.venv\Scripts\python.exe -c "from ultralytics import YOLO; from collections import deque; model = YOLO('yolo11n.pt'); deque(model.track(source='test.mp4', tracker='botsort.yaml', device=0, save=True, stream=True), maxlen=0)"
```

The `deque(..., maxlen=0)` was just a quick command-line way to consume the generator; implement a proper `for result in results:` loop to **save metadata and evidence**, rather than throwing results away.

### Experiment C — Real CCTV-style dataset clips: baseline struggled

Two videos downloaded from the **CAVIAR surveillance dataset**:
- `datasets/caviar/LeftBag.mpg`
- `datasets/caviar/LeftBag_PickedUp.mpg`

Source: https://homepages.inf.ed.ac.uk/rbf/CAVIARDATA1/

CAVIAR footage here is low resolution (~384 × 288). Separate scenarios, **not** synchronized views of the same event.

- YOLO11n produced unreliable bag detections and some visibly wrong labels (e.g., bird/skateboard on people).
- YOLO11m with `imgsz=960`, `conf=0.25`, `classes=[0,24,26,28]` gave cleaner-looking outputs but still did **not** reliably detect/track the relevant bags.
- Class filters can suppress inappropriate labels **without improving true object recognition**; don't conflate removal of displayed false labels with improved metrics.
- Simply upscaling model input to 960 does not recover detail lost in the original video.
- Conclusion: avoid endlessly cycling pretrained model sizes on these clips. Use controlled higher-resolution footage and later consider targeted annotation/fine-tuning only if justified.

One representative command that processed both:

```powershell
.\.venv\Scripts\python.exe -c "from ultralytics import YOLO; from collections import deque; model = YOLO('yolo11m.pt'); videos = ['LeftBag_PickedUp.mpg', 'LeftBag.mpg']; [(print('Processing:', video), deque(model.track(source='datasets/caviar/' + video, tracker='botsort.yaml', device=0, imgsz=960, conf=0.25, classes=[0,24,26,28], save=True, stream=True, project='runs/experiments', name=video.replace('.mpg', '_yolo11m')), maxlen=0)) for video in videos]"
```

### Experiment D — Controlled synthetic office-laptop footage: best current test case

Generated a video using Google Flow / Veo 3.1 Fast from a fixed CCTV-style office reference image. The office has a **large foreground desk containing a black laptop** and a **back-right desk near the door**. Scenario: employee approaches laptop, picks it up, carries it toward back-right desk, places it there.

Several early generated clips were problematic (office redesign, duplicate laptop, detached keyboard, obscured final location). The **latest chosen clip** was deemed sufficiently coherent for an initial experiment, with the final desk placement somewhat obscured. It is **synthetic** and does **not** constitute realistic ground truth.

- Expected local filename: `laptop_test.mp4` in project root.
- Ran YOLO11m + BoT-SORT with `imgsz=960`, `conf=0.15`, `classes=[0,63]` (0 = person, 63 = laptop), CUDA device 0.

```powershell
.\.venv\Scripts\python.exe -c "from ultralytics import YOLO; from collections import deque; model = YOLO('yolo11m.pt'); deque(model.track(source='laptop_test.mp4', tracker='botsort.yaml', device=0, imgsz=960, conf=0.15, classes=[0,63], stream=True, save=True, project='runs/experiments', name='laptop_tracking'), maxlen=0)"
```

- The resulting annotated video was shared as `laptop_test.avi`.
- Visual review: laptop initially detected as **track ID 1** at foreground desk, roughly 0.92 confidence at first; person as **track ID 2**; laptop still tracked near pickup (~0.81 confidence around 4 seconds).
- Around ~5 seconds the detector/tracker **loses the laptop** as it is carried.
- A **possible** later laptop detection around the back-right desk showed a **new ID 7**, low confidence about **0.21**; final frames did not provide a reliable confirmed laptop sighting.
- These numbers are **observations from annotated-video inspection, not an exported CSV or independently measured benchmark**. If implementing automated logic, use the underlying model outputs and video frames, not these approximate descriptions as ground truth.

**Critical interpretation:** Do NOT automatically claim the laptop's final verified location was the back-right desk. Distinguish:
- **Last confirmed detection** (foreground/pickup region).
- **Possible later sighting** (back-right area, tentative; tracking ID changed and confidence low).
- **Uncertainty** due to occlusion, low-confidence recognition, lost tracking.

This is a concrete case where the evidence-guided aspect matters.

## 4. Current project stage vs. what is NOT implemented

**Done and verified:** Python+GPU installation, image inference, video inference/tracking, a few experimental annotated videos.

**Not yet built or demonstrated:**
- Persisting detection or tracking data to a structured store.
- Video timestamp alignment with real clock times.
- Camera ingestion schema and multi-camera cross-camera matching.
- Automated last-seen retrieval/search API and UI.
- Natural-language search/embedding retrieval.
- Object identity calibration, confidence fusion, evidence chain, occlusion reasoning.
- Formal precision/recall, tracking IDF1/HOTA, retrieval accuracy, uncertainty calibration evaluation.

The user has **Antigravity IDE**, wants to **ship a working implementation**, and prefers avoiding redundant installation steps.

## 5. Immediate next development task (recommended)

**Build a minimal, local, reproducible *single-camera sighting indexer*** before jumping to cross-camera matching or React screens.

### Functional target

Given a video file and an explicit `camera_id` (e.g., `office_cam_01`):

1. Iterate frames with YOLO11m + BoT-SORT using `stream=True` and CUDA device 0 if available.
2. For each candidate detection, persist: `camera_id`, `source_video`, `frame_index`, `timestamp_seconds`, `class_id`, `class_name`, `track_id` if present, `confidence`, bounding box (`xyxy`), and path to saved evidence frame or relevant clip.
3. Preserve separate per-camera/per-video track namespaces. IDs are **not** global object identities.
4. Save small contact frames/crops only as needed, avoiding enormous frame dumps; keep source video and evidence paths associated.
5. Expose a CLI command to print sightings for class `laptop`, ordered chronologically; return **last confirmed detection** and separately list low-confidence or possibly connected sightings, without silently stitching IDs.
6. Run against `laptop_test.mp4` and demonstrate that the data is reconstructable and the uncertain later prediction is not asserted as verified.
7. Add basic tests for timestamp calculation, serialization, confidence filtering, missing/empty detections, and output ordering.

**Pragmatic initial storage:** JSONL or SQLite are acceptable. SQLite is preferable once repeated video imports and queries matter. Use a simple modular Python codebase and environment-specific `requirements`/dependency snapshot.

**Potential initial layout** (suggested, not an existing tree):

```text
group8-visual-asset-retrieval/
  .venv/                   # existing; do not commit
  datasets/caviar/         # existing/expected
  laptop_test.mp4          # expected/available locally
  src/
    ingest.py              # video frame + tracking data extraction
    storage.py             # SQLite/JSONL persistence
    retrieval.py           # timestamped sightings and evidence ranking
    models.py              # data schema
  tests/
  outputs/                 # indexes/evidence generated locally
  requirements.txt
  .gitignore
```

**Acceptance criteria**: a command can ingest `laptop_test.mp4`, produce a persistent record of detections with camera/time/confidence, and a second command can answer a simple `laptop` last-seen query with correctly linked evidence and an honest uncertainty statement. It should **not** claim multi-camera identity matching yet.

### Suggested first agent actions

- Inspect the **actual** folder contents and `git status` before changing anything. Never overwrite existing user code, generated assets, or datasets blindly.
- Confirm the `.venv` interpreter and CUDA using tests above.
- Validate file paths/video metadata (`fps`, `frame_count`, codec) with OpenCV or available utilities.
- Implement incremental processing and structured storage; avoid accumulating full inference results in RAM.
- Verify timestamps are video-relative initially; do not invent absolute wall-clock times without camera/video metadata.
- Keep `datasets/`, `runs/`, large videos/weights, `.venv/`, and local caches out of Git; do not delete the user’s files.
- Document tradeoffs and what remains unproven.

## 6. Planned later stack (not all confirmed installed)

- **OpenCV + FFmpeg:** frame extraction, decoding, clipping, output conversion.
- **YOLO11m / PyTorch + CUDA:** candidate object detection (current baseline).
- **BoT-SORT:** within-camera tracking (baseline).
- **OpenCLIP:** joint text/image embeddings for queries such as “black laptop” (later).
- **ChromaDB:** embedding similarity retrieval (later; not a source of ground-truth identity).
- **SQLite then PostgreSQL if justified:** structured camera/event/evidence metadata.
- **FastAPI:** local/backend API for video ingestion, queries, and evidence retrieval (later).
- **React + TypeScript + Vite:** user interface for video upload, search, sightings, uncertainty and evidence clips (later).
- **Pytest:** automated tests; Postman optional for API manual testing.
- **CVAT:** optional annotation for ground truth and evaluation; can be hosted/used online as appropriate.

Avoid automatically installing every prospective tool or training large models before a reproducible baseline exists. `yolo11m.pt` is a model file, not a distinct framework.

## 7. Dataset and evaluation constraints

- **CAVIAR**: freely accessible sample video sequences; very low resolution; useful for failure modes but not an appropriate sole validation dataset. Observe licensing/attribution terms at source.
- **ABODA**: another potential abandoned-object dataset; **not confirmed downloaded**.
- **MEVA**: potential multi-camera research dataset; large, select clips/obtain appropriate access rather than downloading everything; **not confirmed downloaded**.
- **Controlled real recordings**: strongly recommended later, preferably recordings where the team can annotate exact asset locations/times/identity and has consent/authorization.
- Avoid privacy-violating data collection; document legal authority to process any externally provided CCTV footage.
- Evaluate not just detector precision/recall but **last-seen retrieval accuracy**, frame/temporal localization error, evidence validity, cross-camera linkage accuracy, and whether uncertainty is calibrated.

## 8. Important boundaries and reminders for the coding agent

1. The user wants **practical implementation**, not a generic AI essay.
2. Do not confuse **detection confidence** with certainty that a sighting is the *same physical item* across time/cameras.
3. BoT-SORT's IDs can change when an item is occluded or detection fails; IDs aren't cross-camera identities.
4. The model's inability to spot the bag in 384 × 288 footage is not a CUDA failure.
5. The synthetic office clip has a visually implied destination but no reliably confirmed final detection, so report limits.
6. The primary problem is **time-consuming manual retrieval of misplaced assets from multiple CCTV recordings**. Make this central in app UX and documentation; similar-looking items are an advanced challenge, not the opening pitch.
7. Prioritize **traceable evidence and honest uncertainty** over a polished but unsupported answer.
8. If dependencies need installation, inspect installed versions and install **inside `.venv`** rather than globally.
9. Real camera hardware is **not** required for the current phase; prerecorded streams suffice.
10. The team's submitted FYP proposal is the academic baseline. This handoff describes development experiments and intentions, not a replacement for that proposal.

---

**Ready-to-build task:** Create the single-camera YOLO sighting indexer with timestamped evidence records and a minimal query; evaluate it on `laptop_test.mp4` before implementing natural-language retrieval and multi-camera matching.

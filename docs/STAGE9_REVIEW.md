# Stage 9b human frame review

The local review tool uses the existing Python environment and saved Stage 9a
reports. It does not load models or change PostgreSQL, tracking exports, prediction
JSONL or the original human-review CSV. The separate SQLite file stores human
annotations with append-only revision history and detects conflicting edits.

## Start on Windows PowerShell

From the repository root:

```powershell
.\.venv\Scripts\python.exe -m src.stage9_review `
  --report outputs/stage9/20261116T164229528159Z/analysis.json `
  --reviewer "Goodluck Kassa"
```

Open http://127.0.0.1:8765 . If the tool is already running, just open the page;
do not start a second server on the same port. Ctrl+C stops it. An alternative
`--port` is supported. The default annotation database is
`human_review.sqlite3` beside the selected report. Keep this file: it contains
review history. JSON history can be obtained through the page's export link.

The server binds only to 127.0.0.1. This is a local research tool, without user
accounts or a production access-control system. Do not expose it on a network.
Only the selected video's numbered frames are served; no arbitrary file endpoint
exists. Source video hashes are checked when the server starts.

## Review procedure

1. Begin with the interaction jump (approximately 2.8 s), then step through to
   4.2 s. Use the later jump at 6 s and review through the final frame.
2. Orange boxes are unreviewed detector predictions. Toggle them off when judging
   the actual image. Every frame is accessible, including frames with no detection.
3. Enter your reviewer name and choose whether the target is visible, absent or
   uncertain. "Absent" means no visible target in that frame, not physical absence
   from the room or a verified departure.
4. For visible objects, enter a manual object ID and optionally a visually
   supported surface, then drag a green bounding box. For multiple objects use
   different IDs. Predictions are never automatically copied into human boxes.
   Use the same manual ID across frames only where identity is visually supported.
5. Inspect surrounding frames before selecting a motion label. One frame cannot
   establish movement. Leave motion unknown, surface blank, or visibility uncertain
   when evidence is insufficient. Record ambiguities in notes.
6. Save the frame annotation. Corrections create new revisions; a stale browser
   edit is rejected until the frame is reloaded. Unsaved changes prompt before
   navigation. Use Clear human boxes to redraw a mistaken box.

The Play frames control is a slow review sequence, not a real-time video player.
Frames are decoded individually, so observed playback speed does not measure
indexing throughput or source frame rate.

## Verification and current limits

Prerequisite check decoded frames 67, 84, 100, 144, 216 and 239 successfully at
640 x 360. Existing FastAPI, Uvicorn, OpenCV, Pillow, httpx and pytest are present;
no new dependency installation was needed.

Five new integration tests use tiny generated videos and temporary review stores.
They cover frame images, missing/out-of-range detections, persistence/revisions,
invalid annotations, cross-origin write rejection, video hash mismatch and wrong
report scope. Browser checks on the actual clip verified frame jumps and overlays.
No real-footage human labels were created during those checks.

```powershell
.\.venv\Scripts\python.exe -m unittest tests.test_stage9_review -v
$env:HF_HUB_OFFLINE = '1'
$env:TRANSFORMERS_OFFLINE = '1'
$env:PYTHONPATH = (Join-Path (Get-Location) 'scene-search-prototype')
.\.venv\Scripts\python.exe -m pytest -q
```

The combined Python suite passed 77 tests (28 subtests). The initial review tool
explicitly accepts backpack reports (class 24); other target classes are rejected
to prevent mismatched review instructions. Form editing is disabled during frame
loading and saving so drafts cannot silently disappear during requests.
Human annotation is
still required; this tool does not establish accuracy or automatically reconcile
track IDs. A subsequent evaluator should compare reviewed boxes and trajectories
with predictions, distinguish uncertain/unreviewed frames, and report sample
size rather than infer accuracy from one clip. Motion labels here are per-frame
review context; interval-level movement evaluation needs explicit interval labels.

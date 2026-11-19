# Stage 13 — G507 real-footage evaluation preparation

Stage 13's inspection, clip preparation, evaluation tooling and bounded processing
checks are implemented. A [first four-query positive-subset evaluation](evaluation/STAGE13_RESULTS.md)
now uses human-confirmed AI-assisted intervals: visual Hit@1 1/4, Hit@5 2/4,
mean Precision@5 0.30 and bounded candidate MRR 0.3242. Four flagged queries
remain excluded and G507 has no reviewed index descriptions. **The full
effectiveness evaluation is not complete.** This small subset is not general accuracy.

A separate [object-crop experiment](evaluation/STAGE13_OBJECT_CROPS.md) now
reproduces that baseline exactly and improves Hit@1 to 4/4 and mean Precision@5
to 0.65 on the same four known positives. The isolated index contains 140
frames and 39 unreviewed detector crops. These preliminary results require
new verified queries and held-out testing before generalization claims.

[Validation round 2](evaluation/STAGE13_VALIDATION_ROUND2.md) now has six new
30-second clips, 90 frame vectors and 36 crop proposals prepared with frozen
settings. Its [confirmed results](evaluation/STAGE13_VALIDATION_ROUND2_RESULTS.md)
show combined Hit@1 2/2 and Precision@5 0.70 on the two confirmed positives;
all three confirmed-negative queries return candidates. q03 stays unresolved.
A candidate-inspection frontend prototype can begin; reliable no-match behavior
and broad effectiveness remain unestablished.

An [AI-assisted interval draft](evaluation/STAGE13_INTERVAL_DRAFT.md) is now
available with 27 observation/action entries and click-to-play proposed times.
The original template remains frozen; four positive queries were subsequently
confirmed, while flagged actions and negative queries still require review.

## What was inspected

- Existing upload/registration, extraction, YOLO, CLIP, PostgreSQL schemas,
  retrieval modes, playback and tests: [project audit](STAGE13_PROJECT_AUDIT.md).
- All **11** actual G507 AVI files, including the specified nine morning files
  and two additional afternoon recordings: [inventory summary](evaluation/STAGE13_INVENTORY.md).
- Every source fully decoded with FFmpeg; declared/decoded frame counts matched.
  All are H.264, 1920×1072, nominal 30 fps. Nine morning filenames are contiguous
  from 10:20 to 11:05, with small media-duration overruns rather than established
  frame-perfect capture continuity. Eleven files contain about 55 minutes of media.
- Contact sheets for every file, followed by denser previews for selected clips.
  Bags, backpacks, handled cases and crowding are useful candidates. Small items,
  placement/pickup actions, occlusion bounds and absence require continuous review.
- The first original five-minute AVI has 8,998 available decoded timestamps,
  approximately 2/30s ahead of frame-index/FPS; the last two are unavailable.
  Time-based cuts consequently shifted source content. Extraction now selects
  decoded frame ranges and resets the derived CFR timeline; new IDs end `_f2`.
  Offsets use decoded-frame/FPS units, not AVI PTS or verified capture UTC.

## Local artifacts

Inventory remains under `datasets/mevid/G507/evaluation/inventory/`. Active corrected review artifacts are under `datasets/mevid/G507/evaluation/frame_aligned/`; the first derivatives remain preserved in the parent evaluation directory.

- Parent `inventory/`: `inventory.json`, `inventory.csv`, `contact_sheets/`.
- `clips/<video_id>/video.mp4` and `provenance.json`: **seven 40-second clips**,
  totaling 4 minutes 40 seconds, from four morning recordings. H.264 MP4 clips
  retain the original geometry, omit audio and map to selected decoded source frames. Filename wall-clock offsets remain nominal.
- `clips_manifest.json`: source/derivative hashes, selected source frame, camera/date/filename clock,
  frame counts, offsets, candidate scenario and explicit verification status.
- `previews/`: 12 sampled frames per clip, with frame/time sidecars.
- `annotations.proposed.json`: **eight unverified query hypotheses** (six proposed
  positive/action questions, two proposed negative questions). Visibility intervals
  remain unset; negative hypotheses are not confirmed absence labels.
- `review.html`: continuous clip players, source mapping and proposed observations.

The [local independent-review pack](http://127.0.0.1:8772/frame_aligned/review.html) is running.
Review this **before** looking at retrieval predictions. Follow the
[annotation and metric protocol](evaluation/STAGE13_PROTOCOL.md), preserve the
draft, and save a separate `annotations.verified.json`. A reviewer must confirm
every relevant interval across all seven clips, or reject unsupported queries.

Reports under `outputs/stage13/` include raw retrieval/detector diagnostics,
indexing runs, 21 corrected source/derivative frame mapping checks (maximum mean absolute pixel difference 1.70/255 after lossy encoding), source/DB preservation,
container timestamp verification and `evaluation_pending.json`.

## Actual processing and search checks

| Run | Scope/settings | Stored vectors | Elapsed incl. repeat | Repeat inserts |
|---|---|---:|---:|---:|
| Small smoke | 2 clips, every 4s, max 10, batch 2 | 20 | 23.98s | 0 |
| First collection (preserved) | 7 clips, every 2s, max 20, batch 2 | 140 | 63.03s | 0 |
| Corrected frame-aligned collection | 7 `_f2` clips, every 2s, max 20, batch 2 | 140 | 65.86s | 0 |
| Original AVI smoke | One complete 300s original, every 30s, max 10, batch 2 | 10 | 45.95s | 0 |

Each run used its own marked `stage13_...` PostgreSQL schema. Public data was not
used for new indexing. The active corrected collection schema is `stage13_g507_eval_f2`; the first collection is preserved as `stage13_g507_eval`. The smoke
schemas are `stage13_g507_smoke` and `stage13_g507_original_smoke`.

Peak PyTorch CUDA allocation was 630,009,344 bytes (about 0.59 GiB), and process
RSS **at the end** of the corrected seven-clip run was 1,371,443,200 bytes (about 1.28 GiB).
These are process measurements, not total VRAM or peak system-RAM guarantees.
The long-source smoke is deliberately sparse; it does not establish retrieval
coverage between samples or sustained full-dataset processing readiness.

Twenty-four warmed query/mode API requests (eight queries × three modes) completed
against the seven-clip collection. All returned windows stayed within the selected
recording collection. Media returned a valid 206 byte-range response, and a
derived clip played to its measured 40-second end in the browser.

Reviewed text coverage is **0/7**. Text-only returned no candidates; hybrid results
were identical to visual-only for all eight diagnostic queries. Each visual and
hybrid diagnostic returned the requested top ten windows, including the unverified
negative hypotheses. These are raw behavior checks, not verified relevance or
false-positive accuracy. Cold warm-up costs and warm request timings are saved
separately in `frame_aligned_retrieval_diagnostics.json`; earlier diagnostics remain preserved.

Warm API median latency over these eight diagnostic queries was 0.1174s visual,
0.2768s text and 0.3799s hybrid (loopback HTTP included). The initial visual/text
warm-ups took 13.931s and 6.500s; hybrid then reused both loaded encoders. Text's
zero coverage makes this a timing smoke test, not a fair effectiveness comparison
or a controlled production benchmark.

Long-video readiness is therefore mixed: multi-file indexing/search, camera IDs,
source mappings and duplicate prevention are exercised; capture-date/UTC search,
continuous stream jobs and full-density 45-minute processing remain unverified or
unimplemented. Dates and nominal wall-clock mappings are preserved in the manifest,
while the application itself still returns relative video times.

## Observed problems and targeted next experiments

- **Small-object sensitivity:** on two inspected frames YOLO11n returned no
  portable candidates at imgsz 640, but returned backpack candidates at 1280.
  Other detections changed category or disappeared. Ten raw inference runs on
  five frames are insufficient for detector recall. Compare resolution/crop
  strategies against verified boxes before selecting a setting.
- **Full-frame visual context:** the current CLIP preprocessing resizes then center
  crops. With a wide G507 frame, this discards portions of the sides, including
  entry areas. Small bags also occupy little of the encoded image. Test detector
  crops or full-image padding as a separately versioned preprocessing/index
  experiment; changing preprocessing requires a new encoder identity and reindex.
- **No semantic rejection:** ranking returns candidates for any description.
  Verified negative queries are needed to assess this; do not tune an absence
  threshold on the same seven selected clips.
- **Temporal sampling:** two-second samples can miss brief transitions; placement
  cannot be proved from a before/after pair. Evaluate short event visibility first,
  then test a denser sampling baseline with measured resource cost.
- **Missing text coverage:** no independent reviewed G507 captions exist. Do not
  embed the ground-truth answers as search descriptions to manufacture hybrid
  performance. Prepare/freeze separate reviewed descriptions if testing this mode.
- **Scaling:** full PNG storage, exact vector scans and per-range verified media
  snapshots have I/O costs. Workload estimates precede execution; no 45-minute
  full-density GPU job was run. No retraining or model replacement was introduced.

Safe tooling fixes include interrupted-extraction cleanup and atomic clip-plus-
provenance publication; immutable registry profiles and qualified isolated-schema
guards; verified-only metric gating; source-validated text coverage qualification;
decoded-frame-based extraction with a real FFmpeg frame-mapping regression; and
warmed latency measurement. A missing local table now fails instead of falling
through to public tables. The existing Stage 12 app remains available unchanged.

Reusable code added: `src/stage13_dataset.py`, `stage13_prepare.py`,
`stage13_index.py`, `stage13_evaluate.py` and `stage13_preview.py`, each with a
corresponding test module. The preview reuses Stage 12's app/player; dependencies
remain the existing OpenCV, FFmpeg, Ultralytics, cached encoders and PostgreSQL stack.

## Reproduce

Use the existing environment from the repository root. Source video and generated
artifacts stay ignored by Git. No dependency upgrades or large model downloads
were introduced. Source/model caches must already exist.

```powershell
$env:HF_HUB_OFFLINE = '1'
$env:TRANSFORMERS_OFFLINE = '1'

.\.venv\Scripts\python.exe -m src.stage13_dataset --root datasets/mevid/G507 --output datasets/mevid/G507/evaluation/inventory --full-decode
.\.venv\Scripts\python.exe -m src.stage13_prepare --root datasets/mevid/G507 --output datasets/mevid/G507/evaluation/frame_aligned --selections docs/evaluation/stage13_g507_candidates.json

# Show the workload first; no DB/model initialization without --execute.
.\.venv\Scripts\python.exe -m src.stage13_index --manifest datasets/mevid/G507/evaluation/frame_aligned/clips_manifest.json --schema stage13_g507_eval_f2 --output-root outputs/stage13/collection --every-seconds 2 --max-frames 20 --max-recordings 7 --batch-size 2
# Add --execute to index. Repeated existing samples add zero vector rows.

# Independent annotation review; leave this running while reviewing.
.\.venv\Scripts\python.exe -m http.server 8772 --bind 127.0.0.1 --directory datasets/mevid/G507/evaluation

# Optional retrieval preview, after independent annotation review:
.\.venv\Scripts\python.exe -m src.stage13_preview --schema stage13_g507_eval_f2 --storage-root datasets/mevid/G507/evaluation/frame_aligned/clips --port 8771

# Once the separate verified file actually exists; use a new output name per run:
.\.venv\Scripts\python.exe -m src.stage13_evaluate --annotations datasets/mevid/G507/evaluation/frame_aligned/annotations.verified.json --schema stage13_g507_eval_f2 --output outputs/stage13/verified_evaluation_run1.json

$env:GROUP8_TEST_POSTGRES = '1'
$env:PYTHONPATH = (Join-Path (Get-Location) 'scene-search-prototype')
.\.venv\Scripts\python.exe -m pytest --import-mode=importlib -q
```

Extraction reuses matching derived bundles, rejects provenance mismatches and
preserves the existing annotation draft on rerun. Evaluation refuses to overwrite
its output. Use complete, reviewed labels for every clip in the manifest. The
evaluator reports Hit@1, Hit@5, Precision@5, bounded-candidate MRR, warm latency and
negative candidate-return behavior only where valid; it does not invent Recall@5.

## Verification and completion status

**168 tests and 85 subtests passed** with PostgreSQL integration enabled. Coverage
includes metric fixtures, label/scope/source validation, incomplete text coverage,
schema isolation, immutable registration, workload caps, interrupted extraction, exact decoded-frame extraction,
preview wiring and the existing application regressions. Independent review
findings were fixed and rechecked.

All 11 original file hashes, seven active derivative hashes and the frozen draft
were verified. Six public table content digests match the prior application
baseline. New indexing exists only in dedicated schemas. No surveillance footage
was sent to external services; source licensing/citations are recorded in
[attribution](evaluation/MEVID_ATTRIBUTION.md).

**Stage 13 remains incomplete beyond the first confirmed positive subset.**
Next: investigate the observed object/state ranking failures, resolve ambiguous
action queries, establish full-scope negatives and broaden independent review.
If comparing text/hybrid, separately prepare reviewed index descriptions first.

Stage 14 requires the [minimum MediaMTX integration plan](STAGE13_PROJECT_AUDIT.md),
verified capture timestamp provenance, bounded segment jobs and reconnect tests.
No running MediaMTX process or repository MediaMTX configuration was found during
this check; the user-reported external configuration still needs locating and
verification. No major MediaMTX implementation was started.

# Stage 9a — saved-track import and image-space analysis

Implemented in `src/object_state.py` (pure analysis) and `src/stage9.py` (CLI and
artifacts), with tests in `tests/test_object_state.py` and `tests/test_stage9_cli.py`.
No existing experiment was rewritten, no weights were loaded/downloaded and no
database changes were applied. Source is being preserved on `codex/stage9-review-evaluation-20261116`.
Runtime footage, credentials, databases and generated reports remain local.

## Run on Windows PowerShell

From the project root:

```powershell
Set-Location 'C:\Users\hp\Documents\group8-visual-asset-retrieval'
.\.venv\Scripts\python.exe -m src.stage9 `
  --video scene-search-prototype/videos/test01.mp4 `
  --labels scene-search-prototype/runs/detect/backpack_tracking/labels `
  --camera CAM-TEST `
  --video-id test01_863e5601e3b5 `
  --confirm-save-conf `
  --confirm-cfr-stride-one `
  --assume-static-camera
```

The default output is a new UTC timestamp directory beneath `outputs/stage9/`.
`--output <fresh-directory>` selects another location. Existing output directories
are rejected, even if empty. An interrupted partial report is preserved; choose
a fresh directory for another attempt. Omit `--assume-static-camera` to retain
spatial metrics without assigning low/movement candidate states to usable windows.

The two confirmation flags are input-format assertions, not interactive prompts.
The old saved export lacks a run manifest. This importer requires confidence
columns and one-based suffixes from a **full, stride-one, constant-FPS** video
export. Do not use it on confidence-free TXT, segmentation rows, sampled-frame
exports, partial/renumbered videos or variable-frame-rate footage. For other
exports, first provide an adapter with trustworthy frame/time metadata. Label
filenames and a source hash recorded *now* cannot prove the original export came
from that exact file; the report states that limitation.

The installed Ultralytics `Results.save_txt` implementation and its
[official reference](https://docs.ultralytics.com/reference/engine/results/)
confirm the optional confidence/track-ID columns and append-to-file behavior.

## Outputs and interpretation

- `observations.jsonl`: each target-class detection, zero-based frame index,
  unrounded timestamp, pixel bbox/center, detector score, nullable local ID,
  source label/line, camera/recording and scope key. Untracked rows remain separate.
- `analysis.json`: source video/export/code SHA-256 hashes, metadata, import
  versions, assumptions, thresholds, missing-export ranges, per-track summaries,
  local spatial windows, gaps, possible fragment pairs, concurrent tracks and all
  latest observed detector candidates. Written last as the completed-report marker.
- `human_review.csv`: predictions with **blank** human identity/box/surface/motion
  fields. It is a review worksheet, not newly manufactured ground truth.

Track IDs are scoped to the recording/camera/source-export hashes, never linked
across sources. There is no database access and no automatic text indexing. The
default class is COCO backpack **24**; `--class-id` supports another exported class.
Original model/tracker versions/settings were not saved with the labels; the
current installed package versions cannot establish those historical settings.

The spatial metric is the diagonal of the axis-aligned envelope of observed bbox
centers, divided by the image diagonal. Net first-to-last center displacement is
also reported. Extent captures slow drift and excursions that return to the
starting point, while avoiding accumulated per-frame jitter as a movement score.
An envelope diagonal can exceed the distance between any particular pair of
observed centers; it is explicitly a summary statistic.

Tracks split at every missing frame. Default one-second windows overlap by one
observed boundary frame **within the same uninterrupted track segment** so a
change at a window edge is retained. Window counts therefore overlap; use the
top-level unique counts for coverage. No interpolation bridges gaps.

Defaults are exploratory heuristics:

| Parameter | Default | Meaning |
| --- | ---: | --- |
| `--window-seconds` | 1.0 | Local bucket size; boundaries overlap by one frame |
| `--low-extent` | 0.01 | Low center-envelope extent / image diagonal |
| `--movement-extent` | 0.03 | Larger extent produces an image-displacement candidate |
| `--minimum-span-seconds` | 0.5 | Minimum observed span for interpreting low/intermediate extent |
| `--fragment-gap-seconds` | 1.0 | List temporally nearby nonoverlapping IDs as unresolved pairs |

At 640 × 360, the low and larger-extent cutoffs are approximately **7.34 px** and
**22.03 px**. They are not calibrated probabilities or validated universal motion
thresholds. A large adjacent displacement may produce a candidate even within a
short window; one observation cannot. Camera motion must be assessed separately.

Low image displacement does not prove physical stationarity. Bbox pose changes,
camera motion, jitter, false positives and occlusion matter. A fragment pair means
only temporal proximity; it does not establish matching identity. Missing files
mean no exported observation, not confirmed object absence or a completed capture.
Latest candidates are unreviewed detections, not guaranteed present-day locations.

## Actual backpack export result

Final executed report:
`outputs/stage9/test01-stage9-20261116-v2/analysis.json`.
An earlier `test01-audit-20261116` output is preserved but predates the boundary
overlap and exact-integer review corrections; use the v2 report above.

The importer read **223 label files**, with **376 rows across all classes**.
There are **197 backpack observations in 197 distinct video frames**, including
**16 untracked detections**. The first label suffix 1 is frame index 0/time 0.

| Local ID | Observations | First–last time (seconds) | Internal missing target frames |
| --- | ---: | --- | --- |
| 17 | 38 | 1.375000–3.208333 | 69–73 and 75–76 |
| 26 | 2 | 3.458333–3.500000 | None within this very short fragment |
| 27 | 141 | 4.125000–9.958333 | None |

The latest observed detector candidate is **frame 239**, **9.958333 s**, track
**27**, score **0.660495**, bbox approximately
`[469.15, 159.47, 556.54, 266.37]` pixels. Camera is the caller-supplied **CAM-TEST**;
there is no verified recording-start UTC or inferred physical surface in the
machine report. This candidate is not a claim of a uniquely identified bag.

Track 27's full-span center extent is **9.19 px** and net displacement **0.51 px**.
Its 6.0833–7.0833, 7.0833–8.0833, 8.0833–9.0833 and 9.0833–9.9583 s windows are
low-displacement candidates under the declared static-camera assumption.
The 5.0833–6.0833 s window is indeterminate (**8.89 px** extent), illustrating why
the report does not label every post-placement sample stationary.

Possible fragment pairs 17→26, 17→27 and 26→27 remain unresolved. The 17→27 endpoint
center difference is **82.64 px** over **0.9167 s**, with **21 intervening frames**
without that ID continuity. That is a spatial difference between fragments, not
proof of one object's trajectory. The true relocation is not automatically
reconstructed: movement-phase detections are short, gappy and fragmented.

Existing `STAGE7_GROUND_TRUTH.json` describes the backpack staying on the desk at
6, 7, 8 and 9 s. Visual inspection of saved 4 s and 9 s images agrees with desk
placement, and late track-27 low-displacement candidates are qualitatively
consistent with that reviewed sequence. These checks do **not** measure tracking
identity correctness, detector precision/recall or movement accuracy. The blank
worksheet still needs independent frame/box review before those measurements.

## Tests and reproduction

```powershell
# Fast Stage 9 suite: 19 tests
.\.venv\Scripts\python.exe -m unittest tests.test_object_state tests.test_stage9_cli -v

# Root suite, including existing local model/data tests
$env:HF_HUB_OFFLINE = '1'
$env:TRANSFORMERS_OFFLINE = '1'
.\.venv\Scripts\python.exe -m unittest discover -s tests -v

# Prototype tests (includes nested Stage 5/6 helpers)
$env:PYTHONPATH = (Join-Path (Get-Location) 'scene-search-prototype')
.\.venv\Scripts\python.exe -m pytest scene-search-prototype -q

# All collected tests in both development paths
.\.venv\Scripts\python.exe -m pytest -q
```

The corrupt-video negative test intentionally triggers FFmpeg/OpenCV's
`moov atom not found` diagnostic; the expected exception is handled and the test
passes. Existing model tests can emit cached-model or library warnings. Tests
with real legacy outputs are environment-dependent and may initialize legacy
stores; new Stage 9 tests use temporary files only.

Tests cover hand-calculated geometry/timing, numeric ordering, missing/untracked
rows, malformed/duplicate labels, exact integer identity, slow drift, window-edge
displacement, gaps, concurrent objects, camera assumptions, corrupt input,
input hashes, blank review fields and refusal to overwrite. Independent code
review found and rechecked the integer/window-boundary corrections.

Final combined run: **72 tests passed, 23 subtests passed** in **51.44 s** using
the all-tests command above. This includes 30 original root tests, 19 new tests,
and 23 prototype tests. PostgreSQL row counts remained 15 scenes, 9 candidates,
1 recording and 1 temporal window. Final report code hashes match delivered code.

During this task, another `scene-search-prototype/object_state.py`, its four tests
and `data/stage9/` outputs appeared in the shared checkout. They were preserved
and are a separate local implementation, not dependencies of this CLI. Their
tests increased prototype discovery from the original 19 to 23; no result from
that implementation is presented here as this change's output.

## Next increment

Create independent box/identity annotations around the 2.8–4.2 s interaction and
late 6–10 s interval, including false positives, occlusion and class flips. Assess
movement candidates and fragmented IDs against those annotations before adding
association rules. Then adapt the already working OpenCLIP encoder to pgvector
as Stage 10, retaining separate 512-d visual and 384-d text spaces.

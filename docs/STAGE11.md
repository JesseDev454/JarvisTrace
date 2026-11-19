# Stage 11: hybrid window retrieval

This increment combines the Stage 10 CLIP image index with existing MiniLM scene
descriptions in a read-only command-line search. It returns fixed video windows,
source/camera identifiers, evidence and each channel's rank contribution. It does
not add a web search screen or live-stream ingestion.

## Run

From the repository root in PowerShell, with Stage 10 already indexed:

```powershell
$env:HF_HUB_OFFLINE = '1'
$env:TRANSFORMERS_OFFLINE = '1'
.\.venv\Scripts\python.exe -m src.stage11 search --query "a black backpack was moved from a chair to a desk" --camera CAM-TEST
.\.venv\Scripts\python.exe -m src.stage11 search --query "backpack on a desk" --camera CAM-TEST --start-seconds 4 --end-seconds 8
.\.venv\Scripts\python.exe -m src.stage11 compare --queries-file docs/evaluation/stage11_queries.json --camera CAM-TEST
```

Comparison writes a new `outputs/stage11/<run>/comparison.json` containing visual,
text and hybrid rankings from the same candidate pools. Search prints JSON. Models
must already be cached; this stage does not download or upgrade them. Use
`--visual-weight 0` or `--text-weight 0` to disable a branch and its model loading.
There is no migration or production database write in these commands.

## Ranking and provenance

CLIP uses 512-dimensional image/text vectors; MiniLM uses its separate
384-dimensional description space. The vectors and cosine scores are never
compared across models. Default windows are non-overlapping four-second ranges,
clipped at source duration and requested bounds. End bounds are exclusive.
Each channel votes once per source/window after duplicate collapse. The score is
`visual_weight/(60 + visual_rank) + text_weight/(60 + text_rank)` for channels
that returned that window. Evidence retains original description ranges.

Text requires an exact matching manual frame annotation or explicit temporal
index entry, correct embedding model and valid source/time references. Arbitrary
scenes and unreviewed captions are excluded. Registry metadata and current source
bytes must match the pinned visual source. Legacy approval records did not store
signed source hashes at annotation time; this binding is checked now. Eligibility
does not establish that every historical annotation is accurate.

One known incorrect chair label at four seconds is excluded by exact video,
scene ID and description hash in `docs/evaluation/stage11_text_exclusions.json`.
The underlying annotation is preserved. A changed description does not inherit
that exclusion automatically.

## Observed diagnostic results

The six-case suite ran on the existing single synthetic ten-second clip, with ten
visual frame candidates and nine eligible text records after the exclusion. The
movement query ranked 0–4 seconds first in all three methods. Hybrid ranked 0–4
seconds first for the desk query, while visual-only ranked 4–8 first. The chair
query still ranked 4–8 first in hybrid. The red-suitcase negative query returned
candidates in every method. These are diagnostics, not independent relevance
annotations, accuracy measurements or evidence that hybrid improves retrieval.

RRF scores are not probabilities. There is no calibrated absence threshold or
automatic proof of movement, identity or physical placement. Candidate pools are
capped at 100 per branch before window collapse, and displayed evidence at five
items per channel/window. Text retrieval currently scans qualified annotations;
source verification hashes local videos per query. Large collections need a
separate scaling design and benchmark.

## Verification

```powershell
$env:GROUP8_TEST_POSTGRES = '1'
$env:PYTHONPATH = (Join-Path (Get-Location) 'scene-search-prototype')
.\.venv\Scripts\python.exe -m pytest --import-mode=importlib -q
```

PostgreSQL tests use disposable owned schemas. Tests cover source drift, manual
and temporal provenance, model separation, filters, boundary clipping, duplicate
collapse, exact RRF arithmetic and disabled channels. Live comparison and
filter/preservation reports remain local under `outputs/stage11`.

Next: playable search results with timestamp seeking, followed by independent
evaluation on several 20–60 second recordings. MediaMTX ingestion follows the
retrieval/playback flow; this stage does not claim live CCTV readiness.

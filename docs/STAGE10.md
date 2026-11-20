# Stage 10a CLIP image retrieval in PostgreSQL

This increment implements actual text-to-image retrieval using the existing
cached OpenCLIP **ViT-B-32 / laion2b_s34b_b79k** encoder. New image vectors are
**512-dimensional** and are stored separately from the existing **384-dimensional
MiniLM** scene vectors. It does not generate/index VLM captions, infer physical
identity, reconstruct actions, or change Stage 9 annotation provenance.

## What changed

- `src/clip_encoder.py`: extracted the existing encoder, normalization and
  protocol so Stage 10 does not import Chroma or the historical verification module.
  `src/visual_search.py` re-exports compatible names for the preserved CLI/tests.
- `src/pg_config.py`: local environment loading, independent psycopg connections,
  explicit migration command. No credentials are hardcoded or printed.
- `migrations/`: checked adoption of the existing schema, a new visual table,
  and immutable visual-source metadata. No existing table/record is rewritten.
- `src/visual_index.py`: validated vectors, idempotent inserts, content/metadata
  guards, exact cosine search, filters and one best match per source timestamp.
- `src/stage10.py`: CLI migration, indexing, search and stats outside HTTP handlers.
  This increment has no new search web interface or playable-clip endpoint.
- Tests exercise pure validation, actual decoding/cropping and disposable-schema
  PostgreSQL integration. Temporary schemas are cleaned up without touching the
  public data.

## Database changes actually applied

Alembic now records revision **0003_visual_source_snapshot**:

1. `0001_existing_baseline` checks required existing tables/columns and verifies
   `scenes.embedding` is declared `vector(384)`. It adopts that pre-existing schema
   without creating, altering or dropping legacy tables. It is not a greenfield
   installer and refuses incompatible/missing prototype tables.
2. `0002_visual_embeddings` creates `visual_embeddings`, with recording FK,
   source/video/frame/time, frame-or-crop variant, image hash/path, encoder identity,
   preprocessing description, optional bbox/tracking provenance and `vector(512)`.
   A unique sample/encoder constraint makes repeated imports idempotent.
3. `0003_visual_source_snapshot` adds pinned camera/path/FPS/duration to visual
   rows. These prevent legacy registry updates from relabelling existing evidence.
   Earlier unsnapshotted visual rows are hidden from search until explicit,
   source-hash-verified reindexing fills them. Camera/path changes and source-byte
   replacement under an indexed video ID are rejected for future writes.

The source registry may round FPS to four decimals. Index timestamps use the
actual decoded CFR metadata; validation tolerates only that small registry rounding
and a bounded duration discrepancy. Variable-frame-rate footage is unsupported
by this first adapter and needs presentation-timestamp ingestion later.

Search returns the pinned source path **and its indexed content hash**. It does
not claim that a mutable file at that path still has identical bytes indefinitely.
The later playback service must verify the source or retain immutable recordings.
Stored PNG evidence is content-addressed and verified when reused by indexing.

The backup is local at
`outputs/backups/group8_before_stage10_20261116.dump`. It was restored into a
new disposable PostgreSQL database; row counts and content digests matched in
all four original tables. Only that temporary restore database was removed.
`outputs/backups/stage10_restore_verification.json` records the archive hash and
original table digests. After migration/indexing the four table digests still
matched: **15 scenes, 9 frame candidates, 1 recording, 1 temporal window**.

Automatic destructive downgrades of the visual index are deliberately disabled.
The backed-up legacy tables and runtime media remain local; no generated data or
database backup should be committed to GitHub.

## Windows PowerShell commands

Run from `C:\Users\hp\Documents\group8-visual-asset-retrieval`. Existing dotenv
configuration targets the working PostgreSQL instance on loopback port 15432.
It must contain the correct existing credentials; never paste them into commands.

```powershell
# Explicit schema migration (already applied here; rerunning is safe)
.\.venv\Scripts\python.exe -m src.stage10 migrate

# Use the already cached weights without downloading anything
$env:HF_HUB_OFFLINE = '1'
$env:TRANSFORMERS_OFFLINE = '1'

# Full frames only: useful for wider scene context and objects YOLO missed
.\.venv\Scripts\python.exe -m src.stage10 index `
  --video-id test01_863e5601e3b5 --confirm-cfr

# Optionally add raw-pixel crops using matching Stage 9 detector observations
.\.venv\Scripts\python.exe -m src.stage10 index `
  --video-id test01_863e5601e3b5 `
  --tracking-report outputs/stage9/20261116T164229528159Z/analysis.json `
  --confirm-cfr

.\.venv\Scripts\python.exe -m src.stage10 stats
.\.venv\Scripts\python.exe -m src.stage10 search `
  --query "a black backpack on a wooden desk" --camera CAM-TEST --limit 3

# Compare full-frame results; time bounds are relative source seconds, end exclusive
.\.venv\Scripts\python.exe -m src.stage10 search `
  --query "a blue water bottle on a desk" --variant frame `
  --video-id test01_863e5601e3b5 --start-seconds 6 --end-seconds 10
```

`--every-seconds` defaults to 1, rounded to a whole source-frame stride.
`--max-frames` defaults to 300 sampled frames, not 300 seconds for every setting.
`--batch-size` defaults to 8. Reports state actual interval, sample limit, last
sample timestamp and whether all planned samples were reached. Complete sampling
means the scheduled representative samples were processed, not every source frame
was understood. Each GPU/storage batch commits independently; a later failure
leaves a clearly labelled partial index and a failure report. Retry is idempotent.
This is not yet the persistent job scheduler or live-stream worker.

PNG evidence is under `outputs/stage10/images/`; each invocation writes a fresh
report in `outputs/stage10/runs/`. Crops are decoded from unannotated source pixels.
Their provenance includes the detector score, local track ID and hashes of the
Stage 9 report/observations, explicitly marked **unreviewed_detector_output**.
AI-assisted/human boxes are not silently used as detector predictions.

## Executed real-model checks and limits

- Existing synthetic video: **640 x 360, 24 FPS, 240 frames, 10 seconds**.
- Indexed **10 full frames** at seconds 0–9, plus **7 optional backpack crops**.
  Crops are missing at some sample times because Stage 9 has no backpack row there;
  full-frame indexing still covers those times.
- **17 stored 512-dimensional vectors**, with norms approximately 1.0. Repeating
  the same import inserted **zero duplicates** and retained 17 rows.
- Initial measured run: **0.239 s** decoding, **16.575 s** encoder loading,
  **2.869 s** embedding/storage, **19.690 s** total. The repeat run was **13.170 s**
  including **12.290 s** model loading and **0.660 s** embedding/storage. Loading
  happens once per process; a persistent worker can reuse the instance. These
  small-clip measurements do not establish long-video throughput or live latency.
- Camera/time filtering, incompatible-encoder exclusion and per-frame deduplication
  were exercised against PostgreSQL. Cosine similarity is relevance, not confidence.
- Current mixed frame/crop search uses the strongest compatible variant per frame;
  it is not calibrated multimodal fusion or the Stage 11 hybrid ranker.

Actual mixed-variant results (top 3 seconds; rounded cosine scores):

| Query | Results |
| --- | --- |
| a black backpack on a wooden desk | 5 s (0.3439), 0 s (0.3370), 2 s (0.3351) |
| a black backpack resting on a chair | 7 s (0.3253), 9 s (0.3222), 2 s (0.3209) |
| a person carrying a backpack | 6 s (0.3187), 0 s (0.2780), 9 s (0.2706) |
| a blue water bottle on a desk | 6 s (0.2298), 7 s (0.2150), 9 s (0.1988) |
| a red suitcase in a corridor (negative description) | 7 s (0.1708), 2 s (0.1688), 5 s (0.1612) |

**The chair and carrying rankings contain wrong contextual/action matches.** The
desk query also ranks chair observations highly. Full-frame-only comparisons
also confuse placement (the desk description ranked 3 s first). This integration
works technically but does not establish reliable relationship/action retrieval.
CLIP's appearance similarity and small crops can omit the context needed for
those distinctions. Temporal event understanding belongs to later evaluated work.

The negative description still returns candidates; no calibrated rejection/absence
threshold is implemented. Do not present the smaller negative-query score as a
validated decision boundary. `outputs/stage10/live_verification.json` contains
the complete query results, per-query timings, encoder identity, memory figure
and full-frame comparisons. This is a single synthetic clip and a functionality
check, not Precision@K, recall, action accuracy or a real-surveillance benchmark.

## Tests

Final verification with PostgreSQL tests enabled: **98 passed, 44 subtests passed**
in **65.64 seconds**. `pip check` found no broken requirements. No packages were
installed/upgraded; the additional runtime pins are in `requirements-stage10.txt`
for reproducibility on top of the existing environment. Model weights were cached,
and live indexing/search were executed with Hugging Face offline mode enabled.

The five-query verification process reported **590.4 MiB** peak PyTorch CUDA
allocation; query timings after model loading were approximately **0.053–0.124 s**.
These are observed small-sample/process measurements, not total VRAM consumption,
production latency guarantees or a controlled benchmark.

```powershell
$env:GROUP8_TEST_POSTGRES = '1'
.\.venv\Scripts\python.exe -m unittest tests.test_visual_index tests.test_stage10 -v

$env:PYTHONPATH = (Join-Path (Get-Location) 'scene-search-prototype')
.\.venv\Scripts\python.exe -m pytest -q
```

PostgreSQL tests create uniquely named disposable `stage10_test_...` schemas,
seed tiny fixtures and drop only their owned schemas. Without the opt-in variable,
those tests are skipped; other tests still run. Coverage includes migration/data
preservation, bad vectors, rollback, filtering, idempotency, encoder separation,
metadata overwrite protection, rounded FPS, raw decoding, sampling caps and crops.
The historical CLIP/Chroma test suite also verifies compatible encoder extraction.

## Next

Use several different recordings, real footage and independent relevant-interval
annotations to evaluate appearance versus relationship queries. Then implement
reviewed scene-text/visual rank fusion and playable timestamped results. Keep the
current encoder/index as the CLIP-only baseline; do not tune thresholds to make
this single synthetic example look accurate.

# Stage 13 repository audit

> Point-in-time audit from before the current workspace implementation. The
> table describes what existed at that time, not current Stage 12 capabilities.
> The current investigator app, authentication, scoped access, upload queue,
> and MediaMTX segment hook are documented in the
> [frontend and local operations guide](JARVISTRACE_FRONTEND_AND_OPERATIONS.md).

Inspected implementations and tests before processing G507. The Stage 12 app
and historical experiments are separate entry points.

| Capability | Verified implementation and limit |
|---|---|
| Upload | No upload endpoint in Stage 12. Registration and indexing are offline commands. |
| Multiple files | Registry and vector index are keyed by recording ID. Stage 12 lists/filter multiple cameras and recordings. Stage 10 indexes one file per call; Stage 13 adds a bounded offline manifest driver, not a job scheduler. |
| AVI | Stage 10 uses OpenCV without extension restrictions. Existing tests use MJPG AVI; G507's actual H.264 AVI decoding was tested separately. Browser AVI support is a separate question, so evaluation uses mapped H.264 MP4 derivatives. |
| Frame extraction | `stage10.index_recording` decodes sequentially and saves raw content-addressed PNG samples. Legacy `ingest_video.ingest` also extracts change/guard-filtered JPEG candidates; these are not validated event boundaries. |
| Timestamps | Stage 10 uses frame index / measured FPS, requiring CFR interpretation. Filters and playback use relative seconds. No capture-start/timezone fields or date filters exist in the PostgreSQL app. `created_at` is insertion time. Stage 13 keeps filename dates and original offsets in an external manifest without inventing UTC. |
| YOLO | Legacy `compare_object_crops.py` runs Ultralytics. Stage 9 imports saved detector/tracker labels. Stage 10 can take source-validated Stage 9 crop reports, but full-frame indexing needs no detector. G507 smoke indexing uses full frames only. |
| Visual retrieval | A **text query** is encoded with OpenCLIP ViT-B-32/laion2b_s34b_b79k, then compared to matching 512D image embeddings. No image-upload query UI exists. |
| Text retrieval | The same text query uses 384D MiniLM against exact matching manual frame descriptions or explicit temporal-index records. `ReviewedTextIndex` checks source bindings and provenance. G507 initially has no such records. |
| Hybrid | `fuse_windows` assigns separate channel ranks to four-second source-isolated windows, collapses duplicates and sums RRF contributions, default k=60. It does not combine YOLO confidence with cosine scores or compare 384D and 512D vectors directly. |
| Scores/results | Cosine similarities and rank contributions are returned. Stage 12 displays ranked windows, evidence and source seeking. None is calibrated confidence or proof of absence. |
| Provenance | Visual rows pin video ID, camera, source SHA, source path/FPS/duration, sampled frame/time, image hash/path, encoder/preprocessing identity, variant/box and optional unreviewed detector provenance. |
| Reruns | The visual unique key deduplicates identical samples/encoders. Reindexing still decodes and embeds; partial earlier batches can remain if a later batch fails. Changed sources need new IDs. Legacy metadata upserts can overwrite profiles; Stage 13 uses immutable registration instead. |
| Resources | Stage 10 image inference is batch bounded, but collects sample metadata and writes all selected PNGs first. Default cap 300 samples; no background queue/resume lease. Exact database scans, source hashing, and playback's per-request verified snapshot remain scaling costs. |
| Evaluation | Stage 9 measures reviewed detector observations, not general semantic retrieval. Stage 10/11 comparison files contain diagnostics without independent relevance labels. Stage 13 adds verified-only collection retrieval metrics. |

Source evidence: `src/stage10.py`, `visual_index.py`, `stage11.py`,
`hybrid_search.py`, `reviewed_text.py`, `stage12.py`, `stage12.html`,
`scene-search-prototype/ingest_video.py`, `compare_object_crops.py`, migrations and
tests for Stages 9–12. SQLite/Chroma historical cross-camera verification code is
preserved; it is not reintroduced into the current architecture.

## Remaining live validation

The local workspace now has a MediaMTX configuration, completed-segment hook,
idempotent registration, durable indexing jobs, and catalog refresh. The binary
is not bundled, and the integration has not yet been exercised against an actual
MediaMTX process. Before a sustained simulation, install the documented binary
and verify one complete segment from publish through indexing, search, and source
playback. Then measure capture timestamps, segment boundaries, reconnect gaps,
coverage, and longer-recording throughput. A physical CCTV connection still needs
authorized credentials, network controls, and camera-specific testing.

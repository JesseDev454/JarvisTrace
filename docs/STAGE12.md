# Stage 12: search and source playback

> This page records the original read-only Stage 12 prototype and its test
> results. The current application adds accounts, camera permissions, uploads,
> durable indexing jobs, and a MediaMTX segment hook. Use the
> [frontend and local operations guide](JARVISTRACE_FRONTEND_AND_OPERATIONS.md)
> for the current setup and behavior; the historical constraints below describe
> the prototype at the time this page was written.

JarvisTrace now has a local search page for the Stage 11 index. Describe an object,
choose a camera/recording and optional relative time range, and select a ranked
window to play the source at its start time. Expand “Why this window matched” to
inspect visual observations, text provenance and channel rank contributions.

## Launch

From the repository root in PowerShell:

```powershell
$env:HF_HUB_OFFLINE = '1'
$env:TRANSFORMERS_OFFLINE = '1'
.\.venv\Scripts\python.exe -m src.stage12 --port 8770
```

Open [JarvisTrace locally](http://127.0.0.1:8770). The cached CLIP/MiniLM models
load on the first search, so that query takes longer. Visual-only and reviewed
text-only modes load just their selected model. Only one search runs at a time;
another receives a retry message. Stop the server with Ctrl+C.

The default allowed storage folder is `scene-search-prototype/videos`. To use
other indexed local storage, pass `--storage-root "C:\your\recordings"`; repeat
the option for multiple folders. Custom roots replace the default. Sources must
already be registered and indexed through Stage 10. Restart Stage 12 after adding
recordings or changing configured storage; its playback catalog is a startup
snapshot. Missing, changed, conflicting or out-of-folder sources are unavailable.

## Using playback

- Select a time-window button to seek and play. Replay returns to its beginning.
- “Pause at window end” stops once near the end; native browser time updates can
  overshoot by a fraction of a second. This is navigation, not a frame-accurate cut.
- Native Play can then continue into surrounding footage. Selecting/replaying a
  result arms the boundary pause again. Uncheck it to play continuously.
- The source codec must be supported by the browser. H.264 MP4 is recommended;
  unsupported or corrupt media shows an error. Original files are never transcoded
  or modified by this app.

Time filters use recording-relative seconds and an exclusive end, not calendar
dates. Scores are ranking contributions, not confidence percentages. An empty list
means no indexed candidates under these filters; a returned candidate does not
establish presence, identity, movement or physical placement. The one synthetic
clip still has the retrieval limitations recorded in `STAGE11.md`.

## API and source handling

- `GET /api/recordings`: safe recording metadata and opaque playback URLs.
- `POST /api/search`: `query`, optional `mode` (`hybrid`, `visual`, `text`),
  `camera_id`, `video_id`, `start_seconds`, `end_seconds`, `limit` (1–20).
- `GET` / `HEAD /media/{opaque_id}`: verified media, full or single byte range.
  Suffix/open-ended ranges work; unavailable ranges return 416. Multi-range
  requests are unsupported. `If-Range` uses the source hash ETag.

No API accepts filesystem paths. Playback requires pinned indexed content inside
configured roots. Each media request copies the source into a private temporary
snapshot while hashing it and serves only that verified snapshot, preventing
replacement between verification and response from changing served bytes. Missing
or changed sources return 409. Small snapshots stay in memory up to 8 MiB; larger
ones use temporary disk space and close after the response. This repeated full
read/copy is acceptable for these short local clips; sustained CCTV collections
need managed immutable media storage and a measured cache design.

The CLI binds only to `127.0.0.1`, disables proxy-header trust, rejects external
clients, untrusted Host values and cross-origin requests. It has no authentication
or shared-user authorization and must remain local. Do not expose it through a
public proxy. Returned search evidence omits filesystem paths; errors omit
connection details. No migration, indexing, caption generation or database write
runs through Stage 12. Models remain cached, with no package upgrades required.

## Verification

```powershell
$env:GROUP8_TEST_POSTGRES = '1'
$env:PYTHONPATH = (Join-Path (Get-Location) 'scene-search-prototype')
.\.venv\Scripts\python.exe -m pytest --import-mode=importlib -q
```

Final run: **130 tests and 55 subtests passed**. Stage 12 tests cover request
validation, filters/modes, empty/busy/error responses, safe metadata, opaque IDs,
storage containment, missing/changed sources, replacement after verification,
byte ranges, HEAD and origin/Host rejection. An independent review checked the
snapshot and window-boundary fixes.

Live browser verification on `test01` checked hybrid results, evidence expansion,
4–8 second seek/play/pause/replay and native continuation to the 10.0267-second
media end. The indexed frame timeline ends at 10 seconds; the MP4 container has a
slightly longer media duration. Live API checks covered visual/text modes,
camera filtering and empty out-of-range searches. All six public table content
digests remained unchanged; local reports are under `outputs/stage12`.

Next work is evaluating this flow with several independently annotated 20–60
second recordings, then adding MediaMTX segment capture and indexing. This page
plays indexed recordings; live RTSP capture is a later increment.

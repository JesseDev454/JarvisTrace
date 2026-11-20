# JarvisTrace frontend and local operations

This document describes the first usable app shell and the local demo workflow.
The browser UI is investigator-first; administration is a secondary section.

## What is implemented

- Sign in and first-run creation of one administrator account.
- Investigator search, camera/recording/time filters, candidate review, source
  playback, evidence previews, About, and the separately linked evaluation page.
- Administrator screens for recordings, MP4 import, processing/retry, users and
  camera assignments, and MediaMTX camera paths.
- Server-enforced investigator camera scope for search, source playback, and
  evidence images. Passwords are PBKDF2-HMAC-SHA256 hashes; session cookies are
  random, `HttpOnly`, `SameSite=Strict`, and stored in PostgreSQL only as hashes.
- Streaming upload into a private local storage folder, H.264/CFR metadata checks,
  MP4 container verification, immutable recording IDs, durable PostgreSQL jobs,
  retries, and a separate worker. The worker binds indexing to the source hash
  stored at ingest and fails the job if the managed source has changed.
- Opaque source/evidence URLs with path-root and content-hash verification.
- A MediaMTX hook that validates a completed segment under the configured
  recording root, copies it into managed storage, registers its UTC epoch start
  when the configured filename contains one, and queues indexing idempotently.

The app and MediaMTX listeners are configured for loopback-only access. The
browser does not receive RTSP credentials or filesystem paths.

## Requirements and first run

Use Node.js 20.19+ or 22.12+, the repository's Python virtual environment,
PostgreSQL with the existing project schema/pgvector extension, and FFmpeg tools
`ffmpeg`, `ffprobe`. For simulated live footage, use MediaMTX 1.21.3 or later
from the [official releases](https://github.com/bluenviron/mediamtx/releases).
It can be on `PATH`, or you can invoke it by its full path; it does not need to
be copied into the repository. The current workstation's binary is
`C:\Users\hp\Downloads\mediamtx_v1.21.3_windows_amd64\mediamtx.exe`.

1. Copy `scene-search-prototype/.env.example` to `scene-search-prototype/.env`.
   Set the local database password and generate a local MediaMTX hook token
   before starting the API. This PowerShell command stores a random token in
   `.env` without printing it:

   ```powershell
   .\.venv\Scripts\python.exe -c "from pathlib import Path; import re,secrets; p=Path('scene-search-prototype/.env'); s=p.read_text(encoding='utf-8'); s,n=re.subn(r'(?m)^JARVISTRACE_MEDIAMTX_TOKEN=.*$', 'JARVISTRACE_MEDIAMTX_TOKEN='+secrets.token_urlsafe(32), s); assert n == 1, 'Expected one token setting in .env'; p.write_text(s,encoding='utf-8')"
   ```

   Do not commit `.env`.
2. Install frontend packages and create the static production bundle:

   ```powershell
   cd frontend
   npm ci
   npm run build
   cd ..
   ```

3. Back up the intended database, confirm the existing prototype schema, and
   apply additive migrations. The current Alembic head is
   `0005_indexed_coverage_only`: `0004_jarvistrace_workspace` adds workspace
   accounts, cameras, recording metadata, and jobs; `0005_indexed_coverage_only`
   ensures queued and processing footage has no searchable coverage. The command
   checks/adopts the expected baseline before applying the revisions:

   ```powershell
   .\.venv\Scripts\python.exe -m src.stage10 migrate
   ```

   The baseline migration never replaces the historical prototype tables. If
   the existing baseline check fails, stop and compare the schema before changing
   it. Verify the head, workspace tables, and existing recording/index rows after
   migration. Run the command again to confirm it is idempotent.
4. In one PowerShell window, start the local API:

   ```powershell
   .\.venv\Scripts\python.exe -m src.stage12 --port 8770
   ```

5. In a browser, open `http://127.0.0.1:8770`. On first run create the
   administrator account. Add one MediaMTX camera path from **Manage recordings
   → Cameras** or import a recording for an upload camera.

The React production build is served at `/`. The previous read-only prototype is
still available at `/stage12`.

## Import and index a video

1. Open **Manage recordings → Import footage**.
2. Choose an H.264, constant-frame-rate `.mp4`, select a configured camera, and
   enter a capture start only when the source time and time zone are verified.
   Leaving the capture start blank preserves relative clip time without guessing
   an absolute CCTV time.
3. Start one worker in another PowerShell window:

   ```powershell
   .\.venv\Scripts\python.exe -m src.stage12_worker
   ```

   For a one-job diagnostic, use `python -m src.stage12_worker --once`.
4. Watch **Processing** until the job is ready, then search. A failed job can be
   retried after its input or environment problem is fixed.

The upload limit is 1 GiB. The current worker samples at one-second intervals,
at most 300 samples per job. Longer sources become searchable through the last
verified sample; the displayed coverage watermark is partial. Incremental resume
and longer-recording throughput are later performance work.

## Simulate a live source through MediaMTX

The repository config records completed H.264/fMP4 segments under
`mediamtx/recordings/<path>/` and invokes a local PowerShell hook when a segment
closes. MediaMTX exposes `$MTX_PATH` and `$MTX_SEGMENT_PATH` to that hook, as
documented in its [hook reference](https://mediamtx.org/docs/features/hooks).
Its [recording guide](https://mediamtx.org/docs/features/record) describes
segment duration, format, and retention settings.

1. Start the API and one indexing worker from the repository root, using the
   same local database configuration:

   ```powershell
   .\.venv\Scripts\python.exe -m src.stage12 --port 8770
   .\.venv\Scripts\python.exe -m src.stage12_worker
   ```

   The UI is served from the production bundle at `http://127.0.0.1:8770`.
   Create an administrator on first run, then create a MediaMTX camera path that
   matches the publisher path, for example `CAM_TEST`.
2. In the MediaMTX PowerShell window, load the same token from `.env` without
   printing it and point the hook at the recording directory:

   ```powershell
   $env:JARVISTRACE_MEDIAMTX_TOKEN = (& .\.venv\Scripts\python.exe -c "from dotenv import dotenv_values; print(dotenv_values('scene-search-prototype/.env')['JARVISTRACE_MEDIAMTX_TOKEN'])")
   $env:JARVISTRACE_MEDIAMTX_RECORDINGS_ROOT = (Join-Path (Get-Location) 'mediamtx\recordings')
   $mediamtx = 'C:\Users\hp\Downloads\mediamtx_v1.21.3_windows_amd64\mediamtx.exe'
   & $mediamtx --validate-conf='.\config\mediamtx\mediamtx.yml'
   & $mediamtx '.\config\mediamtx\mediamtx.yml'
   ```

   The API reads this token directly from `.env`; the hook calls
   `http://127.0.0.1:8770/api/mediamtx/segment` with the same secret. Start both
   Start MediaMTX from the repository root after validation. The API and hook
   must use the same token; keep the MediaMTX recording root on the same local
   machine. Confirm all listeners shown at startup are on loopback.
3. Publish a reviewed local H.264 sample from another window:

   ```powershell
   .\scripts\publish_mediamtx_sample.ps1 `
     -VideoPath '.\scene-search-prototype\videos\your-sample.mp4' `
     -CameraPath 'CAM_TEST'
   ```

4. Let MediaMTX close a segment (the default is 60 seconds). Confirm the
   completion hook returns HTTP 202, then watch **Processing** until the worker
   marks the job **Ready**. The same camera path, relative segment path, and
   content hash produce an idempotent event. Replaying a clip does not simulate
   wall-clock CCTV timestamps unless the segment names and capture times are
   explicitly verified.

5. For a reconnect exercise, stop the publisher, record the time, leave it
   disconnected long enough to observe a gap, then start it again. Compare the
   UTC starts of the last segment before and first segment after the interruption.
   A segment timestamp alone does not prove that every frame inside the interval
   was captured.

This config is a loopback simulation, not a physical camera connection or a live
browser-monitor screen. A real RTSP camera still needs an authorized source URL,
credential handling, network allow-listing, reconnect/backoff behavior, and a
measured capture-to-index test. The UI marks a MediaMTX path as configured; that
label does not assert that a publisher is online.

## Verification

Frontend checks:

```powershell
cd frontend
npm run typecheck
npm test
npm run build
```

On a new machine, install the Playwright browsers once after `npm ci`:

```powershell
npx playwright install chromium firefox
```

Python Stage 12 focused checks (run from repository root):

```powershell
.\.venv\Scripts\python.exe -m pytest `
  tests/test_stage12.py `
  tests/test_stage12_ingest.py `
  tests/test_stage12_workspace_api.py `
  tests/test_stage12_workspace_store.py `
  tests/test_stage12_workspace_security.py `
  tests/test_stage12_worker.py -q
```

Check that enabled MediaMTX listeners stay on loopback:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_mediamtx_config.py -q
& 'C:\Users\hp\Downloads\mediamtx_v1.21.3_windows_amd64\mediamtx.exe' `
  --validate-conf='config/mediamtx/mediamtx.yml'
```

The browser tests use mocked APIs and a tiny checked-in H.264 fixture. They verify
investigator/admin navigation, upload UI states, responsive search, playback
failure, and axe WCAG 2.1 A/AA checks for the desktop screen set and phone search.
They do not replace a PostgreSQL integration run, a MediaMTX binary test, or an
authorized physical-camera test.

## Local verification status

The MediaMTX acceptance evidence below was captured on 2026-11-19 against the local Windows setup. The full automated suites were rerun on 2026-11-20 after the crop/model comparison was documented. The simulation remained on loopback; this does not commission a physical CCTV camera.

### Environment and database

- MediaMTX v1.21.3, FFmpeg/FFprobe 9.0.2, Python 3.13.17, Node.js 24.20.0, npm 11.19.0, and PostgreSQL 17.12 with pgvector were used.
- PostgreSQL database `group8_fyp` accepted the configured application role. Alembic is at `0005_indexed_coverage_only`. The migration was applied twice; the second run made no schema changes. A backup was taken before migrating the existing project database.
- The PostgreSQL integration suite executed against the real database: **214 passed, 88 subtests passed** in 107.03 seconds. The CFR ingestion regression tests also passed, including acceptance of a timestamp-uniform MP4 with small fMP4 average-rate rounding and rejection of a clip with a terminal frame gap.
- Full Python suite rerun on 2026-11-20 with `GROUP8_TEST_POSTGRES=1` and the project `PYTHONPATH`: **219 passed, 93 subtests passed** in 115.78 seconds. This supersedes the earlier test count above; the integration run exercised the configured local PostgreSQL test database.
- The database has 21 MediaMTX jobs, all `ready`: 20 for `CAM-TEST` and one restricted-camera fixture. There are also 1,129 `CAM-TEST` and 61 restricted-fixture visual embedding rows. The pre-existing `test01` recording remains as a separate legacy row.

### MediaMTX capture and recovery

The real MediaMTX completion hook wrote decodable local H.264 segments and registered them through the production FastAPI API. The hook used the same local token configured for MediaMTX and the API. A duplicate completion event returned HTTP 202 and reused its existing recording/job. A bad token returned HTTP 401, and an out-of-root segment path returned HTTP 422.

A source-hash-validated CFR metadata mismatch initially caused a segment to fail. FFprobe reported an average rate of 30.033351862 fps for a nominal 30 fps, uniformly timestamped fMP4. Ingestion now checks FFmpeg `vfrdet` timestamp intervals as well as the average rate, preserving the dropped-frame rejection. Retrying the corrected event completed indexing without creating a second recording. The regression is covered in `tests/test_stage12_ingest.py`.

During the `CAM-TEST` publisher interruption, the segment ending at approximately `2026-11-19 20:21:26.839 UTC` was followed by a segment starting at `20:22:53.075 UTC`, a measured indexed-timestamp gap of **86.236 seconds**. The API stayed up; later segments were accepted and indexed. Segment timestamps show where indexed recordings do not cover time. They cannot establish whether the camera captured continuously inside a segment or prove that the camera itself stopped during a gap.

Measured estimated segment-end-to-ready latency across the 21 test jobs was **145.225 seconds minimum, 1,426.980 seconds median, and 2,257.989 seconds maximum**. This run includes time when the indexing worker was stopped and a queue accumulated, so these values describe this test run and are not a steady-state indexing benchmark. Exact segment and job IDs, event receipt times, and job completion times are in `outputs/implementation/live-acceptance/results.json`.

### Retrieval and investigator access

Retrieval settings stayed fixed: 4-second windows, RRF `k=60`, visual/text weights `1/1`, result limit 10, and candidate pool 100.

- `q01` (dark backpack), expected at 0–6 seconds in `1795118729-820657.mp4`: **miss**. None of its top 10 windows overlap the reviewed interval. This is an observed retrieval false negative; the demo release gate remains closed.
  A read-only rank trace found 60 one-second full-frame samples, no crop samples, and no text candidates in this clip. Frames inside 0–6 seconds ranked 21st, 27th, 29th, 43rd, 51st, and 55th in visual similarity, which explains why the fixed top-10 window output omitted them. No timestamp or seek-offset error was found. This points to retrieval ranking/index coverage, not a basis to change the reviewed interval.
- The same `q01` query in `1795119642-605566.mp4` returned windows overlapping the reviewed 0–30 second interval.
- `q02` (dark bag on a bench) in `1795119642-605566.mp4` returned an overlapping candidate at 8–12 seconds. The production playback element sought to the candidate and stopped at the candidate window end.
- The human-verified absent query `q04` returned 10 candidates in each of `1795118729-820657.mp4` and `1795119642-605566.mp4`. These are false-positive suggestions. The interface labels them “Possible matches” and says ranked candidates are not proof of presence or absence.

#### Optional similarity cutoffs

Stage 12 accepts optional `JARVISTRACE_MIN_VISUAL_SIMILARITY` and
`JARVISTRACE_MIN_TEXT_SIMILARITY` settings in the local `.env`. Each value must
be a finite cosine similarity from -1 to 1. When both are unset, searches keep
the existing ranked-candidate behavior. When a cutoff is set, a window must
clear at least one configured channel cutoff; hybrid ranking still orders
eligible windows by reciprocal rank fusion. An unconfigured channel does not
qualify a window once either cutoff is active. The API reports `matches`,
`no_strong_match`, or `no_candidates`, plus raw and accepted window counts.
The investigator screen explains the difference without claiming an item was
absent.

No cutoff is enabled by default. The live-index query check produced top visual
similarities of 0.326 for “dark backpack”, 0.255 for “dark bag on a bench”, and
0.271 for “blue handled case”. The reviewed-text channel returned a caption
about a backpack sitting on a chair at 0.477 for the bench query; the blue-case
query scored 0.468 against a backpack caption. The near scores and wrong-place
caption show that these checks do not establish a safe cutoff. Keep thresholds
unset until a source-disjoint calibration and held-out evaluation show
acceptable positive retention and fewer negative returns. Similarity is
ranking evidence, not a probability or proof of presence.

No new model was installed. A larger image-text encoder needs a measured
comparison against frozen ViT-B/32 on reviewed held-out queries before
replacement; all indexed vectors must be rebuilt with a single encoder if the
model changes.

#### Isolated detector-crop comparison (2026-11-20)

An isolated comparison copied 105 source-matched full-frame embeddings into a
private baseline schema and added 65 YOLO11n crops in a separate schema. The
public index snapshot stayed unchanged. On the clip 05 backpack miss, the
relevant interval moved from outside the top 10 to rank 8, but Hit@5 remained
zero and the leading crop evidence was detector-labelled `laptop`; this is not
a verified retrieval recovery. Both q04 negative searches still returned 10
candidates, and the top cosine rose from 0.2154 to 0.2752 on clip 05 and from
0.1705 to 0.2743 on clip 06. Keep the production index and retrieval settings
unchanged. The detailed method, per-case results, evidence hashes, and larger
encoder download blocker are in
[`docs/evaluation/CAMTEST_CROP_MODEL_COMPARISON_2026-11-20.md`](evaluation/CAMTEST_CROP_MODEL_COMPARISON_2026-11-20.md).

A temporary investigator session assigned only to `CAM-TEST` saw no restricted-camera recordings. A search forced to `CAM-RESTRICTED` and direct access to its media both returned HTTP 403. Positive searches through that scoped API session returned only authorized `CAM-TEST` candidates. The live production browser remained in the administrator session; completing the investigator sign-in and search flow in a browser is still outstanding.

### Frontend verification

`npm ci` and `npm run build` passed in the earlier acceptance run; the production build runs the TypeScript check. The full Playwright suite was rerun on 2026-11-20 with the installed Chrome executable and Firefox: **26 passed, 2 expected Firefox skips**. The skips are the Chromium-only phone screenshot/accessibility checks. The coverage-gap regression passed in both browsers after verifying the long gap and reconnect gap. Its test-fixture screenshot is `outputs/implementation/live-acceptance/coverage-gaps-ui.png`.

The production bundle served by FastAPI was opened in the local browser as an administrator. Its search page showed the two measured `CAM-TEST` intervals (about 12m 47s and 1m 26s) with the caveat that timestamps do not confirm continuous capture. The screenshot above is a browser-test preview using those same measured intervals; it is not a capture of the authenticated production database page.

### Release gate and next checks

**The local demo is not marked ready.** The first positive query missed its reviewed interval in clip 05, and the scoped investigator workflow has not yet been completed in a production browser session. The pinned Chromium build also remains unavailable because the Playwright CDN timed out, although the browser suite passed with the cached revision.

Next, keep the retrieval settings fixed and investigate the `q01`/clip 05 miss against its reviewed frames and indexed evidence. Do not relabel the interval or tune settings to hide the miss. Then sign in as the scoped investigator in the production browser and repeat both positive searches, the negative query, and restricted-camera checks. When CDN access is available, install Chromium revision 1243 and rerun `npm test` without the fallback executable. Repeat the latency run with the indexing worker already active and record that separate measurement before calling the demo ready.

The full machine-readable evidence, including database revision, job/event identifiers, query outcomes, gap calculation, latency, and test results, is saved at `outputs/implementation/live-acceptance/results.json`. The research metrics shown in the app remain limited to the small, single-camera G507 validation round 2; they are not general accuracy estimates or proof of presence/absence. Full limitations remain in `docs/evaluation/STAGE13_VALIDATION_ROUND2_RESULTS.md`.

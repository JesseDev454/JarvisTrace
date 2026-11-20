# Group 8 — Stage 2: Real Video Ingestion (Add-on)

**Add-on to the working `scene-search-prototype` from Stage 1.** Do not delete or replace `app.py`, `seed.py`, `.env`, your `.venv` or Docker data.

This phase does **not** recognize backpacks, run a VLM, or claim to understand scenes. It extracts time-stamped JPG frames from **actual footage**, chooses some representative candidates using a deliberately simple pixel-change heuristic, and stores their paths and timestamps in PostgreSQL. You can **optionally label real frames by hand** and search them through the Stage 1 `/search` API. AI-generated descriptions come in Stage 3.

## 1. Place the files

Extract the ZIP at:

`C:\Users\hp\Documents\group8-visual-asset-retrieval`

The ZIP already contains the `scene-search-prototype\` directory, so it only adds `ingest_video.py`, `label_candidates.py`, tests, and this guide to your **existing** prototype directory.

Open a new CMD and run **one command at a time**:

```bat
cd /d "%USERPROFILE%\Documents\group8-visual-asset-retrieval"
.venv\Scripts\activate
cd scene-search-prototype
mkdir videos
```

If Windows says `videos` already exists, that's fine. Add these lines to your existing `.gitignore` (do not replace the file):

```text
videos/
data/
```

**Do not commit surveillance videos, extracted images, or `.env` to GitHub.** Use footage you have permission to record, and avoid capturing people or sensitive information for this first test.

## 2. Capture a short test video

Record 20–60 seconds with your phone or webcam. Place a backpack on a chair; move it to a desk; put a bottle nearby. Use a stationary camera, adequate lighting and no people. Copy the MP4 into:

`scene-search-prototype\videos\test01.mp4`

If your phone records a codec OpenCV cannot read, use FFmpeg to convert it:

```bat
ffmpeg -i videos\test01.mp4 -c:v libx264 -pix_fmt yuv420p -r 30 -c:a aac videos\test01_fixed.mp4
```

In that case, ingest `test01_fixed.mp4` rather than the original. We assume constant-frame-rate video for prototype timestamps; variable-frame-rate video requires a more careful timestamp-based decoder later.

## 3. Extract frames and store metadata

Make sure Docker Desktop is running and the `group8-postgres` container is Up:

```bat
docker ps
```

Then:

```bat
python ingest_video.py --video videos\test01.mp4 --camera CAM-TEST --every 1 --guard 5 --threshold 0.08 --max-minutes 5
```

The script reads the first five minutes at most, examines approximately one frame per second, saves the first frame plus candidates whose downscaled grayscale image differs from the last saved candidate, and saves a guard frame at least every five seconds. This is **not** validated meaningful-scene-change detection; motion, lighting, and camera movement can trigger it, and very brief events can be missed. We will evaluate/tune it against regular fixed sampling later.

Files appear under `data\frames\<video-id>\` alongside a `manifest.json`. PostgreSQL receives `video_recordings` and `frame_candidates` metadata tables; **the video bytes remain on disk**.

If Docker is unavailable, add `--dry-run` to extract images/manifest only.

## 4. Inspect and optionally label a handful of real frames

Copy the exact `Video ID:` shown by the ingestion script, then:

```bat
python label_candidates.py --video-id PASTE_ACTUAL_VIDEO_ID_HERE --limit 10 --open-images
```

This will open each saved JPG in Windows' image viewer and ask you to **type what you personally see**. Enter an accurate description (e.g. `A black backpack sits on a chair.`). Press Enter to skip a frame, or type `q` to exit. These descriptions are **manual labels**, not AI predictions. Only a few labels are needed to test the new end-to-end data path.

Ensure you have previously run `python seed.py` so the existing `scenes` table exists.

## 5. Search real-video descriptions

Start the original server:

```bat
python -m uvicorn app:app --reload --host 127.0.0.1 --port 8000
```

Go to `http://127.0.0.1:8000/docs`, run `GET /search` with a query that describes one of your **manually labeled** frames, and verify the result includes the `video_id`, `CAM-TEST`, and plausible clip start/end seconds. The five fictional demo scenes are still in the same table and may also appear until we separate datasets.

## 6. Technical cautions

- The `video_id` identifies a file by its size and sampled content; filenames and local source paths remain on your laptop. This is only an MVP identifier, not proof that two objects are identical.
- Pixel-change selection is a starting heuristic. For stronger evaluation, compare with fixed-interval sampling and optionally PySceneDetect for actual video cuts (which are a different kind of change).
- Clip times are *windows around sampled frames*, not measured event boundaries. A relevant object might enter or leave outside that window.
- Use local-only development (`127.0.0.1`); this prototype lacks API authentication and secure file streaming.
- Do not run `label_candidates.py` and share screenshots without reviewing them for private information.

## Stage 2 acceptance criteria

- [ ] Real MP4 opens and yields FPS/duration.
- [ ] JPG frames + `manifest.json` created from the video.
- [ ] Metadata for the **real** file appears in PostgreSQL.
- [ ] At least one JPG is manually described and indexed.
- [ ] `GET /search` returns that scene with real video ID and timestamp.

## Next: Stage 3

Replace manual descriptions with a measured VLM inference pipeline on extracted frames, compare CLIP as baseline, and add video clip playback. Do **not** report Stage 2's manual labels as automated scene understanding.

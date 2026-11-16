# Group 8 — Stage 6: Overlapping temporal-window indexing (pilot)

This add-on uses the **existing** Stage 1–5 prototype. It generates temporal event captions from overlapping sets of extracted surveillance frames, then optionally indexes **only user-reviewed windows** into the existing `scenes` table, retaining timestamp, video, camera and provenance information.

## Installation: avoid nested folders

Extract the contents of this ZIP into `C:\Users\hp\Documents\group8-visual-asset-retrieval\` so these two new Python files appear **directly alongside** `app.py`, `auto_caption_frames.py`, and `temporal_caption_test.py`:

- `scene-search-prototype\temporal_window_index.py`
- `scene-search-prototype\search_temporal_windows.py`

And this README plus a helper test under `scene-search-prototype\tests\`.

If extraction creates `scene-search-prototype\scene-search-prototype\` by mistake, move only the Stage 6 files up one folder. Do not replace your current `.env`, `app.py` or previous scripts.

## Before starting

Run Docker Desktop and ensure `group8-postgres` is running. The frame candidates from Stage 2 must still be present in PostgreSQL and under `data\frames\test01_863e5601e3b5`.

Open **CMD** and run **one command at a time**:

```bat
cd /d "%USERPROFILE%\Documents\group8-visual-asset-retrieval"
.venv\Scripts\activate
cd scene-search-prototype
dir temporal_window_index.py
python -m unittest discover -s tests -p "test_stage6_windows.py" -v
```

## 1. List proposed windows (no AI inference, no database writes)

```bat
python temporal_window_index.py --video-id test01_863e5601e3b5 --window-seconds 4 --stride-seconds 3 --max-windows 3 --frames-per-window 4 --list
```

For a 10-second video with 0, 2, 3, 4, 5, 6, 7, 8, 9 second candidates, you should see something close to:

- Window 1: 0–4 seconds, sampled 0, 2, 3, 4
- Window 2: 3–7 seconds, sampled 3, 4, 6, 7
- Window 3: 6–10 seconds, sampled 6, 7, 8, 9

These overlap so an event at a boundary is less likely to be missed. The displayed intervals are **candidate retrieval windows, not verified exact start/end times of the event**.

## 2. Generate preview descriptions (no database writes)

```bat
python temporal_window_index.py --video-id test01_863e5601e3b5 --window-seconds 4 --stride-seconds 3 --max-windows 3 --frames-per-window 4 --model HuggingFaceTB/SmolVLM2-2.2B-Instruct --max-side 512 --max-new-tokens 96
```

The existing SmolVLM2-2.2B checkpoint should be cached. The model is loaded **only once** for this run and used for each window. The generated captions and benchmark information are saved to:

`data\stage6\test01_863e5601e3b5\<timestamp>_HuggingFaceTB_SmolVLM2-2.2B-Instruct_temporal_windows.json`

Note the **actual report file path printed in CMD**; do not type `<timestamp>` literally.

If CUDA runs out of memory, retry with `--max-side 384` or `--frames-per-window 3`. Do **not** reinstall PyTorch or delete existing VLM weights.

## 3. Review captions before indexing

Compare each generated caption with **all of its listed frame images**. Use the uploaded video or saved JPGs. In earlier experiments the model correctly described the backpack moving from a chair to a desk in one sequence, but it also hallucinated other objects and locations. **Never assume all three window captions are correct.**

Write down only the acceptable window numbers. If none is accurate, **do not index** them. Keep the benchmark report and improve the experiment instead.

## 4. Index only explicitly approved windows

Example: if only window 1 is accurate, use the actual report path printed by the preview command:

```bat
python temporal_window_index.py --index-report "data\stage6\test01_863e5601e3b5\PASTE_ACTUAL_REPORT_FILENAME_HERE.json" --approve 1
```

For two approved windows use `--approve 1,3`. The program **does not interpret a preview as approval**, and won't index anything without `--approve`.

The script creates `temporal_window_index` to record which VLM model, frames, timestamps and report produced each approved scene. It inserts the corresponding description embedding into the existing `scenes` table. It doesn't overwrite human-labelled frame descriptions. Re-running with the same model/prompt/window is idempotent.

## 5. Test event retrieval

**Option A: search only approved temporal windows from this video:**

```bat
python search_temporal_windows.py --video-id test01_863e5601e3b5 --q "black backpack moved from chair to desk"
```

**Option B: original FastAPI API.** Keep your existing app running, open `http://127.0.0.1:8000/docs`, and search `GET /search` for `black backpack moved from chair to desk`. Since `scenes` also contains manually entered descriptions and fictional seed examples, results can mix the different sources. Confirm the `video_id` and timestamps when evaluating.

Important: the returned `similarity` is a cosine embedding similarity, **not** a probability of true movement or identity.

## Technical limits and research notes

- This uses **existing Stage 2 frame candidates**; it does not read every video frame. Fast or short events can still be missed.
- The Stage 2 `--max-minutes 5` flag means ingestion may have only processed the first five minutes of long videos. Stage 6 does **not** secretly index the rest.
- Overlapping windows can return repeated evidence of the same event; that is expected at this pilot stage.
- One successful scene is not proof of generalization. Evaluate more footage with human-checked labels before reporting improvements.
- Keep both human/single-frame descriptions and temporal descriptions; the new provenance table distinguishes the latter.
- Never add `.env`, source recordings, cached model weights or generated frames to GitHub. Existing `.gitignore` may need `data/`, `videos/`, and `runs/` added.

### Optional: checks

```bat
python -m py_compile temporal_window_index.py search_temporal_windows.py
python -m unittest discover -s tests -p "test_stage6_windows.py" -v
```

The add-on has passed local syntax and selection-helper tests. It has **not** been tested against your actual Docker database or 8GB GPU; use the preview before any indexing.

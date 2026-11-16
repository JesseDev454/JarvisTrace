# Group 8 — Stage 4: YOLO-guided object-focused captions

This add-on **does not overwrite** the existing Stage 1–3 code. Place `compare_object_crops.py` beside `auto_caption_frames.py` in `scene-search-prototype`.

## Purpose

Compare captions from the same vision-language model and the same prompt on:

- the full video frame;
- a larger crop centred on a detected backpack, retaining surrounding context;
- and full-frame fallback if YOLO misses the backpack.

The experiment is **not** a claim that crops are more accurate; we evaluate the result against the original frames.

## Prerequisites

- Stage 2 video indexed into PostgreSQL. Docker `group8-postgres` is running.
- Stage 3 `auto_caption_frames.py` beside this script, already working with the model.
- Existing Python `.venv` contains Ultralytics, Torch, Transformers, PIL, OpenCV.
- The `yolo11n.pt` weights already downloaded, or allow Ultralytics to download them.

## Commands (Windows CMD, from `scene-search-prototype`)

First, inspect detections and crops **without loading the big VLM**:

```bat
python compare_object_crops.py --video-id test01_863e5601e3b5 --limit 4 --only-crops
```

Open the images in `data\stage4\test01_863e5601e3b5`. Each frame has an annotated version and, if a backpack was detected, an expanded crop. In particular, the 4-second frame should be marked `NO BACKPACK` at the default confidence threshold based on your previous test.

Then compare full-frame and crop captions on the *same* model:

```bat
python compare_object_crops.py --video-id test01_863e5601e3b5 --limit 4 --model HuggingFaceTB/SmolVLM2-2.2B-Instruct --max-new-tokens 64
```

The script reuses the `caption_one()` function and `PROMPT` in your local Stage 3 script, **including** the repetition-control edit you already made. You do not need to change the Stage 3 script again.

To test whether a lower YOLO threshold finds the 4-second bag (may increase false positives):

```bat
python compare_object_crops.py --video-id test01_863e5601e3b5 --limit 4 --conf 0.08 --only-crops
```

If the detector still misses it, **don't fake a box**. We keep the full-frame caption instead. Later, we can assess a larger YOLO model, tracking across frames, and temporal context.

## Expected evaluation

| Time | Visual reference (verify against original video) |
|---|---|
| 0s | Black backpack resting on an office chair |
| 2s | Person approaching; backpack remains on chair |
| 3s | Person handling backpack while it's still supported by chair |
| 4s | Person moving/placing the backpack on wooden desk |

Compare both captions on: presence of backpack, accurate supporting surface, accurate person interaction, invented objects and inference time. A crop may omit important context, so it is not automatically better.

## Notes

- All files are saved under `data/stage4/<video-id>/` (normally ignored by Git).
- No `--index` option: database contents remain untouched.
- YOLO runs on CPU for 4 frames by default to avoid competing with the 2.2B model for VRAM.
- For fairness, full and crop captioning use the same VLM, prompt, decoding parameters and `--max-new-tokens` value. Image content is the only deliberate difference.
- This is a **single-frame crop comparison**, not yet a multi-frame temporal model. Don't claim it reconstructs movement or proves object identity.

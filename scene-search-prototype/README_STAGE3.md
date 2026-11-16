# Group 8 — Stage 3: Automatic scene captions (add-on)

**Purpose:** Caption the actual JPG frames already extracted and stored by Stage 2. Record speed and peak GPU memory, optionally embed AI captions and make them searchable through the existing FastAPI endpoint. This is a single-frame VLM benchmark; it does **not** yet reason over time, prove object identity, or evaluate retrieval quality.

## Requirements
- Earlier Stage 1 and Stage 2 scripts are installed and working in `scene-search-prototype`.
- Existing Docker PostgreSQL container (`group8-postgres`) is running, and `.env` already works.
- Your existing `.venv` contains `torch`, `transformers`, `Pillow`, `sentence-transformers`, `psycopg`, `pgvector`, etc. **Do not reinstall PyTorch.**
- CUDA is available on the RTX 4060. Close games/GPU-intensive apps before loading the VLM.
- A small internet download is needed the first time each VLM is used. 500M is the initial plumbing/compatibility test, **not the final research model**.

## 1. Install this add-on
Extract the ZIP to `C:\Users\hp\Documents\group8-visual-asset-retrieval`.
It adds `auto_caption_frames.py`, a README, and standalone tests under your existing `scene-search-prototype` folder. It does **not** replace `app.py`, `.env`, video files, or existing annotations.

In **Windows CMD** (one command per line):

```bat
cd /d "%USERPROFILE%\Documents\group8-visual-asset-retrieval"
.venv\Scripts\activate
cd scene-search-prototype
docker ps
```

## 2. List the actual frame candidates (no model download)
Your Stage 2 test video ID from the earlier `/search` response is `test01_863e5601e3b5`. Confirm that's still correct:

```bat
python auto_caption_frames.py --video-id test01_863e5601e3b5 --limit 3 --list
```

It should print candidate IDs, timestamps, and `EXISTS` for the frame paths. If the ID differs after re-ingestion, use the actual `video_id` from `ingest_video.py`.

## 3. First AI captions — benchmark only

```bat
python auto_caption_frames.py --video-id test01_863e5601e3b5 --limit 3
```

On the first run, Hugging Face downloads `HuggingFaceTB/SmolVLM2-500M-Video-Instruct`. The script processes 3 frames, prints the generated caption and latency for each, and reports peak VRAM usage. All results are also saved under `data\stage3\<video-id>\` in a JSONL benchmark report.

**The database is unchanged by this first run.** Compare the AI outputs to the real JPGs and your manual descriptions. Treat inaccurate objects, colors, relations, and invented motion as errors.

## 4. Index VLM-generated descriptions after checking quality

```bat
python auto_caption_frames.py --video-id test01_863e5601e3b5 --limit 3 --index
```

`--index` computes text embeddings using the **same existing** `all-MiniLM-L6-v2` model as `/search`, inserts AI captions into `scenes`, and stores model provenance, latency and peak VRAM in a separate `ai_frame_captions` table. Manual labels in `frame_candidates.description` remain unchanged. Reruns with the **same candidate and VLM** skip duplicate indexing.

Now start the existing API if it isn't running:

```bat
python -m uvicorn app:app --reload --host 127.0.0.1 --port 8000
```

Open `http://127.0.0.1:8000/docs`, use `GET /search`, and try `black backpack on a chair` or another phrase actually supported by the AI captions. Older manually written demo records still coexist in the `scenes` table; do not treat mixed results as evaluation metrics.

## 5. Test a stronger model later
After checking the 500M model's outputs and the GPU peak, try the published 2.2B model on just **one** image (larger download and VRAM requirement):

```bat
python auto_caption_frames.py --video-id test01_863e5601e3b5 --limit 1 --model HuggingFaceTB/SmolVLM2-2.2B-Instruct
```

If out of memory, close other GPU apps, retry with `--max-side 512`, or return to the 500M model. Do **not** change the working PyTorch/CUDA setup or install Windows-incompatible `flash-attn` just for the initial benchmark. This script uses PyTorch SDPA.

## Optional inspection and testing

```bat
python -m pytest tests\test_stage3_helpers.py -q
```

## Research cautions
- AI-generated descriptions may hallucinate small objects, colors and relations. Visually review frames and captions before trusting them.
- One frame does **not** establish an object's movement history, its unique identity or whether it stayed at a location.
- Models can describe people; avoid exposing personal videos and sensitive data. Store local videos and embeddings securely.
- Indexing all frames before checking quality is not advisable. Begin with 3–5, then 10–20 images, measuring caption accuracy, latency, and GPU memory.
- The current script does **not** use CLIP, YOLO or true multi-frame temporal reasoning yet; those are separate milestones.
- It deliberately preserves the earlier manual labels for side-by-side comparisons. The existing API does not yet filter source type; do not use mixed demo/manual/AI results to calculate Recall@k.

## References for the implementation APIs
- Hugging Face SmolVLM2 model: https://huggingface.co/HuggingFaceTB/SmolVLM2-500M-Video-Instruct
- 2.2B model: https://huggingface.co/HuggingFaceTB/SmolVLM2-2.2B-Instruct
- Transformers image-text-to-text guide: https://huggingface.co/docs/transformers/tasks/image_text_to_text

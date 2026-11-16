# Group 8 — Stage 5: Temporal scene understanding pilot

This is a **read-only experiment** that adds `temporal_caption_test.py` to the existing `scene-search-prototype`. It asks the same vision-language model to examine several timestamped frames together instead of describing them independently. The script neither inserts into nor changes PostgreSQL. Keep your Stage 1–4 files and `.env` unchanged.

## Setup

Unzip into `C:\Users\hp\Documents\group8-visual-asset-retrieval\`, so the Python file appears in the existing `scene-search-prototype` folder. Start Docker Desktop and the `group8-postgres` container if needed, then in CMD:

```bat
cd /d "%USERPROFILE%\Documents\group8-visual-asset-retrieval"
.venv\Scripts\activate
cd scene-search-prototype
python temporal_caption_test.py --video-id test01_863e5601e3b5 --limit 4 --list
```

## Run the comparison

```bat
python temporal_caption_test.py --video-id test01_863e5601e3b5 --limit 4 --model HuggingFaceTB/SmolVLM2-2.2B-Instruct --max-side 512 --max-new-tokens 96
```

The model is already cached. **Multi-image input can use more GPU memory than single-frame input**, so if you encounter CUDA out-of-memory, try `--max-side 384` or `--limit 2`. The model might still hallucinate; do not treat its caption as ground truth.

Expect a caption about a black backpack initially resting on a chair, a person approaching and handling it, and the backpack eventually being seen on a desk. Do not infer who owns it or assume it was carried away. Review against actual frame images at 0, 2, 3, and 4 seconds.

Reports are saved under `data\stage5\<video-id>\`. Do not use the output as a production scene description until evaluated.

## Experimental caution

Comparing this temporal output directly with a single-frame output is **not controlled** if image dimensions and number of input images differ. This is a feasibility pilot. For quantitative evaluation, match prompts, models, and generation settings, and test a larger, annotated set of events. Also review the 9-second YOLO result, which reports two backpacks and may contain a false positive.

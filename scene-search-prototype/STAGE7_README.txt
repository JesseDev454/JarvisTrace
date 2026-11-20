GROUP 8 FYP -- STAGE 7: VISION-MODEL COMPARISON

Unzip these files DIRECTLY into your current scene-search-prototype folder.
The ZIP contains no parent folders; it avoids the earlier nested-folder issue.

Windows CMD (one command at a time; .venv activated, cwd scene-search-prototype):

1. Check files and timestamps without loading a model:
   python stage7_compare_vlms.py --window 3 --list

2. Test your already-cached SmolVLM2 with structured object-state prompt:
   python stage7_compare_vlms.py --window 3 --model HuggingFaceTB/SmolVLM2-2.2B-Instruct --max-side 448 --max-new-tokens 240

3. When ready to test a second local model:
   python stage7_compare_vlms.py --window 3 --model Qwen/Qwen3-VL-2B-Instruct --max-side 448 --max-new-tokens 240

4. If Qwen hits GPU OOM, retry with --max-side 320.

5. After Window 3 works, test all windows with --window all.

6. Compare each result with STAGE7_GROUND_TRUTH.json and fill the generated
   human_review.csv. Do not auto-index model outputs based on fluency.

DEPENDENCIES:
Uses your existing torch, transformers, accelerate, Pillow installation.
Qwen3-VL requires a compatible recent transformers version (your 5.23 should
have support). The model will download gigabytes on first use. The first
model download may be slow; test cached SmolVLM2 first.

CAVEATS:
- Designed for 8GB CUDA GPU, but Qwen3-VL may exceed VRAM for four images;
  try smaller max-side, otherwise use cloud API or GGUF quantized runtime.
- It uses explicit labelled STILL images from the sample, not full video.
- Ground truth is human labelled for ONE test clip; not a full evaluation.
- Creates files in data/stage7. Does not modify PostgreSQL.
- If the model returns invalid JSON, retain the raw response and mark it;
  do not silently repair the model's response.
- For privacy, local model does not upload your frames to cloud API.

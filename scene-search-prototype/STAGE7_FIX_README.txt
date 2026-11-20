Group 8 / Stage 7 short-prompt fix

This ZIP contains two files at the root (NO nested directory):
- stage7_compare_vlms_v2.py
- STAGE7_FIX_README.txt

Extract both DIRECTLY into C:\Users\hp\Documents\group8-visual-asset-retrieval\scene-search-prototype
The original stage7_compare_vlms.py remains intact for comparison.

Activate .venv, change into scene-search-prototype, run:

python stage7_compare_vlms_v2.py --window 3 --model HuggingFaceTB/SmolVLM2-2.2B-Instruct --max-side 448 --max-new-tokens 110

Changes:
- Reduces verbose JSON demand to two-line location/change answer.
- Crops top 6% of CCTV frames containing text/time overlays by default.
- Adds no_repeat_ngram_size=3 to discourage timestamp loops.
- Limits Qwen3-VL processor pixel budget for possible future tests.
- Never changes PostgreSQL, preserves old Stage 7 reports.

After reviewing first test, optional Qwen test:
python stage7_compare_vlms_v2.py --window 3 --model Qwen/Qwen3-VL-2B-Instruct --max-side 384 --max-new-tokens 110

The Qwen model is a large NEW download; do not run unless you intend to download it.
For a controlled study of effects of overlays, add --keep-overlay.
All outputs remain experimental and must be checked against frames and ground truth.

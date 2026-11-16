"""Stage 5 experiment: caption chronological frames together with SmolVLM2.

Read-only: no inserts/updates to PostgreSQL. Uses the existing Stage 2
frame_candidates table via Stage 3's get_candidates().

Examples:
    python temporal_caption_test.py --video-id test01_863e5601e3b5 --limit 4 --list
    python temporal_caption_test.py --video-id test01_863e5601e3b5 --limit 4 --model HuggingFaceTB/SmolVLM2-2.2B-Instruct
"""
from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parent

TEMPORAL_PROMPT = (
    "These are chronological frames from one stationary surveillance camera. "
    "Describe only changes involving clearly visible portable objects. "
    "Identify an object's initial location, relevant interaction with people, "
    "and its final location if the frames show these. "
    "Distinguish someone approaching or touching an object from actually carrying it. "
    "Do not invent people, objects, movements between sampled frames, or clock times. "
    "If the action or destination is not visually clear, say so. "
    "Respond in two short factual sentences, prioritizing object movement over room details."
)


def make_messages(rows: list[dict], pictures: list):
    """Produce the user message with a timestamp immediately before each image."""
    if not rows or len(rows) != len(pictures):
        raise ValueError("Need matching nonempty rows and pictures")
    content = [{"type": "text", "text": "Frames are in chronological order:"}]
    for row, picture in zip(rows, pictures):
        content.append({"type": "text", "text": f"At {float(row['timestamp_seconds']):.2f} seconds:"})
        content.append({"type": "image", "image": picture})
    content.append({"type": "text", "text": TEMPORAL_PROMPT})
    return [{"role": "user", "content": content}]


def load_images(rows: list[dict], max_side: int):
    from PIL import Image
    pictures = []
    for row in rows:
        path = Path(row["frame_path"])
        if not path.is_file():
            raise FileNotFoundError(f"Missing extracted frame: {path}")
        with Image.open(path) as im:
            picture = im.convert("RGB")
        picture.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)
        pictures.append(picture)
    return pictures


def main():
    from auto_caption_frames import get_candidates, load_vlm, safe_name

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video-id", required=True)
    parser.add_argument("--limit", type=int, default=4, help="First N chronological candidates")
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--list", action="store_true", help="Print frames without loading the VLM")
    parser.add_argument("--model", default="HuggingFaceTB/SmolVLM2-2.2B-Instruct")
    parser.add_argument("--max-side", type=int, default=512,
                        help="Resize each frame for memory efficiency (default: 512)")
    parser.add_argument("--max-new-tokens", type=int, default=96)
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()
    if not (2 <= args.limit <= 8) or args.offset < 0 or args.max_side < 224 or args.max_new_tokens < 24:
        parser.error("Use --limit 2..8, --offset >=0, --max-side >=224, and --max-new-tokens >=24")

    rows = get_candidates(args.video_id, args.limit, args.offset)
    if len(rows) < 2:
        parser.error("Need at least two timestamped frames. Ensure PostgreSQL is running.")
    print(f"Temporal experiment: {len(rows)} frames from {args.video_id}")
    for row in rows:
        p = Path(row["frame_path"])
        print(f"  {float(row['timestamp_seconds']):>5.2f}s | {'EXISTS' if p.is_file() else 'MISSING'} | {p}")
    if args.list:
        return

    pictures = load_images(rows, args.max_side)
    # Do not instantiate a YOLO model here. Keep GPU available for the VLM.
    processor, model, device, dtype = load_vlm(args.model, args.cpu)
    import torch
    messages = make_messages(rows, pictures)
    inputs = processor.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=True,
        return_dict=True, return_tensors="pt"
    ).to(device, dtype=dtype)
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
    started = time.perf_counter()
    try:
        with torch.inference_mode():
            generated = model.generate(
                **inputs, do_sample=False, max_new_tokens=args.max_new_tokens,
                repetition_penalty=1.15, no_repeat_ngram_size=3,
            )
    except torch.cuda.OutOfMemoryError as exc:
        raise SystemExit("GPU out of memory with multiple frames. Retry with "
                         "--max-side 384 or --limit 2. No data was changed.") from exc
    if device == "cuda":
        torch.cuda.synchronize()
    duration_ms = round((time.perf_counter() - started) * 1000, 1)
    input_len = inputs["input_ids"].shape[1]
    caption = processor.batch_decode(
        generated[:, input_len:], skip_special_tokens=True
    )[0].strip()
    vram_mib = (round(torch.cuda.max_memory_allocated() / (1024 * 1024), 1)
                if device == "cuda" else None)

    report = {
        "video_id": args.video_id,
        "model_id": args.model,
        "method": "multi_image_temporal_caption",
        "timestamps_seconds": [float(r["timestamp_seconds"]) for r in rows],
        "frame_paths": [str(r["frame_path"]) for r in rows],
        "max_side": args.max_side,
        "max_new_tokens": args.max_new_tokens,
        "prompt": TEMPORAL_PROMPT,
        "caption": caption,
        "inference_ms": duration_ms,
        "peak_gpu_memory_mib": vram_mib,
        "manual_descriptions": [r.get("manual_description") for r in rows],
        "database_updated": False,
    }
    dest = ROOT / "data" / "stage5" / safe_name(args.video_id)
    dest.mkdir(parents=True, exist_ok=True)
    filename = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ_") + safe_name(args.model) + ".json"
    output = dest / filename
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print("\nTEMPORAL CAPTION:", caption)
    print(f"Inference: {duration_ms:.0f} ms | GPU peak: {vram_mib} MiB")
    print("Report:", output)
    print("Preview only: no database changes were made.")


if __name__ == "__main__":
    main()

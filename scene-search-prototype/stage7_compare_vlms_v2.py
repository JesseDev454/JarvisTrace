"""Stage 7 v2: short, grounded comparison of local VLMs on hand-reviewed surveillance windows.

Run from the existing scene-search-prototype directory. This script never writes to PostgreSQL.
"""
import argparse
import csv
import datetime as dt
import json
import re
import time
from pathlib import Path

WINDOWS = {
    1: (0.0, 4.0, (0.0, 2.0, 3.0, 4.0)),
    2: (3.0, 7.0, (3.0, 4.0, 6.0, 7.0)),
    3: (6.0, 10.0, (6.0, 7.0, 8.0, 9.0)),
}

PROMPT = """These are four chronological surveillance images from one fixed camera.
Only describe the BLACK BACKPACK using what is visibly supported by the images.
Do not read or repeat any printed dates, timestamps, labels or camera text on the images.
A person touching a bag is not necessarily carrying it. Do not invent motion.
Reply in exactly TWO SHORT LINES:
Location: Where the backpack is in the earliest and latest images.
Change: Say what changed, or say 'No observed backpack movement' if it remained in place.
If uncertain, say uncertain. Do not output JSON."""


def time_from_name(p):
    match = re.search(r"_(\d+\.\d+)\.jpe?g$", p.name, re.IGNORECASE)
    return float(match.group(1)) if match else None


def get_window_frames(folder: Path, window_num: int):
    candidates = {}
    for p in folder.glob("*.jpg"):
        t = time_from_name(p)
        if t is not None:
            candidates[round(t, 3)] = p
    expected = WINDOWS[window_num][2]
    missing = [t for t in expected if round(t, 3) not in candidates]
    if missing:
        raise FileNotFoundError(f"Missing frames for window {window_num}: {missing} under {folder}")
    return [(t, candidates[round(t, 3)]) for t in expected]


def create_review_csv(path: Path, rows):
    fields = ["window", "model", "correct_backpack_locations_0_to_4", "correct_backpack_movement", "correct_last_visible_location", "invented_events_or_objects", "invented_timestamps", "reviewer_notes"]
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({"window": row["window"], "model": row["model"]})


def load_model(model_name):
    import torch
    from transformers import AutoModelForImageTextToText, AutoProcessor
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable. Check PyTorch/CUDA in your existing .venv.")
    print(f"Loading {model_name} into CUDA (uncached model weights may need downloading)...", flush=True)
    t0 = time.perf_counter()
    model = AutoModelForImageTextToText.from_pretrained(
        model_name,
        dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation="sdpa",
    ).eval()
    if "Qwen3-VL" in model_name:
        # Prevent Qwen's dynamic-resolution processor from expanding small frames
        # into excessive visual tokens on consumer GPUs.
        processor = AutoProcessor.from_pretrained(
            model_name, min_pixels=224 * 224, max_pixels=512 * 512
        )
    else:
        processor = AutoProcessor.from_pretrained(model_name)
    print(f"Model loaded in {time.perf_counter() - t0:.1f}s")
    return model, processor


def caption_window(model, processor, frames, max_side, max_new_tokens, keep_overlay=False):
    import torch
    from PIL import Image
    content = [{"type": "text", "text": PROMPT}]
    for t, path in frames:
        with Image.open(path) as im:
            img = im.convert("RGB")
        if not keep_overlay:
            # CCTV title and clock occupy the top edge of this test footage;
            # crop them out BEFORE scaling to reduce OCR-like distraction.
            top = max(1, round(img.height * 0.06))
            img = img.crop((0, top, img.width, img.height))
        if max(img.size) > max_side:
            img.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)
        content.append({"type": "text", "text": f"Frame at {t:.1f} seconds:"})
        content.append({"type": "image", "image": img})
    messages = [{"role": "user", "content": content}]
    inputs = processor.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True,
        return_dict=True, return_tensors="pt",
    )
    model_device = next(model.parameters()).device
    inputs = inputs.to(model_device)
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    with torch.inference_mode():
        output = model.generate(
            **inputs, do_sample=False, max_new_tokens=max_new_tokens,
            repetition_penalty=1.15,
            no_repeat_ngram_size=3,
        )
    torch.cuda.synchronize()
    elapsed_ms = round((time.perf_counter() - t0) * 1000)
    new_tokens = output[:, inputs["input_ids"].shape[1]:]
    response = processor.batch_decode(new_tokens, skip_special_tokens=True)[0].strip()
    peak_mib = round(torch.cuda.max_memory_allocated() / (1024**2), 1)
    return response, elapsed_ms, peak_mib


def main():
    parser = argparse.ArgumentParser(description="Compare vision language models on three fixed surveillance windows; never indexes the database.")
    parser.add_argument("--video-id", default="test01_863e5601e3b5")
    parser.add_argument("--window", default="3", choices=["1", "2", "3", "all"])
    parser.add_argument("--model", default="HuggingFaceTB/SmolVLM2-2.2B-Instruct")
    parser.add_argument("--max-side", type=int, default=448)
    parser.add_argument("--max-new-tokens", type=int, default=110)
    parser.add_argument("--keep-overlay", action="store_true", help="Do not remove the top CCTV timestamp bar")
    parser.add_argument("--list", action="store_true")
    args = parser.parse_args()
    if args.max_side < 224 or args.max_side > 1024:
        parser.error("--max-side must be between 224 and 1024")
    selected = list(WINDOWS) if args.window == "all" else [int(args.window)]
    frame_root = Path("data") / "frames" / args.video_id
    if not frame_root.is_dir():
        parser.error(f"Frame directory not found: {frame_root} (run from scene-search-prototype)")
    frame_groups = [(num, get_window_frames(frame_root, num)) for num in selected]
    for num, frames in frame_groups:
        start, end, _ = WINDOWS[num]
        print(f"Window {num} ({start:g}-{end:g}s): " + ", ".join(f"{t:g}s {path.name}" for t, path in frames))
    if args.list:
        return
    model, processor = load_model(args.model)
    output_dir = Path("data") / "stage7" / args.video_id
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    safe_model = args.model.replace("/", "_").replace("\\", "_")
    report = output_dir / f"{timestamp}_{safe_model}_short_prompt_comparison.jsonl"
    rows = []
    for num, frames in frame_groups:
        print(f"\nGenerating Window {num} description...", flush=True)
        try:
            response, elapsed_ms, peak_mib = caption_window(model, processor, frames, args.max_side, args.max_new_tokens, args.keep_overlay)
        except torch_out_of_memory() as ex:
            print("CUDA out-of-memory. Retry with --max-side 320 and/or only one window.")
            raise
        start, end, _ = WINDOWS[num]
        row = {"video_id": args.video_id, "window": num, "start_s": start, "end_s": end,
               "sampled_times_s": [t for t, _ in frames], "model": args.model,
               "prompt": PROMPT, "max_side": args.max_side, "max_new_tokens": args.max_new_tokens,
               "keep_overlay": args.keep_overlay,
               "description_raw": response, "inference_ms": elapsed_ms, "gpu_peak_mib": peak_mib,
               "human_verified": False}
        with report.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
        rows.append(row)
        print(f"Window {num}: {response}\nInference {elapsed_ms}ms; CUDA peak {peak_mib}MiB")
    review_file = output_dir / f"{timestamp}_{safe_model}_short_prompt_human_review.csv"
    create_review_csv(review_file, rows)
    print(f"\nSaved report: {report}")
    print(f"Saved human review sheet: {review_file}")
    print("DATABASE UNCHANGED. Review against STAGE7_GROUND_TRUTH.json before indexing anything.")


def torch_out_of_memory():
    import torch
    return torch.cuda.OutOfMemoryError


if __name__ == "__main__":
    main()

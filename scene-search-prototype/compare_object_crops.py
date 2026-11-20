"""Stage 4: compare full-frame and YOLO-guided object-crop scene captions.

Requires Stage 2 ingestion and Stage 3 auto_caption_frames.py in the same directory.
Does not modify the database; all reports/images stay under data/stage4/.

Example:
 python compare_object_crops.py --video-id test01_863e5601e3b5 --limit 4 --only-crops
 python compare_object_crops.py --video-id test01_863e5601e3b5 --limit 4 --model HuggingFaceTB/SmolVLM2-2.2B-Instruct
"""
from __future__ import annotations

import argparse
import json
import math
import re
import time
from datetime import datetime, timezone
from pathlib import Path

from auto_caption_frames import caption_one, get_candidates, load_vlm

HERE = Path(__file__).resolve().parent
BACKPACK_CLASS = 24
COCO_CLASSES = [0, 24, 56]   # person, backpack, chair


def safe_name(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_.-]+", "_", value).strip("._") or "unnamed"


def context_bbox(
    box: tuple[float, float, float, float], image_size: tuple[int, int],
    width_fraction: float = 0.55, height_fraction: float = 0.75,
    width_multiplier: float = 3.5, height_multiplier: float = 2.6,
) -> tuple[int, int, int, int]:
    """Expand a target box while preserving surrounding surfaces and nearby people.

    The crop stays entirely inside the image and keeps its intended width/height,
    including when the backpack is close to the image boundary.
    """
    image_w, image_h = image_size
    x1, y1, x2, y2 = map(float, box)
    if image_w < 1 or image_h < 1 or x2 <= x1 or y2 <= y1:
        raise ValueError("Invalid image size or detection box")

    target_w = min(image_w, max(1, math.ceil(width_fraction * image_w),
                                math.ceil(width_multiplier * (x2 - x1))))
    target_h = min(image_h, max(1, math.ceil(height_fraction * image_h),
                                math.ceil(height_multiplier * (y2 - y1))))
    cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
    left = min(max(0, round(cx - target_w / 2)), image_w - target_w)
    top = min(max(0, round(cy - target_h / 2)), image_h - target_h)
    return int(left), int(top), int(left + target_w), int(top + target_h)


def select_best_target(detections: list[dict], class_id: int = BACKPACK_CLASS):
    candidates = [d for d in detections if d["class_id"] == class_id]
    return max(candidates, key=lambda d: d["confidence"], default=None)


def detect_frame(model, frame_path: Path, confidence: float, device: str, img_size: int):
    """Return saved annotations plus the best backpack box, if detected."""
    import cv2
    result = model.predict(
        source=str(frame_path), device=device, imgsz=img_size, conf=confidence,
        classes=COCO_CLASSES, verbose=False, save=False
    )[0]
    boxes: list[dict] = []
    if result.boxes is not None:
        for detection in result.boxes:
            class_id = int(detection.cls.item())
            boxes.append({
                "class_id": class_id,
                "label": result.names.get(class_id, str(class_id)),
                "confidence": round(float(detection.conf.item()), 4),
                "xyxy": [round(float(v), 2) for v in detection.xyxy[0].tolist()],
            })
    return result.plot(), boxes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video-id", required=True)
    parser.add_argument("--limit", type=int, default=4)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--only-crops", action="store_true",
                        help="Detect backpacks and save debug images; do not load the VLM")
    parser.add_argument("--model", default="HuggingFaceTB/SmolVLM2-2.2B-Instruct")
    parser.add_argument("--detector", default="yolo11n.pt")
    parser.add_argument("--conf", type=float, default=0.15)
    parser.add_argument("--yolo-device", default="cpu", help="CPU avoids competing with VLM for VRAM")
    parser.add_argument("--imgsz", type=int, default=960)
    parser.add_argument("--max-side", type=int, default=768)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--cpu", action="store_true", help="Run the VLM on CPU (slow)")
    args = parser.parse_args()
    if args.limit <= 0 or args.offset < 0 or not 0 < args.conf < 1:
        parser.error("--limit must be >0; --offset >=0; --conf between 0 and 1")

    rows = get_candidates(args.video_id, args.limit, args.offset)
    if not rows:
        parser.error("No frames found. Check your video ID and that Docker/PostgreSQL are running.")

    from PIL import Image
    import cv2
    from ultralytics import YOLO

    out_dir = HERE / "data" / "stage4" / safe_name(args.video_id)
    out_dir.mkdir(parents=True, exist_ok=True)
    report_file = out_dir / (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ_")
                             + ("crops" if args.only_crops else safe_name(args.model)) + ".jsonl")

    print(f"Found {len(rows)} frames. Output: {out_dir}", flush=True)
    print("Running YOLO detections first, before loading the vision-language model...", flush=True)
    detector = YOLO(args.detector)
    experiments = []
    for row in rows:
        original = Path(row["frame_path"])
        if not original.is_file():
            print(f"MISSING frame: {original}")
            continue
        annotation, boxes = detect_frame(detector, original, args.conf, args.yolo_device, args.imgsz)
        target = select_best_target(boxes)
        prefix = f"candidate_{int(row['id']):04d}_{float(row['timestamp_seconds']):07.2f}s"
        annotated_path = out_dir / f"{prefix}_detected.jpg"
        if not cv2.imwrite(str(annotated_path), annotation):
            raise OSError(f"Could not write {annotated_path}")
        crop_path = None
        crop_box = None
        if target is not None:
            with Image.open(original) as img:
                frame = img.convert("RGB")
                crop_box = context_bbox(tuple(target["xyxy"]), frame.size)
                crop_path = out_dir / f"{prefix}_context_crop.jpg"
                frame.crop(crop_box).save(crop_path, quality=95)
        entry = {
            "video_id": row["video_id"],
            "frame_candidate_id": int(row["id"]),
            "timestamp_seconds": float(row["timestamp_seconds"]),
            "original_frame": str(original),
            "annotated_frame": str(annotated_path),
            "crop_image": str(crop_path) if crop_path else None,
            "crop_box": list(crop_box) if crop_box else None,
            "backpack_detection": target,
            "all_detections": boxes,
            "manual_description": row.get("manual_description"),
        }
        experiments.append(entry)
        detections = ", ".join(f"{b['label']} {b['confidence']:.2f}" for b in boxes) or "none"
        status = "CROP SAVED" if crop_path else "NO BACKPACK: full-frame fallback"
        print(f"  {entry['timestamp_seconds']:>5.1f}s | {detections} | {status}", flush=True)

    # Avoid retaining a detector while loading the large VLM on an 8GB RTX 4060.
    del detector
    if args.yolo_device != "cpu":
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if not args.only_crops:
        processor, model, device, dtype = load_vlm(args.model, args.cpu)
        print("Using the SAME PROMPT and generation settings in both conditions.", flush=True)
        for entry in experiments:
            original_path = Path(entry["original_frame"])
            full, ms_full, full_peak = caption_one(
                original_path, processor, model, device, dtype,
                args.max_side, args.max_new_tokens,
            )
            entry["full_frame_caption"] = full
            entry["full_frame_ms"] = ms_full
            entry["full_frame_peak_mib"] = full_peak
            print(f"\n{entry['timestamp_seconds']:.1f}s FULL : {full} ({ms_full:.0f} ms)")
            if entry["crop_image"]:
                crop, ms_crop, crop_peak = caption_one(
                    Path(entry["crop_image"]), processor, model, device, dtype,
                    args.max_side, args.max_new_tokens,
                )
                entry["crop_caption"] = crop
                entry["crop_ms"] = ms_crop
                entry["crop_peak_mib"] = crop_peak
                print(f"{entry['timestamp_seconds']:.1f}s CROP : {crop} ({ms_crop:.0f} ms)")
            else:
                entry["crop_caption"] = None
                entry["crop_ms"] = None
                print("No crop: YOLO missed the backpack. Full-frame result retained.")
            if entry["manual_description"]:
                print("MANUAL:", entry["manual_description"])

    with report_file.open("w", encoding="utf-8") as f:
        for entry in experiments:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    print(f"\nSaved {len(experiments)} results to {report_file}")
    print("Database unchanged. Compare the images and captions before deciding what to index.")


if __name__ == "__main__":
    main()

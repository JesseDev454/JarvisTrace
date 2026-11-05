"""Single-camera video ingestion using Ultralytics YOLO11 + BoT-SORT with stream=True."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Optional, Sequence

import cv2
import numpy as np
import torch
from ultralytics import YOLO

from src.models import (
    SightingRecord,
    VideoMetadata,
    compute_source_identity,
    compute_timestamp_seconds,
    normalize_source_video,
    parse_recording_start_datetime,
)
from src.storage import SightingStore


def resolve_device(requested_device: Optional[str | int] = None) -> str | int:
    """Select CUDA device 0 when available unless explicitly overridden."""
    if requested_device is not None:
        return requested_device
    return 0 if torch.cuda.is_available() else "cpu"


def inspect_video_file(video_path: str | Path) -> dict[str, float | int]:
    """Read video FPS, frame count, and dimensions via OpenCV."""
    path = Path(video_path)
    if not path.exists():
        raise FileNotFoundError(f"Video file not found: {path}")

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        cap.release()
        raise RuntimeError(f"OpenCV could not open video file: {path}")

    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    if fps <= 0:
        fps = 25.0  # Safe fallback if container omits FPS header
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    cap.release()

    duration_seconds = round(frame_count / fps, 4) if frame_count > 0 else 0.0
    return {
        "fps": fps,
        "frame_count": frame_count,
        "width": width,
        "height": height,
        "duration_seconds": duration_seconds,
    }


def save_evidence_images(
    frame_bgr: np.ndarray,
    sighting: SightingRecord,
    evidence_dir: Path,
) -> tuple[str, str]:
    """Save an annotated full frame and a padded object crop for a sighting."""
    frames_dir = evidence_dir / "frames"
    crops_dir = evidence_dir / "crops"
    frames_dir.mkdir(parents=True, exist_ok=True)
    crops_dir.mkdir(parents=True, exist_ok=True)

    track_tag = f"t{sighting.local_track_id}" if sighting.local_track_id is not None else "untracked"
    stem = (
        f"{sighting.camera_id}_{sighting.class_name}_{track_tag}_"
        f"f{sighting.frame_index:05d}_{ int(round(sighting.timestamp_seconds * 1000)):07d}ms"
    )
    frame_file = frames_dir / f"{stem}.jpg"
    crop_file = crops_dir / f"{stem}_crop.jpg"

    h, w = frame_bgr.shape[:2]
    x1, y1, x2, y2 = [int(round(v)) for v in sighting.bbox_xyxy]
    x1_clamped = max(0, min(w - 1, x1))
    y1_clamped = max(0, min(h - 1, y1))
    x2_clamped = max(x1_clamped + 1, min(w, x2))
    y2_clamped = max(y1_clamped + 1, min(h, y2))

    # Draw bounding box and camera-scoped evidence label on a copy of the frame
    annotated = frame_bgr.copy()
    color = (0, 200, 0) if sighting.confidence >= 0.50 else (0, 165, 255)
    cv2.rectangle(annotated, (x1_clamped, y1_clamped), (x2_clamped, y2_clamped), color, 2)
    label = (
        f"{sighting.camera_id} | {sighting.class_name} "
        f"id={sighting.local_track_id if sighting.local_track_id is not None else 'N/A'} "
        f"conf={sighting.confidence:.2f} t={sighting.timestamp_seconds:.2f}s"
    )
    text_y = max(24, y1_clamped - 8)
    cv2.putText(
        annotated,
        label,
        (x1_clamped, text_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        color,
        2,
        cv2.LINE_AA,
    )
    cv2.imwrite(str(frame_file), annotated)

    # Save padded crop around detection
    pad = 16
    cx1 = max(0, x1_clamped - pad)
    cy1 = max(0, y1_clamped - pad)
    cx2 = min(w, x2_clamped + pad)
    cy2 = min(h, y2_clamped + pad)
    crop = frame_bgr[cy1:cy2, cx1:cx2]
    if crop.size > 0:
        cv2.imwrite(str(crop_file), crop)

    return frame_file.as_posix(), crop_file.as_posix()


def extract_sightings_from_boxes(
    boxes: object,
    names: dict[int, str],
    camera_id: str,
    source_video: str,
    frame_index: int,
    fps: float,
    run_id: Optional[int] = None,
    evidence_clip_path: Optional[str] = None,
) -> list[SightingRecord]:
    """Convert Ultralytics Boxes from a single frame into SightingRecord instances."""
    if boxes is None or len(boxes) == 0:
        return []

    timestamp_seconds = compute_timestamp_seconds(frame_index, fps)
    xyxy_list = boxes.xyxy.cpu().tolist()
    conf_list = boxes.conf.cpu().tolist()
    cls_list = [int(c) for c in boxes.cls.cpu().tolist()]
    if getattr(boxes, "id", None) is not None and boxes.id is not None:
        id_list: list[Optional[int]] = [int(t) for t in boxes.id.cpu().tolist()]
    else:
        id_list = [None] * len(xyxy_list)

    records: list[SightingRecord] = []
    for bbox, conf, cls_id, track_id in zip(xyxy_list, conf_list, cls_list, id_list):
        class_name = str(names.get(cls_id, f"class_{cls_id}")).lower()
        records.append(
            SightingRecord(
                run_id=run_id,
                camera_id=camera_id,
                source_video=source_video,
                frame_index=frame_index,
                timestamp_seconds=timestamp_seconds,
                class_id=cls_id,
                class_name=class_name,
                local_track_id=track_id,
                confidence=round(float(conf), 4),
                bbox_xyxy=(
                    round(float(bbox[0]), 2),
                    round(float(bbox[1]), 2),
                    round(float(bbox[2]), 2),
                    round(float(bbox[3]), 2),
                ),
                evidence_clip_path=evidence_clip_path,
            )
        )
    return records


def ingest_video(
    video_path: str | Path,
    camera_id: str,
    store: SightingStore,
    model_path: str = "yolo11m.pt",
    tracker: str = "botsort.yaml",
    imgsz: int = 960,
    conf_threshold: float = 0.15,
    confirmed_conf_threshold: float = 0.50,
    classes: Optional[Iterable[int]] = (0, 63),
    device: Optional[str | int] = None,
    output_dir: str | Path = "outputs",
    save_annotated_video: bool = True,
    keyframe_interval: int = 12,
    recording_start_utc: Optional[str] = None,
    recording_timezone: Optional[str] = None,
    time_offset_seconds: Optional[float] = None,
    source_identity: Optional[str] = None,
) -> tuple[VideoMetadata, list[SightingRecord]]:
    """Run YOLO + BoT-SORT with stream=True on a local video and persist all sightings."""
    vpath = Path(video_path)
    video_info = inspect_video_file(vpath)
    fps = float(video_info["fps"])
    resolved_device = resolve_device(device)
    out_root = Path(output_dir)
    evidence_dir = out_root / "evidence" / camera_id / vpath.stem
    if evidence_dir.exists():
        for subdir_name in ("frames", "crops"):
            sub = evidence_dir / subdir_name
            if sub.exists():
                for old_file in sub.glob("*.jpg"):
                    old_file.unlink(missing_ok=True)
    evidence_dir.mkdir(parents=True, exist_ok=True)

    run_name = f"{camera_id}_{vpath.stem}"
    annotated_dir = (out_root / "annotated_videos").resolve()
    expected_clip_path = (out_root / "annotated_videos" / run_name / f"{vpath.stem}.avi").as_posix()

    _, norm_start_iso, norm_tz = parse_recording_start_datetime(recording_start_utc, recording_timezone)
    resolved_identity = source_identity or compute_source_identity(vpath)
    norm_video = normalize_source_video(vpath)

    ingested_at_utc = datetime.now(timezone.utc).isoformat()
    metadata = VideoMetadata(
        camera_id=camera_id,
        source_video=norm_video,
        source_identity=resolved_identity,
        fps=fps,
        frame_count=int(video_info["frame_count"]),
        width=int(video_info["width"]),
        height=int(video_info["height"]),
        duration_seconds=float(video_info["duration_seconds"]),
        model_name=str(model_path),
        tracker_name=str(tracker),
        conf_threshold=float(conf_threshold),
        recording_start_utc=norm_start_iso,
        recording_timezone=norm_tz,
        time_offset_seconds=float(time_offset_seconds) if time_offset_seconds is not None else None,
        annotated_video_path=expected_clip_path if save_annotated_video else norm_video,
        ingested_at_utc=ingested_at_utc,
    )
    run_id = store.create_ingestion_run(metadata, replace_existing_for_source=True)
    metadata = VideoMetadata(**{**metadata.to_dict(), "run_id": run_id})

    model = YOLO(model_path)
    class_filter = list(classes) if classes is not None else None
    results_stream = model.track(
        source=str(vpath),
        tracker=tracker,
        device=resolved_device,
        imgsz=imgsz,
        conf=conf_threshold,
        classes=class_filter,
        stream=True,
        save=save_annotated_video,
        project=str(annotated_dir),
        name=run_name,
        exist_ok=True,
        verbose=False,
    )

    all_sightings: list[SightingRecord] = []
    seen_tracks: set[tuple[str, Optional[int]]] = set()
    peak_per_track: dict[tuple[str, Optional[int]], tuple[SightingRecord, np.ndarray]] = {}
    last_any_per_track: dict[tuple[str, Optional[int]], tuple[SightingRecord, np.ndarray]] = {}
    last_confirmed_per_track: dict[tuple[str, Optional[int]], tuple[SightingRecord, np.ndarray]] = {}

    clip_ref = expected_clip_path if save_annotated_video else vpath.as_posix()
    actual_save_dir: Optional[Path] = None

    for frame_idx, result in enumerate(results_stream):
        if actual_save_dir is None and getattr(result, "save_dir", None):
            actual_save_dir = Path(str(result.save_dir))
        frame_bgr: Optional[np.ndarray] = getattr(result, "orig_img", None)
        names = getattr(result, "names", model.names)
        frame_sightings = extract_sightings_from_boxes(
            boxes=result.boxes,
            names=names,
            camera_id=camera_id,
            source_video=vpath.as_posix(),
            frame_index=frame_idx,
            fps=fps,
            run_id=run_id,
            evidence_clip_path=clip_ref,
        )

        for sighting in frame_sightings:
            tkey = (sighting.class_name, sighting.local_track_id)
            should_save_now = False

            # 1. Always save first sighting of a track
            if tkey not in seen_tracks:
                seen_tracks.add(tkey)
                should_save_now = True

            # 2. Save periodic keyframes or low-confidence tentative candidates
            if keyframe_interval > 0 and (frame_idx % keyframe_interval == 0):
                should_save_now = True
            if sighting.confidence < confirmed_conf_threshold:
                should_save_now = True

            if should_save_now and frame_bgr is not None:
                f_path, c_path = save_evidence_images(frame_bgr, sighting, evidence_dir)
                sighting.evidence_frame_path = f_path
                sighting.evidence_crop_path = c_path

            if frame_bgr is not None:
                # Buffer only 1 frame per track key for peak, last confirmed, and last overall
                prev_peak = peak_per_track.get(tkey)
                if prev_peak is None or sighting.confidence > prev_peak[0].confidence:
                    peak_per_track[tkey] = (sighting, frame_bgr.copy())

                last_any_per_track[tkey] = (sighting, frame_bgr.copy())
                if sighting.confidence >= confirmed_conf_threshold:
                    last_confirmed_per_track[tkey] = (sighting, frame_bgr.copy())

            all_sightings.append(sighting)

    # Flush guaranteed evidence images for peak, last-confirmed, and last-any sighting of each track
    for snapshot_dict in (peak_per_track, last_confirmed_per_track, last_any_per_track):
        for sighting, buffered_frame in snapshot_dict.values():
            if sighting.evidence_frame_path is None:
                f_path, c_path = save_evidence_images(buffered_frame, sighting, evidence_dir)
                sighting.evidence_frame_path = f_path
                sighting.evidence_crop_path = c_path

    # Verify actual saved video extension if Ultralytics wrote .avi or .mp4
    if save_annotated_video:
        actual_dir = annotated_dir / run_name
        for candidate in actual_dir.glob(f"{vpath.stem}.*"):
            resolved_clip = candidate.as_posix()
            for s in all_sightings:
                s.evidence_clip_path = resolved_clip
            break

    store.insert_sightings(all_sightings)
    return metadata, all_sightings


def ingest_multiple_videos(
    sources: Sequence[dict[str, Any]],
    store: SightingStore,
    model_path: str = "yolo11m.pt",
    tracker: str = "botsort.yaml",
    imgsz: int = 960,
    conf_threshold: float = 0.15,
    confirmed_conf_threshold: float = 0.50,
    classes: Optional[Iterable[int]] = (0, 63),
    device: Optional[str | int] = None,
    output_dir: str | Path = "outputs",
    save_annotated_video: bool = True,
    keyframe_interval: int = 12,
) -> list[tuple[VideoMetadata, list[SightingRecord]]]:
    """Ingest multiple camera recordings sequentially with isolated run IDs and track namespaces.

    Each item in `sources` is a dict containing:
        - `video_path` (or `video`): str | Path
        - `camera_id`: str
        - `recording_start_utc` (optional): ISO-8601 start datetime
        - `recording_timezone` (optional): timezone string
        - `time_offset_seconds` (optional): relative offset in seconds
        - `source_identity` (optional): explicit source identity string
    """
    results: list[tuple[VideoMetadata, list[SightingRecord]]] = []
    for entry in sources:
        v_path = entry.get("video_path") or entry.get("video")
        cam_id = entry.get("camera_id")
        if not v_path or not cam_id:
            raise ValueError(f"Each ingestion source requires 'video_path' and 'camera_id', got: {entry}")
        meta, recs = ingest_video(
            video_path=v_path,
            camera_id=str(cam_id),
            store=store,
            model_path=str(entry.get("model_path", model_path)),
            tracker=str(entry.get("tracker", tracker)),
            imgsz=int(entry.get("imgsz", imgsz)),
            conf_threshold=float(entry.get("conf_threshold", conf_threshold)),
            confirmed_conf_threshold=float(entry.get("confirmed_conf_threshold", confirmed_conf_threshold)),
            classes=entry.get("classes", classes),
            device=entry.get("device", device),
            output_dir=output_dir,
            save_annotated_video=bool(entry.get("save_annotated_video", save_annotated_video)),
            keyframe_interval=int(entry.get("keyframe_interval", keyframe_interval)),
            recording_start_utc=entry.get("recording_start_utc"),
            recording_timezone=entry.get("recording_timezone"),
            time_offset_seconds=entry.get("time_offset_seconds"),
            source_identity=entry.get("source_identity"),
        )
        results.append((meta, recs))
    return results



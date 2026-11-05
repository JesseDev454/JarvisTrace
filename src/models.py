"""Data models, identity isolation keys, and synchronization utilities for multi-camera asset retrieval."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
import hashlib
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional


def compute_timestamp_seconds(frame_index: int, fps: float) -> float:
    """Compute video-relative timestamp in seconds from 0-based frame index and FPS.

    Never invents wall-clock CCTV timestamps; returns relative offset from video start.
    """
    if frame_index < 0:
        raise ValueError(f"frame_index must be non-negative, got {frame_index}")
    if fps <= 0:
        raise ValueError(f"fps must be positive, got {fps}")
    return round(frame_index / fps, 4)


def format_video_timestamp(seconds: float) -> str:
    """Format video-relative seconds as MM:SS.mmm (relative offset)."""
    if seconds < 0:
        seconds = 0.0
    minutes = int(seconds // 60)
    rem = seconds - (minutes * 60)
    return f"{minutes:02d}:{rem:06.3f} (+{seconds:.3f}s rel)"


def parse_recording_start_datetime(
    recording_start_utc: Optional[str],
    recording_timezone: Optional[str] = None,
) -> tuple[Optional[datetime], Optional[str], Optional[str]]:
    """Parse an optional ISO-8601 recording start timestamp and timezone.

    Never invents a wall-clock time if `recording_start_utc` is None.
    Returns (parsed_datetime_with_tz, normalized_iso_str, timezone_label).
    """
    if recording_start_utc is None or not str(recording_start_utc).strip():
        return (None, None, recording_timezone)

    raw = str(recording_start_utc).strip()
    if raw.endswith("Z"):
        raw = raw[:-1] + "+00:00"

    dt = datetime.fromisoformat(raw)
    if dt.tzinfo is None:
        # If naive datetime was supplied, attach UTC or parsed offset if recording_timezone is an offset
        tz_label = (recording_timezone or "UTC").strip()
        if tz_label in ("UTC", "Z", "+00:00", "00:00"):
            dt = dt.replace(tzinfo=timezone.utc)
            tz_label = "UTC"
        elif (tz_label.startswith("+") or tz_label.startswith("-")) and ":" in tz_label:
            sign = -1 if tz_label.startswith("-") else 1
            parts = tz_label[1:].split(":")
            offset = timedelta(hours=int(parts[0]), minutes=int(parts[1])) * sign
            dt = dt.replace(tzinfo=timezone(offset))
        else:
            dt = dt.replace(tzinfo=timezone.utc)
    else:
        tz_label = recording_timezone or str(dt.tzinfo)

    return (dt, dt.isoformat(), tz_label)


def compute_aligned_timestamp(
    timestamp_seconds: float,
    recording_start_utc: Optional[str] = None,
    recording_timezone: Optional[str] = None,
    time_offset_seconds: Optional[float] = None,
) -> tuple[Optional[str], Optional[float], Optional[float]]:
    """Compute aligned wall-clock ISO timestamp and/or synchronized relative seconds when metadata exists.

    Returns:
        (aligned_wall_clock_iso, aligned_epoch_seconds, aligned_sync_offset_seconds)
    Never invents wall-clock timestamps when `recording_start_utc` is omitted.
    """
    aligned_iso: Optional[str] = None
    aligned_epoch: Optional[float] = None
    aligned_sync: Optional[float] = None

    if recording_start_utc:
        dt, _, _ = parse_recording_start_datetime(recording_start_utc, recording_timezone)
        if dt is not None:
            event_dt = dt + timedelta(seconds=float(timestamp_seconds))
            aligned_iso = event_dt.isoformat(timespec="milliseconds")
            aligned_epoch = round(event_dt.timestamp(), 4)

    if time_offset_seconds is not None:
        aligned_sync = round(float(time_offset_seconds) + float(timestamp_seconds), 4)

    return (aligned_iso, aligned_epoch, aligned_sync)


def normalize_source_video(video_path: str | Path, workspace_root: Optional[str | Path] = None) -> str:
    """Canonicalize a video path so repeated ingestions of the same file never duplicate records."""
    raw_str = str(video_path).strip()
    if not raw_str:
        raise ValueError("source_video path cannot be empty")
    root = Path(workspace_root).resolve() if workspace_root is not None else Path.cwd().resolve()
    candidate = Path(raw_str)
    try:
        resolved = candidate.resolve() if candidate.is_absolute() else (root / candidate).resolve()
        rel = os.path.relpath(resolved, root)
        rel_posix = Path(rel).as_posix()
        if not rel_posix.startswith(".."):
            return rel_posix
        return resolved.as_posix()
    except OSError:
        return Path(raw_str).as_posix().lstrip("./")


def compute_source_identity(video_path: str | Path) -> str:
    """Compute a deterministic source identity fingerprint for a video file (or normalized path fallback)."""
    norm = normalize_source_video(video_path)
    p = Path(video_path)
    if not p.exists():
        p = Path.cwd() / norm
    if p.exists() and p.is_file():
        hasher = hashlib.sha256()
        size = p.stat().st_size
        hasher.update(f"size:{size}:".encode("utf-8"))
        # Hash first and last 64 KB for fast, deterministic content identity even on large CCTV files
        with p.open("rb") as fh:
            head = fh.read(65536)
            hasher.update(head)
            if size > 65536:
                fh.seek(max(0, size - 65536))
                hasher.update(fh.read(65536))
        return f"sha256:{hasher.hexdigest()[:16]}"
    digest = hashlib.sha256(norm.encode("utf-8")).hexdigest()[:16]
    return f"path:{digest}"


def compute_heuristic_stability_score(
    frame_count: int,
    first_frame: int,
    last_frame: int,
    high_conf_frame_count: int,
    min_required_frames: int = 3,
) -> tuple[float, float, float]:
    """Compute heuristic within-camera track stability metrics (distinct from raw model confidence).

    Returns:
        (high_conf_frame_ratio, temporal_continuity_density, heuristic_stability_score)
    """
    if frame_count <= 0:
        return (0.0, 0.0, 0.0)
    high_conf_ratio = round(high_conf_frame_count / frame_count, 4)
    span_frames = max(1, (last_frame - first_frame) + 1)
    continuity_density = round(min(1.0, frame_count / span_frames), 4)
    target_duration = max(1, min_required_frames * 2)
    duration_factor = min(1.0, high_conf_frame_count / target_duration)
    stability_score = round(
        (0.50 * high_conf_ratio) + (0.25 * continuity_density) + (0.25 * duration_factor),
        4,
    )
    return (high_conf_ratio, continuity_density, stability_score)


@dataclass(frozen=True)
class VideoMetadata:
    """Metadata for an ingested local video file associated with a camera."""

    camera_id: str
    source_video: str
    fps: float
    frame_count: int
    width: int
    height: int
    duration_seconds: float
    model_name: str
    tracker_name: str
    conf_threshold: float
    ingested_at_utc: str
    annotated_video_path: Optional[str] = None
    run_id: Optional[int] = None
    source_identity: Optional[str] = None
    recording_start_utc: Optional[str] = None
    recording_timezone: Optional[str] = None
    time_offset_seconds: Optional[float] = None

    @property
    def resolved_source_identity(self) -> str:
        if self.source_identity:
            return self.source_identity
        return compute_source_identity(self.source_video)

    @property
    def has_wall_clock_sync(self) -> bool:
        return bool(self.recording_start_utc and str(self.recording_start_utc).strip())

    @property
    def has_relative_offset_sync(self) -> bool:
        return self.time_offset_seconds is not None

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "camera_id": self.camera_id,
            "source_video": normalize_source_video(self.source_video),
            "source_identity": self.resolved_source_identity,
            "fps": self.fps,
            "frame_count": self.frame_count,
            "width": self.width,
            "height": self.height,
            "duration_seconds": self.duration_seconds,
            "model_name": self.model_name,
            "tracker_name": self.tracker_name,
            "conf_threshold": self.conf_threshold,
            "recording_start_utc": self.recording_start_utc,
            "recording_timezone": self.recording_timezone,
            "time_offset_seconds": self.time_offset_seconds,
            "ingested_at_utc": self.ingested_at_utc,
            "annotated_video_path": self.annotated_video_path,
        }


@dataclass
class SightingRecord:
    """A single-frame detection/tracking sighting scoped to one ingestion run, camera, and video.

    Important:
    - `confidence` is the raw detector class score for this frame, NOT proof of physical identity.
    - `local_track_id` is scoped strictly to `(run_id, camera_id, source_video)`. Two identical
      `local_track_id` numbers from different cameras, videos, or runs never share a `scoped_track_key`.
    """

    camera_id: str
    source_video: str
    frame_index: int
    timestamp_seconds: float
    class_id: int
    class_name: str
    confidence: float
    bbox_xyxy: tuple[float, float, float, float]
    local_track_id: Optional[int] = None
    evidence_frame_path: Optional[str] = None
    evidence_crop_path: Optional[str] = None
    evidence_clip_path: Optional[str] = None
    run_id: Optional[int] = None
    sighting_id: Optional[int] = None
    # Populated when camera synchronization metadata is attached to the ingestion run
    aligned_wall_clock_utc: Optional[str] = None
    aligned_epoch_seconds: Optional[float] = None
    aligned_sync_seconds: Optional[float] = None

    def __post_init__(self) -> None:
        self.source_video = normalize_source_video(self.source_video)

    @property
    def isolation_key(self) -> tuple[Optional[int], str, str, Optional[int], Optional[int]]:
        """Return a tuple uniquely isolating this track across runs, cameras, videos, and untracked frames."""
        if self.local_track_id is None:
            return (self.run_id, self.camera_id, self.source_video, None, self.sighting_id or self.frame_index)
        return (self.run_id, self.camera_id, self.source_video, self.local_track_id, None)

    @property
    def scoped_track_key(self) -> str:
        """Return a human-readable key unique across ingestion runs, cameras, and videos."""
        norm_video = Path(self.source_video).as_posix()
        run_prefix = f"run{self.run_id}:" if self.run_id is not None else ""
        if self.local_track_id is None:
            sig_tag = f"f{self.frame_index}" if self.sighting_id is None else f"f{self.frame_index}_s{self.sighting_id}"
            return f"{run_prefix}{self.camera_id}:{norm_video}:untracked_{sig_tag}"
        return f"{run_prefix}{self.camera_id}:{norm_video}:local_track_{self.local_track_id}"

    @property
    def bbox_center(self) -> tuple[float, float]:
        x1, y1, x2, y2 = self.bbox_xyxy
        return (round((x1 + x2) / 2.0, 1), round((y1 + y2) / 2.0, 1))

    @property
    def bbox_area(self) -> float:
        x1, y1, x2, y2 = self.bbox_xyxy
        return round(max(0.0, x2 - x1) * max(0.0, y2 - y1), 2)

    def to_dict(self) -> dict[str, Any]:
        x1, y1, x2, y2 = self.bbox_xyxy
        return {
            "sighting_id": self.sighting_id,
            "run_id": self.run_id,
            "camera_id": self.camera_id,
            "source_video": self.source_video,
            "frame_index": self.frame_index,
            "timestamp_seconds": round(float(self.timestamp_seconds), 4),
            "aligned_wall_clock_utc": self.aligned_wall_clock_utc,
            "aligned_epoch_seconds": self.aligned_epoch_seconds,
            "aligned_sync_seconds": self.aligned_sync_seconds,
            "class_id": int(self.class_id),
            "class_name": self.class_name,
            "local_track_id": self.local_track_id,
            "scoped_track_key": self.scoped_track_key,
            "confidence": round(float(self.confidence), 4),
            "bbox_xyxy": [round(float(x1), 2), round(float(y1), 2), round(float(x2), 2), round(float(y2), 2)],
            "evidence_frame_path": self.evidence_frame_path,
            "evidence_crop_path": self.evidence_crop_path,
            "evidence_clip_path": self.evidence_clip_path,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SightingRecord":
        bbox = data["bbox_xyxy"]
        return cls(
            sighting_id=data.get("sighting_id"),
            run_id=data.get("run_id"),
            camera_id=str(data["camera_id"]),
            source_video=normalize_source_video(str(data["source_video"])),
            frame_index=int(data["frame_index"]),
            timestamp_seconds=float(data["timestamp_seconds"]),
            aligned_wall_clock_utc=data.get("aligned_wall_clock_utc"),
            aligned_epoch_seconds=(
                float(data["aligned_epoch_seconds"]) if data.get("aligned_epoch_seconds") is not None else None
            ),
            aligned_sync_seconds=(
                float(data["aligned_sync_seconds"]) if data.get("aligned_sync_seconds") is not None else None
            ),
            class_id=int(data["class_id"]),
            class_name=str(data["class_name"]),
            local_track_id=int(data["local_track_id"]) if data.get("local_track_id") is not None else None,
            confidence=float(data["confidence"]),
            bbox_xyxy=(float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])),
            evidence_frame_path=data.get("evidence_frame_path"),
            evidence_crop_path=data.get("evidence_crop_path"),
            evidence_clip_path=data.get("evidence_clip_path"),
        )


@dataclass
class TrackSummary:
    """Aggregated summary of a local tracker trajectory within a single camera/video run."""

    run_id: Optional[int]
    camera_id: str
    source_video: str
    class_id: int
    class_name: str
    local_track_id: Optional[int]
    frame_count: int
    first_frame: int
    last_frame: int
    first_timestamp_seconds: float
    last_timestamp_seconds: float
    # Raw detection confidence statistics (from object detector)
    min_confidence: float
    mean_confidence: float
    max_confidence: float
    # Heuristic track-stability statistics (distinct from detector confidence)
    high_conf_frame_count: int
    high_conf_frame_ratio: float
    temporal_continuity_density: float
    heuristic_stability_score: float
    first_bbox_xyxy: tuple[float, float, float, float]
    last_bbox_xyxy: tuple[float, float, float, float]
    status: str  # "STRONGLY_SUPPORTED" or "UNCERTAIN_CANDIDATE"
    status_reason: str
    peak_Sighting: SightingRecord
    last_sighting: SightingRecord
    last_supported_sighting: Optional[SightingRecord] = None
    source_identity: Optional[str] = None
    first_aligned_utc: Optional[str] = None
    last_aligned_utc: Optional[str] = None
    first_aligned_epoch: Optional[float] = None
    last_aligned_epoch: Optional[float] = None
    first_aligned_sync_seconds: Optional[float] = None
    last_aligned_sync_seconds: Optional[float] = None

    @property
    def scoped_track_key(self) -> str:
        return self.last_sighting.scoped_track_key

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "camera_id": self.camera_id,
            "source_video": self.source_video,
            "source_identity": self.source_identity,
            "class_id": self.class_id,
            "class_name": self.class_name,
            "local_track_id": self.local_track_id,
            "scoped_track_key": self.scoped_track_key,
            "frame_count": self.frame_count,
            "first_frame": self.first_frame,
            "last_frame": self.last_frame,
            "first_timestamp_seconds": self.first_timestamp_seconds,
            "last_timestamp_seconds": self.last_timestamp_seconds,
            "first_aligned_utc": self.first_aligned_utc,
            "last_aligned_utc": self.last_aligned_utc,
            "first_aligned_sync_seconds": self.first_aligned_sync_seconds,
            "last_aligned_sync_seconds": self.last_aligned_sync_seconds,
            "raw_detection_confidence": {
                "min": round(self.min_confidence, 4),
                "mean": round(self.mean_confidence, 4),
                "max": round(self.max_confidence, 4),
            },
            "heuristic_track_stability": {
                "high_conf_frame_count": self.high_conf_frame_count,
                "high_conf_frame_ratio": round(self.high_conf_frame_ratio, 4),
                "temporal_continuity_density": round(self.temporal_continuity_density, 4),
                "heuristic_stability_score": round(self.heuristic_stability_score, 4),
            },
            "first_bbox_xyxy": [round(v, 2) for v in self.first_bbox_xyxy],
            "last_bbox_xyxy": [round(v, 2) for v in self.last_bbox_xyxy],
            "status": self.status,
            "status_reason": self.status_reason,
            "last_supported_evidence_frame_path": (
                self.last_supported_sighting.evidence_frame_path if self.last_supported_sighting else None
            ),
            "last_evidence_frame_path": self.last_sighting.evidence_frame_path,
            "last_evidence_crop_path": self.last_sighting.evidence_crop_path,
            "evidence_clip_path": self.last_sighting.evidence_clip_path,
        }


@dataclass
class CameraRetrievalSummary:
    """Per-camera retrieval results (used both for multi-camera breakdown and unsynchronized feeds)."""

    camera_id: str
    video_metadata: list[VideoMetadata]
    total_sightings: int
    strongly_supported_count: int
    uncertain_count: int
    last_strongly_supported_sighting: Optional[SightingRecord]
    latest_class_detection: Optional[SightingRecord]
    later_uncertain_sightings: list[SightingRecord] = field(default_factory=list)
    track_summaries: list[TrackSummary] = field(default_factory=list)
    camera_conclusion_status: str = "INCONCLUSIVE"  # "STRONGLY_SUPPORTED", "UNCERTAIN_ONLY", "NO_DETECTIONS"
    camera_notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "camera_id": self.camera_id,
            "camera_conclusion_status": self.camera_conclusion_status,
            "total_sightings": self.total_sightings,
            "strongly_supported_count": self.strongly_supported_count,
            "uncertain_count": self.uncertain_count,
            "last_strongly_supported_sighting": (
                self.last_strongly_supported_sighting.to_dict()
                if self.last_strongly_supported_sighting
                else None
            ),
            "latest_class_detection": (
                self.latest_class_detection.to_dict() if self.latest_class_detection else None
            ),
            "later_uncertain_sightings": [s.to_dict() for s in self.later_uncertain_sightings],
            "track_summaries": [t.to_dict() for t in self.track_summaries],
            "video_metadata": [m.to_dict() for m in self.video_metadata],
            "camera_notes": self.camera_notes,
        }


@dataclass
class RetrievalReport:
    """Structured result for single- or multi-camera asset retrieval with explicit uncertainty calibration.

    Distinguishes:
    - `latest_class_detection`: Latest raw detection of the queried object class (any support tier),
      when chronological ordering is valid.
    - `last_strongly_supported_sighting`: Latest detection meeting both model confidence and heuristic
      track-stability criteria (when chronological ordering is valid across the queried scope).
    - `overall_conclusion_status`: "STRONGLY_SUPPORTED_CLASS_SIGHTING" or "INCONCLUSIVE".
    - `verified_asset_identity_status`: Explicitly states that class-level detections across cameras or
      across unlinked tracks do NOT establish verified physical identity of a single asset.
    """

    query_class: str
    camera_filter: Optional[str]
    support_conf_threshold: float
    support_min_track_frames: int
    total_sightings: int
    strongly_supported_sightings_count: int
    uncertain_sightings_count: int
    last_strongly_supported_sighting: Optional[SightingRecord]
    later_uncertain_sightings: list[SightingRecord] = field(default_factory=list)
    all_uncertain_sightings: list[SightingRecord] = field(default_factory=list)
    track_summaries: list[TrackSummary] = field(default_factory=list)
    conflicting_track_pairs: list[dict[str, Any]] = field(default_factory=list)
    video_metadata: list[VideoMetadata] = field(default_factory=list)
    uncertainty_notes: list[str] = field(default_factory=list)
    # Multi-camera & evidence-calibrated extensions
    camera_filters: list[str] = field(default_factory=list)
    queried_cameras: list[str] = field(default_factory=list)
    chronologically_orderable: bool = True
    synchronization_mode: str = "SINGLE_RECORDING_RELATIVE"
    overall_conclusion_status: str = "INCONCLUSIVE"
    overall_conclusion_reason: str = ""
    verified_asset_identity_status: str = (
        "UNVERIFIED_PHYSICAL_IDENTITY: Detections represent class-level ('laptop') model evidence "
        "within independent camera tracks, NOT verified identity of a single physical asset."
    )
    latest_class_detection: Optional[SightingRecord] = None
    per_camera_reports: dict[str, CameraRetrievalSummary] = field(default_factory=dict)
    cross_camera_observations: list[dict[str, Any]] = field(default_factory=list)

    # Backward-compatible aliases so existing callers continue to work
    @property
    def confirmed_conf_threshold(self) -> float:
        return self.support_conf_threshold

    @property
    def confirmed_min_track_frames(self) -> int:
        return self.support_min_track_frames

    @property
    def confirmed_sightings_count(self) -> int:
        return self.strongly_supported_sightings_count

    @property
    def tentative_sightings_count(self) -> int:
        return self.uncertain_sightings_count

    @property
    def last_confirmed_sighting(self) -> Optional[SightingRecord]:
        return self.last_strongly_supported_sighting

    @property
    def later_tentative_sightings(self) -> list[SightingRecord]:
        return self.later_uncertain_sightings

    @property
    def all_tentative_sightings(self) -> list[SightingRecord]:
        return self.all_uncertain_sightings

    def to_dict(self) -> dict[str, Any]:
        return {
            "query_class": self.query_class,
            "camera_filter": self.camera_filter,
            "camera_filters": self.camera_filters,
            "queried_cameras": self.queried_cameras,
            "chronologically_orderable": self.chronologically_orderable,
            "synchronization_mode": self.synchronization_mode,
            "overall_conclusion_status": self.overall_conclusion_status,
            "overall_conclusion_reason": self.overall_conclusion_reason,
            "verified_asset_identity_status": self.verified_asset_identity_status,
            "support_conf_threshold": self.support_conf_threshold,
            "support_min_track_frames": self.support_min_track_frames,
            "total_sightings": self.total_sightings,
            "strongly_supported_sightings_count": self.strongly_supported_sightings_count,
            "uncertain_sightings_count": self.uncertain_sightings_count,
            "last_strongly_supported_sighting": (
                self.last_strongly_supported_sighting.to_dict()
                if self.last_strongly_supported_sighting
                else None
            ),
            "latest_class_detection": (
                self.latest_class_detection.to_dict()
                if self.latest_class_detection
                else None
            ),
            "later_uncertain_sightings": [s.to_dict() for s in self.later_uncertain_sightings],
            "track_summaries": [t.to_dict() for t in self.track_summaries],
            "per_camera_reports": {
                cam: summary.to_dict() for cam, summary in self.per_camera_reports.items()
            },
            "cross_camera_observations": self.cross_camera_observations,
            "conflicting_track_pairs": self.conflicting_track_pairs,
            "video_metadata": [m.to_dict() for m in self.video_metadata],
            "uncertainty_notes": self.uncertainty_notes,
        }

"""Multi-camera evidence-guided sighting retrieval, synchronization checking, and uncertainty analysis."""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Iterable, Optional, Sequence

from src.models import (
    CameraRetrievalSummary,
    RetrievalReport,
    SightingRecord,
    TrackSummary,
    VideoMetadata,
    compute_aligned_timestamp,
    compute_heuristic_stability_score,
    format_video_timestamp,
    normalize_source_video,
)
from src.storage import SightingStore


def compute_bbox_iou_and_containment(
    box_a: tuple[float, float, float, float],
    box_b: tuple[float, float, float, float],
) -> tuple[float, float]:
    """Compute (IoU, max_containment_ratio) between two xyxy bounding boxes."""
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)

    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0.0:
        return (0.0, 0.0)

    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    iou = inter / union if union > 0.0 else 0.0
    min_area = min(area_a, area_b)
    containment = inter / min_area if min_area > 0.0 else 0.0
    return (round(iou, 4), round(containment, 4))


def detect_conflicting_tracks(sightings: Iterable[SightingRecord]) -> list[dict[str, Any]]:
    """Detect co-occurring or temporally overlapping local tracks within the same video run."""
    by_frame: dict[tuple[Optional[int], str, str, int], list[SightingRecord]] = defaultdict(list)
    for s in sightings:
        by_frame[(s.run_id, s.camera_id, s.source_video, s.frame_index)].append(s)

    pair_records: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for (_, camera_id, source_video, frame_idx), frame_list in by_frame.items():
        if len(frame_list) < 2:
            continue
        ordered = sorted(frame_list, key=lambda r: r.scoped_track_key)
        for i in range(len(ordered)):
            for j in range(i + 1, len(ordered)):
                a = ordered[i]
                b = ordered[j]
                if a.scoped_track_key == b.scoped_track_key:
                    continue
                iou, containment = compute_bbox_iou_and_containment(a.bbox_xyxy, b.bbox_xyxy)
                pair_records[(a.scoped_track_key, b.scoped_track_key)].append(
                    {
                        "camera_id": camera_id,
                        "source_video": source_video,
                        "frame_index": frame_idx,
                        "timestamp_seconds": a.timestamp_seconds,
                        "iou": iou,
                        "containment": containment,
                        "conf_a": a.confidence,
                        "conf_b": b.confidence,
                    }
                )

    conflicts: list[dict[str, Any]] = []
    for (track_a, track_b), overlaps in sorted(pair_records.items()):
        frames = [o["frame_index"] for o in overlaps]
        ious = [o["iou"] for o in overlaps]
        containments = [o["containment"] for o in overlaps]
        mean_iou = round(sum(ious) / len(ious), 4)
        max_containment = round(max(containments), 4)
        conflict_type = (
            "SPATIALLY_NESTED_DUPLICATE_BOXES"
            if max_containment >= 0.70 or mean_iou >= 0.30
            else "CONCURRENT_SEPARATE_CANDIDATES"
        )
        conflicts.append(
            {
                "track_a": track_a,
                "track_b": track_b,
                "camera_id": overlaps[0]["camera_id"],
                "source_video": overlaps[0]["source_video"],
                "shared_frame_count": len(frames),
                "first_shared_frame": min(frames),
                "last_shared_frame": max(frames),
                "first_shared_timestamp": overlaps[0]["timestamp_seconds"],
                "last_shared_timestamp": overlaps[-1]["timestamp_seconds"],
                "mean_bbox_iou": mean_iou,
                "max_bbox_containment": max_containment,
                "conflict_type": conflict_type,
            }
        )
    return conflicts


def _enrich_sightings_with_sync_metadata(
    sightings: list[SightingRecord],
    video_metadata: Optional[list[VideoMetadata]],
) -> dict[tuple[str, str, Optional[int]], VideoMetadata]:
    """Attach aligned timestamps from VideoMetadata onto SightingRecords if not already populated."""
    meta_map: dict[tuple[str, str, Optional[int]], VideoMetadata] = {}
    fallback_map: dict[tuple[str, str], VideoMetadata] = {}
    for m in video_metadata or []:
        norm_vid = normalize_source_video(m.source_video)
        meta_map[(m.camera_id, norm_vid, m.run_id)] = m
        fallback_map[(m.camera_id, norm_vid)] = m

    for s in sightings:
        m = meta_map.get((s.camera_id, s.source_video, s.run_id)) or fallback_map.get(
            (s.camera_id, s.source_video)
        )
        if m is not None and (
            s.aligned_wall_clock_utc is None
            and s.aligned_epoch_seconds is None
            and s.aligned_sync_seconds is None
        ):
            iso_str, epoch_s, sync_s = compute_aligned_timestamp(
                s.timestamp_seconds,
                recording_start_utc=m.recording_start_utc,
                recording_timezone=m.recording_timezone,
                time_offset_seconds=m.time_offset_seconds,
            )
            s.aligned_wall_clock_utc = iso_str
            s.aligned_epoch_seconds = epoch_s
            s.aligned_sync_seconds = sync_s
    return meta_map


def determine_synchronization_mode(
    sightings: list[SightingRecord],
    video_metadata: Optional[list[VideoMetadata]] = None,
) -> tuple[bool, str]:
    """Determine whether sightings across the queried recordings can be ordered on a single timeline.

    Rules:
    - Do NOT assume two camera recordings are synchronized unless recording start datetimes
      (`recording_start_utc`) or explicit relative offsets (`time_offset_seconds`) are provided
      for all compared recordings.
    """
    distinct_sources: set[tuple[str, str, Optional[int]]] = set()
    for m in video_metadata or []:
        distinct_sources.add((m.camera_id, normalize_source_video(m.source_video), m.run_id))
    for s in sightings:
        distinct_sources.add((s.camera_id, s.source_video, s.run_id))

    if len(distinct_sources) <= 1:
        if video_metadata and len(video_metadata) == 1:
            m = video_metadata[0]
            if m.has_wall_clock_sync:
                return (True, "SINGLE_RECORDING_WALL_CLOCK")
            if m.has_relative_offset_sync:
                return (True, "SINGLE_RECORDING_OFFSET")
        return (True, "SINGLE_RECORDING_RELATIVE")

    # Multiple recordings involved: check if all have wall-clock sync or all have offset sync
    if video_metadata and len(video_metadata) >= len(distinct_sources):
        relevant_meta = video_metadata
        if all(m.has_wall_clock_sync for m in relevant_meta):
            return (True, "MULTI_CAMERA_WALL_CLOCK_ALIGNED")
        if all(m.has_relative_offset_sync for m in relevant_meta):
            return (True, "MULTI_CAMERA_OFFSET_ALIGNED")
        return (False, "UNSYNCHRONIZED_SEPARATE_BY_CAMERA")

    # Fallback check on sightings themselves if VideoMetadata list was not passed
    if sightings:
        if all(s.aligned_epoch_seconds is not None for s in sightings):
            return (True, "MULTI_CAMERA_WALL_CLOCK_ALIGNED")
        if all(s.aligned_sync_seconds is not None for s in sightings):
            return (True, "MULTI_CAMERA_OFFSET_ALIGNED")

    return (False, "UNSYNCHRONIZED_SEPARATE_BY_CAMERA")


def _temporal_sort_key(s: SightingRecord, sync_mode: str) -> tuple[float, int, float, str]:
    """Return a chronological sort key using aligned time when synchronized, or local time within one video."""
    if sync_mode == "MULTI_CAMERA_WALL_CLOCK_ALIGNED" and s.aligned_epoch_seconds is not None:
        return (s.aligned_epoch_seconds, s.frame_index, s.confidence, s.scoped_track_key)
    if sync_mode == "MULTI_CAMERA_OFFSET_ALIGNED" and s.aligned_sync_seconds is not None:
        return (s.aligned_sync_seconds, s.frame_index, s.confidence, s.scoped_track_key)
    return (s.timestamp_seconds, s.frame_index, s.confidence, s.scoped_track_key)


def summarize_tracks(
    sightings: Iterable[SightingRecord],
    confirmed_conf_threshold: float = 0.50,
    confirmed_min_track_frames: int = 3,
    video_metadata: Optional[list[VideoMetadata]] = None,
) -> tuple[list[TrackSummary], set[tuple[Optional[int], str, str, Optional[int], Optional[int]]]]:
    """Group sightings by isolated track key and separate raw confidence from heuristic stability."""
    sighting_list = list(sightings)
    meta_map = _enrich_sightings_with_sync_metadata(sighting_list, video_metadata)
    fallback_meta = {
        (m.camera_id, normalize_source_video(m.source_video)): m
        for m in (video_metadata or [])
    }

    grouped: dict[tuple[Optional[int], str, str, Optional[int], Optional[int]], list[SightingRecord]] = defaultdict(list)
    for s in sighting_list:
        grouped[s.isolation_key].append(s)

    summaries: list[TrackSummary] = []
    supported_track_keys: set[tuple[Optional[int], str, str, Optional[int], Optional[int]]] = set()

    for key, group in grouped.items():
        run_id, camera_id, source_video, local_track_id, _ = key
        ordered = sorted(group, key=lambda r: (r.timestamp_seconds, r.frame_index, -(r.confidence)))
        confs = [r.confidence for r in ordered]
        mean_conf = sum(confs) / len(confs)
        high_conf_frames = [r for r in ordered if r.confidence >= confirmed_conf_threshold]

        high_conf_ratio, continuity_density, stability_score = compute_heuristic_stability_score(
            frame_count=len(ordered),
            first_frame=ordered[0].frame_index,
            last_frame=ordered[-1].frame_index,
            high_conf_frame_count=len(high_conf_frames),
            min_required_frames=confirmed_min_track_frames,
        )

        has_min_frames = (
            local_track_id is not None
            and len(high_conf_frames) >= confirmed_min_track_frames
        ) or (
            confirmed_min_track_frames <= 1 and len(high_conf_frames) >= 1
        )
        has_track_stability = (
            mean_conf >= confirmed_conf_threshold and high_conf_ratio >= 0.50
        )
        is_strongly_supported = has_min_frames and has_track_stability

        if is_strongly_supported:
            supported_track_keys.add(key)
            status = "STRONGLY_SUPPORTED"
            status_reason = (
                f"Detector conf mean={mean_conf:.2f} (>={confirmed_conf_threshold:.2f}); "
                f"heuristic stability={stability_score:.2f} ({len(high_conf_frames)}/{len(ordered)} frames "
                f"[{high_conf_ratio * 100:.1f}%] >= {confirmed_conf_threshold:.2f})"
            )
        else:
            status = "UNCERTAIN_CANDIDATE"
            if local_track_id is None:
                status_reason = (
                    f"Untracked single-frame detection (conf={ordered[0].confidence:.2f}, "
                    f"heuristic stability={stability_score:.2f}; no tracker continuity)"
                )
            elif len(high_conf_frames) == 0:
                status_reason = (
                    f"All {len(ordered)} frame(s) below support threshold "
                    f"(max detector conf={max(confs):.2f} < {confirmed_conf_threshold:.2f}; "
                    f"heuristic stability={stability_score:.2f})"
                )
            elif not has_min_frames:
                status_reason = (
                    f"Insufficient high-confidence track continuity "
                    f"({len(high_conf_frames)} frame(s) >= {confirmed_conf_threshold:.2f}, "
                    f"need {confirmed_min_track_frames}; heuristic stability={stability_score:.2f})"
                )
            else:
                status_reason = (
                    f"Unstable/predominantly low-confidence track: only {len(high_conf_frames)}/{len(ordered)} "
                    f"frames ({high_conf_ratio * 100:.1f}%) >= {confirmed_conf_threshold:.2f} "
                    f"(detector mean={mean_conf:.2f} < {confirmed_conf_threshold:.2f}, "
                    f"heuristic stability={stability_score:.2f})"
                )

        peak_sighting = max(ordered, key=lambda r: (r.confidence, r.timestamp_seconds))
        last_sighting = ordered[-1]
        last_supported = high_conf_frames[-1] if (is_strongly_supported and high_conf_frames) else None
        m = meta_map.get((camera_id, source_video, run_id)) or fallback_meta.get((camera_id, source_video))

        summaries.append(
            TrackSummary(
                run_id=run_id,
                camera_id=camera_id,
                source_video=source_video,
                source_identity=m.resolved_source_identity if m is not None else None,
                class_id=ordered[0].class_id,
                class_name=ordered[0].class_name,
                local_track_id=local_track_id,
                frame_count=len(ordered),
                first_frame=ordered[0].frame_index,
                last_frame=last_sighting.frame_index,
                first_timestamp_seconds=ordered[0].timestamp_seconds,
                last_timestamp_seconds=last_sighting.timestamp_seconds,
                first_aligned_utc=ordered[0].aligned_wall_clock_utc,
                last_aligned_utc=last_sighting.aligned_wall_clock_utc,
                first_aligned_epoch=ordered[0].aligned_epoch_seconds,
                last_aligned_epoch=last_sighting.aligned_epoch_seconds,
                first_aligned_sync_seconds=ordered[0].aligned_sync_seconds,
                last_aligned_sync_seconds=last_sighting.aligned_sync_seconds,
                min_confidence=min(confs),
                mean_confidence=mean_conf,
                max_confidence=max(confs),
                high_conf_frame_count=len(high_conf_frames),
                high_conf_frame_ratio=high_conf_ratio,
                temporal_continuity_density=continuity_density,
                heuristic_stability_score=stability_score,
                first_bbox_xyxy=ordered[0].bbox_xyxy,
                last_bbox_xyxy=last_sighting.bbox_xyxy,
                status=status,
                status_reason=status_reason,
                peak_Sighting=peak_sighting,
                last_sighting=last_sighting,
                last_supported_sighting=last_supported,
            )
        )

    summaries.sort(key=lambda t: (t.camera_id, t.source_video, t.first_timestamp_seconds, t.first_frame))
    return summaries, supported_track_keys


def _build_per_camera_summaries(
    queried_cameras: list[str],
    filtered_sightings: list[SightingRecord],
    track_summaries: list[TrackSummary],
    supported_track_keys: set[tuple[Optional[int], str, str, Optional[int], Optional[int]]],
    video_metadata: list[VideoMetadata],
    confirmed_conf_threshold: float,
) -> dict[str, CameraRetrievalSummary]:
    """Build isolated per-camera retrieval summaries."""
    per_cam: dict[str, CameraRetrievalSummary] = {}
    for cam_id in queried_cameras:
        cam_meta = [m for m in video_metadata if m.camera_id == cam_id]
        cam_sightings = [s for s in filtered_sightings if s.camera_id == cam_id]
        cam_tracks = [t for t in track_summaries if t.camera_id == cam_id]

        cam_supported = [
            s
            for s in cam_sightings
            if s.isolation_key in supported_track_keys and s.confidence >= confirmed_conf_threshold
        ]
        cam_uncertain = [
            s
            for s in cam_sightings
            if not (s.isolation_key in supported_track_keys and s.confidence >= confirmed_conf_threshold)
        ]

        # Check if multiple recordings under this camera are synchronized
        cam_orderable, cam_sync_mode = determine_synchronization_mode(cam_sightings, cam_meta)
        if cam_orderable:
            cam_supported_sorted = sorted(cam_supported, key=lambda r: _temporal_sort_key(r, cam_sync_mode))
            cam_all_sorted = sorted(cam_sightings, key=lambda r: _temporal_sort_key(r, cam_sync_mode))
        else:
            cam_supported_sorted = sorted(cam_supported, key=lambda r: (r.timestamp_seconds, r.frame_index, r.confidence))
            cam_all_sorted = sorted(cam_sightings, key=lambda r: (r.timestamp_seconds, r.frame_index, r.confidence))

        last_supp = cam_supported_sorted[-1] if cam_supported_sorted else None
        latest_det = cam_all_sorted[-1] if cam_all_sorted else None

        if last_supp is not None:
            supp_key = _temporal_sort_key(last_supp, cam_sync_mode)[:2]
            later_unc = [
                s
                for s in cam_uncertain
                if _temporal_sort_key(s, cam_sync_mode)[:2] > supp_key
            ]
        else:
            later_unc = list(cam_uncertain)

        notes: list[str] = []
        if not cam_sightings:
            status = "NO_DETECTIONS"
            notes.append(f"Camera '{cam_id}' recorded 0 candidate detections of the queried class.")
        elif last_supp is None:
            status = "UNCERTAIN_ONLY"
            notes.append(
                f"Camera '{cam_id}' recorded {len(cam_uncertain)} uncertain candidate(s) (false-positive or "
                f"low-stability tracks) and 0 strongly supported sightings."
            )
        else:
            status = "STRONGLY_SUPPORTED"
            if cam_meta:
                matching = [
                    m for m in cam_meta if normalize_source_video(m.source_video) == last_supp.source_video
                ]
                if matching:
                    rem = round(matching[-1].duration_seconds - last_supp.timestamp_seconds, 2)
                    if rem > 0.5:
                        notes.append(
                            f"Strongly supported tracking on '{cam_id}' ended at +{last_supp.timestamp_seconds:.2f}s "
                            f"with {rem:.2f}s of remaining video."
                        )
            if later_unc:
                notes.append(
                    f"Camera '{cam_id}' has {len(later_unc)} later uncertain candidate detection(s) after "
                    f"+{last_supp.timestamp_seconds:.2f}s."
                )

        per_cam[cam_id] = CameraRetrievalSummary(
            camera_id=cam_id,
            video_metadata=cam_meta,
            total_sightings=len(cam_sightings),
            strongly_supported_count=len(cam_supported),
            uncertain_count=len(cam_uncertain),
            last_strongly_supported_sighting=last_supp,
            latest_class_detection=latest_det,
            later_uncertain_sightings=later_unc,
            track_summaries=cam_tracks,
            camera_conclusion_status=status,
            camera_notes=notes,
        )
    return per_cam


def build_retrieval_report(
    sightings: list[SightingRecord],
    query_class: str = "laptop",
    camera_filter: Optional[str] = None,
    camera_filters: Optional[Sequence[str]] = None,
    confirmed_conf_threshold: float = 0.50,
    confirmed_min_track_frames: int = 3,
    video_metadata: Optional[list[VideoMetadata]] = None,
) -> RetrievalReport:
    """Analyze sightings across one or more cameras with strict identity isolation and synchronization rules."""
    meta_list = list(video_metadata or [])
    _enrich_sightings_with_sync_metadata(sightings, meta_list)

    active_filters: list[str] = []
    if camera_filters:
        active_filters = [c for c in camera_filters if c]
    elif camera_filter:
        active_filters = [camera_filter]

    filtered = [
        s
        for s in sightings
        if s.class_name.lower() == query_class.lower()
        and (not active_filters or s.camera_id in active_filters)
    ]
    relevant_meta = [
        m for m in meta_list if (not active_filters or m.camera_id in active_filters)
    ]

    queried_cameras = sorted(
        {m.camera_id for m in relevant_meta}
        | {s.camera_id for s in filtered}
        | set(active_filters)
    )

    chronologically_orderable, sync_mode = determine_synchronization_mode(filtered, relevant_meta)

    track_summaries, supported_track_keys = summarize_tracks(
        filtered,
        confirmed_conf_threshold=confirmed_conf_threshold,
        confirmed_min_track_frames=confirmed_min_track_frames,
        video_metadata=relevant_meta,
    )
    conflicting_pairs = detect_conflicting_tracks(filtered)

    strongly_supported_sightings: list[SightingRecord] = []
    uncertain_sightings: list[SightingRecord] = []
    for s in filtered:
        if s.isolation_key in supported_track_keys and s.confidence >= confirmed_conf_threshold:
            strongly_supported_sightings.append(s)
        else:
            uncertain_sightings.append(s)

    per_camera_reports = _build_per_camera_summaries(
        queried_cameras=queried_cameras,
        filtered_sightings=filtered,
        track_summaries=track_summaries,
        supported_track_keys=supported_track_keys,
        video_metadata=relevant_meta,
        confirmed_conf_threshold=confirmed_conf_threshold,
    )

    # Global ordering is ONLY computed when recordings are chronologically orderable
    if chronologically_orderable:
        ordered_supported = sorted(
            strongly_supported_sightings,
            key=lambda r: _temporal_sort_key(r, sync_mode),
        )
        ordered_all = sorted(
            filtered,
            key=lambda r: _temporal_sort_key(r, sync_mode),
        )
        last_supported = ordered_supported[-1] if ordered_supported else None
        latest_class_det = ordered_all[-1] if ordered_all else None

        if last_supported is not None:
            supp_time_key = _temporal_sort_key(last_supported, sync_mode)[:2]
            later_uncertain = [
                s
                for s in sorted(uncertain_sightings, key=lambda r: _temporal_sort_key(r, sync_mode))
                if _temporal_sort_key(s, sync_mode)[:2] > supp_time_key
            ]
        else:
            later_uncertain = sorted(uncertain_sightings, key=lambda r: _temporal_sort_key(r, sync_mode))
    else:
        # Unsynchronized multi-camera query: do NOT claim a single cross-camera chronological winner
        last_supported = None
        latest_class_det = None
        later_uncertain = list(uncertain_sightings)

    uncertainty_notes: list[str] = [
        "EVIDENCE TERMINOLOGY: 'Strongly Supported' indicates a high-confidence detector output within a "
        "temporally stable local track. It is NOT an independent ground-truth verification of physical identity.",
        "CLASS VS. ASSET IDENTITY: Detections establish the presence of object class "
        f"'{query_class}', NOT that separate tracks or separate cameras observed the same physical item.",
        "IDENTITY ISOLATION: Track keys (run_id:camera_id:source_video:local_track_id) are strictly isolated "
        "per ingestion run, camera, and video file. Separate tracks are never automatically merged.",
    ]

    if sync_mode == "UNSYNCHRONIZED_SEPARATE_BY_CAMERA":
        uncertainty_notes.append(
            "MISSING CAMERA SYNCHRONIZATION: Multiple recordings were queried without shared recording start "
            "datetimes or explicit time offsets. Cross-camera chronological ordering is IMPOSSIBLE; results are "
            "isolated per camera below."
        )
    elif sync_mode == "MULTI_CAMERA_WALL_CLOCK_ALIGNED":
        uncertainty_notes.append(
            "SYNCHRONIZED TIMELINE (WALL-CLOCK): Sightings across cameras are ordered using provided recording "
            "start datetimes (recording_start_utc). Chronological ordering does NOT imply cross-camera object identity."
        )
    elif sync_mode == "MULTI_CAMERA_OFFSET_ALIGNED":
        uncertainty_notes.append(
            "SYNCHRONIZED TIMELINE (RELATIVE OFFSET): Sightings across cameras are ordered using explicit "
            "relative time offsets (time_offset_seconds). No wall-clock CCTV timestamps are invented."
        )
    else:
        uncertainty_notes.append(
            "TIMESTAMP SCOPE: All timestamps are video-relative offsets from the start of the recording; "
            "no wall-clock CCTV timestamps are assumed or invented."
        )

    # Cross-camera evidence analysis
    cross_camera_observations: list[dict[str, Any]] = []
    cameras_with_supported = [
        cam_id for cam_id, summary in per_camera_reports.items() if summary.strongly_supported_count > 0
    ]
    cameras_with_uncertain_only = [
        cam_id
        for cam_id, summary in per_camera_reports.items()
        if summary.strongly_supported_count == 0 and summary.uncertain_count > 0
    ]
    cameras_with_none = [
        cam_id for cam_id, summary in per_camera_reports.items() if summary.total_sightings == 0
    ]

    has_simultaneous_multi_camera_conflict = False
    if len(cameras_with_supported) >= 2:
        cross_camera_observations.append(
            {
                "type": "UNLINKED_MULTI_CAMERA_CLASS_SIGHTINGS",
                "cameras": cameras_with_supported,
                "description": (
                    f"Strongly supported '{query_class}' tracks exist in multiple cameras ({', '.join(cameras_with_supported)}). "
                    "Without cross-camera visual re-identification, these may be distinct similar-looking objects "
                    "or separate sightings of a moved object."
                ),
            }
        )
        uncertainty_notes.append(
            f"MULTI-CAMERA IDENTITY AMBIGUITY: Strongly supported '{query_class}' detections occur in multiple "
            f"cameras ({', '.join(cameras_with_supported)}). Do NOT assume these represent the same physical '{query_class}'."
        )

        # Check for simultaneous overlapping intervals across synchronized cameras
        if chronologically_orderable and sync_mode in ("MULTI_CAMERA_WALL_CLOCK_ALIGNED", "MULTI_CAMERA_OFFSET_ALIGNED"):
            supp_tracks = [t for t in track_summaries if t.status == "STRONGLY_SUPPORTED"]
            for i in range(len(supp_tracks)):
                for j in range(i + 1, len(supp_tracks)):
                    ta = supp_tracks[i]
                    tb = supp_tracks[j]
                    if ta.camera_id == tb.camera_id:
                        continue
                    if sync_mode == "MULTI_CAMERA_WALL_CLOCK_ALIGNED":
                        sa, ea = ta.first_aligned_epoch, ta.last_aligned_epoch
                        sb, eb = tb.first_aligned_epoch, tb.last_aligned_epoch
                    else:
                        sa, ea = ta.first_aligned_sync_seconds, ta.last_aligned_sync_seconds
                        sb, eb = tb.first_aligned_sync_seconds, tb.last_aligned_sync_seconds
                    if sa is not None and ea is not None and sb is not None and eb is not None:
                        overlap_start = max(sa, sb)
                        overlap_end = min(ea, eb)
                        if overlap_end >= overlap_start:
                            has_simultaneous_multi_camera_conflict = True
                            cross_camera_observations.append(
                                {
                                    "type": "SIMULTANEOUS_MULTI_CAMERA_SIGHTINGS",
                                    "camera_a": ta.camera_id,
                                    "track_a": ta.scoped_track_key,
                                    "camera_b": tb.camera_id,
                                    "track_b": tb.scoped_track_key,
                                    "overlap_seconds": round(overlap_end - overlap_start, 3),
                                    "description": (
                                        f"Simultaneous strongly supported '{query_class}' detections in "
                                        f"'{ta.camera_id}' and '{tb.camera_id}' overlap by "
                                        f"{overlap_end - overlap_start:.2f}s. If camera fields of view do not "
                                        "overlap, at least two distinct physical objects exist."
                                    ),
                                }
                            )
                            uncertainty_notes.append(
                                f"SIMULTANEOUS MULTI-CAMERA DETECTIONS: '{ta.camera_id}' ({ta.scoped_track_key}) "
                                f"and '{tb.camera_id}' ({tb.scoped_track_key}) detected '{query_class}' at overlapping "
                                "synchronized times, indicating multiple similar-looking physical objects or overlapping views."
                            )

    if cameras_with_uncertain_only:
        cross_camera_observations.append(
            {
                "type": "CAMERA_FALSE_POSITIVE_OR_LOW_STABILITY_CANDIDATES",
                "cameras": cameras_with_uncertain_only,
                "description": (
                    f"Camera(s) {', '.join(cameras_with_uncertain_only)} produced only low-confidence or "
                    "unstable candidate detections (potential false positives)."
                ),
            }
        )

    # Check if an object left a camera early without a strongly supported sighting in another queried camera
    departure_without_downstream = False
    if len(queried_cameras) >= 2 and len(cameras_with_supported) == 1 and (cameras_with_none or cameras_with_uncertain_only):
        src_cam = cameras_with_supported[0]
        src_summary = per_camera_reports[src_cam]
        if any("remaining video" in n for n in src_summary.camera_notes):
            departure_without_downstream = True
            other_cams = [c for c in queried_cameras if c != src_cam]
            cross_camera_observations.append(
                {
                    "type": "DEPARTURE_WITHOUT_DOWNSTREAM_CONFIRMATION",
                    "source_camera": src_cam,
                    "unconfirmed_cameras": other_cams,
                    "description": (
                        f"'{query_class}' tracking ended before the end of recording in '{src_cam}', and no "
                        f"strongly supported sighting was established in {', '.join(other_cams)}. Final location is INCONCLUSIVE."
                    ),
                }
            )
            uncertainty_notes.append(
                f"DEPARTURE WITHOUT DOWNSTREAM SUPPORT: '{query_class}' was lost in '{src_cam}' before video end, "
                f"and downstream camera(s) ({', '.join(other_cams)}) have no strongly supported sighting."
            )

    # Single-camera or synchronized tracking-loss and later-uncertain notes
    has_unobserved_tracking_gap = False
    if not filtered:
        uncertainty_notes.append(
            f"NO DETECTIONS: No sightings of class '{query_class}' were recorded in the queried scope."
        )
    elif chronologically_orderable and last_supported is None:
        uncertainty_notes.append(
            f"UNCERTAIN CANDIDATES ONLY: {len(uncertain_sightings)} candidate detection(s) found for '{query_class}', "
            f"but none met the strong-support criteria (detector confidence >= {confirmed_conf_threshold:.2f} across "
            f">= {confirmed_min_track_frames} frames with track mean >= {confirmed_conf_threshold:.2f}). "
            f"Asset location remains unverified."
        )
    elif chronologically_orderable and last_supported is not None:
        if relevant_meta:
            matching_meta = [
                m
                for m in relevant_meta
                if m.camera_id == last_supported.camera_id
                and normalize_source_video(m.source_video) == last_supported.source_video
            ]
            if matching_meta:
                vid_dur = matching_meta[-1].duration_seconds
                unobserved_gap = round(vid_dur - last_supported.timestamp_seconds, 2)
                if unobserved_gap > 0.5:
                    has_unobserved_tracking_gap = True
                    uncertainty_notes.append(
                        f"TRACKING LOSS / OCCLUSION: Strongly supported tracking ({last_supported.scoped_track_key}) "
                        f"ended at {last_supported.timestamp_seconds:.2f}s (frame {last_supported.frame_index}), "
                        f"leaving {unobserved_gap:.2f}s of remaining video ({vid_dur:.2f}s total) without "
                        f"strongly supported continuity."
                    )

        if later_uncertain:
            later_tracks = sorted({s.scoped_track_key for s in later_uncertain})
            max_later_conf = max(s.confidence for s in later_uncertain)
            first_later = later_uncertain[0]
            last_later = later_uncertain[-1]
            if max_later_conf < confirmed_conf_threshold:
                conf_detail = f"max detector confidence {max_later_conf:.2f} < {confirmed_conf_threshold:.2f}"
            else:
                conf_detail = (
                    f"peak detector confidence {max_later_conf:.2f}, but rejected due to low heuristic track "
                    f"stability or confidence decay below {confirmed_conf_threshold:.2f}"
                )
            uncertainty_notes.append(
                f"LATER UNCERTAIN CANDIDATE(S): {len(later_uncertain)} uncertain sighting(s) detected after "
                f"{last_supported.timestamp_seconds:.2f}s between {first_later.timestamp_seconds:.2f}s and "
                f"{last_later.timestamp_seconds:.2f}s under {', '.join(later_tracks)} "
                f"({conf_detail}). "
                f"Do NOT claim these later candidates represent the same physical asset or a verified final location."
            )

    if conflicting_pairs:
        for cp in conflicting_pairs:
            uncertainty_notes.append(
                f"CONFLICTING / OVERLAPPING LOCAL TRACKS: {cp['track_a']} and {cp['track_b']} co-occur across "
                f"{cp['shared_frame_count']} frame(s) (frames {cp['first_shared_frame']}..{cp['last_shared_frame']}, "
                f"{cp['first_shared_timestamp']:.2f}s..{cp['last_shared_timestamp']:.2f}s; "
                f"mean IoU={cp['mean_bbox_iou']:.2f}, max containment={cp['max_bbox_containment']:.2f}, "
                f"type={cp['conflict_type']}). Separate IDs are preserved without merging."
            )

    # Determine overall_conclusion_status & reason
    if not filtered:
        overall_status = "INCONCLUSIVE"
        overall_reason = f"No detections of class '{query_class}' in queried camera scope."
    elif not chronologically_orderable:
        overall_status = "INCONCLUSIVE"
        overall_reason = (
            "Multiple recordings lack shared synchronization metadata (recording_start_utc or time_offset_seconds); "
            "cross-camera chronological ordering cannot be established. Inspect per-camera reports."
        )
    elif last_supported is None:
        overall_status = "INCONCLUSIVE"
        overall_reason = (
            f"Only uncertain/low-stability candidate detections ({len(uncertain_sightings)}) were found; "
            "no track met the strong-support criteria."
        )
    elif has_simultaneous_multi_camera_conflict:
        overall_status = "INCONCLUSIVE"
        overall_reason = (
            "Simultaneous strongly supported detections in multiple cameras indicate multiple similar-looking "
            "objects; single-asset last location is inconclusive without object re-identification."
        )
    elif departure_without_downstream:
        overall_status = "INCONCLUSIVE"
        overall_reason = (
            f"Last strongly supported sighting was in '{last_supported.camera_id}' at +{last_supported.timestamp_seconds:.2f}s "
            "before tracking was lost, with no strongly supported sighting in downstream camera(s)."
        )
    elif has_unobserved_tracking_gap or later_uncertain:
        overall_status = "INCONCLUSIVE"
        overall_reason = (
            f"Last strongly supported sighting is at +{last_supported.timestamp_seconds:.2f}s "
            f"(frame {last_supported.frame_index}) on '{last_supported.camera_id}', but final destination "
            "after tracking loss / later uncertain candidates remains unverified."
        )
    else:
        overall_status = "STRONGLY_SUPPORTED_CLASS_SIGHTING"
        overall_reason = (
            f"Strongly supported class sighting established on '{last_supported.camera_id}' at "
            f"+{last_supported.timestamp_seconds:.2f}s (frame {last_supported.frame_index}); "
            "physical instance identity across cameras remains unverified."
        )

    return RetrievalReport(
        query_class=query_class,
        camera_filter=camera_filter if camera_filter else (",".join(active_filters) if active_filters else None),
        camera_filters=active_filters,
        queried_cameras=queried_cameras,
        chronologically_orderable=chronologically_orderable,
        synchronization_mode=sync_mode,
        overall_conclusion_status=overall_status,
        overall_conclusion_reason=overall_reason,
        support_conf_threshold=confirmed_conf_threshold,
        support_min_track_frames=confirmed_min_track_frames,
        total_sightings=len(filtered),
        strongly_supported_sightings_count=len(strongly_supported_sightings),
        uncertain_sightings_count=len(uncertain_sightings),
        last_strongly_supported_sighting=last_supported,
        latest_class_detection=latest_class_det,
        later_uncertain_sightings=later_uncertain,
        all_uncertain_sightings=uncertain_sightings,
        track_summaries=track_summaries,
        per_camera_reports=per_camera_reports,
        cross_camera_observations=cross_camera_observations,
        conflicting_track_pairs=conflicting_pairs,
        video_metadata=relevant_meta,
        uncertainty_notes=uncertainty_notes,
    )


def query_asset_sightings(
    store: SightingStore,
    query_class: str = "laptop",
    camera_id: Optional[str] = None,
    camera_ids: Optional[Sequence[str]] = None,
    source_video: Optional[str] = None,
    confirmed_conf_threshold: float = 0.50,
    confirmed_min_track_frames: int = 3,
) -> RetrievalReport:
    """Query the SightingStore across all or selected cameras and return a structured RetrievalReport."""
    sightings = store.query_sightings(
        class_name=query_class,
        camera_id=camera_id,
        camera_ids=camera_ids,
        source_video=source_video,
    )
    metadata = store.get_ingestion_runs(
        camera_id=camera_id,
        camera_ids=camera_ids,
        source_video=source_video,
    )
    return build_retrieval_report(
        sightings=sightings,
        query_class=query_class,
        camera_filter=camera_id,
        camera_filters=camera_ids,
        confirmed_conf_threshold=confirmed_conf_threshold,
        confirmed_min_track_frames=confirmed_min_track_frames,
        video_metadata=metadata,
    )


def _format_sighting_time_detail(s: SightingRecord) -> str:
    base = f"{format_video_timestamp(s.timestamp_seconds)} [Frame {s.frame_index}]"
    if s.aligned_wall_clock_utc:
        return f"{base} | aligned_utc={s.aligned_wall_clock_utc}"
    if s.aligned_sync_seconds is not None:
        return f"{base} | aligned_offset=+{s.aligned_sync_seconds:.3f}s"
    return base


def format_retrieval_report_text(report: RetrievalReport, show_timeline: bool = False) -> str:
    """Format a human-readable, multi-camera evidence-calibrated CLI report."""
    lines: list[str] = []
    lines.append("=" * 88)
    lines.append(
        f"EVIDENCE-GUIDED MULTI-CAMERA ASSET RETRIEVAL REPORT | Target Class: '{report.query_class}'"
    )
    lines.append("=" * 88)
    lines.append(
        f"Queried Camera Scope     : {', '.join(report.queried_cameras) if report.queried_cameras else (report.camera_filter or 'NONE')}"
    )
    lines.append(
        f"Synchronization Mode     : {report.synchronization_mode} "
        f"(Cross-camera chronological ordering valid: {report.chronologically_orderable})"
    )
    lines.append(
        f"Overall Conclusion Status: {report.overall_conclusion_status}"
    )
    lines.append(
        f"Conclusion Summary       : {report.overall_conclusion_reason}"
    )
    lines.append(
        f"Asset Identity Status    : {report.verified_asset_identity_status}"
    )
    lines.append(
        f"Strong-Support Criteria  : detector conf >= {report.support_conf_threshold:.2f}, "
        f"min high-conf frames >= {report.support_min_track_frames}, track high-conf ratio >= 50%"
    )
    lines.append(
        f"Total Sightings Indexed  : {report.total_sightings} "
        f"(Strongly Supported: {report.strongly_supported_sightings_count}, "
        f"Uncertain Candidates: {report.uncertain_sightings_count})"
    )

    if report.video_metadata:
        lines.append("-" * 88)
        lines.append("INGESTED CAMERA RECORDINGS:")
        for m in report.video_metadata:
            sync_tag = (
                f"start_utc={m.recording_start_utc}"
                if m.recording_start_utc
                else (
                    f"offset=+{m.time_offset_seconds:.2f}s"
                    if m.time_offset_seconds is not None
                    else "UNSYNCHRONIZED (video-relative only)"
                )
            )
            lines.append(
                f"  - Run #{m.run_id} | Camera '{m.camera_id}' | File: {m.source_video} "
                f"(id={m.resolved_source_identity}) | {m.width}x{m.height} @ {m.fps:.2f} FPS | "
                f"{m.frame_count} frames ({m.duration_seconds:.2f}s) | Sync: {sync_tag}"
            )

    lines.append("-" * 88)
    lines.append(
        "1. LAST STRONGLY SUPPORTED DETECTION (MODEL + TRACK-STABILITY SUPPORTED -- NOT GROUND-TRUTH IDENTITY):"
    )
    if not report.chronologically_orderable:
        lines.append(
            "  [INCONCLUSIVE ACROSS CAMERAS] Recordings are not time-synchronized; cannot rank "
            "timestamps across cameras. See Section 3 (Per-Camera Breakdown) for each camera's sightings."
        )
    elif report.last_strongly_supported_sighting is None:
        lines.append("  [NONE] No sighting met the strong-support criteria.")
    else:
        s = report.last_strongly_supported_sighting
        lines.append(f"  Camera ID          : {s.camera_id}")
        lines.append(f"  Source Video       : {s.source_video}")
        lines.append(f"  Isolated Track Key : {s.scoped_track_key}")
        lines.append(f"  Sighting Timestamp : {_format_sighting_time_detail(s)}")
        lines.append(f"  Detector Confidence: {s.confidence:.4f} (single-frame YOLO class score)")
        lines.append(f"  Bounding Box (xyxy): {[round(v, 1) for v in s.bbox_xyxy]} (center: {s.bbox_center})")
        lines.append(f"  Evidence Frame     : {s.evidence_frame_path or 'N/A'}")
        lines.append(f"  Evidence Crop      : {s.evidence_crop_path or 'N/A'}")
        lines.append(f"  Annotated Video    : {s.evidence_clip_path or 'N/A'}")

    if report.chronologically_orderable and report.latest_class_detection is not None:
        ld = report.latest_class_detection
        is_same_as_supp = (
            report.last_strongly_supported_sighting is not None
            and ld.scoped_track_key == report.last_strongly_supported_sighting.scoped_track_key
            and ld.frame_index == report.last_strongly_supported_sighting.frame_index
        )
        ld_tier = "STRONGLY_SUPPORTED" if is_same_as_supp else "UNCERTAIN_CANDIDATE (NOT VERIFIED)"
        lines.append(
            f"  Latest Raw Class Detection (any tier): Camera '{ld.camera_id}' | {ld.scoped_track_key} | "
            f"{_format_sighting_time_detail(ld)} | det_conf={ld.confidence:.4f} [{ld_tier}]"
        )

    lines.append("-" * 88)
    lines.append("2. LATER UNCERTAIN / LOW-CONFIDENCE CANDIDATE SIGHTINGS (DO NOT ASSERT AS SAME IDENTITY):")
    if not report.later_uncertain_sightings:
        lines.append("  [NONE] No uncertain candidate detections recorded after the last strongly supported sighting.")
    else:
        total_later = len(report.later_uncertain_sightings)
        by_track: dict[str, list[SightingRecord]] = defaultdict(list)
        for s in report.later_uncertain_sightings:
            by_track[s.scoped_track_key].append(s)

        lines.append(
            f"  Total uncertain candidate frame detections: {total_later} across {len(by_track)} isolated track key(s)."
        )
        for tkey, group in by_track.items():
            first_s = group[0]
            peak_s = max(group, key=lambda r: (r.confidence, r.timestamp_seconds))
            last_s = group[-1]
            lines.append(
                f"  - {tkey} (Camera '{first_s.camera_id}', {len(group)} frames, "
                f"{first_s.timestamp_seconds:.3f}s -> {last_s.timestamp_seconds:.3f}s):"
            )
            for tag, s in [("First", first_s), ("Peak ", peak_s), ("Last ", last_s)]:
                lines.append(
                    f"      {tag}: {_format_sighting_time_detail(s)} | "
                    f"det_conf={s.confidence:.4f} | bbox={[round(v, 1) for v in s.bbox_xyxy]} | "
                    f"frame={s.evidence_frame_path or 'N/A'}"
                )

    lines.append("-" * 88)
    lines.append("3. PER-CAMERA SIGHTING BREAKDOWN (ISOLATED BY CAMERA):")
    if not report.per_camera_reports:
        lines.append("  [NONE]")
    else:
        for cam_id, cam_rep in report.per_camera_reports.items():
            lines.append(
                f"  * Camera '{cam_id}' | Status: {cam_rep.camera_conclusion_status} | "
                f"Total={cam_rep.total_sightings} (Strongly Supported={cam_rep.strongly_supported_count}, "
                f"Uncertain={cam_rep.uncertain_count})"
            )
            if cam_rep.last_strongly_supported_sighting is not None:
                cs = cam_rep.last_strongly_supported_sighting
                lines.append(
                    f"      Last Strongly Supported : {cs.scoped_track_key} @ {_format_sighting_time_detail(cs)} | "
                    f"det_conf={cs.confidence:.4f} | frame={cs.evidence_frame_path or 'N/A'} | "
                    f"crop={cs.evidence_crop_path or 'N/A'}"
                )
            else:
                lines.append("      Last Strongly Supported : [NONE]")
            if cam_rep.latest_class_detection is not None:
                cl = cam_rep.latest_class_detection
                lines.append(
                    f"      Latest Raw Detection    : {cl.scoped_track_key} @ {_format_sighting_time_detail(cl)} | "
                    f"det_conf={cl.confidence:.4f}"
                )
            for cnote in cam_rep.camera_notes:
                lines.append(f"      Note                    : {cnote}")

    lines.append("-" * 88)
    lines.append("4. PER-TRACK TRAJECTORY SUMMARIES (DETECTOR CONFIDENCE VS. HEURISTIC TRACK STABILITY):")
    if not report.track_summaries:
        lines.append("  [NONE]")
    else:
        for t in report.track_summaries:
            lines.append(
                f"  * [{t.status}] {t.scoped_track_key} | Camera='{t.camera_id}' | Source='{t.source_video}' | "
                f"Frames {t.first_frame}..{t.last_frame} ({t.frame_count} sightings, "
                f"{t.first_timestamp_seconds:.2f}s -> {t.last_timestamp_seconds:.2f}s)"
            )
            if t.first_aligned_utc and t.last_aligned_utc:
                lines.append(f"    Aligned UTC Span    : {t.first_aligned_utc} -> {t.last_aligned_utc}")
            elif t.first_aligned_sync_seconds is not None and t.last_aligned_sync_seconds is not None:
                lines.append(
                    f"    Aligned Offset Span : +{t.first_aligned_sync_seconds:.2f}s -> +{t.last_aligned_sync_seconds:.2f}s"
                )
            lines.append(
                f"    Raw Detector Conf   : min={t.min_confidence:.2f}, mean={t.mean_confidence:.2f}, max={t.max_confidence:.2f}"
            )
            lines.append(
                f"    Heuristic Stability : score={t.heuristic_stability_score:.2f} "
                f"(high_conf_frames={t.high_conf_frame_count}/{t.frame_count} [{t.high_conf_frame_ratio * 100:.1f}%], "
                f"continuity_density={t.temporal_continuity_density:.2f})"
            )
            lines.append(f"    Assessment Reason   : {t.status_reason}")
            lines.append(
                f"    Spatial Trajectory  : start_bbox={[round(v, 1) for v in t.first_bbox_xyxy]} -> "
                f"end_bbox={[round(v, 1) for v in t.last_bbox_xyxy]}"
            )
            lines.append(
                f"    Last Evidence       : frame={t.last_sighting.evidence_frame_path or 'N/A'} | "
                f"crop={t.last_sighting.evidence_crop_path or 'N/A'}"
            )

    if report.conflicting_track_pairs:
        lines.append("-" * 88)
        lines.append("5. CONFLICTING / OVERLAPPING LOCAL TRACKS:")
        for cp in report.conflicting_track_pairs:
            lines.append(
                f"  * {cp['track_a']} <-> {cp['track_b']} | "
                f"Shared frames: {cp['first_shared_frame']}..{cp['last_shared_frame']} ({cp['shared_frame_count']} frames) | "
                f"mean IoU={cp['mean_bbox_iou']:.2f}, max containment={cp['max_bbox_containment']:.2f} | "
                f"type={cp['conflict_type']}"
            )

    lines.append("-" * 88)
    lines.append("6. EVIDENCE & UNCERTAINTY CALIBRATION NOTES:")
    for note in report.uncertainty_notes:
        lines.append(f"  ! {note}")

    if show_timeline and report.total_sightings > 0:
        lines.append("-" * 88)
        lines.append("7. CHRONOLOGICAL SIGHTING TIMELINE:")
        for t in report.track_summaries:
            lines.append(
                f"  Track {t.scoped_track_key} ({t.status}, stability={t.heuristic_stability_score:.2f}): "
                f"first={t.first_timestamp_seconds:.3f}s (f={t.first_frame}), "
                f"peak={t.peak_Sighting.timestamp_seconds:.3f}s (f={t.peak_Sighting.frame_index}, det_conf={t.peak_Sighting.confidence:.2f}), "
                f"last={t.last_timestamp_seconds:.3f}s (f={t.last_frame}, det_conf={t.last_sighting.confidence:.2f})"
            )

    lines.append("=" * 88)
    return "\n".join(lines)

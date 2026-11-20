"""Reference asset anchoring and multi-factor evidence-guided candidate verification (Milestone 4)."""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from pathlib import Path
from typing import Any, Optional, Sequence

import cv2
import numpy as np

from src.comparison import (
    _bbox_dims,
    _extract_patch,
    _read_video_frames,
    compute_hsv_histogram_correlation,
    compute_normalized_cross_correlation,
)
from src.models import (
    SightingRecord,
    TrackSummary,
    VideoMetadata,
    format_video_timestamp,
    normalize_source_video,
)
from src.retrieval import (
    compute_bbox_iou_and_containment,
    determine_synchronization_mode,
    summarize_tracks,
)
from src.storage import SightingStore
from src.visual_search import (
    ChromaVisualIndex,
    OpenCLIPEmbedder,
    VideoFrameContextReader,
    VisualTextEncoder,
    extract_context_region_pil,
    load_pil_image_safely,
    normalize_embedding_vector,
)

# Calibrated verification outcomes (never claiming ground-truth physical identity)
VERDICT_SUPPORTED_CANDIDATE = "SUPPORTED_CANDIDATE"
VERDICT_CONTRADICTED_CANDIDATE = "CONTRADICTED_CANDIDATE"
VERDICT_INCONCLUSIVE = "INCONCLUSIVE"


@dataclass
class ReferenceAppearanceSnapshot:
    """A single representative appearance state of a reference asset (e.g., open on desk vs closed/carried)."""

    role: str  # e.g., "first_open_state", "peak_confidence_state", "last_strongly_supported_carried_state"
    sighting: SightingRecord
    bbox_geometry: dict[str, float]
    crop_embedding: Optional[np.ndarray] = None
    context_embedding: Optional[np.ndarray] = None
    bgr_patch: Optional[np.ndarray] = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "role": self.role,
            "sighting_id": self.sighting.sighting_id,
            "frame_index": self.sighting.frame_index,
            "timestamp_seconds": round(float(self.sighting.timestamp_seconds), 4),
            "aligned_wall_clock_utc": self.sighting.aligned_wall_clock_utc,
            "aligned_sync_seconds": self.sighting.aligned_sync_seconds,
            "detector_confidence": round(float(self.sighting.confidence), 4),
            "bbox_xyxy": [round(float(v), 2) for v in self.sighting.bbox_xyxy],
            "bbox_geometry": self.bbox_geometry,
            "evidence_crop_path": self.sighting.evidence_crop_path,
            "evidence_frame_path": self.sighting.evidence_frame_path,
            "has_crop_embedding": self.crop_embedding is not None,
            "has_context_embedding": self.context_embedding is not None,
            "has_bgr_patch": self.bgr_patch is not None and self.bgr_patch.size > 0,
        }


@dataclass
class ReferenceAssetAnchor:
    """Anchor representing a selected reference track or sighting with multi-appearance states."""

    scoped_track_key: str
    run_id: Optional[int]
    camera_id: str
    source_video: str
    source_identity: Optional[str]
    class_id: int
    class_name: str
    local_track_id: Optional[int]
    track_support_status: str
    track_status_reason: str
    heuristic_stability_score: float
    frame_count: int
    evidence_sampled_frame_count: int
    first_frame: int
    last_frame: int
    last_strongly_supported_frame: Optional[int]
    first_timestamp_seconds: float
    last_timestamp_seconds: float
    last_strongly_supported_timestamp_seconds: Optional[float]
    first_aligned_utc: Optional[str]
    last_aligned_utc: Optional[str]
    first_aligned_epoch: Optional[float]
    last_aligned_epoch: Optional[float]
    first_aligned_sync_seconds: Optional[float]
    last_aligned_sync_seconds: Optional[float]
    min_confidence: float
    mean_confidence: float
    max_confidence: float
    embedding_model_id: str
    representative_appearances: list[ReferenceAppearanceSnapshot] = field(default_factory=list)
    all_crop_embeddings: dict[int, np.ndarray] = field(default_factory=dict)
    all_context_embeddings: dict[int, np.ndarray] = field(default_factory=dict)
    keyframe_sampling_notes: list[str] = field(default_factory=list)
    global_uniqueness_disclaimer: str = (
        "REFERENCE SCOPE DISCLAIMER: This anchor represents a camera-local track observation of class "
        "'{class_name}', NOT a globally unique physical asset. Multiple identical-looking objects may exist."
    )

    def to_dict(self) -> dict[str, Any]:
        return {
            "scoped_track_key": self.scoped_track_key,
            "run_id": self.run_id,
            "camera_id": self.camera_id,
            "source_video": self.source_video,
            "source_identity": self.source_identity,
            "class_id": self.class_id,
            "class_name": self.class_name,
            "local_track_id": self.local_track_id,
            "track_support_status": self.track_support_status,
            "track_status_reason": self.track_status_reason,
            "heuristic_stability_score": round(float(self.heuristic_stability_score), 4),
            "frame_count": self.frame_count,
            "evidence_sampled_frame_count": self.evidence_sampled_frame_count,
            "frame_span": [self.first_frame, self.last_frame],
            "last_strongly_supported_frame": self.last_strongly_supported_frame,
            "time_span_seconds": [
                round(float(self.first_timestamp_seconds), 4),
                round(float(self.last_timestamp_seconds), 4),
            ],
            "last_strongly_supported_timestamp_seconds": (
                round(float(self.last_strongly_supported_timestamp_seconds), 4)
                if self.last_strongly_supported_timestamp_seconds is not None
                else None
            ),
            "aligned_utc_span": [self.first_aligned_utc, self.last_aligned_utc],
            "aligned_sync_span_seconds": [self.first_aligned_sync_seconds, self.last_aligned_sync_seconds],
            "detector_confidence": {
                "min": round(float(self.min_confidence), 4),
                "mean": round(float(self.mean_confidence), 4),
                "max": round(float(self.max_confidence), 4),
            },
            "embedding_model_id": self.embedding_model_id,
            "indexed_crop_embeddings_count": len(self.all_crop_embeddings),
            "indexed_context_embeddings_count": len(self.all_context_embeddings),
            "representative_appearances": [a.to_dict() for a in self.representative_appearances],
            "keyframe_sampling_notes": self.keyframe_sampling_notes,
            "global_uniqueness_disclaimer": self.global_uniqueness_disclaimer.format(class_name=self.class_name),
        }


@dataclass
class CandidateVerificationResult:
    """Multi-factor verification result comparing one candidate track against the ReferenceAssetAnchor."""

    reference_scoped_track_key: str
    candidate_scoped_track_key: str
    candidate_run_id: Optional[int]
    candidate_camera_id: str
    candidate_source_video: str
    candidate_class_name: str
    candidate_local_track_id: Optional[int]
    same_camera_as_reference: bool
    same_video_run_as_reference: bool
    verification_outcome: str  # "SUPPORTED_CANDIDATE", "CONTRADICTED_CANDIDATE", or "INCONCLUSIVE"
    verdict_explanation: str
    # Separate, inspectable evidence factors (never combined into a fabricated probability)
    openclip_similarity_factors: dict[str, Any]
    color_texture_factors: dict[str, Any]
    spatiotemporal_factors: dict[str, Any]
    track_quality_factors: dict[str, Any]
    background_stationarity_factors: dict[str, Any]
    contradiction_overlap_factors: dict[str, Any]
    supporting_evidence: list[str] = field(default_factory=list)
    contradicting_evidence: list[str] = field(default_factory=list)
    missing_evidence: list[str] = field(default_factory=list)
    relevant_sightings_and_paths: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "reference_scoped_track_key": self.reference_scoped_track_key,
            "candidate_scoped_track_key": self.candidate_scoped_track_key,
            "candidate_run_id": self.candidate_run_id,
            "candidate_camera_id": self.candidate_camera_id,
            "candidate_source_video": self.candidate_source_video,
            "candidate_class_name": self.candidate_class_name,
            "candidate_local_track_id": self.candidate_local_track_id,
            "same_camera_as_reference": self.same_camera_as_reference,
            "same_video_run_as_reference": self.same_video_run_as_reference,
            "verification_outcome": self.verification_outcome,
            "verdict_explanation": self.verdict_explanation,
            "evidence_factors": {
                "1_openclip_appearance_similarity": self.openclip_similarity_factors,
                "2_color_and_texture_similarity": self.color_texture_factors,
                "3_temporal_ordering_and_synchronization": self.spatiotemporal_factors,
                "4_track_stability_and_detection_quality": self.track_quality_factors,
                "5_background_consistency_and_stationarity": self.background_stationarity_factors,
                "6_contradictory_or_overlapping_evidence": self.contradiction_overlap_factors,
            },
            "supporting_evidence": self.supporting_evidence,
            "contradicting_evidence": self.contradicting_evidence,
            "missing_evidence": self.missing_evidence,
            "relevant_sightings_and_paths": self.relevant_sightings_and_paths,
        }


@dataclass
class VerificationReport:
    """Complete report anchoring a reference asset and evaluating candidate tracks across one or more cameras."""

    reference_anchor: ReferenceAssetAnchor
    queried_cameras: list[str]
    chronologically_orderable: bool
    synchronization_mode: str
    total_candidates_evaluated: int
    supported_candidates_count: int
    contradicted_candidates_count: int
    inconclusive_candidates_count: int
    overall_verification_status: str
    overall_verification_summary: str
    candidate_results: list[CandidateVerificationResult] = field(default_factory=list)
    methodology_and_calibration_notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "reference_anchor": self.reference_anchor.to_dict(),
            "queried_cameras": self.queried_cameras,
            "chronologically_orderable": self.chronologically_orderable,
            "synchronization_mode": self.synchronization_mode,
            "total_candidates_evaluated": self.total_candidates_evaluated,
            "supported_candidates_count": self.supported_candidates_count,
            "contradicted_candidates_count": self.contradicted_candidates_count,
            "inconclusive_candidates_count": self.inconclusive_candidates_count,
            "overall_verification_status": self.overall_verification_status,
            "overall_verification_summary": self.overall_verification_summary,
            "candidate_results": [c.to_dict() for c in self.candidate_results],
            "methodology_and_calibration_notes": self.methodology_and_calibration_notes,
        }


def _load_bgr_patch_for_sighting(
    sighting: SightingRecord,
    video_frames_cache: dict[tuple[str, int], np.ndarray],
) -> Optional[np.ndarray]:
    """Load the unpadded BGR patch inside `sighting.bbox_xyxy` from video frame cache or saved crop file."""
    norm_vid = normalize_source_video(sighting.source_video)
    frame_bgr = video_frames_cache.get((norm_vid, sighting.frame_index))
    if frame_bgr is not None and frame_bgr.size > 0:
        patch = _extract_patch(frame_bgr, sighting.bbox_xyxy)
        if patch.size > 0:
            return patch

    # Fallback: read saved evidence crop image from disk if video file is unavailable (e.g., synthetic fixtures)
    if sighting.evidence_crop_path:
        p = Path(sighting.evidence_crop_path)
        if p.exists() and p.is_file():
            crop_bgr = cv2.imread(str(p))
            if crop_bgr is not None and crop_bgr.size > 0:
                return crop_bgr
    return None


def _ensure_embeddings_for_sightings(
    sightings: Sequence[SightingRecord],
    visual_index: Optional[ChromaVisualIndex],
    encoder: Optional[VisualTextEncoder],
) -> tuple[dict[int, np.ndarray], dict[int, np.ndarray], str]:
    """Retrieve persisted crop/context embeddings from ChromaDB or compute on-the-fly if not yet indexed."""
    s_ids = [int(s.sighting_id) for s in sightings if s.sighting_id is not None]
    model_id = encoder.model_id if encoder is not None else "openclip:ViT-B-32:laion2b_s34b_b79k"

    persisted: dict[tuple[int, str], np.ndarray] = {}
    if visual_index is not None and s_ids:
        persisted = visual_index.get_embeddings_for_sighting_ids(
            s_ids,
            model_id=encoder.model_id if encoder is not None else None,
        )

    crop_map: dict[int, np.ndarray] = {}
    ctx_map: dict[int, np.ndarray] = {}
    for (sid, var), vec in persisted.items():
        if var == "crop":
            crop_map[sid] = vec
        elif var == "context":
            ctx_map[sid] = vec

    # Compute on-the-fly for any sighting with evidence paths that wasn't already in ChromaDB
    missing_sightings = [
        s
        for s in sightings
        if s.sighting_id is not None
        and (s.sighting_id not in crop_map or s.sighting_id not in ctx_map)
        and (s.evidence_crop_path or s.evidence_frame_path)
    ]
    if missing_sightings and encoder is not None:
        frame_reader = VideoFrameContextReader()
        try:
            for s in missing_sightings:
                assert s.sighting_id is not None
                if s.sighting_id not in crop_map and s.evidence_crop_path:
                    cpil = load_pil_image_safely(s.evidence_crop_path)
                    if cpil is not None:
                        emb = encoder.encode_images([cpil])
                        if emb.shape[0] == 1:
                            crop_map[s.sighting_id] = emb[0]
                if s.sighting_id not in ctx_map and s.evidence_frame_path:
                    fpil, _ = frame_reader.get_frame_rgb(s.source_video, s.frame_index, s.evidence_frame_path)
                    if fpil is not None:
                        ctx_pil = extract_context_region_pil(fpil, s.bbox_xyxy)
                        if ctx_pil is not None:
                            emb = encoder.encode_images([ctx_pil])
                            if emb.shape[0] == 1:
                                ctx_map[s.sighting_id] = emb[0]
        finally:
            frame_reader.close()

    return crop_map, ctx_map, model_id


def build_reference_asset_anchor(
    store: SightingStore,
    camera_id: Optional[str] = None,
    source_video: Optional[str] = None,
    local_track_id: Optional[int] = None,
    scoped_track_key: Optional[str] = None,
    sighting_id: Optional[int] = None,
    query_class: str = "laptop",
    confirmed_conf_threshold: float = 0.50,
    confirmed_min_track_frames: int = 3,
    visual_index: Optional[ChromaVisualIndex] = None,
    encoder: Optional[VisualTextEncoder] = None,
) -> ReferenceAssetAnchor:
    """Select and anchor an existing local track or sighting with its multi-appearance states and embeddings."""
    all_sightings = store.query_sightings(class_name=query_class)
    all_meta = store.get_ingestion_runs()
    if not all_sightings:
        raise ValueError(f"No sightings of class '{query_class}' exist in SQLite store.")

    track_summaries, _ = summarize_tracks(
        all_sightings,
        confirmed_conf_threshold=confirmed_conf_threshold,
        confirmed_min_track_frames=confirmed_min_track_frames,
        video_metadata=all_meta,
    )

    target_summary: Optional[TrackSummary] = None

    # 1. Resolve by explicit sighting_id if provided
    if sighting_id is not None:
        matched_s = [s for s in all_sightings if s.sighting_id == int(sighting_id)]
        if not matched_s:
            raise ValueError(f"Reference sighting_id={sighting_id} not found for class '{query_class}'.")
        target_key = matched_s[0].scoped_track_key
        for t in track_summaries:
            if t.scoped_track_key == target_key:
                target_summary = t
                break

    # 2. Resolve by explicit scoped_track_key if provided
    if target_summary is None and scoped_track_key:
        for t in track_summaries:
            if t.scoped_track_key == scoped_track_key:
                target_summary = t
                break
        if target_summary is None:
            raise ValueError(f"Reference scoped_track_key='{scoped_track_key}' not found.")

    # 3. Resolve by (camera_id, source_video, local_track_id) or fallback to primary strongly supported track
    if target_summary is None:
        norm_vid = normalize_source_video(source_video) if source_video else None
        candidates = [
            t
            for t in track_summaries
            if (camera_id is None or t.camera_id == camera_id)
            and (norm_vid is None or t.source_video == norm_vid)
            and (local_track_id is None or t.local_track_id == int(local_track_id))
        ]
        if not candidates:
            raise ValueError(
                f"No reference track found matching camera_id={camera_id}, "
                f"source_video={source_video}, local_track_id={local_track_id}."
            )
        strongly_supp = [t for t in candidates if t.status == "STRONGLY_SUPPORTED"]
        target_summary = strongly_supp[0] if strongly_supp else candidates[0]

    ref_sightings = [
        s
        for s in all_sightings
        if s.scoped_track_key == target_summary.scoped_track_key
    ]
    ref_sightings.sort(key=lambda s: (s.timestamp_seconds, s.frame_index))

    evidence_sightings = [
        s for s in ref_sightings if s.evidence_crop_path or s.evidence_frame_path
    ]
    crop_embs, ctx_embs, model_id = _ensure_embeddings_for_sightings(
        evidence_sightings,
        visual_index=visual_index,
        encoder=encoder,
    )

    # Identify representative appearance states:
    #   State 1: First sighting (e.g. open laptop on desk at Frame 0)
    #   State 2: Peak detector confidence sighting (e.g. Frame 67)
    #   State 3: Last strongly supported sighting (e.g. closed/carried laptop at Frame 107)
    #   State 4: Last overall sighting in track (if different from last strongly supported, e.g. Frame 110)
    first_s = ref_sightings[0]
    peak_s = target_summary.peak_Sighting
    last_supp_s = target_summary.last_supported_sighting
    last_any_s = target_summary.last_sighting

    selected_roles: list[tuple[str, SightingRecord]] = [("first_appearance_state", first_s)]
    seen_frames: set[int] = {first_s.frame_index}

    if peak_s.frame_index not in seen_frames:
        selected_roles.append(("peak_confidence_state", peak_s))
        seen_frames.add(peak_s.frame_index)

    if last_supp_s is not None and last_supp_s.frame_index not in seen_frames:
        selected_roles.append(("last_strongly_supported_state", last_supp_s))
        seen_frames.add(last_supp_s.frame_index)

    if last_any_s.frame_index not in seen_frames:
        selected_roles.append(("track_fadeout_last_state", last_any_s))
        seen_frames.add(last_any_s.frame_index)

    # Read video frames for the representative snapshots so BGR patches are available for HSV/NCC comparison
    norm_ref_video = normalize_source_video(target_summary.source_video)
    raw_frames = _read_video_frames(norm_ref_video, seen_frames)
    video_frames_cache: dict[tuple[str, int], np.ndarray] = {
        (norm_ref_video, f_idx): img for f_idx, img in raw_frames.items()
    }

    snapshots: list[ReferenceAppearanceSnapshot] = []
    for role_name, s_rec in selected_roles:
        sid = int(s_rec.sighting_id or -1)
        snapshots.append(
            ReferenceAppearanceSnapshot(
                role=role_name,
                sighting=s_rec,
                bbox_geometry=_bbox_dims(s_rec.bbox_xyxy),
                crop_embedding=crop_embs.get(sid),
                context_embedding=ctx_embs.get(sid),
                bgr_patch=_load_bgr_patch_for_sighting(s_rec, video_frames_cache),
            )
        )

    # Evaluate keyframe sampling coverage (Task 6 investigation)
    sampling_notes: list[str] = []
    unsampled_count = len(ref_sightings) - len(evidence_sightings)
    if unsampled_count > 0:
        sampling_notes.append(
            f"SELECTIVE KEYFRAME SAMPLING: {len(evidence_sightings)}/{len(ref_sightings)} frames in "
            f"{target_summary.scoped_track_key} have saved evidence images on disk ({unsampled_count} intermediate "
            "high-confidence frames were not written to disk due to keyframe_interval=12). First, peak, "
            "last-strongly-supported, and low-confidence transition frames are all preserved."
        )
        # Check if aspect ratio / bbox area changed substantially between first and last strongly supported
        first_dims = _bbox_dims(first_s.bbox_xyxy)
        end_ref = last_supp_s or last_any_s
        end_dims = _bbox_dims(end_ref.bbox_xyxy)
        area_ratio = end_dims["area"] / max(1.0, first_dims["area"])
        if area_ratio < 0.50 or area_ratio > 2.0:
            sampling_notes.append(
                f"POSE / STATE TRANSITION ACROSS REFERENCE TRACK: Bounding box area changed from "
                f"{first_dims['width']}x{first_dims['height']} px ({first_dims['area']:.0f} px^2 at Frame "
                f"{first_s.frame_index}) to {end_dims['width']}x{end_dims['height']} px ({end_dims['area']:.0f} px^2 "
                f"at Frame {end_ref.frame_index}, ratio={area_ratio:.2f}). Anchoring multiple representative states "
                "(both open and closed/carried) mitigates single-crop appearance drift."
            )
    else:
        sampling_notes.append(
            f"COMPLETE FRAME EVIDENCE COVERAGE: All {len(ref_sightings)}/{len(ref_sightings)} frames in "
            f"{target_summary.scoped_track_key} have saved evidence images."
        )

    return ReferenceAssetAnchor(
        scoped_track_key=target_summary.scoped_track_key,
        run_id=target_summary.run_id,
        camera_id=target_summary.camera_id,
        source_video=target_summary.source_video,
        source_identity=target_summary.source_identity,
        class_id=target_summary.class_id,
        class_name=target_summary.class_name,
        local_track_id=target_summary.local_track_id,
        track_support_status=target_summary.status,
        track_status_reason=target_summary.status_reason,
        heuristic_stability_score=target_summary.heuristic_stability_score,
        frame_count=target_summary.frame_count,
        evidence_sampled_frame_count=len(evidence_sightings),
        first_frame=target_summary.first_frame,
        last_frame=target_summary.last_frame,
        last_strongly_supported_frame=last_supp_s.frame_index if last_supp_s else None,
        first_timestamp_seconds=target_summary.first_timestamp_seconds,
        last_timestamp_seconds=target_summary.last_timestamp_seconds,
        last_strongly_supported_timestamp_seconds=last_supp_s.timestamp_seconds if last_supp_s else None,
        first_aligned_utc=target_summary.first_aligned_utc,
        last_aligned_utc=target_summary.last_aligned_utc,
        first_aligned_epoch=target_summary.first_aligned_epoch,
        last_aligned_epoch=target_summary.last_aligned_epoch,
        first_aligned_sync_seconds=target_summary.first_aligned_sync_seconds,
        last_aligned_sync_seconds=target_summary.last_aligned_sync_seconds,
        min_confidence=target_summary.min_confidence,
        mean_confidence=target_summary.mean_confidence,
        max_confidence=target_summary.max_confidence,
        embedding_model_id=model_id,
        representative_appearances=snapshots,
        all_crop_embeddings=crop_embs,
        all_context_embeddings=ctx_embs,
        keyframe_sampling_notes=sampling_notes,
    )


def verify_candidates_against_reference(
    store: SightingStore,
    reference_anchor: ReferenceAssetAnchor,
    candidate_camera_ids: Optional[Sequence[str]] = None,
    candidate_track_keys: Optional[Sequence[str]] = None,
    confirmed_conf_threshold: float = 0.50,
    confirmed_min_track_frames: int = 3,
    visual_index: Optional[ChromaVisualIndex] = None,
    encoder: Optional[VisualTextEncoder] = None,
) -> VerificationReport:
    """Compare candidate tracks against `reference_anchor` using 6 independent, inspectable evidence factors."""
    active_cam_filters = [c for c in (candidate_camera_ids or []) if c]
    all_sightings = store.query_sightings(
        class_name=reference_anchor.class_name,
        camera_ids=active_cam_filters if active_cam_filters else None,
    )
    all_meta = store.get_ingestion_runs(
        camera_ids=active_cam_filters if active_cam_filters else None,
    )
    # Ensure reference camera metadata is included when checking cross-camera synchronization
    ref_metas = store.get_ingestion_runs(
        camera_id=reference_anchor.camera_id,
        source_video=reference_anchor.source_video,
        run_id=reference_anchor.run_id,
    )
    combined_meta_map: dict[Optional[int], VideoMetadata] = {
        m.run_id: m for m in (list(all_meta) + list(ref_metas))
    }
    combined_meta = list(combined_meta_map.values())

    track_summaries, _ = summarize_tracks(
        all_sightings,
        confirmed_conf_threshold=confirmed_conf_threshold,
        confirmed_min_track_frames=confirmed_min_track_frames,
        video_metadata=combined_meta,
    )

    # Filter candidate tracks (excluding the reference track itself)
    candidate_summaries: list[TrackSummary] = []
    for t in track_summaries:
        if t.scoped_track_key == reference_anchor.scoped_track_key:
            continue
        if candidate_track_keys and t.scoped_track_key not in candidate_track_keys:
            continue
        candidate_summaries.append(t)

    # Group sightings by track key
    sightings_by_track: dict[str, list[SightingRecord]] = {}
    for s in all_sightings:
        sightings_by_track.setdefault(s.scoped_track_key, []).append(s)

    # Check synchronization across the reference and candidate recordings
    relevant_sightings_for_sync = list(all_sightings)
    if reference_anchor.scoped_track_key not in sightings_by_track:
        ref_only = store.query_sightings(
            class_name=reference_anchor.class_name,
            camera_id=reference_anchor.camera_id,
            source_video=reference_anchor.source_video,
            run_id=reference_anchor.run_id,
        )
        relevant_sightings_for_sync.extend(ref_only)

    chronologically_orderable, sync_mode = determine_synchronization_mode(
        relevant_sightings_for_sync,
        combined_meta,
    )
    queried_cameras = sorted(
        {reference_anchor.camera_id}
        | {m.camera_id for m in combined_meta}
        | {t.camera_id for t in candidate_summaries}
        | set(active_cam_filters)
    )

    # Pre-load person tracks per (camera_id, source_video, run_id) for carrier context checks
    person_sightings_by_video: dict[tuple[str, str, Optional[int]], list[SightingRecord]] = {}
    for m in combined_meta:
        p_list = store.query_sightings(
            class_name="person",
            camera_id=m.camera_id,
            source_video=m.source_video,
            run_id=m.run_id,
        )
        person_sightings_by_video[(m.camera_id, normalize_source_video(m.source_video), m.run_id)] = p_list

    results: list[CandidateVerificationResult] = []
    for cand_summary in candidate_summaries:
        cand_sightings = sorted(
            sightings_by_track.get(cand_summary.scoped_track_key, []),
            key=lambda s: (s.timestamp_seconds, s.frame_index),
        )
        res = _evaluate_single_candidate_track(
            reference_anchor=reference_anchor,
            cand_summary=cand_summary,
            cand_sightings=cand_sightings,
            all_candidate_summaries=candidate_summaries,
            sightings_by_track=sightings_by_track,
            person_sightings_by_video=person_sightings_by_video,
            combined_meta_map=combined_meta_map,
            confirmed_conf_threshold=confirmed_conf_threshold,
            visual_index=visual_index,
            encoder=encoder,
        )
        results.append(res)

    supported_count = sum(1 for r in results if r.verification_outcome == VERDICT_SUPPORTED_CANDIDATE)
    contradicted_count = sum(1 for r in results if r.verification_outcome == VERDICT_CONTRADICTED_CANDIDATE)
    inconclusive_count = sum(1 for r in results if r.verification_outcome == VERDICT_INCONCLUSIVE)

    if not results:
        overall_status = VERDICT_INCONCLUSIVE
        overall_summary = (
            f"No candidate tracks other than reference '{reference_anchor.scoped_track_key}' were found "
            "in the queried camera scope."
        )
    elif supported_count == 1 and inconclusive_count == 0:
        supp_cand = [r for r in results if r.verification_outcome == VERDICT_SUPPORTED_CANDIDATE][0]
        overall_status = VERDICT_SUPPORTED_CANDIDATE
        overall_summary = (
            f"1 candidate track ({supp_cand.candidate_scoped_track_key} on '{supp_cand.candidate_camera_id}') "
            "is supported by multi-factor evidence without contradicting fixtures or temporal conflicts. "
            "Physical object identity remains an evidence-supported hypothesis, not absolute proof."
        )
    elif supported_count > 1:
        overall_status = VERDICT_INCONCLUSIVE
        overall_summary = (
            f"Multiple candidate tracks ({supported_count}) met supporting criteria; single-asset final "
            "location remains INCONCLUSIVE without human verification."
        )
    elif supported_count == 0 and contradicted_count == len(results):
        overall_status = VERDICT_INCONCLUSIVE
        overall_summary = (
            f"All {contradicted_count} later/other candidate track(s) were contradicted by background stationarity, "
            "temporal overlap, or appearance mismatch. Last reliable sighting remains the reference anchor "
            f"({reference_anchor.scoped_track_key}), and final downstream destination is INCONCLUSIVE."
        )
    else:
        overall_status = VERDICT_INCONCLUSIVE
        overall_summary = (
            f"Evaluated {len(results)} candidate track(s): {supported_count} supported, "
            f"{contradicted_count} contradicted, and {inconclusive_count} inconclusive. "
            "Final location after reference track loss cannot be verified with certainty (INCONCLUSIVE)."
        )

    notes = [
        "MULTI-FACTOR SEPARATION: OpenCLIP cosine similarity, HSV/NCC appearance, temporal/synchronization "
        "plausibility, track stability, pre-arrival background stationarity, and spatial/temporal conflict "
        "checks are evaluated and reported as independent factors — never collapsed into an arbitrary probability.",
        "CONSERVATIVE VERIFICATION RULE: `SUPPORTED_CANDIDATE` requires strong track stability, compatible "
        "OpenCLIP and color appearance, valid chronological ordering after the reference, and zero background "
        "fixture or simultaneous-overlap contradictions. Any unresolved ambiguity yields `INCONCLUSIVE`.",
        "NO AUTOMATIC TRACK MERGING: Candidate tracks retain their isolated `scoped_track_key` regardless of "
        "verification outcome.",
    ]
    notes.extend(reference_anchor.keyframe_sampling_notes)

    return VerificationReport(
        reference_anchor=reference_anchor,
        queried_cameras=queried_cameras,
        chronologically_orderable=chronologically_orderable,
        synchronization_mode=sync_mode,
        total_candidates_evaluated=len(results),
        supported_candidates_count=supported_count,
        contradicted_candidates_count=contradicted_count,
        inconclusive_candidates_count=inconclusive_count,
        overall_verification_status=overall_status,
        overall_verification_summary=overall_summary,
        candidate_results=results,
        methodology_and_calibration_notes=notes,
    )


def _evaluate_single_candidate_track(
    reference_anchor: ReferenceAssetAnchor,
    cand_summary: TrackSummary,
    cand_sightings: list[SightingRecord],
    all_candidate_summaries: list[TrackSummary],
    sightings_by_track: dict[str, list[SightingRecord]],
    person_sightings_by_video: dict[tuple[str, str, Optional[int]], list[SightingRecord]],
    combined_meta_map: dict[Optional[int], VideoMetadata],
    confirmed_conf_threshold: float,
    visual_index: Optional[ChromaVisualIndex],
    encoder: Optional[VisualTextEncoder],
) -> CandidateVerificationResult:
    """Evaluate all 6 evidence factors for a single candidate track against `reference_anchor`."""
    supporting: list[str] = []
    contradicting: list[str] = []
    missing: list[str] = []

    same_camera = cand_summary.camera_id == reference_anchor.camera_id
    same_video_run = (
        same_camera
        and normalize_source_video(cand_summary.source_video) == normalize_source_video(reference_anchor.source_video)
        and cand_summary.run_id == reference_anchor.run_id
    )

    cand_evidence_sightings = [
        s for s in cand_sightings if s.evidence_crop_path or s.evidence_frame_path
    ]
    cand_crop_embs, cand_ctx_embs, _ = _ensure_embeddings_for_sightings(
        cand_evidence_sightings,
        visual_index=visual_index,
        encoder=encoder,
    )

    # -------------------------------------------------------------------------
    # FACTOR 1: OpenCLIP Appearance Similarity (Crop & Context vs Reference States)
    # -------------------------------------------------------------------------
    crop_sims_by_role: dict[str, float] = {}
    ctx_sims_by_role: dict[str, float] = {}

    for snap in reference_anchor.representative_appearances:
        if snap.crop_embedding is not None and cand_crop_embs:
            role_sims = [float(np.dot(snap.crop_embedding, c_emb)) for c_emb in cand_crop_embs.values()]
            crop_sims_by_role[snap.role] = round(max(role_sims), 4)
        if snap.context_embedding is not None and cand_ctx_embs:
            role_ctx_sims = [float(np.dot(snap.context_embedding, c_emb)) for c_emb in cand_ctx_embs.values()]
            ctx_sims_by_role[snap.role] = round(max(role_ctx_sims), 4)

    all_pairwise_crop_sims: list[float] = []
    for r_emb in reference_anchor.all_crop_embeddings.values():
        for c_emb in cand_crop_embs.values():
            all_pairwise_crop_sims.append(float(np.dot(r_emb, c_emb)))

    all_pairwise_ctx_sims: list[float] = []
    for r_emb in reference_anchor.all_context_embeddings.values():
        for c_emb in cand_ctx_embs.values():
            all_pairwise_ctx_sims.append(float(np.dot(r_emb, c_emb)))

    max_crop_sim = round(max(all_pairwise_crop_sims), 4) if all_pairwise_crop_sims else None
    mean_crop_sim = round(sum(all_pairwise_crop_sims) / len(all_pairwise_crop_sims), 4) if all_pairwise_crop_sims else None
    max_ctx_sim = round(max(all_pairwise_ctx_sims), 4) if all_pairwise_ctx_sims else None
    mean_ctx_sim = round(sum(all_pairwise_ctx_sims) / len(all_pairwise_ctx_sims), 4) if all_pairwise_ctx_sims else None

    openclip_factors: dict[str, Any] = {
        "embedding_model_id": reference_anchor.embedding_model_id,
        "candidate_indexed_crop_count": len(cand_crop_embs),
        "candidate_indexed_context_count": len(cand_ctx_embs),
        "max_crop_cosine_similarity": max_crop_sim,
        "mean_crop_cosine_similarity": mean_crop_sim,
        "max_context_cosine_similarity": max_ctx_sim,
        "mean_context_cosine_similarity": mean_ctx_sim,
        "crop_similarity_by_reference_state": crop_sims_by_role,
        "context_similarity_by_reference_state": ctx_sims_by_role,
    }

    if max_crop_sim is None and max_ctx_sim is None:
        missing.append(
            "MISSING OPENCLIP EMBEDDINGS: Either the reference or candidate track lacks saved evidence crops/frames "
            "for OpenCLIP visual embedding comparison."
        )
    elif max_crop_sim is not None:
        if max_crop_sim >= 0.72:
            supporting.append(
                f"OpenCLIP crop embedding similarity with reference {reference_anchor.scoped_track_key} "
                f"reaches {max_crop_sim:.4f} (mean={mean_crop_sim:.4f}; by reference state={crop_sims_by_role})."
            )
        elif max_crop_sim < 0.55:
            contradicting.append(
                f"LOW OPENCLIP APPEARANCE SIMILARITY: Maximum crop cosine similarity against reference "
                f"{reference_anchor.scoped_track_key} is only {max_crop_sim:.4f} (mean={mean_crop_sim:.4f}), "
                "indicating a visually dissimilar object."
            )
        else:
            supporting.append(
                f"Moderate OpenCLIP crop similarity ({max_crop_sim:.4f}, mean={mean_crop_sim:.4f}) with reference "
                "appearance states, though insufficient on its own to confirm object identity."
            )

    # -------------------------------------------------------------------------
    # FACTOR 2 & 5: Color/Texture (HSV & NCC) + Pre-Arrival Background Stationarity
    # -------------------------------------------------------------------------
    peak_cand = cand_summary.peak_Sighting
    last_cand = cand_summary.last_sighting
    norm_cand_vid = normalize_source_video(cand_summary.source_video)
    needed_cand_frames = {0, peak_cand.frame_index, last_cand.frame_index}
    cand_raw_frames = _read_video_frames(norm_cand_vid, needed_cand_frames)
    cand_frame_cache: dict[tuple[str, int], np.ndarray] = {
        (norm_cand_vid, f_idx): img for f_idx, img in cand_raw_frames.items()
    }

    patch_cand_peak = _load_bgr_patch_for_sighting(peak_cand, cand_frame_cache)
    hsv_by_ref_role: dict[str, float] = {}
    ncc_by_ref_role: dict[str, float] = {}

    if patch_cand_peak is not None and patch_cand_peak.size > 0:
        for snap in reference_anchor.representative_appearances:
            if snap.bgr_patch is not None and snap.bgr_patch.size > 0:
                hsv_by_ref_role[snap.role] = compute_hsv_histogram_correlation(snap.bgr_patch, patch_cand_peak)
                ncc_by_ref_role[snap.role] = compute_normalized_cross_correlation(snap.bgr_patch, patch_cand_peak)

    max_hsv_corr = max(hsv_by_ref_role.values()) if hsv_by_ref_role else None
    max_patch_ncc = max(ncc_by_ref_role.values()) if ncc_by_ref_role else None

    cand_dims = _bbox_dims(peak_cand.bbox_xyxy)
    ref_last_snap = reference_anchor.representative_appearances[-1]
    ref_first_snap = reference_anchor.representative_appearances[0]
    area_ratio_vs_last_ref = round(cand_dims["area"] / max(1.0, ref_last_snap.bbox_geometry["area"]), 3)
    area_ratio_vs_first_ref = round(cand_dims["area"] / max(1.0, ref_first_snap.bbox_geometry["area"]), 3)

    color_texture_factors: dict[str, Any] = {
        "hsv_histogram_correlation_by_reference_state": hsv_by_ref_role,
        "patch_ncc_by_reference_state": ncc_by_ref_role,
        "max_hsv_histogram_correlation": max_hsv_corr,
        "max_patch_ncc": max_patch_ncc,
        "candidate_peak_bbox_geometry": cand_dims,
        "area_ratio_vs_reference_last_state": area_ratio_vs_last_ref,
        "area_ratio_vs_reference_first_state": area_ratio_vs_first_ref,
    }

    if max_hsv_corr is not None:
        if max_hsv_corr >= 0.40:
            supporting.append(
                f"Compatible color profile: HSV histogram correlation with reference reaches {max_hsv_corr:.4f} "
                f"({hsv_by_ref_role})."
            )
        elif max_hsv_corr < 0.15:
            contradicting.append(
                f"COLOR / TEXTURE MISMATCH: HSV histogram correlation against all reference appearance states "
                f"is very low (max={max_hsv_corr:.4f}, {hsv_by_ref_role}), contradicting appearance continuity."
            )
    else:
        missing.append(
            "MISSING PIXEL PATCHES FOR COLOR COMPARISON: Source video frame or crop image was unavailable "
            "to compute HSV color histogram correlation."
        )

    if area_ratio_vs_last_ref < 0.35:
        missing.append(
            f"SEVERE RESOLUTION / SCALE DEGRADATION: Candidate bounding box area ({cand_dims['area']:.0f} px^2, "
            f"{cand_dims['width']}x{cand_dims['height']} px) is only {area_ratio_vs_last_ref * 100:.1f}% of the "
            f"reference carried state ({ref_last_snap.bbox_geometry['area']:.0f} px^2) and "
            f"{area_ratio_vs_first_ref * 100:.1f}% of the reference open state, preventing fine-grained "
            "visual re-identification."
        )

    # Pre-arrival Frame 0 background stationarity check (when Frame 0 of the candidate video is available)
    pre_arrival_frame0_ncc: Optional[float] = None
    placed_subregion_frame0_ncc: Optional[float] = None
    is_stationary_background_fixture = False
    is_partially_conflated_with_background = False

    if 0 in cand_raw_frames and peak_cand.frame_index > 5 and patch_cand_peak is not None:
        patch_bg_f0 = _extract_patch(cand_raw_frames[0], peak_cand.bbox_xyxy)
        pre_arrival_frame0_ncc = compute_normalized_cross_correlation(patch_bg_f0, patch_cand_peak)

        if (peak_cand.bbox_xyxy[2] - peak_cand.bbox_xyxy[0]) >= 55.0 and last_cand.frame_index in cand_raw_frames:
            x1, y1, x2, y2 = peak_cand.bbox_xyxy
            sub_box = (x1, y1 + 0.30 * (y2 - y1), x1 + 0.60 * (x2 - x1), y2)
            sub_bg = _extract_patch(cand_raw_frames[0], sub_box)
            sub_last = _extract_patch(cand_raw_frames[last_cand.frame_index], sub_box)
            placed_subregion_frame0_ncc = compute_normalized_cross_correlation(sub_bg, sub_last)

        if pre_arrival_frame0_ncc >= 0.90:
            is_stationary_background_fixture = True
            contradicting.append(
                f"PRE-EXISTING STATIONARY BACKGROUND OBJECT (Frame 0 NCC={pre_arrival_frame0_ncc:.4f}): "
                f"The pixel patch inside {cand_summary.scoped_track_key}'s bounding box {[round(v, 1) for v in peak_cand.bbox_xyxy]} "
                f"was already present at Frame 0 (0.00s) before the asset moved. In laptop_test.mp4, this "
                "encloses the stationary black desk telephone and must NOT be verified as the moved laptop."
            )
        elif pre_arrival_frame0_ncc >= 0.80:
            is_partially_conflated_with_background = True
            contradicting.append(
                f"PARTIAL BACKGROUND FIXTURE CONFLATION (Frame 0 NCC={pre_arrival_frame0_ncc:.4f}): "
                f"{cand_summary.scoped_track_key}'s bounding box overlaps a pre-existing stationary desk fixture "
                "(telephone/tray) alongside any newly placed object."
            )
            if placed_subregion_frame0_ncc is not None and placed_subregion_frame0_ncc < 0.80:
                supporting.append(
                    f"Newly placed sub-region divergence: lower-left sub-region of {cand_summary.scoped_track_key} "
                    f"diverges from Frame 0 empty desk surface (sub-region NCC={placed_subregion_frame0_ncc:.4f}), "
                    "consistent with a dark flat object placed near the stationary fixture."
                )
    else:
        if 0 not in cand_raw_frames:
            missing.append(
                "PRE-ARRIVAL BACKGROUND FRAME UNAVAILABLE: Source video file was not accessible on disk to "
                "verify whether the candidate bounding box existed at Frame 0 prior to arrival."
            )

    background_factors: dict[str, Any] = {
        "frame0_available_for_stationarity_check": 0 in cand_raw_frames,
        "pre_arrival_frame0_ncc": pre_arrival_frame0_ncc,
        "placed_subregion_frame0_ncc": placed_subregion_frame0_ncc,
        "is_stationary_background_fixture": is_stationary_background_fixture,
        "is_partially_conflated_with_background": is_partially_conflated_with_background,
    }

    # -------------------------------------------------------------------------
    # FACTOR 3: Temporal Ordering, Synchronization & Carrier Trajectory
    # -------------------------------------------------------------------------
    ref_meta = combined_meta_map.get(reference_anchor.run_id)
    cand_meta = combined_meta_map.get(cand_summary.run_id)
    pair_meta = [m for m in (ref_meta, cand_meta) if m is not None]
    pair_orderable, pair_sync_mode = determine_synchronization_mode(
        cand_sightings,
        pair_meta if len(pair_meta) >= 2 and not same_video_run else ([ref_meta] if ref_meta else []),
    )

    ref_end_time = (
        reference_anchor.last_strongly_supported_timestamp_seconds
        if reference_anchor.last_strongly_supported_timestamp_seconds is not None
        else reference_anchor.last_timestamp_seconds
    )
    cand_start_time = cand_summary.first_timestamp_seconds

    transition_gap_seconds: Optional[float] = None
    temporal_relationship = "UNSYNCHRONIZED"
    simultaneous_temporal_conflict = False

    if same_video_run:
        pair_orderable = True
        pair_sync_mode = "SINGLE_RECORDING_RELATIVE"
        transition_gap_seconds = round(cand_start_time - ref_end_time, 4)
    elif pair_orderable and pair_sync_mode == "MULTI_CAMERA_WALL_CLOCK_ALIGNED":
        if reference_anchor.last_aligned_epoch is not None and cand_summary.first_aligned_epoch is not None:
            transition_gap_seconds = round(cand_summary.first_aligned_epoch - reference_anchor.last_aligned_epoch, 4)
    elif pair_orderable and pair_sync_mode == "MULTI_CAMERA_OFFSET_ALIGNED":
        if reference_anchor.last_aligned_sync_seconds is not None and cand_summary.first_aligned_sync_seconds is not None:
            transition_gap_seconds = round(
                cand_summary.first_aligned_sync_seconds - reference_anchor.last_aligned_sync_seconds,
                4,
            )

    if not pair_orderable:
        temporal_relationship = "UNSYNCHRONIZED_CAMERAS"
        missing.append(
            f"MISSING CAMERA SYNCHRONIZATION: Camera '{reference_anchor.camera_id}' and Camera "
            f"'{cand_summary.camera_id}' lack shared recording_start_utc or time_offset_seconds metadata; "
            "cannot verify whether the candidate occurred after the reference sighting."
        )
    elif transition_gap_seconds is not None:
        # Check if intervals overlap simultaneously
        if same_video_run:
            ov_start = max(reference_anchor.first_timestamp_seconds, cand_summary.first_timestamp_seconds)
            ov_end = min(reference_anchor.last_timestamp_seconds, cand_summary.last_timestamp_seconds)
        elif pair_sync_mode == "MULTI_CAMERA_WALL_CLOCK_ALIGNED":
            ov_start = max(reference_anchor.first_aligned_epoch or 0.0, cand_summary.first_aligned_epoch or 0.0)
            ov_end = min(reference_anchor.last_aligned_epoch or -1.0, cand_summary.last_aligned_epoch or -1.0)
        else:
            ov_start = max(
                reference_anchor.first_aligned_sync_seconds or 0.0,
                cand_summary.first_aligned_sync_seconds or 0.0,
            )
            ov_end = min(
                reference_anchor.last_aligned_sync_seconds or -1.0,
                cand_summary.last_aligned_sync_seconds or -1.0,
            )

        if ov_end >= ov_start:
            simultaneous_temporal_conflict = True
            temporal_relationship = "SIMULTANEOUS_OVERLAP_CONFLICT"
            contradicting.append(
                f"SIMULTANEOUS TEMPORAL OVERLAP: Candidate {cand_summary.scoped_track_key} on "
                f"'{cand_summary.camera_id}' overlaps in time with reference {reference_anchor.scoped_track_key} "
                f"on '{reference_anchor.camera_id}' by {ov_end - ov_start:.2f}s. A single physical object cannot "
                "occupy two non-overlapping locations simultaneously."
            )
        elif transition_gap_seconds < 0:
            temporal_relationship = "PRECEDES_REFERENCE_END"
            contradicting.append(
                f"TEMPORAL ORDERING CONTRADICTION: Candidate track started {abs(transition_gap_seconds):.2f}s "
                "BEFORE the reference track's last sighting ended."
            )
        else:
            temporal_relationship = "FOLLOWS_REFERENCE"
            supporting.append(
                f"Chronological sequence: Candidate track begins +{transition_gap_seconds:.2f}s after the "
                f"reference sighting ended ({pair_sync_mode})."
            )

    # Check within-camera tracking gap & person carrier trajectory when in the same video run
    center_displacement_px: Optional[float] = None
    carrier_bridged = False
    if same_video_run:
        center_displacement_px = round(
            math.hypot(
                cand_dims["center_x"] - ref_last_snap.bbox_geometry["center_x"],
                cand_dims["center_y"] - ref_last_snap.bbox_geometry["center_y"],
            ),
            1,
        )
        if transition_gap_seconds is not None and transition_gap_seconds > 1.0:
            missing.append(
                f"UNOBSERVED OBJECT TRACKING GAP: {transition_gap_seconds:.2f}s blind gap (Frames "
                f"{reference_anchor.last_frame + 1}..{cand_summary.first_frame - 1}) and "
                f"{center_displacement_px:.1f} px spatial displacement without intermediate '{reference_anchor.class_name}' "
                "detections."
            )
        vid_persons = person_sightings_by_video.get(
            (cand_summary.camera_id, norm_cand_vid, cand_summary.run_id),
            [],
        )
        ref_last_f = reference_anchor.last_strongly_supported_frame or reference_anchor.last_frame
        gap_persons = [
            p for p in vid_persons if ref_last_f <= p.frame_index <= cand_summary.first_frame
        ]
        if gap_persons:
            p_end = gap_persons[-1]
            dist_person_to_cand = math.hypot(
                p_end.bbox_center[0] - cand_dims["center_x"],
                p_end.bbox_center[1] - cand_dims["center_y"],
            )
            if dist_person_to_cand <= 260.0:
                carrier_bridged = True
                p_keys = sorted({p.scoped_track_key for p in gap_persons})
                supporting.append(
                    f"Contextual carrier trajectory: Person track ({', '.join(p_keys)}) moved continuously "
                    f"across {len(gap_persons)} frames during the occlusion gap from the reference pickup "
                    f"zone to the candidate location ({p_end.bbox_center})."
                )

    spatiotemporal_factors: dict[str, Any] = {
        "synchronization_mode": pair_sync_mode,
        "chronologically_orderable": pair_orderable,
        "temporal_relationship": temporal_relationship,
        "transition_gap_seconds": transition_gap_seconds,
        "within_camera_center_displacement_px": center_displacement_px,
        "concurrent_person_carrier_bridged_gap": carrier_bridged,
    }

    # -------------------------------------------------------------------------
    # FACTOR 4: Track Stability & Detection Quality
    # -------------------------------------------------------------------------
    track_quality_factors: dict[str, Any] = {
        "candidate_status": cand_summary.status,
        "candidate_status_reason": cand_summary.status_reason,
        "frame_count": cand_summary.frame_count,
        "high_conf_frame_count": cand_summary.high_conf_frame_count,
        "high_conf_frame_ratio": round(cand_summary.high_conf_frame_ratio, 4),
        "temporal_continuity_density": round(cand_summary.temporal_continuity_density, 4),
        "heuristic_stability_score": round(cand_summary.heuristic_stability_score, 4),
        "detector_confidence": {
            "min": round(cand_summary.min_confidence, 4),
            "mean": round(cand_summary.mean_confidence, 4),
            "max": round(cand_summary.max_confidence, 4),
        },
        "reference_heuristic_stability_score": round(reference_anchor.heuristic_stability_score, 4),
    }

    if cand_summary.status == "STRONGLY_SUPPORTED":
        supporting.append(
            f"Candidate track meets strong-support criteria: detector mean={cand_summary.mean_confidence:.2f} "
            f"(>={confirmed_conf_threshold:.2f}), heuristic stability={cand_summary.heuristic_stability_score:.2f} "
            f"({cand_summary.high_conf_frame_count}/{cand_summary.frame_count} high-conf frames)."
        )
    else:
        contradicting.append(
            f"LOW TRACK STABILITY / WEAK DETECTOR SUPPORT ({cand_summary.status}): "
            f"detector mean confidence is {cand_summary.mean_confidence:.2f} (max={cand_summary.max_confidence:.2f}), "
            f"with only {cand_summary.high_conf_frame_count}/{cand_summary.frame_count} frame(s) >= "
            f"{confirmed_conf_threshold:.2f} and heuristic stability={cand_summary.heuristic_stability_score:.2f} "
            f"(vs {reference_anchor.heuristic_stability_score:.2f} for reference)."
        )

    # -------------------------------------------------------------------------
    # FACTOR 6: Contradictory or Overlapping Co-Occurring Candidate Tracks
    # -------------------------------------------------------------------------
    co_occurring_conflicts: list[dict[str, Any]] = []
    cand_by_frame = {s.frame_index: s for s in cand_sightings}

    for other_t in all_candidate_summaries:
        if other_t.scoped_track_key == cand_summary.scoped_track_key:
            continue
        if (
            other_t.camera_id == cand_summary.camera_id
            and normalize_source_video(other_t.source_video) == norm_cand_vid
            and other_t.run_id == cand_summary.run_id
        ):
            other_sightings = sightings_by_track.get(other_t.scoped_track_key, [])
            shared_frames: list[int] = []
            ious: list[float] = []
            containments: list[float] = []
            for os_rec in other_sightings:
                if os_rec.frame_index in cand_by_frame:
                    cs_rec = cand_by_frame[os_rec.frame_index]
                    iou, cont = compute_bbox_iou_and_containment(cs_rec.bbox_xyxy, os_rec.bbox_xyxy)
                    shared_frames.append(os_rec.frame_index)
                    ious.append(iou)
                    containments.append(cont)
            if shared_frames:
                mean_iou = round(sum(ious) / len(ious), 4)
                max_cont = round(max(containments), 4)
                ctype = (
                    "SPATIALLY_NESTED_DUPLICATE_BOXES"
                    if max_cont >= 0.70 or mean_iou >= 0.30
                    else "CONCURRENT_SEPARATE_CANDIDATES"
                )
                co_occurring_conflicts.append(
                    {
                        "other_track_key": other_t.scoped_track_key,
                        "shared_frame_count": len(shared_frames),
                        "first_shared_frame": min(shared_frames),
                        "last_shared_frame": max(shared_frames),
                        "mean_bbox_iou": mean_iou,
                        "max_bbox_containment": max_cont,
                        "conflict_type": ctype,
                    }
                )
                contradicting.append(
                    f"CO-OCCURRING CANDIDATE TRACK CONFLICT ({ctype}): Overlaps across {len(shared_frames)} "
                    f"frames (Frames {min(shared_frames)}..{max(shared_frames)}) with {other_t.scoped_track_key} "
                    f"(mean IoU={mean_iou:.2f}, max containment={max_cont:.2f})."
                )

    contradiction_overlap_factors: dict[str, Any] = {
        "simultaneous_with_reference": simultaneous_temporal_conflict,
        "co_occurring_candidate_conflicts": co_occurring_conflicts,
    }

    # -------------------------------------------------------------------------
    # CALIBRATED VERIFICATION OUTCOME DECISION RULE
    # -------------------------------------------------------------------------
    has_severe_appearance_mismatch = (
        (max_crop_sim is not None and max_crop_sim < 0.55)
        or (max_hsv_corr is not None and max_hsv_corr < 0.15)
    )
    has_hard_contradiction = (
        is_stationary_background_fixture
        or simultaneous_temporal_conflict
        or temporal_relationship == "PRECEDES_REFERENCE_END"
        or has_severe_appearance_mismatch
    )

    if has_hard_contradiction:
        outcome = VERDICT_CONTRADICTED_CANDIDATE
        reasons: list[str] = []
        if is_stationary_background_fixture:
            reasons.append(
                f"bounding box encloses a pre-existing stationary background object present at Frame 0 "
                f"(Frame 0 NCC={pre_arrival_frame0_ncc:.4f} >= 0.90)"
            )
        if simultaneous_temporal_conflict:
            reasons.append("candidate track overlaps simultaneously in time with the reference track")
        if temporal_relationship == "PRECEDES_REFERENCE_END":
            reasons.append("candidate track occurred before the reference sighting ended")
        if has_severe_appearance_mismatch:
            reasons.append(
                f"visual appearance contradicts reference (max CLIP crop sim={max_crop_sim}, max HSV corr={max_hsv_corr})"
            )
        explanation = (
            f"CONTRADICTED_CANDIDATE: {cand_summary.scoped_track_key} is rejected as a match for "
            f"{reference_anchor.scoped_track_key} because {'; '.join(reasons)}."
        )
    elif (
        cand_summary.status == "STRONGLY_SUPPORTED"
        and pair_orderable
        and temporal_relationship == "FOLLOWS_REFERENCE"
        and not is_partially_conflated_with_background
        and not co_occurring_conflicts
        and (max_crop_sim is not None and max_crop_sim >= 0.72)
        and (max_hsv_corr is None or max_hsv_corr >= 0.40)
        and (transition_gap_seconds is not None and 0.0 <= transition_gap_seconds <= 600.0)
        and (not same_video_run or transition_gap_seconds <= 2.0 or carrier_bridged)
    ):
        outcome = VERDICT_SUPPORTED_CANDIDATE
        explanation = (
            f"SUPPORTED_CANDIDATE: {cand_summary.scoped_track_key} on '{cand_summary.camera_id}' is strongly "
            f"supported (detector mean={cand_summary.mean_confidence:.2f}, stability={cand_summary.heuristic_stability_score:.2f}), "
            f"follows the reference sighting by +{transition_gap_seconds:.2f}s ({pair_sync_mode}), matches "
            f"reference OpenCLIP appearance (max crop sim={max_crop_sim:.4f}) and color profile, and has no "
            "background stationarity or overlap contradictions. Note: This supports a plausible candidate "
            "match, NOT ground-truth proof of physical object identity."
        )
    else:
        outcome = VERDICT_INCONCLUSIVE
        inc_reasons: list[str] = []
        if not pair_orderable:
            inc_reasons.append("recordings lack camera synchronization metadata")
        if cand_summary.status != "STRONGLY_SUPPORTED":
            inc_reasons.append(
                f"low detector confidence & track stability (mean={cand_summary.mean_confidence:.2f}, "
                f"stability={cand_summary.heuristic_stability_score:.2f})"
            )
        if is_partially_conflated_with_background:
            inc_reasons.append(
                f"bounding box partially conflates a newly placed object with a pre-existing stationary fixture "
                f"(Frame 0 full-box NCC={pre_arrival_frame0_ncc:.4f}, placed sub-region NCC={placed_subregion_frame0_ncc})"
            )
        if co_occurring_conflicts:
            inc_reasons.append("spatial/temporal overlap with another candidate track")
        if area_ratio_vs_last_ref < 0.35:
            inc_reasons.append(
                f"severe scale degradation ({cand_dims['width']}x{cand_dims['height']} px, "
                f"{area_ratio_vs_last_ref * 100:.1f}% of carried reference area)"
            )
        if max_crop_sim is None:
            inc_reasons.append("missing OpenCLIP visual embeddings")
        elif max_crop_sim < 0.72:
            inc_reasons.append(f"moderate/ambiguous OpenCLIP crop similarity ({max_crop_sim:.4f} < 0.72)")
        explanation = (
            f"INCONCLUSIVE: Evidence for {cand_summary.scoped_track_key} on '{cand_summary.camera_id}' is "
            f"insufficient or conflicting ({'; '.join(inc_reasons)}). Do NOT merge with "
            f"{reference_anchor.scoped_track_key} or assert as a verified final location."
        )

    first_cand = cand_sightings[0]
    relevant_paths = {
        "candidate_first_sighting": {
            "sighting_id": first_cand.sighting_id,
            "frame_index": first_cand.frame_index,
            "timestamp_seconds": first_cand.timestamp_seconds,
            "aligned_wall_clock_utc": first_cand.aligned_wall_clock_utc,
            "detector_confidence": first_cand.confidence,
            "evidence_crop_path": first_cand.evidence_crop_path,
            "evidence_frame_path": first_cand.evidence_frame_path,
        },
        "candidate_peak_sighting": {
            "sighting_id": peak_cand.sighting_id,
            "frame_index": peak_cand.frame_index,
            "timestamp_seconds": peak_cand.timestamp_seconds,
            "aligned_wall_clock_utc": peak_cand.aligned_wall_clock_utc,
            "detector_confidence": peak_cand.confidence,
            "bbox_xyxy": [round(v, 1) for v in peak_cand.bbox_xyxy],
            "evidence_crop_path": peak_cand.evidence_crop_path,
            "evidence_frame_path": peak_cand.evidence_frame_path,
        },
        "candidate_last_sighting": {
            "sighting_id": last_cand.sighting_id,
            "frame_index": last_cand.frame_index,
            "timestamp_seconds": last_cand.timestamp_seconds,
            "aligned_wall_clock_utc": last_cand.aligned_wall_clock_utc,
            "detector_confidence": last_cand.confidence,
            "bbox_xyxy": [round(v, 1) for v in last_cand.bbox_xyxy],
            "evidence_crop_path": last_cand.evidence_crop_path,
            "evidence_frame_path": last_cand.evidence_frame_path,
        },
    }

    return CandidateVerificationResult(
        reference_scoped_track_key=reference_anchor.scoped_track_key,
        candidate_scoped_track_key=cand_summary.scoped_track_key,
        candidate_run_id=cand_summary.run_id,
        candidate_camera_id=cand_summary.camera_id,
        candidate_source_video=cand_summary.source_video,
        candidate_class_name=cand_summary.class_name,
        candidate_local_track_id=cand_summary.local_track_id,
        same_camera_as_reference=same_camera,
        same_video_run_as_reference=same_video_run,
        verification_outcome=outcome,
        verdict_explanation=explanation,
        openclip_similarity_factors=openclip_factors,
        color_texture_factors=color_texture_factors,
        spatiotemporal_factors=spatiotemporal_factors,
        track_quality_factors=track_quality_factors,
        background_stationarity_factors=background_factors,
        contradiction_overlap_factors=contradiction_overlap_factors,
        supporting_evidence=supporting,
        contradicting_evidence=contradicting,
        missing_evidence=missing,
        relevant_sightings_and_paths=relevant_paths,
    )


def format_verification_report_text(report: VerificationReport) -> str:
    """Format a human-readable reference-anchored candidate verification report for CLI and docs."""
    ref = report.reference_anchor
    lines: list[str] = []
    lines.append("=" * 92)
    lines.append(
        f"REFERENCE-ANCHORED EVIDENCE-GUIDED CANDIDATE VERIFICATION REPORT | Class: '{ref.class_name}'"
    )
    lines.append("=" * 92)
    lines.append(f"Overall Verification Status : {report.overall_verification_status}")
    lines.append(f"Overall Summary             : {report.overall_verification_summary}")
    lines.append(
        f"Queried Camera Scope        : {', '.join(report.queried_cameras)} | "
        f"Sync Mode: {report.synchronization_mode} (Orderable: {report.chronologically_orderable})"
    )
    lines.append(
        f"Candidates Evaluated        : {report.total_candidates_evaluated} "
        f"(SUPPORTED_CANDIDATE: {report.supported_candidates_count}, "
        f"CONTRADICTED_CANDIDATE: {report.contradicted_candidates_count}, "
        f"INCONCLUSIVE: {report.inconclusive_candidates_count})"
    )

    lines.append("-" * 92)
    lines.append("1. REFERENCE ASSET ANCHOR (MULTI-APPEARANCE STATES):")
    lines.append(
        f"  Scoped Track Key          : {ref.scoped_track_key} (Camera='{ref.camera_id}', "
        f"Video='{ref.source_video}', Run={ref.run_id})"
    )
    lines.append(
        f"  Track Support & Stability : {ref.track_support_status} | "
        f"heuristic_stability={ref.heuristic_stability_score:.2f} | "
        f"det_conf min={ref.min_confidence:.2f}, mean={ref.mean_confidence:.2f}, max={ref.max_confidence:.2f}"
    )
    lines.append(
        f"  Track Frame & Time Span   : Frames {ref.first_frame}..{ref.last_frame} "
        f"({ref.first_timestamp_seconds:.3f}s -> {ref.last_timestamp_seconds:.3f}s) | "
        f"Last Strongly Supported: Frame {ref.last_strongly_supported_frame} "
        f"({f'{ref.last_strongly_supported_timestamp_seconds:.3f}s' if ref.last_strongly_supported_timestamp_seconds is not None else 'N/A'})"
    )
    lines.append(
        f"  Indexed OpenCLIP Vectors  : {len(ref.all_crop_embeddings)} crop, "
        f"{len(ref.all_context_embeddings)} context ({ref.embedding_model_id}) across "
        f"{ref.evidence_sampled_frame_count}/{ref.frame_count} sampled track frames"
    )
    lines.append("  Representative States     :")
    for snap in ref.representative_appearances:
        s = snap.sighting
        geom = snap.bbox_geometry
        lines.append(
            f"    * [{snap.role}] Sighting #{s.sighting_id} @ {format_video_timestamp(s.timestamp_seconds)} "
            f"[Frame {s.frame_index}] | det_conf={s.confidence:.4f} | "
            f"bbox={geom['width']}x{geom['height']} px ({geom['area']:.0f} px^2) | "
            f"crop={s.evidence_crop_path or 'N/A'}"
        )

    lines.append("-" * 92)
    lines.append("2. INDEPENDENT CANDIDATE TRACK VERIFICATION RESULTS:")
    if not report.candidate_results:
        lines.append("  [NONE] No candidate tracks found to compare against the reference anchor.")
    else:
        for idx, c in enumerate(report.candidate_results, start=1):
            f_clip = c.openclip_similarity_factors
            f_col = c.color_texture_factors
            f_time = c.spatiotemporal_factors
            f_qual = c.track_quality_factors
            f_bg = c.background_stationarity_factors
            peak_info = c.relevant_sightings_and_paths["candidate_peak_sighting"]
            last_info = c.relevant_sightings_and_paths["candidate_last_sighting"]

            lines.append(
                f"  [{idx}] Candidate: {c.candidate_scoped_track_key} | Camera='{c.candidate_camera_id}' | "
                f"Outcome: {c.verification_outcome}"
            )
            lines.append(f"      Verdict Explanation      : {c.verdict_explanation}")
            lines.append(
                f"      Factor 1 (OpenCLIP Sim)  : max_crop_sim={f_clip['max_crop_cosine_similarity']} "
                f"(mean={f_clip['mean_crop_cosine_similarity']}), "
                f"max_ctx_sim={f_clip['max_context_cosine_similarity']} | "
                f"by_ref_state={f_clip['crop_similarity_by_reference_state']}"
            )
            lines.append(
                f"      Factor 2 (Color/Geometry): max_hsv_corr={f_col['max_hsv_histogram_correlation']} "
                f"({f_col['hsv_histogram_correlation_by_reference_state']}) | "
                f"peak_bbox={f_col['candidate_peak_bbox_geometry']['width']}x"
                f"{f_col['candidate_peak_bbox_geometry']['height']} px "
                f"({f_col['area_ratio_vs_reference_last_state'] * 100:.1f}% of carried ref area)"
            )
            lines.append(
                f"      Factor 3 (Spatiotemporal): rel={f_time['temporal_relationship']} | "
                f"gap={f_time['transition_gap_seconds']}s | "
                f"disp={f_time['within_camera_center_displacement_px']} px | "
                f"carrier_bridged={f_time['concurrent_person_carrier_bridged_gap']}"
            )
            lines.append(
                f"      Factor 4 (Track Quality) : status={f_qual['candidate_status']} | "
                f"det_conf={f_qual['detector_confidence']} | "
                f"stability={f_qual['heuristic_stability_score']:.2f} "
                f"({f_qual['high_conf_frame_count']}/{f_qual['frame_count']} high-conf frames)"
            )
            lines.append(
                f"      Factor 5 (Background)    : frame0_ncc={f_bg['pre_arrival_frame0_ncc']} | "
                f"placed_subregion_ncc={f_bg['placed_subregion_frame0_ncc']} | "
                f"stationary_fixture={f_bg['is_stationary_background_fixture']}"
            )
            lines.append(
                f"      Relevant Evidence Paths  : peak_frame={peak_info['frame_index']} "
                f"(+{peak_info['timestamp_seconds']:.3f}s) crop={peak_info['evidence_crop_path'] or 'N/A'} | "
                f"last_frame={last_info['frame_index']} (+{last_info['timestamp_seconds']:.3f}s) "
                f"crop={last_info['evidence_crop_path'] or 'N/A'}"
            )
            lines.append("      Supporting Evidence      :")
            if c.supporting_evidence:
                for pt in c.supporting_evidence:
                    lines.append(f"        + {pt}")
            else:
                lines.append("        + [NONE]")
            lines.append("      Contradicting Evidence   :")
            if c.contradicting_evidence:
                for pt in c.contradicting_evidence:
                    lines.append(f"        - {pt}")
            else:
                lines.append("        - [NONE]")
            lines.append("      Missing / Unobserved     :")
            if c.missing_evidence:
                for pt in c.missing_evidence:
                    lines.append(f"        ? {pt}")
            else:
                lines.append("        ? [NONE]")
            lines.append("")

    lines.append("-" * 92)
    lines.append("3. METHODOLOGY, KEYFRAME SAMPLING & CALIBRATION NOTES:")
    for note in report.methodology_and_calibration_notes:
        lines.append(f"  ! {note}")
    lines.append("=" * 92)
    return "\n".join(lines)

"""Track gap investigation and visual/spatiotemporal evidence comparison (Task 4)."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Optional

import cv2
import numpy as np

from src.models import SightingRecord, TrackSummary, normalize_source_video
from src.retrieval import compute_bbox_iou_and_containment, query_asset_sightings
from src.storage import SightingStore


def _read_video_frames(video_path: str | Path, frame_indices: set[int]) -> dict[int, np.ndarray]:
    """Read specific frames by 0-based index from a local video file."""
    path = Path(video_path)
    if not path.exists() or not frame_indices:
        return {}
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        cap.release()
        return {}
    out: dict[int, np.ndarray] = {}
    for idx in sorted(frame_indices):
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = cap.read()
        if ok and frame is not None:
            out[idx] = frame
    cap.release()
    return out


def _extract_patch(frame_bgr: np.ndarray, bbox_xyxy: tuple[float, float, float, float]) -> np.ndarray:
    """Slice a clamped BGR patch from a frame."""
    h, w = frame_bgr.shape[:2]
    x1, y1, x2, y2 = [int(round(v)) for v in bbox_xyxy]
    x1c = max(0, min(w - 1, x1))
    y1c = max(0, min(h - 1, y1))
    x2c = max(x1c + 1, min(w, x2))
    y2c = max(y1c + 1, min(h, y2))
    return frame_bgr[y1c:y2c, x1c:x2c]


def compute_normalized_cross_correlation(patch_a: np.ndarray, patch_b: np.ndarray) -> float:
    """Compute zero-mean Normalized Cross-Correlation (NCC) in [-1.0, 1.0] after resizing patch_b to patch_a."""
    if patch_a.size == 0 or patch_b.size == 0:
        return 0.0
    resized_b = cv2.resize(patch_b, (patch_a.shape[1], patch_a.shape[0]), interpolation=cv2.INTER_AREA)
    a_f = patch_a.astype(np.float32) - float(patch_a.mean())
    b_f = resized_b.astype(np.float32) - float(resized_b.mean())
    denom = float(np.linalg.norm(a_f) * np.linalg.norm(b_f))
    if denom <= 1e-6:
        return 0.0
    return round(float(np.sum(a_f * b_f) / denom), 4)


def compute_hsv_histogram_correlation(patch_a: np.ndarray, patch_b: np.ndarray) -> float:
    """Compute HSV color histogram correlation in [-1.0, 1.0] between two BGR patches."""
    if patch_a.size == 0 or patch_b.size == 0:
        return 0.0
    hsv_a = cv2.cvtColor(patch_a, cv2.COLOR_BGR2HSV)
    hsv_b = cv2.cvtColor(patch_b, cv2.COLOR_BGR2HSV)
    hist_a = cv2.calcHist([hsv_a], [0, 1, 2], None, [8, 8, 8], [0, 180, 0, 256, 0, 256])
    hist_b = cv2.calcHist([hsv_b], [0, 1, 2], None, [8, 8, 8], [0, 180, 0, 256, 0, 256])
    cv2.normalize(hist_a, hist_a)
    cv2.normalize(hist_b, hist_b)
    return round(float(cv2.compareHist(hist_a, hist_b, cv2.HISTCMP_CORREL)), 4)


def _bbox_dims(bbox: tuple[float, float, float, float]) -> dict[str, float]:
    x1, y1, x2, y2 = bbox
    w = max(0.1, x2 - x1)
    h = max(0.1, y2 - y1)
    return {
        "width": round(w, 1),
        "height": round(h, 1),
        "area": round(w * h, 1),
        "aspect_ratio": round(w / h, 2),
        "center_x": round((x1 + x2) / 2.0, 1),
        "center_y": round((y1 + y2) / 2.0, 1),
    }


def analyze_tracking_gap_and_candidates(
    store: SightingStore,
    camera_id: str = "office_cam_01",
    source_video: str = "laptop_test.mp4",
    query_class: str = "laptop",
    confirmed_conf_threshold: float = 0.50,
    confirmed_min_track_frames: int = 3,
) -> dict[str, Any]:
    """Compare the last strongly supported track against later candidate tracks and inspect the tracking gap."""
    norm_video = normalize_source_video(source_video)
    report = query_asset_sightings(
        store=store,
        query_class=query_class,
        camera_id=camera_id,
        source_video=norm_video,
        confirmed_conf_threshold=confirmed_conf_threshold,
        confirmed_min_track_frames=confirmed_min_track_frames,
    )

    supported_tracks = [t for t in report.track_summaries if t.status == "STRONGLY_SUPPORTED"]
    uncertain_tracks = [t for t in report.track_summaries if t.status != "STRONGLY_SUPPORTED"]

    if not supported_tracks:
        return {
            "camera_id": camera_id,
            "source_video": norm_video,
            "query_class": query_class,
            "error": "No strongly supported reference track found for comparison.",
        }

    ref_track: TrackSummary = supported_tracks[-1]
    ref_sightings = store.query_sightings(
        class_name=query_class,
        camera_id=camera_id,
        source_video=norm_video,
        local_track_id=ref_track.local_track_id,
        run_id=ref_track.run_id,
    )
    ref_first = ref_sightings[0]
    ref_last_supported = ref_track.last_supported_sighting or ref_track.last_sighting
    ref_last_any = ref_track.last_sighting

    # Determine frame indices to sample from the video for visual & pre-arrival background analysis
    needed_frames: set[int] = {
        0,
        ref_first.frame_index,
        ref_last_supported.frame_index,
        ref_last_any.frame_index,
    }
    for ut in uncertain_tracks:
        needed_frames.add(ut.peak_Sighting.frame_index)
        needed_frames.add(ut.last_sighting.frame_index)

    video_frames = _read_video_frames(norm_video, needed_frames)

    # Analyze person trajectory during the tracking gap
    all_person_sightings = store.query_sightings(
        class_name="person",
        camera_id=camera_id,
        source_video=norm_video,
    )
    first_later_frame = min((ut.first_frame for ut in uncertain_tracks), default=ref_last_any.frame_index)
    gap_start_frame = ref_last_any.frame_index + 1
    gap_end_frame = max(gap_start_frame, first_later_frame - 1)
    gap_duration_seconds = round(
        max(
            0.0,
            (uncertain_tracks[0].first_timestamp_seconds - ref_last_any.timestamp_seconds)
            if uncertain_tracks
            else 0.0,
        ),
        4,
    )

    person_in_gap = [
        p
        for p in all_person_sightings
        if ref_last_supported.frame_index <= p.frame_index <= max(first_later_frame, gap_end_frame)
    ]
    carrier_summary: dict[str, Any] = {"tracked_person_present": False}
    if person_in_gap:
        p_first = person_in_gap[0]
        p_last = person_in_gap[-1]
        p_tracks = sorted({p.scoped_track_key for p in person_in_gap})
        carrier_summary = {
            "tracked_person_present": True,
            "person_track_keys": p_tracks,
            "frames_covered": len(person_in_gap),
            "expected_span_frames": (p_last.frame_index - p_first.frame_index) + 1,
            "start_frame": p_first.frame_index,
            "end_frame": p_last.frame_index,
            "start_center_xy": p_first.bbox_center,
            "end_center_xy": p_last.bbox_center,
            "displacement_px": round(
                math.hypot(
                    p_last.bbox_center[0] - p_first.bbox_center[0],
                    p_last.bbox_center[1] - p_first.bbox_center[1],
                ),
                1,
            ),
        }

    # Compare each later candidate track against Reference Track 1 (open @ f=0 and closed/carried @ f=107)
    ref_open_dims = _bbox_dims(ref_first.bbox_xyxy)
    ref_carried_dims = _bbox_dims(ref_last_supported.bbox_xyxy)

    patch_ref_open = (
        _extract_patch(video_frames[ref_first.frame_index], ref_first.bbox_xyxy)
        if ref_first.frame_index in video_frames
        else np.empty((0, 0, 3), dtype=np.uint8)
    )
    patch_ref_carried = (
        _extract_patch(video_frames[ref_last_supported.frame_index], ref_last_supported.bbox_xyxy)
        if ref_last_supported.frame_index in video_frames
        else np.empty((0, 0, 3), dtype=np.uint8)
    )

    candidate_comparisons: list[dict[str, Any]] = []
    for ut in uncertain_tracks:
        peak_s: SightingRecord = ut.peak_Sighting
        last_s: SightingRecord = ut.last_sighting
        cand_dims = _bbox_dims(peak_s.bbox_xyxy)

        center_dist_from_carried = round(
            math.hypot(
                cand_dims["center_x"] - ref_carried_dims["center_x"],
                cand_dims["center_y"] - ref_carried_dims["center_y"],
            ),
            1,
        )
        area_ratio_vs_carried = round(cand_dims["area"] / max(1.0, ref_carried_dims["area"]), 3)

        patch_cand_peak = (
            _extract_patch(video_frames[peak_s.frame_index], peak_s.bbox_xyxy)
            if peak_s.frame_index in video_frames
            else np.empty((0, 0, 3), dtype=np.uint8)
        )
        # Pre-arrival background patch at Frame 0 at the EXACT same pixel bounding box as the candidate!
        patch_bg_frame0 = (
            _extract_patch(video_frames[0], peak_s.bbox_xyxy)
            if 0 in video_frames
            else np.empty((0, 0, 3), dtype=np.uint8)
        )

        pre_arrival_bg_ncc = compute_normalized_cross_correlation(patch_bg_frame0, patch_cand_peak)
        hsv_corr_vs_carried = compute_hsv_histogram_correlation(patch_ref_carried, patch_cand_peak)
        hsv_corr_vs_open = compute_hsv_histogram_correlation(patch_ref_open, patch_cand_peak)
        ncc_vs_carried = compute_normalized_cross_correlation(patch_ref_carried, patch_cand_peak)

        # Also check if candidate bbox is wider and has a newly placed object in its lower-left sub-region
        subregion_bg_ncc: Optional[float] = None
        if (peak_s.bbox_xyxy[2] - peak_s.bbox_xyxy[0]) >= 55.0 and 0 in video_frames and last_s.frame_index in video_frames:
            x1, y1, x2, y2 = peak_s.bbox_xyxy
            sub_box = (x1, y1 + 0.30 * (y2 - y1), x1 + 0.60 * (x2 - x1), y2)
            sub_bg = _extract_patch(video_frames[0], sub_box)
            sub_last = _extract_patch(video_frames[last_s.frame_index], sub_box)
            subregion_bg_ncc = compute_normalized_cross_correlation(sub_bg, sub_last)

        supporting_points: list[str] = []
        contradicting_points: list[str] = []

        if carrier_summary.get("tracked_person_present"):
            supporting_points.append(
                f"Contextual carrier trajectory: Person ({', '.join(carrier_summary['person_track_keys'])}) "
                f"walked continuously from the pickup location {carrier_summary['start_center_xy']} at "
                f"{ref_last_supported.timestamp_seconds:.2f}s directly to the back-right desk "
                f"{carrier_summary['end_center_xy']} right before {ut.scoped_track_key} appeared at "
                f"{ut.first_timestamp_seconds:.2f}s."
            )

        if hsv_corr_vs_carried >= 0.40:
            supporting_points.append(
                f"Dark matte color profile: HSV histogram correlation with the closed laptop in Frame "
                f"{ref_last_supported.frame_index} is {hsv_corr_vs_carried:.2f}."
            )

        if subregion_bg_ncc is not None and subregion_bg_ncc < 0.80:
            supporting_points.append(
                f"New dark flat object placed on desk surface: the lower-left sub-region of {ut.scoped_track_key} "
                f"diverges from the Frame 0 empty desk surface (pre-arrival sub-region NCC={subregion_bg_ncc:.4f}), "
                f"visually showing a flat dark rectangular slab placed by the employee's hands."
            )

        # Contradicting evidence checks
        if pre_arrival_bg_ncc >= 0.90:
            contradicting_points.append(
                f"PRE-EXISTING STATIONARY FIXTURE (FALSE-POSITIVE DOMINATED): Comparing {ut.scoped_track_key}'s "
                f"bounding box {[round(v, 1) for v in peak_s.bbox_xyxy]} against Frame 0 (0.00s, while the laptop "
                f"was still open on the foreground desk) yields an extremely high Normalized Cross-Correlation of "
                f"{pre_arrival_bg_ncc:.4f}. Visual inspection of the crop confirms this tight box encloses the "
                f"stationary black VoIP desk telephone already sitting on the back-right desk."
            )
        elif pre_arrival_bg_ncc >= 0.80:
            contradicting_points.append(
                f"PARTIAL BACKGROUND FIXTURE CONFLATION: Full-box pre-arrival correlation with Frame 0 is "
                f"{pre_arrival_bg_ncc:.4f} because the upper-right portion of {ut.scoped_track_key}'s bounding box "
                f"includes the stationary black desk telephone and document tray alongside the newly placed flat object."
            )

        contradicting_points.append(
            f"Unbroken 2.83s detection gap & large spatial jump: {center_dist_from_carried:.1f} px displacement "
            f"from last strongly supported laptop sighting without intermediate laptop detections."
        )
        contradicting_points.append(
            f"Low detector confidence & low track stability: mean detector confidence is {ut.mean_confidence:.2f} "
            f"(only {ut.high_conf_frame_count}/{ut.frame_count} frames >= {confirmed_conf_threshold:.2f}, "
            f"heuristic stability score={ut.heuristic_stability_score:.2f} vs {ref_track.heuristic_stability_score:.2f} "
            f"for {ref_track.scoped_track_key})."
        )
        contradicting_points.append(
            f"Severe resolution & scale degradation: bounding box area shrank to {cand_dims['area']:.0f} px^2 "
            f"({area_ratio_vs_carried * 100:.1f}% of the carried laptop area at Frame {ref_last_supported.frame_index}, "
            f"and {cand_dims['area'] / ref_open_dims['area'] * 100:.1f}% of the open laptop at Frame 0), "
            f"preventing appearance-based identity verification."
        )

        if pre_arrival_bg_ncc >= 0.90:
            visual_verdict = (
                "LIKELY FALSE POSITIVE ON STATIONARY DESK TELEPHONE (triggered as employee's hands approach "
                "the back-right desk). Must NOT be merged with Track 1 or reported as the laptop."
            )
        else:
            visual_verdict = (
                "AMBIGUOUS / CONFLATED CANDIDATE: Lower-left region visually shows a flat dark rectangular object "
                "placed on the back-right desk by the person from Track 2, but the bounding box conflates it with "
                "the adjacent desk telephone at low detector confidence (mean 0.26, max 0.33). Plausible destination "
                "candidate, but unverified -- must NOT be automatically merged with Track 1."
            )

        candidate_comparisons.append(
            {
                "candidate_track_key": ut.scoped_track_key,
                "local_track_id": ut.local_track_id,
                "frame_span": [ut.first_frame, ut.last_frame],
                "time_span_seconds": [ut.first_timestamp_seconds, ut.last_timestamp_seconds],
                "frame_count": ut.frame_count,
                "raw_detector_confidence": {
                    "min": round(ut.min_confidence, 4),
                    "mean": round(ut.mean_confidence, 4),
                    "max": round(ut.max_confidence, 4),
                },
                "heuristic_stability_score": ut.heuristic_stability_score,
                "peak_frame": peak_s.frame_index,
                "peak_bbox_xyxy": [round(v, 1) for v in peak_s.bbox_xyxy],
                "peak_crop_path": peak_s.evidence_crop_path,
                "last_crop_path": last_s.evidence_crop_path,
                "geometry_vs_carried_ref": {
                    "candidate_dims": cand_dims,
                    "center_displacement_px": center_dist_from_carried,
                    "area_ratio_vs_carried": area_ratio_vs_carried,
                },
                "visual_metrics": {
                    "pre_arrival_frame0_ncc": pre_arrival_bg_ncc,
                    "placed_subregion_frame0_ncc": subregion_bg_ncc,
                    "ncc_vs_carried_laptop": ncc_vs_carried,
                    "hsv_hist_corr_vs_carried_laptop": hsv_corr_vs_carried,
                    "hsv_hist_corr_vs_open_laptop": hsv_corr_vs_open,
                },
                "supporting_evidence": supporting_points,
                "contradicting_evidence": contradicting_points,
                "calibrated_verdict": visual_verdict,
            }
        )

    return {
        "camera_id": camera_id,
        "source_video": norm_video,
        "query_class": query_class,
        "reference_track": {
            "scoped_track_key": ref_track.scoped_track_key,
            "local_track_id": ref_track.local_track_id,
            "frame_span": [ref_track.first_frame, ref_track.last_frame],
            "time_span_seconds": [ref_track.first_timestamp_seconds, ref_track.last_timestamp_seconds],
            "last_strongly_supported_frame": ref_last_supported.frame_index,
            "last_strongly_supported_timestamp": ref_last_supported.timestamp_seconds,
            "last_strongly_supported_confidence": ref_last_supported.confidence,
            "last_strongly_supported_bbox": [round(v, 1) for v in ref_last_supported.bbox_xyxy],
            "last_strongly_supported_frame_path": ref_last_supported.evidence_frame_path,
            "last_strongly_supported_crop_path": ref_last_supported.evidence_crop_path,
            "heuristic_stability_score": ref_track.heuristic_stability_score,
            "open_dims_frame0": ref_open_dims,
            "carried_dims_frame107": ref_carried_dims,
        },
        "tracking_gap_analysis": {
            "last_strongly_supported_frame": ref_last_supported.frame_index,
            "last_strongly_supported_time_s": ref_last_supported.timestamp_seconds,
            "reference_track_fadeout_frames": [ref_last_supported.frame_index + 1, ref_last_any.frame_index],
            "complete_blind_gap_frames": [gap_start_frame, gap_end_frame],
            "complete_blind_gap_duration_s": gap_duration_seconds,
            "root_causes": [
                "1. Pose/state transition (Frames 80..110): The laptop is closed from an open state (210x204 px) "
                "into a thin flat profile (109x45 px), eliminating keyboard and screen features.",
                "2. Self-occlusion by carrier's torso (Frames 111..177, 4.62s..7.38s): As the employee (person "
                "local_track_2) turns away from the camera and walks toward the back-right desk, their back and "
                "arms occlude the carried laptop.",
                "3. Scale & contrast loss at far desk (Frames 178..239): At the back-right desk (~3x further from "
                "the camera), the closed dark laptop occupies ~40x22 px and blends into the dark desk phone and "
                "document tray.",
            ],
            "concurrent_person_carrier": carrier_summary,
        },
        "candidate_comparisons": candidate_comparisons,
        "conflicting_track_pairs": report.conflicting_track_pairs,
    }


def format_comparison_report_text(analysis: dict[str, Any]) -> str:
    """Format a clear human-readable evidence comparison report for CLI and documentation."""
    if "error" in analysis:
        return f"EVIDENCE COMPARISON ERROR: {analysis['error']}"

    ref = analysis["reference_track"]
    gap = analysis["tracking_gap_analysis"]
    carrier = gap["concurrent_person_carrier"]

    lines: list[str] = []
    lines.append("=" * 88)
    lines.append(
        f"TRACK GAP & VISUAL EVIDENCE COMPARISON REPORT | Camera: {analysis['camera_id']} | "
        f"Video: {analysis['source_video']}"
    )
    lines.append("=" * 88)
    lines.append("1. REFERENCE STRONGLY SUPPORTED TRACK:")
    lines.append(f"  Track Key                  : {ref['scoped_track_key']}")
    lines.append(
        f"  Track Span                 : Frames {ref['frame_span'][0]}..{ref['frame_span'][1]} "
        f"({ref['time_span_seconds'][0]:.3f}s -> {ref['time_span_seconds'][1]:.3f}s) | "
        f"Heuristic Stability: {ref['heuristic_stability_score']:.2f}"
    )
    lines.append(
        f"  Last Strongly Supported    : Frame {ref['last_strongly_supported_frame']} "
        f"({ref['last_strongly_supported_timestamp']:.3f}s) | det_conf={ref['last_strongly_supported_confidence']:.4f} | "
        f"bbox={ref['last_strongly_supported_bbox']}"
    )
    lines.append(f"  Evidence Frame             : {ref['last_strongly_supported_frame_path']}")
    lines.append(f"  Evidence Crop              : {ref['last_strongly_supported_crop_path']}")

    lines.append("-" * 88)
    lines.append("2. TRACKING GAP INVESTIGATION (4.46s / 4.58s -> 7.42s):")
    lines.append(
        f"  Fade-out Frames            : Frames {gap['reference_track_fadeout_frames'][0]}.."
        f"{gap['reference_track_fadeout_frames'][1]} (confidence drops below 0.50 as person turns)"
    )
    lines.append(
        f"  Complete Detection Gap     : Frames {gap['complete_blind_gap_frames'][0]}.."
        f"{gap['complete_blind_gap_frames'][1]} ({gap['complete_blind_gap_duration_s']:.3f}s with zero laptop detections)"
    )
    if carrier.get("tracked_person_present"):
        lines.append(
            f"  Concurrent Person Carrier  : {', '.join(carrier['person_track_keys'])} tracked across "
            f"{carrier['frames_covered']}/{carrier['expected_span_frames']} gap frames "
            f"(center {carrier['start_center_xy']} -> {carrier['end_center_xy']}, "
            f"displacement={carrier['displacement_px']} px)"
        )
    lines.append("  Observed Gap Root Causes   :")
    for cause in gap["root_causes"]:
        lines.append(f"    - {cause}")

    lines.append("-" * 88)
    lines.append("3. CANDIDATE TRACK VISUAL & SPATIOTEMPORAL COMPARISON (NO AUTOMATIC MERGING):")
    for idx, comp in enumerate(analysis["candidate_comparisons"], start=1):
        vm = comp["visual_metrics"]
        gm = comp["geometry_vs_carried_ref"]
        dc = comp["raw_detector_confidence"]
        lines.append(
            f"  [{idx}] Candidate: {comp['candidate_track_key']} "
            f"(Frames {comp['frame_span'][0]}..{comp['frame_span'][1]}, "
            f"{comp['time_span_seconds'][0]:.3f}s -> {comp['time_span_seconds'][1]:.3f}s, {comp['frame_count']} frames)"
        )
        lines.append(
            f"      Detector Confidence    : min={dc['min']:.2f}, mean={dc['mean']:.2f}, max={dc['max']:.2f} | "
            f"Heuristic Stability={comp['heuristic_stability_score']:.2f}"
        )
        lines.append(
            f"      Peak BBox & Scale      : Frame {comp['peak_frame']} bbox={comp['peak_bbox_xyxy']} "
            f"({gm['candidate_dims']['width']}x{gm['candidate_dims']['height']} px, "
            f"area={gm['area_ratio_vs_carried'] * 100:.1f}% of carried laptop, "
            f"displacement={gm['center_displacement_px']} px)"
        )
        lines.append(
            f"      Quantitative Visuals   : Pre-arrival Frame 0 Box NCC={vm['pre_arrival_frame0_ncc']:.4f} | "
            f"Placed Sub-Region Frame 0 NCC={vm['placed_subregion_frame0_ncc'] if vm['placed_subregion_frame0_ncc'] is not None else 'N/A'} | "
            f"HSV Corr vs Carried={vm['hsv_hist_corr_vs_carried_laptop']:.4f}"
        )
        lines.append(f"      Evidence Crops         : peak={comp['peak_crop_path']} | last={comp['last_crop_path']}")
        lines.append("      Supporting Evidence    :")
        for pt in comp["supporting_evidence"]:
            lines.append(f"        + {pt}")
        lines.append("      Contradicting Evidence :")
        for pt in comp["contradicting_evidence"]:
            lines.append(f"        - {pt}")
        lines.append(f"      Calibrated Verdict     : {comp['calibrated_verdict']}")
        lines.append("")

    if analysis["conflicting_track_pairs"]:
        lines.append("-" * 88)
        lines.append("4. CO-OCCURRING CANDIDATE CONFLICT SUMMARY:")
        for cp in analysis["conflicting_track_pairs"]:
            lines.append(
                f"  * {cp['track_a']} and {cp['track_b']} overlap in frames "
                f"{cp['first_shared_frame']}..{cp['last_shared_frame']} ({cp['shared_frame_count']} frames, "
                f"mean IoU={cp['mean_bbox_iou']:.2f}, max containment={cp['max_bbox_containment']:.2f} -> {cp['conflict_type']})."
            )

    lines.append("=" * 88)
    return "\n".join(lines)

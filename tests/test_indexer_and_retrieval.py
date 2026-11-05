"""Comprehensive tests for Group 8 sighting indexing, identity isolation, conflict detection, and retrieval."""

from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path

import cv2

from src.comparison import analyze_tracking_gap_and_candidates
from src.models import (
    SightingRecord,
    VideoMetadata,
    compute_heuristic_stability_score,
    compute_timestamp_seconds,
    format_video_timestamp,
    normalize_source_video,
)
from src.retrieval import (
    build_retrieval_report,
    detect_conflicting_tracks,
    query_asset_sightings,
)
from src.storage import SightingStore
from src.visual_search import (
    ChromaVisualIndex,
    OpenCLIPEmbedder,
    extract_context_region_pil,
    load_pil_image_safely,
    make_embedding_id,
    make_model_identifier,
    normalize_embedding_vector,
)


class TestSightingModelsAndTimestamps(unittest.TestCase):
    def test_compute_timestamp_seconds_precision(self) -> None:
        self.assertEqual(compute_timestamp_seconds(0, 24.0), 0.0)
        self.assertEqual(compute_timestamp_seconds(24, 24.0), 1.0)
        self.assertEqual(compute_timestamp_seconds(107, 24.0), 4.4583)
        self.assertEqual(compute_timestamp_seconds(115, 24.0), 4.7917)

    def test_compute_timestamp_invalid_inputs(self) -> None:
        with self.assertRaises(ValueError):
            compute_timestamp_seconds(-1, 24.0)
        with self.assertRaises(ValueError):
            compute_timestamp_seconds(10, 0.0)
        with self.assertRaises(ValueError):
            compute_timestamp_seconds(10, -30.0)

    def test_normalize_source_video_equivalence(self) -> None:
        norm_a = normalize_source_video("laptop_test.mp4")
        norm_b = normalize_source_video("./laptop_test.mp4")
        norm_c = normalize_source_video(str(Path.cwd() / "laptop_test.mp4"))
        self.assertEqual(norm_a, "laptop_test.mp4")
        self.assertEqual(norm_a, norm_b)
        self.assertEqual(norm_b, norm_c)

    def test_sighting_serialization_and_isolated_scoped_track_key(self) -> None:
        record = SightingRecord(
            sighting_id=5,
            run_id=1,
            camera_id="office_cam_01",
            source_video="./laptop_test.mp4",
            frame_index=107,
            timestamp_seconds=4.4583,
            class_id=63,
            class_name="laptop",
            local_track_id=1,
            confidence=0.7621,
            bbox_xyxy=(574.38, 357.2, 683.27, 401.74),
            evidence_frame_path="outputs/evidence/frame_107.jpg",
            evidence_crop_path="outputs/evidence/crop_107.jpg",
            evidence_clip_path="outputs/annotated_videos/laptop_test.avi",
        )
        self.assertEqual(record.source_video, "laptop_test.mp4")
        self.assertEqual(
            record.scoped_track_key,
            "run1:office_cam_01:laptop_test.mp4:local_track_1",
        )
        as_dict = record.to_dict()
        restored = SightingRecord.from_dict(as_dict)
        self.assertEqual(restored.camera_id, "office_cam_01")
        self.assertEqual(restored.source_video, "laptop_test.mp4")
        self.assertEqual(restored.local_track_id, 1)
        self.assertAlmostEqual(restored.confidence, 0.7621, places=4)
        self.assertEqual(restored.bbox_xyxy, (574.38, 357.2, 683.27, 401.74))

        # Untracked detections at different frames must NOT share an isolation_key or scoped_track_key
        untracked_a = SightingRecord(
            run_id=1,
            camera_id="office_cam_01",
            source_video="laptop_test.mp4",
            frame_index=10,
            timestamp_seconds=0.4167,
            class_id=63,
            class_name="laptop",
            local_track_id=None,
            confidence=0.22,
            bbox_xyxy=(0.0, 0.0, 10.0, 10.0),
        )
        untracked_b = SightingRecord(
            run_id=1,
            camera_id="office_cam_01",
            source_video="laptop_test.mp4",
            frame_index=90,
            timestamp_seconds=3.75,
            class_id=63,
            class_name="laptop",
            local_track_id=None,
            confidence=0.25,
            bbox_xyxy=(20.0, 20.0, 40.0, 40.0),
        )
        self.assertNotEqual(untracked_a.isolation_key, untracked_b.isolation_key)
        self.assertNotEqual(untracked_a.scoped_track_key, untracked_b.scoped_track_key)
        self.assertIn("rel", format_video_timestamp(4.5))

    def test_separation_of_detector_confidence_and_heuristic_stability(self) -> None:
        # High raw confidence on 1 frame out of 20 frames -> high max_confidence, but low heuristic stability
        ratio, density, stability = compute_heuristic_stability_score(
            frame_count=20,
            first_frame=0,
            last_frame=39,
            high_conf_frame_count=1,
            min_required_frames=3,
        )
        self.assertAlmostEqual(ratio, 0.05)
        self.assertAlmostEqual(density, 0.50)
        self.assertLess(stability, 0.25)


class TestIdentityIsolationAndRetrieval(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp_dir = tempfile.TemporaryDirectory()
        self.base = Path(self.tmp_dir.name)
        self.db_path = self.base / "test_sightings.db"
        self.jsonl_path = self.base / "test_sightings.jsonl"
        self.store = SightingStore(db_path=self.db_path, jsonl_path=self.jsonl_path)

    def tearDown(self) -> None:
        self.tmp_dir.cleanup()

    def _create_run(
        self,
        camera_id: str = "office_cam_01",
        source_video: str = "laptop_test.mp4",
        replace: bool = True,
    ) -> int:
        meta = VideoMetadata(
            camera_id=camera_id,
            source_video=source_video,
            fps=24.0,
            frame_count=240,
            width=1280,
            height=720,
            duration_seconds=10.0,
            model_name="yolo11m.pt",
            tracker_name="botsort.yaml",
            conf_threshold=0.15,
            ingested_at_utc="2026-11-04T18:00:00+00:00",
            annotated_video_path="outputs/annotated_videos/laptop_test.avi",
        )
        return self.store.create_ingestion_run(meta, replace_existing_for_source=replace)

    def test_two_recordings_same_camera_id_never_collide(self) -> None:
        """Task 3: Two different recordings assigned to the same camera ID must not collide on local_track_id=1."""
        run_a = self._create_run(camera_id="office_cam_01", source_video="morning_clip.mp4")
        run_b = self._create_run(camera_id="office_cam_01", source_video="afternoon_clip.mp4")
        self.assertNotEqual(run_a, run_b)

        # Both recordings produce local_track_id = 1 for class "laptop"
        sightings_a = [
            SightingRecord(
                run_id=run_a,
                camera_id="office_cam_01",
                source_video="morning_clip.mp4",
                frame_index=i,
                timestamp_seconds=round(i / 24.0, 4),
                class_id=63,
                class_name="laptop",
                local_track_id=1,
                confidence=0.88,
                bbox_xyxy=(100.0, 200.0, 220.0, 300.0),
            )
            for i in range(4)
        ]
        sightings_b = [
            SightingRecord(
                run_id=run_b,
                camera_id="office_cam_01",
                source_video="afternoon_clip.mp4",
                frame_index=i,
                timestamp_seconds=round(i / 24.0, 4),
                class_id=63,
                class_name="laptop",
                local_track_id=1,
                confidence=0.22,
                bbox_xyxy=(800.0, 300.0, 860.0, 340.0),
            )
            for i in range(4)
        ]
        self.store.insert_sightings(sightings_a + sightings_b)

        report = query_asset_sightings(
            store=self.store,
            query_class="laptop",
            camera_id="office_cam_01",
            confirmed_conf_threshold=0.50,
            confirmed_min_track_frames=3,
        )
        self.assertEqual(len(report.track_summaries), 2)
        keys = {t.scoped_track_key for t in report.track_summaries}
        self.assertEqual(
            keys,
            {
                f"run{run_a}:office_cam_01:morning_clip.mp4:local_track_1",
                f"run{run_b}:office_cam_01:afternoon_clip.mp4:local_track_1",
            },
        )
        by_video = {t.source_video: t for t in report.track_summaries}
        self.assertEqual(by_video["morning_clip.mp4"].status, "STRONGLY_SUPPORTED")
        self.assertEqual(by_video["afternoon_clip.mp4"].status, "UNCERTAIN_CANDIDATE")

    def test_repeated_ingestion_of_same_video_prevents_duplicates(self) -> None:
        """Task 3: Re-ingesting the same video (even with ./ path variation) replaces prior run without duplicates."""
        run1 = self._create_run(camera_id="office_cam_01", source_video="laptop_test.mp4")
        s1 = SightingRecord(
            run_id=run1,
            camera_id="office_cam_01",
            source_video="laptop_test.mp4",
            frame_index=0,
            timestamp_seconds=0.0,
            class_id=63,
            class_name="laptop",
            local_track_id=1,
            confidence=0.92,
            bbox_xyxy=(535.3, 452.3, 745.6, 656.0),
        )
        # Also pass duplicate sighting within the same batch
        self.store.insert_sightings([s1, s1])
        self.assertEqual(len(self.store.query_sightings(class_name="laptop")), 1)

        # Re-ingest using ./laptop_test.mp4 path alias
        run2 = self._create_run(camera_id="office_cam_01", source_video="./laptop_test.mp4")
        self.assertNotEqual(run1, run2)
        s2 = SightingRecord(
            run_id=run2,
            camera_id="office_cam_01",
            source_video="./laptop_test.mp4",
            frame_index=0,
            timestamp_seconds=0.0,
            class_id=63,
            class_name="laptop",
            local_track_id=1,
            confidence=0.92,
            bbox_xyxy=(535.3, 452.3, 745.6, 656.0),
        )
        self.store.insert_sightings([s2])

        runs = self.store.get_ingestion_runs(camera_id="office_cam_01")
        self.assertEqual(len(runs), 1)
        self.assertEqual(runs[0].run_id, run2)

        db_rows = self.store.query_sightings(class_name="laptop")
        jsonl_rows = SightingStore.load_jsonl(self.jsonl_path)
        self.assertEqual(len(db_rows), 1)
        self.assertEqual(len(jsonl_rows), 1)
        self.assertEqual(db_rows[0].run_id, run2)

    def test_conflicting_overlapping_tracks_and_incorrect_identity_isolation(self) -> None:
        """Task 5: Conflicting tracks in the same frames are flagged and never merged."""
        run_id = self._create_run()
        records = [
            # Track 1: strongly supported foreground laptop (frames 0..3)
            SightingRecord(
                run_id=run_id,
                camera_id="office_cam_01",
                source_video="laptop_test.mp4",
                frame_index=f,
                timestamp_seconds=round(f / 24.0, 4),
                class_id=63,
                class_name="laptop",
                local_track_id=1,
                confidence=0.88,
                bbox_xyxy=(535.0, 452.0, 745.0, 656.0),
            )
            for f in range(4)
        ] + [
            # Track 7 (stationary desk phone box) and Track 12 (wider overlapping box) co-occurring at frames 200..202
            SightingRecord(
                run_id=run_id,
                camera_id="office_cam_01",
                source_video="laptop_test.mp4",
                frame_index=f,
                timestamp_seconds=round(f / 24.0, 4),
                class_id=63,
                class_name="laptop",
                local_track_id=7,
                confidence=0.20,
                bbox_xyxy=(911.0, 308.0, 946.0, 331.0),
            )
            for f in (200, 201, 202)
        ] + [
            SightingRecord(
                run_id=run_id,
                camera_id="office_cam_01",
                source_video="laptop_test.mp4",
                frame_index=f,
                timestamp_seconds=round(f / 24.0, 4),
                class_id=63,
                class_name="laptop",
                local_track_id=12,
                confidence=0.31,
                bbox_xyxy=(880.0, 308.0, 947.0, 351.0),
            )
            for f in (200, 201, 202)
        ]
        self.store.insert_sightings(records)

        report = query_asset_sightings(store=self.store, query_class="laptop", camera_id="office_cam_01")
        self.assertEqual(len(report.track_summaries), 3)
        self.assertEqual(len(report.conflicting_track_pairs), 1)
        conflict = report.conflicting_track_pairs[0]
        self.assertEqual(conflict["shared_frame_count"], 3)
        self.assertEqual(conflict["conflict_type"], "SPATIALLY_NESTED_DUPLICATE_BOXES")
        self.assertGreater(conflict["max_bbox_containment"], 0.95)
        self.assertTrue(any("CONFLICTING / OVERLAPPING LOCAL TRACKS" in n for n in report.uncertainty_notes))

    def test_missing_detections_and_uncertain_only_results(self) -> None:
        """Task 5: Missing detections and uncertain-only tracks produce calibrated uncertainty."""
        empty_report = build_retrieval_report([], query_class="laptop")
        self.assertIsNone(empty_report.last_strongly_supported_sighting)
        self.assertEqual(empty_report.total_sightings, 0)
        self.assertTrue(any("NO DETECTIONS" in n for n in empty_report.uncertainty_notes))

        # Track 7 style: 10 frames where 3 spike above 0.50 (up to 0.63) but 7 are ~0.20 (mean ~ 0.30 < 0.50)
        spike_confs = [0.20, 0.21, 0.55, 0.63, 0.51, 0.22, 0.19, 0.20, 0.18, 0.21]
        records = [
            SightingRecord(
                run_id=1,
                camera_id="office_cam_01",
                source_video="laptop_test.mp4",
                frame_index=178 + i,
                timestamp_seconds=round((178 + i) / 24.0, 4),
                class_id=63,
                class_name="laptop",
                local_track_id=7,
                confidence=c,
                bbox_xyxy=(905.0, 308.0, 946.0, 335.0),
            )
            for i, c in enumerate(spike_confs)
        ]
        u_report = build_retrieval_report(
            records,
            query_class="laptop",
            confirmed_conf_threshold=0.50,
            confirmed_min_track_frames=3,
        )
        self.assertIsNone(u_report.last_strongly_supported_sighting)
        self.assertEqual(len(u_report.track_summaries), 1)
        ts = u_report.track_summaries[0]
        self.assertEqual(ts.status, "UNCERTAIN_CANDIDATE")
        # Raw max detector confidence (0.63) is preserved separately from heuristic stability score (0.525)
        self.assertAlmostEqual(ts.max_confidence, 0.63)
        self.assertLess(ts.mean_confidence, 0.50)
        self.assertAlmostEqual(ts.heuristic_stability_score, 0.525)
        self.assertTrue(any("UNCERTAIN CANDIDATES ONLY" in n for n in u_report.uncertainty_notes))


class TestRealLaptopVideoVerification(unittest.TestCase):
    """Task 1 & Task 4: Verify the actual SQLite database, evidence images, and gap analysis on laptop_test.mp4."""

    @unittest.skipUnless(
        Path("outputs/sightings.db").exists() and Path("laptop_test.mp4").exists(),
        "Real outputs/sightings.db and laptop_test.mp4 required for live verification test.",
    )
    def test_verify_last_supported_sighting_at_4_458s_and_gap_comparison(self) -> None:
        store = SightingStore(db_path="outputs/sightings.db", jsonl_path=None)
        report = query_asset_sightings(
            store=store,
            query_class="laptop",
            camera_id="office_cam_01",
            source_video="laptop_test.mp4",
        )
        last_s = report.last_strongly_supported_sighting
        self.assertIsNotNone(last_s)
        assert last_s is not None

        # Verify Frame 107 (4.4583s) metadata, bbox, confidence, and evidence files on disk
        self.assertEqual(last_s.camera_id, "office_cam_01")
        self.assertEqual(last_s.source_video, "laptop_test.mp4")
        self.assertEqual(last_s.local_track_id, 1)
        self.assertEqual(last_s.frame_index, 107)
        self.assertAlmostEqual(last_s.timestamp_seconds, 4.4583, places=4)
        self.assertAlmostEqual(last_s.confidence, 0.7621, places=4)
        self.assertEqual(last_s.bbox_xyxy, (574.38, 357.2, 683.27, 401.74))

        self.assertIsNotNone(last_s.evidence_frame_path)
        self.assertIsNotNone(last_s.evidence_crop_path)
        assert last_s.evidence_frame_path is not None and last_s.evidence_crop_path is not None
        self.assertTrue(os.path.exists(last_s.evidence_frame_path))
        self.assertTrue(os.path.exists(last_s.evidence_crop_path))

        frame_img = cv2.imread(last_s.evidence_frame_path)
        crop_img = cv2.imread(last_s.evidence_crop_path)
        self.assertIsNotNone(frame_img)
        self.assertIsNotNone(crop_img)
        assert frame_img is not None and crop_img is not None
        self.assertEqual(frame_img.shape, (720, 1280, 3))
        self.assertGreater(crop_img.shape[0], 40)
        self.assertGreater(crop_img.shape[1], 100)

        # Verify gap and candidate comparison analysis (Tracks 1, 7, and 12)
        analysis = analyze_tracking_gap_and_candidates(
            store=store,
            camera_id="office_cam_01",
            source_video="laptop_test.mp4",
            query_class="laptop",
        )
        self.assertEqual(analysis["reference_track"]["local_track_id"], 1)
        self.assertEqual(analysis["tracking_gap_analysis"]["complete_blind_gap_frames"], [111, 177])
        self.assertEqual(len(analysis["candidate_comparisons"]), 2)

        by_id = {c["local_track_id"]: c for c in analysis["candidate_comparisons"]}
        self.assertIn(7, by_id)
        self.assertIn(12, by_id)
        # Track 7 has >0.95 NCC with Frame 0 stationary desk telephone
        self.assertGreater(by_id[7]["visual_metrics"]["pre_arrival_frame0_ncc"], 0.95)
        # Track 12 has lower-left placed object divergence from Frame 0
        self.assertLess(by_id[12]["visual_metrics"]["placed_subregion_frame0_ncc"], 0.80)


class TestMultiCameraRetrievalEvaluation(unittest.TestCase):
    """Task 4: Multi-camera synthetic evaluation fixtures (7 core scenarios)."""

    def setUp(self) -> None:
        self.tmp_dir = tempfile.TemporaryDirectory()
        self.base = Path(self.tmp_dir.name)
        self.db_path = self.base / "multi_cam.db"
        self.jsonl_path = self.base / "multi_cam.jsonl"
        self.store = SightingStore(db_path=self.db_path, jsonl_path=self.jsonl_path)

    def tearDown(self) -> None:
        self.tmp_dir.cleanup()

    def _register_camera_run(
        self,
        camera_id: str,
        source_video: str,
        duration_seconds: float = 10.0,
        recording_start_utc: str | None = None,
        recording_timezone: str | None = None,
        time_offset_seconds: float | None = None,
        source_identity: str | None = None,
    ) -> int:
        meta = VideoMetadata(
            camera_id=camera_id,
            source_video=source_video,
            source_identity=source_identity or f"id:{camera_id}:{source_video}",
            fps=24.0,
            frame_count=int(duration_seconds * 24),
            width=1280,
            height=720,
            duration_seconds=duration_seconds,
            model_name="yolo11m.pt",
            tracker_name="botsort.yaml",
            conf_threshold=0.15,
            recording_start_utc=recording_start_utc,
            recording_timezone=recording_timezone,
            time_offset_seconds=time_offset_seconds,
            ingested_at_utc="2026-11-05T12:00:00+00:00",
        )
        return self.store.create_ingestion_run(meta, replace_existing_for_source=True)

    def _make_track(
        self,
        run_id: int,
        camera_id: str,
        source_video: str,
        local_track_id: int,
        frames: list[int],
        conf: float,
        bbox: tuple[float, float, float, float] = (200.0, 200.0, 320.0, 300.0),
    ) -> list[SightingRecord]:
        return [
            SightingRecord(
                run_id=run_id,
                camera_id=camera_id,
                source_video=source_video,
                frame_index=f,
                timestamp_seconds=round(f / 24.0, 4),
                class_id=63,
                class_name="laptop",
                local_track_id=local_track_id,
                confidence=conf,
                bbox_xyxy=bbox,
                evidence_frame_path=f"outputs/evidence/{camera_id}/f_{f}.jpg",
                evidence_crop_path=f"outputs/evidence/{camera_id}/c_{f}.jpg",
            )
            for f in frames
        ]

    def test_scenario_1_two_cameras_recording_same_time_period(self) -> None:
        """1. Two cameras recording the same time period with simultaneous sightings."""
        run_cam1 = self._register_camera_run(
            "office_cam_01",
            "cam1_sync.mp4",
            recording_start_utc="2026-11-05T09:00:00+01:00",
        )
        run_cam2 = self._register_camera_run(
            "lab_cam_02",
            "cam2_sync.mp4",
            recording_start_utc="2026-11-05T09:00:00+01:00",
        )
        # Both cameras detect a laptop during frames 24..28 (1.0s..1.17s after 09:00:00)
        s_cam1 = self._make_track(run_cam1, "office_cam_01", "cam1_sync.mp4", 1, [24, 25, 26, 27], 0.90)
        s_cam2 = self._make_track(run_cam2, "lab_cam_02", "cam2_sync.mp4", 1, [25, 26, 27, 28], 0.88)
        self.store.insert_sightings(s_cam1 + s_cam2)

        report = query_asset_sightings(store=self.store, query_class="laptop")
        self.assertTrue(report.chronologically_orderable)
        self.assertEqual(report.synchronization_mode, "MULTI_CAMERA_WALL_CLOCK_ALIGNED")
        # Simultaneous strongly supported detections in two cameras -> INCONCLUSIVE single-asset location
        self.assertEqual(report.overall_conclusion_status, "INCONCLUSIVE")
        obs_types = {o["type"] for o in report.cross_camera_observations}
        self.assertIn("SIMULTANEOUS_MULTI_CAMERA_SIGHTINGS", obs_types)
        self.assertIn("UNLINKED_MULTI_CAMERA_CLASS_SIGHTINGS", obs_types)
        # Track keys remain isolated despite both having local_track_id=1
        self.assertEqual(len(report.track_summaries), 2)

    def test_scenario_2_two_cameras_with_different_start_times(self) -> None:
        """2. Two cameras with different start times order sightings by aligned wall-clock time."""
        # Cam 1 starts at 09:00:00 and sees laptop at relative +8.0s (09:00:08)
        run_cam1 = self._register_camera_run(
            "office_cam_01",
            "cam1_early.mp4",
            duration_seconds=9.0,
            recording_start_utc="2026-11-05T09:00:00+01:00",
        )
        # Cam 2 starts 5 minutes later at 09:05:00 and sees laptop at relative +2.0s (09:05:02)
        run_cam2 = self._register_camera_run(
            "corridor_cam_02",
            "cam2_later.mp4",
            duration_seconds=2.5,
            recording_start_utc="2026-11-05T09:05:00+01:00",
        )
        s_cam1 = self._make_track(run_cam1, "office_cam_01", "cam1_early.mp4", 1, [190, 191, 192, 193], 0.91)
        s_cam2 = self._make_track(run_cam2, "corridor_cam_02", "cam2_later.mp4", 1, [46, 47, 48, 49], 0.86)
        self.store.insert_sightings(s_cam1 + s_cam2)

        report = query_asset_sightings(store=self.store, query_class="laptop")
        self.assertTrue(report.chronologically_orderable)
        self.assertEqual(report.synchronization_mode, "MULTI_CAMERA_WALL_CLOCK_ALIGNED")
        self.assertIsNotNone(report.last_strongly_supported_sighting)
        assert report.last_strongly_supported_sighting is not None
        # Even though Cam 2 relative timestamp (~2.04s) < Cam 1 relative timestamp (~8.04s),
        # Cam 2 occurred at 09:05:02 vs 09:00:08, so Cam 2 is the latest strongly supported sighting!
        self.assertEqual(report.last_strongly_supported_sighting.camera_id, "corridor_cam_02")
        self.assertIn("2026-11-05T09:05:02", report.last_strongly_supported_sighting.aligned_wall_clock_utc or "")
        self.assertIn("UNVERIFIED_PHYSICAL_IDENTITY", report.verified_asset_identity_status)

    def test_scenario_3_missing_camera_synchronization_metadata(self) -> None:
        """3. Missing camera synchronization metadata prevents cross-camera ranking and groups by camera."""
        run_cam1 = self._register_camera_run("office_cam_01", "cam1_unsync.mp4", recording_start_utc=None)
        run_cam2 = self._register_camera_run("corridor_cam_02", "cam2_unsync.mp4", recording_start_utc=None)
        s_cam1 = self._make_track(run_cam1, "office_cam_01", "cam1_unsync.mp4", 1, [10, 11, 12, 13], 0.89)
        s_cam2 = self._make_track(run_cam2, "corridor_cam_02", "cam2_unsync.mp4", 1, [50, 51, 52, 53], 0.85)
        self.store.insert_sightings(s_cam1 + s_cam2)

        report = query_asset_sightings(store=self.store, query_class="laptop")
        self.assertFalse(report.chronologically_orderable)
        self.assertEqual(report.synchronization_mode, "UNSYNCHRONIZED_SEPARATE_BY_CAMERA")
        self.assertEqual(report.overall_conclusion_status, "INCONCLUSIVE")
        # Cross-camera winner must NOT be asserted when unsynchronized
        self.assertIsNone(report.last_strongly_supported_sighting)
        # Per-camera reports preserve each camera's independent last strongly supported sighting
        self.assertIn("office_cam_01", report.per_camera_reports)
        self.assertIn("corridor_cam_02", report.per_camera_reports)
        self.assertIsNotNone(report.per_camera_reports["office_cam_01"].last_strongly_supported_sighting)
        self.assertIsNotNone(report.per_camera_reports["corridor_cam_02"].last_strongly_supported_sighting)

    def test_scenario_4_similar_looking_objects_in_different_cameras(self) -> None:
        """4. Similar-looking laptops in different cameras are never claimed to be the same physical object."""
        run_cam1 = self._register_camera_run("office_cam_01", "cam1.mp4", time_offset_seconds=0.0)
        run_cam2 = self._register_camera_run("lobby_cam_03", "cam3.mp4", time_offset_seconds=20.0)
        s_cam1 = self._make_track(run_cam1, "office_cam_01", "cam1.mp4", 1, [0, 1, 2, 3], 0.93)
        s_cam2 = self._make_track(run_cam2, "lobby_cam_03", "cam3.mp4", 1, [0, 1, 2, 3], 0.91)
        self.store.insert_sightings(s_cam1 + s_cam2)

        report = query_asset_sightings(store=self.store, query_class="laptop")
        self.assertEqual(len(report.track_summaries), 2)
        self.assertIn("UNVERIFIED_PHYSICAL_IDENTITY", report.verified_asset_identity_status)
        self.assertTrue(
            any("MULTI-CAMERA IDENTITY AMBIGUITY" in n for n in report.uncertainty_notes)
        )

    def test_scenario_5_one_camera_producing_false_positive_detection(self) -> None:
        """5. A later false-positive detection in Camera 2 does not override Camera 1's strongly supported sighting."""
        run_cam1 = self._register_camera_run(
            "office_cam_01",
            "cam1.mp4",
            duration_seconds=4.5,
            recording_start_utc="2026-11-05T09:00:00+01:00",
        )
        run_cam2 = self._register_camera_run(
            "corridor_cam_02",
            "cam2.mp4",
            duration_seconds=10.0,
            recording_start_utc="2026-11-05T09:01:00+01:00",
        )
        # Camera 1 has a strongly supported track
        s_cam1 = self._make_track(run_cam1, "office_cam_01", "cam1.mp4", 1, [96, 97, 98, 99], 0.88)
        # Camera 2 (1 minute later) only has a 2-frame low-confidence false-positive blip (conf=0.22)
        s_cam2_fp = self._make_track(run_cam2, "corridor_cam_02", "cam2.mp4", 4, [48, 49], 0.22)
        self.store.insert_sightings(s_cam1 + s_cam2_fp)

        report = query_asset_sightings(store=self.store, query_class="laptop")
        self.assertIsNotNone(report.last_strongly_supported_sighting)
        assert report.last_strongly_supported_sighting is not None
        # Last strongly supported sighting remains Camera 1!
        self.assertEqual(report.last_strongly_supported_sighting.camera_id, "office_cam_01")
        # Latest raw class detection (any tier) is Camera 2's uncertain candidate
        self.assertIsNotNone(report.latest_class_detection)
        assert report.latest_class_detection is not None
        self.assertEqual(report.latest_class_detection.camera_id, "corridor_cam_02")
        self.assertEqual(len(report.later_uncertain_sightings), 2)
        self.assertEqual(report.per_camera_reports["corridor_cam_02"].camera_conclusion_status, "UNCERTAIN_ONLY")

    def test_scenario_6_repeated_multi_camera_ingestion_without_duplicate_evidence(self) -> None:
        """6. Repeated ingestion across multiple cameras replaces runs cleanly without duplicate evidence."""
        r1 = self._register_camera_run("office_cam_01", "cam1.mp4", source_identity="sha256:aaa111")
        r2 = self._register_camera_run("corridor_cam_02", "cam2.mp4", source_identity="sha256:bbb222")
        self.store.insert_sightings(
            self._make_track(r1, "office_cam_01", "cam1.mp4", 1, [0, 1, 2], 0.90)
            + self._make_track(r2, "corridor_cam_02", "cam2.mp4", 1, [0, 1, 2], 0.85)
        )
        self.assertEqual(len(self.store.query_sightings(class_name="laptop")), 6)

        # Re-ingest Camera 1 with identical source_identity
        r1_new = self._register_camera_run("office_cam_01", "./cam1.mp4", source_identity="sha256:aaa111")
        self.store.insert_sightings(
            self._make_track(r1_new, "office_cam_01", "./cam1.mp4", 1, [0, 1, 2], 0.90)
        )
        self.assertEqual(len(self.store.get_ingestion_runs()), 2)
        self.assertEqual(len(self.store.query_sightings(class_name="laptop")), 6)
        self.assertEqual(len(SightingStore.load_jsonl(self.jsonl_path)), 6)

    def test_scenario_7_object_leaving_one_camera_without_supported_sighting_in_another(self) -> None:
        """7. Object leaves Camera 1 early (at 2.0s of a 10.0s clip) with 0 sightings in Camera 2 -> INCONCLUSIVE."""
        run_cam1 = self._register_camera_run(
            "office_cam_01",
            "cam1_departure.mp4",
            duration_seconds=10.0,
            recording_start_utc="2026-11-05T09:00:00+01:00",
        )
        self._register_camera_run(
            "corridor_cam_02",
            "cam2_empty.mp4",
            duration_seconds=10.0,
            recording_start_utc="2026-11-05T09:00:00+01:00",
        )
        # Laptop only seen in first 2 seconds (frames 45..48) of Cam 1, then leaves view; 0 detections in Cam 2
        s_cam1 = self._make_track(run_cam1, "office_cam_01", "cam1_departure.mp4", 1, [45, 46, 47, 48], 0.89)
        self.store.insert_sightings(s_cam1)

        report = query_asset_sightings(store=self.store, query_class="laptop")
        self.assertEqual(report.overall_conclusion_status, "INCONCLUSIVE")
        self.assertIsNotNone(report.last_strongly_supported_sighting)
        assert report.last_strongly_supported_sighting is not None
        self.assertEqual(report.last_strongly_supported_sighting.camera_id, "office_cam_01")
        self.assertEqual(report.per_camera_reports["corridor_cam_02"].camera_conclusion_status, "NO_DETECTIONS")
        obs_types = {o["type"] for o in report.cross_camera_observations}
        self.assertIn("DEPARTURE_WITHOUT_DOWNSTREAM_CONFIRMATION", obs_types)


class DummyDeterministicEncoder:
    """Lightweight deterministic image/text encoder for fast unit testing of ChromaDB + SQLite integration."""

    def __init__(self, model_id: str = "openclip:ViT-B-32:dummy_v1", dim: int = 8) -> None:
        import numpy as np

        self._model_id = model_id
        self._dim = dim
        self._np = np

    @property
    def model_id(self) -> str:
        return self._model_id

    @property
    def embedding_dim(self) -> int:
        return self._dim

    @property
    def device_str(self) -> str:
        return "cpu"

    def encode_images(self, pil_images: list[object]) -> object:
        import numpy as np
        from PIL import Image

        if not pil_images:
            return np.empty((0, self._dim), dtype=np.float32)
        rows = []
        for img in pil_images:
            assert isinstance(img, Image.Image)
            arr = np.asarray(img.convert("RGB"), dtype=np.float32)
            # Encode mean RGB & spatial gradients into an L2-normalized vector
            r_mean = float(arr[:, :, 0].mean()) / 255.0
            g_mean = float(arr[:, :, 1].mean()) / 255.0
            b_mean = float(arr[:, :, 2].mean()) / 255.0
            vec = np.zeros(self._dim, dtype=np.float32)
            vec[0] = r_mean + 0.05
            vec[1] = g_mean + 0.05
            vec[2] = b_mean + 0.05
            vec[3] = float(arr.std()) / 255.0 + 0.01
            for k in range(4, self._dim):
                vec[k] = 0.1 * ((k + 1) / self._dim)
            rows.append(vec)
        return normalize_embedding_vector(np.vstack(rows))

    def encode_text(self, query_text: str) -> object:
        import numpy as np

        q = query_text.lower()
        vec = np.full(self._dim, 0.05, dtype=np.float32)
        if "laptop" in q or "red" in q:
            vec[0] = 1.0  # Strongly aligns with high-R test images
        elif "telephone" in q or "blue" in q:
            vec[2] = 1.0  # Strongly aligns with high-B test images
        elif "person" in q or "green" in q:
            vec[1] = 1.0  # Strongly aligns with high-G test images
        else:
            vec[0] = -1.0
            vec[1] = -1.0
            vec[2] = -1.0
        return normalize_embedding_vector(vec)


class TestOpenCLIPAndChromaDBVisualRetrieval(unittest.TestCase):
    """Milestone 3: Unit and live tests for OpenCLIP embedding extraction and ChromaDB visual search."""

    def setUp(self) -> None:
        self.tmp_dir = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        self.base = Path(self.tmp_dir.name)
        self.db_path = self.base / "visual_test.db"
        self.chroma_dir = self.base / "chroma_test"
        self.ev_dir = self.base / "evidence"
        self.ev_dir.mkdir(parents=True, exist_ok=True)
        self.store = SightingStore(db_path=self.db_path, jsonl_path=None)
        self.encoder = DummyDeterministicEncoder()

    def tearDown(self) -> None:
        self.tmp_dir.cleanup()

    def _save_solid_image(self, path: Path, rgb: tuple[int, int, int], size: tuple[int, int] = (120, 90)) -> str:
        from PIL import Image

        path.parent.mkdir(parents=True, exist_ok=True)
        img = Image.new("RGB", size, color=rgb)
        img.save(path, format="JPEG")
        return path.as_posix()

    def _register_run_and_sightings(
        self,
        camera_id: str = "office_cam_01",
        source_video: str = "cam1.mp4",
        rgb_color: tuple[int, int, int] = (240, 20, 20),
        class_name: str = "laptop",
        local_track_id: int = 1,
        frames: list[int] = [0, 1, 2, 3],
        conf: float = 0.90,
    ) -> tuple[int, list[SightingRecord]]:
        meta = VideoMetadata(
            camera_id=camera_id,
            source_video=source_video,
            source_identity=f"id:{camera_id}:{source_video}",
            fps=24.0,
            frame_count=240,
            width=640,
            height=360,
            duration_seconds=10.0,
            model_name="yolo11m.pt",
            tracker_name="botsort.yaml",
            conf_threshold=0.15,
            recording_start_utc="2026-11-05T09:00:00+01:00",
            ingested_at_utc="2026-11-05T12:00:00+00:00",
        )
        run_id = self.store.create_ingestion_run(meta, replace_existing_for_source=True)
        records: list[SightingRecord] = []
        for f in frames:
            crop_p = self._save_solid_image(
                self.ev_dir / camera_id / f"crop_{class_name}_t{local_track_id}_f{f}.jpg",
                rgb=rgb_color,
                size=(80, 60),
            )
            frame_p = self._save_solid_image(
                self.ev_dir / camera_id / f"frame_{class_name}_t{local_track_id}_f{f}.jpg",
                rgb=rgb_color,
                size=(320, 240),
            )
            records.append(
                SightingRecord(
                    run_id=run_id,
                    camera_id=camera_id,
                    source_video=source_video,
                    frame_index=f,
                    timestamp_seconds=round(f / 24.0, 4),
                    class_id=63 if class_name == "laptop" else 0,
                    class_name=class_name,
                    local_track_id=local_track_id,
                    confidence=conf,
                    bbox_xyxy=(50.0, 50.0, 130.0, 110.0),
                    evidence_crop_path=crop_p,
                    evidence_frame_path=frame_p,
                )
            )
        persisted = self.store.insert_sightings(records)
        return run_id, persisted

    def test_1_embedding_dimensions_and_l2_normalization(self) -> None:
        """Verify 1D and 2D embedding L2 normalization, zero-norm rejection, and deterministic IDs."""
        import numpy as np

        vec_1d = normalize_embedding_vector([3.0, 4.0, 0.0, 0.0])
        self.assertEqual(vec_1d.shape, (4,))
        self.assertAlmostEqual(float(np.linalg.norm(vec_1d)), 1.0, places=6)

        batch_2d = normalize_embedding_vector([[1.0, 2.0, 2.0], [0.0, 3.0, 4.0]])
        self.assertEqual(batch_2d.shape, (2, 3))
        np.testing.assert_allclose(np.linalg.norm(batch_2d, axis=1), [1.0, 1.0], rtol=1e-5)

        with self.assertRaises(ValueError):
            normalize_embedding_vector([0.0, 0.0, 0.0])

        model_id = make_model_identifier("ViT-B-32", "laion2b_s34b_b79k")
        emb_id_a = make_embedding_id(42, "crop", model_id)
        emb_id_b = make_embedding_id(42, "crop", model_id)
        emb_id_ctx = make_embedding_id(42, "context", model_id)
        self.assertEqual(emb_id_a, emb_id_b)
        self.assertNotEqual(emb_id_a, emb_id_ctx)

    def test_2_context_region_extraction_and_corrupted_file_handling(self) -> None:
        """Verify context region expansion around bbox and graceful handling of missing/corrupted images."""
        from PIL import Image

        # Missing file returns None
        self.assertIsNone(load_pil_image_safely(self.base / "nonexistent.jpg"))

        # Empty / corrupted file returns None without raising
        corrupt_path = self.base / "corrupt.jpg"
        corrupt_path.write_bytes(b"not_a_valid_jpeg_stream")
        self.assertIsNone(load_pil_image_safely(corrupt_path))

        empty_path = self.base / "empty.jpg"
        empty_path.write_bytes(b"")
        self.assertIsNone(load_pil_image_safely(empty_path))

        # Context crop is larger than tight bbox and clamped cleanly to frame bounds
        full_img = Image.new("RGB", (640, 480), color=(100, 120, 140))
        ctx_img = extract_context_region_pil(full_img, (200.0, 150.0, 280.0, 210.0))
        self.assertIsNotNone(ctx_img)
        assert ctx_img is not None
        self.assertGreater(ctx_img.width, 80)
        self.assertGreater(ctx_img.height, 60)
        self.assertLessEqual(ctx_img.width, 640)
        self.assertLessEqual(ctx_img.height, 480)

    def test_3_persistent_storage_reload_and_repeated_indexing_without_duplicates(self) -> None:
        """Verify ChromaDB persistence across reloads and idempotent re-indexing without duplicate entries."""
        _, persisted = self._register_run_and_sightings(frames=[0, 1, 2])
        self.assertEqual(len(persisted), 3)

        index_a = ChromaVisualIndex(persist_dir=self.chroma_dir, collection_name="test_col")
        res_1 = index_a.index_sightings_from_store(self.store, encoder=self.encoder)
        # 3 sightings * 2 variants (crop + context) = 6 embeddings
        self.assertEqual(res_1["sightings_indexed"], 3)
        self.assertEqual(res_1["total_embeddings_upserted"], 6)
        self.assertEqual(index_a.count(), 6)

        # Re-index the exact same sightings again -> count must remain 6 (no duplicates)
        res_2 = index_a.index_sightings_from_store(self.store, encoder=self.encoder)
        self.assertEqual(res_2["collection_total_embeddings"], 6)
        self.assertEqual(index_a.count(), 6)

        # Reload a fresh ChromaVisualIndex instance from the same directory -> 6 embeddings persisted
        index_reloaded = ChromaVisualIndex(persist_dir=self.chroma_dir, collection_name="test_col")
        self.assertEqual(index_reloaded.count(), 6)
        stats = index_reloaded.get_index_stats(store=self.store)
        self.assertEqual(stats["total_embeddings"], 6)
        self.assertEqual(stats["distinct_sightings_indexed"], 3)
        self.assertEqual(stats["stale_embeddings_detected"], 0)
        self.assertEqual(stats["by_embedding_variant"], {"context": 3, "crop": 3})

    def test_4_stale_embeddings_pruned_after_reingestion(self) -> None:
        """When an ingestion run is replaced in SQLite, stale embeddings from the old run are removed/excluded."""
        run_1, sightings_1 = self._register_run_and_sightings(
            camera_id="office_cam_01",
            source_video="cam1.mp4",
            frames=[0, 1, 2, 3],
        )
        old_sighting_ids = {s.sighting_id for s in sightings_1}
        index = ChromaVisualIndex(persist_dir=self.chroma_dir, collection_name="test_stale")
        index.index_sightings_from_store(self.store, encoder=self.encoder)
        self.assertEqual(index.count(), 8)  # 4 sightings * 2 variants

        # Re-ingest the same camera + video with only 2 frames (replaces run_1 in SQLite via ON DELETE CASCADE)
        run_2, sightings_2 = self._register_run_and_sightings(
            camera_id="office_cam_01",
            source_video="cam1.mp4",
            frames=[0, 1],
        )
        self.assertNotEqual(run_1, run_2)
        new_sighting_ids = {s.sighting_id for s in sightings_2}
        self.assertTrue(old_sighting_ids.isdisjoint(new_sighting_ids))

        # Before re-indexing, search automatically prunes the 8 stale embeddings from run_1
        search_before_reindex = index.search_by_description(
            query_text="black laptop",
            store=self.store,
            encoder=self.encoder,
        )
        self.assertEqual(search_before_reindex.stale_embeddings_excluded, 8)
        self.assertEqual(len(search_before_reindex.ranked_sighting_matches), 0)
        self.assertEqual(index.count(), 0)

        # Now index the new run -> exactly 4 embeddings (2 sightings * 2 variants)
        reindex_summary = index.index_sightings_from_store(self.store, encoder=self.encoder)
        self.assertEqual(reindex_summary["collection_total_embeddings"], 4)
        search_after = index.search_by_description(
            query_text="black laptop",
            store=self.store,
            encoder=self.encoder,
        )
        self.assertEqual(len(search_after.ranked_sighting_matches), 2)
        for m in search_after.ranked_sighting_matches:
            self.assertEqual(m.sighting.run_id, run_2)
            self.assertIn(m.sighting.sighting_id, new_sighting_ids)

    def test_5_missing_evidence_files_skipped_gracefully_during_indexing(self) -> None:
        """Sightings with missing or corrupted evidence files are skipped without crashing indexing."""
        _, sightings = self._register_run_and_sightings(frames=[0, 1])
        # Delete the crop file of sighting 0 on disk
        assert sightings[0].evidence_crop_path is not None
        Path(sightings[0].evidence_crop_path).unlink()

        index = ChromaVisualIndex(persist_dir=self.chroma_dir, collection_name="test_missing")
        summary = index.index_sightings_from_store(self.store, encoder=self.encoder)
        # Sighting 0 has only context (1), Sighting 1 has crop + context (2) -> total 3 embeddings, 1 missing file
        self.assertEqual(summary["missing_or_corrupt_evidence_files"], 1)
        self.assertEqual(summary["crop_embeddings_indexed"], 1)
        self.assertEqual(summary["context_embeddings_indexed"], 2)
        self.assertEqual(summary["total_embeddings_upserted"], 3)

    def test_6_camera_filtering_model_compatibility_and_empty_results(self) -> None:
        """Verify camera/class filtering, model compatibility isolation, and empty/irrelevant query handling."""
        self._register_run_and_sightings(
            camera_id="office_cam_01",
            source_video="cam1.mp4",
            rgb_color=(240, 20, 20),  # aligns with 'laptop' in DummyDeterministicEncoder
            class_name="laptop",
            local_track_id=1,
            frames=[0, 1, 2],
        )
        self._register_run_and_sightings(
            camera_id="corridor_cam_02",
            source_video="cam2.mp4",
            rgb_color=(20, 20, 240),  # aligns with 'telephone' in DummyDeterministicEncoder
            class_name="laptop",
            local_track_id=1,
            frames=[0, 1, 2],
            conf=0.28,
        )
        index = ChromaVisualIndex(persist_dir=self.chroma_dir, collection_name="test_filters")
        index.index_sightings_from_store(self.store, encoder=self.encoder)

        # 1. Camera filter restricts results strictly to corridor_cam_02
        rep_cam2 = index.search_by_description(
            query_text="black laptop",
            store=self.store,
            encoder=self.encoder,
            camera_id="corridor_cam_02",
        )
        self.assertEqual(len(rep_cam2.ranked_sighting_matches), 3)
        self.assertTrue(all(m.sighting.camera_id == "corridor_cam_02" for m in rep_cam2.ranked_sighting_matches))
        # Track in corridor_cam_02 had conf=0.28 -> preserved as UNCERTAIN_CANDIDATE
        self.assertTrue(all(m.track_status == "UNCERTAIN_CANDIDATE" for m in rep_cam2.ranked_sighting_matches))

        # 2. Incompatible embedding model ID returns 0 results cleanly without mixing vector spaces
        other_model_encoder = DummyDeterministicEncoder(model_id="openclip:ViT-L-14:other_weights", dim=8)
        rep_incompatible = index.search_by_description(
            query_text="black laptop",
            store=self.store,
            encoder=other_model_encoder,
        )
        self.assertEqual(len(rep_incompatible.ranked_sighting_matches), 0)
        self.assertEqual(rep_incompatible.total_indexed_embeddings_searched, 0)

        # 3. High min_similarity threshold filters out low-similarity / irrelevant queries
        rep_irrelevant = index.search_by_description(
            query_text="unrelated object xyz",
            store=self.store,
            encoder=self.encoder,
            min_similarity=0.50,
        )
        self.assertEqual(len(rep_irrelevant.ranked_sighting_matches), 0)
        self.assertTrue(any("NO VISUAL MATCHES" in n for n in rep_irrelevant.uncertainty_and_calibration_notes))

    @unittest.skipUnless(
        Path("outputs/sightings.db").exists() and Path("outputs/chroma_db").exists(),
        "Real outputs/sightings.db and outputs/chroma_db required for live OpenCLIP verification test.",
    )
    def test_7_live_openclip_chromadb_laptop_and_telephone_queries(self) -> None:
        """Verify live OpenCLIP ViT-B-32 + ChromaDB index on laptop_test.mp4 across the 3 milestone queries."""
        import numpy as np

        store = SightingStore(db_path="outputs/sightings.db", jsonl_path=None)
        index = ChromaVisualIndex(persist_dir="outputs/chroma_db")
        encoder = OpenCLIPEmbedder.get_instance()

        self.assertEqual(encoder.embedding_dim, 512)
        q_vec = encoder.encode_text("black laptop")
        self.assertEqual(q_vec.shape, (512,))
        self.assertAlmostEqual(float(np.linalg.norm(q_vec)), 1.0, places=5)

        stats = index.get_index_stats(store=store)
        self.assertEqual(stats["total_embeddings"], 172)
        self.assertEqual(stats["distinct_sightings_indexed"], 86)
        self.assertEqual(stats["stale_embeddings_detected"], 0)

        # Query 1: "black laptop" -> Track 1 (STRONGLY_SUPPORTED) ranks #1
        rep_laptop = index.search_by_description("black laptop", store=store, encoder=encoder, top_k=5)
        self.assertGreaterEqual(len(rep_laptop.track_candidate_groups), 3)
        self.assertEqual(
            rep_laptop.track_candidate_groups[0].scoped_track_key,
            "run1:office_cam_01:laptop_test.mp4:local_track_1",
        )
        self.assertEqual(rep_laptop.track_candidate_groups[0].track_support_status, "STRONGLY_SUPPORTED")

        # Query 2: "person holding a black laptop" -> Track 1 (carried frames 60..110) and Track 2 (person @ f=96) top-ranked
        rep_holding = index.search_by_description(
            "person holding a black laptop",
            store=store,
            encoder=encoder,
            top_k=5,
        )
        self.assertEqual(
            rep_holding.track_candidate_groups[0].scoped_track_key,
            "run1:office_cam_01:laptop_test.mp4:local_track_1",
        )
        self.assertEqual(rep_holding.ranked_sighting_matches[0].sighting.frame_index, 60)

        # Query 3: "black desk telephone" -> Track 7 (the stationary desk telephone false positive) ranks #1!
        rep_phone = index.search_by_description("black desk telephone", store=store, encoder=encoder, top_k=5)
        self.assertEqual(
            rep_phone.track_candidate_groups[0].scoped_track_key,
            "run1:office_cam_01:laptop_test.mp4:local_track_7",
        )
        self.assertEqual(rep_phone.track_candidate_groups[0].track_support_status, "UNCERTAIN_CANDIDATE")
        self.assertGreater(
            rep_phone.track_candidate_groups[0].max_clip_similarity,
            rep_laptop.track_candidate_groups[3].max_clip_similarity,
        )


if __name__ == "__main__":
    unittest.main()

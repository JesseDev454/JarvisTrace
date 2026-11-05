"""CLI entry point for Group 8 multi-camera sighting ingestion, retrieval, and track comparison."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Sequence

from src.comparison import analyze_tracking_gap_and_candidates, format_comparison_report_text
from src.models import format_video_timestamp
from src.retrieval import format_retrieval_report_text, query_asset_sightings
from src.storage import SightingStore


def _parse_camera_list(raw_values: Sequence[str] | str | None) -> list[str]:
    """Parse one or more camera IDs (supporting space-separated or comma-separated lists)."""
    if not raw_values:
        return []
    if isinstance(raw_values, str):
        raw_values = [raw_values]
    out: list[str] = []
    for item in raw_values:
        for part in str(item).split(","):
            stripped = part.strip()
            if stripped and stripped not in out:
                out.append(stripped)
    return out


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m src.cli",
        description=(
            "Group 8 — Evidence-Guided Multi-Camera Visual Asset Retrieval System "
            "(Multi-Camera Sighting Indexer, Evidence-Calibrated Retrieval & Track Comparison CLI)"
        ),
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # 1. ingest subcommand
    ingest_p = subparsers.add_parser(
        "ingest",
        help="Run YOLO11 + BoT-SORT (stream=True) on a local video and persist sightings to SQLite & JSONL.",
    )
    ingest_p.add_argument(
        "--video",
        required=True,
        help="Path to local video file (e.g. laptop_test.mp4).",
    )
    ingest_p.add_argument(
        "--camera-id",
        required=True,
        help="Explicit camera identifier (e.g. office_cam_01).",
    )
    ingest_p.add_argument(
        "--start-time",
        default=None,
        help="Optional ISO-8601 recording start datetime (e.g. 2026-11-05T09:00:00+01:00). Never invented if omitted.",
    )
    ingest_p.add_argument(
        "--timezone",
        default=None,
        help="Optional recording timezone label or UTC offset (e.g. +01:00 or UTC).",
    )
    ingest_p.add_argument(
        "--time-offset",
        type=float,
        default=None,
        help="Optional relative synchronization offset in seconds (when wall-clock start time is unknown).",
    )
    ingest_p.add_argument(
        "--source-identity",
        default=None,
        help="Optional explicit source identity token (defaults to SHA-256 content fingerprint).",
    )
    ingest_p.add_argument(
        "--model",
        default="yolo11m.pt",
        help="Ultralytics YOLO model weights (default: yolo11m.pt).",
    )
    ingest_p.add_argument(
        "--tracker",
        default="botsort.yaml",
        help="Tracker configuration file (default: botsort.yaml).",
    )
    ingest_p.add_argument(
        "--imgsz",
        type=int,
        default=960,
        help="Inference image size (default: 960).",
    )
    ingest_p.add_argument(
        "--conf",
        type=float,
        default=0.15,
        help="Minimum candidate detection confidence to record (default: 0.15).",
    )
    ingest_p.add_argument(
        "--confirmed-conf",
        type=float,
        default=0.50,
        help="Detector confidence threshold for strongly supported sightings vs uncertain candidates (default: 0.50).",
    )
    ingest_p.add_argument(
        "--classes",
        type=int,
        nargs="*",
        default=[0, 63],
        help="COCO class IDs to detect/track (default: 0=person, 63=laptop).",
    )
    ingest_p.add_argument(
        "--device",
        default=None,
        help="Compute device override (default: CUDA device 0 if available, else cpu).",
    )
    ingest_p.add_argument(
        "--db-path",
        default="outputs/sightings.db",
        help="SQLite database path (default: outputs/sightings.db).",
    )
    ingest_p.add_argument(
        "--jsonl-path",
        default="outputs/sightings.jsonl",
        help="JSONL export path (default: outputs/sightings.jsonl).",
    )
    ingest_p.add_argument(
        "--output-dir",
        default="outputs",
        help="Directory for saved evidence frames, crops, and annotated videos (default: outputs).",
    )
    ingest_p.add_argument(
        "--keyframe-interval",
        type=int,
        default=12,
        help="Save evidence keyframes every N frames in addition to track start/peak/last/uncertain frames (default: 12).",
    )
    ingest_p.add_argument(
        "--no-annotated-video",
        action="store_true",
        help="Disable saving Ultralytics full annotated video clip.",
    )

    # 2. ingest-batch subcommand
    batch_p = subparsers.add_parser(
        "ingest-batch",
        help="Ingest multiple camera recordings from a JSON manifest file.",
    )
    batch_p.add_argument(
        "--manifest",
        required=True,
        help="Path to JSON file containing a list of camera recording objects.",
    )
    batch_p.add_argument(
        "--model",
        default="yolo11m.pt",
        help="Default YOLO model weights if not specified per entry (default: yolo11m.pt).",
    )
    batch_p.add_argument(
        "--tracker",
        default="botsort.yaml",
        help="Default tracker configuration (default: botsort.yaml).",
    )
    batch_p.add_argument(
        "--imgsz",
        type=int,
        default=960,
        help="Default inference image size (default: 960).",
    )
    batch_p.add_argument(
        "--conf",
        type=float,
        default=0.15,
        help="Default minimum candidate confidence (default: 0.15).",
    )
    batch_p.add_argument(
        "--confirmed-conf",
        type=float,
        default=0.50,
        help="Default strong-support confidence threshold (default: 0.50).",
    )
    batch_p.add_argument(
        "--classes",
        type=int,
        nargs="*",
        default=[0, 63],
        help="Default COCO class IDs (default: 0=person, 63=laptop).",
    )
    batch_p.add_argument(
        "--device",
        default=None,
        help="Compute device override.",
    )
    batch_p.add_argument(
        "--db-path",
        default="outputs/sightings.db",
        help="SQLite database path (default: outputs/sightings.db).",
    )
    batch_p.add_argument(
        "--jsonl-path",
        default="outputs/sightings.jsonl",
        help="JSONL export path (default: outputs/sightings.jsonl).",
    )
    batch_p.add_argument(
        "--output-dir",
        default="outputs",
        help="Directory for saved evidence frames, crops, and annotated videos.",
    )
    batch_p.add_argument(
        "--no-annotated-video",
        action="store_true",
        help="Disable saving Ultralytics full annotated video clips.",
    )

    # 3. query subcommand
    query_p = subparsers.add_parser(
        "query",
        help="Query last strongly supported detection and uncertain candidates across all or selected cameras.",
    )
    query_p.add_argument(
        "--class-name",
        default="laptop",
        help="Object class name to retrieve (default: laptop).",
    )
    query_p.add_argument(
        "--camera-id",
        nargs="*",
        default=None,
        help="Optional camera ID(s) to filter (omit to search all indexed cameras; accepts multiple IDs or comma-separated list).",
    )
    query_p.add_argument(
        "--source-video",
        default=None,
        help="Optional source video filter.",
    )
    query_p.add_argument(
        "--confirmed-conf",
        type=float,
        default=0.50,
        help="Minimum detector confidence required for a strongly supported sighting (default: 0.50).",
    )
    query_p.add_argument(
        "--min-track-frames",
        type=int,
        default=3,
        help="Minimum high-confidence frames in a local track for strong support (default: 3).",
    )
    query_p.add_argument(
        "--db-path",
        default="outputs/sightings.db",
        help="SQLite database path (default: outputs/sightings.db).",
    )
    query_p.add_argument(
        "--show-timeline",
        action="store_true",
        help="Include chronological track milestone timeline in text output.",
    )
    query_p.add_argument(
        "--list-all",
        action="store_true",
        help="Print every frame-level sighting in addition to the summary.",
    )
    query_p.add_argument(
        "--json",
        action="store_true",
        help="Emit structured JSON report instead of formatted text.",
    )

    # 4. compare-tracks subcommand
    comp_p = subparsers.add_parser(
        "compare-tracks",
        help="Investigate tracking gaps and compare visual/spatiotemporal evidence across candidate tracks.",
    )
    comp_p.add_argument(
        "--camera-id",
        default="office_cam_01",
        help="Camera ID to analyze (default: office_cam_01).",
    )
    comp_p.add_argument(
        "--video",
        default="laptop_test.mp4",
        help="Source video path to analyze (default: laptop_test.mp4).",
    )
    comp_p.add_argument(
        "--class-name",
        default="laptop",
        help="Target asset class name (default: laptop).",
    )
    comp_p.add_argument(
        "--confirmed-conf",
        type=float,
        default=0.50,
        help="Detector confidence threshold for strongly supported tracks (default: 0.50).",
    )
    comp_p.add_argument(
        "--min-track-frames",
        type=int,
        default=3,
        help="Minimum high-confidence frames for strongly supported tracks (default: 3).",
    )
    comp_p.add_argument(
        "--db-path",
        default="outputs/sightings.db",
        help="SQLite database path (default: outputs/sightings.db).",
    )
    comp_p.add_argument(
        "--json",
        action="store_true",
        help="Emit structured JSON comparison report.",
    )

    # 5. index-embeddings subcommand
    idx_p = subparsers.add_parser(
        "index-embeddings",
        help="Extract OpenCLIP crop and context embeddings for SQLite sightings and persist in ChromaDB.",
    )
    idx_p.add_argument(
        "--db-path",
        default="outputs/sightings.db",
        help="SQLite database path (default: outputs/sightings.db).",
    )
    idx_p.add_argument(
        "--chroma-dir",
        default="outputs/chroma_db",
        help="ChromaDB persistence directory (default: outputs/chroma_db).",
    )
    idx_p.add_argument(
        "--collection",
        default="sighting_visual_embeddings",
        help="ChromaDB collection name (default: sighting_visual_embeddings).",
    )
    idx_p.add_argument(
        "--clip-model",
        default="ViT-B-32",
        help="OpenCLIP model architecture (default: ViT-B-32).",
    )
    idx_p.add_argument(
        "--clip-pretrained",
        default="laion2b_s34b_b79k",
        help="OpenCLIP pretrained weights tag (default: laion2b_s34b_b79k).",
    )
    idx_p.add_argument(
        "--device",
        default=None,
        help="Compute device override (default: cuda if available, else cpu).",
    )
    idx_p.add_argument(
        "--camera-id",
        nargs="*",
        default=None,
        help="Optional camera ID(s) to index (omit to index all cameras).",
    )
    idx_p.add_argument(
        "--class-name",
        default=None,
        help="Optional detected object class filter (e.g. laptop, person).",
    )
    idx_p.add_argument(
        "--rebuild",
        action="store_true",
        help="Delete and recreate the ChromaDB collection before indexing.",
    )
    idx_p.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for OpenCLIP GPU image encoding (default: 32).",
    )
    idx_p.add_argument(
        "--json",
        action="store_true",
        help="Emit structured JSON indexing summary.",
    )

    # 6. search subcommand
    search_p = subparsers.add_parser(
        "search",
        help="Search indexed camera sightings using natural-language descriptions via OpenCLIP + ChromaDB.",
    )
    search_p.add_argument(
        "--query",
        "-q",
        required=True,
        help="Natural-language description to search for (e.g. 'black laptop', 'person holding a black laptop').",
    )
    search_p.add_argument(
        "--top-k",
        "-k",
        type=int,
        default=10,
        help="Maximum number of ranked sighting matches to return (default: 10).",
    )
    search_p.add_argument(
        "--camera-id",
        nargs="*",
        default=None,
        help="Optional camera ID(s) to filter (omit to search all cameras; accepts space- or comma-separated IDs).",
    )
    search_p.add_argument(
        "--class-name",
        default=None,
        help="Optional detected object class filter (e.g. laptop, person; omit to search all indexed classes).",
    )
    search_p.add_argument(
        "--variant",
        choices=["crop", "context"],
        default=None,
        help="Optional embedding variant filter ('crop' or 'context'; omit to search both).",
    )
    search_p.add_argument(
        "--min-similarity",
        type=float,
        default=None,
        help="Optional minimum OpenCLIP cosine similarity threshold.",
    )
    search_p.add_argument(
        "--confirmed-conf",
        type=float,
        default=0.50,
        help="Detector confidence threshold for strongly supported track classification (default: 0.50).",
    )
    search_p.add_argument(
        "--min-track-frames",
        type=int,
        default=3,
        help="Minimum high-confidence frames for strongly supported track classification (default: 3).",
    )
    search_p.add_argument(
        "--db-path",
        default="outputs/sightings.db",
        help="SQLite database path (default: outputs/sightings.db).",
    )
    search_p.add_argument(
        "--chroma-dir",
        default="outputs/chroma_db",
        help="ChromaDB persistence directory (default: outputs/chroma_db).",
    )
    search_p.add_argument(
        "--collection",
        default="sighting_visual_embeddings",
        help="ChromaDB collection name (default: sighting_visual_embeddings).",
    )
    search_p.add_argument(
        "--clip-model",
        default="ViT-B-32",
        help="OpenCLIP model architecture (default: ViT-B-32).",
    )
    search_p.add_argument(
        "--clip-pretrained",
        default="laion2b_s34b_b79k",
        help="OpenCLIP pretrained weights tag (default: laion2b_s34b_b79k).",
    )
    search_p.add_argument(
        "--device",
        default=None,
        help="Compute device override (default: cuda if available, else cpu).",
    )
    search_p.add_argument(
        "--json",
        action="store_true",
        help="Emit structured JSON search report.",
    )

    # 7. index-stats subcommand
    stats_p = subparsers.add_parser(
        "index-stats",
        help="Report ChromaDB visual embedding index statistics and SQLite synchronization status.",
    )
    stats_p.add_argument(
        "--db-path",
        default="outputs/sightings.db",
        help="SQLite database path (default: outputs/sightings.db).",
    )
    stats_p.add_argument(
        "--chroma-dir",
        default="outputs/chroma_db",
        help="ChromaDB persistence directory (default: outputs/chroma_db).",
    )
    stats_p.add_argument(
        "--collection",
        default="sighting_visual_embeddings",
        help="ChromaDB collection name (default: sighting_visual_embeddings).",
    )
    stats_p.add_argument(
        "--json",
        action="store_true",
        help="Emit structured JSON statistics report.",
    )

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "ingest":
        from src.ingest import ingest_video, resolve_device

        store = SightingStore(db_path=args.db_path, jsonl_path=args.jsonl_path)
        device = resolve_device(args.device)
        print(
            f"[INGEST] Video='{args.video}' | Camera='{args.camera_id}' | "
            f"Model='{args.model}' | Tracker='{args.tracker}' | Device={device}"
        )
        metadata, sightings = ingest_video(
            video_path=args.video,
            camera_id=args.camera_id,
            store=store,
            model_path=args.model,
            tracker=args.tracker,
            imgsz=args.imgsz,
            conf_threshold=args.conf,
            confirmed_conf_threshold=args.confirmed_conf,
            classes=args.classes,
            device=args.device,
            output_dir=args.output_dir,
            save_annotated_video=not args.no_annotated_video,
            keyframe_interval=args.keyframe_interval,
            recording_start_utc=args.start_time,
            recording_timezone=args.timezone,
            time_offset_seconds=args.time_offset,
            source_identity=args.source_identity,
        )
        by_class: dict[str, int] = {}
        for s in sightings:
            by_class[s.class_name] = by_class.get(s.class_name, 0) + 1

        print(
            f"[INGEST COMPLETE] Run ID={metadata.run_id} | Camera='{metadata.camera_id}' | "
            f"Source='{metadata.source_video}' (id={metadata.resolved_source_identity}) | "
            f"Frames={metadata.frame_count} @ {metadata.fps:.2f} FPS ({metadata.duration_seconds:.2f}s)"
        )
        print(f"  Total Sightings Persisted : {len(sightings)} -> {by_class}")
        print(f"  SQLite Store              : {args.db_path}")
        print(f"  JSONL Export              : {args.jsonl_path}")
        print(f"  Annotated Video Reference : {metadata.annotated_video_path}")
        return 0

    if args.command == "ingest-batch":
        from src.ingest import ingest_multiple_videos

        manifest_path = Path(args.manifest)
        if not manifest_path.exists():
            raise FileNotFoundError(f"Batch manifest not found: {manifest_path}")
        sources = json.loads(manifest_path.read_text(encoding="utf-8"))
        store = SightingStore(db_path=args.db_path, jsonl_path=args.jsonl_path)
        batch_results = ingest_multiple_videos(
            sources=sources,
            store=store,
            model_path=args.model,
            tracker=args.tracker,
            imgsz=args.imgsz,
            conf_threshold=args.conf,
            confirmed_conf_threshold=args.confirmed_conf,
            classes=args.classes,
            device=args.device,
            output_dir=args.output_dir,
            save_annotated_video=not args.no_annotated_video,
        )
        print(f"[BATCH INGEST COMPLETE] Processed {len(batch_results)} camera recording(s).")
        for meta, recs in batch_results:
            print(
                f"  - Run #{meta.run_id} | Camera='{meta.camera_id}' | Video='{meta.source_video}' "
                f"(id={meta.resolved_source_identity}) | Sightings={len(recs)}"
            )
        return 0

    if args.command == "query":
        store = SightingStore(db_path=args.db_path, jsonl_path=None)
        cam_list = _parse_camera_list(args.camera_id)
        single_cam = cam_list[0] if len(cam_list) == 1 else None
        multi_cams = cam_list if len(cam_list) > 1 else None

        report = query_asset_sightings(
            store=store,
            query_class=args.class_name,
            camera_id=single_cam,
            camera_ids=multi_cams,
            source_video=args.source_video,
            confirmed_conf_threshold=args.confirmed_conf,
            confirmed_min_track_frames=args.min_track_frames,
        )
        if args.json:
            print(json.dumps(report.to_dict(), indent=2))
            return 0

        print(format_retrieval_report_text(report, show_timeline=args.show_timeline))

        if args.list_all:
            sightings = store.query_sightings(
                class_name=args.class_name,
                camera_id=single_cam,
                camera_ids=multi_cams,
                source_video=args.source_video,
            )
            supported_keys = {
                t.scoped_track_key
                for t in report.track_summaries
                if t.status == "STRONGLY_SUPPORTED"
            }
            print("\nCHRONOLOGICAL FRAME-BY-FRAME SIGHTINGS (WITHIN CAMERA):")
            print("-" * 88)
            for s in sightings:
                tier = (
                    "STRONGLY_SUPPORTED"
                    if (s.scoped_track_key in supported_keys and s.confidence >= args.confirmed_conf)
                    else "UNCERTAIN_CANDIDATE"
                )
                print(
                    f"  Frame {s.frame_index:4d} | {format_video_timestamp(s.timestamp_seconds)} | "
                    f"{s.scoped_track_key:46s} | det_conf={s.confidence:.4f} [{tier}] | "
                    f"bbox={[round(v, 1) for v in s.bbox_xyxy]} | "
                    f"ev={s.evidence_frame_path or '-'}"
                )
        return 0

    if args.command == "compare-tracks":
        store = SightingStore(db_path=args.db_path, jsonl_path=None)
        analysis = analyze_tracking_gap_and_candidates(
            store=store,
            camera_id=args.camera_id,
            source_video=args.video,
            query_class=args.class_name,
            confirmed_conf_threshold=args.confirmed_conf,
            confirmed_min_track_frames=args.min_track_frames,
        )
        if args.json:
            print(json.dumps(analysis, indent=2))
            return 0
        print(format_comparison_report_text(analysis))
        return 0

    if args.command == "index-embeddings":
        from src.visual_search import ChromaVisualIndex, OpenCLIPEmbedder

        store = SightingStore(db_path=args.db_path, jsonl_path=None)
        index = ChromaVisualIndex(persist_dir=args.chroma_dir, collection_name=args.collection)
        encoder = OpenCLIPEmbedder.get_instance(
            model_name=args.clip_model,
            pretrained=args.clip_pretrained,
            device=args.device,
        )
        cam_list = _parse_camera_list(args.camera_id)
        summary = index.index_sightings_from_store(
            store=store,
            encoder=encoder,
            camera_ids=cam_list if cam_list else None,
            class_name=args.class_name,
            rebuild=args.rebuild,
            batch_size=args.batch_size,
        )
        if args.json:
            print(json.dumps(summary, indent=2))
            return 0

        print(
            f"[INDEX-EMBEDDINGS COMPLETE] Model='{summary['embedding_model_id']}' "
            f"(dim={summary['embedding_dim']}, device={summary['device']})"
        )
        print(f"  ChromaDB Directory           : {summary['persist_dir']} ({summary['collection_name']})")
        print(f"  SQLite Sightings Scanned     : {summary['total_sqlite_sightings_scanned']}")
        print(f"  Sightings With Evidence      : {summary['sightings_with_evidence_paths']}")
        print(f"  Sightings Indexed            : {summary['sightings_indexed']}")
        print(
            f"  Embeddings Upserted          : {summary['total_embeddings_upserted']} "
            f"(crop={summary['crop_embeddings_indexed']}, context={summary['context_embeddings_indexed']})"
        )
        print(f"  Missing/Corrupt Files Skipped: {summary['missing_or_corrupt_evidence_files']}")
        print(f"  Stale Embeddings Pruned      : {summary['stale_embeddings_pruned']}")
        print(f"  Total Collection Embeddings  : {summary['collection_total_embeddings']}")
        return 0

    if args.command == "search":
        from src.visual_search import (
            ChromaVisualIndex,
            OpenCLIPEmbedder,
            format_visual_search_report_text,
        )

        store = SightingStore(db_path=args.db_path, jsonl_path=None)
        index = ChromaVisualIndex(persist_dir=args.chroma_dir, collection_name=args.collection)
        encoder = OpenCLIPEmbedder.get_instance(
            model_name=args.clip_model,
            pretrained=args.clip_pretrained,
            device=args.device,
        )
        cam_list = _parse_camera_list(args.camera_id)
        report = index.search_by_description(
            query_text=args.query,
            store=store,
            encoder=encoder,
            top_k=args.top_k,
            camera_ids=cam_list if cam_list else None,
            class_name=args.class_name,
            variant=args.variant,
            min_similarity=args.min_similarity,
            confirmed_conf_threshold=args.confirmed_conf,
            confirmed_min_track_frames=args.min_track_frames,
        )
        if args.json:
            print(json.dumps(report.to_dict(), indent=2))
            return 0
        print(format_visual_search_report_text(report))
        return 0

    if args.command == "index-stats":
        from src.visual_search import ChromaVisualIndex, format_index_stats_text

        store = SightingStore(db_path=args.db_path, jsonl_path=None)
        index = ChromaVisualIndex(persist_dir=args.chroma_dir, collection_name=args.collection)
        stats = index.get_index_stats(store=store)
        if args.json:
            print(json.dumps(stats, indent=2))
            return 0
        print(format_index_stats_text(stats))
        return 0

    return 1


if __name__ == "__main__":
    sys.exit(main())

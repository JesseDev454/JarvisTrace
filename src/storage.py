"""SQLite and JSONL persistence for single- and multi-camera video ingestion and sightings."""

from __future__ import annotations

from contextlib import closing
import json
import sqlite3
from pathlib import Path
from typing import Iterable, Optional, Sequence

from src.models import (
    SightingRecord,
    VideoMetadata,
    compute_aligned_timestamp,
    compute_source_identity,
    normalize_source_video,
    parse_recording_start_datetime,
)


class SightingStore:
    """Persistent store backed by SQLite (and optional JSONL export)."""

    def __init__(
        self,
        db_path: str | Path = "outputs/sightings.db",
        jsonl_path: Optional[str | Path] = "outputs/sightings.jsonl",
    ) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.jsonl_path = Path(jsonl_path) if jsonl_path is not None else None
        if self.jsonl_path is not None:
            self.jsonl_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON;")
        return conn

    def _init_db(self) -> None:
        with closing(self._connect()) as conn, conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS ingestion_runs (
                    run_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    camera_id TEXT NOT NULL,
                    source_video TEXT NOT NULL,
                    source_identity TEXT,
                    fps REAL NOT NULL,
                    frame_count INTEGER NOT NULL,
                    width INTEGER NOT NULL,
                    height INTEGER NOT NULL,
                    duration_seconds REAL NOT NULL,
                    model_name TEXT NOT NULL,
                    tracker_name TEXT NOT NULL,
                    conf_threshold REAL NOT NULL,
                    recording_start_utc TEXT,
                    recording_timezone TEXT,
                    time_offset_seconds REAL,
                    annotated_video_path TEXT,
                    ingested_at_utc TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS sightings (
                    sighting_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_id INTEGER NOT NULL,
                    camera_id TEXT NOT NULL,
                    source_video TEXT NOT NULL,
                    frame_index INTEGER NOT NULL,
                    timestamp_seconds REAL NOT NULL,
                    class_id INTEGER NOT NULL,
                    class_name TEXT NOT NULL,
                    local_track_id INTEGER,
                    confidence REAL NOT NULL,
                    x1 REAL NOT NULL,
                    y1 REAL NOT NULL,
                    x2 REAL NOT NULL,
                    y2 REAL NOT NULL,
                    evidence_frame_path TEXT,
                    evidence_crop_path TEXT,
                    evidence_clip_path TEXT,
                    FOREIGN KEY (run_id) REFERENCES ingestion_runs(run_id) ON DELETE CASCADE
                );

                CREATE INDEX IF NOT EXISTS idx_sightings_class_time
                    ON sightings(class_name, camera_id, timestamp_seconds, frame_index);

                CREATE INDEX IF NOT EXISTS idx_sightings_run
                    ON sightings(run_id);
                """
            )

            # Backward-compatible schema migration for existing SQLite databases
            existing_cols = {
                str(r["name"])
                for r in conn.execute("PRAGMA table_info(ingestion_runs);").fetchall()
            }
            migrations = [
                ("source_identity", "TEXT"),
                ("recording_start_utc", "TEXT"),
                ("recording_timezone", "TEXT"),
                ("time_offset_seconds", "REAL"),
            ]
            for col_name, col_type in migrations:
                if col_name not in existing_cols:
                    conn.execute(f"ALTER TABLE ingestion_runs ADD COLUMN {col_name} {col_type};")

    def create_ingestion_run(
        self,
        metadata: VideoMetadata,
        replace_existing_for_source: bool = True,
    ) -> int:
        """Register a video ingestion run.

        When `replace_existing_for_source=True` (default), any prior runs for the same
        `(camera_id, normalized_source_video)` or `(camera_id, source_identity)` are removed
        first so repeated ingestion never creates duplicate sighting records.
        """
        norm_video = normalize_source_video(metadata.source_video)
        src_identity = metadata.source_identity or compute_source_identity(metadata.source_video)
        _, norm_start_iso, norm_tz = parse_recording_start_datetime(
            metadata.recording_start_utc,
            metadata.recording_timezone,
        )

        with closing(self._connect()) as conn, conn:
            if replace_existing_for_source:
                conn.execute(
                    """
                    DELETE FROM ingestion_runs
                    WHERE camera_id = ? AND (source_video = ? OR (source_identity IS NOT NULL AND source_identity = ?))
                    """,
                    (metadata.camera_id, norm_video, src_identity),
                )
            cursor = conn.execute(
                """
                INSERT INTO ingestion_runs (
                    camera_id, source_video, source_identity, fps, frame_count, width, height,
                    duration_seconds, model_name, tracker_name, conf_threshold,
                    recording_start_utc, recording_timezone, time_offset_seconds,
                    annotated_video_path, ingested_at_utc
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    metadata.camera_id,
                    norm_video,
                    src_identity,
                    metadata.fps,
                    metadata.frame_count,
                    metadata.width,
                    metadata.height,
                    metadata.duration_seconds,
                    metadata.model_name,
                    metadata.tracker_name,
                    metadata.conf_threshold,
                    norm_start_iso,
                    norm_tz,
                    float(metadata.time_offset_seconds) if metadata.time_offset_seconds is not None else None,
                    metadata.annotated_video_path,
                    metadata.ingested_at_utc,
                ),
            )
            run_id = int(cursor.lastrowid)
        return run_id

    def insert_sightings(self, sightings: Iterable[SightingRecord]) -> list[SightingRecord]:
        """Insert a batch of SightingRecords into SQLite (deduplicating within-run duplicates) and sync JSONL."""
        persisted: list[SightingRecord] = []
        seen_signatures: set[tuple[object, ...]] = set()

        with closing(self._connect()) as conn, conn:
            run_sync_map: dict[int, tuple[Optional[str], Optional[str], Optional[float]]] = {}
            for r in conn.execute(
                "SELECT run_id, recording_start_utc, recording_timezone, time_offset_seconds FROM ingestion_runs"
            ).fetchall():
                run_sync_map[int(r["run_id"])] = (
                    r["recording_start_utc"],
                    r["recording_timezone"],
                    float(r["time_offset_seconds"]) if r["time_offset_seconds"] is not None else None,
                )

            existing_rows = conn.execute(
                "SELECT run_id, camera_id, source_video, frame_index, class_id, local_track_id, x1, y1, x2, y2 FROM sightings"
            ).fetchall()
            for r in existing_rows:
                seen_signatures.add(
                    (
                        int(r["run_id"]),
                        str(r["camera_id"]),
                        normalize_source_video(str(r["source_video"])),
                        int(r["frame_index"]),
                        int(r["class_id"]),
                        int(r["local_track_id"]) if r["local_track_id"] is not None else None,
                        round(float(r["x1"]), 2),
                        round(float(r["y1"]), 2),
                        round(float(r["x2"]), 2),
                        round(float(r["y2"]), 2),
                    )
                )

            for s in sightings:
                if s.run_id is None:
                    raise ValueError("SightingRecord.run_id must be set before inserting into SQLite.")
                norm_video = normalize_source_video(s.source_video)
                s.source_video = norm_video
                x1, y1, x2, y2 = [round(float(v), 2) for v in s.bbox_xyxy]
                sig = (
                    int(s.run_id),
                    s.camera_id,
                    norm_video,
                    int(s.frame_index),
                    int(s.class_id),
                    s.local_track_id,
                    x1,
                    y1,
                    x2,
                    y2,
                )
                if sig in seen_signatures:
                    continue
                seen_signatures.add(sig)

                sync_info = run_sync_map.get(int(s.run_id))
                if sync_info is not None:
                    aligned_iso, aligned_epoch, aligned_sync = compute_aligned_timestamp(
                        s.timestamp_seconds,
                        recording_start_utc=sync_info[0],
                        recording_timezone=sync_info[1],
                        time_offset_seconds=sync_info[2],
                    )
                    s.aligned_wall_clock_utc = aligned_iso
                    s.aligned_epoch_seconds = aligned_epoch
                    s.aligned_sync_seconds = aligned_sync

                cur = conn.execute(
                    """
                    INSERT INTO sightings (
                        run_id, camera_id, source_video, frame_index, timestamp_seconds,
                        class_id, class_name, local_track_id, confidence,
                        x1, y1, x2, y2,
                        evidence_frame_path, evidence_crop_path, evidence_clip_path
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        s.run_id,
                        s.camera_id,
                        norm_video,
                        s.frame_index,
                        s.timestamp_seconds,
                        s.class_id,
                        s.class_name.lower(),
                        s.local_track_id,
                        s.confidence,
                        x1,
                        y1,
                        x2,
                        y2,
                        s.evidence_frame_path,
                        s.evidence_crop_path,
                        s.evidence_clip_path,
                    ),
                )
                s.sighting_id = int(cur.lastrowid)
                persisted.append(s)

        if self.jsonl_path is not None:
            self.export_jsonl(self.jsonl_path)
        return persisted

    def export_jsonl(self, path: Optional[str | Path] = None) -> Path:
        """Write all currently stored sightings in chronological order to a JSONL file."""
        target = Path(path) if path is not None else self.jsonl_path
        if target is None:
            raise ValueError("No JSONL path specified.")
        target.parent.mkdir(parents=True, exist_ok=True)
        records = self.query_sightings()
        with target.open("w", encoding="utf-8") as fh:
            for record in records:
                fh.write(json.dumps(record.to_dict(), ensure_ascii=False) + "\n")
        return target

    @staticmethod
    def load_jsonl(path: str | Path) -> list[SightingRecord]:
        """Load SightingRecords from a JSONL file."""
        target = Path(path)
        if not target.exists():
            return []
        records: list[SightingRecord] = []
        with target.open("r", encoding="utf-8") as fh:
            for line in fh:
                stripped = line.strip()
                if stripped:
                    records.append(SightingRecord.from_dict(json.loads(stripped)))
        return records

    def get_ingestion_runs(
        self,
        camera_id: Optional[str] = None,
        camera_ids: Optional[Sequence[str]] = None,
        source_video: Optional[str] = None,
        run_id: Optional[int] = None,
    ) -> list[VideoMetadata]:
        """Return stored video ingestion metadata for all or selected cameras."""
        clauses: list[str] = []
        params: list[object] = []

        selected_cameras = list(camera_ids) if camera_ids else ([camera_id] if camera_id else [])
        if len(selected_cameras) == 1:
            clauses.append("camera_id = ?")
            params.append(selected_cameras[0])
        elif len(selected_cameras) > 1:
            placeholders = ", ".join("?" for _ in selected_cameras)
            clauses.append(f"camera_id IN ({placeholders})")
            params.extend(selected_cameras)

        if source_video:
            clauses.append("source_video = ?")
            params.append(normalize_source_video(source_video))
        if run_id is not None:
            clauses.append("run_id = ?")
            params.append(int(run_id))

        where_sql = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        sql = f"""
            SELECT run_id, camera_id, source_video, source_identity, fps, frame_count, width, height,
                   duration_seconds, model_name, tracker_name, conf_threshold,
                   recording_start_utc, recording_timezone, time_offset_seconds,
                   annotated_video_path, ingested_at_utc
            FROM ingestion_runs
            {where_sql}
            ORDER BY run_id ASC
        """
        with closing(self._connect()) as conn:
            rows = conn.execute(sql, params).fetchall()
        return [
            VideoMetadata(
                run_id=int(r["run_id"]),
                camera_id=str(r["camera_id"]),
                source_video=normalize_source_video(str(r["source_video"])),
                source_identity=(
                    str(r["source_identity"])
                    if r["source_identity"]
                    else compute_source_identity(str(r["source_video"]))
                ),
                fps=float(r["fps"]),
                frame_count=int(r["frame_count"]),
                width=int(r["width"]),
                height=int(r["height"]),
                duration_seconds=float(r["duration_seconds"]),
                model_name=str(r["model_name"]),
                tracker_name=str(r["tracker_name"]),
                conf_threshold=float(r["conf_threshold"]),
                recording_start_utc=r["recording_start_utc"],
                recording_timezone=r["recording_timezone"],
                time_offset_seconds=(
                    float(r["time_offset_seconds"]) if r["time_offset_seconds"] is not None else None
                ),
                annotated_video_path=r["annotated_video_path"],
                ingested_at_utc=str(r["ingested_at_utc"]),
            )
            for r in rows
        ]

    def query_sightings(
        self,
        class_name: Optional[str] = None,
        camera_id: Optional[str] = None,
        camera_ids: Optional[Sequence[str]] = None,
        source_video: Optional[str] = None,
        min_confidence: Optional[float] = None,
        local_track_id: Optional[int] = None,
        run_id: Optional[int] = None,
    ) -> list[SightingRecord]:
        """Query sightings ordered by (camera_id, source_video, run_id, timestamp_seconds, frame_index)."""
        clauses: list[str] = []
        params: list[object] = []

        if class_name is not None:
            clauses.append("LOWER(s.class_name) = LOWER(?)")
            params.append(class_name)

        selected_cameras = list(camera_ids) if camera_ids else ([camera_id] if camera_id else [])
        if len(selected_cameras) == 1:
            clauses.append("s.camera_id = ?")
            params.append(selected_cameras[0])
        elif len(selected_cameras) > 1:
            placeholders = ", ".join("?" for _ in selected_cameras)
            clauses.append(f"s.camera_id IN ({placeholders})")
            params.extend(selected_cameras)

        if source_video is not None:
            clauses.append("s.source_video = ?")
            params.append(normalize_source_video(source_video))
        if min_confidence is not None:
            clauses.append("s.confidence >= ?")
            params.append(float(min_confidence))
        if local_track_id is not None:
            clauses.append("s.local_track_id = ?")
            params.append(int(local_track_id))
        if run_id is not None:
            clauses.append("s.run_id = ?")
            params.append(int(run_id))

        where_sql = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        sql = f"""
            SELECT s.sighting_id, s.run_id, s.camera_id, s.source_video, s.frame_index,
                   s.timestamp_seconds, s.class_id, s.class_name, s.local_track_id,
                   s.confidence, s.x1, s.y1, s.x2, s.y2,
                   s.evidence_frame_path, s.evidence_crop_path, s.evidence_clip_path,
                   r.recording_start_utc, r.recording_timezone, r.time_offset_seconds
            FROM sightings s
            LEFT JOIN ingestion_runs r ON s.run_id = r.run_id
            {where_sql}
            ORDER BY s.camera_id ASC, s.source_video ASC, s.run_id ASC,
                     s.timestamp_seconds ASC, s.frame_index ASC, s.confidence DESC, s.sighting_id ASC
        """
        with closing(self._connect()) as conn:
            rows = conn.execute(sql, params).fetchall()

        results: list[SightingRecord] = []
        for r in rows:
            ts = float(r["timestamp_seconds"])
            aligned_iso, aligned_epoch, aligned_sync = compute_aligned_timestamp(
                ts,
                recording_start_utc=r["recording_start_utc"],
                recording_timezone=r["recording_timezone"],
                time_offset_seconds=(
                    float(r["time_offset_seconds"]) if r["time_offset_seconds"] is not None else None
                ),
            )
            results.append(
                SightingRecord(
                    sighting_id=int(r["sighting_id"]),
                    run_id=int(r["run_id"]),
                    camera_id=str(r["camera_id"]),
                    source_video=normalize_source_video(str(r["source_video"])),
                    frame_index=int(r["frame_index"]),
                    timestamp_seconds=ts,
                    aligned_wall_clock_utc=aligned_iso,
                    aligned_epoch_seconds=aligned_epoch,
                    aligned_sync_seconds=aligned_sync,
                    class_id=int(r["class_id"]),
                    class_name=str(r["class_name"]),
                    local_track_id=int(r["local_track_id"]) if r["local_track_id"] is not None else None,
                    confidence=float(r["confidence"]),
                    bbox_xyxy=(float(r["x1"]), float(r["y1"]), float(r["x2"]), float(r["y2"])),
                    evidence_frame_path=r["evidence_frame_path"],
                    evidence_crop_path=r["evidence_crop_path"],
                    evidence_clip_path=r["evidence_clip_path"],
                )
            )
        return results

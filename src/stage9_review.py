"""Local, human-operated Stage 9 review. Start with python -m src.stage9_review."""
from __future__ import annotations

import argparse
from contextlib import closing
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import sqlite3
from typing import Literal

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, Response
from pydantic import BaseModel, Field


class HumanObject(BaseModel):
    object_id: str = Field(min_length=1, max_length=100)
    bbox_xyxy: tuple[float, float, float, float]
    surface: str = Field(default='', max_length=100)


class Annotation(BaseModel):
    reviewer: str = Field(min_length=1, max_length=100)
    visibility: Literal['present', 'absent', 'uncertain']
    motion: Literal['unknown', 'no_observed_movement', 'movement_visible']
    objects: list[HumanObject] = Field(default_factory=list, max_length=50)
    notes: str = Field(default='', max_length=2000)
    expected_revision: int = Field(ge=0)


def create_app(report_path: Path, review_db: Path, *, default_reviewer: str = '') -> FastAPI:
    """Bind one immutable prediction report and a separate append-only review store."""
    report_path, review_db = Path(report_path).resolve(), Path(review_db).resolve()
    report = json.loads(report_path.read_text(encoding='utf-8'))
    if report.get('report_format') != 'group8-stage9-v1':
        raise ValueError('Unsupported Stage 9 report')
    if report.get('target_class_id') != 24:
        raise ValueError('This review tool currently supports backpack reports (class 24) only')
    video_path = Path(report['provenance']['video_path'])
    with video_path.open('rb') as f:
        if hashlib.file_digest(f, 'sha256').hexdigest() != report['provenance']['video_sha256']:
            raise ValueError('Source video hash differs from report')
    info = report['video']
    count, width, height = info['frame_count'], info['width'], info['height']
    rows = [json.loads(line) for line in report_path.with_name('observations.jsonl').read_text().splitlines() if line.strip()]
    scope = report['provenance']['scope_id']
    if len(rows) != report['analysis']['observation_count'] or any(r.get('scope_id') != scope for r in rows):
        raise ValueError('Observation file does not match report scope/count')
    predictions = {}
    for row in rows:
        predictions.setdefault(row['frame_index'], []).append(row)
    review_db.parent.mkdir(parents=True, exist_ok=True)
    with closing(sqlite3.connect(review_db)) as conn, conn:
        conn.execute('CREATE TABLE IF NOT EXISTS metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL)')
        existing = conn.execute("SELECT value FROM metadata WHERE key='scope'").fetchone()
        if existing and existing[0] != scope:
            raise ValueError('Review database belongs to a different report scope')
        conn.execute("INSERT OR IGNORE INTO metadata VALUES ('scope', ?)", (scope,))
        conn.execute('CREATE TABLE IF NOT EXISTS reviews (frame INTEGER NOT NULL, revision INTEGER NOT NULL, payload TEXT NOT NULL, PRIMARY KEY(frame,revision))')
    app = FastAPI(title='Stage 9 Human Review', docs_url=None, redoc_url=None)

    def frame_check(index):
        if not 0 <= index < count:
            raise HTTPException(404, 'Frame outside recording')

    def latest(index):
        with closing(sqlite3.connect(review_db)) as conn:
            row = conn.execute('SELECT payload FROM reviews WHERE frame=? ORDER BY revision DESC LIMIT 1', (index,)).fetchone()
        return json.loads(row[0]) if row else None

    @app.get('/', response_class=HTMLResponse)
    def page():
        return Path(__file__).with_name('stage9_review.html').read_text(encoding='utf-8')

    @app.get('/api/info')
    def metadata():
        return {**info, 'camera_id': report['camera_id'], 'video_id': report['video_id'],
                'scope_id': scope, 'default_reviewer': default_reviewer}

    @app.get('/api/evaluation')
    def evaluation():
        from src.stage9_evaluate import evaluate_report
        return evaluate_report(report_path, review_db)

    @app.get('/api/frames/{index}')
    def frame(index: int):
        frame_check(index)
        return {'frame_index': index, 'timestamp_seconds': index/info['fps'],
                'predictions': predictions.get(index, []), 'annotation': latest(index)}

    @app.get('/api/frames/{index}/image')
    def image(index: int):
        import cv2
        frame_check(index)
        cap = cv2.VideoCapture(str(video_path))
        try:
            cap.set(cv2.CAP_PROP_POS_FRAMES, index)
            ok, pixels = cap.read()
            if not ok:
                raise HTTPException(422, 'Frame could not be decoded')
            ok, jpg = cv2.imencode('.jpg', pixels)
            if not ok:
                raise HTTPException(500, 'Frame could not be encoded')
            return Response(jpg.tobytes(), media_type='image/jpeg', headers={'Cache-Control': 'no-store'})
        finally:
            cap.release()

    @app.put('/api/frames/{index}/annotation')
    def save(index: int, annotation: Annotation, request: Request):
        frame_check(index)
        if (request.headers.get('X-Review-Client') != 'stage9' or
                (request.headers.get('origin') and request.headers['origin'] != str(request.base_url).rstrip('/'))):
            raise HTTPException(403, 'Same-origin review client required')
        if (not annotation.reviewer.strip() or
                (annotation.visibility == 'present' and not annotation.objects) or
                (annotation.visibility == 'absent' and annotation.objects)):
            raise HTTPException(422, 'Set a reviewer; present needs boxes and absent needs no boxes')
        ids = set()
        for obj in annotation.objects:
            x1, y1, x2, y2 = obj.bbox_xyxy
            if (not obj.object_id.strip() or obj.object_id in ids or
                    not all(math.isfinite(v) for v in obj.bbox_xyxy) or
                    not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height)):
                raise HTTPException(422, 'Each object needs a unique ID and a finite box within the frame')
            ids.add(obj.object_id)
        with closing(sqlite3.connect(review_db)) as conn, conn:
            conn.execute('BEGIN IMMEDIATE')
            revision = conn.execute('SELECT coalesce(max(revision),0) FROM reviews WHERE frame=?', (index,)).fetchone()[0]
            if revision != annotation.expected_revision:
                raise HTTPException(409, 'Annotation changed; reload frame before saving')
            saved = annotation.model_dump(exclude={'expected_revision'}) | {
                'frame_index': index, 'timestamp_seconds': index/info['fps'], 'revision': revision+1,
                'reviewed_at_utc': datetime.now(timezone.utc).isoformat(), 'scope_id': scope,
                'annotation_source': 'human_review'}
            conn.execute('INSERT INTO reviews VALUES (?,?,?)', (index, revision+1, json.dumps(saved, allow_nan=False)))
        return saved

    @app.get('/api/export')
    def export():
        with closing(sqlite3.connect(review_db)) as conn:
            history = [json.loads(r[0]) for r in conn.execute('SELECT payload FROM reviews ORDER BY frame,revision')]
        return {'format': 'group8-stage9-human-review-v1', 'scope_id': scope,
                'video_sha256': report['provenance']['video_sha256'],
                'history': history, 'note': 'Annotation history; annotation_source distinguishes human review from AI-assisted drafts.'}

    return app


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report', required=True, type=Path)
    parser.add_argument('--review-db', type=Path)
    parser.add_argument('--port', type=int, default=8765)
    parser.add_argument('--reviewer', default='', help='Prefill the reviewer name; creates no annotations')
    args = parser.parse_args()
    app = create_app(args.report, args.review_db or args.report.parent/'human_review.sqlite3',
                     default_reviewer=args.reviewer)
    import uvicorn
    uvicorn.run(app, host='127.0.0.1', port=args.port)


if __name__ == '__main__':
    main()

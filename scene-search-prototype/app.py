"""Minimal, local-only scene description retrieval prototype for Group 8 FYP.

No video or VLM ingestion yet; inputs are manually described scene records.
"""
from functools import lru_cache
from pathlib import Path
import os

import psycopg
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Query
from pgvector.psycopg import register_vector
from pydantic import BaseModel, Field, model_validator
from psycopg.rows import dict_row
from sentence_transformers import SentenceTransformer

load_dotenv(Path(__file__).with_name('.env'))

EMBEDDING_MODEL = 'sentence-transformers/all-MiniLM-L6-v2'

app = FastAPI(title='Group 8 Scene Search — Prototype', version='0.1.0')


def open_database():
    password = os.getenv('DB_PASSWORD')
    if not password:
        raise RuntimeError('DB_PASSWORD missing. Create .env from .env.example first.')

    conn = psycopg.connect(
        host=os.getenv('DB_HOST', '127.0.0.1'),
        port=int(os.getenv('DB_PORT', '15432')),
        dbname=os.getenv('DB_NAME', 'group8_fyp'),
        user=os.getenv('DB_USER', 'fyp_user'),
        password=password,
        row_factory=dict_row,
        connect_timeout=5,
    )
    register_vector(conn)
    return conn


@lru_cache(maxsize=1)
def get_embedder():
    # Keep this first prototype on CPU. Reserve the RTX 4060 for later video/VLM tests.
    return SentenceTransformer(EMBEDDING_MODEL, device='cpu')


def embed(text: str):
    return get_embedder().encode(text, normalize_embeddings=True).astype('float32')


class SceneIn(BaseModel):
    video_id: str = Field(min_length=1, max_length=120)
    camera_id: str = Field(min_length=1, max_length=120)
    start_seconds: float = Field(ge=0)
    end_seconds: float = Field(gt=0)
    description: str = Field(min_length=5, max_length=1000)

    @model_validator(mode='after')
    def validate_time_range(self):
        if self.end_seconds <= self.start_seconds:
            raise ValueError('end_seconds must be greater than start_seconds')
        return self


@app.get('/health')
def health():
    try:
        with open_database() as conn:
            version = conn.execute("SELECT extversion FROM pg_extension WHERE extname='vector'").fetchone()
        return {'status': 'ok', 'database': 'connected', 'pgvector': version['extversion'] if version else None}
    except (psycopg.Error, RuntimeError) as exc:
        raise HTTPException(status_code=503, detail=f'Database unavailable: {exc.__class__.__name__}') from exc


@app.post('/scenes', status_code=201)
def add_scene(scene: SceneIn):
    embedding = embed(scene.description)
    try:
        with open_database() as conn:
            result = conn.execute(
                '''INSERT INTO scenes
                   (video_id, camera_id, start_seconds, end_seconds, description, embedding_model, embedding)
                   VALUES (%s, %s, %s, %s, %s, %s, %s)
                   ON CONFLICT (video_id, start_seconds, end_seconds, description)
                   DO UPDATE SET camera_id = EXCLUDED.camera_id,
                                 embedding_model = EXCLUDED.embedding_model,
                                 embedding = EXCLUDED.embedding
                   RETURNING id''',
                (scene.video_id, scene.camera_id, scene.start_seconds, scene.end_seconds,
                 scene.description, EMBEDDING_MODEL, embedding),
            ).fetchone()
        return {'id': result['id'], 'status': 'stored', 'model': EMBEDDING_MODEL}
    except psycopg.Error as exc:
        raise HTTPException(status_code=503, detail='Database insert failed') from exc


@app.get('/search')
def search(q: str = Query(min_length=2, max_length=300), limit: int = Query(default=5, ge=1, le=20)):
    query_embedding = embed(q)
    try:
        with open_database() as conn:
            rows = conn.execute(
                '''WITH query_embedding AS (SELECT %s::vector AS embedding)
                   SELECT s.id, s.video_id, s.camera_id, s.start_seconds,
                          s.end_seconds, s.description,
                          1 - (s.embedding <=> q.embedding) AS similarity
                   FROM scenes AS s CROSS JOIN query_embedding AS q
                   WHERE s.embedding_model = %s
                   ORDER BY s.embedding <=> q.embedding
                   LIMIT %s''',
                (query_embedding, EMBEDDING_MODEL, limit),
            ).fetchall()
        return {'query': q, 'results': [
            {**row, 'start_seconds': float(row['start_seconds']),
             'end_seconds': float(row['end_seconds']),
             'similarity': round(row['similarity'], 4)} for row in rows
        ]}
    except psycopg.Error as exc:
        raise HTTPException(status_code=503, detail='Database search failed') from exc

"""Stage 3: benchmark automatic, single-frame scene captions and optionally index them.

Requires the existing Stage 1 app.py and Stage 2 frame_candidates tables.
Does not change manually entered frame_candidates.description values.
"""
from __future__ import annotations

import argparse
import json
import re
import time
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
DEFAULT_MODEL = 'HuggingFaceTB/SmolVLM2-500M-Video-Instruct'


PROMPT = (
    "Examine this image carefully. "
    "Describe the portable objects that are clearly visible. "
    "For each visible object, state its type, color, and location. "
    "Explain whether it is resting on a surface or being held, "
    "only when that relationship is visually clear. "
    "Do not assume an object exists because it is commonly found "
    "in this environment. "
    "Do not invent objects, locations, or actions. "
    "Omit uncertain details. "
    "Respond in one or two factual sentences."
)




def safe_name(name: str) -> str:
    return re.sub(r'[^A-Za-z0-9_.-]+', '_', name).strip('._') or 'unnamed'


def get_candidates(video_id: str, limit: int, offset: int):
    from app import open_database
    with open_database() as conn:
        rows = conn.execute('''
            SELECT f.id, f.video_id, f.timestamp_seconds,
                   f.clip_start_seconds, f.clip_end_seconds, f.frame_path,
                   f.description AS manual_description, r.camera_id
              FROM frame_candidates AS f
              JOIN video_recordings AS r ON r.video_id = f.video_id
             WHERE f.video_id = %s
             ORDER BY f.timestamp_seconds, f.id
             LIMIT %s OFFSET %s
        ''', (video_id, limit, offset)).fetchall()
    return rows


def load_vlm(model_id: str, cpu: bool):
    import torch
    from transformers import AutoModelForImageTextToText, AutoProcessor

    if not cpu and not torch.cuda.is_available():
        raise RuntimeError('CUDA not available. Check your (.venv), or use --cpu (much slower).')
    device = 'cpu' if cpu else 'cuda'
    dtype = torch.float32 if cpu else torch.float16
    print(f'Loading {model_id} on {device} (first run downloads model weights)...', flush=True)
    start = time.perf_counter()
    processor = AutoProcessor.from_pretrained(model_id)
    model = AutoModelForImageTextToText.from_pretrained(
        model_id, dtype=dtype, attn_implementation='sdpa'
    ).to(device)
    model.eval()
    print(f'Model loaded in {time.perf_counter() - start:.1f}s', flush=True)
    return processor, model, device, dtype


def caption_one(image_path: Path, processor, model, device: str, dtype, max_side: int,
                max_new_tokens: int):
    import torch
    from PIL import Image

    with Image.open(image_path) as handle:
        picture = handle.convert('RGB')
    picture.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)
    messages = [{'role': 'user', 'content': [
        {'type': 'image', 'image': picture},
        {'type': 'text', 'text': PROMPT},
    ]}]
    inputs = processor.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=True,
        return_dict=True, return_tensors='pt'
    ).to(device, dtype=dtype)

    if device == 'cuda':
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
    began = time.perf_counter()
    with torch.inference_mode():
        generated = model.generate(**inputs, do_sample=False,
                                   max_new_tokens=max_new_tokens, repetition_penalty=1.15,
                                   no_repeat_ngram_size=3,)
    if device == 'cuda':
        torch.cuda.synchronize()
    inference_ms = (time.perf_counter() - began) * 1000
    input_tokens = inputs['input_ids'].shape[1]
    caption = processor.batch_decode(
        generated[:, input_tokens:], skip_special_tokens=True
    )[0].strip()
    peak_mib = round(torch.cuda.max_memory_allocated() / (1024 * 1024), 1) if device == 'cuda' else None
    return caption, round(inference_ms, 1), peak_mib


def store_caption(row: dict, model_id: str, caption: str, elapsed_ms: float,
                  peak_mib: float | None) -> str:
    """Persist only AI-generated descriptions, with explicit provenance.

    Existing manual descriptions in frame_candidates remain unchanged.
    Subsequent runs using the same frame and model do not add repeated scene rows.
    """
    from app import EMBEDDING_MODEL, embed, open_database
    vector = embed(caption)  # CPU text model; independent of the VLM image encoder
    with open_database() as conn:
        conn.execute('''CREATE TABLE IF NOT EXISTS ai_frame_captions (
            frame_candidate_id BIGINT NOT NULL REFERENCES frame_candidates(id),
            model_id TEXT NOT NULL,
            scene_id BIGINT REFERENCES scenes(id),
            caption TEXT NOT NULL,
            inference_ms DOUBLE PRECISION NOT NULL,
            peak_vram_mib DOUBLE PRECISION,
            created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
            PRIMARY KEY(frame_candidate_id, model_id)
        )''')
        already = conn.execute('''SELECT scene_id FROM ai_frame_captions
            WHERE frame_candidate_id = %s AND model_id = %s''',
            (row['id'], model_id)).fetchone()
        if already is not None:
            return f"already indexed (scene {already['scene_id']})"
        stored = conn.execute('''INSERT INTO scenes
            (video_id, camera_id, start_seconds, end_seconds,
             description, embedding_model, embedding)
            VALUES (%s,%s,%s,%s,%s,%s,%s)
            ON CONFLICT (video_id, start_seconds, end_seconds, description)
            DO UPDATE SET embedding_model=EXCLUDED.embedding_model,
                          embedding=EXCLUDED.embedding
            RETURNING id''',
            (row['video_id'], row['camera_id'], row['clip_start_seconds'],
             row['clip_end_seconds'], caption, EMBEDDING_MODEL, vector)).fetchone()
        conn.execute('''INSERT INTO ai_frame_captions
            (frame_candidate_id, model_id, scene_id, caption, inference_ms, peak_vram_mib)
            VALUES (%s,%s,%s,%s,%s,%s)''',
            (row['id'], model_id, stored['id'], caption, elapsed_ms, peak_mib))
    return f"indexed as scene {stored['id']}"


def main() -> None:
    parser = argparse.ArgumentParser(
        description='Caption a few existing Stage 2 frames with a vision-language model.'
    )
    parser.add_argument('--video-id', required=True, help='Exact Video ID printed by ingest_video.py')
    parser.add_argument('--limit', type=int, default=3, help='Frames to test (default 3)')
    parser.add_argument('--offset', type=int, default=0, help='Skip this many earlier frames')
    parser.add_argument('--list', action='store_true', help='List selected frames without downloading a model')
    parser.add_argument('--index', action='store_true', help='Save AI captions in PostgreSQL scenes for /search')
    parser.add_argument('--model', default=DEFAULT_MODEL, help='Hugging Face model ID')
    parser.add_argument('--max-side', type=int, default=768, help='Resize image so longest edge <= N')
    parser.add_argument('--max-new-tokens', type=int, default=96)
    parser.add_argument('--cpu', action='store_true', help='CPU fallback; much slower')
    args = parser.parse_args()
    if args.limit < 1 or args.offset < 0 or args.max_side < 224 or args.max_new_tokens < 16:
        parser.error('Use --limit >=1, --offset >=0, --max-side >=224 and --max-new-tokens >=16')
    rows = get_candidates(args.video_id, args.limit, args.offset)
    if not rows:
        parser.error('No candidate frames for this Video ID. Verify Stage 2 ingestion and ID.')
    print(f'Found {len(rows)} frame candidates for {args.video_id}:')
    for row in rows:
        path = Path(row['frame_path'])
        print(f"  candidate {row['id']} @ {float(row['timestamp_seconds']):.2f}s | "
              f"{'EXISTS' if path.is_file() else 'MISSING'} | {path}")
    if args.list:
        return
    if any(not Path(row['frame_path']).is_file() for row in rows):
        parser.error('Some extracted frame files are missing; re-ingest the recording.')

    processor, model, device, dtype = load_vlm(args.model, args.cpu)
    output_dir = HERE / 'data' / 'stage3' / safe_name(args.video_id)
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / (datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ_')
                             + safe_name(args.model) + '.jsonl')
    print(f'Writing benchmark report: {out_path}', flush=True)
    successes = 0
    with out_path.open('w', encoding='utf-8') as file:
        for row in rows:
            path = Path(row['frame_path'])
            print(f"\nCandidate {row['id']} at {float(row['timestamp_seconds']):.2f}s", flush=True)
            try:
                caption, ms, peak = caption_one(
                    path, processor, model, device, dtype, args.max_side, args.max_new_tokens
                )
            except RuntimeError as exc:
                if 'out of memory' in str(exc).lower():
                    raise SystemExit('CUDA out of memory. Close GPU apps, use a smaller model '
                                     'or reduce --max-side to 512. Database is unchanged.') from exc
                raise
            if len(caption) < 5:
                print('Caption was empty/too short; skipped.')
                continue
            success = {'video_id': row['video_id'], 'camera_id': row['camera_id'],
                       'frame_candidate_id': row['id'],
                       'timestamp_seconds': float(row['timestamp_seconds']),
                       'clip_start_seconds': float(row['clip_start_seconds']),
                       'clip_end_seconds': float(row['clip_end_seconds']),
                       'frame_path': str(path), 'model_id': args.model,
                       'caption': caption,
                       'manual_description': row['manual_description'],
                       'inference_ms': ms, 'peak_vram_mib': peak}
            print('AI caption:', caption)
            print(f'Inference: {ms:.0f} ms | GPU peak: {peak if peak is not None else "N/A"} MiB')
            if row['manual_description']:
                print('Manual label:', row['manual_description'])
            if args.index:
                success['index_status'] = store_caption(row, args.model, caption, ms, peak)
                print('Search:', success['index_status'])
            file.write(json.dumps(success, ensure_ascii=False) + '\n')
            file.flush()
            successes += 1
    print(f'\nFinished: {successes}/{len(rows)} captions. Benchmark report: {out_path}')
    if args.index:
        print('New AI captions are available in the existing GET /search API.')
    else:
        print('Preview only: database unchanged. Add --index when ready to search AI captions.')


if __name__ == '__main__':
    main()

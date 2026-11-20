"""Stage 6: preview overlapping temporal captions; selectively index reviewed windows.

Works alongside the existing Stage 1-5 Group 8 prototype. No updates to app.py.

Examples:
  python temporal_window_index.py --video-id test01_863e5601e3b5 --list
  python temporal_window_index.py --video-id test01_863e5601e3b5 --max-windows 3
  python temporal_window_index.py --index-report "data\\stage6\\...\\report.json" --approve 1

Crucial: preview does not change the database. Indexing requires an explicit approval
of selected window numbers, after comparing captions with source frames.
"""
from __future__ import annotations

import argparse
import json
import math
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PROMPT_VERSION = "stage6_temporal_v1"


def select_evenly(rows: list[dict], maximum: int) -> list[dict]:
    """Pick chronological samples including the first/last frame in each window."""
    if maximum < 2:
        raise ValueError("maximum must be at least 2")
    if len(rows) <= maximum:
        return rows[:]
    indices = [round(i * (len(rows) - 1) / (maximum - 1)) for i in range(maximum)]
    return [rows[i] for i in indices]


def make_windows(rows: list[dict], duration: float, window_seconds: float,
                 stride_seconds: float, frames_per_window: int,
                 max_windows: int) -> list[dict]:
    """Construct overlapping time-bounded windows from Stage 2 candidates.

    Important: a window's edges are *requested retrieval bounds*, not precise
    detected event boundaries. No unobserved movements should be inferred.
    """
    if duration <= 0 or window_seconds <= 0 or stride_seconds <= 0 or max_windows < 1:
        raise ValueError("duration/window/stride/max_windows must be positive")
    if frames_per_window < 2:
        raise ValueError("frames_per_window must be >=2")
    ordered = sorted(rows, key=lambda row: (float(row['timestamp_seconds']), row['id']))
    windows: list[dict] = []
    for i in range(math.ceil(duration / stride_seconds)):
        if len(windows) >= max_windows:
            break
        start = round(i * stride_seconds, 3)
        if start >= duration:
            break
        end = round(min(start + window_seconds, duration), 3)
        # Include exact end frame, if available. At shared boundaries, a frame
        # may legitimately occur in two overlapping windows.
        candidates = [row for row in ordered
                      if start - 1e-7 <= float(row['timestamp_seconds']) <= end + 1e-7]
        if len(candidates) < 2:
            continue
        chosen = select_evenly(candidates, frames_per_window)
        windows.append({
            'window_number': len(windows) + 1,
            'start_seconds': start,
            'end_seconds': end,
            'rows': chosen,
        })
    return windows


def get_video_rows(video_id: str) -> tuple[dict, list[dict]]:
    from app import open_database
    with open_database() as conn:
        video = conn.execute('''SELECT video_id, camera_id, duration_seconds
            FROM video_recordings WHERE video_id=%s''', (video_id,)).fetchone()
        if video is None:
            raise ValueError(f'Unknown video ID: {video_id}')
        rows = conn.execute('''SELECT id, video_id, timestamp_seconds, frame_path,
            description AS manual_description FROM frame_candidates
            WHERE video_id=%s ORDER BY timestamp_seconds, id''', (video_id,)).fetchall()
    return video, rows


def make_report(video: dict, windows: list[dict], args, model_id: str) -> dict:
    from temporal_caption_test import TEMPORAL_PROMPT
    return {
        'report_format': 'group8-stage6-v1',
        'created_at_utc': datetime.now(timezone.utc).isoformat(),
        'video_id': video['video_id'],
        'camera_id': video['camera_id'],
        'video_duration_seconds': float(video['duration_seconds']),
        'model_id': model_id,
        'prompt_version': PROMPT_VERSION,
        'prompt': TEMPORAL_PROMPT,
        'window_seconds': args.window_seconds,
        'stride_seconds': args.stride_seconds,
        'frames_per_window': args.frames_per_window,
        'max_side': args.max_side,
        'max_new_tokens': args.max_new_tokens,
        'database_updated': False,
        'windows': [],
    }


def capture_window(window: dict, processor, model, device: str, dtype,
                   max_side: int, max_new_tokens: int) -> tuple[str, float, float | None]:
    from temporal_caption_test import make_messages, load_images
    import torch

    images = load_images(window['rows'], max_side)
    messages = make_messages(window['rows'], images)
    inputs = processor.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=True,
        return_dict=True, return_tensors='pt'
    ).to(device, dtype=dtype)
    if device == 'cuda':
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
    start = time.perf_counter()
    try:
        with torch.inference_mode():
            output = model.generate(
                **inputs, do_sample=False, max_new_tokens=max_new_tokens,
                repetition_penalty=1.15, no_repeat_ngram_size=3,
            )
    except torch.cuda.OutOfMemoryError as exc:
        raise RuntimeError('CUDA out of memory; retry --max-side 384 or --frames-per-window 3') from exc
    if device == 'cuda':
        torch.cuda.synchronize()
    duration_ms = round((time.perf_counter() - start) * 1000, 1)
    input_length = inputs['input_ids'].shape[1]
    caption = processor.batch_decode(output[:, input_length:], skip_special_tokens=True)[0].strip()
    memory = round(torch.cuda.max_memory_allocated() / (1024 * 1024), 1) if device == 'cuda' else None
    return caption, duration_ms, memory


def preview(args) -> None:
    from auto_caption_frames import load_vlm, safe_name

    video, rows = get_video_rows(args.video_id)
    windows = make_windows(rows, float(video['duration_seconds']), args.window_seconds,
                           args.stride_seconds, args.frames_per_window, args.max_windows)
    if not windows:
        raise ValueError('No windows with at least 2 candidate frames. Re-ingest with denser sampling.')

    print(f"Video {video['video_id']} ({float(video['duration_seconds']):.2f}s): ")
    for window in windows:
        times = [float(row['timestamp_seconds']) for row in window['rows']]
        print(f"  Window {window['window_number']}: {window['start_seconds']:.1f}–{window['end_seconds']:.1f}s; "
              f"sampled frames {', '.join(f'{x:g}' for x in times)}s")
    if args.list:
        return
    missing = [str(row['frame_path']) for w in windows for row in w['rows']
               if not Path(row['frame_path']).is_file()]
    if missing:
        raise FileNotFoundError('Missing extracted frame: ' + missing[0])

    processor, model, device, dtype = load_vlm(args.model, args.cpu)
    report = make_report(video, windows, args, args.model)
    output_dir = ROOT / 'data' / 'stage6' / safe_name(args.video_id)
    output_dir.mkdir(parents=True, exist_ok=True)
    output = output_dir / (datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ_')
                           + safe_name(args.model) + '_temporal_windows.json')
    try:
        for window in windows:
            caption, elapsed, vram = capture_window(
                window, processor, model, device, dtype, args.max_side, args.max_new_tokens
            )
            result = {
                'window_number': window['window_number'],
                'start_seconds': window['start_seconds'],
                'end_seconds': window['end_seconds'],
                'frame_candidate_ids': [int(row['id']) for row in window['rows']],
                'timestamps_seconds': [float(row['timestamp_seconds']) for row in window['rows']],
                'frame_paths': [str(row['frame_path']) for row in window['rows']],
                'manual_labels': [row.get('manual_description') for row in window['rows']],
                'caption': caption,
                'inference_ms': elapsed,
                'peak_gpu_memory_mib': vram,
            }
            report['windows'].append(result)
            print(f"\nWindow {window['window_number']} "
                  f"[{window['start_seconds']:.1f}–{window['end_seconds']:.1f}s]: {caption}")
            print(f"Inference: {elapsed:.0f}ms | GPU peak: {vram}MiB")
            # Save intermediate progress if the model fails in a later window.
            output.write_text(json.dumps(report, indent=2, ensure_ascii=False, default=str), encoding='utf-8')
    except Exception:
        print(f'Partial report, if any: {output}')
        raise
    print(f'\nPreview report: {output}')
    print('DATABASE UNCHANGED. Review each caption against its actual frames before indexing.')
    print('To index only approved windows (example):')
    print(f'  python temporal_window_index.py --index-report "{output}" --approve 1')


def parse_approved(value: str) -> set[int]:
    try:
        approved = {int(x.strip()) for x in value.split(',') if x.strip()}
    except ValueError as exc:
        raise ValueError('--approve must be comma-separated window numbers, e.g. 1,3') from exc
    if not approved or any(n <= 0 for n in approved):
        raise ValueError('--approve needs positive window numbers, e.g. 1,3')
    return approved


def index_report(report_path: Path, approval: str) -> None:
    from app import EMBEDDING_MODEL, embed, open_database

    report = json.loads(report_path.read_text(encoding='utf-8'))
    if report.get('report_format') != 'group8-stage6-v1':
        raise ValueError('Unsupported report format')
    if report.get('prompt_version') != PROMPT_VERSION:
        raise ValueError('Prompt version mismatch; do not index a modified/old report')
    if len(report.get('windows', [])) == 0:
        raise ValueError('The report contains no completed windows')
    selected = parse_approved(approval)
    available = {w['window_number'] for w in report['windows']}
    if not selected.issubset(available):
        raise ValueError(f'Unknown window number(s): {sorted(selected - available)}')

    print(f"About to index {len(selected)} explicitly approved windows from {report['video_id']}")
    with open_database() as conn:
        # Keep this metadata separate from manual labels and individual frame captions.
        conn.execute('''CREATE TABLE IF NOT EXISTS temporal_window_index (
            video_id TEXT NOT NULL REFERENCES video_recordings(video_id),
            model_id TEXT NOT NULL,
            prompt_version TEXT NOT NULL,
            start_seconds NUMERIC(12,3) NOT NULL,
            end_seconds NUMERIC(12,3) NOT NULL,
            scene_id BIGINT NOT NULL REFERENCES scenes(id),
            frame_candidate_ids JSONB NOT NULL,
            frame_times_seconds JSONB NOT NULL,
            caption TEXT NOT NULL,
            inference_ms DOUBLE PRECISION NOT NULL,
            peak_gpu_memory_mib DOUBLE PRECISION,
            report_path TEXT NOT NULL,
            created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            PRIMARY KEY(video_id, model_id, prompt_version, start_seconds, end_seconds)
        )''')
        # This query validates that the recording still exists. Never trust only
        # IDs or times provided by a saved JSON file.
        video = conn.execute('''SELECT video_id, camera_id, duration_seconds
            FROM video_recordings WHERE video_id=%s''', (report['video_id'],)).fetchone()
        if video is None or video['camera_id'] != report['camera_id']:
            raise ValueError('Report video/camera does not match the local database')
        for window in report['windows']:
            if window['window_number'] not in selected:
                continue
            start = float(window['start_seconds'])
            end = float(window['end_seconds'])
            caption = str(window['caption']).strip()
            if not (0 <= start < end <= float(video['duration_seconds']) + 0.01):
                raise ValueError('Invalid time boundaries in report')
            if len(caption) < 8:
                raise ValueError('Refusing empty or very short caption')
            existing = conn.execute('''SELECT caption,scene_id FROM temporal_window_index
                WHERE video_id=%s AND model_id=%s AND prompt_version=%s
                  AND start_seconds=%s AND end_seconds=%s''',
                (report['video_id'], report['model_id'], PROMPT_VERSION, start, end)).fetchone()
            if existing is not None:
                if existing['caption'] != caption:
                    raise ValueError('This window was already indexed with different text. '
                                     'Do not silently overwrite reviewed results.')
                print(f"  Window {window['window_number']}: already indexed (scene {existing['scene_id']})")
                continue
            # Verify all candidate IDs still belong to the actual recording.
            ids = [int(n) for n in window['frame_candidate_ids']]
            matches = conn.execute('''SELECT COUNT(*) AS n FROM frame_candidates
                WHERE video_id=%s AND id=ANY(%s)''', (report['video_id'], ids)).fetchone()['n']
            if matches != len(set(ids)) or len(ids) < 2:
                raise ValueError('Frame references do not match the database')
            vector = embed(caption)
            scene = conn.execute('''INSERT INTO scenes
                (video_id,camera_id,start_seconds,end_seconds,description,embedding_model,embedding)
                VALUES (%s,%s,%s,%s,%s,%s,%s)
                ON CONFLICT (video_id,start_seconds,end_seconds,description)
                DO UPDATE SET embedding_model=EXCLUDED.embedding_model,
                              embedding=EXCLUDED.embedding, camera_id=EXCLUDED.camera_id
                RETURNING id''',
                (report['video_id'], report['camera_id'], start, end, caption,
                 EMBEDDING_MODEL, vector)).fetchone()
            conn.execute('''INSERT INTO temporal_window_index
                (video_id,model_id,prompt_version,start_seconds,end_seconds,scene_id,
                 frame_candidate_ids,frame_times_seconds,caption,inference_ms,
                 peak_gpu_memory_mib,report_path)
                VALUES (%s,%s,%s,%s,%s,%s,%s::jsonb,%s::jsonb,%s,%s,%s,%s)''',
                (report['video_id'], report['model_id'], PROMPT_VERSION, start, end,
                 scene['id'], json.dumps(ids), json.dumps(window['timestamps_seconds']),
                 caption, window['inference_ms'], window['peak_gpu_memory_mib'],
                 str(report_path.resolve())))
            print(f"  Window {window['window_number']}: indexed as scene {scene['id']} "
                  f"({start:g}–{end:g}s)")
    print('Approved captions are now searchable through the existing GET /search API.')
    print('The temporal_window_index table records their provenance; manual labels remain unchanged.')


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--video-id', help='Stage 2 video ID for listing/preview')
    parser.add_argument('--list', action='store_true', help='List proposed windows without loading the VLM')
    parser.add_argument('--window-seconds', type=float, default=4.0)
    parser.add_argument('--stride-seconds', type=float, default=3.0)
    parser.add_argument('--frames-per-window', type=int, default=4)
    parser.add_argument('--max-windows', type=int, default=3)
    parser.add_argument('--model', default='HuggingFaceTB/SmolVLM2-2.2B-Instruct')
    parser.add_argument('--max-side', type=int, default=512)
    parser.add_argument('--max-new-tokens', type=int, default=96)
    parser.add_argument('--cpu', action='store_true')
    parser.add_argument('--index-report', type=Path, help='Index reviewed captions from a saved preview report')
    parser.add_argument('--approve', help='Explicitly approve numbered windows (e.g. 1,2)')
    args = parser.parse_args()

    if args.index_report is not None:
        if not args.approve:
            parser.error('--index-report requires --approve 1 or --approve 1,3')
        index_report(args.index_report, args.approve)
        return
    if not args.video_id:
        parser.error('--video-id is required for listing and preview')
    if args.approve:
        parser.error('--approve is only supported with --index-report')
    if (args.window_seconds <= 0 or args.stride_seconds <= 0 or args.frames_per_window < 2
            or args.max_windows < 1 or args.max_side < 224 or args.max_new_tokens < 24):
        parser.error('Invalid parameters. Use positive window and stride, frames>=2, '
                     'max_windows>=1, max_side>=224 and tokens>=24')
    preview(args)


if __name__ == '__main__':
    main()

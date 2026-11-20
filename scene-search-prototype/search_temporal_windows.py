"""Search only explicitly indexed Stage 6 temporal windows from one recording.

Examples:
  python search_temporal_windows.py --video-id test01_863e5601e3b5 --q "backpack moved from chair to desk"
"""
from __future__ import annotations
import argparse

from app import EMBEDDING_MODEL, embed, open_database


def search(video_id: str, q: str, limit: int = 5) -> list[dict]:
    vector = embed(q)
    with open_database() as conn:
        rows = conn.execute('''WITH query_embedding AS (SELECT %s::vector AS embedding)
            SELECT tw.start_seconds, tw.end_seconds, tw.caption,
                   tw.frame_times_seconds, s.camera_id,
                   1 - (s.embedding <=> q.embedding) AS similarity
            FROM temporal_window_index AS tw
            JOIN scenes AS s ON s.id = tw.scene_id
            CROSS JOIN query_embedding AS q
            WHERE tw.video_id = %s AND s.embedding_model = %s
            ORDER BY s.embedding <=> q.embedding
            LIMIT %s''', (vector, video_id, EMBEDDING_MODEL, limit)).fetchall()
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--video-id', required=True)
    parser.add_argument('--q', required=True, help='Natural-language event query')
    parser.add_argument('--limit', type=int, default=5)
    args = parser.parse_args()
    if args.limit < 1 or not args.q.strip():
        parser.error('--limit must be positive and --q nonempty')
    try:
        rows = search(args.video_id, args.q, args.limit)
    except Exception as exc:
        if 'temporal_window_index' in str(exc):
            parser.error('No Stage 6 temporal index table yet. Index an approved report first.')
        raise
    if not rows:
        print('No approved temporal windows indexed for that video yet.')
        return
    print(f'Query: {args.q}')
    for i, row in enumerate(rows, 1):
        times = row['frame_times_seconds']
        print(f"{i}. {float(row['start_seconds']):.1f}–{float(row['end_seconds']):.1f}s "
              f"({row['camera_id']}, similarity {float(row['similarity']):.4f})")
        print(f"   {row['caption']}")
        print(f"   Frames sampled at: {', '.join(str(t) for t in times)}s")


if __name__ == '__main__':
    main()

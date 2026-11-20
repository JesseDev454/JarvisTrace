"""Optional human annotations of real frames, reusing existing semantic search API.

Only label a frame if the described object is visible; captions are NOT inferred.
"""
import argparse
import os
from pathlib import Path

from app import EMBEDDING_MODEL, embed, open_database


def main():
    parser = argparse.ArgumentParser(description='Manually label extracted real-video frame candidates')
    parser.add_argument('--video-id', required=True)
    parser.add_argument('--limit', type=int, default=10)
    parser.add_argument('--open-images', action='store_true', help='Open each JPG in your default image viewer (Windows)')
    args = parser.parse_args()
    if args.limit < 1:
        parser.error('--limit must be greater than 0')

    with open_database() as conn:
        rows = conn.execute('''SELECT f.id, f.timestamp_seconds, f.clip_start_seconds,
                   f.clip_end_seconds, f.frame_path, r.camera_id
            FROM frame_candidates f JOIN video_recordings r USING (video_id)
            WHERE f.video_id = %s AND f.description IS NULL
            ORDER BY f.timestamp_seconds LIMIT %s''', (args.video_id, args.limit)).fetchall()

    if not rows:
        print('No unlabeled candidates found for this video ID. Check ingest output.')
        return
    print(f'Found {len(rows)} unlabeled frames. Press ENTER to skip; type q to stop.')
    print('Use only what you can actually see in the image; do not invent objects.')
    for row in rows:
        image = Path(row['frame_path'])
        print(f"\nFrame {row['id']} @ {float(row['timestamp_seconds']):.2f}s — {image}")
        if not image.is_file():
            print('Image file missing. Skipping.')
            continue
        if args.open_images:
            if os.name != 'nt':
                print('Automatic image opening is supported only on Windows here.')
            else:
                os.startfile(str(image))
        description = input('What is visibly in this frame? ').strip()
        if description.lower() in ('q', 'quit', 'exit'):
            break
        if len(description) < 5:
            print('Skipped.')
            continue
        embedding = embed(description)
        with open_database() as conn:
            conn.execute('''INSERT INTO scenes
                (video_id, camera_id, start_seconds, end_seconds,
                 description, embedding_model, embedding)
                VALUES (%s,%s,%s,%s,%s,%s,%s)
                ON CONFLICT(video_id,start_seconds,end_seconds,description)
                DO UPDATE SET camera_id=EXCLUDED.camera_id,
                              embedding_model=EXCLUDED.embedding_model,
                              embedding=EXCLUDED.embedding''',
                (args.video_id, row['camera_id'], row['clip_start_seconds'],
                 row['clip_end_seconds'], description, EMBEDDING_MODEL, embedding))
            conn.execute('''UPDATE frame_candidates SET description=%s WHERE id=%s''',
                         (description, row['id']))
        print('Indexed. Search using the existing /search API.')


if __name__ == '__main__':
    main()

"""Create an isolated prototype table and insert five example scene descriptions."""
from app import EMBEDDING_MODEL, embed, open_database

DEMO_SCENES = [
    ('camera_01.mp4', 'CAM-01', 15.0, 22.0, 'A black backpack is resting on a chair beside a desk.'),
    ('camera_01.mp4', 'CAM-01', 60.0, 68.0, 'A silver laptop lies open on the office table.'),
    ('camera_02.mp4', 'CAM-02', 110.0, 118.0, 'A transparent water bottle stands on a shelf.'),
    ('camera_03.mp4', 'CAM-03', 35.0, 43.0, 'A red handbag is placed near the entrance.'),
    ('camera_02.mp4', 'CAM-02', 200.0, 207.0, 'A black phone is left on a wooden desk.'),
]


def main():
    with open_database() as conn:
        conn.execute('''CREATE TABLE IF NOT EXISTS scenes (
            id BIGSERIAL PRIMARY KEY,
            video_id TEXT NOT NULL,
            camera_id TEXT NOT NULL,
            start_seconds NUMERIC(12, 2) NOT NULL CHECK (start_seconds >= 0),
            end_seconds NUMERIC(12, 2) NOT NULL CHECK (end_seconds > start_seconds),
            description TEXT NOT NULL,
            embedding_model TEXT NOT NULL,
            embedding vector(384) NOT NULL,
            created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            UNIQUE (video_id, start_seconds, end_seconds, description)
        )''')
        for video_id, camera_id, start, end, description in DEMO_SCENES:
            conn.execute('''INSERT INTO scenes
                (video_id, camera_id, start_seconds, end_seconds, description, embedding_model, embedding)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (video_id, start_seconds, end_seconds, description) DO NOTHING''',
                (video_id, camera_id, start, end, description, EMBEDDING_MODEL, embed(description)))
    print(f'Seeded up to {len(DEMO_SCENES)} sample scenes. Rerunning is safe.')


if __name__ == '__main__':
    main()

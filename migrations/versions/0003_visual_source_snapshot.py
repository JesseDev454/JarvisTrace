"""Pin visual evidence to immutable source metadata instead of mutable registry rows.

Pre-snapshot visual rows stay unsearchable until an explicit source-hash-verified
reindex fills the snapshot. Do not guess historical camera/path values in SQL.
"""
from alembic import op
import sqlalchemy as sa

revision = '0003_visual_source_snapshot'
down_revision = '0002_visual_embeddings'
branch_labels = depends_on = None


def upgrade():
    op.add_column('visual_embeddings',sa.Column('source_camera_id',sa.Text))
    op.add_column('visual_embeddings',sa.Column('source_video_path',sa.Text))
    op.add_column('visual_embeddings',sa.Column('source_fps',sa.Double))
    op.add_column('visual_embeddings',sa.Column('source_duration_seconds',sa.Double))
    op.create_check_constraint('visual_source_snapshot','visual_embeddings',
        '(source_camera_id IS NULL AND source_video_path IS NULL AND source_fps IS NULL AND source_duration_seconds IS NULL) OR '
        '(source_camera_id IS NOT NULL AND source_video_path IS NOT NULL AND source_fps IS NOT NULL AND '
        'source_duration_seconds IS NOT NULL AND source_fps > 0 AND source_duration_seconds > 0)')


def downgrade():
    raise RuntimeError('Destructive automatic downgrade disabled; preserve visual source provenance')

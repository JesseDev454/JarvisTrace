"""Add the isolated 512-dimensional CLIP visual index."""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import JSONB
from pgvector.sqlalchemy import Vector

revision = '0002_visual_embeddings'
down_revision = '0001_existing_baseline'
branch_labels = depends_on = None


def upgrade():
    op.create_table('visual_embeddings',
        sa.Column('id',sa.BigInteger,primary_key=True,autoincrement=True),
        sa.Column('video_id',sa.Text,sa.ForeignKey('video_recordings.video_id'),nullable=False),
        sa.Column('source_sha256',sa.String(64),nullable=False),
        sa.Column('frame_index',sa.Integer,nullable=False),
        sa.Column('timestamp_seconds',sa.Numeric(16,6),nullable=False),
        sa.Column('variant',sa.Text,nullable=False),
        sa.Column('bbox_key',sa.Text,nullable=False),
        sa.Column('bbox_xyxy',JSONB),
        sa.Column('image_path',sa.Text,nullable=False),
        sa.Column('image_sha256',sa.String(64),nullable=False),
        sa.Column('encoder_key',sa.String(64),nullable=False),
        sa.Column('embedding_model',sa.Text,nullable=False),
        sa.Column('preprocess_version',sa.Text,nullable=False),
        sa.Column('crop_provenance',JSONB),
        sa.Column('embedding',Vector(512),nullable=False),
        sa.Column('created_at',sa.DateTime(timezone=True),server_default=sa.text('now()'),nullable=False),
        sa.CheckConstraint('frame_index >= 0 AND timestamp_seconds >= 0',name='visual_nonnegative_time'),
        sa.CheckConstraint("(variant='frame' AND bbox_xyxy IS NULL) OR (variant='crop' AND bbox_xyxy IS NOT NULL)",name='visual_variant_box'),
        sa.UniqueConstraint('video_id','source_sha256','frame_index','variant','bbox_key','encoder_key',name='visual_sample_encoder_key'))
    op.create_index('visual_recording_time','visual_embeddings',['video_id','timestamp_seconds'])


def downgrade():
    raise RuntimeError('Destructive automatic downgrade disabled; back up and explicitly review removal of the visual index')

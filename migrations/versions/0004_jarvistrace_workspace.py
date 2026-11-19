"""Add accounts, camera access, recording metadata, and durable indexing jobs."""
from alembic import op
import sqlalchemy as sa

revision = '0004_jarvistrace_workspace'
down_revision = '0003_visual_source_snapshot'
branch_labels = depends_on = None


def upgrade():
    op.create_table('app_users',
        sa.Column('user_id', sa.Text, primary_key=True),
        sa.Column('display_name', sa.Text, nullable=False),
        sa.Column('email', sa.Text, nullable=False, unique=True),
        sa.Column('role', sa.Text, nullable=False),
        sa.Column('password_hash', sa.Text, nullable=False),
        sa.Column('is_active', sa.Boolean, nullable=False, server_default=sa.true()),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=False, server_default=sa.text('now()')),
        sa.CheckConstraint("role IN ('administrator','investigator')", name='app_user_role'))
    op.create_table('app_sessions',
        sa.Column('token_hash', sa.String(64), primary_key=True),
        sa.Column('user_id', sa.Text, sa.ForeignKey('app_users.user_id', ondelete='CASCADE'), nullable=False),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=False, server_default=sa.text('now()')),
        sa.Column('last_seen_at', sa.DateTime(timezone=True), nullable=False, server_default=sa.text('now()')),
        sa.Column('expires_at', sa.DateTime(timezone=True), nullable=False))
    op.create_index('app_sessions_expiry', 'app_sessions', ['expires_at'])
    op.create_table('app_cameras',
        sa.Column('camera_id', sa.Text, primary_key=True),
        sa.Column('name', sa.Text, nullable=False),
        sa.Column('source_type', sa.Text, nullable=False),
        sa.Column('mtx_path', sa.Text, unique=True),
        sa.Column('enabled', sa.Boolean, nullable=False, server_default=sa.true()),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=False, server_default=sa.text('now()')),
        sa.CheckConstraint("source_type IN ('upload','mediamtx','rtsp')", name='app_camera_source_type'))
    op.create_table('app_camera_access',
        sa.Column('user_id', sa.Text, sa.ForeignKey('app_users.user_id', ondelete='CASCADE'), primary_key=True),
        sa.Column('camera_id', sa.Text, sa.ForeignKey('app_cameras.camera_id', ondelete='CASCADE'), primary_key=True))
    op.create_index('app_camera_access_camera', 'app_camera_access', ['camera_id'])
    op.create_table('app_recordings',
        sa.Column('video_id', sa.Text, sa.ForeignKey('video_recordings.video_id', ondelete='CASCADE'), primary_key=True),
        sa.Column('filename', sa.Text, nullable=False),
        sa.Column('source_type', sa.Text, nullable=False),
        sa.Column('source_sha256', sa.String(64), nullable=False),
        sa.Column('capture_start_utc', sa.DateTime(timezone=True)),
        sa.Column('capture_timezone', sa.Text),
        sa.Column('coverage_start_seconds', sa.Float, nullable=False, server_default='0'),
        sa.Column('coverage_end_seconds', sa.Float, nullable=False),
        sa.Column('status', sa.Text, nullable=False, server_default='queued'),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=False, server_default=sa.text('now()')),
        sa.CheckConstraint("source_type IN ('upload','mediamtx','rtsp')", name='app_recording_source_type'),
        sa.CheckConstraint("status IN ('queued','processing','ready','failed')", name='app_recording_status'),
        sa.CheckConstraint('coverage_start_seconds >= 0 AND coverage_end_seconds > coverage_start_seconds', name='app_recording_coverage'),
        sa.CheckConstraint("source_sha256 ~ '^[a-f0-9]{64}$'", name='app_recording_sha256'))
    op.create_table('indexing_jobs',
        sa.Column('job_id', sa.Text, primary_key=True),
        sa.Column('video_id', sa.Text, sa.ForeignKey('video_recordings.video_id', ondelete='CASCADE'), nullable=False, unique=True),
        sa.Column('status', sa.Text, nullable=False, server_default='queued'),
        sa.Column('failure_reason', sa.Text),
        sa.Column('attempt_count', sa.Integer, nullable=False, server_default='0'),
        sa.Column('lease_owner', sa.Text),
        sa.Column('lease_until', sa.DateTime(timezone=True)),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=False, server_default=sa.text('now()')),
        sa.Column('updated_at', sa.DateTime(timezone=True), nullable=False, server_default=sa.text('now()')),
        sa.CheckConstraint("status IN ('queued','processing','ready','failed')", name='indexing_job_status'),
        sa.CheckConstraint('attempt_count >= 0', name='indexing_job_attempts'))
    op.create_index('indexing_jobs_claim', 'indexing_jobs', ['status', 'created_at'])
    op.create_table('mediamtx_segment_events',
        sa.Column('event_id', sa.String(64), primary_key=True),
        sa.Column('camera_id', sa.Text, sa.ForeignKey('app_cameras.camera_id'), nullable=False),
        sa.Column('source_sha256', sa.String(64), nullable=False),
        sa.Column('video_id', sa.Text, sa.ForeignKey('video_recordings.video_id', ondelete='CASCADE'), nullable=False),
        sa.Column('received_at', sa.DateTime(timezone=True), nullable=False, server_default=sa.text('now()')))


def downgrade():
    raise RuntimeError('Destructive automatic downgrade disabled; back up and review workspace data explicitly')

"""Adopt only a compatible existing prototype; create/alter no legacy tables."""
from alembic import op
import sqlalchemy as sa

revision = '0001_existing_baseline'
down_revision = None
branch_labels = depends_on = None


def upgrade():
    connection = op.get_bind()
    schema = op.get_context().opts['version_table_schema']
    inspector = sa.inspect(connection)
    required = {
        'video_recordings': {'video_id','local_video_path','camera_id','fps','duration_seconds'},
        'scenes': {'id','video_id','embedding','embedding_model'},
        'frame_candidates': {'id','video_id','timestamp_seconds','frame_path'},
        'temporal_window_index': {'video_id','scene_id'},
    }
    for table, columns in required.items():
        if not inspector.has_table(table,schema=schema):
            raise RuntimeError(f'Existing prototype table missing: {table}; baseline was not adopted')
        actual = {c['name'] for c in inspector.get_columns(table,schema=schema)}
        if not columns <= actual:
            raise RuntimeError(f'Existing table incompatible: {table}; baseline was not adopted')
    quoted = connection.dialect.identifier_preparer.quote(schema)
    declared = connection.execute(sa.text("SELECT format_type(atttypid,atttypmod) FROM pg_attribute WHERE attrelid=to_regclass(:table) AND attname='embedding'"),{'table':f'{quoted}.scenes'}).scalar_one()
    if declared != 'vector(384)':
        raise RuntimeError('Expected existing scenes.embedding vector(384); refusing baseline adoption')


def downgrade():
    # The baseline predates Alembic: never drop its preserved tables.
    pass

"""Do not report queued or processing footage as searchable coverage."""
from alembic import op


revision = '0005_indexed_coverage_only'
down_revision = '0004_jarvistrace_workspace'
branch_labels = depends_on = None


def upgrade():
    op.drop_constraint('app_recording_coverage', 'app_recordings', type_='check')
    op.execute("""UPDATE app_recordings
        SET coverage_end_seconds=coverage_start_seconds
        WHERE status IN ('queued','processing')""")
    op.create_check_constraint('app_recording_coverage', 'app_recordings',
        "coverage_start_seconds >= 0 AND coverage_end_seconds >= coverage_start_seconds "
        "AND (status <> 'ready' OR coverage_end_seconds > coverage_start_seconds)")


def downgrade():
    raise RuntimeError('Destructive coverage downgrade disabled; back up and review workspace data explicitly')

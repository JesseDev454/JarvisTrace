import unittest

from src.stage12_workspace import PostgresWorkspaceStore


class RecordingConnection:
    def __init__(self):
        self.calls = []

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def execute(self, sql, params=()):
        self.calls.append((sql, params))
        if "SELECT u.user_id,u.display_name,u.email,u.role" in sql:
            return Rows([{"user_id": "investigator-7", "display_name": "Investigator",
                          "email": "i@example.test", "role": "investigator"}])
        if "SELECT a.camera_id FROM app_camera_access a" in sql:
            return Rows([{"camera_id": "CAM-A"}])
        if "SELECT c.camera_id,c.name" in sql:
            return Rows([{"camera_id": "CAM-A", "name": "Lobby", "source_type": "mediamtx",
                          "searchable_until": None}])
        if sql.startswith("UPDATE app_sessions"):
            return Rows([])
        raise AssertionError(f"Unexpected SQL: {sql}")


class Rows:
    def __init__(self, rows):
        self.rows = rows

    def fetchall(self):
        return self.rows

    def fetchone(self):
        return self.rows[0] if self.rows else None


class FinishConnection:
    def __init__(self, updated_row):
        self.updated_row = updated_row
        self.calls = []

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def execute(self, sql, params=()):
        self.calls.append((sql, params))
        if sql.lstrip().startswith("UPDATE indexing_jobs"):
            return Rows([self.updated_row] if self.updated_row else [])
        return Rows([])


class QueueConnection:
    def __init__(self):
        self.calls = []

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def execute(self, sql, params=()):
        self.calls.append((sql, params))
        if "RETURNING j.job_id,j.video_id" in sql:
            return Rows([{"job_id": "job-1", "video_id": "video-1", "source_sha256": "a" * 64}])
        return Rows([])


class WorkspaceStoreTests(unittest.TestCase):
    def test_public_investigator_id_is_used_for_camera_access_queries(self):
        connection = RecordingConnection()
        store = PostgresWorkspaceStore(connect=lambda **_kwargs: connection)
        user = store.public_user({"user_id": "investigator-7", "display_name": "Investigator",
                                  "email": "i@example.test", "role": "investigator"})

        self.assertEqual(store.allowed_cameras(user), ["CAM-A"])
        self.assertEqual(store.list_cameras(user)[0]["camera_id"], "CAM-A")
        self.assertEqual(connection.calls[0][1], ("investigator-7",))
        self.assertEqual(connection.calls[1][1], ("investigator", "investigator-7"))
        self.assertIn("CASE WHEN r.status='ready'", connection.calls[1][0])

    def test_session_user_uses_the_same_public_id_shape_for_camera_access(self):
        connection = RecordingConnection()
        store = PostgresWorkspaceStore(connect=lambda **_kwargs: connection)

        user = store.session_user("hashed-session")

        self.assertEqual(user["id"], "investigator-7")
        self.assertEqual(user["cameras"], ["CAM-A"])
        self.assertEqual(connection.calls[-1][1], ("investigator-7",))

    def test_job_completion_is_guarded_by_current_worker_lease(self):
        connection = FinishConnection(None)
        store = PostgresWorkspaceStore(connect=lambda **_kwargs: connection)

        completed = store.finish_job("job-9", "ready", worker_id="worker-old", coverage_end_seconds=12)

        self.assertFalse(completed)
        sql, params = connection.calls[0]
        self.assertIn("lease_owner=%s", sql)
        self.assertIn("status='processing'", sql)
        self.assertIn("lease_until>now()", sql)
        self.assertEqual(params, ("ready", None, "job-9", "worker-old"))
        self.assertEqual(len(connection.calls), 1)

    def test_queued_recording_has_no_searchable_coverage_before_indexing(self):
        connection = QueueConnection()
        store = PostgresWorkspaceStore(connect=lambda **_kwargs: connection)

        store.enqueue_recording(video_id="video-1", camera_id="CAM-A", path="clip.mp4", filename="clip.mp4",
            fps=30, duration=90, source_sha256="a" * 64, source_type="upload")
        claimed=store.claim_job("worker-1")

        insert_sql, insert_params = next(call for call in connection.calls if "INSERT INTO app_recordings" in call[0])
        self.assertIn("coverage_end_seconds,status", insert_sql)
        self.assertEqual(insert_params[6:8], (0, 0))
        claim_sql = next(sql for sql, _params in connection.calls if "WITH candidate AS" in sql)
        self.assertIn("UPDATE app_recordings", claim_sql)
        self.assertIn("status='processing'", claim_sql)
        self.assertIn("RETURNING c.job_id,c.video_id,a.source_sha256", claim_sql)
        self.assertEqual(claimed["source_sha256"],"a"*64)


if __name__ == "__main__":
    unittest.main()

"""Account, access-control, and durable-workspace support for Stage 12."""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import hmac
import re
import secrets
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

PASSWORD_ITERATIONS = 600_000
EMAIL_RE = re.compile(r"^[^\s@]{1,128}@[^\s@]{1,190}\.[^\s@]{2,63}$")


def normalize_email(value: str) -> str:
    email = value.strip().casefold()
    if len(email) > 320 or not EMAIL_RE.fullmatch(email):
        raise ValueError("Enter a valid email address")
    return email


def hash_password(password: str, *, salt: bytes | None = None) -> str:
    if not isinstance(password, str) or not 12 <= len(password) <= 256:
        raise ValueError("Password must be 12 to 256 characters")
    salt = salt or secrets.token_bytes(16)
    digest = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, PASSWORD_ITERATIONS)
    return f"pbkdf2_sha256${PASSWORD_ITERATIONS}${salt.hex()}${digest.hex()}"


def verify_password(password: str, encoded: str) -> bool:
    try:
        algorithm, iterations, salt_hex, digest_hex = encoded.split("$", 3)
        if algorithm != "pbkdf2_sha256" or int(iterations) != PASSWORD_ITERATIONS:
            return False
        if not re.fullmatch(r"[a-f0-9]{32}", salt_hex) or not re.fullmatch(r"[a-f0-9]{64}", digest_hex):
            return False
        candidate = hash_password(password, salt=bytes.fromhex(salt_hex)).split("$", 3)[3]
        return hmac.compare_digest(candidate, digest_hex)
    except (ValueError, TypeError, AttributeError):
        return False


def new_session_token() -> str:
    return secrets.token_urlsafe(32)


def session_token_hash(token: str) -> str:
    return hashlib.sha256(token.encode("ascii")).hexdigest()


def parse_capture_start(local_value: str | None, timezone_name: str | None):
    """Convert an explicitly supplied local capture time to UTC without guessing DST."""
    if local_value is None or not local_value.strip():
        return None, None
    if not timezone_name:
        raise ValueError("A time zone is required with the capture start")
    try:
        zone = ZoneInfo(timezone_name)
        naive = datetime.fromisoformat(local_value)
    except (ValueError, ZoneInfoNotFoundError):
        raise ValueError("Enter a valid capture start and IANA time zone") from None
    if naive.tzinfo is not None:
        raise ValueError("Capture start must be local time without an offset")
    early = naive.replace(tzinfo=zone, fold=0)
    late = naive.replace(tzinfo=zone, fold=1)
    early_roundtrip = early.astimezone(timezone.utc).astimezone(zone).replace(tzinfo=None)
    late_roundtrip = late.astimezone(timezone.utc).astimezone(zone).replace(tzinfo=None)
    if early_roundtrip != naive and late_roundtrip != naive:
        raise ValueError("This local time does not exist in the selected time zone")
    if early.utcoffset() != late.utcoffset():
        raise ValueError("This local time is ambiguous at a daylight-saving change")
    utc_value = early.astimezone(timezone.utc)
    return utc_value, zone.key


class PostgresWorkspaceStore:
    """Small repository layer; all SQL executes inside existing PostgreSQL transactions."""

    def __init__(self, connect=None):
        if connect is None:
            from src.pg_config import open_database
            connect = open_database
        self.connect = connect

    @staticmethod
    def public_user(row, cameras=()):
        if row is None:
            return None
        return {"id": str(row["user_id"]), "display_name": row["display_name"],
                "email": row["email"], "role": row["role"], "cameras": list(cameras)}

    def setup_admin(self, display_name, email, password):
        email = normalize_email(email)
        display_name = display_name.strip()
        if not 1 <= len(display_name) <= 120:
            raise ValueError("Name must be 1 to 120 characters")
        password_hash = hash_password(password)
        import uuid
        with self.connect() as conn:
            conn.execute("SELECT pg_advisory_xact_lock(81743102)")
            if conn.execute("SELECT 1 FROM app_users LIMIT 1").fetchone():
                raise RuntimeError("Administrator setup has already been completed")
            row = conn.execute("""INSERT INTO app_users(user_id,display_name,email,role,password_hash)
                VALUES (%s,%s,%s,'administrator',%s) RETURNING user_id,display_name,email,role""",
                (str(uuid.uuid4()), display_name, email, password_hash)).fetchone()
        return self.public_user(row)

    def authenticate(self, email, password):
        email = normalize_email(email)
        with self.connect(readonly=True) as conn:
            row = conn.execute("""SELECT user_id,display_name,email,role,password_hash,is_active
                FROM app_users WHERE email=%s""", (email,)).fetchone()
        if row is None:
            # Equalize the expensive path for unknown addresses.
            verify_password(password, hash_password("JarvisTrace dummy account password"))
            return None
        if not row["is_active"] or not verify_password(password, row["password_hash"]):
            return None
        return self.public_user(row)

    def user_count(self):
        with self.connect(readonly=True) as conn:
            return conn.execute("SELECT count(*) AS total FROM app_users WHERE is_active").fetchone()["total"]

    def create_session(self, user_id, token_hash, expires_at):
        with self.connect() as conn:
            conn.execute("DELETE FROM app_sessions WHERE expires_at <= now()")
            conn.execute("INSERT INTO app_sessions(token_hash,user_id,expires_at) VALUES (%s,%s,%s)",
                         (token_hash, user_id, expires_at))

    def session_user(self, token_hash):
        with self.connect() as conn:
            row = conn.execute("""SELECT u.user_id,u.display_name,u.email,u.role
                FROM app_sessions s JOIN app_users u USING(user_id)
                WHERE s.token_hash=%s AND s.expires_at>now() AND u.is_active""", (token_hash,)).fetchone()
            if row:
                conn.execute("UPDATE app_sessions SET last_seen_at=now() WHERE token_hash=%s", (token_hash,))
        if not row:
            return None
        public = self.public_user(row)
        return self.public_user(row, self.allowed_cameras(public))

    def delete_session(self, token_hash):
        with self.connect() as conn:
            conn.execute("DELETE FROM app_sessions WHERE token_hash=%s", (token_hash,))

    def allowed_cameras(self, user):
        if user["role"] == "administrator":
            with self.connect(readonly=True) as conn:
                return [r["camera_id"] for r in conn.execute("SELECT camera_id FROM app_cameras WHERE enabled ORDER BY camera_id").fetchall()]
        with self.connect(readonly=True) as conn:
            return [r["camera_id"] for r in conn.execute("""SELECT a.camera_id FROM app_camera_access a
                JOIN app_cameras c USING(camera_id) WHERE a.user_id=%s AND c.enabled ORDER BY a.camera_id""",
                (user["id"],)).fetchall()]

    def list_cameras(self, user):
        with self.connect(readonly=True) as conn:
            rows = conn.execute("""SELECT c.camera_id,c.name,c.source_type,c.mtx_path,c.enabled,
                max(CASE WHEN r.status='ready' THEN r.coverage_end_seconds
                    WHEN r.video_id IS NULL AND EXISTS (SELECT 1 FROM visual_embeddings e
                        WHERE e.video_id=v.video_id AND e.source_camera_id IS NOT NULL)
                    THEN v.duration_seconds END) AS searchable_until
                FROM app_cameras c LEFT JOIN video_recordings v ON v.camera_id=c.camera_id
                LEFT JOIN app_recordings r USING(video_id)
                WHERE c.enabled AND (%s='administrator' OR EXISTS (
                    SELECT 1 FROM app_camera_access a WHERE a.camera_id=c.camera_id AND a.user_id=%s))
                GROUP BY c.camera_id ORDER BY c.camera_id""", (user["role"], user["id"])).fetchall()
        return [dict(camera_id=r["camera_id"], name=r["name"], source_type=r["source_type"],
                     connection_status="configured" if r["source_type"] == "mediamtx" else "available",
                     searchable_until=format_relative(r["searchable_until"])) for r in rows]

    def create_camera(self, camera_id, name, source_type="mediamtx", mtx_path=None):
        camera_id = camera_id.strip()
        name = name.strip()
        if not re.fullmatch(r"[A-Za-z0-9_-]{1,80}", camera_id):
            raise ValueError("Camera ID may contain letters, numbers, underscores, and hyphens")
        if not 1 <= len(name) <= 120:
            raise ValueError("Camera name must be 1 to 120 characters")
        if source_type not in {"upload", "mediamtx"}:
            raise ValueError("Unknown camera source type")
        mtx_path = mtx_path or camera_id
        if source_type == "mediamtx" and not re.fullmatch(r"[A-Za-z0-9_/-]{1,120}", mtx_path):
            raise ValueError("MediaMTX path contains unsupported characters")
        with self.connect() as conn:
            conn.execute("INSERT INTO app_cameras(camera_id,name,source_type,mtx_path) VALUES (%s,%s,%s,%s)",
                         (camera_id, name, source_type, mtx_path if source_type == "mediamtx" else None))
        return {"camera_id": camera_id, "name": name, "source_type": source_type,
                "connection_status": "configured" if source_type == "mediamtx" else "available", "searchable_until": None}

    def camera_for_mtx_path(self, mtx_path):
        with self.connect(readonly=True) as conn:
            return conn.execute("SELECT camera_id FROM app_cameras WHERE mtx_path=%s AND enabled", (mtx_path,)).fetchone()

    def list_users(self):
        with self.connect(readonly=True) as conn:
            rows = conn.execute("""SELECT u.user_id,u.display_name,u.email,u.role,
                coalesce(array_agg(a.camera_id) FILTER (WHERE a.camera_id IS NOT NULL),'{}') AS cameras
                FROM app_users u LEFT JOIN app_camera_access a USING(user_id)
                GROUP BY u.user_id ORDER BY u.created_at""").fetchall()
        return [self.public_user(r, r["cameras"]) for r in rows]

    def create_user(self, display_name, email, password, role, cameras):
        display_name = display_name.strip(); email = normalize_email(email)
        if not 1 <= len(display_name) <= 120:
            raise ValueError("Name must be 1 to 120 characters")
        if role not in {"administrator", "investigator"}:
            raise ValueError("Unknown user role")
        password_hash = hash_password(password)
        import uuid
        user_id = str(uuid.uuid4())
        with self.connect() as conn:
            exists = {r["camera_id"] for r in conn.execute("SELECT camera_id FROM app_cameras WHERE enabled").fetchall()}
            if not set(cameras) <= exists:
                raise ValueError("Select cameras that are configured in JarvisTrace")
            row = conn.execute("""INSERT INTO app_users(user_id,display_name,email,role,password_hash)
                VALUES (%s,%s,%s,%s,%s) RETURNING user_id,display_name,email,role""",
                (user_id, display_name, email, role, password_hash)).fetchone()
            for camera_id in sorted(set(cameras)):
                conn.execute("INSERT INTO app_camera_access(user_id,camera_id) VALUES (%s,%s)", (user_id, camera_id))
        return self.public_user(row, cameras)

    def list_jobs(self):
        with self.connect(readonly=True) as conn:
            rows = conn.execute("""SELECT j.job_id,j.video_id,a.filename,v.camera_id,v.duration_seconds,
                j.status,j.failure_reason,j.created_at,j.updated_at FROM indexing_jobs j
                JOIN video_recordings v USING(video_id) JOIN app_recordings a USING(video_id)
                ORDER BY j.created_at DESC LIMIT 200""").fetchall()
        return [dict(job_id=str(r["job_id"]),video_id=r["video_id"],filename=r["filename"],camera_id=r["camera_id"],
                     duration_seconds=float(r["duration_seconds"]),status=r["status"],failure_reason=r["failure_reason"]) for r in rows]

    def enqueue_recording(self, *, video_id, camera_id, path, filename, fps, duration, source_sha256,
                          source_type, capture_start=None, capture_timezone=None, coverage_start=0,segment_event=None):
        import uuid
        job_id = hashlib.sha256((segment_event[0] if segment_event else str(uuid.uuid4())).encode()).hexdigest()[:32]
        with self.connect() as conn:
            if segment_event:
                event_id=segment_event[0]
                conn.execute("SELECT pg_advisory_xact_lock(hashtext(%s),hashtext(%s))",(event_id,camera_id))
                existing=conn.execute("SELECT video_id FROM mediamtx_segment_events WHERE event_id=%s",(event_id,)).fetchone()
                if existing:
                    prior=conn.execute("SELECT job_id,status FROM indexing_jobs WHERE video_id=%s",(existing['video_id'],)).fetchone()
                    return dict(job_id=prior['job_id'],video_id=existing['video_id'],status=prior['status'],duplicate=True)
                job_id=event_id[:32]
            conn.execute("""INSERT INTO video_recordings(video_id,local_video_path,camera_id,fps,duration_seconds)
                VALUES (%s,%s,%s,%s,%s)""", (video_id, str(path), camera_id, fps, duration))
            conn.execute("""INSERT INTO app_recordings(video_id,filename,source_type,source_sha256,capture_start_utc,
                capture_timezone,coverage_start_seconds,coverage_end_seconds,status)
                VALUES (%s,%s,%s,%s,%s,%s,%s,%s,'queued')""",
                (video_id,filename,source_type,source_sha256,capture_start,capture_timezone,coverage_start,coverage_start))
            conn.execute("INSERT INTO indexing_jobs(job_id,video_id) VALUES (%s,%s)", (job_id,video_id))
            if segment_event:
                conn.execute("""INSERT INTO mediamtx_segment_events(event_id,camera_id,source_sha256,video_id)
                    VALUES (%s,%s,%s,%s)""",(segment_event[0],camera_id,source_sha256,video_id))
        if segment_event:
            return dict(job_id=job_id,video_id=video_id,status='queued',duplicate=False)
        return job_id

    def segment_event(self, event_id):
        with self.connect(readonly=True) as conn:
            return conn.execute("""SELECT e.video_id,j.job_id,j.status FROM mediamtx_segment_events e
                JOIN indexing_jobs j USING(video_id) WHERE e.event_id=%s""",(event_id,)).fetchone()

    def list_recording_metadata(self, user):
        camera_ids = self.allowed_cameras(user)
        with self.connect(readonly=True) as conn:
            if user["role"] == "administrator":
                rows = conn.execute("""SELECT v.video_id,coalesce(r.filename,v.video_id) AS filename,
                    coalesce(r.source_type,'indexed') AS source_type,coalesce(r.source_sha256,'') AS source_sha256,
                    r.capture_start_utc,r.capture_timezone,coalesce(r.coverage_start_seconds,0) AS coverage_start_seconds,
                    coalesce(r.coverage_end_seconds,v.duration_seconds) AS coverage_end_seconds,
                    coalesce(r.status,'ready') AS status,v.camera_id,v.duration_seconds,
                    EXISTS(SELECT 1 FROM visual_embeddings e WHERE e.video_id=v.video_id AND e.source_camera_id IS NOT NULL) AS indexed
                    FROM video_recordings v LEFT JOIN app_recordings r USING(video_id)
                    WHERE EXISTS(SELECT 1 FROM visual_embeddings e WHERE e.video_id=v.video_id AND e.source_camera_id IS NOT NULL)
                    OR r.status IN ('queued','processing','failed') ORDER BY v.video_id""").fetchall()
            elif camera_ids:
                rows = conn.execute("""SELECT v.video_id,coalesce(r.filename,v.video_id) AS filename,
                    coalesce(r.source_type,'indexed') AS source_type,coalesce(r.source_sha256,'') AS source_sha256,
                    r.capture_start_utc,r.capture_timezone,coalesce(r.coverage_start_seconds,0) AS coverage_start_seconds,
                    coalesce(r.coverage_end_seconds,v.duration_seconds) AS coverage_end_seconds,
                    coalesce(r.status,'ready') AS status,v.camera_id,v.duration_seconds,
                    EXISTS(SELECT 1 FROM visual_embeddings e WHERE e.video_id=v.video_id AND e.source_camera_id IS NOT NULL) AS indexed
                    FROM video_recordings v LEFT JOIN app_recordings r USING(video_id)
                    WHERE v.camera_id=ANY(%s) AND (EXISTS(SELECT 1 FROM visual_embeddings e WHERE e.video_id=v.video_id AND e.source_camera_id IS NOT NULL)
                    OR r.status IN ('queued','processing','failed')) ORDER BY v.video_id""", (camera_ids,)).fetchall()
            else:
                rows = []
        return rows

    def finish_job(self, job_id, status, *, worker_id, failure_reason=None, coverage_end_seconds=None):
        with self.connect() as conn:
            row = conn.execute("""UPDATE indexing_jobs SET status=%s,failure_reason=%s,lease_owner=NULL,
                lease_until=NULL,updated_at=now() WHERE job_id=%s AND lease_owner=%s AND status='processing'
                AND lease_until>now()
                RETURNING video_id""", (status, failure_reason, job_id, worker_id)).fetchone()
            if row:
                if coverage_end_seconds is None:
                    conn.execute("UPDATE app_recordings SET status=%s WHERE video_id=%s", (status,row["video_id"]))
                else:
                    conn.execute("UPDATE app_recordings SET status=%s,coverage_end_seconds=%s WHERE video_id=%s",
                                 (status,coverage_end_seconds,row["video_id"]))
            return row is not None

    def retry_job(self, job_id):
        with self.connect() as conn:
            row = conn.execute("""UPDATE indexing_jobs SET status='queued',failure_reason=NULL,attempt_count=0,
                lease_owner=NULL,lease_until=NULL,updated_at=now() WHERE job_id=%s AND status='failed' RETURNING job_id""",
                (job_id,)).fetchone()
            if not row:
                return False
            conn.execute("""UPDATE app_recordings SET status='queued',coverage_end_seconds=coverage_start_seconds
                WHERE video_id=(SELECT video_id FROM indexing_jobs WHERE job_id=%s)""", (job_id,))
            return True

    def claim_job(self, worker_id):
        with self.connect() as conn:
            conn.execute("""WITH expired AS (UPDATE indexing_jobs SET status='failed',
                failure_reason='Worker stopped before completing the job',lease_owner=NULL,lease_until=NULL,updated_at=now()
                WHERE status='processing' AND lease_until<now() AND attempt_count>=3 RETURNING video_id)
                UPDATE app_recordings SET status='failed' WHERE video_id IN (SELECT video_id FROM expired)""")
            return conn.execute("""WITH candidate AS (
                SELECT job_id FROM indexing_jobs WHERE attempt_count<3 AND
                    (status='queued' OR (status='processing' AND lease_until<now()))
                ORDER BY created_at FOR UPDATE SKIP LOCKED LIMIT 1)
                , claimed AS (
                UPDATE indexing_jobs j SET status='processing',attempt_count=attempt_count+1,
                    lease_owner=%s,lease_until=now()+interval '15 minutes',failure_reason=NULL,updated_at=now()
                FROM candidate c WHERE j.job_id=c.job_id
                RETURNING j.job_id,j.video_id)
                UPDATE app_recordings a SET status='processing'
                FROM claimed c WHERE a.video_id=c.video_id
                RETURNING c.job_id,c.video_id,a.source_sha256""", (worker_id,)).fetchone()

    def heartbeat_job(self, job_id, worker_id):
        with self.connect() as conn:
            return conn.execute("""UPDATE indexing_jobs SET lease_until=now()+interval '15 minutes',updated_at=now()
                WHERE job_id=%s AND lease_owner=%s AND status='processing' RETURNING job_id""",
                (job_id,worker_id)).fetchone() is not None


def format_relative(value):
    if value is None:
        return None
    seconds = max(0, int(float(value)))
    return f"{seconds // 3600:02}:{(seconds % 3600) // 60:02}:{seconds % 60:02}"

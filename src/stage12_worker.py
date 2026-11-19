"""Durable single-GPU indexing worker. Run with ``python -m src.stage12_worker``."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import os
from pathlib import Path
import socket
import threading
import time

from src.stage12_workspace import PostgresWorkspaceStore


def _default_indexer(video_id, *, max_frames=300, expected_source_sha256=None):
    from src.stage10 import index_recording
    return index_recording(video_id,every_seconds=1.0,max_frames=max_frames,batch_size=8,
                           output_root=Path('outputs/stage10'),expected_source_sha256=expected_source_sha256)


def process_next(store=None, *, indexer=None, worker_id=None):
    """Claim and process one durable job; dependency injection keeps queue logic testable."""
    store=store or PostgresWorkspaceStore()
    indexer=indexer or _default_indexer
    worker_id=worker_id or f"{socket.gethostname()}-{os.getpid()}"
    job=store.claim_job(worker_id)
    if not job:
        return None
    stop=threading.Event()

    def keep_lease():
        while not stop.wait(60):
            try:
                if not store.heartbeat_job(job['job_id'],worker_id):
                    stop.set()
            except Exception:
                # The existing 15-minute lease allows recovery after a short DB outage.
                pass

    heartbeat=threading.Thread(target=keep_lease,name='indexing-job-heartbeat',daemon=True)
    heartbeat.start()
    try:
        report=indexer(job['video_id'],max_frames=300,expected_source_sha256=job['source_sha256'])
        if report.get('status')!='complete':
            raise RuntimeError('Indexer did not complete')
        if report.get('sampling_complete'):
            coverage_end=report.get('video',{}).get('frame_count')
            fps=report.get('video',{}).get('fps')
            coverage_end=float(coverage_end)/float(fps) if coverage_end and fps else None
        else:
            coverage_end=float(report.get('last_sample_seconds',0))+float(report.get('actual_interval_seconds',1))
        if not store.finish_job(job['job_id'],'ready',worker_id=worker_id,coverage_end_seconds=coverage_end):
            return dict(job_id=job['job_id'],video_id=job['video_id'],status='lease_lost')
        return dict(job_id=job['job_id'],video_id=job['video_id'],status='ready',coverage_end_seconds=coverage_end)
    except Exception as exc:
        reason=f"Indexing failed ({type(exc).__name__}). Review the worker log and retry after correcting the issue."
        if not store.finish_job(job['job_id'],'failed',worker_id=worker_id,failure_reason=reason):
            return dict(job_id=job['job_id'],video_id=job['video_id'],status='lease_lost')
        return dict(job_id=job['job_id'],video_id=job['video_id'],status='failed',failure_reason=reason)
    finally:
        stop.set()
        heartbeat.join(timeout=1)


def run_worker(*,poll_seconds=2.0,max_jobs=None):
    if not 0.2<=poll_seconds<=60:
        raise ValueError('Poll interval must be 0.2 to 60 seconds')
    store=PostgresWorkspaceStore()
    completed=0
    while max_jobs is None or completed<max_jobs:
        result=process_next(store)
        if result is None:
            time.sleep(poll_seconds)
            continue
        completed+=1
        print(f"{result['status']}: {result['job_id']} ({result['video_id']})")
    return completed


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--once',action='store_true',help='Process at most one job and exit')
    parser.add_argument('--poll-seconds',type=float,default=2)
    parser.add_argument('--max-jobs',type=int)
    args=parser.parse_args()
    try:
        if args.once:
            result=process_next()
            print('idle' if result is None else f"{result['status']}: {result['job_id']} ({result['video_id']})")
        else:
            run_worker(poll_seconds=args.poll_seconds,max_jobs=args.max_jobs)
    except Exception as exc:
        parser.exit(1,f"ERROR: {type(exc).__name__}; check the local database and worker configuration.\n")


if __name__=='__main__':
    main()

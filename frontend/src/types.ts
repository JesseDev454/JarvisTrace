export type Role = 'administrator' | 'investigator';

export interface User {
  id: string;
  display_name: string;
  email?: string;
  role: Role;
  cameras?: string[];
}

export interface Recording {
  media_id: string;
  video_id: string;
  camera_id: string;
  filename?: string;
  duration_seconds: number;
  playback_url: string;
  playback_status?: string;
  source_type?: 'upload' | 'simulation' | 'camera' | string;
  status?: 'ready' | 'queued' | 'processing' | 'failed' | string;
  capture_start?: string | null;
  capture_timezone?: string | null;
  coverage?: { start_seconds: number; end_seconds: number } | null;
}

export interface Evidence {
  timestamp_seconds?: number;
  frame_index?: number;
  variant?: string;
  evidence_url?: string;
  image_sha256?: string;
  description?: string;
  provenance_type?: string;
  start_seconds?: number;
  end_seconds?: number;
}

export interface Candidate {
  video_id: string;
  camera_id: string;
  source_sha256: string;
  start_seconds: number;
  end_seconds: number;
  rrf_score: number;
  playback_url?: string | null;
  playback_status?: string;
  channels?: Record<string, { rank: number; rrf_contribution: number; evidence: Evidence[] }>;
}

export interface SearchResponse {
  query: string;
  mode: string;
  results: Candidate[];
  result_status: 'matches' | 'no_strong_match' | 'no_candidates';
  raw_window_counts: Record<string, number>;
  accepted_window_counts: Record<string, number>;
  candidate_counts: { visual: number; text: number };
  limitations: string;
}

export interface Camera {
  camera_id: string;
  name: string;
  source_type: string;
  connection_status: string;
  searchable_until: string | null;
}

export interface Job {
  job_id: string;
  video_id?: string;
  filename: string;
  camera_id: string;
  duration_seconds: number;
  status: 'queued' | 'processing' | 'ready' | 'failed' | string;
  failure_reason?: string | null;
}

export interface EvaluationSummary {
  title: string;
  clips: number;
  cameras: number;
  positives: number;
  negatives: number;
  unresolved: number;
  full_frames: { hit_at_1: string; precision_at_5: string };
  frames_and_crops: { hit_at_1: string; precision_at_5: string };
  absent_item_queries_returning_candidates: string;
  warning: string;
}

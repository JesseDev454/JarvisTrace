export class ApiError extends Error {
  constructor(message: string, readonly status: number) {
    super(message);
    this.name = 'ApiError';
  }
}

export async function requestJSON<T>(path: string, init: RequestInit = {}): Promise<T> {
  const headers = new Headers(init.headers);
  headers.set('Accept', 'application/json');
  if (init.body && !(init.body instanceof FormData) && !headers.has('Content-Type')) {
    headers.set('Content-Type', 'application/json');
  }
  const response = await fetch(path, { credentials: 'same-origin', ...init, headers });
  const payload = await response.json().catch(() => null) as { detail?: unknown } | null;
  if (!response.ok) {
    const detail = typeof payload?.detail === 'string' ? payload.detail : 'The request could not be completed.';
    throw new ApiError(detail, response.status);
  }
  return payload as T;
}

export function formatTime(seconds: number): string {
  const whole = Math.max(0, Math.floor(Number.isFinite(seconds) ? seconds : 0));
  const hours = Math.floor(whole / 3600);
  const minutes = Math.floor((whole % 3600) / 60);
  const remainder = whole % 60;
  return hours > 0
    ? `${String(hours).padStart(2, '0')}:${String(minutes).padStart(2, '0')}:${String(remainder).padStart(2, '0')}`
    : `${String(minutes).padStart(2, '0')}:${String(remainder).padStart(2, '0')}`;
}

export function parseTime(value: string): number | undefined {
  const cleaned = value.trim();
  if (!cleaned) return undefined;
  if (!/^\d{1,3}:\d{2}(?::\d{2})?$/.test(cleaned)) return Number.NaN;
  const parts = cleaned.split(':').map(Number);
  if (parts.some((part) => !Number.isInteger(part)) || parts.slice(1).some((part) => part > 59)) return Number.NaN;
  return parts.length === 2 ? parts[0] * 60 + parts[1] : parts[0] * 3600 + parts[1] * 60 + parts[2];
}

export function recordingLabel(recording: { filename?: string; video_id: string }): string {
  return recording.filename || recording.video_id;
}

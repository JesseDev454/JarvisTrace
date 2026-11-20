import { useCallback, useEffect, useMemo, useRef, useState, type FormEvent } from 'react';
import { ApiError, formatTime, parseTime, recordingLabel, requestJSON } from './api';
import type { Camera, Candidate, EvaluationSummary, Job, Recording, Role, SearchResponse, User } from './types';

type Screen = 'search' | 'recordings' | 'import' | 'processing' | 'users' | 'cameras' | 'about' | 'evaluation';
type AuthMode = 'loading' | 'login' | 'setup';
type SearchPhase = 'idle' | 'loading' | 'results' | 'empty' | 'error';
type EmptySearchStatus = 'no_strong_match' | 'no_candidates';

interface SessionPayload { user: User | null; setup_required?: boolean }

const managementScreens: Screen[] = ['recordings', 'import', 'processing', 'users', 'cameras'];
const managementLinks: Array<{ screen: Screen; label: string }> = [
  { screen: 'recordings', label: 'Recordings' },
  { screen: 'import', label: 'Import footage' },
  { screen: 'processing', label: 'Processing' },
  { screen: 'users', label: 'Users & access' },
  { screen: 'cameras', label: 'Cameras' },
];

function roleLabel(role: Role): string {
  return role === 'administrator' ? 'Administrator' : 'Investigator';
}

function initialScreen(): Screen {
  const value = window.location.hash.replace(/^#\/?/, '');
  return managementScreens.includes(value as Screen) || value === 'about' || value === 'evaluation'
    ? value as Screen
    : 'search';
}

function evidenceFor(candidate: Candidate) {
  return Object.values(candidate.channels ?? {}).flatMap((channel) => channel.evidence ?? []);
}

interface CoverageGap {
  cameraId: string;
  startsAt: number;
  endsAt: number;
}

function measuredCoverageGaps(recordings: Recording[]): CoverageGap[] {
  const byCamera = new Map<string, Array<{ startsAt: number; endsAt: number }>>();
  for (const recording of recordings) {
    if (recording.status !== 'ready' || !recording.capture_start) continue;
    const captureStart = Date.parse(recording.capture_start);
    if (!Number.isFinite(captureStart)) continue;
    const startSeconds = recording.coverage?.start_seconds ?? 0;
    const endSeconds = recording.coverage?.end_seconds ?? recording.duration_seconds;
    if (!Number.isFinite(startSeconds) || !Number.isFinite(endSeconds) || endSeconds <= startSeconds) continue;
    const intervals = byCamera.get(recording.camera_id) ?? [];
    intervals.push({ startsAt: captureStart + startSeconds * 1000, endsAt: captureStart + endSeconds * 1000 });
    byCamera.set(recording.camera_id, intervals);
  }

  const gaps: CoverageGap[] = [];
  for (const [cameraId, intervals] of byCamera) {
    intervals.sort((left, right) => left.startsAt - right.startsAt);
    let coveredUntil = intervals[0]?.endsAt ?? 0;
    for (const interval of intervals.slice(1)) {
      if (interval.startsAt - coveredUntil > 5_000) {
        gaps.push({ cameraId, startsAt: coveredUntil, endsAt: interval.startsAt });
      }
      coveredUntil = Math.max(coveredUntil, interval.endsAt);
    }
  }
  return gaps.sort((left, right) => left.startsAt - right.startsAt);
}

function formatUtcTimestamp(timestamp: number): string {
  return `${new Date(timestamp).toISOString().slice(0, 19).replace('T', ' ')} UTC`;
}

function formatGapDuration(milliseconds: number): string {
  const seconds = Math.max(0, Math.round(milliseconds / 1000));
  const minutes = Math.floor(seconds / 60);
  const remainingSeconds = seconds % 60;
  return minutes ? `about ${minutes}m ${remainingSeconds}s` : `about ${seconds}s`;
}

function StatusDot({ value }: { value: string }) {
  const state = value.toLowerCase().replaceAll(' ', '-');
  return <span className={`status-dot status-${state}`} aria-hidden="true" />;
}

function LoginScreen({ mode, onSignedIn }: { mode: Exclude<AuthMode, 'loading'>; onSignedIn: (user: User) => void }) {
  const [name, setName] = useState('');
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [error, setError] = useState('');
  const [busy, setBusy] = useState(false);
  const setup = mode === 'setup';

  async function submit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    setBusy(true);
    setError('');
    try {
      const payload = await requestJSON<SessionPayload>(setup ? '/api/setup/admin' : '/api/login', {
        method: 'POST',
        body: JSON.stringify(setup ? { display_name: name, email, password } : { email, password }),
      });
      if (!payload.user) throw new Error('Your account could not be loaded. Try again or contact an administrator.');
      onSignedIn(payload.user);
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : 'Sign-in did not complete. Try again.');
    } finally {
      setBusy(false);
    }
  }

  return (
    <main className="auth-shell">
      <section className="auth-card" aria-labelledby="auth-title">
        <Brand />
        <p className="eyebrow">LOCAL VIDEO SEARCH</p>
        <h1 id="auth-title">{setup ? 'Set up your administrator account' : 'Sign in to JarvisTrace'}</h1>
        <p className="muted">Search recordings that your account is allowed to access.</p>
        <form className="stack-form" onSubmit={submit}>
          {setup && <label>Your name<input autoComplete="name" value={name} onChange={(e) => setName(e.target.value)} required /></label>}
          <label>Email address<input type="email" autoComplete="username" value={email} onChange={(e) => setEmail(e.target.value)} required /></label>
          <label>Password<input type="password" autoComplete={setup ? 'new-password' : 'current-password'} value={password} onChange={(e) => setPassword(e.target.value)} minLength={12} required /></label>
          {setup && <p className="field-hint">Use at least 12 characters. The administrator can create investigator accounts after setup.</p>}
          {error && <p role="alert" className="notice notice-error">{error}</p>}
          <button className="button button-primary button-wide" disabled={busy} type="submit">{busy ? 'Please wait…' : setup ? 'Create administrator' : 'Sign in'}</button>
        </form>
      </section>
    </main>
  );
}

function Brand() {
  return <a className="brand" href="#/search" aria-label="JarvisTrace home"><span className="brand-mark" aria-hidden="true">J</span><span>JarvisTrace</span></a>;
}

function TopBar({ user, screen, navigate, signOut }: { user: User; screen: Screen; navigate: (target: Screen) => void; signOut: () => void }) {
  const admin = user.role === 'administrator';
  return (
    <header className="topbar">
      <Brand />
      <nav className="main-nav" aria-label="Main navigation">
        <a href="#/search" aria-current={screen === 'search' ? 'page' : undefined} onClick={(e) => { e.preventDefault(); navigate('search'); }}>Search footage</a>
        {admin && <a href="#/recordings" aria-current={managementScreens.includes(screen) ? 'page' : undefined} onClick={(e) => { e.preventDefault(); navigate('recordings'); }}>Manage recordings</a>}
        <a href="#/about" aria-current={screen === 'about' || screen === 'evaluation' ? 'page' : undefined} onClick={(e) => { e.preventDefault(); navigate('about'); }}>About</a>
      </nav>
      <div className="account-menu">
        <span className="avatar" aria-hidden="true">{user.display_name.trim().slice(0, 1).toUpperCase()}</span>
        <span className="account-copy"><strong>{user.display_name}</strong><small>{roleLabel(user.role)}</small></span>
        <button className="button button-quiet button-signout" onClick={signOut}>Sign out</button>
      </div>
    </header>
  );
}

function SearchPage({
  recordings, phase, results, selected, query, setQuery, camera, setCamera, videoId, setVideoId,
  startText, setStartText, endText, setEndText, error, setError, busy, onSearch, onClear, onSelect,
  playerError, clearPlayerError, onPlaybackError, emptyResultStatus,
}: {
  recordings: Recording[]; phase: SearchPhase; results: Candidate[]; selected: Candidate | null;
  query: string; setQuery: (value: string) => void; camera: string; setCamera: (value: string) => void;
  videoId: string; setVideoId: (value: string) => void; startText: string; setStartText: (value: string) => void;
  endText: string; setEndText: (value: string) => void; error: string; setError: (value: string) => void;
  busy: boolean; onSearch: (event: FormEvent<HTMLFormElement>) => void; onClear: () => void;
  onSelect: (candidate: Candidate) => void; playerError: boolean; clearPlayerError: () => void; onPlaybackError: () => void;
  emptyResultStatus: EmptySearchStatus;
}) {
  const cameraIds = useMemo(() => Array.from(new Set(recordings.map((row) => row.camera_id))).sort(), [recordings]);
  const coverageGaps = useMemo(() => measuredCoverageGaps(recordings), [recordings]);
  const filteredRecordings = camera ? recordings.filter((row) => row.camera_id === camera) : recordings;
  const playerRef = useRef<HTMLVideoElement>(null);
  const selectedEvidence = selected ? evidenceFor(selected) : [];
  const crop = selectedEvidence.find((item) => item.variant === 'crop' && item.evidence_url)
    ?? selectedEvidence.find((item) => item.evidence_url);

  useEffect(() => {
    const video = playerRef.current;
    if (!selected || !video || !selected.playback_url || playerError) return;
    const seekAndPlay = () => {
      video.currentTime = selected.start_seconds;
      void video.play().catch(() => undefined);
    };
    if (video.readyState >= HTMLMediaElement.HAVE_METADATA) seekAndPlay();
    else video.addEventListener('loadedmetadata', seekAndPlay, { once: true });
    return () => video.removeEventListener('loadedmetadata', seekAndPlay);
  }, [selected, playerError]);

  const replay = () => {
    const video = playerRef.current;
    if (!video || !selected) return;
    clearPlayerError();
    video.currentTime = selected.start_seconds;
    void video.play().catch(() => undefined);
  };

  return (
    <main className="page-wrap search-page">
      <div className="page-heading search-heading">
        <div><p className="eyebrow">INVESTIGATOR WORKSPACE</p><h1>Search footage</h1></div>
        <p className="heading-note">Describe what you saw. Review the possible matches in the original recording.</p>
      </div>

      <form className="search-form" onSubmit={onSearch}>
        <div className="query-row">
          <label className="query-field">Describe what you are looking for
            <input aria-label="Describe what you are looking for" value={query} onChange={(e) => { setQuery(e.target.value); setError(''); }} placeholder="For example, a dark backpack beside the waiting room bench" required maxLength={1000} />
          </label>
          <button className="button button-primary search-button" type="submit" disabled={busy}><span aria-hidden="true">⌕</span>{busy ? 'Searching…' : 'Search footage'}</button>
        </div>
        <div className="filters-row">
          <label>Camera<select aria-label="Camera" value={camera} onChange={(e) => { setCamera(e.target.value); setVideoId(''); }}><option value="">All cameras</option>{cameraIds.map((id) => <option key={id} value={id}>{id}</option>)}</select></label>
          <label>Recording<select aria-label="Recording" value={videoId} onChange={(e) => setVideoId(e.target.value)}><option value="">All recordings</option>{filteredRecordings.map((row) => <option key={row.video_id} value={row.video_id}>{recordingLabel(row)}</option>)}</select></label>
          <label>From (MM:SS)<input aria-label="From time" inputMode="numeric" placeholder="00:00" value={startText} onChange={(e) => setStartText(e.target.value)} /></label>
          <label>To (MM:SS)<input aria-label="To time" inputMode="numeric" placeholder="Any time" value={endText} onChange={(e) => setEndText(e.target.value)} /></label>
          <span className="filter-summary">{recordings.length ? `${recordings.length} recording${recordings.length === 1 ? '' : 's'} available` : 'No recordings indexed yet'}</span>
        </div>
      </form>

      {coverageGaps.length > 0 && <section className="coverage-gap-panel" role="region" aria-label="Recording coverage gaps">
        <div className="coverage-gap-heading"><span className="coverage-gap-mark" aria-hidden="true">↗</span><div><h2>Gaps between indexed recordings</h2><p>No indexed recording covers these measured intervals between saved segments.</p></div></div>
        <ul>{coverageGaps.map((gap) => <li key={`${gap.cameraId}-${gap.startsAt}`}><strong>{gap.cameraId}</strong><span>{formatUtcTimestamp(gap.startsAt)} – {formatUtcTimestamp(gap.endsAt)}</span><small>{formatGapDuration(gap.endsAt - gap.startsAt)}</small></li>)}</ul>
        <p className="coverage-gap-note">Times are UTC. Segment timestamps do not confirm continuous capture within a recording.</p>
      </section>}

      {error && <p role="alert" className="notice notice-error search-notice">{error}</p>}
      {phase === 'loading' && <div className="state-panel loading-panel" role="status" aria-live="polite"><span className="spinner" aria-hidden="true" />Searching indexed recordings…</div>}
      {phase === 'idle' && <section className="welcome-panel"><div className="welcome-icon" aria-hidden="true">⌕</div><h2>Start with what you remember</h2><p>Use ordinary descriptions such as “a person carrying a blue case”. Choose a camera or time range if you know it.</p><span className="privacy-note"><span className="privacy-dot" />Search stays on this workstation</span></section>}
      {phase === 'empty' && <section className="state-panel empty-panel"><div className="state-icon" aria-hidden="true">◌</div><h2>{emptyResultStatus === 'no_strong_match' ? 'No strong match found' : 'No indexed candidates for this search'}</h2><p>{emptyResultStatus === 'no_strong_match' ? 'The indexed footage produced candidates, but none met the configured relevance check. This does not establish that the item was absent.' : 'No indexed candidates matched this description and these filters. This does not establish that the item was absent.'}</p><button type="button" className="button button-secondary" onClick={onClear}>Clear filters</button></section>}
      {phase === 'results' && <section className="results-layout" aria-label="Search results and playback">
        <div className="results-column">
          <div className="section-heading"><div><p className="eyebrow">REVIEW EACH CLIP</p><h2>Possible matches</h2></div><span className="result-count">{results.length} candidate{results.length === 1 ? '' : 's'}</span></div>
          <ol className="result-list">
            {results.map((candidate, index) => {
              const evidence = evidenceFor(candidate).find((item) => item.evidence_url);
              const active = selected === candidate;
              return <li key={`${candidate.video_id}-${candidate.start_seconds}-${index}`}>
                <article className={`result-card${active ? ' result-card-selected' : ''}`}>
                  <button className="result-select" onClick={() => onSelect(candidate)} aria-label={`Camera ${candidate.camera_id} from ${formatTime(candidate.start_seconds)} to ${formatTime(candidate.end_seconds)}, select possible match`} aria-pressed={active}>
                    {evidence?.evidence_url ? <img className="result-thumb" src={evidence.evidence_url} alt="" loading="lazy" /> : <span className="result-thumb result-thumb-placeholder" aria-hidden="true"><span>JT</span></span>}
                    <span className="result-copy"><strong>Camera {candidate.camera_id}</strong><span>{formatTime(candidate.start_seconds)} – {formatTime(candidate.end_seconds)}</span><small>{recordings.find((r) => r.video_id === candidate.video_id)?.filename || candidate.video_id}</small></span>
                    <span className="result-open" aria-hidden="true">↗</span>
                  </button>
                </article>
              </li>;
            })}
          </ol>
          <p className="ranking-note">Results are ranked candidates, not confidence scores or proof of presence or absence.</p>
        </div>

        <section className="player-column" ref={undefined} aria-label="Selected footage" role="region">
          {selected ? <>
            <div className="player-heading"><div><p className="eyebrow">SELECTED CLIP</p><h2>Camera {selected.camera_id}</h2></div><span className="clip-range">{formatTime(selected.start_seconds)} <span>to</span> {formatTime(selected.end_seconds)}</span></div>
            {selected.playback_url && !playerError ? <video
              key={selected.playback_url}
              ref={playerRef}
              className="recording-player"
              src={selected.playback_url}
              controls
              playsInline
              preload="metadata"
              aria-label={`Recording from camera ${selected.camera_id}`}
              onError={onPlaybackError}
              onTimeUpdate={(event) => {
                const video = event.currentTarget;
                if (video.currentTime >= selected.end_seconds && !video.paused) {
                  video.pause();
                  video.currentTime = selected.end_seconds;
                }
              }}
            /> : <div className="player-unavailable" role="status"><span aria-hidden="true">▣</span><strong>Playback unavailable</strong><p>The candidate is still selected. Check that this recording is available, then try again.</p><button className="button button-secondary" onClick={replay}>Try again</button></div>}
            <div className="player-actions"><button className="button button-secondary" onClick={replay} disabled={!selected.playback_url}><span aria-hidden="true">↻</span> Replay this clip</button><button className="text-button" onClick={() => { const video = playerRef.current; if (video) { video.currentTime = 0; void video.play().catch(() => undefined); } }}>Play surrounding footage</button></div>
            <div className="object-preview">
              {crop?.evidence_url ? <img src={crop.evidence_url} alt="Object preview from the selected clip" loading="lazy" /> : <div className="object-placeholder" aria-hidden="true">Frame evidence</div>}
              <div><p className="eyebrow">VISUAL EVIDENCE</p><h3>Object preview</h3><p>Sampled at {formatTime(crop?.timestamp_seconds ?? selected.start_seconds)}. Check the full clip before drawing a conclusion.</p></div>
            </div>
            <details className="evidence-details"><summary>Why this window matched</summary><ul>{selectedEvidence.map((item, index) => <li key={`${item.frame_index ?? item.timestamp_seconds}-${index}`}><strong>{item.variant === 'crop' ? 'Detector crop' : item.description ? 'Reviewed description' : 'Full-frame sample'}</strong><span>{item.timestamp_seconds !== undefined ? ` · ${formatTime(item.timestamp_seconds)}` : ''}</span>{item.description && <p>{item.description}</p>}</li>)}</ul></details>
          </> : <div className="player-empty"><div className="player-placeholder-mark" aria-hidden="true">▶</div><p>Select a possible match to review its original recording.</p></div>}
        </section>
      </section>}
      <p className="page-footnote">A result helps you review a recording. It does not identify a person or establish what happened.</p>
    </main>
  );
}

function AboutPage({ onEvaluate }: { onEvaluate: () => void }) {
  return <main className="page-wrap narrow-page about-page">
    <p className="eyebrow">ABOUT THE SYSTEM</p><h1>About JarvisTrace</h1>
    <p className="intro-copy">JarvisTrace helps an investigator find moments in recorded CCTV footage by describing an item or scene in everyday language.</p>
    <section className="about-steps" aria-label="How to use JarvisTrace">
      <article><span>01</span><div><h2>Describe what you remember</h2><p>Write a simple description such as “a dark backpack beside a bench”.</p></div></article>
      <article><span>02</span><div><h2>Choose a possible clip</h2><p>Review the camera, recording, time range, and visual evidence for each ranked candidate.</p></div></article>
      <article><span>03</span><div><h2>Check the original footage</h2><p>Play the recording around the suggested time. A visual match is a lead for review, not a conclusion.</p></div></article>
    </section>
    <section className="about-callout"><span className="callout-icon" aria-hidden="true">i</span><div><h2>What a search result means</h2><p>The system ranks possible matches from indexed samples. It can return a candidate for an item that is not present, and it may miss an item. Results do not identify people or establish actions.</p></div></section>
    <div className="about-links"><a className="button button-primary" href="#/search">Back to search</a><a className="text-button" href="#/evaluation" onClick={(e) => { e.preventDefault(); onEvaluate(); }}>Research evaluation <span aria-hidden="true">→</span></a></div>
    <p className="about-footer">Group 8 · Final year project</p>
  </main>;
}

function EvaluationPage({ data, onBack }: { data: EvaluationSummary | null; onBack: () => void }) {
  return <main className="page-wrap narrow-page evaluation-page">
    <button className="back-link" onClick={onBack}><span aria-hidden="true">←</span> Back to About</button>
    <p className="eyebrow">PROJECT VALIDATION</p><h1>Research evaluation</h1>
    <p className="intro-copy">{data ? `${data.title} · ${data.clips} recorded clips from ${data.cameras} camera${data.cameras === 1 ? '' : 's'}` : 'Loading the latest reviewed results…'}</p>
    {data && <>
      <div className="table-scroll"><table className="evaluation-table"><thead><tr><th scope="col">Measure</th><th scope="col">Full frames</th><th scope="col">Frames + crops</th></tr></thead><tbody>
        <tr><th scope="row">Relevant clip ranked first</th><td>{data.full_frames.hit_at_1}</td><td>{data.frames_and_crops.hit_at_1}</td></tr>
        <tr><th scope="row">Precision in the top five</th><td>{data.full_frames.precision_at_5}</td><td>{data.frames_and_crops.precision_at_5}</td></tr>
        <tr><th scope="row">Absent-item searches that returned candidates</th><td colSpan={2}>{data.absent_item_queries_returning_candidates}</td></tr>
      </tbody></table></div>
      <p className="evaluation-scope">{data.positives} positive queries and {data.negatives} confirmed absent-item queries. {data.unresolved} unresolved query was excluded.</p>
      <section className="evaluation-meaning"><p className="eyebrow">HOW TO READ THIS</p><h2>Object crops improved these rankings</h2><p>The sample is small and comes from a single camera. Searches for absent items still returned possible matches. This result does not establish general accuracy.</p><div className="evaluation-meta"><span>{data.clips} clips reviewed</span><span>{data.cameras} camera</span><span>{data.positives + data.negatives} scored queries</span></div></section>
      <a className="report-link" href="/research-evaluation-report">Read the full results <span aria-hidden="true">↗</span></a>
      <p className="fine-print">{data.warning}</p>
    </>}
  </main>;
}

function ManagementTabs({ screen, navigate }: { screen: Screen; navigate: (target: Screen) => void }) {
  return <nav className="management-tabs" aria-label="Recording management">
    {managementLinks.map(({ screen: target, label }) => <a key={target} href={`#/${target}`} aria-current={screen === target ? 'page' : undefined} onClick={(e) => { e.preventDefault(); navigate(target); }}>{label}</a>)}
  </nav>;
}

function RecordingsPage({ recordings, loading, error, navigate }: { recordings: Recording[]; loading: boolean; error: string; navigate: (target: Screen) => void }) {
  return <section className="management-content">
    <div className="management-heading"><div><p className="eyebrow">FOOTAGE LIBRARY</p><h2>Recordings</h2><p className="muted">Searchable clips and their verified recording coverage.</p></div><button className="button button-primary" onClick={() => navigate('import')}><span aria-hidden="true">＋</span> Import footage</button></div>
    {error && <p className="notice notice-error" role="alert">{error}</p>}
    {loading ? <div role="status" className="state-panel compact-state">Loading recordings…</div> : recordings.length === 0 ? <div className="state-panel compact-state"><h3>No recordings yet</h3><p>Import an H.264 MP4 or configure the MediaMTX simulation to add searchable footage.</p><button className="button button-secondary" onClick={() => navigate('import')}>Import footage</button></div> : <div className="table-scroll"><table className="data-table"><thead><tr><th scope="col">Recording</th><th scope="col">Camera</th><th scope="col">Coverage</th><th scope="col">Duration</th><th scope="col">Source</th><th scope="col">Status</th></tr></thead><tbody>{recordings.map((row) => <tr key={row.video_id}><td><strong>{recordingLabel(row)}</strong><small>{row.video_id}</small></td><td>{row.camera_id}</td><td>{row.coverage ? `${formatTime(row.coverage.start_seconds)} – ${formatTime(row.coverage.end_seconds)}` : row.capture_start ? `${row.capture_start} · ${row.capture_timezone || 'time zone unverified'}` : 'Relative time only'}</td><td>{formatTime(row.duration_seconds)}</td><td>{row.source_type || 'Indexed recording'}</td><td><span className="table-status"><StatusDot value={row.status || 'ready'} />{row.status || 'Ready'}</span></td></tr>)}</tbody></table></div>}
  </section>;
}

function ImportPage({ cameras, onImported }: { cameras: Camera[]; onImported: (message: string) => void }) {
  const [file, setFile] = useState<File | null>(null);
  const [camera, setCamera] = useState('');
  const [start, setStart] = useState('');
  const [timezone, setTimezone] = useState('Africa/Lagos');
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState('');
  const [success, setSuccess] = useState('');

  async function submit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    setError(''); setSuccess('');
    if (!file) { setError('Choose an MP4 video to continue.'); return; }
    if (!file.name.toLowerCase().endsWith('.mp4') || file.type && file.type !== 'video/mp4') {
      setError('This file is not an MP4. Convert it to H.264 MP4 with FFmpeg, then choose it again.'); return;
    }
    if (file.size > 1024 * 1024 * 1024) { setError('The file is larger than the 1 GB upload limit. Choose a shorter recording.'); return; }
    const body = new FormData();
    body.set('file', file); body.set('camera_id', camera); body.set('capture_start_local', start); body.set('capture_timezone', timezone);
    setBusy(true);
    try {
      const result = await requestJSON<{ job_id: string; status: string }>('/api/recordings/upload', { method: 'POST', body });
      const message = `${file.name} is ${result.status}. It will appear in search after indexing completes.`;
      setSuccess(message); onImported(message);
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : 'Upload did not complete. Try again.');
    } finally { setBusy(false); }
  }

  return <section className="management-content import-content">
    <div className="management-heading"><div><p className="eyebrow">ADD SEARCHABLE FOOTAGE</p><h2>Import footage</h2><p className="muted">Choose a recording and camera. Search is available after processing finishes.</p></div></div>
    <form className="import-form" onSubmit={submit}>
      <label className="upload-drop"><span className="upload-icon" aria-hidden="true">↑</span><strong>Choose video file</strong><span>H.264 MP4 · up to 1 GB</span><input aria-label="Choose video file" type="file" accept="video/mp4,.mp4" onChange={(e) => { setFile(e.target.files?.[0] ?? null); setError(''); }} required /></label>
      {file && <p className="selected-file"><strong>{file.name}</strong><span>{(file.size / (1024 * 1024)).toFixed(1)} MB</span></p>}
      <div className="form-grid"><label>Camera<select value={camera} onChange={(e) => setCamera(e.target.value)} required><option value="">Select a camera</option>{cameras.map((item) => <option value={item.camera_id} key={item.camera_id}>{item.camera_id} · {item.name}</option>)}</select></label><label>Verified capture start <span className="optional">(optional)</span><input type="datetime-local" value={start} onChange={(e) => setStart(e.target.value)} /><small>Use only when the source time is confirmed. Otherwise leave it blank.</small></label><label>Time zone<select value={timezone} onChange={(e) => setTimezone(e.target.value)}><option>Africa/Lagos</option><option>UTC</option><option>Europe/London</option><option>America/New_York</option><option>Asia/Dubai</option></select></label></div>
      {error && <p role="alert" className="notice notice-error">{error}</p>}{success && <p role="status" className="notice notice-success">{success}</p>}
      <div className="form-actions"><button className="button button-primary" disabled={busy || !file || !camera} type="submit">{busy ? 'Uploading…' : 'Import recording'}</button><span>Original footage is kept in local storage.</span></div>
    </form>
  </section>;
}

function ProcessingPage({ jobs, loading, error, onRetry }: { jobs: Job[]; loading: boolean; error: string; onRetry: (job: Job) => void }) {
  return <section className="management-content">
    <div className="management-heading"><div><p className="eyebrow">INDEXING QUEUE</p><h2>Processing</h2><p className="muted">A recording becomes searchable after indexing is complete.</p></div></div>
    {error && <p role="alert" className="notice notice-error">{error}</p>}
    {loading ? <div role="status" className="state-panel compact-state">Loading processing jobs…</div> : jobs.length === 0 ? <div className="state-panel compact-state"><h3>No recordings in the queue</h3><p>New imports will appear here while they are being prepared.</p></div> : <div className="table-scroll"><table className="data-table"><thead><tr><th scope="col">Recording</th><th scope="col">Camera</th><th scope="col">Duration</th><th scope="col">Status</th><th scope="col">Failure reason</th><th scope="col"><span className="sr-only">Actions</span></th></tr></thead><tbody>{jobs.map((job) => <tr key={job.job_id}><td><strong>{job.filename}</strong></td><td>{job.camera_id}</td><td>{formatTime(job.duration_seconds)}</td><td><span className="table-status"><StatusDot value={job.status} />{job.status[0].toUpperCase() + job.status.slice(1)}</span></td><td>{job.failure_reason || '—'}</td><td>{job.status === 'failed' && <button className="button button-outline button-small" onClick={() => onRetry(job)}>Retry</button>}</td></tr>)}</tbody></table></div>}
  </section>;
}

function UsersPage({ users, loading, error, onCreated }: { users: User[]; loading: boolean; error: string; onCreated: (user: User) => void }) {
  const [showForm, setShowForm] = useState(false);
  const [name, setName] = useState(''); const [email, setEmail] = useState(''); const [password, setPassword] = useState('');
  const [role, setRole] = useState<Role>('investigator'); const [cameraIds, setCameraIds] = useState(''); const [formError, setFormError] = useState(''); const [busy, setBusy] = useState(false);
  async function submit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault(); setFormError(''); setBusy(true);
    try {
      const response = await requestJSON<{ user: User }>('/api/users', { method: 'POST', body: JSON.stringify({ display_name: name, email, password, role, cameras: cameraIds.split(',').map((v) => v.trim()).filter(Boolean) }) });
      onCreated(response.user); setName(''); setEmail(''); setPassword(''); setCameraIds(''); setShowForm(false);
    } catch (cause) { setFormError(cause instanceof Error ? cause.message : 'User could not be added.'); }
    finally { setBusy(false); }
  }
  return <section className="management-content">
    <div className="management-heading"><div><p className="eyebrow">ACCOUNT PERMISSIONS</p><h2>Users & access</h2><p className="muted">Investigators can search and play footage assigned to them. Administrators manage recordings and cameras.</p></div><button className="button button-primary" onClick={() => setShowForm((value) => !value)}>{showForm ? 'Close form' : '＋ Add user'}</button></div>
    {error && <p className="notice notice-error" role="alert">{error}</p>}
    {showForm && <form className="inline-form" onSubmit={submit}><h3>Add a user</h3><div className="form-grid"><label>Name<input value={name} onChange={(e) => setName(e.target.value)} required /></label><label>Email<input type="email" value={email} onChange={(e) => setEmail(e.target.value)} required /></label><label>Temporary password<input type="password" minLength={12} value={password} onChange={(e) => setPassword(e.target.value)} required /><small>At least 12 characters. Share this with the account holder directly.</small></label><label>Role<select value={role} onChange={(e) => setRole(e.target.value as Role)}><option value="investigator">Investigator</option><option value="administrator">Administrator</option></select></label><label>Assigned cameras <span className="optional">(comma separated IDs)</span><input value={cameraIds} onChange={(e) => setCameraIds(e.target.value)} placeholder="G507, G508" /></label></div>{formError && <p className="notice notice-error" role="alert">{formError}</p>}<button className="button button-primary" disabled={busy}>{busy ? 'Adding…' : 'Create account'}</button></form>}
    {loading ? <div role="status" className="state-panel compact-state">Loading users…</div> : <div className="table-scroll"><table className="data-table"><thead><tr><th scope="col">Name</th><th scope="col">Email</th><th scope="col">Role</th><th scope="col">Assigned cameras</th></tr></thead><tbody>{users.map((item) => <tr key={item.id}><td><strong>{item.display_name}</strong></td><td>{item.email || '—'}</td><td><span className={`role-pill role-${item.role}`}>{roleLabel(item.role)}</span></td><td>{item.cameras?.length ? item.cameras.join(', ') : 'No cameras assigned'}</td></tr>)}</tbody></table></div>}
  </section>;
}

function CamerasPage({ cameras, loading, error, onCreated }: { cameras: Camera[]; loading: boolean; error: string; onCreated: (camera: Camera) => void }) {
  const [showForm, setShowForm] = useState(false); const [cameraId, setCameraId] = useState(''); const [name, setName] = useState(''); const [busy, setBusy] = useState(false); const [formError, setFormError] = useState('');
  async function submit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault(); setBusy(true); setFormError('');
    try { const response = await requestJSON<{ camera: Camera }>('/api/cameras', { method: 'POST', body: JSON.stringify({ camera_id: cameraId.trim(), name: name.trim(), source_type: 'mediamtx' }) }); onCreated(response.camera); setCameraId(''); setName(''); setShowForm(false); }
    catch (cause) { setFormError(cause instanceof Error ? cause.message : 'Camera could not be added.'); }
    finally { setBusy(false); }
  }
  return <section className="management-content">
    <div className="management-heading"><div><p className="eyebrow">VIDEO SOURCES</p><h2>Cameras</h2><p className="muted">Connection status and searchable coverage are shown separately.</p></div><button className="button button-primary" onClick={() => setShowForm((value) => !value)}>{showForm ? 'Close form' : '＋ Add camera'}</button></div>
    <p className="info-strip"><span className="info-mark" aria-hidden="true">i</span> Camera stream addresses and passwords are configured in MediaMTX and stay out of browser responses.</p>
    {error && <p role="alert" className="notice notice-error">{error}</p>}
    {showForm && <form className="inline-form compact-form" onSubmit={submit}><h3>Add a camera record</h3><div className="form-grid"><label>Camera ID<input value={cameraId} onChange={(e) => setCameraId(e.target.value)} required maxLength={120} /></label><label>Display name<input value={name} onChange={(e) => setName(e.target.value)} required maxLength={160} /></label></div>{formError && <p role="alert" className="notice notice-error">{formError}</p>}<button className="button button-primary" disabled={busy}>{busy ? 'Adding…' : 'Add camera'}</button></form>}
    {loading ? <div role="status" className="state-panel compact-state">Loading cameras…</div> : cameras.length === 0 ? <div className="state-panel compact-state"><h3>No camera sources yet</h3><p>Add a camera record, then configure its stream in MediaMTX.</p></div> : <div className="table-scroll"><table className="data-table"><thead><tr><th scope="col">Camera</th><th scope="col">Source</th><th scope="col">Connection</th><th scope="col">Searchable through</th></tr></thead><tbody>{cameras.map((camera) => <tr key={camera.camera_id}><td><strong>{camera.name}</strong><small>{camera.camera_id}</small></td><td>{camera.source_type}</td><td><span className="table-status"><StatusDot value={camera.connection_status} />{camera.connection_status}</span></td><td>{camera.searchable_until || 'No indexed footage yet'}</td></tr>)}</tbody></table></div>}
    <p className="fine-print">A connected stream does not mean its latest segment has finished indexing.</p>
  </section>;
}

function App() {
  const [authMode, setAuthMode] = useState<AuthMode>('loading');
  const [user, setUser] = useState<User | null>(null);
  const [screen, setScreen] = useState<Screen>(initialScreen);
  const [recordings, setRecordings] = useState<Recording[]>([]);
  const [cameras, setCameras] = useState<Camera[]>([]);
  const [jobs, setJobs] = useState<Job[]>([]);
  const [users, setUsers] = useState<User[]>([]);
  const [listLoading, setListLoading] = useState(false);
  const [listError, setListError] = useState('');
  const [query, setQuery] = useState(''); const [camera, setCamera] = useState(''); const [videoId, setVideoId] = useState('');
  const [startText, setStartText] = useState(''); const [endText, setEndText] = useState('');
  const [phase, setPhase] = useState<SearchPhase>('idle'); const [results, setResults] = useState<Candidate[]>([]); const [selected, setSelected] = useState<Candidate | null>(null);
  const [emptyResultStatus, setEmptyResultStatus] = useState<EmptySearchStatus>('no_candidates');
  const [formError, setFormError] = useState(''); const [searchBusy, setSearchBusy] = useState(false);
  const [playerError, setPlayerError] = useState(false); const [importMessage, setImportMessage] = useState('');
  const [evaluation, setEvaluation] = useState<EvaluationSummary | null>(null);
  const requestId = useRef<AbortController | null>(null);

  useEffect(() => {
    let active = true;
    requestJSON<SessionPayload>('/api/session').then((payload) => {
      if (!active) return;
      if (payload.user) { setUser(payload.user); setAuthMode('loading'); }
      else setAuthMode(payload.setup_required ? 'setup' : 'login');
    }).catch((cause: unknown) => {
      if (!active) return;
      setAuthMode(cause instanceof ApiError && cause.status === 404 ? 'setup' : 'login');
    });
    return () => { active = false; };
  }, []);

  const navigate = useCallback((target: Screen) => {
    setScreen(target); setListError('');
    window.history.replaceState(null, '', `#/${target}`);
    if (target !== 'search') { requestId.current?.abort(); setSearchBusy(false); }
  }, []);

  useEffect(() => {
    const onHash = () => setScreen(initialScreen());
    window.addEventListener('hashchange', onHash);
    return () => window.removeEventListener('hashchange', onHash);
  }, []);

  const refreshRecordings = useCallback(async () => {
    const payload = await requestJSON<{ recordings: Recording[] }>('/api/recordings');
    setRecordings(payload.recordings);
  }, []);

  useEffect(() => {
    if (!user) return;
    let active = true;
    setListLoading(true); setListError('');
    requestJSON<{ recordings: Recording[] }>('/api/recordings').then((payload) => { if (active) setRecordings(payload.recordings); }).catch((cause) => { if (active) setListError(cause instanceof Error ? cause.message : 'Recordings could not be loaded.'); }).finally(() => { if (active) setListLoading(false); });
    requestJSON<{ cameras: Camera[] }>('/api/cameras').then((payload) => { if (active) setCameras(payload.cameras); }).catch(() => undefined);
    return () => { active = false; };
  }, [user?.id]);

  useEffect(() => {
    if (!user || screen !== 'processing') return;
    let active = true;
    const refresh = () => requestJSON<{ jobs: Job[] }>('/api/jobs').then((payload) => { if (active) setJobs(payload.jobs); }).catch((cause) => { if (active) setListError(cause instanceof Error ? cause.message : 'Processing queue is unavailable.'); });
    setListLoading(true); void refresh().finally(() => { if (active) setListLoading(false); });
    const timer = window.setInterval(() => { void refresh(); }, 5000);
    return () => { active = false; window.clearInterval(timer); };
  }, [user?.id, screen]);

  useEffect(() => {
    if (!user || screen !== 'users') return;
    let active = true;
    setListLoading(true); setListError('');
    requestJSON<{ users: User[] }>('/api/users').then((payload) => { if (active) setUsers(payload.users); }).catch((cause) => { if (active) setListError(cause instanceof Error ? cause.message : 'Users could not be loaded.'); }).finally(() => { if (active) setListLoading(false); });
    return () => { active = false; };
  }, [user?.id, screen]);

  useEffect(() => {
    if (!user || screen !== 'evaluation') return;
    let active = true;
    requestJSON<EvaluationSummary>('/api/research-evaluation').then((payload) => { if (active) setEvaluation(payload); }).catch(() => { if (active) setEvaluation(null); });
    return () => { active = false; };
  }, [user?.id, screen]);

  async function submitSearch(event: FormEvent<HTMLFormElement>) {
    event.preventDefault(); setFormError('');
    const cleanQuery = query.trim();
    if (!cleanQuery) { setFormError('Enter an item or scene description.'); return; }
    const start = parseTime(startText); const end = parseTime(endText);
    if (Number.isNaN(start) || Number.isNaN(end)) { setFormError('Enter time as MM:SS, for example 01:24.'); return; }
    if (start !== undefined && end !== undefined && end <= start) { setFormError('End time must be after the start time.'); return; }
    requestId.current?.abort();
    const controller = new AbortController(); requestId.current = controller;
    setSearchBusy(true); setPhase('loading'); setResults([]); setSelected(null); setPlayerError(false);
    const body: Record<string, string | number> = { query: cleanQuery, mode: 'hybrid', limit: 10 };
    if (camera) body.camera_id = camera;
    if (videoId) body.video_id = videoId;
    if (start !== undefined) body.start_seconds = start;
    if (end !== undefined) body.end_seconds = end;
    try {
      const payload = await requestJSON<SearchResponse>('/api/search', { method: 'POST', body: JSON.stringify(body), signal: controller.signal });
      if (controller.signal.aborted || requestId.current !== controller) return;
      setResults(payload.results);
      setEmptyResultStatus(payload.result_status === 'no_strong_match' ? 'no_strong_match' : 'no_candidates');
      setPhase(payload.results.length ? 'results' : 'empty');
    } catch (cause) {
      if (controller.signal.aborted || requestId.current !== controller) return;
      setPhase('error');
      setFormError(`Search did not complete. ${cause instanceof Error ? cause.message : 'Search service unavailable. Try again.'}`);
    } finally {
      if (requestId.current === controller) setSearchBusy(false);
    }
  }

  function clearSearch() {
    requestId.current?.abort(); setQuery(''); setCamera(''); setVideoId(''); setStartText(''); setEndText('');
    setFormError(''); setPhase('idle'); setResults([]); setSelected(null); setEmptyResultStatus('no_candidates'); setSearchBusy(false); setPlayerError(false);
  }

  async function signOut() {
    try { await requestJSON('/api/logout', { method: 'POST' }); } catch { /* Cookie expiry remains enforced by the server. */ }
    setUser(null); setAuthMode('login'); setScreen('search'); clearSearch();
  }

  async function loadManagementList(target: Screen) {
    setListError(''); setListLoading(true);
    try {
      if (target === 'recordings') await refreshRecordings();
      if (target === 'cameras') { const payload = await requestJSON<{ cameras: Camera[] }>('/api/cameras'); setCameras(payload.cameras); }
      if (target === 'users') { const payload = await requestJSON<{ users: User[] }>('/api/users'); setUsers(payload.users); }
      if (target === 'processing') { const payload = await requestJSON<{ jobs: Job[] }>('/api/jobs'); setJobs(payload.jobs); }
    } catch (cause) { setListError(cause instanceof Error ? cause.message : 'This information could not be loaded.'); }
    finally { setListLoading(false); }
  }

  async function retryJob(job: Job) {
    try { await requestJSON(`/api/jobs/${encodeURIComponent(job.job_id)}/retry`, { method: 'POST' }); const response = await requestJSON<{ jobs: Job[] }>('/api/jobs'); setJobs(response.jobs); }
    catch (cause) { setListError(cause instanceof Error ? cause.message : 'The job could not be retried.'); }
  }

  if (authMode === 'loading' && !user) return <main className="auth-shell"><div className="loading-brand"><Brand /><p role="status">Opening your workspace…</p></div></main>;
  if (!user) return <LoginScreen mode={authMode === 'setup' ? 'setup' : 'login'} onSignedIn={(next) => { setUser(next); setAuthMode('loading'); navigate('search'); }} />;

  const isAdmin = user.role === 'administrator';
  const openScreen = (target: Screen) => { navigate(target); if (managementScreens.includes(target)) void loadManagementList(target); };
  const onSelect = (candidate: Candidate) => {
    setSelected(candidate); setPlayerError(candidate.playback_status !== 'available');
    if (window.matchMedia('(max-width: 860px)').matches) window.setTimeout(() => document.querySelector('.player-column')?.scrollIntoView({ behavior: 'smooth', block: 'start' }), 50);
  };

  return <div className="app-shell">
    <TopBar user={user} screen={screen} navigate={openScreen} signOut={() => { void signOut(); }} />
    {screen === 'search' && <SearchPage recordings={recordings} phase={phase} results={results} selected={selected} query={query} setQuery={setQuery} camera={camera} setCamera={setCamera} videoId={videoId} setVideoId={setVideoId} startText={startText} setStartText={setStartText} endText={endText} setEndText={setEndText} error={formError} setError={setFormError} busy={searchBusy} onSearch={(event) => { void submitSearch(event); }} onClear={clearSearch} onSelect={onSelect} playerError={playerError} clearPlayerError={() => setPlayerError(false)} onPlaybackError={() => setPlayerError(true)} emptyResultStatus={emptyResultStatus} />}
    {managementScreens.includes(screen) && isAdmin && <main className="page-wrap management-page"><div className="page-heading"><div><p className="eyebrow">ADMINISTRATION</p><h1>Manage recordings</h1></div><p className="heading-note">Keep the footage library, cameras, and access ready for investigators.</p></div><ManagementTabs screen={screen} navigate={openScreen} />
      {screen === 'recordings' && <RecordingsPage recordings={recordings} loading={listLoading} error={listError} navigate={openScreen} />}
      {screen === 'import' && <ImportPage cameras={cameras} onImported={(message) => { setImportMessage(message); openScreen('processing'); }} />}
      {screen === 'processing' && <ProcessingPage jobs={jobs} loading={listLoading} error={listError} onRetry={(job) => { void retryJob(job); }} />}
      {screen === 'users' && <UsersPage users={users} loading={listLoading} error={listError} onCreated={(item) => setUsers((current) => [...current, item])} />}
      {screen === 'cameras' && <CamerasPage cameras={cameras} loading={listLoading} error={listError} onCreated={(item) => setCameras((current) => [...current, item])} />}
      {importMessage && screen === 'processing' && <p role="status" className="notice notice-success import-success">{importMessage}</p>}
    </main>}
    {screen === 'about' && <AboutPage onEvaluate={() => openScreen('evaluation')} />}
    {screen === 'evaluation' && <EvaluationPage data={evaluation} onBack={() => openScreen('about')} />}
    <footer className="site-footer"><span>JarvisTrace</span><span>Group 8 · Final year project</span><button className="text-button" onClick={() => openScreen('about')}>About this system</button></footer>
  </div>;
}

export default App;

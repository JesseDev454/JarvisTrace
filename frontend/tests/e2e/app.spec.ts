import { expect, test, type Page, type Route } from '@playwright/test';
import AxeBuilder from '@axe-core/playwright';
import { readFileSync } from 'node:fs';
import { mkdir } from 'node:fs/promises';
import { join } from 'node:path';
import { fileURLToPath } from 'node:url';

const tinyVideo = Buffer.from(readFileSync(fileURLToPath(new URL('../fixtures/tiny.mp4.b64', import.meta.url)), 'utf8'), 'base64');

const recordings = [
  {
    media_id: 'opaque-demo-01',
    video_id: 'demo01',
    camera_id: 'G507',
    filename: 'waiting-room.mp4',
    duration_seconds: 40,
    playback_url: '/media/opaque-demo-01',
    playback_status: 'available',
    source_type: 'upload',
    status: 'ready',
    coverage: { start_seconds: 0, end_seconds: 40 },
  },
  {
    media_id: 'opaque-demo-02',
    video_id: 'demo02',
    camera_id: 'G508',
    filename: 'reception.mp4',
    duration_seconds: 80,
    playback_url: '/media/opaque-demo-02',
    playback_status: 'available',
    source_type: 'upload',
    status: 'ready',
    coverage: { start_seconds: 0, end_seconds: 80 },
  },
];

const match = {
  video_id: 'demo01',
  camera_id: 'G507',
  source_sha256: 'a'.repeat(64),
  start_seconds: 4,
  end_seconds: 8,
  rrf_score: 0.02,
  playback_url: '/media/opaque-demo-01',
  playback_status: 'available',
  channels: {
    visual: {
      rank: 1,
      rrf_contribution: 0.02,
      evidence: [{
        timestamp_seconds: 5,
        frame_index: 150,
        variant: 'crop',
        evidence_url: '/evidence/opaque-crop-01',
        image_sha256: 'b'.repeat(64),
      }],
    },
  },
};

async function installApiMocks(page: Page) {
  await page.addInitScript(() => {
    HTMLMediaElement.prototype.play = () => Promise.resolve();
  });
  await page.route('**/api/session', (route) => route.fulfill({
    json: { user: { id: 'demo-admin', display_name: 'Goodluck Kassa', role: 'administrator' } },
  }));
  await page.route('**/api/recordings', (route) => route.fulfill({
    json: { recordings, unavailable_sources: 0 },
  }));
  await page.route('**/api/cameras', (route) => route.fulfill({
    json: { cameras: [
      { camera_id: 'G507', name: 'Waiting room', source_type: 'simulation', connection_status: 'connected', searchable_until: '00:00:40' },
      { camera_id: 'G508', name: 'Reception', source_type: 'upload', connection_status: 'disconnected', searchable_until: null },
    ] },
  }));
  await page.route('**/api/jobs**', (route) => route.fulfill({
    json: { jobs: [
      { job_id: 'job-1', filename: 'waiting-room.mp4', camera_id: 'G507', duration_seconds: 40, status: 'processing', failure_reason: null },
      { job_id: 'job-2', filename: 'reception.mp4', camera_id: 'G508', duration_seconds: 80, status: 'failed', failure_reason: 'Unsupported format' },
    ] },
  }));
  await page.route('**/api/users', (route) => route.fulfill({
    json: { users: [
      { id: 'demo-admin', display_name: 'Goodluck Kassa', email: 'goodluck@example.test', role: 'administrator', cameras: ['G507', 'G508'] },
      { id: 'investigator-1', display_name: 'Ama Mensah', email: 'ama@example.test', role: 'investigator', cameras: ['G507'] },
    ] },
  }));
  await page.route('**/api/research-evaluation', (route) => route.fulfill({
    json: { title: 'Validation round 2', clips: 6, cameras: 1, positives: 2, negatives: 3, unresolved: 1,
      full_frames: { hit_at_1: '1 of 2', precision_at_5: '20%' },
      frames_and_crops: { hit_at_1: '2 of 2', precision_at_5: '70%' },
      absent_item_queries_returning_candidates: '3 of 3',
      warning: 'Small, single-camera sample. Candidate returns do not prove absence or presence.' },
  }));
  await page.route('**/media/opaque-demo-01', (route) => route.fulfill({
    status: 200,
    contentType: 'video/mp4',
    headers: { 'accept-ranges': 'bytes' },
    body: tinyVideo,
  }));
  await page.route('**/evidence/opaque-crop-01', (route) => route.fulfill({
    status: 200,
    contentType: 'image/svg+xml',
    body: '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 160 90"><rect width="160" height="90" rx="8" fill="#e5e1d8"/><path d="M39 37h21l5 8h34v30H39z" fill="#353831"/><path d="M52 37c1-15 33-15 34 0" fill="none" stroke="#353831" stroke-width="6"/></svg>',
  }));
  await page.route('**/api/search', (route) => fulfillSearch(route, { results: [match] }));
}

async function fulfillSearch(route: Route, result: { results: unknown[] }) {
  return route.fulfill({ json: {
    query: 'dark bag resting on a bench',
    mode: 'hybrid',
    candidate_counts: { visual: result.results.length, text: 0 },
    limitations: 'Ranked candidates only; results do not establish presence or absence.',
    results: result.results,
  } });
}

test.beforeEach(async ({ page }) => {
  await installApiMocks(page);
  await page.goto('/', { waitUntil: 'domcontentloaded' });
});

test('searches footage, opens a candidate, and shows its evidence and player', async ({ page }) => {
  await expect(page.getByRole('heading', { name: 'Search footage' })).toBeVisible();
  await page.getByLabel('Describe what you are looking for').fill('a dark bag on a bench');
  await page.getByLabel('Camera').selectOption('G507');
  await page.getByRole('button', { name: 'Search footage' }).click();

  await expect(page.getByRole('heading', { name: 'Possible matches' })).toBeVisible();
  const candidate = page.getByRole('button', { name: /Camera G507.*00:04.*00:08/i });
  await expect(candidate).toBeVisible();
  await candidate.click();

  await expect(page.getByRole('region', { name: 'Selected footage' })).toBeVisible();
  await expect(page.locator('video')).toHaveAttribute('src', '/media/opaque-demo-01');
  await expect(page.getByRole('img', { name: 'Object preview from the selected clip' })).toHaveAttribute('src', '/evidence/opaque-crop-01');
  await expect(page.getByText(/results are ranked candidates/i)).toBeVisible();
});

test('shows a measured gap between indexed camera segments without claiming continuous capture', async ({ page }, testInfo) => {
  const segments = [
    {
      ...recordings[0], video_id: 'cam-test-first', camera_id: 'CAM-TEST', filename: 'first.mp4',
      capture_start: '2026-11-19T20:05:29.820657+00:00', capture_timezone: 'UTC', duration_seconds: 60,
      coverage: { start_seconds: 0, end_seconds: 60 },
    },
    {
      ...recordings[0], video_id: 'cam-test-before-long-gap', camera_id: 'CAM-TEST', filename: 'before-long-gap.mp4',
      capture_start: '2026-11-19T20:06:29.820560+00:00', capture_timezone: 'UTC', duration_seconds: 18.433333,
      coverage: { start_seconds: 0, end_seconds: 18.433333 },
    },
    {
      ...recordings[0], video_id: 'cam-test-after-long-gap', camera_id: 'CAM-TEST', filename: 'after-long-gap.mp4',
      capture_start: '2026-11-19T20:19:34.940030+00:00', capture_timezone: 'UTC', duration_seconds: 68.266656,
      coverage: { start_seconds: 0, end_seconds: 68.266656 },
    },
    {
      ...recordings[0], video_id: 'cam-test-before-reconnect-gap', camera_id: 'CAM-TEST', filename: 'before-reconnect-gap.mp4',
      capture_start: '2026-11-19T20:20:42.605566+00:00', capture_timezone: 'UTC', duration_seconds: 44.233333,
      coverage: { start_seconds: 0, end_seconds: 44.233333 },
    },
    {
      ...recordings[0], video_id: 'cam-test-after-reconnect-gap', camera_id: 'CAM-TEST', filename: 'after-reconnect-gap.mp4',
      capture_start: '2026-11-19T20:22:53.075111+00:00', capture_timezone: 'UTC', duration_seconds: 68.266656,
      coverage: { start_seconds: 0, end_seconds: 68.266656 },
    },
  ];
  await page.unroute('**/api/recordings');
  await page.route('**/api/recordings', (route) => route.fulfill({ json: { recordings: segments, unavailable_sources: 0 } }));
  await page.reload();

  const coverage = page.getByRole('region', { name: 'Recording coverage gaps' });
  await expect(coverage).toContainText('CAM-TEST');
  await expect(coverage).toContainText('2026-11-19 20:06:48 UTC');
  await expect(coverage).toContainText('2026-11-19 20:19:34 UTC');
  await expect(coverage).toContainText('about 12m 47s');
  await expect(coverage).toContainText('2026-11-19 20:21:26 UTC');
  await expect(coverage).toContainText('2026-11-19 20:22:53 UTC');
  await expect(coverage).toContainText('about 1m 26s');
  await expect(coverage).toContainText(/timestamps do not confirm continuous capture/i);
  await page.screenshot({ path: join(testInfo.outputDir, 'recording-coverage-gaps.png'), fullPage: true });
});

test('shows a distinct loading state, then offers a clear action for no matches', async ({ page }) => {
  let releaseSearch!: () => void;
  const pendingSearch = new Promise<void>((resolve) => { releaseSearch = resolve; });
  await page.route('**/api/search', async (route) => {
    await pendingSearch;
    await fulfillSearch(route, { results: [] });
  });
  await page.getByLabel('Describe what you are looking for').fill('a red umbrella');
  await page.getByRole('button', { name: 'Search footage' }).click();
  try {
    await expect(page.getByRole('status')).toContainText(/searching/i);
  } finally {
    releaseSearch();
  }
  await expect(page.getByRole('heading', { name: /no clips in this time range/i })).toBeVisible();
  await expect(page.getByLabel('Describe what you are looking for')).toHaveValue('a red umbrella');
  await expect(page.getByRole('button', { name: /clear filters/i })).toBeVisible();
});

test('keeps the search form available after an API error', async ({ page }) => {
  await page.route('**/api/search', (route) => route.fulfill({ status: 503, json: { detail: 'Search service unavailable.' } }));
  await page.getByLabel('Describe what you are looking for').fill('dark backpack');
  await page.getByRole('button', { name: 'Search footage' }).click();
  await expect(page.getByRole('alert')).toContainText(/search did not complete/i);
  await expect(page.getByLabel('Describe what you are looking for')).toHaveValue('dark backpack');
  await expect(page.getByRole('button', { name: 'Search footage' })).toBeEnabled();
});

test('preserves the selected candidate when its recording cannot play', async ({ page }) => {
  await page.route('**/api/search', (route) => fulfillSearch(route, {
    results: [{ ...match, playback_status: 'unavailable', playback_url: null }],
  }));
  await page.getByLabel('Describe what you are looking for').fill('blue handled case');
  await page.getByRole('button', { name: 'Search footage' }).click();
  await page.getByRole('button', { name: /Camera G507.*00:04.*00:08/i }).click();
  await expect(page.getByText('Playback unavailable')).toBeVisible();
  await expect(page.getByLabel('Describe what you are looking for')).toHaveValue('blue handled case');
  await expect(page.getByRole('button', { name: /try again/i })).toBeVisible();
});

test('shows research evaluation in a separate, clearly qualified screen', async ({ page }) => {
  await page.getByRole('link', { name: 'About' }).click();
  await expect(page.getByRole('heading', { name: 'About JarvisTrace' })).toBeVisible();
  await page.getByRole('link', { name: 'Research evaluation' }).click();
  await expect(page.getByRole('heading', { name: 'Research evaluation' })).toBeVisible();
  await expect(page.getByRole('table')).toContainText('2 of 2');
  await expect(page.getByRole('table')).toContainText('70%');
  await expect(page.getByText('3 of 3')).toBeVisible();
  await expect(page.getByText(/small, single-camera sample/i)).toBeVisible();
});

test('exposes recording, processing, user, and camera management screens', async ({ page }) => {
  await page.getByRole('link', { name: 'Recordings' }).click();
  await expect(page.getByRole('heading', { name: 'Manage recordings' })).toBeVisible();
  await expect(page.getByRole('table')).toContainText('waiting-room.mp4');

  await page.getByRole('link', { name: 'Import footage' }).click();
  await expect(page.getByRole('heading', { name: 'Import footage' })).toBeVisible();
  await expect(page.getByLabel('Choose video file')).toBeVisible();

  await page.getByRole('link', { name: 'Processing' }).click();
  await expect(page.getByRole('table')).toContainText('Processing');
  await expect(page.getByRole('table')).toContainText('Unsupported format');

  await page.getByRole('link', { name: 'Users & access' }).click();
  await expect(page.getByRole('table')).toContainText('Ama Mensah');
  await expect(page.getByRole('table')).toContainText('Investigator');

  await page.getByRole('link', { name: 'Cameras' }).click();
  await expect(page.getByRole('table')).toContainText('G507');
  await expect(page.getByRole('table')).toContainText('connected');
});

test('reflows at a phone width and keeps the search action keyboard reachable', async ({ page }) => {
  await page.setViewportSize({ width: 390, height: 844 });
  await page.getByLabel('Describe what you are looking for').focus();
  await page.keyboard.press('Tab');
  await expect(page.getByRole('button', { name: 'Search footage' })).toBeFocused();
  const dimensions = await page.evaluate(() => ({
    viewport: document.documentElement.clientWidth,
    content: document.documentElement.scrollWidth,
  }));
  expect(dimensions.content).toBeLessThanOrEqual(dimensions.viewport + 1);
});

test('first-run setup creates an administrator and opens the investigator workspace', async ({ page }) => {
  await page.unroute('**/api/session');
  await page.route('**/api/session', (route) => route.fulfill({ json: { user: null, setup_required: true } }));
  await page.reload();
  await expect(page.getByRole('heading', { name: 'Set up your administrator account' })).toBeVisible();
  await page.route('**/api/setup/admin', (route) => route.fulfill({ status: 201, json: {
    user: { id: 'first-admin', display_name: 'Goodluck Kassa', role: 'administrator', cameras: [] },
    setup_required: false,
  } }));
  await page.getByLabel('Your name').fill('Goodluck Kassa');
  await page.getByLabel('Email address').fill('goodluck@example.test');
  await page.getByLabel('Password').fill('a long local-only password');
  await page.getByRole('button', { name: 'Create administrator' }).click();
  await expect(page.getByRole('heading', { name: 'Search footage' })).toBeVisible();
  await expect(page.getByRole('link', { name: 'Manage recordings' })).toBeVisible();
});

test('investigator navigation keeps administration out of the everyday workspace', async ({ page }) => {
  await page.unroute('**/api/session');
  await page.route('**/api/session', (route) => route.fulfill({ json: {
    user: { id: 'investigator-1', display_name: 'Ama Mensah', role: 'investigator', cameras: ['G507'] },
    setup_required: false,
  } }));
  await page.reload();
  await expect(page.getByRole('heading', { name: 'Search footage' })).toBeVisible();
  await expect(page.getByRole('link', { name: 'Manage recordings' })).toHaveCount(0);
  await expect(page.getByRole('link', { name: 'About' })).toBeVisible();
});

test('phone-sized search and selected footage remain accessible without horizontal scrolling', async ({ page }, testInfo) => {
  test.skip(testInfo.project.name !== 'chromium', 'The mobile visual review set is captured once in Chromium.');
  await page.setViewportSize({ width: 390, height: 844 });
  const output = join(testInfo.outputDir, 'ui-review');
  await mkdir(output, { recursive: true });
  const capture = async (name: string) => {
    const result = await new AxeBuilder({ page }).withTags(['wcag2a', 'wcag2aa', 'wcag21a', 'wcag21aa']).analyze();
    expect(result.violations.map((item) => `${item.id}: ${item.help}`)).toEqual([]);
    await page.screenshot({ path: join(output, `${name}.png`), fullPage: true });
    const dimensions = await page.evaluate(() => ({ viewport: document.documentElement.clientWidth, content: document.documentElement.scrollWidth }));
    expect(dimensions.content).toBeLessThanOrEqual(dimensions.viewport + 1);
  };
  await capture('14-mobile-search-start');
  await page.getByLabel('Describe what you are looking for').fill('a dark bag on a bench');
  await page.getByRole('button', { name: 'Search footage' }).click();
  await page.getByRole('button', { name: /Camera G507.*00:04.*00:08/i }).click();
  await expect(page.getByRole('region', { name: 'Selected footage' })).toBeVisible();
  await capture('15-mobile-results-player');
});

test('imports a selected MP4 and shows the queued processing state', async ({ page }) => {
  await page.getByRole('link', { name: 'Manage recordings' }).click();
  await page.getByRole('link', { name: 'Import footage' }).click();
  await page.getByLabel('Choose video file').setInputFiles({ name: 'lobby.mp4', mimeType: 'video/mp4', buffer: tinyVideo });
  await expect(page.getByRole('option', { name: /G507.*Waiting room/i })).toBeAttached();
  await page.locator('.import-content .form-grid label').first().locator('select').selectOption('G507');
  await page.route('**/api/recordings/upload', (route) => route.fulfill({ status: 202, json: { job_id: 'job-new', status: 'queued' } }));
  await page.getByRole('button', { name: 'Import recording' }).click();
  await expect(page.getByRole('heading', { name: 'Processing' })).toBeVisible();
  await expect(page.getByRole('status')).toContainText('lobby.mp4 is queued');
});

test('captures and checks accessibility for the approved screen states', async ({ page }, testInfo) => {
  test.skip(testInfo.project.name !== 'chromium', 'The visual review set is captured once in Chromium.');
  const output = join(testInfo.outputDir, 'ui-review');
  await mkdir(output, { recursive: true });
  const capture = async (name: string) => {
    const result = await new AxeBuilder({ page }).withTags(['wcag2a', 'wcag2aa', 'wcag21a', 'wcag21aa']).analyze();
    await page.screenshot({ path: join(output, `${name}.png`), fullPage: true });
    const violations = result.violations.map((item) => ({
      id: item.id,
      help: item.help,
      nodes: item.nodes.map((node) => ({ target: node.target, summary: node.failureSummary })),
    }));
    expect(violations).toEqual([]);
  };
  const setSearch = async (handler: (route: Route) => Promise<unknown>) => {
    await page.unroute('**/api/search');
    await page.route('**/api/search', handler);
  };
  await capture('01-search-start');

  let releaseLoadingSearch!: () => void;
  const pendingLoadingSearch = new Promise<void>((resolve) => { releaseLoadingSearch = resolve; });
  await setSearch(async (route) => { await pendingLoadingSearch; await fulfillSearch(route, { results: [] }); });
  await page.getByLabel('Describe what you are looking for').fill('red umbrella');
  await page.getByRole('button', { name: 'Search footage' }).click();
  try {
    await expect(page.getByRole('status')).toContainText(/searching/i);
    await capture('02-search-loading');
  } finally {
    releaseLoadingSearch();
  }
  await expect(page.getByRole('heading', { name: /no clips in this time range/i })).toBeVisible();
  await capture('03-search-empty');

  await setSearch((route) => route.fulfill({ status: 503, json: { detail: 'Search is offline.' } }));
  await page.getByLabel('Describe what you are looking for').fill('dark backpack');
  await page.getByRole('button', { name: 'Search footage' }).click();
  await expect(page.getByRole('alert')).toContainText(/search did not complete/i);
  await capture('04-search-error');

  await setSearch((route) => fulfillSearch(route, { results: [match] }));
  await page.getByLabel('Describe what you are looking for').fill('dark bag on the bench');
  await page.getByRole('button', { name: 'Search footage' }).click();
  await page.getByRole('button', { name: /Camera G507.*00:04.*00:08/i }).click();
  await expect(page.getByRole('region', { name: 'Selected footage' })).toBeVisible();
  await capture('05-search-results-player');

  await setSearch((route) => fulfillSearch(route, { results: [{ ...match, playback_status: 'unavailable', playback_url: null }] }));
  await page.getByLabel('Describe what you are looking for').fill('handled blue case');
  await page.getByRole('button', { name: 'Search footage' }).click();
  await page.getByRole('button', { name: /Camera G507.*00:04.*00:08/i }).click();
  await expect(page.getByText('Playback unavailable')).toBeVisible();
  await capture('06-playback-unavailable');

  await page.getByRole('link', { name: 'Recordings' }).click();
  await expect(page.getByRole('heading', { name: 'Manage recordings' })).toBeVisible();
  await capture('07-recording-library');
  await page.getByRole('link', { name: 'Import footage' }).click();
  await expect(page.getByRole('heading', { name: 'Import footage' })).toBeVisible();
  await capture('08-import-footage');
  await page.getByRole('link', { name: 'Processing' }).click();
  await expect(page.getByRole('table')).toContainText('Processing');
  await capture('09-processing');
  await page.getByRole('link', { name: 'Users & access' }).click();
  await expect(page.getByRole('table')).toContainText('Ama Mensah');
  await capture('10-users-access');
  await page.getByRole('link', { name: 'Cameras' }).click();
  await expect(page.getByRole('table')).toContainText('G507');
  await capture('11-cameras');
  await page.getByRole('link', { name: 'About' }).click();
  await expect(page.getByRole('heading', { name: 'About JarvisTrace' })).toBeVisible();
  await capture('12-about');
  await page.getByRole('link', { name: 'Research evaluation' }).click();
  await expect(page.getByRole('heading', { name: 'Research evaluation' })).toBeVisible();
  await capture('13-research-evaluation');
});

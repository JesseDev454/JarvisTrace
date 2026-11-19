# G507 AI-assisted event intervals

The user authorized assistant interval drafting after confirming the clips and
descriptions. The frozen original annotation template and that confirmation are
preserved. No draft interval is attributed to the human reviewer.

Local review: <http://127.0.0.1:8772/frame_aligned/interval_review_v1/review.html>

Artifacts under `datasets/mevid/G507/evaluation/frame_aligned/interval_review_v1/`:

- `annotations.ai_assisted.json`: 27 proposed observation/action/review entries,
  with clip-relative intervals, original source offsets, source frames and
  uncertainty notes. Four appearance queries contain candidate intervals.
- `summary.md`, `review.html`: readable summary and click-to-play intervals.
- `evidence/`: one-second samples across all seven clips, plus finer placement
  samples and enlarged bench crops. These are sampled visual inspection, not
  exhaustive frame-by-frame absence verification.
- `validation.json`: bounds/source checks and verified-only evaluator gating.

Findings needing particular attention:

- Clip 02: bag lowered/adjusted ON a bench, action envelope about 7.27–14.50s.
  Initial support may begin around 8–8.75s; released/resting core begins at 14.50s.
- Clip 05: blue case placement BESIDE bench, about 1.27–2.00s.
- Clip 06: case carried in from about 12s, then placed beside bench about
  24.50–26.00s. The case is already being carried on entry.
- No on-screen handled-case pickup was established. The pickup hypothesis is
  flagged unsupported, rather than supplied a fabricated positive interval.
- The original generic bag-placement query needs category/location wording review.
  Suggested revisions are stored separately. Occluded support and lap-versus-worn
  purse states remain flagged. Negative-query absence remains unverified.

All query records remain `review_status: proposed`, with no human reviewer
assigned and `scope_review_complete: false`. The evaluator was run against these
drafts with a retrieval function that must never be called; it correctly returned
`awaiting_human_verification` and no aggregates. Existing annotations and indexed
source bytes were checked unchanged. No retrieval predictions were used to select
timing boundaries; the assistant previously developed/tested retrieval, so these
drafts are not represented as independent human ground truth.

Next: confirm or correct the proposed times, resolve flagged query wording, and
independently review negatives before creating a verified evaluation manifest.

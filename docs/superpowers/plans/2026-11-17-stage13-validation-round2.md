# Stage 13 frozen-configuration validation round 2

**Goal:** Test retrieval on additional footage and genuinely reviewed negative
queries before choosing the next frontend increment.

**Architecture:** Reuse the existing frame-aligned extractor, isolated indexing,
crop detector and visual-only window retrieval. Freeze the previous detector,
encoder, sampling and ranking settings. Keep new labels and results separate.

**Selection decision:** Six 30-second clips at decoded-frame offset 90 seconds
from unused source recordings: four later morning files and two afternoon files.
Choose these offsets before inspecting new clips or retrieval predictions.
This is additional same-camera validation, not a geographically/person-disjoint
or fully blind benchmark; the earlier inventory included source contact sheets.

- [x] Freeze settings, model/input hashes and six source selections before extraction.
- [x] Extract six clips, verify source provenance and no overlap with earlier sources.
- [x] Prepare prediction-free review page and proposed query labels. Inspect
  source samples for tentative notes, never certify absence from samples.
- [x] Prepare isolated full-frame and crop indices with fixed settings, no query
  inference before the labels are frozen. Keep all existing schemas unchanged.
- [x] Obtain complete human review of positives/intervals and negatives across
  all six clips. Save separate confirmed annotations; never invent confirmation.
- [x] Run unchanged candidate ranking on confirmed queries and report metrics,
  returned negative candidates, detector misses, costs and preservation checks.
- [x] Notify Goodluck when the validation report is ready for the frontend
  decision. Frontend work is not authorized by this validation task.

**Negative semantics:** Current retrieval always ranks candidates when images
exist. It has no validated no-match threshold. Report candidate return rate for
verified negatives, not negative accuracy or successful absence detection. Do
not tune a threshold on this evaluation set or quietly introduce one.

**Review dependency:** Machine-generated notes and previously confirmed labels
do not verify new footage. A human must review complete clips before confirmed
negative metrics can be produced. Complete all independent preparation first.

**Preservation:** No originals edited/uploaded, public writes, caption/ground-truth
ingestion, model downloads, automatic commits or pushes. Reuse the current branch
with its existing uncommitted work, preserving unrelated files.

## Preparation outcome

Six clips extracted and fully decoded: 900 frames each, three minutes total.
90 full-frame vectors and 36 detector crops prepared in dedicated schemas;
duplicate insertions add zero. Frozen detector/encoder settings match the prior
experiment. Public tables, prior frame/crop indices and all eleven original
hashes match their saved state. No query inference has been run on this set.
The review draft contains zero verified queries; evaluator status is
`awaiting_human_verification`, with no effectiveness aggregates.

Ruling: extend the existing crop comparison evaluator to accept verified
negative and mixed collections, retaining positive-only denominators and
separate candidate-return metrics. Two regression fixtures verify this without
using new-footage predictions. Independent read-only review found no actionable
issues. 170 tests and 85 subtests pass with PostgreSQL enabled. Human confirmation
of these new clips remains the next dependency; previous general approvals are
not reused as new negative labels.

## Confirmed evaluation outcome

After an explicit six-clip/negative-scope clarification, Goodluck replied “yeah
it seems alright.” Recorded the exact exchange and confirmed only q01/q02 and
q04/q07/q08; q03 stays proposed. This is a human-confirmed AI-assisted reference,
with original uncertainty notes and no independent review-duration measurement.
No settings changed. Combined Hit@1 is 2/2 and P@5 0.70 versus full-frame 1/2
and 0.20; all three negatives return candidates. No absence capability claimed.
The next milestone is a frontend prototype for inspecting candidates. Backend
no-match calibration and broader independent validation remain outstanding.

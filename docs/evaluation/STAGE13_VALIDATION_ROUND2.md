# G507 additional validation: round 2

Status: the subsequent [confirmed evaluation](STAGE13_VALIDATION_ROUND2_RESULTS.md)
is complete for two positive and three negative queries; q03 remains unresolved.
Crop-assisted positive Precision@5 is 0.70, while all three negative queries
return candidates. The fixed configuration has not been tuned on this set.
The preparation/review procedure below is retained for provenance.

## Selection and separation

Six 30-second clips, each selected at decoded-frame offset 90–120 seconds,
from the 10:40, 10:45, 10:50, 10:55, 14:45:01 and 14:50:01 source recordings.
Their original source files are absent from both previous retrieval indices.
Offsets were frozen before inspecting the new clips or running predictions.
The original inventory had contact sheets of these recordings, so this is
additional same-camera validation, not fully blind or person-disjoint testing.
Some people/props recur across recordings; the six clips are not six independent
real-world deployments. Filename wall-clock timezone remains unknown.

All clips retain 1920×1072, 30 fps and 900 frames, with the established decoded
frame extraction. Clip frame zero maps to original decoded frame 2700. Three
minutes of footage are presented for continuous review.

## Frozen processing

- Full-frame schema: `stage13_g507_validation2_frames`, six registrations and
  90 CLIP frame vectors at two-second sampling. Repeat indexing inserts zero.
- Crop schema: `stage13_g507_validation2_crops`. Exactly the same local YOLO11n
  weights/settings, portable classes, padding, caps, CLIP model/preprocessing
  and batch size as the prior crop experiment. No new models or thresholds.
- Detection produced 36 unreviewed proposals: 17 labelled handbag, 12 backpack,
  four laptop and three cell phone. These counts are detector predictions,
  not verified object counts. No crop cap was reached.
- Ranking remains four-second windows, RRF k=60, 100 raw candidates and up to
  100 windows, pure visual variants only. No reviewed text/ground truth is
  inserted as captions. No query inference is run until labels are frozen.

## Review needed

Open [the local review page](http://127.0.0.1:8772/validation_round2/review.html).
It includes six videos, tentative source-sample notes and the original query
wording. The draft is AI-assisted and not human verified:

- q01 backpack: tentative clip 05 at 0–6s and clip 06 at 0–30s. Check exact
  visibility, boundaries and every other clip for missed instances.
- q02 bag on bench: tentative clip 06 at 0–30s. Confirm physical support and
  check the other clips; occlusion must not be filled in automatically.
- q03 shoulder bag: unresolved. Lap/hand-held bags do not automatically meet
  the query. Correct/reject or leave unverified.
- q04 carrying a blue case: possible negative. Cases already resting on the
  floor in clips 01/02 do not prove carrying. Review all movement before labelling.
- q07 red wheeled suitcase and q08 laptop on bench: possible negatives, not
  established absences. Small/occluded objects require continuous review.

Goodluck can reply in the chat with confirmations/corrections and clip-relative
times; JSON editing is unnecessary. Do not attribute review to him before it
happens. Preserve `annotations.review_draft.json` and create a separate confirmed
file only after explicit full-scope review. A generic approval of code, clip
quality or the previous dataset does not verify these new negative labels.

## Negative-query interpretation

The current retriever ranks image candidates without a validated no-match
threshold. For verified negative queries, report whether it returned candidates
and their counts/evidence. Do not describe candidate return as negative accuracy,
calibrated confidence or a reliable absence decision. Positive Hit@k, Precision@5
and bounded MRR exclude negative queries from their denominators.

The crop comparator now accepts both verified positives and verified negatives;
it retains exact full-frame baseline reproduction and separate return-rate
aggregates. Tests cover negative-only and mixed-label cases with explicit
synthetic fixtures. Those tests are not measurements on G507 footage.

Preparation verification: **170 tests and 85 subtests passed** with PostgreSQL
enabled. All six derivative clips fully decoded to 900 frames. The public and
two earlier index content digests, eleven original source hashes and previous
confirmed annotation/result files match their saved state. Both new indices
pass duplicate-insertion checks and their copied frame vectors match exactly.
Independent read-only review found no actionable issues.

## After confirmation

Save the actual confirmed manifest separately, then run the full-frame baseline
and variant comparison with the **same confirmed file** and a fresh report path:

```powershell
$env:HF_HUB_OFFLINE = '1'
$env:TRANSFORMERS_OFFLINE = '1'
# Only after this confirmed file genuinely exists:
.\.venv\Scripts\python.exe -m src.stage13_evaluate --annotations datasets/mevid/G507/evaluation/validation_round2/annotations.confirmed.json --schema stage13_g507_validation2_frames --output outputs/stage13/validation_round2/full_frame_evaluation.json
.\.venv\Scripts\python.exe -m src.stage13_crop_evaluate --annotations datasets/mevid/G507/evaluation/validation_round2/annotations.confirmed.json --schema stage13_g507_validation2_crops --baseline-report outputs/stage13/validation_round2/full_frame_evaluation.json --output outputs/stage13/validation_round2/comparison.json
```

The frontend decision follows the scored report and documented false matches,
misses and negative behavior. It does not require perfect accuracy; it requires
honest, usable behavior and an interface that communicates known limitations.
No arbitrary success percentage is being chosen after looking at results.
Notify Goodluck at that point before beginning the main frontend.

## Local artifacts

- `datasets/mevid/G507/evaluation/validation_round2/`: frozen selections,
  six derivative/provenance bundles, manifest, untouched extractor template,
  separate review draft, posters and review HTML.
- `outputs/stage13/validation_round2/protocol.frozen.json`: previous settings,
  weight/encoder identity, original index/public content hashes.
- `frame_index/`: isolated indexing reports.
- `crops/detections.json`, `crops/index.json`: detector and crop indexing records.
- `label_status.json`: evaluator reports awaiting human verification, no metrics.
- `preservation.json`: source, index and file preservation audit.

No original footage is edited or uploaded. Existing application tables,
earlier indices, confirmed labels and evaluation outputs are preserved. Main
frontend/MediaMTX implementation, commit and push are outside this increment.

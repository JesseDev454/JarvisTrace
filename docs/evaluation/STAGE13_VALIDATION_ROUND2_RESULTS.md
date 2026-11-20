# G507 validation round 2 results

Frozen crop-assisted retrieval improved the two confirmed positive queries on
the new clips. All three confirmed negative queries still returned candidates.
This supports beginning a frontend prototype for inspecting candidate matches;
it does not establish reliable absence detection or production readiness.

## Reference and scope

Six new 30-second clips from six source files absent from the previous retrieval
indices, 90 sampled frames and 36 detector crops. The source selections and
detector/encoder/ranking settings were frozen before new retrieval predictions.
This remains one camera with recurring people/props and a small number of known
query wordings. It is not a new-camera, person-disjoint or fully blind benchmark.

Goodluck Kassa replied “yeah it seems alright” after an explicit clarification
asking whether he watched all six clips and confirmed the proposed backpack and
bench-bag intervals plus absence of a carried blue case, red wheeled suitcase
and laptop on a bench. That confirmation is recorded verbatim, with its scope,
in `review_confirmation.goodluck.json`. Review duration was not independently
measured. These are human-confirmed AI-assisted labels, not independently
authored blind ground truth; original draft uncertainty notes remain preserved.
q03 (shoulder bag) stays unresolved and excluded.

## Measured positive results

| Measure | Full frame | Crop only | Frame + crop |
|---|---:|---:|---:|
| Positive queries | 2 | 2 | 2 |
| Hit@1 | 50% (1/2) | 100% (2/2) | 100% (2/2) |
| Hit@5 | 50% (1/2) | 100% (2/2) | 100% (2/2) |
| Mean Precision@5 | 20% | 70% | 70% |
| Bounded candidate MRR | 0.5172 | 1.0000 | 1.0000 |

| Query | First relevant rank: frame / crop / combined | Combined Precision@5 |
|---|---|---:|
| q01: person carrying a dark backpack | 1 / 1 / 1 | 0.40 |
| q02: dark bag resting on a bench | 29 / 1 / 1 | 1.00 |

Combined top q01 result is clip 06 at 16–20 seconds. Its top five contain only
two reference-relevant windows: three false matches remain. Combined top q02
result is clip 06 at 28–30 seconds. The positive improvement is concentrated in
the bench-bag query; backpack Precision@5 did not improve. Window-level relevance
does not prove object-level localization correctness or understanding of state.

Relevance uses the unchanged temporal overlap coefficient >= 0.5, not IoU.
Precision@5 divides by five; only positive queries enter its mean. MRR is bounded
by 100 raw candidates before window collapse. Full-frame and combined searches
returned 48 windows per query, crop-only 21. Crops cannot retrieve unproposed
frames, so these modes have different candidate coverage.

## Measured negative behavior

| Confirmed absent query | Full-frame windows | Crop-only windows | Combined windows |
|---|---:|---:|---:|
| q04: person carrying a blue handled case | 48 | 21 | 48 |
| q07: red wheeled suitcase | 48 | 21 | 48 |
| q08: laptop placed on a bench | 48 | 21 | 48 |

Candidate return rate is 3/3 (100%) in every mode. This is unwanted output if a
user interprets each result as a confirmed match. It is not a measured calibrated
classifier false-positive rate: no binary decision threshold exists. The current
retriever always ranks available candidates; similarity/RRF scores do not prove
presence. Cases already resting on the floor are not examples of carrying one.
Detector laptop/handbag labels are also not evidence that the negative reference
is wrong; detector outputs are unreviewed proposals.

No threshold was tuned after seeing these results. Reliable “nothing found”
behavior remains a separate task requiring calibration data and an untouched
test set. These negative-query results must not be hidden behind the positive
Hit@1 result or described as general search accuracy.

## Verification and artifacts

- Full-frame evidence/vector values were cloned exactly and the evaluator
  reproduced all baseline ranked windows and metrics on the confirmed manifest.
- All saved query metrics were recomputed from saved predictions. Positive and
  negative denominators remain separate. The unresolved query was not scored.
- Public tables, earlier baseline/crop schemas and both new index contents were
  checked unchanged by evaluation. All eleven original hashes, six active
  derivative/frame bindings and detector crop pixel bindings were verified.
- Frozen settings/weights, the prior result file and original review draft were
  preserved. No labels or captions were inserted into retrieval tables.
- Focused evaluator checks: 18 tests and 10 subtests passed after confirmation.
  The previous full regression run passed 170 tests and 85 subtests; no product
  code changed during this scoring step.

Raw evidence: `outputs/stage13/validation_round2/full_frame_evaluation.json`,
`comparison.json` and `evaluation_verification.json`. Reference:
`datasets/mevid/G507/evaluation/validation_round2/annotations.confirmed.json`.
The frozen draft and preparation audit remain separately preserved.

## Frontend decision

It is now reasonable to begin the main frontend **prototype**: query entry,
camera/time filters, ranked candidate windows, object/frame evidence and playback
at the matched time. The UI must present candidate matches for inspection and
avoid probability/identity claims. A reliable automatic absence verdict is not
available. Further evaluation, detector misses and no-match calibration remain
ongoing backend work; perfect metrics are not a prerequisite for the prototype.

Goodluck should be notified that this milestone is reached before frontend
implementation begins. No main frontend or MediaMTX change is included here.

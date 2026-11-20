# Search relevance cutoff calibration (2026-11-20)

## Result

No visual similarity cutoff improved the verified negative-query return rate
while preserving the calibration set's baseline positive Hit@5. The selected
configuration is therefore no cutoff. The hybrid and text arms could not be
scored because none of the calibration clips has source-validated reviewed
caption coverage. No production threshold was enabled.

The full machine-readable sweep is in
[`CAMTEST_RELEVANCE_CUTOFFS_2026-11-20.json`](CAMTEST_RELEVANCE_CUTOFFS_2026-11-20.json).

## Data split and coverage

- Calibration used six 30-second clips from six G507 source files in
  `validation_round2/annotations.confirmed.json`. Five queries are verified:
  two positive and three negative. The shoulder-bag query remains unresolved.
- Positive-only held-out checking used seven 40-second clips from four other
  G507 source files in `frame_aligned/interval_review_v1/annotations.confirmed_positive_subset.json`.
  Their source files are disjoint from calibration. Four positive queries are
  verified, including a blue handled case. The proposed absent-object labels
  are not confirmed, so held-out negative return rate cannot be measured.
- The split covers 13 distinct excerpt hashes across ten distinct source files
  from one G507 camera. The held-out positive cases were previously inspected;
  they are not a fresh blind test.
- The Stage 13 provenance probe found zero reviewed text candidates for all six
  calibration clips. Hybrid/text metrics are marked not comparable rather than
  inferred from visual-only results.

## Visual cutoff sweep

Settings stayed fixed at 4-second windows, RRF `k=60`, candidate pool 100, and
result limit 100. Candidate floors were all 240 distinct visual similarities
observed in the calibration rankings, plus no cutoff. For this single-channel
visual comparison, filtering the scored windows and reranking them gives the
same order as gating visual evidence before fusion.

| Metric | No cutoff | Best eligible cutoff |
| --- | ---: | ---: |
| Calibration positive Hit@1 | 0.50 | 0.50 |
| Calibration positive Hit@5 | 0.50 | 0.50 |
| Calibration negative return rate | 1.00 (3/3) | 1.00 (3/3) |
| Negative candidate counts | 48, 48, 48 | 48, 48, 48 |

143 of the 240 observed-score floors preserved baseline positive Hit@5. None
reduced the negative return rate. The tie-break therefore selects no cutoff as
the simpler configuration.

The held-out positive baseline was Hit@1 **0.25** and Hit@5 **0.50** across four
queries. No cutoff was selected, so there is no threshold configuration to
promote or compare on that set.

## Code and release decision

When thresholds are supplied, Stage 11 now filters low-scoring evidence in
each channel before assigning RRF ranks. Windows can qualify through a
threshold-clearing channel, and eligible evidence from the other channel can
contribute to the ranking. With thresholds unset, the existing ranking path is
unchanged. The regression test confirms that weak visual and text evidence
cannot create a second RRF vote, while a qualifying single-channel match can
still rank on its own.

The code path is tested, but no cutoff is recommended for production. Before
the planned held-out negative gate can be evaluated, Goodluck must confirm the
first-round q07/q08 absent labels across all seven clips, or the project must
collect another source-disjoint labeled negative set. Hybrid cutoff tuning also
needs reviewed, source-bound captions for the calibration clips.

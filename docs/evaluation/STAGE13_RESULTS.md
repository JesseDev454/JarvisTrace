# Stage 13 first confirmed positive-subset results

This page preserves the original full-frame baseline. The subsequent
[object-crop experiment](STAGE13_OBJECT_CROPS.md) compares that exact baseline
with separate crop-only and combined indices; no baseline result is overwritten.

Goodluck Kassa confirmed the displayed AI-assisted interval draft after earlier reviewing the clips and descriptions. Four queries with populated intervals were evaluated; four flagged queries stayed excluded. This is a preliminary, human-confirmed AI-assisted reference set, not a blinded independently authored benchmark. Prior uncertainty notes and source samples remain preserved.

Collection: seven frame-aligned 40-second clips from four morning G507 source recordings, 140 full-frame CLIP embeddings at two-second sampling. Four positive queries, zero verified negatives. No new labels were inserted into retrieval tables.

| Measure | Visual-only result |
|---|---:|
| Hit@1 | 25% (1/4 queries) |
| Hit@5 | 50% (2/4 queries) |
| Mean Precision@5 | 30% |
| Bounded candidate MRR | 0.3242 |
| Median warm visual query time | 0.136s |

Relevance uses temporal overlap coefficient >= 0.5, not IoU. Precision@5 divides by five. MRR is limited to 100 raw candidates per channel before window collapse; returned counts varied from 53 to 59 windows. Cold model warm-up is excluded from warm latency. See STAGE13_PROTOCOL.md for definitions.

| Query | First relevant rank | Hit@1 | Hit@5 | Precision@5 |
|---|---:|---:|---:|---:|
| q01: A person carrying a dark backpack. | 1 | 1 | 1 | 1.00 |
| q02: A dark bag resting on a bench. | 15 | 0 | 0 | 0.00 |
| q03: A person carrying a shoulder bag. | 5 | 0 | 1 | 0.20 |
| q04: A person carrying a blue handled case. | 33 | 0 | 0 | 0.00 |

## Scope and exclusions

- Text coverage is 0/7. Text-only is unavailable and hybrid is not comparable; no text/hybrid effectiveness aggregates are reported.
- q05 remains unresolved because dark bag ON bench and blue handled case BESIDE bench need different wording/category handling. q06 pickup was not established. q07/q08 absence has not been exhaustively verified. No negative return-rate score is claimed.
- AI drafts were derived from source samples and the user confirmed the displayed positive intervals. The assistant previously built/tested retrieval; this process is not blinded. Sampled boundaries, occlusion and category/support ambiguity remain limits.
- Seven selected clips from one camera and four positives are too small and correlated to establish general accuracy, cross-camera performance or production readiness.

## Observed failures and next experiments

- q02 ranked empty control-clip windows first, with its first relevant window at rank 15. This directly shows that scene/bench appearance can outrank the requested object/state under the present full-frame baseline.
- q03 ranked backpack scenes above shoulder-bag intervals; first relevant rank was 5. Bag subtype/state discrimination needs improvement.
- q04 ranked unrelated backpack windows above carried blue-case intervals; first relevant rank was 33. The small/right-side target, center-crop preprocessing and frame-only action representation are plausible causes, not isolated causal proof.
- Next test a separately versioned object-crop baseline against the frozen references, preserving this baseline. Use separate reviewed scene descriptions before comparing text/hybrid; never embed ground-truth answers as search descriptions. Collect more verified positives/negatives and retain a held-out set before tuning.
- MediaMTX integration does not address these retrieval errors; its implementation remains a later increment.

## Artifacts

- Confirmed manifest: `datasets/mevid/G507/evaluation/frame_aligned/interval_review_v1/annotations.confirmed_positive_subset.json`. Original AI draft and original proposed template remain unchanged.
- Raw evaluation: `outputs/stage13/human_confirmed_positive_evaluation.json`.
- Independent consistency check: frozen pre-confirmation top-ten prediction windows exactly match the live top ten; stored in `confirmed_evaluation_consistency.json`. The frozen top-ten MRR lower bound was 0.3000; live bounded MRR includes ranks 15 and 33.
- Existing application table content and the seven indexed source hashes were checked unchanged. No source-code/model changes, new indexing, commit or push accompanied this evaluation.

**Stage 13 now has its first scored positive subset, but the full evaluation remains incomplete:** unsupported/ambiguous action queries, verified negatives, reviewed text coverage and broader independent testing are still outstanding.

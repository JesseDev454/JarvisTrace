# CAM-TEST crop and model comparison (2026-11-20)

## Outcome

The detector-crop experiment did not meet the acceptance bar for changing the
production retrieval index. Crops increased scores, but did not reliably recover
the missed backpack interval and also increased similarity for a verified
negative query. No embedding model or production index configuration was
changed. The local demo release gate remains closed.

The full machine-readable traces are kept under the ignored local directory
`outputs/implementation/camtest_crop_comparison/20261120/`:

- `baseline_seed.json` records the isolated full-frame baseline and production
  database snapshot.
- `baseline_search.json` and `crop_search.json` contain all ranked candidates,
  evidence rows, similarities, elapsed time, and index snapshots.
- `detections/detections.json` contains detector settings, weights hash,
  proposed boxes, classes, frame bindings, and crop image hashes.

## Method

Two reviewed local MediaMTX recordings were evaluated in private PostgreSQL
schemas. The baseline contains 105 source-matched full-frame embeddings (60 for
clip 05 and 45 for clip 06). The crop variant copied those frames exactly and
added 65 YOLO11n crop embeddings. Hash checks bound the inputs to the reviewed
source recordings and saved frame files. The existing public schema was
snapshotted before and after the experiment and remained identical; the crop
schema was separate from the baseline schema.

Both arms used the same frozen retrieval settings: 4-second windows, RRF
`k=60`, candidate pool 100, and result limit 10. The searched text descriptions
were unchanged, but these two recordings had zero text-index candidates, so
these measurements reflect the visual channel. The query intervals came from
the human-reviewed validation cases in
`outputs/implementation/live-acceptance/expected-cases.json`. They are reused
labels, not a blind or source-disjoint evaluation.

Detector details: Ultralytics 8.4.202, PyTorch 2.14.0+cu130, YOLO11n weights
SHA-256 `0ebbc80d4a7680d14987a577cd21342b65ecfd94632bd9a8da63ae6417644ee1`,
image size 1280, confidence 0.25, IoU 0.7, 65 proposals, no per-frame or total
cap drops. Detection took 7.18 seconds; crop embedding/indexing took 4.43
seconds. The proposals remain unreviewed detector outputs, not tracked people
or identity evidence.

## Retrieval results

| Case | Reviewed interval | Full-frame baseline | With crops | Interpretation |
| --- | --- | --- | --- | --- |
| q01, backpack, clip 05 | 0–6s | No top-10 overlap | Relevant window at rank 8; Hit@1/5 still 0 | A rank-8 window is not a successful top-five recovery. Its leading crop evidence at 0s is detector-labelled `laptop`, not an identified backpack. Treat as an ambiguous overlap, not a confirmed improvement. |
| q01, backpack, clip 06 | 0–30s | Rank 1 overlaps | Rank 1 overlaps | Both variants pass this broad interval check; it does not demonstrate a crop benefit. |
| q02, bag on bench, clip 06 | 0–30s | Rank 1 overlaps | Rank 1 overlaps | The broad interval makes this a weak check. The previous live run's specific 8–12s candidate was not present in the crop top 10; the crop top result was 0–4s. |
| q04, absent blue handled case, clip 05 | Absent | 10 suggestions | 10 suggestions | No reduction in negative-query candidates. Top visual cosine rose from 0.2154 to 0.2752. |
| q04, absent blue handled case, clip 06 | Absent | 10 suggestions | 10 suggestions | No reduction in negative-query candidates. Top visual cosine rose from 0.1705 to 0.2743. |

Cosine values are uncalibrated ranking similarities, not probabilities or
presence confidence. A higher score on the negative cases is a reason to avoid
introducing a crop-based cutoff or boosting crops by score alone.

## Larger encoder feasibility

The current index uses OpenCLIP `ViT-B-32:laion2b_s34b_b79k`. A candidate
OpenCLIP ViT-L/14 checkpoint was not benchmarked: Hugging Face downloads stalled
with zero-byte incomplete files, and a direct range request to the OpenAI
Azure-hosted checkpoint timed out before returning data. That checkpoint is
about 933 MB. This is a network/download blocker, not evidence that the larger
encoder performs better or worse. No new dependencies or model weights were
added to the repository.

## Decision and next evidence

- Keep production embeddings, crop weighting, score gates, and retrieval
  settings unchanged.
- Do not claim that object crops fix the clip 05 backpack false negative.
- Before reconsidering crops, visually review the stored proposals and collect
  more reviewed positive and negative cases with narrower event intervals,
  including clips from different scenes/cameras.
- Retry a larger encoder only when its checkpoint can be retrieved reliably;
  evaluate it side by side on the same frozen held-out cases before rebuilding
  any index. If selected later, all indexed vectors must use the same encoder.
- Keep the demo release gate closed until the clip 05 positive case and the
  investigator production-browser workflow are verified.

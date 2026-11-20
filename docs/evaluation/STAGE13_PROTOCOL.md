# G507 independent evaluation protocol

The proposed local manifest is an annotation template, not ground truth. Neither
discovery contact sheets nor YOLO/CLIP predictions verify full event intervals.

## Human review

1. Open the local review pack and watch all seven 40-second clips continuously.
   Use its native player to inspect uncertain moments. Do this before inspecting
   retrieval predictions. The contact sheets are sparse navigation aids.
2. Preserve `annotations.proposed.json`. Save edits in a separate
   `annotations.verified.json`. For each query, review **all seven** clips, list
   every relevant clip-relative interval and correct unsupported attributes.
   Reject unsupported action queries rather than asserting a placement/pickup
   because an object appears in two different sampled frames.
3. A negative label requires exhaustive review of the whole collection for that
   target. Do not infer absence from a detector returning no boxes or from one
   empty frame. Ambiguous small/occluded objects remain unverified.
4. Set `review_status: verified`, identify the actual `reviewer`, keep
   `independent_of_predictions: true`, and set `scope_review_complete: true` only
   when those statements are true. Do not attribute agent proposals to Goodluck.
5. For positives, fill `relevant_intervals` with `video_id`, `start_seconds`,
   `end_seconds`. Use exclusive interval ends within the clip duration. Negatives
   use an empty interval list. Record event type, attributes and occlusion notes.
   All query `scope_video_ids` must equal the seven-clip collection. If changing
   scope, create a new manifest/index collection and report the change.

Example structure, with deliberately illustrative values rather than G507 labels:

```json
{"video_id":"EXAMPLE_CLIP_ID","start_seconds":3.2,"end_seconds":8.5}
```

Use the active `frame_aligned/` review set (IDs ending `_f2`). Clip frame 0 maps
to `source_frame_start` in the decoded AVI. Thus original decoded frame = source
frame start + clip frame; original nominal CFR offset = `source_offset_seconds`
+ clip-relative time. This is deliberately separate from AVI PTS, which can start
nonzero or be missing. Add this to
`filename_recording_start` for an **unspecified-timezone filename wall time**;
never call it UTC without establishing the original timezone.

## Retrieval comparison

All modes use identical natural-language query strings and collection scope:
visual is CLIP text-to-image, text is MiniLM text-to-reviewed-description, and
hybrid is window RRF. Use the existing models/settings without tuning to this
small selected set. Do not insert query answers or ground-truth event intervals
as retrieval descriptions. Any future text descriptions must be a separately
prepared, reviewed, frozen index with disclosed annotation provenance and coverage.
Without such text coverage, text/hybrid effectiveness cannot be fairly compared.

The runner verifies the isolated schema contains exactly the manifest video IDs
and source hashes. Proposed queries are skipped; malformed verified labels fail.
No labels means `awaiting_human_verification`, with no accuracy aggregates.
Before timed queries, each mode is warmed once using the first verified query;
warm-up cost is reported separately. Timed values exclude indexing.

A returned window is relevant when its video ID matches a labelled interval and
`intersection_duration / min(window_duration, labelled_interval_duration) >= 0.5`.
This is a temporal overlap coefficient, **not temporal IoU**. Exact duplicate
video/time windows are collapsed before scoring.

- **Hit@1/Hit@5:** fraction of positive queries with at least one relevant window
  in the first one/five returned windows. Hit@5 is not Recall@5.
- **Precision@5:** relevant windows in the first five divided by five, including
  missing ranks in the denominator. This measures window relevance, not the
  fraction of individual physical objects found.
- **MRR:** reciprocal rank of the first relevant window, zero when absent, averaged
  over positive queries. It is bounded by the current retrieval protocol: at most
  100 raw candidates per channel before window collapse and 100 returned windows.
  It is not exhaustive collection recall.
- **Negatives:** report candidate-return rate and returned counts. The system has
  no calibrated rejection threshold, so a ranked result does not establish
  presence. Do not call these counts calibrated false-positive accuracy.
- **Latency:** warm per-query/mode elapsed time, with cold warm-up costs separate.
  Indexing costs and process memory come from the independent indexing run report.

Report positive/negative counts, reviewed/skipped queries, actual indexed coverage,
selected clips and source videos. Exclude text/hybrid aggregate comparisons when
source-validated reviewed text is missing for any clip. Never report the tests of
synthetic metric fixtures as real G507 retrieval scores.

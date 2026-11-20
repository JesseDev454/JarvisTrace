# Evaluate the human-reviewed Stage 9 sample

The review page now has an **Update evaluation** button. Start it with
`--reviewer "Goodluck Kassa"` to prefill the reviewer name. This preference does
not create or approve any annotations. Saved annotations retain their recorded
reviewer names and revision history.

From the project root, run the standalone evaluator:

```powershell
.\.venv\Scripts\python.exe -m src.stage9_evaluate `
  --report outputs/stage9/20261116T164229528159Z/analysis.json
```

The evaluator reads the review SQLite database without changing it. Optional
`--output <new-file.json>` exports results and rejects existing output files.
The original prediction reports, tracking exports and PostgreSQL are unchanged.

Only the latest **certain** annotation for each frame is scored. Uncertain and
unreviewed frames are excluded; an empty usable sample returns `awaiting_review`
and unavailable precision/recall, rather than an accuracy claim. At the time of
verification, the real review store had one uncertain frame, zero evaluable
frames and 239 unreviewed frames. This tool did not create any real-footage labels.

Detection metrics are true positives, false positives, false negatives, precision,
recall and mean matched box intersection-over-union (IoU). Default matching IoU is
0.5, configurable by `--iou-threshold`. This is an evaluation criterion, not a
detector-confidence threshold. One-to-one matching maximizes the number of valid
matches using augmenting paths with deterministic IoU preferences; it does not
optimize total IoU among every equally sized assignment. Duplicate detections can
count as false positives; unmatched human boxes count as misses. Frame annotations
must enumerate the visible targets for frame-level precision to be meaningful.

The evaluator counts changes of local track ID only when the same manual object
identity has matched tracked detections in adjacent certain reviewed video frames.
A matched tracked observation followed by a missed/untracked observation starts a
track gap. These are descriptive sample counts, not IDF1, MOTA, physical identity
verification or a reconstruction across occlusions. Human IDs must themselves be
visually justified. Motion accuracy remains explicitly unevaluated until independent
interval-level ground truth exists; a frame's motion-context label is insufficient.

Report, prediction-file and review-history SHA-256 hashes identify evaluation
inputs. Six new unit fixtures verify empty samples, latest revisions, exclusions,
known box overlap/misses/duplicates, matching conflicts and adjacent track changes.
Review integration tests exercise the evaluation endpoint and confirm that the
configured name creates no annotations. Independent code review found no important
issues. Final combined suite: **84 passed, 28 subtests passed** in **46.23 seconds**.

```powershell
.\.venv\Scripts\python.exe -m unittest tests.test_stage9_evaluate tests.test_stage9_review -v
$env:HF_HUB_OFFLINE = '1'
$env:TRANSFORMERS_OFFLINE = '1'
$env:PYTHONPATH = (Join-Path (Get-Location) 'scene-search-prototype')
.\.venv\Scripts\python.exe -m pytest -q
```

# G507 portable-object crop experiment

The fixed YOLO-crop experiment improved ranking on the same four confirmed
positive queries. It preserves the full-frame baseline and uses a separate
`stage13_g507_object_crops_v1` database schema. This is an exploratory result
on four known queries from one camera, not 100% system accuracy or a held-out
evaluation. No negative-query performance was measured.

| Measure | Full frame | Crop only | Frame + crop |
|---|---:|---:|---:|
| Hit@1 | 25% (1/4) | 100% (4/4) | 100% (4/4) |
| Hit@5 | 50% (2/4) | 100% (4/4) | 100% (4/4) |
| Mean Precision@5 | 30% | 65% | 65% |
| Bounded candidate MRR | 0.3242 | 1.0000 | 1.0000 |

| Query | Full-frame first relevant rank | Crop-only rank | Combined rank |
|---|---:|---:|---:|
| Person carrying a dark backpack | 1 | 1 | 1 |
| Dark bag resting on a bench | 15 | 1 | 1 |
| Person carrying a shoulder bag | 5 | 1 | 1 |
| Person carrying a blue handled case | 33 | 1 | 1 |

Combined Precision@5 is 1.0, 0.8, 0.4 and 0.4 respectively. False matches
remain, particularly for shoulder bags and handled cases. Matching labelled
windows does not prove which object in the window caused the score or that
CLIP understands carrying, resting or an action transition. The best case
result was clip 06 at 20–24 seconds; the bench-bag result was clip 06 at
32–36 seconds. No label was used to choose those predictions.

## Fixed protocol

- Same seven frame-aligned 40-second clips, four approved positive queries,
  source hashes, intervals and encoder as the baseline. Four proposed queries
  stay excluded. The human-confirmed AI-assisted reference is not blinded.
- YOLO11n from existing local weights, image size 1280, confidence 0.25, NMS
  IoU 0.7; portable COCO classes 24/26/28/39/63/67. Ten percent padding on
  each side, clamped to source pixels; at most eight unique crops per frame
  and 500 total, fixed before inspecting retrieval outcomes. No retraining,
  model downloads or query-specific filtering/tuning.
- Detect on exactly the 140 already indexed raw PNG frames (two-second
  sampling). Encode crops with the same cached OpenCLIP ViT-B-32,
  `laion2b_s34b_b79k`, 512 dimensions, batch size two. Copy full-frame
  evidence/vector values directly with SQL; row IDs and creation times are new.
- Existing Stage 11 pure visual search, four-second windows, RRF k=60,
  100 raw candidates and up to 100 windows. Combined selects the best image
  variant per sampled frame before window collapse; it does not combine
  detector confidence with CLIP similarity. Crop-only has 25 returned windows;
  full frame has 53–59 and combined has 55–59, so candidate coverage differs.
- Relevance remains overlap coefficient >= 0.5, not temporal IoU; Precision@5
  divides by five. MRR is bounded by the candidate cap. Scores are rankings,
  not calibrated absence probabilities or object identities.

## Workload and verification

The two-frame smoke produced zero proposals. The complete pass produced
39 proposals: 22 labelled backpack, 13 handbag, two suitcase, one cell phone
and one laptop. These are detector labels, not human-verified category counts.
Six clips had crops; the empty control clip had none. No per-frame or total
cap was reached. Crop-only cannot retrieve a frame with no proposal; a missed
detection must not be interpreted as absence. The contact sheet contains
questionable classifications and partial/occluded views, so detector recall
or precision is not claimed.

Measured detector-loop time was 40.32 seconds; crop embedding/insertion time
was 8.79 seconds. Both exclude Python/model startup; the embedding timer also
excludes manifest validation and baseline cloning. Peak allocated PyTorch CUDA
memory was approximately 96 MiB for the detector and 601 MiB for crop encoding,
not total GPU memory. Warm mean query times in this run were 0.096s full frame,
0.082s crop only and 0.092s combined; four timed queries per mode are insufficient
for a performance benchmark. Index size is 140 frames + 39 crops = 179 vectors.
Repeating each crop insertion added zero rows.

The evaluator required exact full-frame metrics and all ranked windows to
reproduce the frozen baseline. Crop hashes were reconstructed from parent
pixels and boxes; source/frame/profile/encoder provenance was checked. The
post-run audit checks all eleven originals, all seven active derivatives,
baseline database content, frozen annotations/results and public table content.
No ground-truth labels, generated descriptions or reviewed text were indexed.
The production preview retains its original baseline schema.

Regression verification: **168 tests and 85 subtests passed**, with PostgreSQL
integration and importlib collection enabled. Independent review findings on
crop pixel binding, resume conflicts and report overwrite behavior were fixed.

## Local evidence and commands

Artifacts under `outputs/stage13/object_crops/`:

- `full/detections.json`: every frame, raw proposal, chosen crop, settings and hashes.
- `full/index.json`: counts, duplicate insertion and indexing resource results.
- `comparison.json`: complete per-query rankings, evidence, metrics and timings.
- `crop_contact_sheet.jpg`: all 39 proposed crops for inspection.
- `preservation_before.json` and `preservation_after.json`: preservation audit.

Run each command with a fresh output folder/schema/report for a new experiment;
completed reports are deliberately preserved.

```powershell
$env:HF_HUB_OFFLINE = '1'
$env:TRANSFORMERS_OFFLINE = '1'
.\.venv\Scripts\python.exe -m src.stage13_crops detect --baseline stage13_g507_eval_f2 --weights scene-search-prototype/yolo11n.pt --output outputs/stage13/object_crops/full
.\.venv\Scripts\python.exe -m src.stage13_crops index --manifest outputs/stage13/object_crops/full/detections.json --target stage13_g507_object_crops_v1
.\.venv\Scripts\python.exe -m src.stage13_crop_evaluate --annotations datasets/mevid/G507/evaluation/frame_aligned/interval_review_v1/annotations.confirmed_positive_subset.json --schema stage13_g507_object_crops_v1 --output outputs/stage13/object_crops/comparison.json --baseline-report outputs/stage13/human_confirmed_positive_evaluation.json
```

The next effectiveness step is more verified positive and negative queries on
new clips, with a held-out split before tuning. Keep this configuration frozen
for that check. Review detector misses and state/action false matches separately.
Text/hybrid evaluation still requires separately authored, reviewed descriptions
that do not copy evaluation answers. MediaMTX streaming remains a separate later
integration task; this experiment does not establish live CCTV readiness.

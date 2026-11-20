# G507 object-crop experiment plan

**Goal:** Compare frozen full-frame, detector-crop-only and combined visual
retrieval on the same seven clips and four approved positive queries.

**Architecture:** Keep Stage 13 baseline schema and results untouched. Detect
portable objects on the exact 140 existing sampled frames, record raw detector
provenance, encode derived crops using the existing CLIP encoder, and clone
baseline frame vectors exactly into a separate owned schema. Reuse Stage 11
window ranking and the confirmed-subset scorer, without adding text captions.

**Chosen defaults:** Existing local YOLO11n, imgsz 1280, confidence .25, IoU .7,
portable COCO classes 24/26/28/39/63/67, 10% box padding, at most eight unique
crops/frame and 500 total crops, CLIP batch size two. These defaults are fixed
before seeing crop retrieval results. No query-dependent filtering or retuning.

## Tasks

- [x] Freeze baseline schemas/results, source/image/encoder identity, approved
  annotation hash and experiment settings. Preserve unrelated working changes.
- [x] TDD detector manifest/crop validation and schema-specific cloning/indexing.
  Validate source/frame hashes and deterministic box bounds; no original edits,
  public fallback, model downloads or ground-truth caption insertion.
- [x] Run a two-frame detector smoke check, then the bounded 140-frame detection
  pass. Inspect actual crop count/resource cost before encoding.
- [x] Index verified derived crops separately; preserve exact baseline vectors;
  repeat insertion must add zero rows. Isolate detector/preprocessing variants.
- [x] TDD and execute full-frame/crop/combined comparison with identical text
  queries, temporal labels, camera/time scope, candidate caps and fusion settings.
  Save per-query ranks, evidence, latency and metrics; verify baseline reproduction.
- [x] Independent code review, regression checks, source/database preservation,
  and a report of measured outcomes and limits.

## Outcome

39 detector proposals on 140 frames; separate index has 179 vectors. Repeated
crop insertion adds zero. Full-frame baseline rankings/metrics reproduced exactly.
Crop-only and combined Hit@1 are 4/4 versus baseline 1/4; mean P@5 is 0.65 versus
0.30. This is the same four known positives, with no verified negative queries.
Independent review fixes cover pixel reconstruction, exact frame binding,
conflicting existing metadata/vectors, and early completed-report preservation.
168 tests and 85 subtests pass with PostgreSQL. Public/source/baseline/frozen-label
audits pass. See `docs/evaluation/STAGE13_OBJECT_CROPS.md` for the full protocol.

## Constraints

- Four positive queries are a human-confirmed AI-assisted exploratory subset,
  not a blind or held-out benchmark. No negative performance or generalized
  improvement claims. Frozen raw results allow exact comparison.
- Combined visual retrieval picks the best frame/crop per sampled frame before
  window collapse, as the existing Stage 10/11 implementation does. It is not a
  new weighted sum of detector confidences and CLIP scores.
- Baseline parameters include two-second samples, four-second windows, k=60 RRF,
  100 raw candidates and 100 result windows. Crops can alter candidate allocation.
- Model-inference code runs offline, outside the API. No production deployment,
  main-branch merge, commit or push is part of this experiment.

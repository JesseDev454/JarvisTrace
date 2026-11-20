# Assisted review of the backpack sequence

At Goodluck Kassa's request, Codex visually inspected 40 decoded frames:
67 through 101 inclusive, plus 144, 168, 192, 216 and 239. All frames were inspected
as raw chronological images without detector overlays before drafting boxes.

Draft annotations and image previews are in
`outputs/stage9/assisted-review-20261116/`. `assisted_annotations.json` preserves
source video hash, run scope and AI authorship. Boxes are approximate visible
extents estimated visually; no detector boxes were copied or interpolated.
`assisted_review.sqlite3` holds the same 40 drafts for the local frame viewer.
The original human review database is unchanged; its preservation hash is saved
in `preservation.json`. PostgreSQL and prediction exports were not modified.

View the completed drafts:

```powershell
.\.venv\Scripts\python.exe -m src.stage9_review `
  --report outputs/stage9/20261116T164229528159Z/analysis.json `
  --review-db outputs/stage9/assisted-review-20261116/assisted_review.sqlite3 `
  --reviewer "Goodluck Kassa" --port 8766
```

Open http://127.0.0.1:8766 and jump to the interaction or later interval. Draft
frames display **AI-assisted draft** and green boxes. Stored reviewer is
**Codex AI-assisted visual review**, not Goodluck. A subsequent human correction
saved in the viewer receives its own reviewer and revision; AI history remains.
Default evaluation excludes assisted drafts. The original human-only review page
on port 8765 remains separate.

## Findings and limits

The inspected sequence shows handling near the chair, lifting/carrying toward
the desk and placement on it. Support is left unspecified during the transfer.
The five late samples show the bag on the desk; they do not establish the state
of every intervening frame or its current physical location. Identity continuity
is a visual draft interpretation of this clip, not automatic tracker-ID merging.

The diagnostic comparison at IoU >= 0.5 yields **14 matched detections and 26
misses across the 40 draft frames**, with no unmatched backpack predictions.
This is a comparison to AI-assisted reference boxes, not independently measured
human-ground-truth accuracy. Most sampled transfer frames lack exported backpack
observations, which explains why the saved tracks do not reconstruct the full
event. Investigate the original detector outputs and tracker class changes before
attributing all missing rows to the detector itself.

Reproduce that diagnostic comparison:

```powershell
.\.venv\Scripts\python.exe -m src.stage9_evaluate `
  --report outputs/stage9/20261116T164229528159Z/analysis.json `
  --review-db outputs/stage9/assisted-review-20261116/assisted_review.sqlite3 `
  --include-assisted
```

`assisted_comparison.json` records this result. Omitting `--include-assisted`
excludes these 40 drafts, leaves precision/recall unavailable and reports the
excluded draft count. Regression coverage checks that an assisted revision cannot
silently resurrect an older human annotation as the latest scored state.

Verification: **85 tests passed, 28 subtests passed** in 46.54 seconds. Independent
review found no important issues in assisted-label exclusion. All 40 saved rows
carry AI authorship, and the original human review store's SHA-256 is unchanged.

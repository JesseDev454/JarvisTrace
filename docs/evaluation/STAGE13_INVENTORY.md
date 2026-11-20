# G507 inventory summary

All 11 AVI files were probed, hashed and fully decoded by FFmpeg with error-on-decode-failure. Every decoded frame count matched its declared count. All are H.264, 1920×1072 and nominal 30 fps. No original file was edited.

| Filename start (2018-05-18) | Duration (s) | Frames | Bytes | Gap from prior filename end (s) |
|---|---:|---:|---:|---:|
| 10:20:00 | 300.000000 | 9000 | 88784004 | None |
| 10:25:00 | 300.066667 | 9002 | 107107062 | 0.0 |
| 10:30:00 | 300.100000 | 9003 | 113814232 | 0.0 |
| 10:35:00 | 300.000000 | 9000 | 109569288 | 0.0 |
| 10:40:00 | 300.033333 | 9001 | 99892344 | 0.0 |
| 10:45:00 | 300.066667 | 9002 | 100977086 | 0.0 |
| 10:50:00 | 300.000000 | 9000 | 87082248 | 0.0 |
| 10:55:00 | 300.400000 | 9012 | 91304422 | 0.0 |
| 11:00:00 | 300.166667 | 9005 | 112694706 | 0.0 |
| 14:45:01 | 300.366667 | 9011 | 106491722 | 13201.0 |
| 14:50:01 | 300.100000 | 9003 | 106339786 | 0.0 |

Morning nine: 2700.833334s of measured media, nominal 10:20–11:05 (45 minutes). All 11: 3301.300001s (55.0217 minutes), 1,124,056,900 bytes.

Morning filenames have no missing nominal intervals. Individual media durations overrun filename boundaries by up to 0.4s; this does not establish duplicated or missing capture frames. The afternoon pair begins after a 13,201-second nominal gap. Filename timezone and actual capture-clock synchronization are unverified.

Contact sheets (nine discovery samples per source) were visually inspected locally. They show a fixed indoor waiting-room view with benches, pedestrian movement, backpacks/shoulder bags and handled cases. Small handheld items and event boundaries need closer review. A contact sheet is not an exhaustive absence test. Compression/block artifacts are visible in some sampled imagery despite successful decoding.

Full local detail: `datasets/mevid/G507/evaluation/inventory/inventory.json`, `inventory.csv`, and `contact_sheets/`. The morning block is the primary scope; afternoon sources are inventoried but not used in the selected seven clips.

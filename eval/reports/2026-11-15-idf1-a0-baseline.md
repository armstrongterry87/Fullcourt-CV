# IDF1 baseline for the A0 default tracker (task 003, step 4)

Date: 2026-11-15. This is the project's first real accuracy measurement. Everything before it, including the task 001 comparisons, measured self-consistency. This is the first comparison against a hand-corrected key.

## What was evaluated
- Tracker output: `runs/task001_eval/A0/xbotgo_tracks.csv` (committed A0 default; 45 unique IDs over the full clip). The tracker code that produced it is committed as **a9144a9** ("Task 001: tracker presets, detection cache, eval (ByteTrack A0 default)"). The CSV was generated from the working tree just before that commit, and a later uncommitted change flipped the default tracker to ByteTrack A0. A0 was selected explicitly for this run, so the output is unaffected, but the run predates the commit and was not produced from a clean checkout of it.
- Key: `data/ground_truth/xbotgo_10s_mot.txt` (289 boxes, 40 frames, 9 players, read-only).
- Frame map: sidecar `frame_map`. MOT frames 1..40 map to original frames 307..541, stride 6. Only those 40 frames are scored.
- Sample size: **1 game clip, 10 s, 40 frames, 9 players.** This is very small, so treat the numbers as indicative only.
- Matching: IoU distance with a 0.5 threshold. Metrics come from `motmetrics` 1.4.0 (MIT licence, added to `requirements.txt`).

Command (from the repo root):
```
.venv/Scripts/python.exe eval/score_tracking.py runs/task001_eval/A0/xbotgo_tracks.csv --label A0
```
The script takes several CSVs at once (label defaults to the parent directory name), `--modes` to choose a filter, and `--json out.json`.

**Reviewable surface: the matching distances are a local IoU implementation, not the library's.** `motmetrics` 1.4.0's `iou_matrix` crashes on NumPy 2 (`np.asfarray` was removed in NumPy 2.0). I chose not to downgrade NumPy and wrote `iou_dist()` in `eval/score_tracking.py` (boxes in tlwh, distance = 1 - IoU, set to NaN above 0.5). Only the accumulator and the metric computation (IDF1, switches, MT/ML and so on) come from motmetrics 1.4.0. Every number in the table below depends on `iou_dist()`, so a sign error or a transposed box there would shift the scores without crashing. It needs review.

## Results
| Filter | IDF1 | IDP | IDR | ID switches | Precision | Recall | MT / PT / ML (of 9) | FP | Misses |
|---|---|---|---|---|---|---|---|---|---|
| players only | 0.619 | 0.562 | 0.689 | 11 | 0.811 | 0.993 | 9 / 0 / 0 | 67 | 2 |
| players + referees | 0.602 | 0.535 | 0.689 | 11 | 0.777 | 1.000 | 9 / 0 / 0 | 83 | 0 |

**Primary number: players only (IDF1 0.619).** The key has no referees, and the tracker's "referee" class mixes real players with true officials and spectators. Keeping it adds 16 false positives but recovers only 2 matches.

The gap between the filters is small here (2 matched boxes recovered, 16 extra FPs, IDF1 -0.017). In this window, the class vote therefore costs about 2 true player boxes out of 289. Task 002's role vote has little to fix in this clip. One clip is a weak basis for that conclusion.

Sanity checks:
- Overlap: the GT and the tracker CSV both cover all 40 labelled frames (40/40).
- Distinct tracker IDs inside the 40 frames: **17** against 9 in the key. This reproduces task 001's count and is the ID-fragmentation story in miniature. It matches the IDP of 0.56 and the 11 switches.
- Tracker rows with `track_id < 0` (untracked detections) are dropped before scoring: 27 in players mode, 39 in players+referees mode. This matches the key, which was built without them. They are not counted as false positives, so precision is slightly flattering to the tracker on that point.

## Caveats (read before using any number here)
1. **The key was built from the tracker's own boxes, and no new boxes were drawn.** Recall is therefore close to 1 by construction: every key box is a tracker box. The 0.993 recall says nothing about detection quality. Only the identity metrics (IDF1, IDP/IDR, ID switches, MT/ML) carry real information, because the IDs were hand-corrected. Do not report recall as a detection result.
2. **Precision is biased downward and is not an absolute figure.** Real players the tracker found but the key left unlabelled count as false positives. The README names player 2 in frames 38-40 and player 8 in frames 18-20 as examples. Some of the 67 FPs are therefore real players.
3. **The bias is systematic, so comparisons between configs on the same key remain meaningful.** The same people are unlabelled for every config, so a difference in precision or IDF1 between configs is informative even though the absolute value is pessimistic.
4. **The bias is not neutral across detection thresholds.** A setting that detects more people produces more apparent false positives, and some of them are real unlabelled players. Precision and IDF1 will therefore look worse at lower thresholds and better at higher ones partly as an artifact. Task 002's grid varies this threshold, so a "higher threshold is better" conclusion must not be drawn from this key alone. Weight the ID metrics (switches, MT/ML) more than precision there. Also, a setting that detects a player the key omits is penalised, and a setting that misses one is not. To fix this, add the missing boxes in CVAT and re-export.
5. One 10 s clip with 9 players. Differences of a few switches or a few IDF1 points between configs are probably noise.

## Comparison to previous report
No previous accuracy report exists against this key. Earlier reports (`2026-11-15-botsort-vs-bytetrack*.md`) used self-consistency metrics. This report is the baseline that later runs should be compared against.

## Failure cases
Not yet itemised per frame. The 11 ID switches and the 8 extra tracker IDs (17 vs 9) are the failure. I have not yet listed the frames. I can add a per-switch frame list (GT ID, old and new tracker ID, MOT frame, time) when you want it, so a human can watch those moments.

## Verdict
A0 keeps all 9 players mostly tracked but fragments identities: IDF1 is 0.619, with 11 ID switches and 17 tracker IDs for 9 people in 10 s. Recall is circular and precision is pessimistic because of how the key was built, so use the ID metrics for comparing configs, and treat threshold comparisons with care.

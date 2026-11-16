# Tracker self-consistency: BoT-SORT vs ByteTrack (Task 001)

This is a **self-consistency comparison, not an accuracy measurement.** `data/ground_truth/` is empty and there are no hand-labelled track IDs. No ID-switch rate, MOTA or IDF1 can be claimed. The metrics only describe how fragmented the tracker output looks.

## What was evaluated
- Clip: `data/raw/xbotgo.mp4`, 1 clip, 719 frames, 23.951 fps (30 s), panning camera. **Sample size is one 30 s clip, no held-out games. Treat differences of a few IDs as noise.**
- Detector: `models/weights.pt` (fine-tuned RF-DETR Small), threshold 0.40. Detections were cached once (`runs/cache/xbotgo__wfc2f7c4e__small__thr0.40.npz`) and replayed for every preset, so all presets see identical detections (detection is deterministic run to run, verified by the caller).
- Code: `scripts/baseline_track.py` and `src/tracking/factory.py` `PRESETS` as of 2026-11-15 (measured from the working tree just before commit `a9144a9`; files measured: `scripts/baseline_track.py`, `src/tracking/factory.py`, `src/detection/cache.py`. Note `a9144a9` and its child `abbfbe4` share the same commit message but are different commits; `abbfbe4` carries Step 0). Metrics script: `eval/track_consistency.py`. Raw outputs: `runs/task001_eval/<preset>/`. Each preset has its own directory; no collisions.

## Baseline correction
The task brief states ByteTrack = 43 IDs / 31.2% ball. **That is not reproducible.** `models/weights.pt` was replaced at 19:18:50 and the brief's table came from the older checkpoint. With the current weights, A0 gives **45 IDs / 28.8% ball**, and the A0 CSV is byte-identical to a pre-change run (`runs/finetuned/xbotgo_tracks.csv`). All comparisons here use 45. (Gate passed: A0 = 45 / 28.8%. Sanity: B0 = 50, B1 = 49 reproduced.)

## Results

Ball detection: **28.8% of frames, identical for all presets.** The ball never passes through `tracker.update`, so this is a detector number and cannot differ between trackers.

| Preset | Tracker | Buffer | Other | Unique player IDs | IDs < 2 s (span / presence) | IDs >= 5 s (span / presence) | Tracker ms/frame | End-to-end fps* |
|---|---|---|---|---|---|---|---|---|
| A0 | ByteTrack | 1 s | default | **45** | 12 / 16 | 24 / 21 | 0.69 | 3.33 |
| A1 | ByteTrack | 3 s | | **43** | 11 / 14 | 24 / 21 | 0.70 | 3.33 |
| B0 | BoT-SORT | 1 s | CMC off | 50 | 14 / 20 | 20 / 15 | 1.02 | 3.32 |
| B1 | BoT-SORT | 1 s | CMC sparseOptFlow | 49 | 15 / 18 | 19 / 15 | 12.96 | 3.20 |
| B2 | BoT-SORT | 3 s | CMC sparseOptFlow | 46 | 14 / 16 | 19 / 15 | 12.64 | 3.20 |
| B3 | BoT-SORT | 3 s | CMC orb | 45 | 14 / 17 | 19 / 16 | 11.89 | 3.21 |
| B4 | BoT-SORT | 3 s | CMC, activation/high-conf thr 0.5 | 58 | 22 / 25 | 24 / 19 | 12.96 | 3.20 |

*End-to-end fps = 1 / (0.30 + tracker seconds), where 0.30 s/frame is the measured CPU detection cost on this machine. Detection dominates, so tracker choice changes throughput by about 4%. CMC costs about 11 to 12 ms/frame extra, which is irrelevant next to detection but would matter on a GPU.

Lifetime is shown two ways. **Span** = (last_frame - first_frame + 1) / fps over player-labelled rows; it includes frames where the ID was absent or labelled referee. **Presence** = distinct player-labelled frames / fps. About 80% of tracks have gaps (A0: 36 of 45), so span overstates ">= 5 s" by 3 to 5 IDs and understates "< 2 s" by 2 to 4. Presence is the honest reading of "long-lived"; BoT-SORT has fewer long IDs than ByteTrack under presence for every preset (15 to 19 vs 21), and under span for all but B4, so the inflation does not manufacture the verdict.

**What "unique player IDs" counts:** every track labelled "player" in at least one frame. A track labelled referee in most of its frames still counts fully, and a track that flickers is also counted in the referee totals. This is common here: 14 to 19 IDs per preset appear in both lists, i.e. a large share of the 45 (19/45 = 42% for A0) are not clean players.

**Direct evidence of label flicker:** A1's largest gap in player-labelled frames is 97 frames (4.1 s), longer than its own 72-frame lost-track budget. The track therefore stayed alive while the detector's label flipped to "referee". So an ID can count as ">= 5 s" by span while much of it was labelled referee. This supports the hypothesis below that detector flicker, not camera motion, drives fragmentation.

Generated table (all columns, including both lifetimes): `eval/reports/2026-11-15-botsort-vs-bytetrack-table.md`, produced by `eval/track_consistency.py --out`.

## B0 vs B1: the CMC contrast
B0 and B1 differ only in `enable_cmc`, so this is the only pair that isolates CMC. B0 (no CMC) = 50 IDs, B1 (CMC) = 49. A one-ID difference on one clip is within noise: **CMC did not measurably help on this clip.** Short-lived IDs by span went the wrong way (14 to 15); by presence they improved (20 to 18), so there is no consistent CMC signal. The camera-pan benefit that motivated the task is not visible in these metrics.

## Verdict
BoT-SORT is **not better than ByteTrack here**. The best BoT-SORT preset (B3, 45) only ties the default ByteTrack (A0, 45), and BoT-SORT at matching buffers is equal or worse (1 s: 49 to 50 vs 45; 3 s: 45 to 46 vs 43). It also has fewer long-lived IDs by presence (15 to 19 vs 21; by span 19 to 24 vs 24). The task criterion "fewer than 43 IDs" is **not met** by any BoT-SORT preset (best = 45), and the brief's 43 target (older checkpoint) itself was measured on a different checkpoint. Only A1 (ByteTrack, 3 s buffer) reaches 43, a ByteTrack-only change. That 43 is partly an artifact of wrong merges (see Merge check), so it is not an endorsement of A1.

Best read on why (hypotheses, not tested):
- The dominant fragmentation is probably not camera motion. 14 to 19 IDs flicker player/referee, and the detector is thresholded at 0.4 with many low-confidence boxes, so tracks break from detections appearing and vanishing. CMC cannot fix that.
- BoT-SORT's default activation threshold (0.7) and high-conf threshold (0.6) sit well above a 0.4 detector threshold, so many detections never start a track or are only used in a second association. B4 (lowering to 0.5) made things much worse (58), which suggests low-confidence detections spawn many junk tracks; this is consistent with the detector, not the tracker, being the bottleneck.
- Lengthening the buffer from 1 s to 3 s helped both families by about 2 to 4 IDs, the largest consistent effect in the grid.
- The camera pan may be too slow or small for CMC to matter, or sparse optical flow may fail on a gym floor with sparse texture. Unverified.

## Caveat: fewer IDs is not automatically better
Two players merging into one persistent ID would lower every metric in the table while making tracking worse. Nothing here detects that. Longer buffers (A1, B2 to B4) raise that risk. The annotated video should be reviewed by hand before trusting any ID count.

## Merge check
Added after the fact. **These findings come from a manual crop check of A1's reconnections after gaps over 1 s in `runs/task001_eval/A1/xbotgo_tracks.csv`. They were produced by hand, not measured by the evaluator, and have not been independently verified.**

| A1 ID | Frames | Student's reading |
|---|---|---|
| 28 | 666 -> 709 | Blue-jersey player to a different white-shirt kid. **Wrong merge.** |
| 20 | 500 -> 538 | White-jersey player to a spectator by the door. **Wrong merge.** |
| 0 | 159 -> 192 | Similar white/red jersey. Probably correct. |

Two of A1's three long-gap reconnections are wrong merges (n = 3, one clip, so this is a small sample, but the failure is qualitative). The "fewer IDs is not automatically better" caveat above was borne out: A1 reached 43 partly by merging different people. The measured ID counts are unchanged; what changes is how to read them.

Methodological lesson: a lower unique-ID count cannot be read as better tracking without a merge check, and no metric in the table above detects a merge. This is why task 002 makes the merge check a required part of the eval. The same check has not been run on B2 to B4 (also 3 s buffers), so their counts carry the same unresolved risk.

## Recommendation
Rule: simplest preset within 10% of the fewest IDs (43, so <= 47.3 qualifies: A0, A1, B2, B3). **Winner: A0 (ByteTrack, defaults)**, the simplest. A1 is **no longer a candidate on its ID count**: the merge check above found that its 43 is partly an artifact of wrong merges. **A0 remains the recommendation**, now on stronger evidence than the simplicity tie-break alone. Do not switch the default to BoT-SORT on this evidence. Real conclusions need labelled tracks (even about 10 s of hand-labelled IDs would allow IDF1) and more than one clip.

Annotated video of the winner (same CSV as the no-video run, verified byte-identical): `c:\Users\Minim\Desktop\Startup VS Code\runs\task001_eval_video\A0\xbotgo_tracked.mp4`

## Failure cases
No ground truth, so no scored misses. What to watch in the video: players briefly labelled "referee" (spectators by the doors, out of scope for this task); the 12 A0 IDs under 2 s (list them via `eval/track_consistency.py` extended, or filter the CSV by lifetime); moments where two players cross.

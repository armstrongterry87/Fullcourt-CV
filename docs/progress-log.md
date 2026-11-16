# Progress Log

Newest entries on top.

## 2026-11-15
**Task 001: ByteTrack vs BoT-SORT for player tracking (steps 1-4 done, review done, not committed)**

**Headline: the task's premise did not hold on this clip.** BoT-SORT did not beat ByteTrack. Its camera motion compensation (CMC) gave no consistent signal. The one change that helped was a longer lost-track buffer, and it helped ByteTrack most. The ByteTrack default (A0, 45 unique player IDs) is the scored winner. The best BoT-SORT preset only ties it (B3, 45) at about 18x the tracker cost. Evidence: `eval/reports/2026-11-15-botsort-vs-bytetrack.md` (table: `eval/reports/2026-11-15-botsort-vs-bytetrack-table.md`). This is a self-consistency comparison on one 30 s clip, not an accuracy result.

What got done:
- `scripts/baseline_track.py` now takes `--tracker {bytetrack,botsort}` and `--preset` (explicit grid row, overrides `--tracker`). ByteTrack is preset A0. The code default is still `botsort` (B1). See the open decision below.
- New `src/tracking/factory.py`: a `TrackerSettings` dataclass and a `PRESETS` grid. This is the single home for tracker constants.
- New `src/detection/cache.py`: save raw detections to `.npz` once, then replay every tracker variant from identical detections (`--save-detections`, `--detections`).
- The frame is passed to BoT-SORT `update()` so CMC runs.
- Review found one BLOCKING issue and one output-path bug. Both were fixed in the 13 review fixes applied by model-builder. A re-review of the fixed diff found no remaining blocking issues. See the review section below.
- `eval/track_consistency.py` (written by evaluator) computes the metrics from the tracker CSVs. It was reviewed separately; its arithmetic was confirmed by an independent re-implementation of all seven rows.
- Script now prints fps to 3 decimals. The clip is 23.951 fps; the old one-decimal print showed "24.0 fps", which is why the brief said 24.

Results (one clip, 719 frames, 23.951 fps; source: `eval/reports/2026-11-15-botsort-vs-bytetrack.md`):

Lifetimes are given two ways. **Span** = first-to-last player-labelled frame, which includes gaps. **Presence** = distinct player-labelled frames / fps, which excludes gaps. Presence is the honest reading of "long-lived". Span is shown for comparison only.

| Preset | Tracker | Buffer | Other | Unique player IDs | IDs < 2 s (span / presence) | IDs >= 5 s (span / presence) | Tracker ms/frame | End-to-end fps |
|---|---|---|---|---|---|---|---|---|
| A0 | ByteTrack | 1 s | default | **45** | 12 / 16 | 24 / 21 | 0.69 | 3.33 |
| A1 | ByteTrack | 3 s | | **43** | 11 / 14 | 24 / 21 | 0.70 | 3.33 |
| B0 | BoT-SORT | 1 s | CMC off | 50 | 14 / 20 | 20 / 15 | 1.02 | 3.32 |
| B1 | BoT-SORT | 1 s | CMC sparseOptFlow | 49 | 15 / 18 | 19 / 15 | 12.96 | 3.20 |
| B2 | BoT-SORT | 3 s | CMC sparseOptFlow | 46 | 14 / 16 | 19 / 15 | 12.64 | 3.20 |
| B3 | BoT-SORT | 3 s | CMC orb | 45 | 14 / 17 | 19 / 16 | 11.89 | 3.21 |
| B4 | BoT-SORT | 3 s | CMC, activation/high-conf thr 0.5 | 58 | 22 / 25 | 24 / 19 | 12.96 | 3.20 |

- Ball detection: 28.8% of frames for every preset. The ball never passes through the tracker, so this is a detector number and is reported once.
- End-to-end fps = 1 / (0.30 s detection + tracker time). Detection dominates; tracker choice changes throughput by about 4%.
- Baseline: A0 reproduces 45 IDs / 28.8% ball, and its CSV is byte-identical to `runs/finetuned/xbotgo_tracks.csv`.
- Long-lived IDs by presence: A0 and A1 have 21; every BoT-SORT preset has 15 to 19. By span, BoT-SORT has fewer long-lived IDs for all presets except B4 (24, equal to ByteTrack).
- Winner as scored by the evaluator's rule (simplest preset within 10% of the fewest IDs): **A0**. A1 is nominally best (43) and is a one-parameter change; it is worth adopting only if the video shows no merged players.
- Winner video for the student to watch: `runs/task001_eval_video/A0/xbotgo_tracked.mp4`.

**Read of the results:**
- **CMC: no consistent signal.** B0 vs B1 is the only pair that isolates CMC. Total IDs: 50 vs 49, within noise. Short-lived IDs: 14 to 15 by span, but 20 to 18 by presence. The two measures disagree in direction, so CMC did not measurably help on this clip.
- **Label flicker is the dominant effect.** A1's largest gap in player-labelled frames is 97 frames (4.1 s). Its own lost-track budget is 72 frames (2.4 s), so the track stayed alive while the detector's label flipped to "referee". An ID can count as ">= 5 s" by span while much of that time was labelled referee. This is direct support for the evaluator's hypothesis that detector flicker, not camera motion, drives fragmentation. It is still an inference from one clip.
- B4 (activation and high-confidence thresholds lowered to 0.5) was much worse, at 58 IDs. This is consistent with low-confidence detections spawning junk tracks. Hypothesis, untested.
- A longer buffer (1 s to 3 s) helped both families by about 2 to 4 IDs. It was the largest consistent effect in the grid.

**Caveat on what "45 unique player IDs" counts:** the count includes every track labelled "player" in at least one frame. A track that is mostly labelled "referee" still counts. A track whose label flickers is also counted in the referee totals. 14 to 19 IDs per preset appear in both lists; for A0 that is 19 of 45 (42%). A large share of the 45 are not clean player tracks. Fewer IDs is also not automatically better: two players merged into one ID would lower the count and make tracking worse. Nothing in the metrics detects that.

Decisions and why:
- **Cache detections, then replay the tracker grid.** Detection costs about 0.30 s/frame on CPU (torch is CPU-only, no CUDA). A 7-preset grid that ran inference each time was about 25 min. With the cache it is one 3.6 min detection pass plus cheap replays. This also makes the comparison controlled: every tracker sees the same detections. Detection was checked to be deterministic run to run, which is what makes the replay valid.
- **Cache key uses weights content (SHA-256), not file size.** The reviewer found the key used file size, so a retrain could replay stale detections. Fixed; the stored hash was checked against the on-disk weights.
- **Baseline correction: 45 unique IDs, not 43.** `models/weights.pt` was replaced at 19:18:50, so the old figures came from the previous checkpoint. The current baseline is 45 IDs, 28.8% ball, 12 / 16 IDs under 2 s, and 24 / 21 IDs at 5 s or more (span / presence). The brief's original "13 under 2 s, 18 at 5 s or more" was span-measured on the older checkpoint. It is not comparable to either column above and is kept for history only.
- **Corrected the brief's `lost_track_buffer` units.** The library counts in 30-fps-equivalent units (`lost_track_buffer / 30.0`), not video frames. 3 s = 90, not 72. Brief updated.

Review of the fixed diff (reviewer, Opus, re-review complete):
- The BLOCKING cache-key issue is confirmed fixed (SHA-256 of file content, verified against the on-disk hash).
- The refactor is behaviour-preserving: A0 output is byte-identical to a pre-change run, and the `searchsorted` rewrite was checked against the old indexing on all 719 frames.
- `eval/track_consistency.py` had its own review pass. Its arithmetic was confirmed by independent re-implementation of all seven rows.
- Two SHOULD FIX items were found after the re-review and fixed: a replay hard-failed when the checkpoint was absent, and `--save-detections` help text was stale. Both were verified by the main session. A replay now works with `models/weights.pt` absent and still reproduces 45 IDs / 28.8%.
- Verdict: no remaining blocking issues.

Open decision (for the student and project manager):
- The code default is still `--tracker botsort` (B1, 49 IDs). The evaluator recommends against switching the default to BoT-SORT on this evidence, and the winner is A0. Changing the default is a code change for model-builder, not a docs change. The task brief's "BoT-SORT is the default" box is ticked as an implementation fact, with this caveat written next to it.

Blockers and risks:
- **No ground truth yet.** `data/ground_truth/` is empty. The eval measures tracker self-consistency only. No ID-switch rate, MOTA or IDF1 can be claimed.
- One 30 s clip. Differences of a few IDs are noise.
- The student has not yet read the diff, watched the A0 video for merged IDs, decided the default, or committed.

Next steps:
- Student: read the diff; watch `runs/task001_eval_video/A0/xbotgo_tracked.mp4` for merged IDs and for players labelled "referee"; decide the default preset (A0 vs A1 vs leaving BoT-SORT); commit.
- Hand-label about 10 s of IDs on the clip if IDF1 is wanted. That would allow an ID-level metric.
- Resolve ground truth (hand-charted stats for at least one game) before any accuracy claim.
- Test on more than one clip.

**Later on 2026-11-15: default tracker decided (resolves the open decision above).** The default is now ByteTrack preset A0 (task 002 Step 0). BoT-SORT and all presets stay selectable via `--tracker botsort` and `--preset`. The reason is evidence: BoT-SORT's tracker cost is about 17 to 19 times A0's per frame, with no consistent gain in IDs and fewer long-lived IDs by presence. CMC showed no signal (B0 vs B1).

A1 (43 IDs, the only count below 45) was rejected after a manual merge check by the student, on the annotated video and crops, of A1's reconnections after gaps over 1 s (`runs/task001_eval/A1/xbotgo_tracks.csv`). Two of three are wrong merges: ID 28 joins a blue-jersey player (frame 666) to a different white-shirt player (frame 709), and ID 20 joins a white-jersey player (frame 500) to a spectator by the door (frame 538). ID 0 (frames 159 to 192) is probably correct. So A1's 43 is partly an artifact of merging different people. This is the failure mode the "fewer IDs is not automatically better" caveat in `eval/reports/2026-11-15-botsort-vs-bytetrack.md` predicted, and it is why task 002 makes a merge check a required part of its eval.

## 2026-11-14
- Pivoted from ultimate frisbee to basketball (youth shot & stat tracking).
- Set up project structure and Claude Code agent team (architect on Fable; researcher, model-builder, evaluator on Sonnet; reviewer on Opus; data-engineer and report-writer on Haiku).
- Next: work through the "Before recording any footage" checklist in project-plan.md.

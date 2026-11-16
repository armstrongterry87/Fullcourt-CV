# Task 001: Switch player tracking from ByteTrack to BoT-SORT

**Status:** Steps 1-4 done (plan, build, review with one blocking issue found and fixed, eval, re-review with no remaining blocking issues). Not committed. Outcome: BoT-SORT did not beat ByteTrack on this clip; see `eval/reports/2026-11-15-botsort-vs-bytetrack.md`. Last updated 2026-11-15.
**Owner (main session):** project manager. Plan with `architect`, build with the Sonnet/Haiku agents.

## Why

On `data/raw/xbotgo.mp4` (30 s, 719 frames, 1280x720, 23.951 fps, panning XbotGo camera), the fine-tuned detector `models/weights.pt` with ByteTrack produced **43 unique player IDs** when about 10 are on court. That count came from an earlier checkpoint; the current baseline is below. On the earlier checkpoint, 13 of those IDs lived under 2 seconds (span-measured). Most breaks happen when the camera pans or players cross. BoT-SORT adds **camera motion compensation (CMC)**, which should cut ID switches on panning footage.

## Baseline to beat (fine-tuned model + ByteTrack, full clip)

| Metric | Value |
|---|---|
| Unique player IDs | 45 |
| Player IDs lasting under 2 s (presence / span) | 16 / 12 |
| Player IDs lasting 5 s or more (presence / span) | 21 / 24 |
| Ball found | 28.8% of frames |

Command that produced it:
```
python scripts/baseline_track.py --video data/raw/xbotgo.mp4 --weights models/weights.pt --tracker bytetrack
```

Notes on the baseline:
- `models/weights.pt` was replaced at 19:18:50. The table above is measured on the current checkpoint with ByteTrack (preset A0). The reference file is `runs/finetuned/xbotgo_tracks.csv` (written 19:39, before the BoT-SORT code existed). It is byte-identical to the new A0 output, so the code change did not alter ByteTrack behaviour; the checkpoint change did.
- Lifetimes are given two ways. **Presence** = distinct player-labelled frames / fps. **Span** = first-to-last player-labelled frame, which includes gaps. Presence is the headline. A0 has gaps in 36 of its 45 tracks, so span overstates "5 s or more" and understates "under 2 s". Source: `eval/reports/2026-11-15-botsort-vs-bytetrack.md`.
- History, kept for the record and not comparable to any column above: the brief's original table (43 unique IDs, 13 under 2 s, 18 at 5 s or more, 31.2% ball) was span-measured on the older checkpoint.

## Verified facts about the library (trackers package, already installed)

- `from trackers import BoTSORTTracker`
- Constructor (defaults shown): `BoTSORTTracker(lost_track_buffer=30, frame_rate=30.0, track_activation_threshold=0.7, minimum_consecutive_frames=2, minimum_iou_threshold_first_assoc=0.2, minimum_iou_threshold_second_assoc=0.5, minimum_iou_threshold_unconfirmed_assoc=0.3, high_conf_det_threshold=0.6, enable_cmc=True, cmc_method='sparseOptFlow', cmc_downscale=2, instant_first_frame_activation=True, reid_model=None, reid_config=None)`
- `update(detections, frame=None, timestamp=None, *, cmc_transform=None)`. **CMC only works if the BGR frame is passed**: `tracker.update(detections, frame=frame)`.
- `cmc_method` options: `'orb'`, `'sift'`, `'sparseOptFlow'`, `'ecc'`.
- `lost_track_buffer` is **not** a count of video frames. It is a 30-fps-equivalent count: the library computes `self.maximum_time_without_update = lost_track_buffer / 30.0` (`.venv/Lib/site-packages/trackers/core/botsort/tracker.py:162`), and the parameter docstring (lines 54-57) documents it as "frames at 30 FPS". The value does not depend on the clip's fps: the default 30 is 1.0 s at any fps. ByteTrack does the same (`.../bytetrack/tracker.py:51-55,105`).
- Correct conversion from seconds: `lost_track_buffer = round(lost_seconds * 30)`. 3 s = 90, not 72. Do not scale by the video's fps. Using the video's fps is the bug this note exists to prevent.

## Steps

1. **Plan (architect, Fable).** Read `scripts/baseline_track.py` and this brief. Decide:
   - where tracking code should live (`src/tracking/` vs. the script)
   - the tracker settings to try first, and a small grid of 3–4 variants (e.g. `lost_track_buffer` 30 vs 90 (1 s vs 3 s), `cmc_method` sparseOptFlow vs orb, `track_activation_threshold`)
   - how to keep ByteTrack available as a `--tracker` option for comparison

   Output a short plan with each step assigned to an agent. Keep it brief; Fable is the expensive model.
2. **Build (model-builder, Sonnet).** Implement the plan. Required:
   - `--tracker {bytetrack,botsort}` option, default `botsort`
   - pass `frame=frame` to BoT-SORT
   - tracker settings as CLI options or one config dict, not scattered constants
   - no change to detection, class mapping or output format
3. **Review (reviewer, Opus).** Review the diff before anything is committed. Watch especially for the frame not being passed to `update`, buffer units (frames, not seconds), and referees being dropped from tracking.
4. **Measure (evaluator, Sonnet).** Run ByteTrack and each BoT-SORT variant on `data/raw/xbotgo.mp4` with `models/weights.pt`. Report the four metrics above for each in `eval/reports/<date>-botsort-vs-bytetrack.md`, with a recommendation. Also note processing speed (frames/second), since CMC adds cost.
5. **Log (report-writer, Haiku).** Add entries to `docs/progress-log.md` and `docs/ai-use-log.md`.

## Done when

- [ ] ~~BoT-SORT is the default and ByteTrack is still selectable.~~ **Superseded by task 002 Step 0.** This was the task-001 requirement. It was met in the code at the time (`scripts/baseline_track.py` `--tracker` defaulted to `botsort`). It is no longer the default. The student decided, on the evidence in `eval/reports/2026-11-15-botsort-vs-bytetrack.md`, to make ByteTrack preset A0 the default. A0 is tied with A1 for the most long-lived IDs by presence (21, against 15 to 19 for every BoT-SORT preset). It has 45 unique IDs, tied with B3 for the fewest outside A1, and its tracker time is about 17 to 19 times lower than BoT-SORT's. A1 (43 IDs) was rejected after a manual merge check (see progress log, 2026-11-15). BoT-SORT and all presets remain selectable via `--tracker botsort` and `--preset`.
- [x] Reviewer found no blocking issues. The first review found one BLOCKING issue (detection cache keyed on weights file size, not content) and a self-overwriting output path; both were fixed. The re-review of the fixed diff (Opus) found no remaining blocking issues. Covered: the SHA-256 cache key is confirmed against the on-disk hash; the refactor is behaviour-preserving (A0 output byte-identical to a pre-change run; `searchsorted` rewrite matches the old indexing on all 719 frames); `eval/track_consistency.py` arithmetic confirmed by independent re-implementation of all seven rows. Two SHOULD FIX items were found and fixed afterwards (replay hard-failed without the checkpoint; stale `--save-detections` help text). Both were verified by the main session: replay works with `models/weights.pt` absent and still gives 45 IDs / 28.8%.
- [x] Eval report exists with the comparison table: `eval/reports/2026-11-15-botsort-vs-bytetrack.md`.
- [ ] Unique player IDs are lower than 45 on the same clip (current baseline; not 43), or the report explains why not. Not ticked as a BoT-SORT result: no BoT-SORT preset is below 45 (best B3 = 45, default B1 = 49). The only preset below 45 is A1 (ByteTrack, 3 s buffer, 43), a ByteTrack-only change. The report explains the BoT-SORT result. BoT-SORT also has fewer long-lived IDs by presence for every preset (15 to 19 vs 21 for ByteTrack). A1 is an option for the student to decide on after watching the video for merged IDs.
- [ ] Logs updated; changes committed by the student after reading the diff. Logs are updated (2026-11-15). Not committed. Student still to read the diff, watch the A0 video (`runs/task001_eval_video/A0/xbotgo_tracked.mp4`) for merged IDs, and commit.

## Out of scope (separate tasks)

- Court-mask filter for spectators (people by the doors are being labeled "referee")
- Ball gap filling and a separate ball threshold
- Re-ID appearance models

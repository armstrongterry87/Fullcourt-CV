# Task 002: Honest player counts: one role per track, separate thresholds, court filter

**Status:** Ready to start (after Step 0)
**Owner (main session):** project manager. Plan with `architect`, build with the Sonnet/Haiku agents.

## Step 0: close out task 001 first

1. Set the default tracker to **ByteTrack preset A0** (`DEFAULT_PRESET` / the `--tracker` default). Keep BoT-SORT and A1 selectable.
2. Reason, for the record: A1 (3 s buffer) reached 43 IDs partly by **merging different people**. A crop check of A1's three reconnections after gaps over 1 s (`runs/task001_eval/A1/xbotgo_tracks.csv`) found:
   - ID 28: blue-jersey player (frame 666) → different white-shirt kid (frame 709). Wrong merge.
   - ID 20: white-jersey player (frame 500) → spectator by the door (frame 538). Wrong merge.
   - ID 0: frames 159 → 192, similar white/red jersey. Probably correct.
3. Add this finding to `eval/reports/2026-11-15-botsort-vs-bytetrack.md` under a "Merge check" heading.
4. Stop and let the student review the diff and commit task 001 before starting Step 1.

## Why

The eval for task 001 showed tracker choice is not the bottleneck. The real causes:
- **Per-frame labels flicker.** 14–19 tracks per run switch between "player" and "referee", so they count twice and the player count is inflated.
- **Low-confidence boxes flicker on and off** at the 0.40 threshold and start short-lived tracks.
- **Spectators standing near the court** get detected (now often labeled "referee") and create junk tracks.

## Baseline to beat (your machine, A0, `models/weights.pt`, full `data/raw/xbotgo.mp4`)

| Metric | Value |
|---|---|
| Unique player IDs | 45 |
| Player IDs under 2 s | 12 |
| Player IDs 5 s or more | 24 |
| Tracks appearing as both player and referee | 14–19 |
| Ball found | 28.8% of frames |

## Steps

1. **Plan (architect, Fable).** Short plan covering the three changes below, the experiment grid, and agent assignments. Note that detections can be cached once at the **lowest** threshold in the grid and filtered by confidence for higher ones, so the grid costs one detection pass.

2. **Build (model-builder, Sonnet).**
   - **A. One role per track.** Processing is offline, so assign each track a single role after the run: a confidence-weighted majority vote of its per-frame labels (player vs referee). Write both columns to the CSV (`label_frame`, `role_track`). Count players by `role_track`. Draw the final role in the video.
   - **B. Separate thresholds.** Add `--person-threshold` (people that may start or extend tracks) and `--ball-threshold`. The ball is small and needs a lower bar than people.
   - **C. Court filter (code + calibration tool; full test waits for fixed-camera footage).**
     - `scripts/calibrate_court.py`: opens one frame, the student clicks the court corners, and it saves a polygon to `data/courts/<video_stem>.json`.
     - `--court <json>`: drop any person whose bottom-center point (the feet) is outside the polygon, with a small margin in pixels so players on the line are kept.
     - **Important:** `xbotgo.mp4` pans, so a fixed polygon only fits part of the clip. Test the filter on a short stretch where the camera is roughly still, and say so in the report. Do not tune anything else around this clip's panning.
   - Keep the existing output format, CLI and detection cache behaviour. Additions only.

3. **Review (reviewer, Opus).** Watch especially for:
   - the vote using the wrong labels after tracking (the track/label alignment)
   - the threshold for people accidentally applied to the ball, or the reverse
   - the court test using the bottom of the box (feet), not the center
   - the cache key covering whatever the new thresholds depend on

4. **Measure (evaluator, Sonnet).** Run on `data/raw/xbotgo.mp4` with `models/weights.pt`, tracker A0. Grid:
   - person threshold: 0.4, 0.5, 0.6
   - ball threshold: 0.25, 0.3, 0.4
   - role vote: off vs on

   For each, report:
   - the baseline metrics above
   - **ball false alarms**: frames with 2 or more ball candidates above the threshold, before keeping the top one
   - **merge check**: list every reconnection after a gap over 1 s with the jump distance in box heights. Save before/after crops for the largest jumps to `runs/task002_eval/merge_crops/` and look at them. Fewer IDs is only better if they are not merges.

   Write `eval/reports/<date>-roles-thresholds.md` with a recommendation. Court filter: one short qualitative section only.

5. **Log (report-writer, Haiku).** Update `docs/progress-log.md` and `docs/ai-use-log.md`.

## Done when

- [ ] Step 0 done and task 001 committed by the student
- [ ] Each track has exactly one role; no track is counted as both player and referee
- [ ] Person and ball thresholds are separate options, with recommended defaults chosen from the grid
- [ ] Court calibration tool and `--court` filter exist, are reviewed, and have a short honest test note
- [ ] Eval report includes the merge check
- [ ] Reviewer found no blocking issues; logs updated; student commits after reading the diff

## Out of scope

- Ball gap filling and interpolation (later task)
- Retraining the detector (waits for the student's own fixed-camera footage)
- Ground-truth labeling (task 003)

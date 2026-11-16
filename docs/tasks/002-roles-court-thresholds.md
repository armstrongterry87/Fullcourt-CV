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

- [x] Step 0 done and task 001 committed. HEAD is `abbfbe4`, parent `a9144a9`. Both carry the same commit message but are different commits: `abbfbe4` holds Step 0 (ByteTrack A0 default, the merge check, the superseded box).
- [x] Each track has exactly one role; no track is counted as both player and referee. `role_track` is constant per `track_id` (verified: 0 violations across 47 tracks).
      **The premise at line 19 of this brief was wrong.** Flickering tracks did not count twice in the player total — a both-label track contributes one id. What was double-counted was the *breakdown*: 45 players + 21 referees = 66 for 47 real tracks. The vote fixes that (43 + 4 = 47) and moves the player count by only 2. **43 player tracks for 9 real players is still ~4.8x fragmentation, untouched by this change.**
- [x] Person and ball thresholds are separate options, with recommended defaults chosen from the grid. **Recommended: person 0.4, ball 0.4** (both are the current defaults, inherited from `--threshold`).
      Person 0.4 is the chosen default and the grid supports it: ID switches rise to 18 at 0.5 and 16 at 0.6 against 11 at 0.4, and at 0.6 two key players drop to partially tracked.
      Ball stays 0.4 because 0.25 is **unproven**: it raises detection from 28.8% to 48.0%, but multi-candidate frames go 3 → 31 and large centre-jumps between consecutive detections rise from 4.7% to 15.7%, so some of the gain may be spurious. A few dozen hand-labelled ball positions would settle it. That is the most valuable next measurement for shot detection.
- [~] Court calibration tool and `--court` filter: **code complete, untested on real geometry.** `scripts/calibrate_court.py` exists and round-trips through `save_court`/`load_court`, and the filter demonstrably drops rows (37 of 1,565 person detections in frames 359-503). But the polygon used was a guessed trapezoid never checked against the court lines, so the *tool* is demonstrated and the *calibration* is not. Valid frames on this clip are **359-503** (15-21 s, camera measured locked off at 0.00-0.01 px/frame; it pans at 8-11 s). Real calibration needs a human clicking frame 430.
- [x] Eval report includes the merge check. Note the brief's required ">1 s gap" threshold proved **nearly vacuous**: ByteTrack's lost-track buffer is ~1.25 s and the largest observed reconnection gap was 0.96 s at person 0.4, giving 0 reconnections. Lowered to 0.5 s it finds 12 / 14 / 19 at person 0.4 / 0.5 / 0.6. At 0.4, three look like merges — tracks 6, 15 and 28 — and tracks 6 and 15 are long mixed-person tracks the vote labels "player" for their whole length, so `unique_player_ids_by_role` is not trustworthy at the margin.
- [ ] Reviewer found no blocking issues; logs updated; commits after the diff is read.
      **The final review did not complete.** It cleared `scripts/calibrate_court.py` as privacy-clean, then stopped on a rate limit before reaching the `eval/score_tracking.py` changes and `eval/xbotgo_10s_frame_map.json`. Those two remain unreviewed. See `docs/progress-log.md` for what was checked directly instead.

## Out of scope

- Ball gap filling and interpolation (later task)
- Retraining the detector (waits for the student's own fixed-camera footage)
- Ground-truth labeling (task 003)

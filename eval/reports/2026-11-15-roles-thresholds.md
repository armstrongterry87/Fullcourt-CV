# Task 002: role vote, person/ball thresholds, court filter (eval)

Date: 2026-11-15. One clip (xbotgo, 30 s, 719 frames, 23.951 fps), no held-out games. Differences of a few IDs or a few IDF1 points are noise.

## What was evaluated
- Pipeline: task 002 build (uncommitted at time of run), 5 replays of `scripts/baseline_track.py` off `runs/cache/xbotgo__wfc2f7c4e__small__thr0.25.npz`, preset A0 (ByteTrack), `--threshold 0.40` (rim) passed explicitly, `--no-video`. One `--out-dir` per cell: `runs/task002_eval/p<P>_b<B>/A0/`.
- Gate re-check: the p0.4/b0.4 cell is byte-identical (all columns, incl. the vote columns) to `runs/task002/gate/A0/xbotgo_tracks.csv`, and its first nine columns match the committed task 001 A0 CSV (md5 equal after stripping CR).
- Key: `data/ground_truth/xbotgo_10s_mot.txt` (40 frames, 9 players), scored via `eval/score_tracking.py`. Read the A0 baseline report (`2026-11-15-idf1-a0-baseline.md`) for the key's limits.

## Scorer fixes (all in `eval/`, no changes to `src/` or ground truth)
1. **Labels.** Default label is now the first ancestor directory that is not a preset name (A0, A1, B0-B4); `--label` is repeatable, one per CSV, and a count mismatch is an error. Verified: the 5 grid CSVs now come out as `p0.4_b0.4`, `p0.5_b0.4`, etc. Before this fix all would have been `A0`.
2. **`--role-column {class,role_track}`.** Pre-vote and post-vote CSVs now score differently (below). It errors if the column is missing, so the old nine-column CSVs fail loudly rather than silently scoring `class`.
3. **Frame map.** Numbers-only `eval/xbotgo_10s_frame_map.json` (mot_frame, original_frame, time_s, fps, stride; 40 rows, no paths). Scorer defaults to it, falls back to the gitignored sidecar; `--frame-map` overrides. Nothing was written to `data/ground_truth/`. The documented A0 command now works on a clean checkout, and reproduces IDF1 0.619 / 11 switches.
4. `max_iou` renamed `max_dist` (naming only; arithmetic unchanged).

## Independence assertion (grid reduction)
Asserted from the CSVs, not assumed:
- Ball-varied runs (b0.25, b0.30 vs b0.40 at p0.4): player/referee rows byte-identical, and `unique_player_ids`, `unique_player_ids_by_role`, `tracks_with_both_labels` identical. **Holds.**
- Person-varied runs (p0.5, p0.6 vs p0.4 at b0.4): ball and rim rows identical, `ball_seen_pct` (28.8) and `ball_multi_candidate_frames` (3) identical. **Holds.**
- IDF1/switch scores are identical across the three ball-varied cells for both role columns. The 5-cell reduction is valid.

## Metrics per cell
IDs by span/presence follow the task 001 definitions (span = last-first+1 frames; presence = distinct player-labelled frames; fps 23.951). "IDs" = unique `track_id >= 0` with class player.

| Cell (person/ball) | Unique player IDs | IDs by role | Tracks with both labels | IDs <2 s (span/pres) | IDs >=5 s (span/pres) | Ball multi-cand. frames | Ball % |
|---|---|---|---|---|---|---|---|
| 0.4 / 0.40 | 45 | 43 | 19 | 12 / 16 | 24 / 21 | 3 | 28.8 |
| 0.5 / 0.40 | 47 | 44 | 15 | 15 / 16 | 20 / 18 | 3 | 28.8 |
| 0.6 / 0.40 | 50 | 49 | 11 | 15 / 18 | 22 / 19 | 3 | 28.8 |
| 0.4 / 0.30 | 45 | 43 | 19 | 12 / 16 | 24 / 21 | 17 | 39.1 |
| 0.4 / 0.25 | 45 | 43 | 19 | 12 / 16 | 24 / 21 | **31** | **48.0** |

IDF1 block (40 frames, 9 key players). Ball cells share the p0.4 numbers.

| Cell | Role column | Mode | IDF1 | ID switches | Precision | MT / PT / ML | Tracker IDs in window (key: 9) |
|---|---|---|---|---|---|---|---|
| p0.4 (any ball) | class | players | 0.619 | 11 | 0.811 | 9/0/0 | 17 |
| p0.4 (any ball) | role_track | players | 0.602 | 11 | 0.777 | 9/0/0 | 17 |
| p0.5 | class | players | 0.603 | 18 | 0.837 | 9/0/0 | 18 |
| p0.5 | role_track | players | 0.589 | 18 | 0.801 | 9/0/0 | 18 |
| p0.6 | class | players | 0.633 | 16 | 0.868 | 7/2/0 | 19 |
| p0.6 | role_track | players | 0.622 | 16 | 0.840 | 7/2/0 | 19 |

Note on p0.6: MT 7, PT 2, ML 0 (two players only partially tracked, recall 0.907). Higher threshold lost real players. Full table including players+referees mode and IDP/IDR: `runs/task002_eval/score_class.json`, `score_role_track.json` (gitignored).

Reading the IDF1 block:
- **The vote has no effect on ID switches, MT/ML or tracker IDs in this window.** Under `role_track`, "players" mode equals "players+referees" mode in every cell: every tracked box in the 40 scored frames belongs to a track voted player. The only change is that the 18 referee-labelled boxes now count as players, which lowers IDF1 by about 0.015 and precision by about 0.03. On this key the vote neither helps nor hurts identity quality. It is a relabelling.
- **Switches rise from 11 (p0.4) to 18 (p0.5) and 16 (p0.6).** IDF1 moves 0.60 to 0.63 but the highest-threshold cell loses two players (PT 2, recall 0.907). Per the precision caveat below, I do not read the IDF1 gain at p0.6 as an improvement. No threshold beats p0.4 on identity quality; they are all within noise of each other on a 9-player key, but the direction of the switch count is the opposite of the "higher threshold is better" story.

## The role vote: what it does and does not fix
- **The task brief's premise is wrong.** `docs/tasks/002-roles-court-thresholds.md:19` says flickering tracks "count twice and the player count is inflated." They do not: a track labelled both ways contributes exactly one id to the player set. What was inflated is the *breakdown* (45 players + 21 referees = 66 for 47 actual tracks, double-counting 19). The vote gives 43 + 4 = 47, and moves the player count by only 2. That is the correct result, not a bug. The reviewer independently recomputed the vote on all 47 tracks with 0 mismatches. I also confirmed here that after the vote no track has more than one `role_track` value.
- **43 player tracks for 9 real players is still about 4.8x fragmentation, untouched by this change.** The vote is a labelling fix, not a tracking fix.
- **The vote depends on tracks being pure, and task 001 showed they are not.** A merged track that is a player for 200 frames and a spectator for 100 gets `role_track = "player"` for all 300 frames. The vote launders a tracking error into a confident-looking label. The merge check below must be read before trusting `unique_player_ids_by_role`, and the merge check says the tracks are not pure.

## Merge check
Required threshold was reconnections after a gap over 1 s. **That condition is nearly vacuous**: the ByteTrack lost-track buffer is 30 frames at 23.951 fps (about 1.25 s) and the largest observed reconnection gap is 0.96 s (p0.4), 1.04 s (p0.5), 1.00 s (p0.6). Reconnections over 1 s: 0 at p0.4, 1 at p0.5, 0 or 1 at p0.6 (the 1.00 s one is borderline at frame granularity). So I lowered the gap to 0.5 s and checked all of those. Jump distance = centre displacement between the last box before and the first box after the gap, divided by mean box height. Counts (all classes of person track):

| Person thr | Reconnections >0.5 s | Jump >1 box height | >2 | Median jump (box heights) |
|---|---|---|---|---|
| 0.4 | 12 | 4 | 1 | 0.68 |
| 0.5 | 14 | 2 | 0 | 0.57 |
| 0.6 | 19 | 2 | 0 | 0.24 |

**Caveat on the metric:** the camera pans, so a pixel jump mixes real movement and camera motion; box-height jump is a weak merge indicator. I therefore looked at crops (`runs/task002_eval/merge_crops/`, left two panels = tight crops before/after the gap, right two = full frames, boxes in green; `git check-ignore` confirms the folder is ignored via `.gitignore:16 runs/`). `reconnections.csv` there lists every one. My reading of the low-resolution crops (a human should confirm):
- **p0.4 id 6, frames 191 to 209 (jump 2.44):** before = a dark-blue player on court; after = a different, smaller figure near the bench. Looks like a **merge**. Track has 574 frames.
- **p0.4 id 15, frames 212 to 231 (1.78) and 192 to 212 (1.74):** before = blue/white player; after = a person in red/pink near the far bench. Looks like a **merge** (and the track flips to class referee after the gap).
- **p0.4 id 28, frames 465 to 488 (1.50):** before = player in blue in a group; after = a person in dark clothing standing apart. Likely a **merge** (class flips player to referee).
- Smaller jumps (<1 box height) and the p0.5/p0.6 top cases (1.1 to 1.3) I did not inspect individually beyond the saved crops; I have not viewed those, so I make no claim about them.

Fewer IDs at a higher threshold is therefore not automatically better. p0.5/p0.6 have fewer large jumps (I only viewed p0.4 crops) but also more switches and more IDs, so the trade is not a clean win either way. Because id 6 and id 15 are long, mixed-person tracks that the vote labels "player" in full, `unique_player_ids_by_role` is not trustworthy at the margin.

## Ball threshold (potentially the most valuable result, weigh with care)
- Ball detection rate rises from **28.8% (0.40) to 39.1% (0.30) to 48.0% (0.25)** of 719 frames. This is a detection rate, not accuracy: there are no ball labels, so I cannot say how many of the added detections are real balls.
- **Correction to the brief's framing:** "only 3 multi-candidate frames" is the figure at 0.40. At 0.25 it is **31**, and at 0.30 it is 17. Ambiguous frames grow about tenfold, so the 0.25 floor is not clean.
- Indirect evidence of added false positives: median detection confidence of the selected ball falls from 0.64 (0.40 floor) to 0.54 (0.30) to 0.47 (0.25) (by definition, the new detections are low confidence), and consecutive-frame pairs with a centre jump over 100 px rise from 7 of 148 (4.7%) at 0.40 to 17 of 199 (8.5%) at 0.30 to 40 of 255 (15.7%) at 0.25. Camera panning could explain some of it, so this is suggestive, not proof.
- Verdict on this point: the rate increase is real but the quality of the extra 19% is unmeasured. Before choosing 0.25, someone needs to label ball positions on a few dozen frames, or watch the 31 multi-candidate frames. This is the single most useful next measurement for shot detection.

## Court filter
The court filter shipped code-only. `scripts/calibrate_court.py` did not exist at review time and is being written now, so the filter has never seen real geometry: the builder's smoke test used a made-up rectangle and dropped zero rows. I did not run it in any grid cell and none of the numbers above depend on it. It is untested on real data and I make no claim about its effect. Its only valid frames on this clip are **359 to 503** (15 to 21 s), where the camera was measured as locked off; outside that range the camera pans and a fixed court polygon is wrong.

## Caveats
- **Precision is biased and not neutral across thresholds.** The key was built from tracker boxes with no new boxes drawn, so unlabelled real players count as false positives and higher thresholds flatter themselves. I weight ID switches and MT/ML over precision. Recall is circular (every key box is a tracker box) and is not quoted as a result.
- One 30 s clip (10 s keyed), 9 players, no held-out game. Every difference above is within what I would expect from noise except possibly the ball-rate change.

## Comparison to previous report
Baseline (`2026-11-15-idf1-a0-baseline.md`): A0 IDF1 0.619, 11 switches, 17 IDs in window. The p0.4/b0.4 cell reproduces it exactly. Moving the person threshold to 0.5 or 0.6 gives more switches (18, 16), not fewer. The role vote changes IDF1 by -0.017 (class to role_track), which is the same as the previously measured cost of including referees.

## Failure cases for a human to watch
- Original frames 191 to 209, track 6 (about 8.0 to 8.7 s): suspected merge, 2.44 box heights.
- Frames 192 to 231, track 15 (8.0 to 9.6 s): two suspected merges.
- Frames 465 to 488, track 28 (19.4 to 20.4 s): suspected merge, flips to referee.
- Ball: the 31 multi-candidate frames at ball 0.25 (list them with `ball_multi_candidate_frames` in the p0.4_b0.25 CSV; I have not itemised them).
- p0.6: the two partially tracked key players (not identified here).

## Verdict
The independence assumption holds, so the 5-cell grid is valid, and the scorer is now reproducible from a clean checkout with correct per-CSV labels and a role-column switch. The role vote is a labelling fix that changes no identity metric on this key, the tracks it votes over are visibly impure (at least three likely merges at p0.4), and no person threshold improves identity quality. The ball floor at 0.25 raises detection to 48.0% but also raises multi-candidate frames from 3 to 31 and implausible jumps from 4.7% to 15.7%, so its accuracy is unproven until ball positions are labelled.

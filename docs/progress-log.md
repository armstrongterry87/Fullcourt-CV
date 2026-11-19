# Progress Log

Newest entries on top.

## 2026-11-18
*(unattended run, Task 005 Part A. Nothing committed. Morning summary and all questions: `docs/tasks/005-questions.md`. Physical checks: `eval/reports/2026-11-18-motion-compensation.md`.)*

**Task 005 Part A: camera motion compensation built and checked on one 30 s clip. It works at the frame centre and misses four design targets. Part B has not started.**

What got built:
- Design: `docs/design/005-motion-compensation.md`, written by the architect agent (Fable). An Opus stand-in design is kept beside it as `005-motion-compensation-opus-standin.md`. See the A1 notes in the questions file.
- Build: `src/camera/motion.py` (per-frame homographies, keyframe graph, views, quality, cache), `scripts/stabilize.py`, a mosaic mode in `scripts/calibrate_court.py`, `--motion` in `scripts/detect_shots.py` and `src/shots/detect.py`, `src/court/geometry.py`, and `tests/test_motion.py`. `ALGO_VERSION` is 3.
- Eval: `eval/motion_checks.py`, `eval/motion_frames.py`, `eval/motion_adjacent.py`, and the A3 report.
- Tests: 128 passed (98 existing, 30 new). These counts come from run notes given to the report-writer. The report-writer has no shell and did not re-run them.
- Review: the reviewer (Opus) found no blocking issues and 3 should-fix items. All three were fixed: a cap on chained rescue keyframes, recovery in pass 2, and a check that the motion file's fps and frame count match the video. A2-Q10 also records a pass-1 stale-pose fix and a jump-check skip after long gaps.

Results (A3, one clip: `data/raw/xbotgo.mp4`, 719 frames, 23.951 fps; all numbers from the A3 report):

| Check | Raw | Stabilised | Target | Result |
|---|---|---|---|---|
| Frame-centre pose vs direct fit | n/a | median 0.52 px, p95 1.42 px | n/a | sound at centre |
| Independent drift, max corner (190 pairs) | n/a | median 3.78 px, p95 10.5 px | median <= 2, p95 <= 5 | miss |
| Independent drift, mean corner | n/a | median 2.07 px, p95 5.68 px | same | narrow miss |
| Coverage | n/a | 99.86% at >= 0.6; 30.0% at 0.9; longest unreliable run 1 frame | >= 95%; <= 1 s | pass |
| Left rim (view 0), p95 spread | 23.4 px | 7.8 view px (about 3 raw-equivalent px) | <= 25% of raw; <= 0.5 W | 0.33 of raw: miss on ratio, pass on absolute |
| Right rim (view 1), p95 spread | 1.9 px | 2.0 px | same | no power (raw already tight) |
| Static person, track 34 (n = 209), p95 | 110.6 px | 9.7 view px | <= 15 px | pass (one person) |
| Bench seat A, full gate (n = 114), p95 | 41.1 px | 32.7 px | <= 15 px | fail |
| On-court controls (tracks 6, 19) | n/a | do not look static | must not look static | pass |
| Closure error | n/a | not measured (graph is a tree, no loops) | <= 5 px | cannot assess |

Structure: 4 keyframes (frames 0, 207, 237, 274), 2 views, anchor keyframe 1, max 2 hops. Keyframe 3 hangs on a 20-frame chained edge, so 430 frames (60% of the clip) are capped at 0.6. The left rim is stretched 2.65x in view 0.

Decisions made, and why:
- **Frames are matched to keyframes, not to each other.** Chaining frame to frame would add error on every step. Re-anchoring reuses old keyframes when the camera returns to a view.
- **Two views, not one mosaic.** A flat canvas stretches content without limit as the pan grows. Each view is bounded at 1.5x stretch around its keyframe (an untuned value). A narrow pan still gives one view, as the brief expects. This clip gives two.
- **Rescue mode in pass 1 (not in the design).** The fast pan blurred every 5 Hz sample, so none matched the first keyframe well enough to start a new one. The builder chains frames until a sharp frame can start a keyframe (A2-Q1). The cost is chained edges with accumulated error. This is the biggest weakness the A3 checks found.
- **Every detection is masked, rims included** (design Q8). This avoids mapping class ids to names. It costs only the small rim box.
- **Static-people picks.** Picks were made by eye before the stabilised numbers were read. Two exceptions, both disclosed in the report: track 35 was a bad pick, and the bench-A subset (frames 372-590) was chosen after the full result was seen. The full-gate number is the one reported as the verdict.
- **Stand-in probe numbers kept as leads only** (pan about 88 degrees, focal about 920 px). A3 agrees with the pan (frame centres sweep about 85 degrees). The rim bearings in the A3 report depend on an assumed focal length of 800 to 1050 px, so they are not measured.

Blockers and risks:
- 30 s, one clip. It says nothing about drift over a full game.
- The chained edge to keyframe 3 caps 60% of the clip at 0.6 and adds corner error.
- Quality 0.6 is permissive. Single-frame corner jumps of 19 to 47 px (frames 221-250 and 615-616) still score 0.6.
- Rim detections exist only while the camera is still, so rim stability cannot show whether stabilisation works during a pan.
- Interactive mosaic clicking in a real window has not been tested (A2-Q3).
- Undecided: rims clicked on the mosaic (design) or on raw keyframes (stand-in proposal).
- Footage of minors: the mosaic (`data/frames/xbotgo_mosaic.jpg`) and debug video (`runs/`) show bystanders. Design 12.2 says both folders are git-ignored. `*.jpg` is not ignored globally, so the scripts must refuse other output folders. I did not re-check `.gitignore`.

Next steps:
- Owner: decide the four items in the morning summary (doc handling, whether to fix the A3 findings before Part B, a real-window click test of the mosaic mode, and mosaic vs raw rim clicks).
- Owner: watch `runs/xbotgo_motion_debug.mp4` yourself. The agent's visual check used extracted frames and montages.
- Owner: review the diff and commit by hand. Nothing is committed.
- Part B waits for `data/raw/game1.mp4` and hand-labelled shots.

How the system works (plain English, for the student to defend):
1. The camera rotates on a tripod, so each frame is the same gym seen from a slightly different direction. Two images from a camera that only rotates (fixed zoom) are related by a homography: a 3x3 matrix with 8 free numbers that maps every pixel of one image to the other. It is exact for pure rotation. It breaks if the camera zooms or moves position (parallax).
2. For each frame we need one matrix that maps the frame into a single fixed picture, the reference. We estimate it from background features only. SIFT finds keypoints and descriptors. They are matched to a keyframe's keypoints with a ratio test (Lowe 0.75) and a mutual check. RANSAC then fits one homography and keeps the inliers. The fit uses a fixed seed, so reruns match.
3. Players and the ball are masked out before features are found. The masks come from the same frame's detections, dilated, so moving people do not vote for their own motion.
4. We do not chain frame to frame, because errors add up. Keyframes (sampled about 5 times a second) are matched to each other, and each frame is matched to a keyframe. When the camera returns to a view, the old keyframes are reused. This closes the loop without any global optimisation.
5. Chaining is the fallback for frames too blurred to match a keyframe, usually during a fast pan. A chained frame carries the error of every link, so it is capped at quality 0.6.
6. The reference is the keyframe with the fewest hops to all the others. Composition: the rightmost matrix is applied first, so `H_frame_to_ref = H_key_to_ref @ H_frame_to_key`.
7. A flat canvas stretches content as the pan angle grows (1.5x at 35 degrees, 2.4x at 50). The reference is therefore split into views. Each view is a keyframe's plane extended to a canvas, with stretch bounded at 1.5x. Each rim is scored in its own view.
8. Quality per frame: 0.9 (direct match, enough inliers, low error, inliers spread across the frame), 0.6 (everything else, including chained frames), or 0 (unreliable, with the pose set to NaN so no code can use it by accident). Shots that depend on unreliable frames become `unknown`. They are never guessed.
9. Scoring: the shot engine is unchanged. Feet and ball centres are transformed to view pixels. Box corners are not transformed, because a box's corners do not map to a box. The rebuilt box is a centre-and-width model. The court is clicked once on the mosaic and fitted to feet in feet.
10. What the checks can and cannot show: nothing measures camera pose directly. The A3 checks are consistency checks (a rim should not move, seated people should not move, consecutive frames should agree with a direct fit). They do not measure shot accuracy. The drift check uses the same SIFT and RANSAC code as the build, so a systematic error common to both would not show. The scoreboard landmark is the only independent check.

## 2026-11-17
*(unattended overnight run, began this morning and continued past midnight. Task 004. Nothing committed. Morning summary and all questions: `docs/tasks/004-questions.md`.)*

**Task 004: shot scoring engine, built on synthetic tests only. No accuracy claim exists.**

What got built:
- Design doc `docs/design/004-shot-scoring.md`. Every court dimension carries a V/S/X/D/U status, and no value was filled in from memory.
- `src/court/geometry.py`, `src/shots/trajectory.py`, `src/shots/detect.py`, `src/shots/config.py`, `scripts/detect_shots.py`.
- `scripts/label_shots.py`, a hand-labelling tool (open, step, label, save). Its arrow keys were tested only through mocks.
- `eval/score_shots.py`, which scores predicted shots against a label CSV within ±1 s.
- 98 tests reported passing (`.venv/Scripts/pytest.exe tests/ -q`), up from 84 after the step 5 fixes. The docs agent did not re-run them. No new dependencies, no scipy (Q1), and nothing writes an image.

Results:
- **Nothing has been validated on video (Q3).** All acceptance is synthetic. The XbotGo clip is not a test case for any scoring rule.
- `scripts/detect_shots.py` on the XbotGo clip emits zero attempts, with `n_flights_lost=1`. This is the correct outcome. The clip ends with the ball still airborne, and the engine declined to guess.

Court dimensions:
- FIBA moved to V from the FIBA 2024 Official Basketball Rules (Art. 2.1, 2.4.3, 2.4.5).
- NFHS moved to S. The source is NFHS Rule 1 quoted verbatim, with section numbers, in a secondary gymnasium-equipment design guide.
- The 63 in (5.25 ft) basket offset is confirmed two independent ways for NFHS: the equipment chain (48 + 6 + 9 in) and the three-point straight segments meeting the arc at the basket's offset. This sets the origin of every 2/3 call.
- NCAA and NBA stay `verified = False` (Q12, Q20, Q22). NCAA's basket offset is an assumption that it matches NFHS.
- Method: the research step's failed fetches had saved their PDFs anyway. The main session extracted text from them with the local `pdftotext`. The research agent has no shell and could not do this itself.

Step 5 review (reviewer, Opus): no blocking issues. Three fixes closed paths to confidently wrong output:
- Miss-after-loss is capped at 0.6 and needs real detections on both sides.
- Rim labels are checked against the homography. Before this, mirrored court coordinates could publish at 0.9 confidence.
- `provisional_geometry` caps `points_conf` at 0.6 for NCAA and NBA. Before this, their unverified constants reached the CSV at 0.9.
- The reviewer also reported 6 SHOULD FIX and 6 NIT items. Their status is not recorded in the run's notes.

Decisions made without a person (all in `docs/tasks/004-questions.md`):
- **Q5: `points` is the shot value** (1, 2 or 3) on every row, misses included. Points scored is derived. Three deliverables depend on this, so it needs confirmation.
- **Q15: shooter feet at take-off, not release.** A deliberate deviation from the brief. Airborne feet map too far from the camera.
- **Q13: a label marks the outcome frame**, not the release frame.
- **Q27 (scorer): drop-and-count with a loud banner and exit code 2**, rather than a hard fail. A clean file prints one quiet line.
- The second block numbered Q27 (the step 5 fixes) is renumbered Q28.

Licence finding: `roboflow/sports` is MIT, but its soccer example imports Ultralytics (AGPL-3.0). Library modules were not checked. Nothing was copied. `aarnavshah12/shot-tracker` has no licence file and must not be copied from.

Housekeeping: a stray synthetic label file, `data/labels/selftest_xbotgo_shots.csv`, was deleted (Q11). It held test rows, not labels.

**Not done, and relevant to the commit:**
- **Step 6 (architect post-build review) is not confirmed.** Design section 10 is still the placeholder "To be filled in at step 6." Treat the task as not fully closed.
- The keyboard check of the labelling tool's arrow keys (Q10) has not happened.
- The court polygon in `data/courts/xbotgo.json` is a guessed trapezoid, not a calibration.
- Hand-labelling of shots on the XbotGo clip has not started. The task's "works on `data/raw/xbotgo.mp4`" box is open.
- Nothing is committed.

Next steps:
- Student: run `scripts/label_shots.py` in a real window and check the arrow keys.
- Student: calibrate the court on frame 430.
- Student: confirm Q5 and Q15. Close NCAA and NBA from the design doc section 5 list (about 10 minutes), or accept them as provisional.
- Confirm whether step 6 runs, then review the diff and commit by hand.
- Decide whether shot labels stay in `data/labels/` or move to `data/ground_truth/` (Q14).
- Once shots are hand-labelled, score them with `eval/score_shots.py`. Only then is there an accuracy number, and it needs a citation to its report.

## 2026-11-16
*(session began 2026-11-15 and ran past midnight; tasks 002 and 003 below.)*

**Task 003: the project's first real accuracy measurement.** A 10 s window of `data/raw/xbotgo.mp4` (original frames 307-546, every 6th frame = 40 labelled frames) was pre-filled from the A0 run, hand-corrected in CVAT, and exported to `data/ground_truth/xbotgo_10s_mot.txt` (289 boxes, 9 real players). `eval/score_tracking.py` scores it with motmetrics 1.4.0 (MIT).

- **A0 scores IDF1 0.619 with 11 ID switches**, all 9 players mostly-tracked, against 17 tracker IDs used in that window. Everything before this was self-consistency, not accuracy. Source: `eval/reports/2026-11-15-idf1-a0-baseline.md`.
- Two caveats that limit what the key can prove. **Recall (0.993) is circular**: the key was built from the tracker's own boxes, so it says nothing about detection. **Precision is biased downward and not neutral across thresholds** — unlabelled real players count as false positives, so a higher threshold flatters itself. Weight ID switches and MT/ML over precision.
- The clip was chosen by measuring camera motion over all 719 frames: the camera pans at 8-11 s (4-5 px/frame) and is locked off at 15-21 s (0.00-0.01 px/frame). That same measurement gives the court filter its only valid range, frames 359-503.

**Task 002: the premise did not hold here either.** Evidence: `eval/reports/2026-11-15-roles-thresholds.md`.

- **The role vote is a relabelling only.** It changes no ID switches, no MT/ML and no tracker IDs on the answer key, and costs about 0.017 IDF1. It moves the player count 45 -> 43. The brief's premise (line 19) that flickering tracks "count twice and the player count is inflated" was wrong: a both-label track contributes one id. What was double-counted was the breakdown (45 players + 21 referees = 66 for 47 real tracks); the vote fixes that (43 + 4 = 47). **43 player tracks for 9 real players is still ~4.8x fragmentation, untouched.**
- **Person threshold stays 0.4.** ID switches rise to 18 at 0.5 and 16 at 0.6 against 11 at 0.4, and at 0.6 two key players drop to partially tracked. Ball stays 0.4: the 0.25 floor lifts detection from 28.8% to 48.0%, but multi-candidate frames go 3 -> 31 and large centre-jumps between consecutive detections rise from 4.7% to 15.7%, so part of the gain may be spurious. **A few dozen hand-labelled ball positions would settle it, and that is the most valuable next measurement for shot detection.**
- The grid cost one detection pass plus 5 replays instead of 27 runs, because the ball never reaches `tracker.update` and the vote is a post-process. The evaluator asserted that independence rather than assuming it: ball-varied runs have byte-identical player rows, person-varied runs identical ball rows.
- **Court filter is code-complete and untested on real geometry.** `scripts/calibrate_court.py` exists and round-trips; the filter drops 37 of 1,565 person detections in frames 359-503. But the polygon used was a guessed trapezoid never checked against the court lines. Real calibration needs a human clicking frame 430.
- The brief's ">1 s gap" merge-check threshold proved nearly vacuous (ByteTrack's buffer is ~1.25 s; largest observed gap 0.96 s, giving 0 reconnections). At 0.5 s it finds 12/14/19 at person 0.4/0.5/0.6. At 0.4 three look like merges (tracks 6, 15, 28); tracks 6 and 15 are long mixed-person tracks the vote labels "player" throughout, so `unique_player_ids_by_role` is not trustworthy at the margin.

**Not done, and relevant to the commit:**
- **The final review did not complete.** It cleared `scripts/calibrate_court.py` as privacy-clean, then stopped on a rate limit before reaching the `eval/score_tracking.py` changes and `eval/xbotgo_10s_frame_map.json`. Those remain unreviewed. The main session checked the scorer's IoU function numerically (7 hand-computed cases, correct orientation) and verified the replay gate reproduces byte-identically, but that is not a review pass.
- Two open items flagged and not acted on: `data/ground_truth/xbotgo_10s_README.md` identifies nine minors by jersey number and clothing in a committable file; and `ultralytics-thop` (AGPL-3.0) and `ultralytics-platform` (AGPL-3.0-only) are installed in `.venv` as orphans, which `requirements.txt`'s planned `pip freeze` lock would pin into the manifest. CLAUDE.md forbids Ultralytics AGPL.
- The A0 annotated video has still not been reviewed by hand for merged IDs.

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
- Winner video, to be reviewed by hand: `runs/task001_eval_video/A0/xbotgo_tracked.mp4`.

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

Open decision (not settled by the eval):
- The code default is still `--tracker botsort` (B1, 49 IDs). The evaluator recommends against switching the default to BoT-SORT on this evidence, and the winner is A0. Changing the default is a code change for model-builder, not a docs change. The task brief's "BoT-SORT is the default" box is ticked as an implementation fact, with this caveat written next to it.

Blockers and risks:
- **No ground truth yet.** `data/ground_truth/` is empty. The eval measures tracker self-consistency only. No ID-switch rate, MOTA or IDF1 can be claimed.
- One 30 s clip. Differences of a few IDs are noise.
- Still outstanding: the diff has not been read, the A0 video has not been reviewed for merged IDs, the default has not been decided, and nothing has been committed.

Next steps:
- Student: read the diff; watch `runs/task001_eval_video/A0/xbotgo_tracked.mp4` for merged IDs and for players labelled "referee"; decide the default preset (A0 vs A1 vs leaving BoT-SORT); commit.
- Hand-label about 10 s of IDs on the clip if IDF1 is wanted. That would allow an ID-level metric.
- Resolve ground truth (hand-charted stats for at least one game) before any accuracy claim.
- Test on more than one clip.

**Later on 2026-11-15: default tracker decided (resolves the open decision above).** The default is now ByteTrack preset A0 (task 002 Step 0). BoT-SORT and all presets stay selectable via `--tracker botsort` and `--preset`. The reason is evidence: BoT-SORT's tracker cost is about 17 to 19 times A0's per frame, with no consistent gain in IDs and fewer long-lived IDs by presence. CMC showed no signal (B0 vs B1).

A1 (43 IDs, the only count below 45) was rejected after a manual merge check, done by hand on the annotated video and crops, of A1's reconnections after gaps over 1 s (`runs/task001_eval/A1/xbotgo_tracks.csv`). Two of three are wrong merges: ID 28 joins a blue-jersey player (frame 666) to a different white-shirt player (frame 709), and ID 20 joins a white-jersey player (frame 500) to a spectator by the door (frame 538). ID 0 (frames 159 to 192) is probably correct. So A1's 43 is partly an artifact of merging different people. This is the failure mode the "fewer IDs is not automatically better" caveat in `eval/reports/2026-11-15-botsort-vs-bytetrack.md` predicted, and it is why task 002 makes a merge check a required part of its eval.

## 2026-11-14
- Pivoted from ultimate frisbee to basketball (youth shot & stat tracking).
- Set up project structure and Claude Code agent team (architect on Fable; researcher, model-builder, evaluator on Sonnet; reviewer on Opus; data-engineer and report-writer on Haiku).
- Next: work through the "Before recording any footage" checklist in project-plan.md.

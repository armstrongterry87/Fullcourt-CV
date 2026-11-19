
## Morning summary

Task 005 Part A (camera motion compensation) is built and has been checked on `data/raw/xbotgo.mp4` (30 s, 719 frames). **Nothing is committed.** Part B has not started.

**What was done**
- Design: `docs/design/005-motion-compensation.md` (canonical). Opus stand-in kept at `docs/design/005-motion-compensation-opus-standin.md`. See A1 notes below.
- Build (A2): `src/camera/motion.py`, `scripts/stabilize.py`, mosaic mode in `scripts/calibrate_court.py`, `--motion` in `scripts/detect_shots.py` and `src/shots/detect.py`, changes in `src/court/geometry.py`, `tests/test_motion.py`. `ALGO_VERSION` is 3.
- Review (A4): 0 blocking, 3 should-fix, all fixed (see A2-Q10).
- Physical checks (A3): `eval/reports/2026-11-18-motion-compensation.md`.
- Log (A5): progress-log entry and AI-use-log rows dated 2026-11-18.

**Test status:** 128 passed (98 existing + 30 new). These counts come from the run notes. Not re-run here, because the report-writer has no shell.

**A3 verdict, honestly**
- It works at the frame centre across the roughly 85 degree pan. Centre error is median 0.5 px (p95 1.4 px). 718 of 719 frames (99.9%) get a pose at quality 0.6 or better. A far scoreboard stays within about 11 px across the fast pan.
- One clean static person (track 34) drops from 110.6 px to 9.7 px p95.
- It misses targets:
  - Independent drift: max-corner error median 3.8 px, p95 10.5 px, against 2 and 5 px. Mean-corner error is 2.1 and 5.7 px, a narrow miss.
  - 60% of the clip (430 frames) is capped at 0.6 by a 20-frame chained edge to keyframe 3. Only 30.0% of frames reach 0.9.
  - The left rim is stretched 2.65x in view 0. The guideline is 1.25, and the view bound is 1.5.
  - Single-frame corner glitches of 19 to 47 px at frames 221 to 250 and 615 to 616. Quality does not flag them; they stay at 0.6.
  - Bench seat A fails the full-gate static check (p95 32.7 px against 15). Only one valid static person was found, not three.
- Rim stability: the absolute targets pass (left rim p95 7.8 view px, about 3 raw-equivalent px). The check has almost no power, because a rim is only detected while the camera is still.
- Closure error was not measured. The graph has no loops, so the 0.00 is meaningless.
- The two raw rim positions (x = 30 and x = 1250) are two different baskets, at roughly opposite ends of the gym. Their bearings depend on an assumed focal length of 800 to 1050 px.
- A 30 s clip says nothing about drift over a full game.

**Builder follow-ups from the A3 report (not fixed)**
1. Gate points, not the frame centre. `usable()` and `transform_tracks` test stretch only at the frame centre. Points near edges reach stretch 16 to 260.
2. Quality 0.6 is too permissive for 33 to 47 px jumps.
3. Replace the chained edge to keyframe 3 with a direct edge or a loop. This would also give a real closure number.
4. Pose jitter on a still camera: median 3.9 px at the corners, up to 18 px.
5. Left rim stretch 2.65 in view 0. Centre a view on a rim whose stretch exceeds 1.25. Rim boxes cut by the frame edge (53 of 194) bias rim centres.
6. Stale numbers in A2-Q1 and A2-Q4. **Fixed in this file:** `ALGO_VERSION` is 3, and 30.0% is at 0.9.
7. The detector rarely fires on seated people, so the static-people check cannot run at full strength.

**Decisions only the owner can make**
1. Accept the design-doc handling: the Fable doc is canonical, and the Opus stand-in is kept as a separate file, not merged.
2. Whether to fix the A3 findings before Part B. The A3 verdict says shots released during a fast pan should not be trusted even at 0.6. Follow-ups 1 to 3 are the candidates.
3. Interactive mosaic clicking in a real window was never tested (A2-Q3). It needs a run before Part B.
4. Rim calibration: click rims on the mosaic (design) or on raw keyframes (stand-in proposal).

**Not done:** nothing is committed. All Task 005 files are in the working tree. Part B needs `data/raw/game1.mp4` and hand-labelled shots, and there is no accuracy number.

## A1 notes

- The first Fable attempt at the A1 design failed on Fable usage credits.
- An Opus stand-in (main session standing in, per its header) wrote a second design that briefly overwrote `docs/design/005-motion-compensation.md`. It was preserved as `docs/design/005-motion-compensation-opus-standin.md`. The Fable version was restored as canonical. The files do not record which run produced the restored text. The restored doc's header says the architect agent (Fable) wrote it.
- The stand-in's probe numbers appear in the canonical doc's "Probe findings" section, marked as unverified leads.

## A2 questions

**A2-Q1. Pass 1 needed a "rescue mode" the design does not describe.** On `xbotgo.mp4` the first build (5 Hz samples matched only to the current keyframe) found one keyframe and left 68% of frames unreliable: the fast pan blurs the samples, so no sample ever fitted the old keyframe well enough (quality 0.9) to become a new keyframe. Assumption: when a sample fails, pass 1 chains every frame to the previous localised frame until a sharp frame (at least 60% of the first keyframe's feature count) can become a new keyframe, or an old keyframe fits again. The new keyframe's edge is the composed chain (limit 300 frames). Result on the clip: 4 keyframes, 2 views, 99.9% usable. Cost: a chained edge carries accumulated error, and the clip has no loop edges (max_closure_err_px = 0.00 means "no loop to measure", not "perfect"). A3 should check drift independently. `ALGO_VERSION` is 3 (the build A3 measured).

**A2-Q2. Static motion file plus a static calibration writes no extra `shots.csv` columns.** Design 9.3 says the two columns (`view_id`, `camera_quality`) appear "when a motion file is used". Assumption: they appear only when the stabilised path runs (camera_static false), so a static video gives byte-identical output with or without `--motion`.

**A2-Q3. Landmarks and rims on a mosaic.** `calibrate_court.py` had no landmark or rim clicking at all (only the polygon). Mosaic mode adds it: `l` then click and type feet in the console, `a`/`b` two clicks per rim; non-interactive `--rim` and `--landmark` for tests. The default output is `data/courts/<stem>_mosaic.json`, not `<stem>.json`, so the existing polygon file is never overwritten. Pass `--out` to merge elsewhere. Interactive mode was NOT exercised in a real window.

**A2-Q4. Quality of the 0.6 level.** Only 30.0% of xbotgo frames are 0.9 (216 of 719, v3 cache, A3 report); most of the rest are 0.6 (chained frames, or fits just under the 0.9 thresholds). Every shot there is capped at confidence 0.6 (`camera_degraded`). The thresholds are untuned (T values); A3 should say whether they are too strict.

**A2-Q5. `camera_degraded` also fires when the span holds a short unreliable run (quality 0) under max_gap_s,** not only for 0.6. The design says "lowest quality is 0.6"; I used "lowest quality below 0.9".

**A2-Q6. Rim drift window constants** (`10 s` window, at least 10 rim detections, detections within 2 W of the rim) are CHOSEN in `src/shots/detect.py`; the design gives only the 10 s window and 0.5 W.

**A2-Q7. The mosaic has black holes** where people stand in every sampled frame (visible at the right end of `data/frames/xbotgo_mosaic.jpg`). Sampling more frames (`mosaic_max_frames` is 25) would fill them. Not changed.

**A2-Q8. Mask uses dilation 0.15*max(w,h)+8 px.** A very large close-up person (over about 40% of the frame) masks most of the frame; the estimate then fails to unreliable, as intended.

**A2-Q9. `--max-seconds` writes `<stem>__motion__<key8>__first<N>.npz`,** so a clip cache is never taken for the full video's.

**A2-Q10. Reviewer fixes (round 2), new assumptions.**
- Chained keyframe edges: a rescue-born keyframe is first matched DIRECTLY to every keyframe predicted to overlap (and to the current one). If any direct fit is good, only direct edges are kept; the chained edge is kept only when there is none, and records `chain_len` (cache array `key_chain_len`). Any frame whose tree path to the anchor uses a chained edge is capped at quality 0.6, and the keyframe's own frame too. The anchor-path BFS now prefers direct edges over chained ones.
- The pass-1 rescue chain limit is `pass1_chain_limit_s = 12.0` seconds (about 300 frames at 24 fps), converted with the video's fps.
- Pass 1 had the same stale-pose flaw as pass 2: in rescue mode with a non-adjacent last localised frame, it now tries every keyframe for re-anchoring, not only those predicted to overlap.
- Pass 2 recovery: when the last good frame is not the previous frame, or the chain limit was hit, every keyframe is tried, ordered by predicted overlap then nearness in time.
- The jump check is skipped when the allowed jump is at least 2 frame widths (`jump_skip_w`): after a long gap any pose is plausible, and far corners map behind the plane, so the old check rejected genuine recoveries.
- `detect_shots` raises ConfigError when `motion.meta['fps']` differs from `fps` by more than 1e-3, or the tracks reach a frame at or beyond `motion.n_frames`. The CLI now takes fps from the motion file when `--fps` is not given, because the CSV's rounded `time_s` gives a slightly different fps.
- Pass-2 stats (`n_matches`, `n_inliers`, `inlier_ratio`, `spread_cells`) of a keyframe's own frame are filler (the frame is matched to itself); do not use them as quality evidence.

# Camera motion compensation: A3 physical validation on xbotgo.mp4

Date: 2026-11-18. Evaluator: A3 of `docs/tasks/005-camera-motion-compensation.md`, following design 005 section 12.1.

## What was evaluated

| Item | Value |
|---|---|
| Code | HEAD `767ecd6` plus uncommitted working tree (`src/camera/motion.py` is untracked; `ALGO_VERSION = 3`). Nothing in `src/` was edited. |
| Video | `data/raw/xbotgo.mp4`, 1280x720, 23.951 fps, 719 frames (30.0 s). One clip, one game, one camera position. |
| Detection cache | `runs/cache/xbotgo__wfc2f7c4e__small__thr0.25.npz` (used for the mask, the rim detections and the drift pairs) |
| Motion cache | `runs/cache/xbotgo__motion__46052b9a.npz`, `motion_id c1be939b616e02bb`, cv2 5.0.0, default `MotionConfig` |
| Tracks | `runs/finetuned/xbotgo_tracks.csv` (ByteTrack A0 preset, fine-tuned detector; built from the 0.40 threshold detections, not the 0.25 cache the mask used) |
| Debug video | `runs/xbotgo_motion_debug.mp4` (719 frames) |
| Sample size | 1 clip, 0.5 minutes, 0 shots. Rim detections: 194 (conf >= 0.4). Pairs for drift: 200 random + 716 consecutive. People: 1 valid moving-camera static person (track 34) plus partial bench seats (details below). **This is a small, single-clip sample. A 30 s clip says nothing about drift over a full game.** |

No previous motion-compensation report exists, so the comparison is raw versus stabilised, side by side, against the design targets (targets, not gates).

## Reproduce

```
# frames with track ids, and debug-video frames, for the by-eye picks (written under runs/ only)
.venv/Scripts/python eval/motion_frames.py --video data/raw/xbotgo.mp4 --tracks runs/finetuned/xbotgo_tracks.csv --out runs/a3_frames --frames 10 120 250 400 500 540 600 660 705 --debug-video runs/xbotgo_motion_debug.mp4
# rim, static people, scoreboard landmark, drift (200 pairs, seed 0), coverage, structure -> runs/a3_motion_checks.json
.venv/Scripts/python eval/motion_checks.py --video data/raw/xbotgo.mp4 --det runs/cache/xbotgo__wfc2f7c4e__small__thr0.25.npz --motion runs/cache/xbotgo__motion__46052b9a.npz --tracks runs/finetuned/xbotgo_tracks.csv
# consecutive-frame jump check -> runs/a3_adjacent.csv
.venv/Scripts/python eval/motion_adjacent.py --video data/raw/xbotgo.mp4 --det runs/cache/xbotgo__wfc2f7c4e__small__thr0.25.npz --motion runs/cache/xbotgo__motion__46052b9a.npz
```

Per-pair drift rows: `runs/a3_drift_pairs.csv`. Everything in `runs/` is git-ignored.

## Results

### Structure

| Item | Value | Target |
|---|---|---|
| Keyframes | 4, at frames 0, 207, 237, 274 | |
| Views | 2 (view 0: keyframes 0,1,2, canvas 2845x1124; view 1: keyframe 3, canvas 1281x721) | |
| Anchor, hops | keyframe 1 (frame 207); max hops 2 | |
| Chained edge | keyframe 3 (frame 274) hangs on a chained edge, chain length 20 frames | |
| `max_closure_err_px` | 0.00. **Meaningless: the graph is a tree with no loops, nothing was measured.** | <= 5 (cannot be assessed) |
| `is_static` | False | |
| `zoom_suspected` | 0 frames | context |
| Frames usable in view 0 / view 1 / both / neither | 332 / 495 / 109 / 1 (the one unreliable frame) | |
| Stretch at frame centre, median / max | view 0: 1.23 / 2.99; view 1: 1.00 / 2.99 | view <= 1.5 intended |
| Stretch at the rims | left rim (view 0) median 2.54, max 2.65; right rim (view 1) median 0.98, max 0.99 | <= 1.25 (design 12.2 risk 6) |

### Coverage

| Item | Value | Target |
|---|---|---|
| Frames at 0.9 | 216 (30.0%) | |
| Frames at 0.6 or better | 718 (99.86%) | >= 95% |
| Unreliable | 1 frame (203), 0.04 s | |
| Longest unreliable run | 1 frame, 0.04 s | <= 1 s |
| Runs longer than `max_gap_s` (0.125 s = 3 frames) | 0 | |

By time the picture is much worse than the headline: 0.9 share per 100 frames is 1.00, 0.90, 0.26, then 0.00 for frames 300 to 718. **All 430 frames matched to keyframe 3 are capped at 0.6 because of the chained edge, which is 60% of the clip.** Of the 446 directly matched frames at 0.6, 411 miss the 100-inlier bar, 286 miss rms <= 1.0, 61 miss inlier ratio >= 0.5 (counts overlap). So the thresholds are too strict only in the sense that inlier count (40 to 65 on frames 540 to 719) is the binding limit. I cannot say whether the 0.9 bar is "too strict" or the match count is genuinely low, because the chain cap hides it. Another 56 frames at 0.6 are pass-2 chained frames.

### Rim stability (conf >= 0.4, 194 detections, 53 of them cut by the frame edge)

Raw rim detections span x = 30 to 1250 px. Raw clustering (DBSCAN, eps = 71 px = one raw rim width, min 10) gives two clusters. Per view, stabilised clusters use eps = one median rim width in that view.

| Rim cluster | Frames | n | Raw spread of the same detections: MAD / p95 / max (px) | Stabilised: MAD / p95 / max (view px) | MAD, p95 in rim widths (stabilised) | p95 vs raw | Target |
|---|---|---|---|---|---|---|---|
| Left (raw x 30 to 96), view 0 | 19-165 | 85 | 0.61 / 23.4 / 42.4 | 1.89 / 7.79 / 10.9 | 0.009 W, 0.039 W (W = 199.5 view px) | 0.33 | p95 <= 0.5 W and <= 25% of raw; MAD <= 0.15 W |
| same, excluding 7 edge-truncated boxes | | 78 | 0.59 / 13.5 / 42.4 | 1.84 / 3.94 / 5.14 | | 0.29 | |
| Right (raw x 1242 to 1250), view 1 | 382-549 | 103 | 0.49 / 1.88 / 7.28 | 0.58 / 1.98 / 2.76 | 0.009 W, 0.029 W (W = 67.9 view px) | 1.05 | same |
| same, excluding 46 edge-truncated boxes | | 57 | 0.35 / 1.39 / 2.55 | 0.55 / 1.56 / 2.11 | | 1.12 | |

Reading:
- The absolute targets (p95 <= 0.5 W, MAD <= 0.15 W) pass with a very large margin. The relative target (<= 25% of raw) is missed on the left rim (0.33; 0.29 without truncated boxes) and meaningless on the right rim.
- **This check has almost no power on this clip.** Rim detections only exist while the rim sits at the frame edge, and those moments are almost all camera-still stretches (frames 19-150 and 382-549). The raw right-rim spread is already 1.9 px. The left-rim raw spread (23 px) comes only from the first frames of the pan (frames about 150 to 165), where stabilisation does help (max error 42 px to 11 px). The rim is never detected while the camera is in fast motion, so rim stability cannot show whether stabilisation works there.
- View 0 stretches the left rim 2.65x (W = 199 px against 71 px raw), above the 1.5 view bound and the 1.25 guideline. Quoting the spread in view widths flatters the result. In raw-equivalent pixels the stabilised p95 is about 7.8 / 2.6 = 3.0 px.
- Rim boxes at the frame edge are cut off, so their centres are biased. 53 of 194 detections are cut.
- **Noise floor (context):** median change of the raw rim centre between consecutive detections 0.15 px (p95 0.94, n = 161); in view 0 between consecutive detections 0.67 view px (p95 3.4, n = 75, about 0.26 raw-equivalent px). The stabilised spreads above are at the noise floor.

**One rim or two?** Two. In reference coordinates the left rim lands at (-810, 14) while the right-edge rim cannot be mapped into the anchor plane at all (it maps behind it). Using the camera rotation recovered from the matrices with an assumed focal length (920 px from the probe finding, not verified by me; 800 to 1050 px gives the same answer), the left rim is at a bearing of about -57 degrees and the right rim at about +93 degrees from the anchor view, roughly 150 degrees apart, at the two ends of the gym. Frame centres sweep -25 to +60 degrees (an 85 degree pan, consistent with the probe's 88). Raw x = 30 and x = 1250 are not one rim seen at both edges; they are the two end baskets. A third (side) basket is also visible mid-court in some frames (for example near x = 540 in frame 10) but the detector never fired on it with conf >= 0.4.

### Static people

Picks made before looking at any stabilised number. Camera: still for frames 0-160, fast pan 160-290, still 290-540, slow pan back 540-719. The tracker does not hold seated people (activation threshold 0.7, and the detector rarely fires on them, usually labelled "referee" at conf 0.3 to 0.4), so almost no bench person has a track id.

| Person | How picked | Frames, n | Raw MAD / p95 (px) | Stabilised MAD / p95 (view px, median stretch) | Target p95 <= 15 |
|---|---|---|---|---|---|
| Track 34, standing spectator against the right wall (frames 501-718) | by eye from raw frames 500-705, then raw track data confirms smooth motion with the pan (x 915 to 1054) | 501-718, 209 | 43.0 / 110.6 | view 1: 2.3 / 9.7 (stretch 0.98) | pass |
| same, the 63 frames usable in view 0 | | 63 | 20.9 / 35.7 | view 0: 17.4 / 62.0 (stretch **16.3** at the person, so about 3.8 raw-equivalent px) | n/a |
| Bench seat A, seated, raw feet gate x 190-250, y2 474-488 (frames 372-630) | gate read by eye from raw frames and the raw detection list | 114 | 2.1 / 41.1 | view 1: 2.2 / 32.7 | **fail as measured** |
| Bench A, frames 372-590 only (**restriction chosen after seeing the full result**; see below) | | 97 | 1.6 / 24.8 | 1.8 / 5.9 (max 13.0) | pass |
| Bench A, frames 540-590 only (the slow pan back, the only informative part) | | 11 | 2.4 / 16.7 | 2.8 / 7.3 | pass, tiny n |
| Bench seat B (frames 373-519) | gate x 160-178 | 33 | 1.7 / 4.6 | 1.8 / 4.5 | uninformative, camera is still |
| Bench seat C (frames 381-511) | gate x 295-310 | 30 | 2.1 / 6.5 | 2.4 / 6.8 | uninformative, camera is still |
| Track 35 ("standing spectator", frames 548-679) | picked from frame 600 alone. **Bad pick.** | 124 | 123.8 / 274.1 | view 1: 91.0 / 198.8 | **fail, pick was wrong** |
| Control: track 6, on-court player | long-running player | 574 | 113 / 398 | view 1: 75 / 577 | must not look static: it does not |
| Control: track 19, on-court player | long-running player | 458 | 61 / 499 | view 1: 71 / 405 | does not look static |

What went wrong in the picks, in the open:
- **Track 35 was a bad pick.** From one frame it looked like a second standing spectator. Its raw x goes 815 to 1179 over frames 560-660 (3.6 px per frame), while track 34, which really is static, goes 931 to 1010 (0.9 px per frame). Track 35 walks. It stays in the table as a fail of my pick, not of the stabiliser.
- **Bench A fails as measured, and I do not fully know why.** The stabilised position is flat at about x = 166-172 view px across frames 372-590 (raw drifts 198 to 239, so the stabiliser removed about 45 px of drift), then drops to 137-143 at frames 616-630. Two causes are mixed in: (a) frame 615 is a single-frame pose glitch (corner jump of 33 px, section "Jumps" below), and (b) by frames 616-630 the camera has shifted about 50 px, more than the seat spacing of about 30 px, so my fixed x gate probably picks neighbouring seated people. I cannot separate the two from the data, so the honest number for A is the full-gate one (fail, p95 32.7 px), with the 372-590 subset as a diagnostic only.
- A first attempt that followed a seed by nearest neighbour made seats A and B collapse onto the same detections; it was replaced by the fixed gates above. Scoreboard readings were also redone on zoomed crops after the first pass (see next section). These are fixes to my measuring tools, made with the stabilised numbers visible, so they are not blind.
- **Valid evidence for "static people stay put" across a real pan: one person (track 34, n = 209, p95 9.7 px against 110.6 raw) plus 11 frames of bench A.** The design asked for at least 3 and I do not have them, because the detector and tracker rarely hold seated people. Treat the pass as weak.

Where a person sits far from the view centre, view pixels stop meaning much: stretch at track 34's feet is 16.3 in view 0 and 260 for track 35 (view 0). `usable()` gates only on stretch at the frame centre (<= 3.0), so points near a frame edge can be stretched far past it (see Findings).

### Independent landmark check (added by me, not in design 12.1)

The two gym ends carry identical scoreboards that show the same clock, so end-to-end aliasing is a real risk. I read the right edge of each scoreboard by eye on 3x zoomed crops (reading error about 3 to 5 px) and mapped it to the views. Board A is on the first-half scene (frames 10, 120, and the cut-off board at frame 250); board B on the later scene (frames 250 to 705).

| Board | Frame, raw (x,y) | View 1 | View 0 |
|---|---|---|---|
| A | 10 (897,255) / 120 (928,255) / 250 (55,248) | not usable except 250 | (1376.9,482.5) / (1377.5,484.3) / (1382.8,483.9) |
| B | 250 (997,247) / 400 (426,262) / 500 (417,261) / 600 (464,263) / 705 (554,266) | (406.2,260.6) / (402.2,260.9) / (394.9,259.7) / (403.1,260.8) / (402.1,263.2) | frames 250 and 705 only: (2459.0,399.1) / (2451.1,403.8) |

Board B across five frames spanning the fast pan (250) and the slow pan back (705) stays within about 11 px in x and 3.5 px in y in view 1, which is about the reading error plus the corner jitter below. Board A in view 0 is within 6 px across frames 10 to 250 (frames 10 and 120 are camera-still, so only 250 is informative). A and B land about 1070 view px apart, so the two ends are not confused. This is the best independent evidence that the mosaic is geometrically consistent across the pan, and it is one landmark.

### Independent drift (200 random pairs, seed 0, overlap >= 0.5 by the composed homography, direct SIFT fit with the same masks)

Corner error of `inv(H_b) @ H_a` against the direct fit, in px of frame b. "Max corner" is the largest of the 4 corners, "mean corner" the average of the 4; the design says only "corner error", so both are given. Median gap between the frames of a pair: 96 frames (49.5% over 100 frames). 10 of 200 pairs could not be fitted directly and are excluded; this hides the worst cases (8 of the 10 failures, in pairs not on the chained subtree).

| Group | Pairs fitted | Max corner: median / p95 / max | Mean corner: median / p95 | Centre: median / p95 | Target (median <= 2, p95 <= 5) |
|---|---|---|---|---|---|
| All | 190 | 3.78 / 10.5 / 21.5 | 2.07 / 5.68 | 0.52 / 1.42 | **miss** (mean corner median 2.07 and p95 5.68 narrowly) |
| Neither frame on the chained subtree (keyframes 0,1,2) | 20 | 1.70 / 4.31 / 4.45 | 0.88 / 2.42 | 0.18 / 0.45 | pass (n = 20, small) |
| One frame on the chained subtree | 18 | 8.02 / 17.0 / 21.5 | 5.10 / 7.59 | 0.66 / 1.57 | **miss** |
| Both on the chained subtree (keyframe 3) | 152 | 3.80 / 9.79 / 15.4 | 2.08 / 4.76 | 0.54 / 1.40 | miss on max corner |
| Gap <= 30 frames | 33 | 3.11 / 9.16 / 12.3 | 1.39 / 5.23 | 0.34 / 0.93 | |
| Gap > 100 frames | 94 | 4.43 / 13.0 / 21.5 | 2.38 / 5.91 | 0.67 / 1.51 | |

Reading:
- The error lives in the corners, not at the centre: centre error median 0.5 px and p95 1.4 px, corner error several times larger. That is the signature of poorly constrained projective terms, and it means objects near the frame edge (the right rim, sideline players) are stabilised worse than the centre.
- The chained edge is the weakest part. Pairs that cross between the chained subtree and the rest have a median max-corner error of 8.0 px (n = 18) against 1.7 px inside the clean subtree (n = 20). Only 20 "clean" pairs exist because 60% of frames hang on the chain, so the clean number is thin.
- Shared-estimator caveat: the direct fit uses the same SIFT, matcher and RANSAC code as the build. A systematic bias common to both (for example a wrong match between the two symmetric gym ends) would not show here. The scoreboard check above is the only independent check of that.
- Worst pairs (watch them): (364,704) 21.5 px, (532,703) 16.1, (485,713) 15.4, (698,663) 14.0, (707,480) 13.6, (537,298) 13.1, (665,698) 13.0, (701,498) 12.9. All involve a frame in the late stretch (frames 480 to 713).

### Jumps: consecutive-frame check (716 pairs, all fitted)

Composed `inv(H_{t+1}) @ H_t` against a direct fit, max-corner error in px.

| Phase | Pairs | Camera motion median / max (px per frame, mean corner) | Error median / p95 |
|---|---|---|---|
| Frames 0-160, camera still | 159 | 0.5 / 1.5 | 0.81 / 2.68 |
| Frames 160-290, fast pan | 128 | 8.2 / 55.6 | 2.08 / 12.8 |
| Frames 290-540, camera still | 250 | 0.0 / 2.7 | 3.89 / 10.0 |
| Frames 540-719, slow pan back | 179 | 1.2 / 2.5 | 4.01 / 9.04 |
| All | 716 | | 2.75 / 9.75 (p99 17.7, max 47.4) |

- **Pose jitter when the camera is still:** on frames 290-540 the direct fit says the camera moves 0.0 px, but the stabilised pose changes by a median 3.9 px (max 18 px) at the corners from frame to frame. A stabilised video of a still camera would shimmer at its edges.
- **Single-frame glitches (frame numbers):** frames 249 and 250 (45 and 47 px, in the fast pan), 232 (37 px, a keyframe hand-over from -1 to 2), 615 and 616 (33 and 35 px while the camera moves 1.6 px, 43 to 49 inliers), 221-222 (19 and 22 px), 259 (15 px, hand-over to keyframe 3), 537, 538 and 529 (16 to 18 px, camera still). These recover in one or two frames, with no persistent offset in the centre path (frame-centre position in view 1 is smooth through 606-633). In each case quality is 0.6, not 0, so downstream code would still use them.

### Visual (debug video `runs/xbotgo_motion_debug.mp4`)

I extracted frames 0, 100, 160, 190, 205, 230, 260, 280, 300, 450, 540, 600, 650, 718 (montages in `runs/a3_frames/dbg_montage*.jpg`, plus full frames 10, 120, 250, 400, 600). The overlay is a grid of yellow dots, a magenta line at the view-0 canvas edge (visible at the left in frames 100 to 190), and a border that is green at 0.9 and amber at 0.6.
- Frames 0 to about 160 (green): the grid sits still on the wall and ceiling features and the magenta edge stays at x about 35 to 55 in frames 100 to 190. No slide visible.
- Frames 190 onward are amber (0.6). During the fast pan (205 to 280) the grid keeps its place against the windows, vents and scoreboards at montage scale, and its spacing is visibly projective (denser at one side), as it should be. I cannot see drifts below about 10 px by eye at this scale.
- Frames 300 to 718 (view 1): grid and scene move together; dots near the scoreboard and the right basket stay on them at 450, 540, 600, 650 and 718 at montage scale. I saw no jump with the eye, but the frame 615-616 glitch (33 px at one corner) is below what I could see in a half-size tile, so the eye check proves little. The measured checks above are what carry the verdict.
- The grid covers only a part of the frame in view 1 frames; it is a fixed grid in the canvas, so this is expected.

## Findings for the model-builder (not fixed by me)

1. `usable()` / `transform_tracks` gate on stretch at the frame centre only. Points near a frame edge can sit at stretch 16 to 260 in a view (tracks 34 and 35 in view 0), where view pixels are meaningless. Gate per point.
2. Quality 0.6 is permissive: frames with 33 to 47 px corner jumps are 0.6, not 0. `camera_degraded` caps confidence but still scores them.
3. The chained edge to keyframe 3 caps 60% of the clip at 0.6 and measurably adds corner error (8 px median on crossing pairs). A direct edge, or a loop, would fix both and give a real closure number.
4. Pose jitter of a few px at the corners on a still camera (median 3.9 px max-corner, up to 18 px). The centre is fine (0.5 px).
5. The left rim ends up at stretch 2.65 in view 0 (design 12.2 risk 6 asks for a view centred on a rim above 1.25). Boxes cut by the frame edge (53 of 194 rim detections) bias the rim centre.
6. `docs/tasks/005-questions.md` A2-Q1 and A2-Q4 quote the old build (`ALGO_VERSION` 2, 33.7% at 0.9). The v3 cache gives 30.0%.
7. The detector rarely fires on seated people, so the static-people check cannot be run at the intended strength, and later shot attribution near benches will be thin.

## Verdict

Camera motion compensation clearly works for the whole-frame geometry. The pose is sound at the frame centre (0.5 px), the 85 degree pan is mapped with no visible jump, 99.9% of frames get a pose, there is no sign of zoom, the one clean static person drops from 110 px to 9.7 px spread, and an independent landmark (the far scoreboard) stays within about 11 px across the fast pan and the pan back with the two gym ends kept apart. It falls short of the design targets in four places: independent drift is 3.8 px median and 10.5 px p95 (maximum corner) against 2 and 5 px, with the error concentrated at the corners and worst across the chained keyframe; 60% of the clip is capped at 0.6 by that chained edge; the left rim is stretched 2.65x by its view; and the "static people" evidence is one person, not three. Rim stability passes the absolute targets but has no power, since the rim is only detected while the camera is still, so it neither confirms nor contradicts the stabiliser.

For Part B, treat positions near the frame centre as trustworthy to about 1 px and positions near the frame edges (rims at the edge, sideline players) as good to roughly 5 to 10 px, with occasional single-frame jumps of 30 to 47 px during fast pans that the quality score does not flag. Shots released during a fast pan should not be trusted even at 0.6. These are a 30 s, single-camera results and say nothing about drift over a full game.

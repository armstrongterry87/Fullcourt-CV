# Design 005: Camera motion compensation (a "virtual static camera")

Status: design for task 005, step A1. Written 2026-11-18 in an unattended run by the main session standing in for `architect` (Fable was unavailable). Nothing below has been built. The XbotGo numbers in section 1.1 come from a throwaway read-only probe run this session. It is not repo code and nothing was written to disk. Everything else is a decision or a tuning default (**T**).

Read with: `docs/design/004-shot-scoring.md` (the engine this feeds), `docs/tasks/005-camera-motion-compensation.md` (the brief).

---

## 0. Decisions in one screen

| Topic | Decision |
|---|---|
| Motion model | One 3x3 homography per frame, **frame → reference** (`H_i_to_ref`), in full-resolution pixels. |
| Reference | A **real frame** `R` (a keyframe), chosen after pass 1 to minimise the worst stretch. Reference coordinates are therefore just the raw pixels of frame `R`. |
| Features | SIFT on greyscale, long side downscaled to 960 px, 2000 features, `contrastThreshold=0.02`. |
| Matcher | Brute force L2, k=2, Lowe ratio 0.75. |
| Robust fit | `cv2.findHomography(..., cv2.RANSAC, 2.5 px, maxIters=5000, confidence=0.999)`, `cv2.setRNGSeed(0)` before every call. |
| Mask | Every person and ball box **from the detection cache for the same frame index**, conf ≥ 0.25, dilated. Rim boxes are **not** masked. Plus a 16 px border and optional overlay rectangles. |
| Drift | Every frame is matched **directly to an anchored keyframe**, not chained. New keyframe when overlap with all keyframes < 0.60. Revisits re-use old keyframes, which closes loops for free. |
| Reliability | `ok` / `bridged` / `unreliable`, decided by inliers ≥ 40, inlier ratio ≥ 0.30, RMSE ≤ 1.5 px, grid coverage ≥ 4/16, jump ≤ 6 px. |
| Scoring geometry | Ball and rim tests run in each **rim's own view** (the raw pixels of the frame the rim was clicked on), **not** in the global reference. Held and shooter tests stay in raw frame pixels. Feet go raw → ft through `H_ref_to_ft · H_i_to_ref`. |
| Downstream failure | Ball detections in unreliable frames are dropped. Any attempt whose evidence touches an unreliable frame is `unknown` (`camera_unreliable`). Bridged frames cap confidence at 0.6. |
| `camera_static` | `false` with no motion file → refuse to score. `true` with measured motion → loud warning, then score with motion compensation. |
| Cache | `runs/cache/<stem>__motion__<key8>.npz`. The key is a SHA-256 over the video fingerprint, the detection-cache fingerprint, every `MotionConfig` field, `MOTION_VERSION` and `cv2.__version__`. |
| Static video | With no `--motion`, the old code path runs unchanged (byte-identical output). With `--motion` on a static video, every H is identity and the output must be byte-identical too. |

---

## 1. Motion model

### 1.1 What the XbotGo clip actually does (measured this session)

Probe: SIFT (2000 features) on half-resolution greyscale, consecutive frame pairs, person and ball boxes masked from `runs/cache/xbotgo__wfc2f7c4e__small__thr0.25.npz`, OpenCV 5.0.0. Video: 1280x720, 23.951 fps, 719 frames.

| Quantity | Value |
|---|---|
| Inliers per consecutive pair (half-res) | min 65, median 419 |
| Inlier reprojection RMSE (full-res px) | median 0.24, p95 0.83 |
| Horizontal shift per frame (full-res) | median 0.5 px, p95 28.9 px, max 50.8 px |
| Vertical shift per frame | median 0.1 px, max 10.3 px (small tilt) |
| Pairs with \|dx\| > 2 px | 71 of 718 |
| Focal length, from the rotation structure of 49 fast-pan homographies | f ≈ **920 px** (IQR 905-935), so horizontal FOV ≈ 2·atan(640/920) ≈ **70°** |
| Net pan | ≈ 1420 px of f·Δθ ≈ **1.54 rad ≈ 88°** |
| Timeline | roughly still for frames 0-160; fast pan about 160-290 (6.7-12.1 s); still about 290-540 (this contains the locked-off 359-503 used by `calibrate_court.py`); slow pan back about 540-718 |

**This answers task 004 Q29 for the existing clip: the camera moved, by about 90°.** It also explains the rim detections at x = 30 and x = 1250 (Q2). Each end of the court sits at an opposite edge of the frame at the two ends of the pan.

When the probe chained everything to frame 0, frame 0's own centre point reached x ≈ 10 000 px by frame 289. A view about 88° away from frame 0's axis projects almost to infinity on frame 0's image plane. **So frame 0 cannot be the reference** (section 3.3), and **the global reference cannot be where the rim tests run** (section 5.2).

### 1.2 Why a homography

A camera that only **rotates about its optical centre** (pan and tilt), with fixed intrinsics `K`, maps every scene point, at any depth, between two views by

```
p_b ~ K · R_ab · K^-1 · p_a        (homogeneous pixel column vectors)
```

That is a 3x3 homography `H_a_to_b = K R_ab K^-1`. Depth cancels, so the floor, walls, stands and rim all move together under one H. This is why one H per frame is well-posed here, and why the rim (10 ft above the floor) can be stabilised by the same H as the floor.

Separately, the court floor is a plane. So floor pixels map to court feet by one homography `H_ref_to_ft`, whatever the camera does, as long as the reference is fixed.

### 1.3 When it breaks

| Situation | Effect | Detected by | v1 handling |
|---|---|---|---|
| **Camera translates** (handheld, walking, a tripod bumped or moved) | Parallax. Near and far points shift differently, so no single H fits. The floor-plane H is still valid for floor points, but the rim and the ball (off the plane) are wrong. | RMSE rises and inliers split into depth layers. The rotation-consistency check (1.4) fails. | Out of scope. Frames fail the RMSE or ratio test and become `unreliable`. A permanent bump shows as a jump against the keyframes. |
| **Lens centre offset from the pan axis** (a phone on a gimbal: about 3-5 cm) | Small parallax. Estimate: f · t · (1/Z_near − 1/Z_far) ≈ 920 · (0.05 m · 0.52 rad) · (1/5 m − 1/30 m) ≈ **4 px** across one keyframe hop of about 30°, and under 1 px between consecutive frames. | Keyframe-match RMSE | Tolerated. This is why RANSAC uses 2.5 px and not 1 px. A3 measures it. |
| **Zoom** | Optical zoom about a fixed centre is still a homography in principle (`H = K_b R K_a^-1`). It breaks in practice: lens distortion changes with zoom and is not modelled; digital zoom crops and resamples; and the scale of the rim view changes, so pixel thresholds drift. | Rotation-consistency check (inconsistent f) | Out of scope (task brief). The brief states these cameras pan but do not zoom. |
| **Lens distortion** (wide phone lenses) | Straight lines curve, so the H residual grows toward the frame edges. | RMSE and grid coverage | Tolerated at 2.5 px. Undistortion is a follow-up if A3 shows edge residuals above about 1.5 px. |
| **Rolling shutter during fast pans** (up to 51 px/frame measured) | The frame is skewed: rows are captured at different pan angles. | RMSE at pan peaks | Tolerated. A3 reports RMSE against pan speed. |
| **Pan wider than about 150° in total** | Some frames' corners land behind the reference plane (w ≤ 0). | The `out_of_canvas` reason | Those frames are `unreliable`. Score only inside the canvas. |

### 1.4 Rotation-consistency check (diagnostic, not on the critical path)

For each pair of anchored keyframes with a pan above 5°, compute `M = K^-1 · H · K / det(...)^(1/3)` with `K = [[f,0,cx],[0,f,cy],[0,0,1]]`, where `(cx, cy)` is the image centre. Search f over `[0.3, 3.0] × width` in 1% steps, and take the f that minimises the median `‖MᵀM − I‖_F`.

Write `f_est_px` and `rotation_residual` to the motion meta. If the best residual is above 0.02 (**T**), warn: "camera motion is not a pure rotation (zoom or translation?); motion compensation may be wrong". This is the explicit test for the "when it breaks" cases. It never changes any H.

### 1.5 Conventions (the reviewer checks these)

- `H_a_to_b` maps a homogeneous pixel column vector of view `a` to view `b`: `p_b ~ H_a_to_b @ p_a`.
- **Composition is right to left:** `H_a_to_c = H_b_to_c @ H_a_to_b`. Inverse: `H_b_to_a = inv(H_a_to_b)`.
- `cv2.findHomography(src_pts, dst_pts)` returns `H_src_to_dst`. When frame `i` is matched to keyframe `k`, `src` = frame `i` keypoints (the query) and `dst` = keyframe `k` keypoints (the train). That gives `H_i_to_k`, and then **`H_i_to_ref = H_k_to_ref @ H_i_to_k`**.
- Every stored H is in **full-resolution** pixels, normalised so `H[2,2] = 1`. Features are found at processing scale `s = min(1, 960 / max(W, H))` (**T**). With `S = [[s, 0, 0.5s − 0.5], [0, s, 0.5s − 0.5], [0, 0, 1]]` (the exact `cv2.resize` pixel-centre mapping), `H_full = inv(S) @ H_proc @ S`.
- Points are mapped with `cv2.perspectiveTransform`, or the equivalent numpy code. A point whose `w = h3ᵀp ≤ 1e-6 · ‖h3‖` maps to `NaN`, never to a huge finite number.
- Local scale of H at point p: `|det J| = |det H| / |w|³`, with `w = h3ᵀp` and p homogeneous with `p[2] = 1`. The linear stretch is `sqrt(|det J|)`.
- **Standard error metric** between two estimates of one frame's H, used in the jump check, the tests and A3: `frame_err_px(H_a, H_b) = max over the 4 corners and the centre p of frame i of ‖(inv(H_b) @ H_a)(p) − p‖`. It is measured in frame `i`'s own pixels, so the stretch of the reference cannot inflate it.
- Frame indices are 0-based decode order from sequential `cv2.VideoCapture.read()`. That is the same order as `sv.get_video_frames_generator`, which the detection cache and the tracks CSV use. Never seek. fps comes from the video and is stored.

---

## 2. Features and masking

### 2.1 Feature type: SIFT

- **Choice:** `cv2.SIFT_create(nfeatures=2000, contrastThreshold=0.02)` (**T**). It is in OpenCV main since 4.4, the patent has expired, and OpenCV 4.5+ is Apache-2.0. **No new package.**
- **Why not ORB:** ORB's binary descriptors degrade badly under the motion blur of a 50 px/frame pan, and its FAST corners cluster on high-contrast spots (scoreboards, banners). SIFT's scale invariance also covers the small scale change across a keyframe hop.
- **Why not AKAZE:** it would work. SIFT matched well on this exact clip (median 419 inliers at half resolution), so there is no reason to switch.
- **ORB is the documented fallback** if A3 or a full game shows the runtime is unacceptable (section 11, risk 6). Change `MotionConfig.feature` and nothing else.
- The lowered `contrastThreshold` (default 0.04) finds features on low-texture gym walls and wood floors.

### 2.2 Matcher and RANSAC

- `cv2.BFMatcher(cv2.NORM_L2)`, `knnMatch(k=2)`, ratio 0.75 (**T**). At about 2000 x 2000 descriptors, brute force is fast enough and exact. FLANN adds nondeterminism for no gain.
- `cv2.setRNGSeed(cfg.rng_seed)` immediately before every `findHomography` call, so the cache is reproducible.
- `cv2.findHomography(src, dst, cv2.RANSAC, ransacReprojThreshold=2.5, maxIters=5000, confidence=0.999)`. The threshold is in **full-resolution** pixels. Convert it to processing scale (`2.5 · s`) if the fit runs on processing-scale points, or scale the points up to full resolution first. **The build does the latter: keypoints are converted to full-resolution coordinates before the fit**, so `H_full` comes directly and the threshold is in full-resolution pixels.
- OpenCV's RANSAC already refines on the inliers with Levenberg-Marquardt, so no extra refit is needed.
- Why 2.5 px: measured consecutive RMSE p95 is 0.83 px, and keyframe hops add up to about 4 px of lens-offset parallax (1.3). 2.5 px keeps most of the floor and the walls in one consensus without admitting players.
- At least 8 ratio-test matches are needed to attempt a fit. Otherwise the reason is `no_features`.

### 2.3 Mask (built per frame, from the cache, for the **same** frame index)

Inputs: the detection cache (`src/detection/cache.py`) for this video, loaded with `load_detections`. Classes come from the cache meta `classes` (fine-tuned ids `k → classes[k-1]`) or COCO names (baseline), through the same `make_namer` logic as `baseline_track.py`.

For frame `i`, take **only** `cached[i]`, the boxes whose cache `frame_idx == i`. Never use a neighbouring frame and never use tracks-CSV boxes (those are thresholded and court-filtered, which would leave people unmasked). The builder asserts `len(cached) == n_decoded_frames`, and refuses to run otherwise.

| Rule | Value (**T**) |
|---|---|
| Classes masked | person, player, referee, ball, sports ball (`PEOPLE_NAMES ∪ BALL_NAMES` from `src/detection/filtering.py`) |
| Classes **not** masked | **rim**. The rim and backboard are rigid scene structure, and their texture is exactly where accuracy matters. The net sways, but it is a few dozen pixels and RANSAC rejects it. |
| Confidence floor | `mask_conf_min = 0.25`. A lower floor masks more, which is the safe direction. If the cache floor is higher, use the cache floor and record it in the meta. |
| Dilation, each side | `0.15 · w + 8 px` |
| Dilation, top | `0.10 · h + 8 px` |
| Dilation, bottom | `0.25 · h + 8 px`. Glossy gym floors show a player's **reflection** below the feet, and it moves with the player. |
| Frame border | 16 px on every side (encoder and rolling-shutter edge artefacts) |
| Overlay rectangles | `overlay_boxes_px`: zero or more `(x1, y1, x2, y2)` from the CLI (`--overlay-mask`), for burned-in score bugs and watermarks. Pixels fixed to the screen pull the estimate toward identity. |

The mask is a full-resolution `uint8` image (255 = usable) built from integer-rounded, clipped rectangles. It is resized to processing scale with `INTER_NEAREST` and passed as `mask=` to `detectAndCompute`.

**Overlay detector (warning only):** in an `ok` frame where the estimate moves the frame centre by more than 5 px, count the ratio-test matches with displacement under 0.5 px. If more than 15% (**T**) of matches are like that in more than 20 frames, warn: "possible burned-in overlay at <bounding box of those matches>; pass --overlay-mask".

---

## 3. Drift control, keyframes and the reference

### 3.1 Pass 1: forward estimation (one decode)

State: a list of **keyframes**, each with its features (keypoints in full-resolution px plus descriptors, built under its own frame's mask) and an anchored `H_k_to_ref0`, where `ref0` is the first keyframe. Also kept: the previous frame's features and `H_{i-1}_to_ref0`.

For each frame `i`:

1. Detect features under frame `i`'s mask.
2. **Consecutive estimate** `H_i_to_{i-1}`: match to frame `i-1`. It is valid if it has at least 40 inliers. The prediction is `H_pred = H_{i-1}_to_ref0 @ H_i_to_{i-1}`. If it is invalid, `H_pred = H_{i-1}_to_ref0` (assume no motion).
3. **Candidate keyframes:** for every keyframe `k`, predicted overlap = area(frame `i`'s rectangle mapped by `inv(H_k_to_ref0) @ H_pred`, intersected with frame `k`'s rectangle) / frame area. Use `cv2.intersectConvexConvex`. If any mapped corner has w ≤ 0, the overlap is 0. Keep keyframes with overlap ≥ 0.30 (**T**), sorted by overlap, at most 3 (**T**).
4. **Direct estimate:** for each candidate in order, fit `H_i_to_k`. Then `H_direct = H_k_to_ref0 @ H_i_to_k`. Run the quality tests (section 4). **Stop at the first candidate that passes.**
5. **Jump check:** if the consecutive estimate is valid, `jump_px = frame_err_px(H_direct, H_pred)` must be ≤ 6 px (**T**). If it fails, try the next candidate. This is the defence against **repeated structure**: the two ends of a gym look alike (identical baskets and banners), and a direct match to the wrong keyframe gives a confident wrong H. The chain cannot make that mistake over one frame.
6. **Outcome:**
   - Pass: `status = ok`, store `H_direct`.
   - No candidate passes: store `H_pred` (so the chain can continue), `status = unreliable`, with the reason of the best failed candidate (or `jump`).
7. **Re-anchor after a jump run:** if the jump check has failed for more than `round(0.5 · fps)` consecutive frames, and over that run the direct estimates agree with **each other** frame to frame (`frame_err_px(H_direct_i, H_direct_{i-1} @ H_i_to_{i-1}) ≤ 6 px`), then the chain is what drifted. Accept the direct estimates from then on, mark the run's frames `unreliable` (`jump`), and log the warning `reanchor_after_jump at frame N`.
8. **New keyframe:** if frame `i` is `ok` and its best overlap with any keyframe is below 0.60 (**T**), frame `i` becomes a keyframe with `H_i_to_ref0 = H_direct`. A second trigger: a keyframe is also added when the best direct match passed but with fewer than 80 inliers (2 x `min_inliers`). This refreshes keyframes as lighting or crowd changes over a long game.
9. **First keyframe:** the first frame with at least 80 features after masking. Earlier frames are `unreliable` (`no_features`).

**Why this bounds drift:** a frame's error is the error of its keyframe's anchor plus one direct fit, never a sum over frames. Anchor error grows only by one fit (about 0.3-1 px) per keyframe **hop**, and a one-way 90° pan with a 0.60 overlap trigger needs about 4-6 hops. When the camera returns over ground it has seen, the overlap test selects the **old** keyframe, so the return is anchored to the original. That is loop closure without bundle adjustment.

**Not in v1: global bundle adjustment** (jointly refining all keyframe H's from keyframe-to-keyframe matches). Add it only if the A3 loop-closure error exceeds 3 px.

### 3.2 Bridging short gaps (end of pass 1)

For each run `[a, b]` of `unreliable` frames that has `ok` frames at both `a-1` and `b+1`, and length ≤ `max(1, round(0.125 · fps))` (the same 0.125 s as `ShotConfig.max_gap_s`, 3 frames at 23.951 fps):

- Every consecutive estimate `H_j_to_{j-1}` for `j` in `[a, b+1]` must be valid.
- The forward chain from `a-1` must land on `b+1`: `frame_err_px(H_chain_{b+1}, H_direct_{b+1}) ≤ 6 px`.
- If so, store the chained H for `a..b` and set `status = bridged`. Otherwise the run stays `unreliable`.

A textureless frame has no consecutive estimate either, so it can never be bridged. This is intentional (test A2-4).

### 3.3 Choosing the reference frame `R` (after pass 1, pure matrix work)

- **Candidates:** the `ok` keyframes.
- **Score:** for candidate `r`, the score is the maximum, over all keyframes `j` and over the 4 corners and the centre `p` of frame `j`, of the linear stretch `sqrt(|det J|)` of `H_j_to_r = inv(H_r_to_ref0) @ H_j_to_ref0` at `p`. A mapped point with w ≤ 0 makes the stretch infinite.
- **R** = the candidate with the smallest score. Ties go to the lower frame index. A user override is `MotionConfig.reference_frame` (`--reference-frame N`). It must name a frame that is `ok`. If it is not a keyframe, it is used directly through its own `H`.
- **Re-express every frame:** `H_i_to_R = inv(H_R_to_ref0) @ H_i_to_ref0`, then normalise. No re-estimation is needed.

This picks roughly the middle of the pan. Frame 0 is the worst candidate on XbotGo (1.1). **Reference coordinates are the raw pixels of frame R.** That is why a calibration can name its frame, and why an old static calibration (clicked on `frame_index`) can be carried into reference coordinates exactly (section 6).

### 3.4 Canvas and mosaic (for a camera that pans end to end)

Reference coordinates are unbounded floats. The pipeline never needs an image of them; it composes matrices. A canvas is needed only to **show** the gym for calibration.

- **Extent:** the bounding box of every `ok` or `bridged` frame's corners mapped to reference coordinates, **clipped** to the region where the local linear stretch relative to R is ≤ 4 (**T**; about 60° off-axis). The width is then capped at 8192 px and the height at 4096 px (**T**) by a uniform scale.
- **Canvas transform:** `M_ref_to_mosaic = [[m, 0, tx], [0, m, ty], [0, 0, 1]]`, with `m ≤ 1`. It is stored in the motion file. Mosaic pixel `= M_ref_to_mosaic @ ref`. A click maps back with `inv(M)`.
- **Rendering** (`--mosaic`, a second decode reading only the keyframes):
  - Warp each keyframe with `M @ H_k_to_R` (`cv2.warpPerspective`, `INTER_LINEAR`).
  - Each canvas pixel takes the keyframe whose mapped centre is **nearest**. A seam-free blend is not needed.
  - Pixels that are masked (people or ball) in that keyframe fall through to the next-nearest keyframe that has the pixel unmasked. This removes most players from the mosaic.
  - Write `data/frames/<stem>_mosaic.jpg` (JPEG quality 90). `data/frames/*` is git-ignored. The builder asserts the output path resolves under `data/frames/` or `runs/` before writing.
- **Consequence on XbotGo:** the reference sits mid-pan, about 44° from each end. **The rims and the baseline corners sit about 75-80° off R's axis**, where the stretch is 20-27x in x and only 4-5x in y. They are outside the clipped mosaic, and they would be anisotropically smeared in any case. This drives two decisions:
  1. **Landmarks can be clicked on the mosaic or on any `ok` frame.** A click on frame `f` at raw `p` is stored as `H_f_to_R @ p`. Both kinds of click end up in reference coordinates, and the calibration records where each was clicked.
  2. **Rims are always clicked on a raw frame** (`view_frame`), never on the mosaic (section 5.2).

**Alternatives considered:**
- A cylindrical or spherical canvas. It shows a wide pan nicely, but floor → feet is no longer a homography from it, and it needs f. It could be a display aid later.
- One reference per half of the pan. More files and more confusion, and it is unnecessary once the rim tests run in rim views.

---

## 4. Quality and failure

### 4.1 Per-frame numbers stored

| Field | Meaning |
|---|---|
| `n_inliers` | RANSAC inliers of the accepted fit (or of the best failed fit) |
| `inlier_ratio` | inliers / ratio-test matches |
| `rmse_px` | inlier reprojection RMSE, full-resolution px, measured in the keyframe's pixels |
| `coverage` | fraction of a 4x4 grid over frame `i` with at least 1 inlier. Stops a fit that rests on one banner. |
| `jump_px` | `frame_err_px(direct, chained prediction)`, or NaN when there is no valid chain |
| `keyframe` | frame index of the keyframe matched, or -1 |
| `conf` | `min(1, n_inliers/120) · min(1, inlier_ratio/0.5) · clip((2.5 − rmse_px)/1.5, 0, 1)` (**T**), for display and reports only. **No decision reads `conf`.** |
| `status` | 0 = `ok`, 1 = `bridged`, 2 = `unreliable` |
| `reason` | 0 none, 1 `no_features`, 2 `few_inliers`, 3 `low_ratio`, 4 `high_rmse`, 5 `low_coverage`, 6 `jump`, 7 `out_of_canvas`, 8 `before_first_keyframe` |

### 4.2 Definition of reliable (`ok`)

All of the following hold, each **T**:

- `n_inliers ≥ 40`. The probe minimum was 65 at half resolution.
- `inlier_ratio ≥ 0.30`
- `rmse_px ≤ 1.5`
- `coverage ≥ 0.25` (4 of 16 cells)
- `jump_px ≤ 6` (when a valid chain exists)
- all 4 corners of frame `i` map into R with w > 0 and a linear stretch ≤ 50 (**T**). Otherwise the reason is `out_of_canvas`.

`bridged` is defined in section 3.2. Everything else is `unreliable`.

### 4.3 What happens downstream (in `detect_shots`, motion path only)

- **Ball detections in `unreliable` frames are dropped** before trajectory cleaning, so the frame becomes a gap. A position mapped through a bad H is a fabricated position.
- **An attempt is forced to `unknown`** when any frame from `release_frame − takeoff_window_s · fps` to its outcome frame (the take-off window, the flight and the outcome window) is `unreliable`. It gets `result = unknown`, note `camera_unreliable`, `result_conf = 0.0`. Shooter and points are still computed if their own frames are fine.
- **Points are empty** (note `camera_unreliable`) if the take-off frame used for the feet is `unreliable`.
- **Bridged frames** are used. If a bridged frame is the feet frame, `points_conf ≤ 0.6`. If one is in the flight or outcome window, `attempt_conf` and `result_conf` are capped at 0.6. The note is `camera_bridged`. This mirrors how design 004 treats interpolated ball points.
- **Debug JSON gains:**
  - `motion_file`, `reference_frame`
  - `n_frames_ok`, `n_frames_bridged`, `n_frames_unreliable`
  - `unreliable_runs_over_1s`: a list of `[start_frame, end_frame, start_s, end_s]`
  - `n_ball_dets_dropped_camera`
  - `n_flights_lost_camera`: flights whose segment ended at or next to an unreliable frame
- **Rows are not emitted for flights that vanished in an unreliable stretch.** An `unknown` row for a maybe-attempt would hurt attempt precision. The counters make the loss visible (design 004 risk 3), and B1 reports them.

New note codes, added to design 004 section 8.2's list: `camera_unreliable`, `camera_bridged`, `camera_static_mismatch` (section 7).

---

## 5. How the rest of the pipeline uses it

### 5.1 Which space each test runs in

| Test (design 004 section) | Space | Why |
|---|---|---|
| Held (4.3), shooter vote (4.5), `feet_cut_off` (4.5) | **raw frame pixels** of that frame | These compare the ball and the boxes **in the same frame**. Camera motion does not affect them, and raw pixels avoid any reference distortion. |
| Ball trajectory cleaning, flights, rise rule, apex, attempt zone, make/miss, rim drift check (4.1-4.4) | **rim view** of the rim being tested | These compare positions **across frames** and against the rim box, in rim widths `W`. They need a frame-independent space with no anisotropic stretch near the rim. |
| Feet → feet in court coordinates, 2/3, free-throw stationarity (4.6, 4.7) | **court feet** via `H_i_to_ft = H_ref_to_ft @ H_i_to_R` | The floor is a plane, so this is exact projectively, whatever the stretch. |

### 5.2 Rim views (the key decision)

The global reference R is distorted where the rims are: 20x stretch in x against 5x in y at about 78° off-axis on XbotGo. The engine's rules mix x and y thresholds in one unit `W` (for example `zone_above_w = 4 W` vertically while `W` is measured horizontally). In R near a rim, the vertical rules would be about 4x too loose. **So rim tests do not run in R.**

Each rim has a `view_frame`, the frame on which its box was clicked. Choose an `ok` frame where that rim is near the horizontal centre; the calibration tool suggests one (section 6.2).

- The **rim view** of rim `X` is the raw pixel space of `view_frame_X`. The rim box `box_px` is in those raw pixels, unchanged in meaning from design 004.
- Frame `i` → rim view: **`H_i_to_view = inv(H_view_to_R) @ H_i_to_R`** (`MotionTrack.H_between(i, view_frame)`).
- **Ball point:** the box centre `((x1+x2)/2, (y1+y2)/2)` in raw px, mapped by `H_i_to_view`. **Never map the box corners.**
- **Ball width** (for the depth check, 004 section 4.4 step 3): raw width × `sqrt(|det J|)` of `H_i_to_view` at the raw ball centre (formula in 1.5).
- **Feet point:** the bottom-centre `((x1+x2)/2, y2)` in raw px.
  - For court feet: `H_ref_to_ft @ H_i_to_R`.
  - For the take-off choice ("largest y2" in 004 section 4.5): rank frames by the **y of the feet point in the attacked rim's view**, not raw `y2`. Camera tilt (up to 10 px/frame measured) must not decide the take-off frame.
- **Held test with view trajectories:** the trajectory point at frame `i` (real or interpolated) is mapped back to raw with `inv(H_i_to_view)` and tested against frame `i`'s raw boxes. For a real detection this returns its raw centre exactly.
- **Rim detections** (drift check): the centre is mapped raw → rim view. A detection is assigned to the rim whose reference-space centre is nearest, if that is within 3 x the rim's local width in R. Compare in the rim view against the calibrated centre. This turns the drift check into a **live check of the motion compensation itself**.

### 5.3 `detect_shots` structure with motion

- **No `motion`:** the existing code runs as today, untouched.
- **With `motion`**, the work splits into four steps:
  1. **Per rim X:** build ball points in X's view (dropping unreliable frames), run `clean_ball_track` with `ref_w = W_X`, compute `held` (raw, as above), and run a factored-out **candidate finder**. The finder is today's flight loop up to and including the per-rim apex, zone and distance tests, with no lockout and no `judge_outcome`. It returns candidates `(release_frame, s_frame, rim X, zone index, d_W)`.
  2. **Merge** across rims. Candidates whose `release_frame` values are within 1 frame are the same flight. Keep the smaller `d_W`, the existing tie-break of 004 Q24.3.
  3. In `release_frame` order, apply the lockout, then `judge_outcome` on the winning rim's view trajectory, then shooter and points. Then apply the section 4.3 camera rules.
  4. Sort rows and assign `shot_id` as today.
- **Equivalence requirement:** with identity H's (a static video), every rim view equals raw pixels, the per-rim cleanings are identical, and steps 1-3 reduce exactly to today's loop. **Test A2-5 asserts the motion path on a static synthetic video gives a `shots.csv` byte-identical to the no-motion path.** If the refactor cannot meet that, keep the old loop for the no-motion path and accept duplicated code. Equivalence wins over tidiness.

### 5.4 Court calibration in reference coordinates

The fit is `H_ref_to_ft` from `homography_points` whose `px` are reference coordinates (raw pixels of R). It uses the existing `fit_homography` unchanged; it normalises points internally, so large reference coordinates are fine. The RMS limit stays 1.0 ft.

At use: `H_i_to_ft = H_ref_to_ft @ H_i_to_R`. The rim-label check (`_check_rim_labels`) maps each rim centre through `H_ref_to_ft @ H_view_to_R`.

---

## 6. Calibration file and tool

### 6.1 New optional keys (old files load unchanged)

```json
{
  "level": "nfhs",
  "court_length_ft": 84.0,
  "camera_static": false,
  "space": "reference",
  "reference_frame": 412,
  "motion_key": "3fa1c2d9",
  "homography_points": [
    {"name": "baseline_left_corner", "px": [-2210.4, 655.2], "ft": [-25.0, 0.0], "clicked_on": 233},
    {"name": "halfcourt_left_sideline", "px": [402.0, 610.5], "ft": [-25.0, 42.0], "clicked_on": "mosaic"}
  ],
  "rims": [
    {"id": "A", "box_px": [1180.0, 300.0, 1251.0, 324.0], "view_frame": 233, "source": "manual"},
    {"id": "B", "box_px": [28.0, 296.0, 99.0, 320.0], "view_frame": 655, "source": "manual"}
  ]
}
```

- `space`: `"frame"` (the default; today's meaning) or `"reference"`.
- **The calibration frame** `F`:
  - for `"reference"`, it is `reference_frame`;
  - for `"frame"`, it is the existing polygon key `frame_index` if present, else unknown.
  - All `homography_points[].px` are raw pixels of frame `F`.
- `rims[].view_frame`: defaults to `F`. `box_px` is in the raw pixels of `view_frame`.
- `motion_key`: the 8-hex key of the motion file used. If it differs from the motion file passed to `detect_shots`, warn, because the settings changed. The coordinates stay valid while `reference_frame` is the same frame, since they are that frame's raw pixels. If the motion file's `reference_frame` differs from the calibration's, **convert** the points with `H_F_to_newR` and warn. This is exact, so there is no need to refuse. The exception is frame `F` being `unreliable` in the new motion file: then raise `ConfigError`.
- `Calibration` (in `src/court/geometry.py`) gains fields with defaults, appended at the end so positional construction still works: `space: str = "frame"`, `calib_frame: int | None = None`, `motion_key: str | None = None`. `Rim` gains `view_frame: int | None = None`. `homography` stays `H_px_to_ft`, documented as "pixels of `calib_frame`".

### 6.2 `scripts/calibrate_court.py`

Today the script clicks **only the polygon**; `homography_points` and `rims` are written by hand (see Questions, item 1). Additions:

- `--motion PATH` (required with `--mosaic`), `--mosaic PATH`, and `--mode {polygon,landmarks,rims}` (default `polygon`, which keeps today's behaviour).
- `landmarks` shows the mosaic.
  - A click maps to reference coordinates by `inv(M_ref_to_mosaic)`.
  - Key `k` switches to the `ok` keyframe whose reference-space centre is nearest the cursor. A click there maps by `H_k_to_R`.
  - Each landmark is named from the section 4.1 list of design 004, with its `ft` filled in.
  - On save: fit `H_ref_to_ft`, print the RMS, and refuse above 1.0 ft (the existing rule).
- `rims` shows a raw keyframe, suggesting for each rim the `ok` frame where the rim (from cached rim detections mapped to R) sits nearest the frame centre. The user clicks the 2 box corners; they are stored with `view_frame`.
- It writes `space: "reference"`, `reference_frame`, `motion_key` and `camera_static: false`.
- Like today, it **never writes an image**. It only displays.

---

## 7. The `camera_static` flag

Definition of **measured motion**: `p99_disp_px` = the 99th percentile, over `ok` or `bridged` frames, of `frame_err_px(H_i_to_R, I)`. The camera is **static** when `p99_disp_px ≤ 3 px` (**T**). Both values are stored in the motion meta (`is_static`, `p99_disp_px`).

| `camera_static` in the calibration | `--motion` given | Behaviour |
|---|---|---|
| `true` (or absent) | no | **Unchanged from today**, byte-identical. The existing whole-clip drift check still runs. When `--motion` is not passed, `detect_shots.py` also prints a hint if `runs/cache/<stem>__motion__*.npz` exists for the tracks file's stem (`<stem>_tracks.csv`). |
| `true` | yes, `is_static` | Motion path. H ≈ identity. No warning. |
| `true` | yes, **not static** | **Loud warning**: a multi-line banner on stderr, an entry in `debug.warnings`, and note `camera_static_mismatch` on **every** row. The text: "calibration says camera_static=true but the camera moved (p99 N px); scoring WITH motion compensation, with the calibration taken as frame F". If `F` is unknown (no `frame_index` and no `reference_frame`), raise `ConfigError`: "cannot place a static calibration in a moving video; recalibrate with scripts/calibrate_court.py --mode landmarks/rims". |
| `false` | no | **Refuse to score:** `ConfigError` ("camera_static is false: run scripts/stabilize.py and pass --motion"). Exit code 2. |
| `false` | yes | Motion path, normal. |

This replaces design 004 section 10.1 change 2. The per-window drift check of 004 risk 1 is still useful on the static path and stays recommended, but it is not part of this task.

**Consistency checks** when `--motion` is given (`ConfigError` on failure):
- `|fps_tracks − fps_motion| / fps_motion ≤ 1e-3`.
- The tracks' max frame is below the motion file's `n_frames`.
- If the calibration has `frame_width`/`frame_height`, they equal the motion file's.

---

## 8. Motion cache file

### 8.1 Path and key

- The path is `runs/cache/<video_stem>__motion__<key8>.npz`, where `runs/` is git-ignored. It holds numbers only, no image.
- `key8` = the first 8 hex characters of `sha256(json.dumps(key, sort_keys=True, separators=(",", ":")))`.

```python
key = {
  "motion_version": MOTION_VERSION,             # "005.1"; bump on any algorithm change
  "opencv_version": cv2.__version__,            # SIFT output can differ across versions
  "video": {
    "stem": ..., "size_bytes": ..., "sha256": ...,   # full-file SHA-256, streamed in 8 MiB chunks
    "fps": ..., "width": ..., "height": ..., "n_frames": ...   # n_frames = frames actually decoded
  },
  "detections": {
    "file_name": ..., "sha256": ...,            # SHA-256 of the cache .npz bytes
    "meta": {...}                               # the cache's own meta record, verbatim
  },
  "settings": dataclasses.asdict(cfg),          # EVERY MotionConfig field, tuples as lists
}
```

- The full-file SHA-256 costs a few seconds per GB, negligible next to SIFT. The file name is not identity: a re-export with the same name must invalidate the cache.
- `load_motion(path, expected_key)` compares the stored key to the expected one **field by field** and raises `ValueError` naming the first mismatching field, the same style as `load_detections`.
- The `detect_shots` path calls `load_motion(path, None)`. It does not have the video, so it checks only the section 7 consistency rules, and it prints the key's video stem and the first 12 characters of the SHA.

### 8.2 Contents (`np.savez_compressed`, loaded with `allow_pickle=False`)

| Array | dtype, shape | Meaning |
|---|---|---|
| `H_to_ref` | float64 (N,3,3) | **frame → reference**, full-resolution px, `H[2,2] = 1`. Unreliable frames hold the chained guess (used only for the debug overlay). |
| `status` | int8 (N,) | 0 ok, 1 bridged, 2 unreliable |
| `reason` | int8 (N,) | codes in 4.1 |
| `n_inliers` | int32 (N,) | |
| `inlier_ratio`, `rmse_px`, `coverage`, `jump_px`, `conf` | float32 (N,) | NaN where not measured |
| `keyframe` | int32 (N,) | the keyframe matched, or -1 |
| `keyframes` | int32 (K,) | keyframe frame indices, ascending |
| `M_ref_to_mosaic` | float64 (3,3) | canvas transform (3.4) |
| `mosaic_wh` | int32 (2,) | canvas width and height |
| `meta` | 0-d str | JSON: `key`, `key8`, `reference_frame`, `fps`, `frame_wh`, `n_frames`, `is_static`, `p99_disp_px`, `f_est_px`, `rotation_residual`, `mask_conf_min_used`, `warnings`, `created_utc`, `runtime_s` |

---

## 9. API: `src/camera/motion.py`

```python
from __future__ import annotations
from dataclasses import dataclass
from enum import IntEnum
from pathlib import Path
from typing import Iterable, Sequence
import numpy as np

MOTION_VERSION = "005.1"
CACHE_DIR: Path   # same as src.detection.cache.CACHE_DIR (runs/cache)

class Status(IntEnum):
    OK = 0
    BRIDGED = 1
    UNRELIABLE = 2

class Reason(IntEnum):
    NONE = 0; NO_FEATURES = 1; FEW_INLIERS = 2; LOW_RATIO = 3; HIGH_RMSE = 4
    LOW_COVERAGE = 5; JUMP = 6; OUT_OF_CANVAS = 7; BEFORE_FIRST_KEYFRAME = 8

@dataclass(frozen=True)
class MotionConfig:
    """Every motion-estimation setting. All of them go into the cache key."""
    feature: str = "sift"                    # "sift" | "orb" (fallback, section 2.1)
    proc_long_side_px: int = 960
    sift_nfeatures: int = 2000
    sift_contrast_threshold: float = 0.02
    ratio_test: float = 0.75
    min_matches: int = 8
    ransac_thresh_px: float = 2.5            # full-resolution px
    ransac_max_iters: int = 5000
    ransac_confidence: float = 0.999
    rng_seed: int = 0
    mask_conf_min: float = 0.25
    mask_pad_x_frac: float = 0.15
    mask_pad_top_frac: float = 0.10
    mask_pad_bottom_frac: float = 0.25
    mask_pad_px: int = 8
    border_px: int = 16
    overlay_boxes_px: tuple[tuple[int, int, int, int], ...] = ()
    first_keyframe_min_features: int = 80
    keyframe_overlap_new: float = 0.60
    keyframe_overlap_min: float = 0.30
    max_keyframe_candidates: int = 3
    keyframe_refresh_inliers: int = 80
    min_inliers: int = 40
    min_inlier_ratio: float = 0.30
    max_rmse_px: float = 1.5
    coverage_grid: int = 4
    min_coverage: float = 0.25
    max_jump_px: float = 6.0
    reanchor_after_s: float = 0.5
    bridge_max_s: float = 0.125
    max_stretch_reliable: float = 50.0
    static_disp_px: float = 3.0
    mosaic_max_stretch: float = 4.0
    mosaic_max_w_px: int = 8192
    mosaic_max_h_px: int = 4096
    reference_frame: int | None = None       # force R; otherwise chosen by section 3.3

@dataclass
class MotionTrack:
    """Per-frame homographies frame -> reference (raw px of `reference_frame`) plus quality."""
    H_to_ref: np.ndarray        # (N,3,3) float64
    status: np.ndarray          # (N,) int8
    reason: np.ndarray          # (N,) int8
    n_inliers: np.ndarray
    inlier_ratio: np.ndarray
    rmse_px: np.ndarray
    coverage: np.ndarray
    jump_px: np.ndarray
    conf: np.ndarray
    keyframe: np.ndarray
    keyframes: np.ndarray
    reference_frame: int
    fps: float
    frame_wh: tuple[int, int]
    M_ref_to_mosaic: np.ndarray
    mosaic_wh: tuple[int, int]
    meta: dict

    @property
    def n_frames(self) -> int: ...
    @property
    def is_static(self) -> bool: ...                 # meta["is_static"]
    def usable(self, frames: np.ndarray) -> np.ndarray: ...       # bool: status in (OK, BRIDGED)
    def H_between(self, frame_a: int, frame_b: int) -> np.ndarray:
        """H_a_to_b = inv(H_to_ref[b]) @ H_to_ref[a], normalised."""
    def to_ref(self, frames: np.ndarray, pts_px: np.ndarray) -> np.ndarray:
        """(M,) frame indices plus (M,2) raw px -> (M,2) reference px. NaN where w <= eps."""
    def to_view(self, frames: np.ndarray, pts_px: np.ndarray, view_frame: int) -> np.ndarray:
        """Raw px of each frame -> raw px of `view_frame` (a rim view)."""
    def from_view(self, frames: np.ndarray, pts_view: np.ndarray, view_frame: int) -> np.ndarray:
        """Inverse of to_view (used by the held test)."""
    def local_scale(self, frames: np.ndarray, pts_px: np.ndarray, view_frame: int) -> np.ndarray:
        """sqrt(|det J|) of H_i_to_view at each point (ball-width correction)."""
    @classmethod
    def identity(cls, n_frames: int, fps: float, frame_wh: tuple[int, int]) -> "MotionTrack":
        """All-identity, all-OK track (tests; static equivalence)."""

def transform_points(H: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """(N,2) -> (N,2) through one H; NaN where w <= 1e-6 * ||h3||."""

def frame_err_px(H_a: np.ndarray, H_b: np.ndarray, frame_wh: tuple[int, int]) -> float:
    """max over the 4 corners and the centre of || (inv(H_b) @ H_a)(p) - p ||, in frame px."""

def box_feet_px(xyxy: np.ndarray) -> np.ndarray:    # ((x1+x2)/2, y2)
def box_centre_px(xyxy: np.ndarray) -> np.ndarray:  # ((x1+x2)/2, (y1+y2)/2)

def mask_boxes_from_detections(per_frame: Sequence["sv.Detections"], class_names: Sequence[str] | None,
                               cfg: MotionConfig) -> list[np.ndarray]:
    """Per frame, the (n,4) xyxy boxes to mask: people and ball classes, conf >= mask_conf_min.
    `class_names` is the cache meta `classes` (fine-tuned, id k -> names[k-1]) or None (COCO)."""

def build_mask(boxes_xyxy: np.ndarray, frame_wh: tuple[int, int], cfg: MotionConfig) -> np.ndarray:
    """Full-resolution uint8 mask, 255 = usable: dilated boxes, border and overlays removed."""

def estimate_motion(frames: Iterable[np.ndarray], mask_boxes: Sequence[np.ndarray], fps: float,
                    cfg: MotionConfig | None = None) -> MotionTrack:
    """Pass 1 (3.1), bridging (3.2), reference choice and re-expression (3.3), canvas extent (3.4).
    `frames` are BGR full-resolution in decode order; `mask_boxes[i]` belongs to frame i.
    Raises ValueError if the number of frames and the number of mask entries differ."""

def video_fingerprint(video: Path) -> dict: ...
def detection_cache_fingerprint(cache_npz: Path) -> dict: ...
def motion_key(video_fp: dict, det_fp: dict, cfg: MotionConfig) -> dict: ...
def motion_cache_path(video: Path, key: dict, cache_dir: Path = CACHE_DIR) -> Path: ...
def save_motion(path: Path, track: MotionTrack, key: dict) -> None: ...
def load_motion(path: Path, expected_key: dict | None = None) -> MotionTrack: ...

def render_mosaic(keyframe_images: dict[int, np.ndarray], keyframe_masks: dict[int, np.ndarray],
                  track: MotionTrack) -> np.ndarray:
    """Nearest-keyframe-centre composite on the canvas M_ref_to_mosaic (3.4). Returns BGR;
    the caller writes it, and only under data/frames/ or runs/."""
```

`src/camera/__init__.py` is new and empty. `src/camera/motion.py` imports only numpy, cv2, the standard library and `src.detection.filtering`. supervision appears only as a type hint, behind `TYPE_CHECKING`.

**`scripts/stabilize.py`:**

```
python scripts/stabilize.py --video data/raw/xbotgo.mp4 --detections runs/cache/<file>.npz
    [--mosaic] [--debug-video] [--reference-frame N] [--overlay-mask X1,Y1,X2,Y2 ...] [--force]
```

- It prints the motion file path, `key8`, `reference_frame`, ok/bridged/unreliable percentages, the longest unreliable run (frames and seconds), `p99_disp_px`, `is_static`, `f_est_px` and the warnings.
- An existing cache with a matching key is reused, unless `--force`.
- `--debug-video` writes `runs/motion/<stem>_stabilized_debug.mp4`. Each frame shows:
  - the raw frame;
  - the outline of frame R mapped into this frame (`inv(H_i_to_R)` applied to R's corners), in cyan;
  - every **stabilised rim cluster** re-projected into this frame, as a green box. These are the medians of the cached rim detections (conf ≥ 0.40) in reference coordinates, clustered by single linkage at 2 x local rim width. When a calibration is given with `--calibration`, the calibrated rims are drawn instead.
  - this frame's raw rim detections, in red;
  - the status text (`ok 412 inl 0.61 rmse 0.4 kf 233` or `UNRELIABLE few_inliers`);
  - a yellow border on keyframes.
  The visual test is that green sits on red throughout the pan.

---

## 10. Evaluation plan (A3: evaluator, no labels)

Inputs: `data/raw/xbotgo.mp4`, `runs/cache/xbotgo__wfc2f7c4e__small__thr0.25.npz` (the mask source), and the committed A0 tracks `runs/task001_eval/A0/xbotgo_tracks.csv`.

1. **Rim stability.**
   - Cluster the rim detections into rims (the single-linkage rule in section 9).
   - For each rim, take its view frame as the `ok` frame where the rim is nearest the horizontal centre.
   - Report the median absolute deviation from the cluster median and the 95th percentile of the distance to it, in px and in units of that rim's median width `W`:
     - **raw** (raw px), against
     - **stabilised** (that rim's view px; reference px as a secondary column).
   - **Pass:** stabilised MAD ≤ 0.1 W and p95 ≤ 0.5 W per rim. 0.5 W is the engine's own drift threshold.
   - If one cluster fails while coverage passes, inspect those detections. Q2 already suspected that some x = 30 detections are false positives. Report this honestly rather than filtering.
2. **Static people.**
   - **Pre-register** at least 5 track ids, chosen by eye from the debug video, of people seated or standing still for at least 3 s (bench, spectators). List them in the report **before** computing any number.
   - For each, report the feet-point (bottom-centre) spread over its frames, raw against reference px, as MAD and p95.
   - **Pass:** stabilised p95 ≤ max(6 px, 0.1 × median box height). Box jitter from the detector is included and expected.
3. **Coverage.** Report:
   - the % of frames `ok`, `bridged` and `unreliable`, and the per-reason counts;
   - the longest unreliable run in frames and seconds;
   - RMSE and inliers against pan speed (`|dx|` per frame), as a 3-bin table.
   - **Pass:** ok + bridged ≥ 95%, and the longest unreliable run ≤ 1.0 s.
4. **Loop closure (drift).** For frames after 560 (the pan back), estimate a fresh direct H to the reference frame (same features and mask, one fit), where the overlap is ≥ 0.3. Report `frame_err_px(stored, fresh)`, median and max. **Pass:** max ≤ 3 px.
5. **Diagnostics to report:** `f_est_px` (expected about 920), `rotation_residual`, the overlay warning (yes/no), the keyframe count, `reference_frame`, and runtime in ms/frame.
6. **Debug video:** watch it in full. Describe any visible jump, drift of green from red, or wobble, with frame numbers.
7. **Static control:** run `stabilize.py` on a static stretch alone (frames 359-503, cut with `--max-seconds` / a frame range, or just report those frames from the full run). Report `p99_disp_px` over the stretch. Expect it at or below 3 px.
8. Write `eval/reports/<date>-motion-compensation.md` with the commands, the tables and a verdict. No accuracy claim about shots: there are none in this clip that can be scored (design 004 Q3).

**What the synthetic unit tests (A2) must contain, so none passes trivially:**
- **Truth:** a planar textured "gym" (blurred noise plus straight lines) rendered by a virtual camera `K` (f = 900 px, 1280x720) that **rotates**: yaw from −40° to +40°, with ±3° of tilt. `H_i_to_ref_true = K R_ref R_iᵀ K⁻¹`. This includes the perspective terms; **identity and pure-translation pans are not allowed** as the main case.
- **Recovery:** over `ok` frames, median `frame_err_px` ≤ 1.0 and max ≤ 2.0.
- **Sign test:** after a known rightward yaw, a known scene point must land at the predicted reference x within 1 px.
- **Masking:** 8 highly textured blobs moving coherently at 15 px/frame, covering about 40% of the frame, with their boxes passed as `mask_boxes`.
  - With masks: the recovery bounds hold.
  - Without masks (empty boxes): `max frame_err_px > 5`. This proves the scene actually defeats an unmasked estimate, so the mask is doing work.
- **Mask frame alignment:** pass `mask_boxes` shifted by one frame. The error must rise above the correct-alignment error. This catches an off-by-one mask.
- **Drift:** a 600-frame pan of 80° out and back. Max `frame_err_px` ≤ 3 over the whole run, and the final frame (back at the start) ≤ 2 px.
- **Textureless frame:** a uniform grey frame inserted. It must be `UNRELIABLE` (`no_features` or `few_inliers`), never `BRIDGED`, and its neighbours stay `OK`.
- **Static:** 100 frames of one still image with per-frame noise. Every `frame_err_px` against identity is ≤ 0.1, `is_static` is true, and `detect_shots(..., motion=track)` gives a `shots.csv` **byte-identical** to `detect_shots(...)` on the task-004 synthetic make/miss fixtures.
- **Downstream:** a synthetic make whose flight frames are marked `UNRELIABLE` becomes `unknown` with `camera_unreliable`. The same flight with `BRIDGED` frames becomes a make capped at 0.6 with `camera_bridged`.
- **`camera_static`:** `false` with no motion raises `ConfigError`. `true` with a moving track warns and adds `camera_static_mismatch` to every row.
- **Cache key:** changing any single `MotionConfig` field, the video bytes, or the detection cache bytes changes `key8`. Loop over **every** field with `dataclasses.fields`.
- **fps:** use 23.951 and 60 (design 004 section 9).

---

## 11. Risks for Part B (the real game)

1. **Repeated structure across the gym.** Both ends look alike, so a direct match can lock onto the wrong end. The defence is the jump check against the chain (3.1 step 5) plus keyframe preselection by predicted overlap. Residual risk: a long unreliable stretch during which the chain is lost, followed by re-acquisition at the wrong end. **Detect it:** rim stability per rim in B1's run (the 10.1 numbers on game1), and the drift check in rim views flagging `rim_drift`.
2. **The camera follows play tightly, so players fill the frame.** The usable background shrinks to floor and stands, and the stands hold moving spectators. Inliers fall toward 40. **Detect it:** the per-frame inlier histogram and the unreliable share in B1. **Mitigate:** lower `proc_long_side_px` is the wrong lever; raise `sift_nfeatures` to 4000 first.
3. **`unknown` inflation.** Every attempt touching one unreliable frame becomes `unknown`. If fast pans coincide with shots (the camera swings to follow a fast break into a layup), make/miss coverage drops. **B1 must report** the share of `unknown` caused by `camera_unreliable` separately from the design 004 reasons. Do not loosen reliability on one game.
4. **The calibration is tied to one reference frame.** Re-running `stabilize.py` with new settings can pick a different R. Section 6.1 converts automatically, but it fails if frame F has become `unreliable`. **Mitigate:** the owner passes `--reference-frame` on game1 after the first run, and records it.
5. **Rims are clicked off-mosaic.** The task text says to calibrate on the mosaic. On a 90° pan the rims and baseline corners are outside the usable mosaic (3.4), so the tool must offer keyframe clicking. If the owner clicks a rim on the mosaic anyway, the box is anisotropically stretched and every W threshold is wrong. **The tool must not allow rim clicks on the mosaic.**
6. **Runtime on a full game.** About 32 minutes at 30-60 fps is 58k-115k frames. With 1-4 fits per frame, rough CPU cost is 30-60 ms/frame at 960 px, so about 0.5-2 hours. It is a one-time cached cost, but the owner should expect it. A3 measures ms/frame. The fallback is `feature="orb"` or `proc_long_side_px=720`, each changing the key.
7. **Burned-in overlays** (the XbotGo app's score bug or logo, if exported). They pull the estimate toward identity during slow pans. The overlay warning (2.3) and `--overlay-mask` cover it. Check the first game export by eye.
8. **Ball-width correction and the depth check** (design 004 risk 2). The correction in 5.2 changes the ratio by the local scale, about 1-5% between nearby frames, so it does not fix the 0.31 vs 0.35 problem. B2 still decides that on at least 10 labelled makes.
9. **Auto-exposure and LED boards.** Brightness changes during pans, and scrolling LED ribbon boards are "texture that moves". SIFT is fairly robust to the first. Ribbon boards become RANSAC outliers, unless they dominate a frame, in which case add `--overlay-mask`... except they are not fixed to the screen. Watch the per-reason counts.
10. **A camera that translates** (a parent bumps the tripod, or the mount slides on a rail). After the bump the old keyframes no longer fit: a run of `jump` or `high_rmse`, then re-anchoring onto a slightly wrong geometry. The rotation check (1.4) and the rim stability numbers expose it. It is out of scope to fix.

---

## Questions and assumptions

1. **`calibrate_court.py` clicks only the polygon today.** `homography_points` and `rims` are hand-written JSON. A2's "the court and rim are clicked once in reference coordinates" therefore means **adding** a landmark mode and a rim mode (6.2). That is more build than the brief implies. *Assumption:* it is in scope for A2, built as `--mode landmarks|rims`.
2. **The rims are clicked on raw keyframes, not on the mosaic.** This deviates from the brief's wording. *Reason:* on this clip the rims sit about 75-80° off the reference axis, where the mosaic is clipped and stretched 20x in x and 5x in y. A box clicked there would break every W-based threshold (3.4, 5.2). Landmarks may be clicked on either.
3. **Rim tests run in per-rim views, not the global reference.** This is a design decision beyond the brief's "transform to the reference". The brief's intent (one coordinate system per object, as if the camera never moved) is kept: each rim view is a fixed homography of the reference.
4. **The XbotGo camera did move** (about 88° of pan, measured this session by a throwaway probe). This answers task 004 Q29 for the existing clip. It is not yet known whether the owner's game1 will be filmed with tracking on.
5. **There is no provenance link from a tracks CSV to the detection cache.** The mask uses the cache, and scoring uses the tracks. *Assumption:* both come from the same video, enforced only by fps and frame count (section 7). A mask from a different detection run of the same video only changes mask quality, not correctness.
6. **`--motion` is explicit** in `detect_shots.py`; there is no auto-discovery, only a hint. *Reason:* the tracks CSV does not carry a reliable video identity.
7. **Flights lost inside unreliable stretches emit no row**, only counters. This favours attempt precision over recall-visibility. Revisit after B1 if the counters are large.
8. **The focal length (about 920 px) and the pan (about 88°) are probe estimates** from half-resolution consecutive homographies, assuming the principal point is at the image centre. They are diagnostics only; no decision depends on their exact values.
9. **The lens-offset parallax estimate (about 4 px per keyframe hop) assumes a 5 cm offset and 5-30 m depths.** A3 measures the real keyframe-match RMSE. If it exceeds 1.5 px routinely, raise `max_rmse_px` to 2.0 **before** game1, not after.
10. **`MOTION_VERSION = "005.1"`** must be bumped by hand on any algorithm change. The settings dict alone cannot capture code changes.
11. **No new packages:** OpenCV main (SIFT, RANSAC, `intersectConvexConvex`, `warpPerspective`) and numpy. Nothing needs adding to the questions file on dependencies.

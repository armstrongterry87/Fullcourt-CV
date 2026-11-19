# Design 005: Camera motion compensation (a virtual static camera)

Status: design for task 005, step A1. Written 2026-11-18 in an unattended run by the `architect` agent (Fable). Nothing here has been run on video. Every number marked **T** is a tuning parameter with no data behind it; all of them live in one `MotionConfig` dataclass.

**Companion document:** `docs/design/005-motion-compensation-opus-standin.md` is a second design, written by an Opus agent while Fable was briefly unavailable. It contains measurements from a read-only probe of `data/raw/xbotgo.mp4` that this document lacks (see "Probe findings" at the end). This document is the canonical design.

## 1. Problem and the real constraint

The task 004 engine scores shots against a hand-clicked rim box and one pixel-to-feet homography, so it is only correct when the camera never moves. Auto-panning cameras (XbotGo) move constantly, and the engine would then return confident wrong makes and misses. We need to map every frame into fixed coordinates so the existing engine runs unchanged.

The binding constraints:
- **No ground truth for camera motion.** Validation in Part A is physical checks only: a rim must stay put, and seated people must stay put.
- **Thin background evidence.** Gyms have blank walls, players cover much of the frame, and panning blurs the image. The estimate must fail to "unreliable", never to a wrong pose.
- **No new packages.** OpenCV and numpy only.
- **Unknown lens and pan range.** Nobody has measured how far this camera pans or what its field of view is (Q2). The design must work for a narrow pan and a wide one.

## 2. Recommendation

Estimate one homography per frame by matching **masked SIFT features directly to a small set of keyframes** (not frame to frame). Keyframes are tied to one anchor keyframe. Pixel work happens in one or more **views**, each a keyframe's own image plane extended to a mosaic canvas. The existing shot engine runs once per view, unchanged, as if each view were a static camera.

Why this, in one line each:
- **Homography:** exact for a camera that only rotates, 8 numbers, and drawable on one slide.
- **Frame to keyframe:** error does not grow with time, only with the number of keyframe hops to the anchor (a handful).
- **Views:** a single flat mosaic stretches without limit as the pan angle grows (section 6.3). Views bound that stretch at 1.5x. A narrow pan gives exactly one view, which is the task brief's "one mosaic" case.

Alternatives:
- **Rotation model with an estimated focal length and a cylindrical mosaic.** One seamless mosaic for any pan range, and only 3 numbers per frame, so it is steadier on low-texture frames. The cost is that court lines curve on the mosaic, calibration is no longer a plain homography, and a focal-length estimate becomes a new failure point. Switch if A3 shows more than 3 views, or closure error (section 6.5) above 5 px that views do not fix.
- **Per-frame rim detection drives scoring, with no motion model** (design 004's alternative). There is no drift and no mosaic. The cost is that it needs about 90% rim recall (today 27%, confidence at most 0.59) and gives no court coordinates, so no 2/3 calls or shot chart. Switch only if stabilised coverage on real games stays under 80% and the rim detector has been retrained.

## 3. Naming, direction and composition (read before writing any code)

- A homography is always named `H_<from>_to_<to>` and maps a point in `<from>` to `<to>`: `p_to ~ H_from_to_to @ p_from`, in homogeneous coordinates.
- `cv2.findHomography(src, dst)` returns `H_src_to_dst`. Per frame we call `cv2.findHomography(pts_frame, pts_key)`, giving `H_frame_to_key`.
- **Composition: the rightmost matrix is applied first.** `H_a_to_c = H_b_to_c @ H_a_to_b`.

```
H_frame_to_ref  = H_key_to_ref  @ H_frame_to_key
H_frame_to_view = H_ref_to_view @ H_frame_to_ref
H_view_to_ft    = H_ref_to_ft   @ inv(H_ref_to_view)
```

- To draw a frame onto a canvas: `cv2.warpPerspective(frame, H_frame_to_view, (canvas_w, canvas_h))`, with **no** `WARP_INVERSE_MAP` flag.
- The cache stores only `H_frame_to_ref`. Nothing stores the inverse.
- Stored matrices are scaled to unit Frobenius norm, because `H[2,2]` can approach 0 at wide angles and dividing by it is unsafe.

Spaces, with the variable suffix to use:

| Space | Suffix | Meaning |
|---|---|---|
| Raw frame | `_px` | Pixels of the video frame, as today |
| Reference | `_ref` | Pixel coordinates of the anchor keyframe, extended without bound. Used for maths only, never for drawing |
| View v | `_view` | Canvas pixels of view v. Equal to the pixels of mosaic image v |
| Court | `_ft` | Feet, as in design 004 |

## 4. Motion model

**Why a homography.** A camera that rotates about its optical centre with a fixed focal length sees the same rays before and after. Two such images are related exactly by `H = K R K^-1`, whatever the depth of the scene. No 3D structure is needed.

**When it breaks:**

| Cause | Effect | What we do |
|---|---|---|
| Zoom (optical or digital auto-zoom) | Still a homography frame to frame, but the scale changes, so rim-width units and the mosaic are wrong | Out of scope. Counted by the `zoom_suspected` flag (section 7) and warned about |
| Camera translates (handheld, or tripod moved) | Parallax: near and far objects move differently, so no single homography fits | Out of scope. Shows up as low inlier ratio and high closure error |
| Gimbal rotates about its own axis, not the lens centre | A few cm of translation. Roughly 16 px of parallax for an object 3 m away and 3 px at 15 m (assuming an 8 cm offset and a 40 degree pan) | Accepted. RANSAC locks onto the far background, where the rims are. Near-sideline feet carry extra error (section 12, risk 3) |
| Lens distortion, rolling shutter in fast pans | Residuals of a few px at the frame edges | Accepted in v1. Visible in the reprojection error |

## 5. Features, masking, matching

### 5.1 Mask (same frame only)

- The mask for frame `t` is built **only from detections whose `frame_idx == t`** in the detection cache. Each side of a match (frame and keyframe) is masked with its own frame's detections.
- **Every cached detection is masked, whatever its class and confidence.** This removes any class-id-to-name mapping that could be got wrong. Losing the small rim box costs nothing.
- Dilation: each box grows on every side by `0.15 * max(box_w, box_h) + 8 px` (**T**).
- Also masked: an 8 px frame border (**T**), and any `ignore_rects_px` given on the CLI for burned-in graphics (Q3).
- The frames are decoded **sequentially from frame 0**, never by seeking, so the frame index matches the detection cache. If the cache's `n_frames` or `fps` disagrees with the video, raise an error.
- A frame with no detections is used unmasked.

### 5.2 Features: SIFT

- `cv2.SIFT_create(nfeatures=0, contrastThreshold=0.03, edgeThreshold=10, sigma=1.6)` on the grayscale frame (**T** for the contrast threshold; the default is 0.04, lowered for plain gym walls).
- **Bucketing:** split the frame into an 8 x 5 grid and keep the 60 strongest features per cell (at most 2,400) (**T**). This stops bleachers or adverts taking every feature, and spreads the points so the homography is well conditioned.
- **Working size:** frames wider than 1,280 px are downscaled to 1,280 before detection. The result is converted back with `H_native = inv(S) @ H_work @ S`, so every stored matrix is in native pixels.
- Why SIFT over ORB or AKAZE: frames are matched to keyframes up to 40% of a frame away, under blur, so descriptor quality and sub-pixel accuracy matter more than speed. ORB is about 5 times faster but less accurate on low-texture, blurred frames. SIFT has been in OpenCV main since 4.4 and its patent expired in 2020.

### 5.3 Matcher

- `cv2.BFMatcher(cv2.NORM_L2)`, `knnMatch(k=2)`, Lowe ratio 0.75 (**T**).
- Mutual check: keep a match only if it is also the best match in the reverse direction.
- Fewer than 20 matches (**T**): no fit is attempted.

### 5.4 RANSAC

- `cv2.setRNGSeed(0)` before each call, so a re-run gives the same matrices.
- `cv2.findHomography(pts_frame, pts_key, cv2.RANSAC, ransacReprojThreshold=2.0, maxIters=2000, confidence=0.995)` (**T**). The threshold is in working-size pixels.
- OpenCV refines the result on the inliers; no extra refinement step.

## 6. Keyframes, re-anchoring, reference and views

### 6.1 Pass 1: build the keyframe graph (about 5 frames per second)

Process every `max(1, round(fps / 5))`-th frame in order.

1. The first frame with at least 300 features (**T**) becomes keyframe 0.
2. Match each later frame to the **current keyframe**.
3. **Overlap** is the fraction of the frame's area that lands inside the keyframe's image rectangle (project the 4 corners, intersect the polygons).
4. If overlap falls below 0.6 (**T**), first try existing keyframes: predict overlap with each from the last good pose, and match against the best one above 0.6. If that fit is good, it becomes the current keyframe. **This is the re-anchoring step.** A camera that pans away and comes back re-uses the old keyframes, so error does not build up over the game.
5. If no existing keyframe fits, the frame becomes a new keyframe. It must be a clean frame: quality 0.9 (section 7) and under 50% of its area masked (**T**).
6. Every successful keyframe-to-keyframe fit is stored as a graph edge with its matrix and inlier count. A new keyframe is also matched to up to 2 other overlapping keyframes, so the graph has loops to check drift with.

### 6.2 Choosing the reference

- The **anchor** is the keyframe with the smallest worst-case hop count to every other keyframe (the middle of the graph). Ties go to the lowest frame index.
- `H_key_to_ref` for each keyframe is the product along the shortest hop path to the anchor (breadth-first search; ties go to the edge with more inliers).
- A video with one keyframe has that keyframe as the anchor, and `H_key_to_ref` is the identity.
- No global bundle adjustment in v1. Closure error (6.5) says whether one is needed.

### 6.3 Views and the mosaic canvas

A flat canvas stretches content by `1 / cos^2(angle)` horizontally as the pan angle from its centre grows: 1.5x at 35 degrees, 2.4x at 50 degrees, 12x at 73 degrees, infinite at 90. A sideline camera at mid-court plausibly reaches these angles, so one canvas cannot be assumed.

- **Stretch** of a homography at a point is the largest singular value of its 2 x 2 Jacobian there. With rows `h1, h2, h3`, `w = h3 . p` and the mapped point `(x', y')`: `J = [[h1[:2] - x' * h3[:2]], [h2[:2] - y' * h3[:2]]] / w`.
- View 0 is centred on the anchor. Keyframes are visited in hop order. A keyframe joins the view where the stretch at its centre is smallest, if that is at most 1.5 (**T**). Otherwise it starts a new view centred on itself.
- A view's canvas is the bounding box of its member keyframes' projected corners, shifted by a whole-pixel translation `T_v` so the top-left is `(0, 0)`. Then `H_ref_to_view[v] = T_v @ inv(H_viewkey_to_ref)`.
- Canvas cap: 6,000 x 2,500 px (**T**). Beyond it the canvas is clipped and a warning is printed.
- A frame is **usable in view v** when the stretch of `H_frame_to_view` at the frame centre is at most 3.0 (**T**). Its points in any other view are NaN. This also removes points that would map behind the view plane.
- **Mosaic image:** for each view, the per-pixel median over its keyframes plus evenly sampled quality-0.9 frames (25 frames at most, **T**), with masked pixels left out. The median removes most players. Files: `data/frames/<stem>_mosaic.jpg` for view 0 and `data/frames/<stem>_mosaic_v<k>.jpg` for the others.

A narrow pan gives one view and one mosaic, exactly as the task brief describes. A wide pan gives about three: centre, left end, right end.

### 6.4 Pass 2: every frame

Keyframes are frozen. For each frame, in order:

1. Match to the keyframe with the highest predicted overlap (predicted from the last good pose).
2. If that fails, match to the previous good frame and compose: `H_t_to_ref = H_prev_to_ref @ H_t_to_prev`. This **chained** fallback is allowed for at most 10 frames in a row (**T**), and chained frames are never better than quality 0.6. Adjacent frames share the same blur, so this bridges fast pans.
3. If that fails too, the frame is unreliable. Each following frame is tried against every keyframe until one fits.

Unreliable frames are **never interpolated**.

### 6.5 Drift measure

When a keyframe has two different paths to the anchor, project its 4 corners through both and take the largest disagreement in pixels. This is `closure_err_px`. The maximum is stored and reported. Above 5 px (**T**) a warning is printed.

### 6.6 Static video

A video is **measured static** when it has one keyframe and, over all fitted frames, the 99th percentile of frame-corner displacement under `H_frame_to_ref` is under 3 px (**T**). Then every matrix is set to the exact identity, every quality to 0.9, and `is_static = True`. View 0's canvas is the frame itself. Scoring with this motion file must equal scoring without one.

## 7. Quality score and the definition of "unreliable"

Stored per frame: `n_matches`, `n_inliers`, `inlier_ratio` (inliers / matches), `reproj_rms_px` (over inliers), `spread_cells` (how many cells of a 4 x 3 grid hold at least one inlier, 0 to 12), `keyframe`, `chained` (chain length), and `flags`.

Quality uses the same discrete levels as design 004.

**0.0, unreliable**, if any of these hold (all **T**):
- no fit, or `n_inliers < 40`
- `inlier_ratio < 0.25`
- `reproj_rms_px > 1.5`
- `spread_cells < 5`
- sanity check fails: the projected frame outline is not convex, or its area changes by more than 4x
- jump check fails: a frame corner moves more than 12% of the frame width per 1/30 s, relative to the last good frame
- the chain limit of 10 was reached

**0.9, reliable:** matched directly to a keyframe, with `n_inliers >= 100`, `inlier_ratio >= 0.5`, `reproj_rms_px <= 1.0` and `spread_cells >= 7`.

**0.6, degraded:** everything else, including every chained frame.

For unreliable frames `H_frame_to_ref` is NaN, so no code can use it by accident.

`zoom_suspected` flag (warning only): the scale at the frame centre differs from 1 by more than 5% while the centre has moved less than 5% of the frame width. A pure pan cannot do that.

## 8. Cache format and key

Path: `runs/cache/<video_stem>__motion__<key8>.npz`, written with `np.savez_compressed` and read with `allow_pickle=False`, like the detection cache.

| Array | Shape | Content |
|---|---|---|
| `H_frame_to_ref` | (n, 3, 3) float64 | NaN where unreliable |
| `quality` | (n,) float32 | 0.0, 0.6 or 0.9 |
| `n_matches`, `n_inliers`, `keyframe` | (n,) int32 | `keyframe` is -1 when none |
| `inlier_ratio`, `reproj_rms_px` | (n,) float32 | |
| `spread_cells`, `chained`, `flags` | (n,) int16 | |
| `key_frame_idx`, `key_parent`, `key_hops`, `key_view` | (K,) int32 | |
| `key_H_to_ref` | (K, 3, 3) float64 | |
| `view_key` | (V,) int32 | Keyframe each view is centred on |
| `view_H_ref_to_view` | (V, 3, 3) float64 | Includes the canvas shift |
| `view_canvas_wh` | (V, 2) int32 | |
| `meta` | JSON string | Below |

`meta` holds:
- **Video fingerprint:** `video_stem`, file size in bytes, SHA-256 of the first 8 MiB, `n_frames`, `fps`, `frame_width`, `frame_height`. This is stronger than the detection cache, which checks only the stem and fps.
- **Detections:** SHA-256 of the detection cache file. This covers the weights, threshold and classes in one value.
- **Settings:** `dataclasses.asdict(MotionConfig)`, including `ignore_rects_px`. Because the key is built from `asdict`, a new setting can never be left out.
- `algo_version` (an integer bumped by hand when the algorithm changes) and `cv2.__version__`.
- **Results:** `is_static`, `max_closure_err_px`, and `motion_id`.

**Key:** SHA-256 of the canonical JSON of the fingerprint, detection hash, settings, `algo_version` and OpenCV version. The first 8 hex characters go in the filename. On load, **every** key field is compared for strict equality and a mismatch raises `ValueError` naming the field.

**`motion_id`:** SHA-256 of the bytes of `H_frame_to_ref` and `view_H_ref_to_view`. A calibration clicked on a mosaic stores this id. If the motion file is ever recomputed to different matrices, the calibration no longer matches and scoring refuses to run.

## 9. How the rest of the pipeline uses it

### 9.1 Transforming detections (points, never box corners)

Tracking stays in raw pixels. `detect_shots` transforms positions as its first step.

| Object | Points transformed | Rebuilt box in the view |
|---|---|---|
| Player or referee | feet `((x1+x2)/2, y2)`, head `((x1+x2)/2, y1)`, left and right at mid-height | `w' = dist(left', right')`; `x = feet'_x +/- w'/2`; `y2 = feet'_y`; `y1 = min(head'_y, feet'_y - 1)` |
| Ball, rim detection | centre, left and right mid-points, top and bottom mid-points | Centred on `centre'`, width `dist(left', right')`, height `dist(top', bottom')` |
| Calibrated rim box | none | Clicked on the mosaic, so already in view pixels |

The feet point and the ball centre are preserved exactly. The four corners of a box are never transformed.

The **feet-cut-off test** (design 004 section 4.5) compares `y2` with the raw frame height, so it is computed in raw pixels **before** the transform and passed to the engine as a set of `(track_id, frame)` pairs.

### 9.2 Calibration on the mosaic

- `scripts/calibrate_court.py --mosaic <image> --motion <npz> [--view k]` lets the owner click court landmarks and rims on a mosaic. Each rim is clicked once, on the mosaic where it sits nearest the centre.
- The court JSON gains: `"camera_static": false`, `"motion_id"`, a `"view"` integer on every homography point and rim (default 0), and `px` / `box_px` in that view's canvas pixels.
- `load_calibration` maps each landmark from view pixels to reference pixels and fits **one** `H_ref_to_ft` with the existing `fit_homography`. A landmark whose stretch in reference coordinates exceeds 4.0 (**T**) is rejected with a message to pick a landmark nearer the centre.
- The court polygon filter (`baseline_track.py --court`) works in raw pixels and must not be used on panning video in v1 (Q6).

### 9.3 Scoring

With a motion file, `detect_shots` does this for each view that holds a rim:

1. Keep rows from frames that are not unreliable and are usable in the view. Transform them (9.1).
2. Build a calibration for the view: its rims, and `H_view_to_ft` in place of `H_px_to_ft`.
3. Run the existing engine on it. No scoring rule changes.
4. Merge the rows from all views, sort by frame and renumber `shot_id`.

Rules added on top (the engine's own gap logic already turns a lost ball into `unknown`; these make the reason explicit):

- **`camera_unreliable`:** if a run of unreliable frames longer than `max_gap_s` overlaps the span from `release_frame` to the outcome frame, then `result = unknown` and `result_conf = 0`. Shorter runs are ordinary ball gaps and are handled by design 004's interpolation.
- **`camera_degraded`:** if the lowest quality in that span is 0.6, `result_conf` and `points_conf` are capped at 0.6.
- Feet are taken only from frames that are not unreliable. With none in the take-off window, `points` is empty with `camera_unreliable`.
- The median rim fallback is disabled; rims must be manual. The left/right-half drift check is replaced by: the median distance of stabilised rim detections (those within 2 W of the calibrated rim) from the calibrated centre, per 10 s window. Above 0.5 W, that window's shots get `rim_drift`.
- `shots.csv` gains two columns **at the end, only when a motion file is used**: `view_id` and `camera_quality`. The `*_px` columns are then view pixels, and the debug JSON records `coordinate_space`. Without a motion file the output is byte-identical to today.
- New debug counters: `n_unreliable_frames`, `n_unreliable_runs_over_gap`, `n_views`, `max_closure_err_px`.

## 10. The `camera_static` flag

| `camera_static` | Motion file | Behaviour |
|---|---|---|
| true or absent | none | As today, unchanged. If the key is absent, print "camera_static not set; assuming a static camera" |
| true | measured static | Proceed. Identity transform; output equals the row above |
| true | **measured moving** | **Warn loudly** on stderr and in `debug["warnings"]`. Score in raw pixels, since the calibration was clicked on a raw frame, but cap every `result_conf` and `points_conf` at 0.3 with note `camera_moved` |
| false | none | **Refuse:** `ConfigError`, exit code 2, message naming `scripts/stabilize.py` |
| false | given, `motion_id` matches | Stabilised scoring (section 9.3) |
| false | given, `motion_id` missing or different | **Refuse:** `ConfigError`: "recalibrate on the current mosaic" |

- The default stays `true` when the key is absent, so existing files keep working.
- `scripts/stabilize.py` prints whether the video measured static and which value to set.
- Calibrating on a mosaic writes `camera_static: false` and `motion_id` automatically.

## 11. API of `src/camera/motion.py`

```python
@dataclass(frozen=True)
class MotionConfig: ...            # every T above, plus ignore_rects_px: tuple[tuple[int, int, int, int], ...] = ()

@dataclass(frozen=True)
class Features:
    pts_px: np.ndarray             # (N, 2) float32, native pixels
    desc: np.ndarray               # (N, 128) float32

@dataclass(frozen=True)
class PairFit:
    H_src_to_dst: np.ndarray | None
    n_matches: int
    n_inliers: int
    inlier_ratio: float
    reproj_rms_px: float
    spread_cells: int

@dataclass(frozen=True)
class View:
    id: int
    keyframe: int
    H_ref_to_view: np.ndarray
    canvas_w: int
    canvas_h: int

@dataclass(frozen=True)
class MotionTrack:
    meta: dict
    H_frame_to_ref: np.ndarray     # (n, 3, 3), NaN where unreliable
    quality: np.ndarray            # (n,) 0.0 / 0.6 / 0.9
    stats: dict[str, np.ndarray]   # the per-frame and keyframe arrays of section 8
    views: tuple[View, ...]
    is_static: bool
    motion_id: str

    def H_frame_to_view(self, frame: int, view: int = 0) -> np.ndarray | None: ...
    def usable(self, frame: int, view: int = 0) -> bool: ...
    def to_view(self, pts_px: np.ndarray, frames: np.ndarray, view: int = 0) -> np.ndarray: ...
        # (N, 2) raw pixels + (N,) frame indices -> (N, 2) view pixels, NaN where not usable

def build_mask(frame_hw: tuple[int, int], boxes_xyxy_px: np.ndarray, cfg: MotionConfig) -> np.ndarray: ...
    # uint8, 255 = usable. boxes must be the SAME frame's detections.
def extract_features(gray: np.ndarray, mask: np.ndarray, cfg: MotionConfig) -> Features: ...
def match_homography(src: Features, dst: Features, frame_wh: tuple[int, int], cfg: MotionConfig) -> PairFit: ...
def score_quality(fit: PairFit, chained: int, cfg: MotionConfig) -> float: ...
def stretch_at(H: np.ndarray, pt: tuple[float, float]) -> float: ...
def transform_points(H_a_to_b: np.ndarray, pts_a: np.ndarray) -> np.ndarray: ...
def estimate_motion(frame_source: Callable[[], Iterator[np.ndarray]],
                    boxes_per_frame: Sequence[np.ndarray], fps: float,
                    cfg: MotionConfig | None = None, meta: dict | None = None) -> MotionTrack: ...
    # frame_source is called twice (pass 1 and pass 2); each call yields BGR frames from frame 0.
def transform_tracks(tracks: pd.DataFrame, motion: MotionTrack, view: int) -> pd.DataFrame: ...
    # section 9.1; drops rows from frames not usable in the view
def render_mosaic(frame_source: Callable[[], Iterator[np.ndarray]], boxes_per_frame: Sequence[np.ndarray],
                  motion: MotionTrack, view: int, cfg: MotionConfig | None = None) -> np.ndarray: ...
def video_fingerprint(video: Path) -> dict: ...
def motion_key(fingerprint: dict, detections_sha256: str, cfg: MotionConfig) -> str: ...
def cache_path(video: Path, key: str, cache_dir: Path = CACHE_DIR) -> Path: ...
def save_motion(path: Path, motion: MotionTrack) -> None: ...
def load_motion(path: Path, expected: dict | None = None) -> MotionTrack: ...
```

Changes elsewhere, all additive:

```python
# src/court/geometry.py
class Rim:          view: int = 0
class Calibration:  motion_id: str | None = None
def load_calibration(path: Path, cfg: ShotConfig | None = None, motion: MotionTrack | None = None) -> Calibration

# src/shots/detect.py
def detect_shots(tracks, calib, fps, cfg=None, frame_height_px=None,
                 motion: MotionTrack | None = None,
                 feet_cut: set[tuple[int, int]] | None = None) -> DetectionResult
```

CLIs:
- `scripts/detect_shots.py` gains `--motion PATH`.
- `scripts/stabilize.py` takes `--video`, `--detections` (the detection cache), `--out`, `--mosaic`, `--debug-video`, `--calibration`, `--ignore-rect`, `--max-seconds`.
- Both image-writing paths refuse any output folder that is not under `data/frames/` or `runs/`.
- The debug video draws the view outline, a border coloured by quality, and a fixed grid of reference points projected back into the frame. If stabilisation works, the grid sticks to the walls.

Tests the build must include, beyond the task list:
- A pan that mixes pan, tilt and roll, so a wrong composition order fails.
- A direction test: a known frame point must land at a known reference point.
- A moving textured blob that owns most of the features, so the unmasked estimate is wrong by more than 10 px and the masked one is right within 2 px.
- A sweep out and back, twice, with corner error at most 4 px throughout and no larger at the end of the second cycle than the first.
- A test that changes each `MotionConfig` field in turn and checks the key changes.
- A test that the mask for frame `t` is built from boxes of frame `t`.
- Two frame rates (23.951 and 60).

## 12. Evaluation plan for A3 and risks for Part B

### 12.1 A3 on `data/raw/xbotgo.mp4` (evaluator)

All thresholds are targets, not gates. The report gives raw versus stabilised side by side.

| Check | How | Target (**T**) |
|---|---|---|
| Rim stability | Take rim detections with confidence at least 0.4. Per view, cluster the stabilised centres (DBSCAN, `eps` = one median rim width, `min_samples` = 10); each cluster of 20 or more is one rim. Report the median absolute deviation and 95th percentile of distance to the cluster median, in px and in rim widths. Raw baseline: the same detections, in raw pixels | 95th percentile at most 0.5 W and at most 25% of the raw figure; MAD at most 0.15 W |
| Detector noise floor | Median frame-to-frame change of the raw rim centre across consecutive detections | Context only: stabilised spread cannot beat this |
| Static people | At least 3 seated people, **hand-picked by track id from the debug video** and listed in the report. Never selected by their stabilised spread, which would be circular. Report the feet-point MAD and 95th percentile per track, raw versus stabilised. Add 2 on-court players as a control; they must not look static | 95th percentile at most 15 px |
| Independent drift | 200 random pairs of non-unreliable frames with overlap at least 0.5. Match each pair directly and compare with the composed `inv(H_b_to_ref) @ H_a_to_ref` by corner error | Median at most 2 px, 95th percentile at most 5 px |
| Coverage | Share of frames at 0.9, at 0.6 or better, and unreliable; longest unreliable run in frames and seconds; count of runs longer than `max_gap_s` | At least 95% at 0.6 or better; longest run at most 1 s |
| Structure | Keyframes, views, largest hop count, `max_closure_err_px`, `is_static`, `zoom_suspected` count, largest stretch at each rim | Closure at most 5 px |
| Visual | Watch the debug video; describe any slide or jump of the grid, with frame numbers | |

The report must also say whether the raw rim detections at x = 30 and x = 1250 (task 004 Q2) are one rim or two, and note that a 30 s clip says nothing about drift over a full game.

### 12.2 Risks for Part B, most likely first

1. **Unreliable stretches during fast breaks.** Blur and player-filled frames cut the features exactly when shots happen. This shows up as a high `unknown` share, not as wrong answers. Check `n_unreliable_runs_over_gap` and the count of `camera_unreliable` notes against the labelled shots.
2. **Burned-in graphics or digital zoom.** A scoreboard overlay is static in the frame and votes for "no motion"; auto-zoom breaks the scale. Check the `zoom_suspected` count and look at the first frame for overlays (Q3, Q4).
3. **Near-sideline feet error.** Gimbal parallax plus mosaic stretch adds error to feet near the camera. It affects 2/3 calls near the arc only. Mitigated by the existing `near_arc` band and the 0.6 cap on degraded frames.
4. **The tripod is moved or bumped at half-time.** Every keyframe is then wrong. It shows up as a long unreliable stretch or large closure error. v1 answer: split the video there and stabilise and calibrate each part separately.
5. **Keyframes go stale over a full game** (crowd moves, lights change). Inlier ratio falls slowly over time. Plot quality against time in the B1 report.
6. **Residual anisotropy at the rims.** Views bound the stretch at 1.5x, but horizontal and vertical stretch still differ, and design 004's thresholds are all in rim widths. The ball-to-rim width ratio is unaffected (both stretch together), so B2's depth decision still stands on its own. If the largest stretch at a rim exceeds 1.25, centre a view on that rim.
7. **Run time.** Roughly 0.1 s per frame for SIFT and matching, plus the 5 Hz first pass: about 2 to 3 hours for a 40-minute game at 30 fps on a laptop CPU. This is an estimate, not a measurement.
8. **Tracking still runs in raw pixels,** so ID switches during pans can break shooter attribution. Out of scope here.
9. **Footage of minors.** The mosaic and the debug video show bench players and spectators. They are written only under `data/frames/` and `runs/`, both git-ignored (checked: `.gitignore` covers `data/frames/*`, `runs/` and `*.mp4`; `*.jpg` is **not** ignored globally, which is why the scripts must refuse other folders).

Licensing: no new dependency. SIFT ships in OpenCV main (patent expired 2020). Nothing is copied from another repo.

### 12.3 Next steps

| Step | Agent | Verify by |
|---|---|---|
| Build `src/camera/motion.py`, single-view path first, then views; the tests of section 11 | model-builder | New tests pass, the 98 existing tests still pass, and each new test fails when its fix is reverted |
| `scripts/stabilize.py`, the mosaic mode of `calibrate_court.py`, `--motion` in `detect_shots.py` | model-builder | Static synthetic video: `shots.csv` byte-identical with and without `--motion` |
| Confirm the OpenCV version in `.venv` has `cv2.SIFT_create`; licence note for the installed OpenCV wheel | researcher | One-line report with the version and licence |
| A3 report | evaluator | Table of 12.1 with reproduce commands in `eval/reports/<date>-motion-compensation.md` |
| Review against the task's A4 list plus section 3 | reviewer | No blocking issues |

## Questions and assumptions

**Q1. Multiple views instead of one mosaic.** The task brief asks for one reference view and one mosaic. **Assumption:** one reference coordinate system for the maths, but up to about three views for pixel work, because a flat mosaic stretches without limit on a wide pan. A narrow pan gives one view, matching the brief. **Why it matters:** the owner may have to click on two or three mosaic images, and each rim is scored in its own view.

**Q2. The pan range and lens field of view are unknown.** Nobody has measured them, and I did not view the footage. **Assumption:** the pan may be wide enough to need views. A3 reports the number of views and the stretch at each rim, which settles it. (Superseded in part: see the probe findings below.)

**Q3. Does the XbotGo video carry a burned-in scoreboard, logo or timestamp?** **Assumption:** no. If it does, pass `--ignore-rect` for each graphic; it is part of the cache key.

**Q4. Is digital auto-zoom or electronic stabilisation switched on in the XbotGo app?** The task says the camera pans but does not zoom. **Assumption:** true. The `zoom_suspected` count in A3 tests it.

**Q5. "Static plus measured motion: warn loudly."** **Assumption:** warn, and also cap `result_conf` and `points_conf` at 0.3 with note `camera_moved`. A warning alone would still publish 0.9-confidence results against a rim box that is in the wrong place. A static video is unaffected.

**Q6. The court polygon filter in `baseline_track.py --court` is in raw pixels.** It is wrong on panning video. **Assumption:** it is not used on panning video in v1. Filtering by court feet after stabilisation is a follow-up.

**Q7. Static snap.** **Assumption:** a video whose measured motion stays under 3 px is treated as exactly static (identity matrices). Tripod vibration below that is ignored, as it is today.

**Q8. All detections are masked, including rims.** The brief says "every person and ball box". **Assumption:** masking every class is safer than mapping class ids to names, and costs only the small rim box.

**Q9. `default camera_static = true` when the key is absent.** This keeps old files working, but `data/courts/xbotgo.json` has no such key and is probably a panning clip. **Assumption:** keep the default, print a notice, and have `stabilize.py` tell the owner which value to set.

**Q10. Two new columns at the end of `shots.csv`** (`view_id`, `camera_quality`), written only when a motion file is used. **Assumption:** acceptable, since the scorer reads the first five columns by name.

**Q11. Not read in this session:** the body of `scripts/calibrate_court.py` (function list only), `src/court/polygon.py`, and `scripts/baseline_track.py` beyond its CLI flags. The mosaic click mode is specified by behaviour, not by where it slots into that script.

**Q12. Missing information that would sharpen this design:** the camera's distance from the sideline, its lens setting (1x or ultra-wide), and whether the tripod is ever moved during a game.

## Probe findings (added by the main session from the Opus stand-in's report; not verified by the architect)

An Opus agent ran a throwaway read-only probe on `data/raw/xbotgo.mp4` and reported these. Treat them as leads to check in A3, not facts:

- The camera pans about 88 degrees in total. This answers task 004 Q29 for this clip: it is a panning clip, and the pan is wide.
- Focal length is about 920 px, so the horizontal field of view is about 70 degrees.
- Pan speed peaked at about 51 px per frame. The camera was roughly still for frames 0-160, panned fast over 160-290, was still over 290-540, then panned slowly back.
- Consecutive frames matched well during the pan: at least 65 inliers at half resolution, with fit error p95 0.83 px.
- Frame 0 is a poor reference: chained to frame 0, a point mapped to x of about 10,000 px.

Consequences for this design: they support the multi-view decision (Q1) and the middle-of-graph anchor (6.2). The Opus design also proposed (a) clicking rims on raw keyframes rather than on the mosaic, and (b) checking reliability against the frame-to-frame chain to catch the two gym ends matching each other by mistake. Neither is adopted here, but (b) is worth considering if A3 shows wrong matches between the two ends of the gym. Its other differences are in `005-motion-compensation-opus-standin.md`.

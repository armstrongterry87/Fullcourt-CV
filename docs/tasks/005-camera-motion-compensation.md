# Task 005: Camera motion compensation (a "virtual static camera") and first real scoring test

**Status:** Part A is ready now and can run unattended. Part B starts once the owner has labelled shots on real footage.
**Owner (main session):** project manager. Plan with `architect`, build with the Sonnet/Haiku agents, review with `reviewer`.

## Why

Fixed-tripod game footage is hard to find, and many target programs already own auto-panning cameras (XbotGo, Veo-style). These cameras **pan but don't zoom**. The scoring engine from task 004 assumes a static camera, and on panning video it would return confidently wrong makes and misses (task 004 Q29, risk 1).

The fix: measure the camera's motion from the background (walls, court lines, stands), and map every frame into one fixed **reference view**. Then detections, the rim and the court calibration all live in one coordinate system, as if the camera never moved. The court is calibrated once, on a mosaic of the whole visible gym.

For a camera rotating on a tripod with no zoom, frame-to-frame motion is close to a pure homography, so this is well-posed.

## Ground rules (same as task 004)

- Runs unattended. Unclear points go to `docs/tasks/005-questions.md` as a question plus the assumption made, and the work continues.
- **Do not commit.** The owner reviews and commits.
- Do not edit `data/ground_truth/`, `data/labels/`, `models/`, or earlier `runs/` outputs.
- No new packages. Use OpenCV's built-in feature matchers (ORB, AKAZE or SIFT, which have been in OpenCV main since 4.4) and numpy. If something else seems necessary, note it in the questions file.
- **Footage shows minors.** Any image or video output (mosaic, debug overlay) goes only under `data/frames/` or `runs/`, which are git-ignored. Check `.gitignore` covers it before writing.
- Additions only. Existing CLIs keep working. A static-camera video must give the same results as before.

---

## Part A: build and validate motion compensation (now, on `data/raw/xbotgo.mp4`)

### A1. Design (architect, Fable)
Add a design doc `docs/design/005-motion-compensation.md`. Cover:
- **Motion model.** Frame→reference homography. Explain why a homography (pure rotation, fixed focal length) and when it breaks (zoom, a handheld camera that moves position).
- **Features and masking.** Detect features only on the background: mask out every person and ball box (dilated) from the cached detections, so moving players don't drive the estimate. Choose the feature type, matcher and RANSAC settings.
- **Drift control.** Chaining frame-to-frame homographies drifts. Use keyframes and re-anchor to the reference (or to a keyframe already anchored) whenever there is enough overlap. Define how the reference view is chosen, and how a canvas larger than one frame (a mosaic) handles a camera that pans from one end to the other.
- **Quality and failure.** Per-frame confidence (inlier count, inlier ratio, reprojection error). Define when a frame is "unreliable", and what happens downstream: shots in unreliable frames become `unknown`, never a guess.
- **How the rest of the pipeline uses it.** Transform box points (feet = bottom-centre, ball centre, rim box) into reference coordinates. Calibrate the court on the mosaic. Have `detect_shots` read reference coordinates when a motion file exists.
- **The `camera_static` flag.** It is parsed but never used (`src/court/geometry.py:274`, task 004 Q29). Make it mean something. Static plus measured motion → warn loudly. Non-static with no motion file → refuse to score.
- Evaluation plan for Part A (below), plus the risks for Part B.

### A2. Build (model-builder, Sonnet)
- `src/camera/motion.py`: estimate per-frame homographies to the reference with masking, keyframes and quality scores. Save them as a per-video cache (keyed like the detection cache, so changing the video or the settings invalidates it).
- `scripts/stabilize.py`: computes the cache for a video. Optional flags:
  - `--mosaic` writes a mosaic image of the reference canvas to `data/frames/<stem>_mosaic.jpg`, for court calibration
  - `--debug-video` writes an overlay video to `runs/` with the reference canvas outline and the stabilised rim position
- `scripts/calibrate_court.py` accepts the mosaic image, so the court and rim are clicked once in reference coordinates.
- `scripts/detect_shots.py` / `src/shots/detect.py`: when a motion file is given, transform all positions to the reference before any scoring logic, and send low-quality frames to `unknown`.
- **Unit tests** with synthetic data: generate a planar "court" image, apply known pans (homographies), add moving foreground blobs, and check:
  - the recovered homographies match the truth within tolerance
  - masking makes the estimate ignore the blobs
  - drift over a long synthetic pan stays bounded
  - a frame with no texture is flagged as unreliable
  - static video → identity homographies, and the scoring output is unchanged

### A3. Validate on the XbotGo clip (evaluator, Sonnet)
No labels are needed. These are physical checks:
- **Rim stability.** Raw rim detections span x = 30 to 1250 px (task 004 Q2). In reference coordinates, the detections of each rim should cluster tightly. Report each rim's spread in px (median absolute deviation and 95th percentile), raw versus stabilised.
- **Static people.** Bench players and spectators who stay seated should keep near-constant feet positions in reference coordinates. Report their spread, raw versus stabilised.
- **Coverage.** The percentage of frames with a reliable homography, and the longest unreliable stretch.
- Look at the debug video yourself and describe any visible drift or jumps.
- Write `eval/reports/<date>-motion-compensation.md`: the commands to reproduce, the table, and an honest verdict.

### A4. Review (reviewer, Opus)
Watch especially for:
- the homography direction (frame→reference versus reference→frame), and composition order when chaining
- boxes transformed by their corners instead of the feet / centre points the design calls for
- the mask using detections from the wrong frame
- cache keys that miss a setting
- any image of footage written outside git-ignored folders
- tests that pass trivially (for example, an identity-only synthetic pan)

### A5. Log (report-writer, Haiku)
Update `docs/progress-log.md`, and write a morning summary at the top of `docs/tasks/005-questions.md`.

**Part A done when:** tests pass, the XbotGo report shows rim spread clearly reduced (or explains honestly why not), the review has no blocking issues, and nothing is committed.

---

## Part B: first real scoring test (after the owner supplies footage and labels)

**The owner does these by hand:**
1. Put the game video at `data/raw/game1.mp4`.
2. Log every shot with `scripts/label_shots.py --video data/raw/game1.mp4`, which writes `data/labels/game1_shots.csv`.
3. After Part A: run `scripts/stabilize.py --mosaic` on it, then calibrate the court and rim on the mosaic with `scripts/calibrate_court.py`.

**Then the agents:**
- **B1 (evaluator):** run detection → tracking → stabilisation → shot detection on `game1.mp4`, and score it with `eval/score_shots.py` against the owner's labels. Report attempt precision and recall, make/miss accuracy, 2/3 accuracy, and the points difference, plus the share of shots that came back `unknown` and why.
- **B2 (architect, Fable):** task 004 risk 2. The depth-check window `[0.35, 0.75]` is likely wrong (the XbotGo ball/rim ratio is 0.31). Once **at least 10 labelled makes** exist, decide whether to retune it or replace it with the horizontal-speed test from design doc 004 section 10. Do not tune on fewer.
- **B3 (model-builder → reviewer):** apply that decision, re-score, and report before versus after. Do not tune anything else on this single game. Record every tuned value in the report.

## Out of scope

- Zooming cameras, and handheld cameras that move position (a camera that translates breaks the homography model)
- Using the camera motion inside the tracker to reduce ID switches (a good follow-up: re-score task 003's IDF1 with motion-aware tracking)
- Retraining the detector, and the dashboard

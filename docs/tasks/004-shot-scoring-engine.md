# Task 004: Shot and scoring engine (footage-independent build)

**Status:** Ready. Designed to run unattended overnight.
**Owner (main session):** project manager. Plan with `architect`, build with the Sonnet/Haiku agents, review with `reviewer`.

## Why

Scoring is what organizers will pay for: makes and misses, 2s, 3s, free throws, and a running score. Everything so far has been detection and tracking. This task builds the scoring logic **before** we have good footage, using synthetic data and unit tests, so that when real fixed-camera footage arrives we only have to tune and evaluate it.

The XbotGo clip has exactly one shot (released at about 29.1 s, frame 697), and the clip ends with the ball still in the air. **Do not tune, test or report anything on it for scoring.** It contains no usable make/miss signal.

## Ground rules for this run (unattended)

- Nobody is available to answer questions. If something is unclear, **write the question and the assumption you made** to `docs/tasks/004-questions.md`, then continue with that assumption.
- **Do not commit.** The owner reviews the diff in the morning and commits.
- Do not edit `data/ground_truth/`, `models/`, or anything under `runs/` from earlier tasks.
- Do not install new packages. Use what is already in `requirements.txt` / `requirements-dev.txt` (numpy, opencv, pandas, scikit-learn, pytest). If something else seems necessary, note it in the questions file instead.
- Keep existing CLIs and output formats working. Additions only.
- Licences: anything borrowed from an outside project must be MIT, BSD or Apache-2.0, and named in the design doc.

## Steps

### 1. Design (architect, Fable)
Write `docs/design/004-shot-scoring.md`. Cover:
- **Event model.** Shot attempt (release frame, shooter track id, release position), outcome (make, miss, unknown), points (1, 2, 3), plus a confidence per field.
- **Algorithms**, rule-based for v1:
  - ball trajectory cleaning: smoothing, gap filling for short gaps, outlier rejection (the detector sometimes jumps to a false ball, e.g. frames 690 and 692 of the XbotGo CSV)
  - shot attempt: the ball leaves a player and rises toward a rim
  - make/miss: the ball centre passes downward from above the rim top to below the rim bottom, within the rim's horizontal span, within a short window. Everything else is a miss, or "unknown" when the ball is lost near the rim
  - shooter attribution: the nearest player to the ball at release
  - 2 vs 3: the shooter's **feet** at release, mapped to court coordinates through a homography, compared against the 3-point arc
  - free throw: the shooter is at the free-throw line, and other players are lined up along the lane and stationary. v1 may be rule-based with low confidence
- **Court geometry.** High school (NFHS), NCAA, NBA and FIBA dimensions: 3-point distance and corner distance, free-throw line, basket position from the baseline, court size. **Verify every number against an official rulebook or governing-body source and cite it.** Do not use memory.
- **Failure modes** a single cheap fixed camera will hit (occlusion at the rim, the ball behind the backboard, a far-side rim that is very small), and how each degrades: unknown, not wrong.
- **Evaluation plan.** Shot attempt precision and recall within ±1 s, make/miss accuracy, 2/3 accuracy and points error per game, plus the label format the owner will fill in by hand.
- A short survey of existing open-source basketball shot-detection projects: what is reusable, and under what licence.

### 2. Build (model-builder, Sonnet)
In `src/court/` and `src/shots/`:
- `src/court/geometry.py`: court specs per level (from the design doc's cited numbers), a homography from 4+ image↔court point pairs, image-to-court mapping, and `is_three(court_xy, level)`.
- `src/shots/trajectory.py`: cleaning and gap filling of ball positions.
- `src/shots/detect.py`: shot attempts, make/miss, shooter attribution, and points, from a tracks CSV (the existing `scripts/baseline_track.py` output format) plus a court calibration file. Output: `shots.csv`.
- A `--shots` flag (or a separate `scripts/detect_shots.py`) that runs it on an existing tracks CSV.
- **Unit tests in `tests/`** with synthetic trajectories: a clean make, a rim-out miss, an air ball, a ball lost behind the backboard (expect unknown), a corner 3, a long 2 just inside the arc, and a free throw. All tests must pass.

### 3. Labeling tool (data-engineer, Haiku)
`scripts/label_shots.py`: an OpenCV video player for logging shots by hand while watching. Space pauses, arrow keys step frames, `m` = make, `x` = miss, `2` / `3` / `f` set the shot type, `u` undoes. It saves `data/labels/<video_stem>_shots.csv` (frame, time_s, result, points, notes). Document the keys at the top of the file. This is what the owner will use on the first real game footage.

### 4. Scorer (evaluator, Sonnet)
`eval/score_shots.py`: compares a predicted `shots.csv` against a label CSV, matching attempts within ±1 s. It reports attempt precision and recall, make/miss accuracy, 2/3 accuracy, and the total points difference. Test it on synthetic predicted/label pairs, with no real footage.

### 5. Review (reviewer, Opus)
Watch especially for:
- the homography direction (image→court versus court→image) and units (feet versus metres)
- feet (box bottom-centre), not box centre, used for shooter position
- image y growing downward when testing "above the rim" and "below the rim"
- any court dimension not traceable to a cited source
- tests that pass trivially

### 6. Final check (architect, Fable)
Read the built code against the design doc. List any mismatches and the top three risks for the first real-footage test in `docs/design/004-shot-scoring.md` under "Post-build review".

### 7. Log (report-writer, Haiku)
Update `docs/progress-log.md`. Write a short morning summary at the top of `docs/tasks/004-questions.md`: what was built, which tests pass, and what needs a decision.

## Done when

- [ ] Design doc with cited court dimensions and an evaluation plan
- [ ] `src/court/geometry.py`, `src/shots/trajectory.py`, `src/shots/detect.py` built, with passing synthetic tests
- [ ] `scripts/label_shots.py` works on `data/raw/xbotgo.mp4` (open, step, label, save)
- [ ] `eval/score_shots.py` works on synthetic data
- [ ] Reviewer found no blocking issues, or they are listed in the questions file
- [ ] Nothing committed; morning summary written

## Out of scope

- Tuning on any real footage (none suitable yet)
- Per-frame homography for panning cameras (v1 is a static camera)
- Retraining the detector
- Dashboard or UI

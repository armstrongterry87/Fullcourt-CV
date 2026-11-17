# Task 004: questions and assumptions

Unattended run started 2026-11-17. Nobody was available to answer, so each item below
records a question and the assumption the work proceeded under. Each needs a decision
or a confirmation.

## Morning summary (read first)

**Needs a person, in this order:**

0. **Before filming the first game: tripod, auto-tracking OFF (Q29).** This is the step 6 review's top risk and the assumption the whole design rests on. The engine assumes a static camera, but XbotGo-type mounts track the play automatically, and the rim detections in the existing clip span x = 30 to 1250 — consistent with a camera that moved. If the first game is filmed with tracking on, the static rim box sits over empty air for much of it and the engine returns **confident wrong** makes and misses at 0.9, not unknowns. Worse, the calibration file's `camera_static` flag is parsed and **never used** (`geometry.py:274`, no reference in `detect.py`; verified), so nothing will stop it. Also: was auto-tracking on for `xbotgo.mp4`? Nobody knows, and the answer changes how the existing clip should be read.
1. **Keyboard check of `scripts/label_shots.py` (Q10).** The arrow-key codes were tested only through mocks. No window was opened. The task's "works on `data/raw/xbotgo.mp4`" box is not met until someone runs the tool in a real window.
2. **The court polygon in `data/courts/xbotgo.json` is a guessed trapezoid, not a calibration.** Nobody has checked it against the court lines, and every court coordinate and 2/3 call depends on it. It needs a human to click the lines on frame 430 with `scripts/calibrate_court.py`.
3. **NCAA and NBA court constants are unverified** (`verified = False`; Q12, Q20, Q22). NFHS (the default) and FIBA are sourced. The NCAA basket offset is an assumption that it matches NFHS. The document list at the end of design section 5 would close both in about 10 minutes. Q20 also records an unconfirmed claim that some states use a 22 ft 1¾ in three-point line, so check your gym's rules. (Correction to the brief: the court constants are Q12/Q20/Q22; Q25 is about bridged makes.)
4. **Confirm Q5.** `points` is the shot value (1, 2 or 3) on every row, misses included. Points scored is derived. The main session decided this because three deliverables depend on it. If wrong, the change is one line in each of three places.
5. **Confirm Q15.** The shooter's feet are taken at take-off, not at release, as a deliberate deviation from the brief. An airborne jumper's feet map to a point too far from the camera.
6. **Step 6 is complete** (updated by the main session after this summary was first written). Design section 10 holds the architect's post-build review: 11 build deviations accepted, 6 recommended changes, and three ranked risks for the first real footage. **Risk 1** is the static-camera assumption (item 0 above). **Risk 2 is that make recall collapses:** the depth-check window is `[0.35, 0.75]`, but the ball/rim width ratio in the XbotGo clip is **0.31** — below the window's floor (reproduced by the main session: ball 22.2 px over f680-f709 against a 71.1 px rim). If that holds on real footage, almost every make returns `depth_check_failed`. The architect also concluded the depth check **cannot catch the air ball it was written for** at a 40 ft camera distance, and recommends replacing it with a horizontal-speed test once the first 10 makes are logged. **Risk 3** is attempts vanishing silently, with no counter. None of the recommended changes were implemented: step 6 asks for the review, not the fixes.

**Highest-risk assumptions for real footage:** Q13 (a label marks the outcome frame, not release), Q16 (a net-hidden make is scored 0.6), Q25 (bridged makes may fall to `unknown` more often than Q16 expects).

**What was built:** the design doc (`docs/design/004-shot-scoring.md`), `src/court/geometry.py`, `src/shots/trajectory.py`, `src/shots/detect.py`, `src/shots/config.py`, `scripts/detect_shots.py`, `scripts/label_shots.py` and `eval/score_shots.py`. No new dependencies. No scipy (Q1). Nothing writes an image.

**Tests:** 98 passing with `.venv/Scripts/pytest.exe tests/ -q` (84 before the step 5 fixes), per the main session's run. The docs agent has no shell and did not re-run them. The builder mutation-tested the original suite (make direction, feet logic, depth check, rise rule), and two new tests were mutation-checked.

**Results: none is an accuracy claim (Q3).** All acceptance is synthetic. On the XbotGo clip, `scripts/detect_shots.py` emits zero attempts with `n_flights_lost=1`. That is the correct result: the clip ends with the ball still airborne, so the engine declined to guess. Nothing should be tuned on it.

**Step 5 review:** no blocking issues. Three fixes closed paths to confidently wrong output: miss-after-loss is capped at 0.6 and needs real detections; rim labels are checked against the homography; and `provisional_geometry` caps `points_conf` at 0.6 for NCAA and NBA. The reviewer reported 6 SHOULD FIX and 7 NIT items, and **all 13 were addressed** — see the "Step 5 review disposition" table at the end of this file for each item and how it was fixed. Nothing from the review is outstanding.

**Licence:** `roboflow/sports` is MIT, but its soccer example imports Ultralytics (AGPL-3.0). Nothing was copied, and all task 004 code is original (Q21). `aarnavshah12/shot-tracker` has no licence file. Do not copy from it.

**Housekeeping:** the second block numbered Q27 (the step 5 review fixes) is renumbered Q28, and a duplicated header was removed. Nothing is committed. Review the diff and commit by hand.

---

## Q1. May we use `scipy`? It is installed but not declared.

`scipy 1.18.1` is importable in `.venv`, but it is **not** in `requirements.txt` or
`requirements-dev.txt`. The brief says to use what is already declared there
(numpy, opencv, pandas, scikit-learn, pytest) and not to install anything new.

**Assumption:** treat `scipy` as unavailable. Trajectory smoothing and gap filling are
implemented with numpy only. If a Savitzky-Golay filter or spline interpolation would
be materially better, the design doc says so and the implementation stays numpy-based.

**Why it matters:** numpy-only smoothing is a little cruder, but it keeps the pipeline
runnable from the declared dependency set alone.

## Q2. The rim is detected in only 27% of frames, at low confidence. Should make/miss depend on it?

Measured on the committed A0 run (`runs/task001_eval/A0/xbotgo_tracks.csv`):

| | |
|---|---|
| Rim detections | 194 of 719 frames (27%) |
| Confidence | median 0.43, range 0.40-0.59 |
| Box size | ~71 x 24 px (median) |
| Centre x | median 1241.8, range 30-1250 (i.e. near both frame edges) |

The make/miss rule in the brief is defined against the rim's top, bottom and horizontal
span. With the rim visible in a quarter of frames and never above 0.59 confidence, a
per-frame rim box is a weak foundation, and the x range spanning both edges suggests
either two rims or false positives.

**Assumption:** the rim position used for scoring is a **static estimate per calibration**
(from the court calibration file, or a robust median of rim detections over a stretch
where the camera is still), not the per-frame detection. Per-frame rim boxes may refine
it when confident. v1 targets a static camera, so a static rim is consistent with that.

**Why it matters:** this changes what the court calibration file has to contain — it
needs a rim position, not just a court polygon.

## Q3. The XbotGo clip cannot validate any scoring rule. Is synthetic-only acceptance enough?

The brief is explicit that the one shot in the clip (released ~frame 697, 29.1 s) has no
usable make/miss signal because the clip ends with the ball still in the air, and that
nothing should be tuned or reported on it.

**Assumption:** all acceptance for this task is synthetic unit tests plus the review.
No accuracy claim of any kind is made. The first real-footage test is treated as
unvalidated until labelled footage exists.

**Why it matters:** "all tests pass" in this task means the logic matches the design's
intent, not that it works on video.

## Q4. Which rule level is the default for `is_three` and court specs?

The project targets youth/high-school footage, so NFHS is the natural default, but the
brief asks for NFHS, NCAA, NBA and FIBA specs.

**Assumption:** all four levels are implemented, and **NFHS is the default**, since the
stated target is youth/high-school games. The level is an explicit parameter everywhere
it matters, so nothing silently assumes one.

## Q5. (Labelling tool) Is `points` the shot value or the points scored?

The label CSV has the columns `frame, time_s, result, points, notes`. There is no separate
shot-type column, so the 2/3 information has to live in `points`. A miss still needs its
type, because 2/3 accuracy is scored on misses too.

**Assumption:** `points` is the **shot value** (1 for a free throw, 2, or 3) on every row,
including misses. Points scored is derived: `points` if `result` is `make`, otherwise 0. This
matches the "points (1, 2, 3)" in the event model. A miss therefore has `points` 2 or 3, and
the scorer must not read `points` as points scored.

**Why it matters:** `eval/score_shots.py` and the design doc both depend on this. If the
scorer treats `points` as points scored, every miss on a 3 would look like a 0 and the 2/3
accuracy would silently drop. Please confirm before the scorer is written.

## Q6. (Labelling tool) How do `result` and shot type combine into one row?

The brief leaves open whether the row is created on `m`/`x` and amended by `2`/`3`/`f`, or
the other way round.

**Assumption:** the row is created on `m` or `x`. The shot type is taken from:
- a type key pressed **just before** the result (kept as the "next shot type"), or
- a type key pressed **just after** the result, which amends that row if its type is still
  unset.
Each row takes at most one type. A type key pressed after a row is already typed sets the
type for the next shot. If no type key is pressed for a shot, its type is **2 points**.
`u` removes the last row added in this session. Rows already in an existing CSV are never
removed by undo. If nothing was added this session, `u` does nothing and prints a message.

**Why it matters:** the default of 2 is silent. A free throw or 3 that is not keyed will be
recorded as a 2. Review the rows at the end of a session, or check the summary line at the
end of the run, which counts shot values.

## Q7. (Labelling tool) Which frame does a label attach to?

**Assumption:** the frame **on screen when the key is pressed**, whether paused or playing.
`time_s = frame / fps`, with fps read from the video. During playback the loop shows a new
frame every pass, so a key lands on the frame displayed at that moment. A human reaction
time (about 0.2-0.4 s) means a label made while playing is that much **later** than the shot.
The scorer matches within +-1 s, so this is inside tolerance, but it is a systematic bias.

**Why it matters:** the owner should pause on the shot when possible. If the bias turns out to
matter, a reaction-time offset can be added as a single constant. I did not add one now.

## Q8. (Labelling tool) Append or overwrite an existing label file?

**Assumption:** **append**. If `data/labels/<stem>_shots.csv` exists, its rows are loaded and
kept, and new rows are added. The file is never overwritten. A file with the wrong header or a
bad row stops the script before any write. Every change is saved at once, through a temp file
and a rename, so a crash cannot leave a half-written CSV. Rows are written sorted by frame.

**Why it matters:** a labelling session can span several runs without losing work. The cost is
that a mistake in an earlier run can only be fixed by hand-editing the CSV, because undo does
not reach loaded rows. Remove the file to start again.

## Q9. (Labelling tool) Extra keys beyond the brief

The brief lists the keys exactly, so these are additions, each small:
- `q` or `Esc` saves and quits. The brief has no quit key and the window needs one. Closing
  the window with its X also saves.
- `Up` / `Down` jump one second (`round(fps)` frames), for speed. `Left` / `Right` step one frame.
- Keys are accepted in upper or lower case (caps lock).

`notes` is always empty from this tool. It is kept as a column for hand edits.

**Assumption:** these additions are acceptable. Remove them if the brief means exactly the
listed keys and nothing else.

## Q10. (Labelling tool) Arrow-key codes are unverified on a real window

The arrow keys are read with `cv2.waitKeyEx`. The codes used are the Windows values
(2424832 / 2490368 / 2555904 / 2621440) and the X11 values (65361-65364). I could not open a
real window in this unattended run.

**Assumption:** these codes are right for OpenCV 5.0.0 on Windows, which is what `.venv` has.
Tests drive the arrow tokens through mocked codes. Nothing proves the physical keys are read
correctly.

**Why it matters:** this is the first thing to check with a keyboard. If an arrow does nothing,
`ARROW_CODES` in `scripts/label_shots.py` needs the code that the on-screen hint can reveal.
This is an open item for a human at a keyboard. The mocked loop proves the logic, not that the
keys feel right.

## Q11. (Labelling tool) A demo CSV is in `data/labels/`, and it should be deleted

To show a real CSV, I wrote `data/labels/selftest_xbotgo_shots.csv` with the non-interactive
path. It holds two made shots at frames 2 and 26. It is **test data**, not labels.

**Assumption:** the owner deletes it before the first real session. It is named so it cannot
collide with `xbotgo_shots.csv`, which the window tool reads. It is not gitignored, and `git
status` shows it as untracked. Do not commit it.

**Why it matters:** a stray file in `data/labels/` could be mistaken for real labels by the
scorer later.

**RESOLVED by the main session, 2026-11-17.** The file has been deleted rather than left for
the morning. Its contents were two synthetic rows (`make 3` at frame 2, `make 2` at frame 26)
from the non-interactive self-test, it was untracked, and it was never committed. `data/labels/`
now holds only `.gitkeep`. No decision is needed — this is recorded only so the deletion is
visible. Regenerate it any time with the `--keys` self-test command in the step 3 report.

## Q5 RESOLVED: `points` is the shot value

Decided by the main session as project manager, since nobody was available and three
deliverables depend on it: **`points` is the shot value (1, 2 or 3) on every row, misses
included.** Points scored is derived (`points` if `result == "make"`, else 0).

The reason is that the brief requires 2/3 accuracy as a metric, and that has to be scorable
on misses. If `points` meant points scored, a missed 3 would record 0 and its shot type would
be unrecoverable. The design doc and `eval/score_shots.py` were both told to follow this.
Confirm it is what you wanted; if not, the change is one line in each of the three places.

---

# Questions from the design (step 1, architect)

The architect numbered these Q5-Q9, which collided with the labelling tool's Q5-Q11.
They are renumbered Q12-Q16 here; the content is unchanged.

## Q12. Several court dimensions are not verified from an official source.

The official NFHS, NCAA and FIBA documents are paywalled, blocked (HTTP 403), over the
fetch size limit, or image-only PDFs this environment cannot render. Section 5 of
`docs/design/004-shot-scoring.md` marks every number with a verification status
(V / S / X / D / U) and no value was filled in from memory.

The one that matters most: **the basket centre's distance from the end line for
NFHS / NCAA / NBA (63 in) rests on a secondary source only.**

**Assumption:** the US basket offset (5.25 ft) and backboard offset (4 ft) are implemented
as PROVISIONAL with `verified = False`. The owner confirms them from the five document links
at the end of section 5 (about 10 minutes).

**Why it matters:** an error of a few inches in the basket offset is much smaller than the
expected homography error, so it will not flip a 2/3 call outside the 1 ft uncertainty band.
But the brief's step 5 requires every dimension traceable to a cited source, so this blocks
that criterion until confirmed. Confirming either the basket offset or the backboard offset
confirms both, since verified NBA equipment numbers tie them together.

## Q13. What moment does a hand label mark?

**Assumption:** `m` / `x` is pressed when the result is visible at the rim, so `shots.csv`
`frame` is the **outcome frame**, not the release frame. Release is kept in separate columns
(`release_frame`, `release_time_s`).

**Why it matters:** matching on release would put every label 0.5-1.5 s late and break the
±1 s tolerance. If the owner prefers to label at release, the scorer should match on
`release_time_s` instead — a one-line change.

## Q14. Should finished shot labels live in `data/labels/` or `data/ground_truth/`?

CLAUDE.md calls `data/ground_truth/` the answer key that only the evaluator touches, but the
task brief writes shot labels to `data/labels/`.

**Assumption:** follow the task brief (`data/labels/`).

**Why it matters:** the tracking answer key from task 003 lives in `data/ground_truth/`, so
shot labels and tracking labels are currently in different places. The owner should decide
whether finished shot labels move, and whether the no-agent-edits rule extends to them.

## Q15. Jump shots: feet at release, or at take-off?

The brief says the shooter's feet **at release**.

**Assumption:** feet at **take-off** — the frame in the 0.5 s before release where the
shooter's box bottom is lowest on screen.

**Why it matters:** a jump shooter is airborne at release, and airborne feet map through the
homography to a point farther from the camera, which would bias every jump shot's court
position. The rules also value a shot by where the shooter left the floor. This is a
deliberate deviation from the brief's wording.

## Q16. A make with the ball hidden by the net.

**Assumption:** real detections above and below the rim within 0.30 s count as a **make at
confidence 0.6**, even when the frames between them are missing.

**Why it matters:** the net hides the ball for a few frames on many real makes, so the
stricter rule would turn most of them into `unknown`. This is the first rule to check against
real footage, and it is the one most likely to produce a confident wrong answer.

## Q17. Free-throw precision and recall: how are empty predicted `points` treated?

**Assumption:** over matched pairs only. Precision = (pred 1 and label 1) / (pred 1). Recall = (pred 1 and label 1) / (label 1). A prediction with empty `points` is not a free throw, so it lowers recall and not precision.

**Why it matters:** a predictor that leaves points blank scores zero FT recall, which is the honest reading, but design 8.3 does not say so.

## Q18. Attempt F1 and sample-size warning (additions to 8.3).

**Assumption:** the scorer also prints attempt F1 = 2*matched/(pred+label), and a warning when fewer than 30 labelled shots. Neither is in the 8.3 table.

**Why it matters:** the evaluator rules require F1 and a small-sample warning. The 30 threshold is arbitrary.

## Q19. Points error with empty `points` on a predicted make.

**Assumption:** counts as 0 in the predicted total (as 8.3 says) and is tallied in "makes with unknown value". A label make with empty points also counts 0.

**Why it matters:** a blank-heavy prediction understates its own points total, so read the two metrics together.

---

# Verification pass (main session, 2026-11-17)

## Q20 RESOLVED (partly): court dimensions — FIBA and NFHS are now sourced

Q12 recorded that no primary court document could be opened. That is now partly fixed.

**Method:** the research step's failed fetches had nonetheless *saved* their PDFs to the session's
`tool-results` directory, and `pdftotext` is installed locally at `/mingw64/bin/pdftotext`. The
research agent has no Bash and could not reach it. Extracting those files locally produced the text.

**Closed:**
- **FIBA, all rows S -> V**, from the primary *Official Basketball Rules 2024 v1.0a*: court 28 x 15 m
  (Art. 2.1), three-point arc 6.75 m to the outer edge with the basket point 1.575 m from the inner
  edge of the endline mid-point (Art. 2.4.5), straight segments 0.90 m from the inner edge of the
  sidelines (Art. 2.4.5), free-throw line 5.80 m / 3.60 m long (Art. 2.4.3).
- **NFHS, rows X/V*/U -> S**, from NFHS Rule 1 quoted verbatim with Section and Article numbers in a
  Draper gymnasium-equipment design guide: court 94 x 50 max and 84 x 50 ideal (S1), three-point
  19 ft 9 in with 63 in straight segments (S4 A1), free-throw line 15 ft from the backboard face (S6),
  **backboard face 4 ft from the end line (S8 A1 — this row was previously U)**, ring 18 in / 10 ft /
  6 in from the board (S10 A1, S11 A1), lane 12 ft wide (S5 A1).
- **The basket centre offset, 63 in (5.25 ft), is confirmed twice over for NFHS** — via the equipment
  chain (48 + 6 + 9) and via the three-point line's own geometry. This was the most important open
  number, since it sets the court-coordinate origin for every 2/3 call. **No longer PROVISIONAL for
  NFHS.** The free-throw line follows as 19.0 ft.

**Still open — these remain `verified = False` in the code:**
- **NCAA and NBA basket offsets.** The NCAA rule book exceeds the fetch size limit; the NCAA and NBA
  court diagrams are image-only with no text layer; no OCR tool (`pdftoppm`, `tesseract`) is
  installed. NCAA's 5.25 ft is an *assumption that it matches NFHS*, not a reading.
- NCAA court size and free-throw line.
- Whether any state has moved the NFHS boys' three-point line to 22 ft 1¾ in. Unconfirmed and
  unrefuted; the NFHS 2023-24 release does not mention the three-point line at all. This is why
  `three_point_radius_ft` is overridable per gym — **check the rule set for your own games.**
- FIBA 2026 Article 2 (effective 1 Oct 2026). The FIBA summary page lists Article 2 as revised but
  names only "more flexibility with regard to line colors and court markings"; the article text was
  not parseable. So "no geometry change reported", not verified.

**Assumption:** NFHS (the default) and FIBA are treated as sourced; NCAA and NBA carry
`verified = False` and are flagged in the code. **Why it matters:** the project targets high-school
footage, so the one level that must be right is the one that now is. Installing a PDF renderer or
OCR would close the NCAA and NBA rows in minutes.

## Q21. The `roboflow/sports` licence carve-out is confirmed and narrower than it looked

The repo is MIT, but **its soccer example imports Ultralytics (AGPL-3.0)**: `examples/soccer/main.py`
has `from ultralytics import YOLO`, and `examples/soccer/requirements.txt` lists `ultralytics`.

**Assumption:** library modules only (`sports.common.team`, `sports.common.view`, `sports.common.ball`,
`sports.configs`), never the example scripts. Whether those library modules themselves import
Ultralytics was **not** checked.

**Why it matters:** this repo already had an AGPL contamination scare. Nothing from this project has
been copied in — all task 004 code is original — but the carve-out needs to be verified before
anything is adopted later. `aarnavshah12/shot-tracker` has no licence file (its `LICENSE` 404s), so
treat it as all rights reserved and do not copy from it.

---

# Questions from the build (step 2, model-builder)

These were appended as Q17-Q21 and collided with the scorer's and the verification pass's
entries; they are renumbered Q22-Q26 here with content unchanged.

## Q22. Verified flags for derived and excerpt-sourced values.

The design status table has V / S / D / X / U / PROVISIONAL, and the "What the build uses" table gives
one `verified` per level. Mapping used in `src/court/geometry.py`: NFHS all `True` (S, or D derived from S);
FIBA all `True` (V, or D derived from V); NBA arc radius `True` (V), every other NBA value `False`;
NCAA all `False` (arc and corner are S from a search excerpt only, the rest PROVISIONAL or U).

**Assumption:** S counts as verified only for NFHS, because the design table explicitly sets NFHS `verified = True`.
The NCAA S values stay `False` because the build table says NCAA is `False`. NCAA court length and width are
unavailable (`value=None`); an NCAA gym must give `court_length_ft` in the calibration file.

**Why it matters:** the step 5 review checks that no X or U value is presented as verified; a test enforces it.

## Q23. A second recency limit on the release.

Design 4.3 says rule 1 needs a held frame before the flight, and allows a segment break between them
(release_conf 0.3). It sets no limit on how old that held frame may be.

**Assumption:** the held frame must be within `max_release_gap_s = 0.5` s of the first free frame (CHOSEN, not in
the design), otherwise a ball that reappears seconds later near the rim would be attributed to a stale holder.

**Why it matters:** it can drop an attempt whose release was hidden for more than 0.5 s.

## Q24. Make/miss deviations from design 4.4 as written.

1. The first point past the zone (the exit point) is included as evidence, for both the make test and the miss
   test. Otherwise a fast ball that jumps from the rim box to below the zone (more than W in one frame at 24 fps)
   has no "below the rim" detection at all.
2. Rule 4 (ball lost at the rim) runs before the miss rule as written, which would turn a ball seen falling
   below and outside the rim and then lost into `unknown`. Implemented: if a detection below the rim and outside
   its span was seen before the loss, it is a `miss` (such a ball cannot go in afterwards). The bounce and
   sideways-exit evidence are not accepted after a loss.
3. Two rims qualifying: the closest approach is compared in rim widths (distance / W), not raw pixels, so a
   small far rim is not favoured.

**Why it matters:** each changes a result from `unknown` to `miss` in a specific case. Check these first on real footage.

## Q25. Seeding limits bridged makes.

A segment needs 3 consecutive detections to start (design 4.2 step 3), so after the net hides the ball the
first two detections below the rim are not in the trajectory. The `bridged_crossing` make (0.6) therefore needs
three detections after the gap, and the first of them must itself be below the rim bottom (or be the first
point past the zone). The design says the gate does not apply across a gap, so I did not add a looser restart.

**Why it matters:** real net makes may fall to `unknown` more often than Q16 hopes.

## Q26. Seed choice, ghosts, homography RMS, free throw, and missing inputs.

- Seeds: when several candidate triples pass the gate, the highest total confidence wins, then the smallest
  step cost. Smallest cost alone picks a motionless false ball over the real moving one (a test showed this).
- Homography RMS is over the RANSAC inliers (all points for 4-5 pairs). With 6+ pairs RANSAC can discard bad
  landmarks silently as long as 4 remain; `n_inliers` is stored but not yet reported.
- Free throw (4.7): needs at least one other tracked player, otherwise the "70% still" test has no evidence and
  the shot is scored by the arc. "In front of" the line is taken as toward the basket.
- Rim B with no `court_length_ft` and no level default (NCAA): the frame change is impossible, so points are
  empty with note `no_homography` (the closest note code in design 8.2).
- Median rim fallback labels the left rim `A` and the right rim `B`. That is arbitrary; the homography's rim A
  frame is defined only by the calibration file, so use manual rims whenever there is a homography.
- Frame height (feet-cut-off check) comes from the calibration `frame_height`, or `--frame-height`; if neither
  exists the check is skipped.
- Untracked boxes (`track_id < 0`, as in the old A0 CSV) can vote but cannot win: the shooter is then empty.
- A flight is counted in `n_flights_lost` when it rises from a held frame and ends without being held again but
  reaches no rim's attempt zone. With no rims at all it is still counted.
- `no_rim` is never written: with no rim there is no attempt to attach it to.

## Q27. (Scorer) Invalid input rows: drop-and-count or hard-fail?

Decision (evaluator): drop-and-count, with a loud header. `eval/score_shots.py` drops rows with blank/unparseable/non-finite
`time_s`, a `result` outside make/miss (labels) or make/miss/unknown (predictions), or `points` outside 1/2/3/empty. It
counts them per side, prints an "INPUT ERRORS - INCOMPLETE READ" banner (with line numbers and reasons) above the
"CHECK FIRST" line, and exits with code 2. Clean files print one quiet line. Metric definitions are unchanged.
Reason: a hard fail mid-game loses the whole run; a banner plus nonzero exit cannot be mistaken for a complete read.
Note: a blank `result` is now invalid (it used to count as a non-make). `read_rows(path, kind="pred"|"label")`.

## Q28. Step 5 review fixes: choices made.

- Miss-after-loss is now 0.6 and needs an observed (not interpolated) point below and outside the rim.
- Rim labels vs homography: the rim centre is above the floor, so its mapped foot point is only approximate. The
  check is "nearer its own basket than the other one" (not a fixed tolerance). A mislabelled manual rim raises
  `ConfigError`; I chose raise over silent swap so the owner fixes the file. The median fallback with a
  homography raises `ConfigError`. The check is skipped when the court length is unknown (NCAA, no override).
- `provisional_geometry` (points_conf capped at 0.6) fires when basket offset or corner width is unverified, or
  the arc radius is unverified and not overridden. New codes for design 8.2: `provisional_geometry`, `no_court_length`.
- A frame-adjacent segment break now counts as a loss in `judge_outcome` and as `bridged_crossing` for a make.
  With that, the gap_n test cannot fail on gap_n alone (the segment check also catches the probe); it fails only
  if both are reverted.
- Shooter selection now uses `role_track == "player"` alone when the column exists (class ignored).

---

# Step 5 review disposition (main session)

The step 5 review found **no BLOCKING issues**, 6 SHOULD FIX and 7 NIT. **All 13 were addressed**, so
nothing from that review is outstanding. Recorded here so the disposition does not have to be
reconstructed from the diff.

| # | Finding | Disposition |
|---|---|---|
| SF1 | `judge_outcome`'s gap length unpinned by any test; re-anchor can break a segment with zero frame gap, invisible to the frame-arithmetic loss test | **Fixed.** Loss and `bridged_crossing` tests now also check `tr.seg[...] != tr.seg[...]`; the reviewer's 7-frame trajectory added as a test asserting `unknown`/`ball_lost_at_rim` at 23.951 fps. The mutation check showed the seg-break fix closes the hole *by itself*, which makes `gap_n` redundant for loss detection — flagged for step 6 |
| SF2 | Miss-after-loss returned `result_conf = 0.9` without requiring real detections | **Fixed.** `below_outside` takes `observed_only`, the loss branch passes `True`, and the branch caps at 0.6. A net-occluded make with one interpolated point outside the span now falls to `unknown` |
| SF3 | Median-fallback rim labels never checked against the homography; could publish mirrored court coordinates at 0.9 confidence | **Fixed.** New `_check_rim_labels` maps each rim centre through `H_px_to_ft` and raises `ConfigError` if a rim is nearer the other basket; the median fallback now refuses to run when a homography exists. Raises rather than auto-swapping — deliberately strict |
| SF4 | PROVISIONAL court constants produced `points_conf = 0.9` with nothing in the CSV saying so | **Fixed.** New `provisional_geometry` note caps `points_conf` at 0.6 and adds a `debug["warnings"]` line. Verified by the main session across all four levels: fires for **NCAA and NBA only**; NFHS (default) and FIBA stay uncapped |
| SF5 | The NFHS citation had no URL or filename, and the URL printed under the table is a different document | **Fixed by the main session** in design §5: publisher (Draper Athletic), document title, the five verbatim quoted rule sentences, and the honest caveat that the fetched copy lived in an ephemeral directory and is **not durably re-fetchable as cited**. Preferred fix (re-cite from the KHSAA-hosted NFHS diagram, moving the rows to V) needs a PDF renderer or OCR, neither installed |
| SF6 | The lockout test asserted nothing in the passing case (`all([])` is `True`); both scenes reused `track_id=7` | **Fixed.** Now asserts `len(res.shots) == 1 and res.debug["n_lockout_suppressed"] == 1`, and the second scene's track ids are offset by 100 |
| SF6b | `eval/score_shots.py` silently dropped rows with blank/unparseable `time_s` and accepted any `result` string — and a dropped label *raises* reported recall | **Fixed** by the evaluator: drop-and-count with an "INPUT ERRORS — INCOMPLETE READ" banner naming each bad line and reason, plus exit code 2. Verified by the main session: a clean file gives one quiet line and exit 0; three bad rows give the banner, the per-line reasons and exit 2 |
| N1-N7 | 30 fps literal in a test; dead `RIM[1] if False` debris; `outside_ok` reading backwards; per-frame `class` filtered before `role_track`; `ref_w` using the nearer rim undocumented; rim B's missing court length reported as `no_homography`; `label_shots.py` docstring example writing into the non-gitignored `data/labels/` | **All fixed.** The literal is 23.951, the debris is gone, `outside_ok` is `within_span`, the shooter filter uses `role_track` alone when present, `ref_w` is documented, `no_court_length` is a distinct code (added to design §8.2), and the docstring example writes to `runs/` |

**Test count after the fixes: 98 passing** (84 before), verified by the main session with
`.venv/Scripts/pytest.exe tests/ -q`. Both new tests were mutation-checked — they fail when their fix
is reverted.

---

# Question from the post-build review (step 6, architect)

## Q29. Was auto-tracking on for `xbotgo.mp4`, and will it be on for the first game?

The whole design assumes a static camera: one hand-clicked rim box and one homography for the
entire game. XbotGo-type mounts usually track the play automatically. The rim detections in the
existing clip span x = 30 to 1250 (Q2), which fits a camera that moved during the clip.

Two compounding problems, both verified in the code by the main session:

- The calibration file's `camera_static` flag is parsed (`src/court/geometry.py:274`) and
  **referenced nowhere** in `src/shots/detect.py` or `scripts/detect_shots.py`. Setting it to
  `false` today changes nothing.
- The drift check takes one median over the whole clip per half of the frame
  (`detect.py:161-170`), so it cannot see a camera that sits still most of the time and pans for
  some stretches.

**Assumption:** the architect proceeded as if tracking may have been on, and its recommended
checks catch the problem either way.

**Why it matters:** if tracking is on, the static rim box sits over empty air for much of the
game, and every threshold (all expressed in rim widths) moves with it. The engine then decides
makes and misses against nothing at confidence 0.9 — confident wrong answers, which is exactly
what the design was built to avoid. **Film the first game on a tripod with auto-tracking off.**
Design section 10.1 change 2 (enforce `camera_static`) and 10.2 risk 1 (a per-window drift check)
are the code-side fixes; neither was implemented in this run.

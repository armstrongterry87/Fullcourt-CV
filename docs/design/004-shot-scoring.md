# Design 004: Shot and scoring engine

Status: design for task 004, step 1. Written 2026-11-17 in an unattended run. Nothing here has been validated on video (see Q3); all acceptance is synthetic.

## 1. Problem and the real constraint

From a tracks CSV of one static camera, output each shot attempt with its outcome (make, miss, unknown), its value (1, 2, 3) and the shooter, each with a confidence.

The binding constraint is detection quality, not algorithm choice:
- The rim is detected in 194 of 719 frames (27%), confidence at most 0.59, and at both frame edges.
- The ball detector produces repeatable false positives and short gaps.
- There is no labelled footage, so no threshold below is fitted to data.

Every rule therefore has to fail to "unknown" when its evidence is missing.

## 2. Recommendation

A rule-based engine keyed on a **static, hand-calibrated rim box** and a **static homography**, with all pixel thresholds expressed in rim widths. Every rule can be drawn on one slide and defended.

**Q2 is confirmed, with one change.** The scoring rim is a static box from the calibration file. Per-frame rim detections do **not** refine it in v1: a detector that never exceeds 0.59 confidence cannot improve a hand-clicked box. They are used only as a drift check (section 4.1).

Alternatives:
- **Per-frame rim detection drives make/miss.** Needed for panning cameras. Switch when the rim detector reaches about 90% frame recall on a held-out game and the camera is not static.
- **Learned make/miss classifier on a crop around the rim.** Better at net occlusion and in-line air balls. Switch when at least 300 labelled shots exist across 3 or more games and rule-based make/miss accuracy is under 85%. It is harder to defend in a presentation.

## 3. Event model

One row per attempt in `shots.csv` (columns in section 8.2).

| Field | Meaning |
|---|---|
| `release_frame` | Last frame the ball is held before the flight |
| `shooter_track_id` | Track id of the shooter (`role_track == player`), or empty |
| ball release position | Ball box centre at `release_frame`, pixels |
| feet position | Shooter's box bottom-centre `((x1+x2)/2, y2)` at take-off (section 4.5), in pixels and feet |
| `frame` | Outcome frame: the ball at the rim (section 4.4) |
| `result` | `make`, `miss` or `unknown` |
| `points` | Shot **value**: 1, 2, 3, or empty for unknown. Never points scored |
| `rim_id` | `A` or `B` |

Confidences are four discrete levels, not probabilities: **0.9** (all evidence observed), **0.6** (one piece of evidence weak or interpolated), **0.3** (best guess), **0.0** (field is unknown or empty). The rules that set each level are in section 4.

## 4. Algorithms

Conventions:
- Pixel space is `(x, y)` from the top-left. **y grows downward, so "above the rim" means smaller y.**
- Ball position is the ball box centre. Player position is the box bottom-centre.
- `W` is the calibrated rim box width in pixels. Pixel thresholds are multiples of `W`, so they scale with resolution and with rim distance.
- Every duration is in seconds and converted with `n = max(1, round(seconds * fps))`. fps comes from the tracks CSV (`frame / time_s`) or the CLI. Never assume 30.
- Every number marked **T** is a tuning parameter with no data behind it. Put them all in one `ShotConfig` dataclass.

### 4.1 Calibration file and rim

The existing court JSON gains optional keys. `src/court/polygon.py` and its `Court` dataclass are not changed; `src/court/geometry.py` adds `load_calibration(path)`, which reads the same file. Old files still load, with no rims and no homography.

```json
{
  "level": "nfhs",
  "court_length_ft": 84.0,
  "three_point_radius_ft": null,
  "homography_points": [{"name": "baseline_left_corner", "px": [x, y], "ft": [-25.0, 0.0]}],
  "rims": [{"id": "A", "box_px": [x1, y1, x2, y2], "source": "manual"}],
  "camera_static": true
}
```

- `homography_points`: 4 or more pairs, with `ft` in the **rim A frame**: origin at the centre of the baseline under rim A, x along the baseline, y toward mid-court.
- `court_length_ft` is measured or declared per gym, because high-school courts vary (section 5). It is needed only to convert to the rim B frame: `x_B = -x_A`, `y_B = court_length_ft - y_A`.
- `three_point_radius_ft` overrides the level default when the gym's line differs.
- `rims[].box_px`: the rim ellipse's bounding box, clicked by hand on a still frame. `source` is `manual` or `median`.
- Recommended landmarks, using only numbers from section 5: the two baseline corners `(-25, 0)` and `(25, 0)`, and the half-court line at the sidelines `(-25, L/2)` and `(25, L/2)`. Add more if visible. Lane corners are not recommended because the lane width was not verified.

Fallback when no manual rim exists (`source: median`): split rim detections into left and right of the frame's vertical midline. For each side with at least 30 detections (**T**), take the median box. Accept it only if the median absolute deviation of the centre is at most 0.25 of the median width (**T**). Otherwise there is no rim on that side.

Drift check: if the median centre of that side's detections is more than 0.5 W (**T**) from the calibrated centre, log a warning and add `rim_drift` to `notes` on every shot at that rim. Results at that rim get `result_conf` capped at 0.3.

A rim with `W < 20 px` (**T**) is "small": attempts are still detected, but `result` is always `unknown` with note `rim_small`.

### 4.2 Ball trajectory cleaning (`src/shots/trajectory.py`, numpy only)

Evidence designed around: the two outliers at f690 and f692 both sit at about (83, 468), 336 px from the true ball, at confidence 0.42-0.43. But the legitimate f689 and f691 are at 0.41 and 0.48. **So a confidence threshold cannot separate them**; continuity has to. Confidence is used only as a tie-break.

Steps, in order:

1. **Speed gate.** Maximum plausible ball speed is `36 W per second` (**T**). At W = 71 px and 23.951 fps that is 107 px per frame. The f690 jump is 336 px (rejected); the f688 to f689 move is 53 px (accepted). A detection is accepted if its distance from the last accepted point is at most `max_speed * dt`.
2. **One candidate per frame.** If a frame has several ball detections, keep the one inside the gate closest to the constant-velocity prediction from the last two accepted points. Ties go to higher confidence.
3. **Seeding.** A chain starts only from 3 consecutive detections that pass the gate pairwise. This stops a ghost from anchoring the chain.
4. **Re-anchor.** If 5 consecutive detections (**T**) are rejected but pass the gate among themselves, the chain switches to them.
5. **Ghost list.** A rejected location hit 2 or more times within 0.1 W (**T**) is recorded for the clip. Detections within 0.25 W of it are dropped unless they pass the gate.
6. **Gap fill.** Gaps of at most `max_gap_s = 0.125 s` (**T**; 3 frames at 23.951 fps) are filled by linear interpolation in x and y, and flagged `interp = True`. This bridges the real 2-frame gap at f695-f696. A longer gap ends the segment; nothing is invented across it and the gate does not apply across it.
7. **Smoothing.** A centred 3-frame moving average within each segment, used **only for velocity**. All rim tests use the cleaned, unsmoothed positions, because smoothing blurs a rim bounce.

Output per frame: `x_px, y_px, interp, segment_id, vx, vy` (pixels per second).

On scipy (Q1): a Savitzky-Golay filter would preserve the arc's apex slightly better. That is not material here, since the smoothed signal is used only for the sign of velocity. Stay numpy-only.

### 4.3 Shot attempt

- **Held.** The ball is held in a frame if its centre is inside a player's box padded by 25% of box width on each side and 25% of box height on top (**T**). Use `role_track == player`, not the per-frame class. Otherwise the ball is free.
- **Flight.** A maximal run of free frames within one segment, lasting at least 0.15 s (**T**).

A flight is an attempt at rim R when all of these hold:

1. It starts from a held state. `release_frame` is the last held frame before it. This excludes rebounds and tips that start at the rim.
2. It rises on release: mean `vy < 0` over the first 0.25 s (**T**).
3. Its apex (minimum y) is at least 0.5 W above the rim top: `y_apex <= rim_top - 0.5 W` (**T**).
4. At or after the apex, a free frame lies in the attempt zone: `|x - rim_cx| <= 3 W` and `rim_top - 4 W <= y <= rim_bottom + 1 W` (**T**).

Rules 3 and 4 are tested near the rim on purpose. With one camera, image height only means physical height at the rim's own depth.

- If two rims qualify, pick the one with the smaller minimum distance to the ball.
- No new attempt may be released within 0.75 s (**T**) of the previous attempt's outcome frame.
- A flight that ends by ball loss before reaching the zone is **not** emitted. It is counted in a debug total `n_flights_lost`. The XbotGo shot is this case.

Confidence:
- `attempt_conf`: 0.9 if rules 1-4 hold on real detections; 0.6 if the zone frame or apex is interpolated.
- `release_conf`: 0.9 if the first free frame directly follows a held frame; 0.6 if a filled gap lies between; 0.3 if a segment break lies between.

### 4.4 Make, miss, unknown

Rim box: `x_l, x_r, y_top, y_bot`. The outcome window runs from the first zone frame until the ball is held again, leaves the zone, or 1.5 s (**T**) passes.

Critical box C: `|x - rim_cx| <= 1.5 W` and `y_top - 1.5 W <= y <= y_bot + 1.5 W`.

Decide in this order:

1. **No rim, or a small rim:** `unknown` (`no_rim` or `rim_small`).
2. **Make.** There are real (not interpolated) detections `a` before `b` such that:
   - `y_a < y_top` (above the rim top) and `y_b > y_bot` (below the rim bottom);
   - `t_b - t_a <= 0.30 s` (**T**);
   - y does not decrease between them by more than 0.1 W;
   - the x where the path crosses the rim centre height, linearly interpolated, lies in `[x_l + 0.1 W, x_r - 0.1 W]`;
   - `x_b` lies in `[x_l - 0.25 W, x_r + 0.25 W]`.

   `result_conf` is 0.9 if every frame between `a` and `b` is a real detection or within a filled gap. It is 0.6 if the frames between are missing for longer than `max_gap_s` (note `bridged_crossing`). The net often hides the ball for a few frames, so a strict rule would turn most real makes into unknowns.
3. **Depth check on a make.** If the ball box width at `a` divided by W is outside `[0.35, 0.75]` (**T**; the expected ratio must be measured on the first footage), the result becomes `unknown` (`depth_check_failed`). This is the only defence against an air ball that passes in front of or behind the rim on the camera's line of sight.
4. **Ball lost at the rim.** If any gap longer than `max_gap_s` occurs while the last seen position was inside C, or the segment or clip ends inside the outcome window: `unknown` (`ball_lost_at_rim` or `clip_end`).
5. **Miss.** The window was observed and one of these holds:
   - the ball goes below `y_bot` with x outside `[x_l - 0.25 W, x_r + 0.25 W]`;
   - the ball enters C and then rises by more than 0.5 W (a bounce off the rim or board);
   - the ball leaves the attempt zone sideways without going below `y_top` inside the rim span.

   `result_conf` is 0.9, or 0.6 if any frame inside C was interpolated.
6. **Otherwise** `unknown` (`no_evidence`).

Outcome frame (`frame` in the CSV): for a make, the first frame at or below the rim centre height; for a miss or unknown, the frame of closest approach to the rim centre, or the last frame seen in the zone.

### 4.5 Shooter and feet

- **Shooter.** For each of the last 0.25 s (**T**) of held frames up to `release_frame`, find the player whose box is nearest the ball centre (point-to-rectangle distance, zero inside). The shooter is the track with the most votes.
- If the winner's distance at `release_frame` exceeds 0.5 of its box height (**T**), the shooter is empty (`no_shooter`).
- `shooter_conf`: 0.9 if the winner has at least 80% of the votes; 0.6 for 50-80%; 0.3 below that.
- **Feet.** Box bottom-centre `((x1 + x2) / 2, y2)`, never the box centre. Take it at the **take-off frame**: the frame in the 0.5 s (**T**) before release where the shooter's `y2` is largest. A jump shooter is airborne at release, and airborne feet map through the homography to a point farther from the camera. The rules also value a shot by where the shooter left the floor.
- If `y2` is within 2 px of the frame's bottom edge, the feet are cut off: `points` is empty (`feet_cut_off`).

### 4.6 Two or three (`src/court/geometry.py`)

- The homography is named `H_px_to_ft`, from `cv2.findHomography(pts_px, pts_ft)`. Use least squares for exactly 4 pairs and RANSAC with a 0.5 ft threshold (**T**) for 6 or more. Apply with `cv2.perspectiveTransform`. Units are feet throughout; FIBA metres are converted once in the spec table.
- Store the reprojection RMS in feet. Above 1.0 ft (**T**), raise `ConfigError`.
- Convert feet to the frame of the rim being attacked (section 4.1).

`is_three(feet_ft, level)`, with basket centre `(0, b)`, arc radius `R` and corner half-width `c`:

```
two = |x| <= c  and  (y <= b  or  hypot(x, y - b) <= R)
three = not two
```

For NFHS, `c = R`, so the same formula covers the straight segments. A foot on the line is a two, which the `<=` gives.

- `margin_ft` is the distance from the feet to the nearest part of the line.
- `points_conf`: 0.9 if `margin_ft >= 1.0 ft` (**T**); 0.3 inside that band (`near_arc`). Cap at 0.6 if the homography RMS exceeds 0.5 ft.
- `points` is empty when there is no homography (`no_homography`), no shooter, or the feet map more than 1 ft outside the court (`feet_off_court`).

### 4.7 Free throw (low confidence in v1)

`points = 1` when all of these hold:
- the feet are within 3 ft of the court's long axis and between 1 ft in front of and 3 ft behind the free-throw line (**T**);
- the shooter's feet moved less than 1.0 ft in the 1.0 s before release (**T**);
- no other player's feet are within 4 ft of the shooter at release (**T**);
- at least 70% of the other players moved less than 1.5 ft in that second (**T**).

`points_conf` is at most 0.6, with note `ft_rule`. If only the position matches, the shot is scored by section 4.6.

The brief's "lined up along the lane" is replaced by the stationarity test, because the lane width could not be verified (section 5).

## 5. Court geometry

**Read this first.** I could not open the primary NFHS, NCAA or FIBA rule documents in this environment. Each number below carries a status:

- **V**: read this session on the governing body's own page.
- **S**: the governing body's text, but seen only as a search-result excerpt or through a mirror. Not opened directly.
- **X**: secondary source only.
- **D**: derived by arithmetic from other rows.
- **U**: not found. No value is given from memory.

`geometry.py` must carry a `source` string and a `verified` flag per value. X and U values are PROVISIONAL until the owner checks the documents listed at the end of this section.

### NBA

| Quantity | Value | Status | Source |
|---|---|---|---|
| Three-point arc radius, from the middle of the basket | 23 ft 9 in (23.75 ft) | V | NBA Rule No. 1, Section I |
| Three-point straight lines, distance from the sideline | 3 ft | V | NBA Rule No. 1, Section I |
| Corner distance | 22 ft | D | 25 ft minus 3 ft; depends on the 50 ft width |
| Free-throw line | 15 ft from the plane of the backboard face | V | NBA Rule No. 1, Section I |
| Ring inside diameter; inside edge from backboard; rim height | 18 in; 6 in; 10 ft | V | NBA Rule No. 1, Section II |
| Court size | 94 x 50 ft | X | The rule text refers to the court diagram, which is an image I could not read |
| Basket centre from the baseline | 63 in | U | On the diagram only |

Source: https://official.nba.com/rule-no-1-court-dimensions-equipment

### FIBA

| Quantity | Value | Status | Source |
|---|---|---|---|
| Court size | 28.0 x 15.0 m (91.86 x 49.21 ft), measured from the inner edge of the boundary line | **V** | Official Basketball Rules 2024 v1.0a, Art. 2.1 — primary PDF, read directly |
| Line width | 50 mm | V | FIBA Venue Guide |
| Three-point straight lines, outer edge from the inner edge of the sideline | 0.90 m | **V** | Official Basketball Rules 2024 v1.0a, Art. 2.4.5 — primary PDF, read directly |
| Three-point arc radius, to the outer edge | 6.75 m (22.146 ft), measured from the point beneath the exact centre of the basket | **V** | Same, Art. 2.4.5 |
| Basket centre from the inner edge of the mid-point of the endline | 1.575 m (5.167 ft) | **V** | Same, Art. 2.4.5 |
| Corner distance | 6.60 m (21.654 ft) | D | 7.50 m minus 0.90 m |
| Free-throw line, far edge from the inner edge of the endline | 5.80 m (19.029 ft); 3.60 m long | **V** | Same, Art. 2.4.3 — primary PDF, read directly |

**Verification update, 2026-11-17 (main session).** All FIBA rows above moved from S to **V**. The
official FIBA 2024 rules PDF defeated the research step (over the fetch size limit), but the fetch
had saved the file, and `pdftotext` is available locally at `/mingw64/bin/pdftotext`. Extracting it
yielded the Article 2 text directly. The exact sentences read:

- Art. 2.1: "dimensions of 28 m in length by 15 m in width measured from the inner edge of the boundary line".
- Art. 2.4.5: "An arc of radius 6.75 m measured from the point on the court beneath the exact centre of the opponents' basket to the outer edge of the arc. The distance of the point on the court from the inner edge of the mid-point of the endline is 1.575 m."
- Art. 2.4.3: "its furthest edge 5.80 m from the inner edge of the endline and shall be 3.60 m long".

Sources: https://venueguide.fiba.basketball/vanue-design ; https://assets.fiba.basketball/image/upload/documents-corporate-fiba-official-rules-2024-v10a.pdf (official, could not be parsed); mirror at https://www.scribd.com/document/847032324/Documents-Corporate-Fiba-Official-Rules-2024-v10a

A 2026 edition, valid from 1 October 2026, exists at https://assets.fiba.basketball/image/upload/documents-corporate-fiba-official-rules-2026-v1-0.pdf. I could not read it, so I cannot say whether Article 2 changed.

Metres to feet uses 1 ft = 0.3048 m (the definition of the international foot; not looked up this session).

### NCAA

| Quantity | Value | Status | Source |
|---|---|---|---|
| Three-point arc | 22 ft 1¾ in (22.146 ft) | S | NCAA release, 5 June 2019, via excerpt; the page now redirects |
| Corner distance | 21 ft 7⅞ in (21.656 ft) | S | NCAA.com article, via excerpt. Two secondary sites give 7¾ in and 8 in |
| Three-point line from the sideline in the corner | 40⅛ in | X | Consistent with the row above: 300 in minus 40.125 in is 21 ft 7.875 in |
| Court size | 94 x 50 ft | U | |
| Free-throw line | 15 ft from the backboard face | U | |
| Basket centre from the end line | 63 in | U | |

Sources: https://www.ncaa.org/news/2019/6/5/men-s-basketball-3-point-line-extended-to-international-distance.aspx ; https://www.ncaa.com/news/basketball-men/article/2019-09-26/biggest-rule-differences-between-ncaa-and-nba-basketball

### NFHS (the default level, Q4)

| Quantity | Value | Status | Source |
|---|---|---|---|
| Court size | Not greater than 94 x 50 ft; **ideal high school 84 x 50 ft** | **S** | NFHS Rule 1 Section 1, quoted verbatim |
| Three-point line | 19 ft 9 in (19.75 ft) radius from a point in the middle of the free-throw lane directly below the centre of the basket | **S** | NFHS Rule 1 Section 4 Article 1, quoted verbatim |
| Straight three-point segments, length from the inside edge of the end line | 63 in (5.25 ft) | **S** | Same, Section 4 Article 1 |
| Free-throw line, farthest edge from the plane of the face of the backboard | 15 ft | **S** | NFHS Rule 1 Section 6, quoted verbatim |
| **Backboard front face from the end line** | **4 ft** | **S** | NFHS Rule 1 Section 8 Article 1, quoted verbatim — *was U* |
| Ring inside diameter; upper edge above floor | 18 in; 10 ft | **S** | NFHS Rule 1 Section 10 Article 1 and Section 11 Article 1 |
| Nearest point of the inside edge of the ring, from the plane of the backboard face | 6 in | **S** | NFHS Rule 1 Section 11 Article 1 |
| Free-throw lane width | 12 ft, measured to the outside of each lane boundary | **S** | NFHS Rule 1 Section 5 Article 1 |
| Basket centre from the end line | **63 in (5.25 ft)** | **S, doubly derived** | See the two independent derivations below |
| No separate corner distance | `c = R = 19.75 ft` | D | Follows from the arc radius |

**Verification update, 2026-11-17 (main session).** These rows moved from X/V*/U to **S**, obtained by
extracting a saved fetch artifact with local `pdftotext`. That document **quotes NFHS Rule 1 verbatim
with Section and Article numbers**, which is materially stronger than the search excerpts the research
step had. It is still a secondary publisher, not the NFHS rule book (paywalled; the NFHS diagram
returns HTTP 403), so the status is S rather than V. Caution for anyone re-reading that file: it also
contains NFHS *volleyball* rules, and the volleyball court is 60 x 30 ft — do not cross-wire the two.

**Locator for the NFHS rows (so they are re-findable).** The `Source:` URL printed under this table is
an MHSAA referee story and is **not** the document these rows came from. The actual source is:

- Publisher: **Draper Athletic** (`www.draperathletic.com`)
- Document title: **"GYMNASIUM EQUIPMENT COURT DESIGN & RULES — HIGH SCHOOL BASKETBALL AND VOLLEYBALL"**
- No URL, title metadata or revision date is embedded in the PDF, and the fetched copy lived in this
  session's **ephemeral** `tool-results` directory (`webfetch-1794893931916-fw324f.pdf`), which will be
  cleaned up. **So this citation is not durably re-fetchable as it stands.**
- The quoted sentences, so the numbers can be checked against any NFHS rule book:
  - S4 A1: *"The semicircle has a radius of 19 feet 9 inches from a point in the middle of the free-throw lane directly below the center of the basket. The semicircle shall be extended with a 2-inch wide line perpendicular to the end line, the length of which shall be 63 inches from the inside edge of the end line."*
  - S6: *"It shall be parallel to the end line and shall have its farthest edge 15 feet from the plane of the face of the backboard."*
  - S8 A1: *"Each backboard shall be midway between the sidelines, with the plane of its front face perpendicular to the floor, parallel to the end line, and 4 feet from it."*
  - S11 A1: *"The nearest point of the inside edge of the ring shall be 6 inches from the plane of the face of the backboard."*
  - S10 A1: *"a single metal ring, 18 inches in inside diameter"*.

**Preferred fix:** re-cite these rows from the KHSAA-hosted NFHS 2025-26 court diagram listed in the
owner-action list below, which would move them to **V** and give a stable URL. That needs a PDF
renderer or OCR (`pdftoppm`/`tesseract`), neither of which is installed here — the diagram is
image-only with no text layer.

**The basket centre offset of 63 in is now confirmed two independent ways, both from the same quoted rule text:**

1. **From the equipment chain.** Backboard face 4 ft (48 in) from the end line (Section 8 Art. 1), plus
   6 in from the backboard face to the nearest inside edge of the ring (Section 11 Art. 1), plus the
   9 in ring radius (half of the 18 in inside diameter, Section 10 Art. 1) = **63 in**.
2. **From the three-point line's own geometry.** The straight segments extend 63 in from the inside
   edge of the end line before joining the arc (Section 4 Art. 1). The arc is centred beneath the
   basket, so the junction lies at the basket's own distance from the end line — which makes that
   distance **63 in**.

This was the single most important open number, because it sets the court-coordinate origin for every
2/3 call. **For NFHS it is no longer PROVISIONAL.** The free-throw line from the end line likewise
follows as 4 ft + 15 ft = **19.0 ft**, confirming what the design had assumed.

Source: https://www.mhsaa.com/sports/girls-basketball/stories/be-referee-basketball-measurements

One search excerpt claimed that some states have moved the boys' line to 22 ft 1¾ in. I could not confirm it. This is why `three_point_radius_ft` is overridable per gym.

### What the build uses

| Level | `b` basket offset | `R` | `c` | Free-throw line from baseline | `verified` |
|---|---|---|---|---|---|
| NFHS | **5.25 ft (confirmed, S)** | 19.75 ft | 19.75 ft | **19.0 ft (= 4 ft + 15 ft, confirmed, S)** | `True` |
| NCAA | 5.25 ft (PROVISIONAL, assumed equal to NFHS) | 22.146 ft | 21.656 ft | 19.0 ft (PROVISIONAL) | `False` |
| NBA | 5.25 ft (PROVISIONAL) | 23.75 ft | 22.0 ft | 19.0 ft (PROVISIONAL) | `False` |
| FIBA | **5.167 ft (V)** | **22.146 ft (V)** | 21.654 ft | **19.029 ft (V)** | `True` |

**NFHS is the default level (Q4), and it and FIBA are now both sourced.** NCAA and NBA basket offsets
remain PROVISIONAL: the NCAA rule book exceeded the fetch size limit, the NCAA and NBA court diagrams
are image-only with no text layer, and no OCR tool (`pdftoppm`, `tesseract`) is installed. They stay
`verified = False`. Note that NCAA's 5.25 ft is an *assumption* that it matches NFHS, not a reading.

The US basket offset and backboard offset are tied together by verified NBA equipment numbers: 63 in minus the 9 in ring radius minus the 6 in gap is 48 in. So confirming either one confirms both.

An error of a few inches in `b` is far smaller than the expected homography error, so it will not change the 2/3 calls outside the 1 ft band. It still must be cited before step 5 passes.

### Owner action: confirm the PROVISIONAL and U rows (about 10 minutes)

Open each and read the basket centre offset, backboard offset, court size and three-point reference edge:
- NFHS 2025-26 court diagram: https://khsaa.org/common_documents/Rules-Basketball-Rules%20and%20NFHS%20Guidance/2025-26/NFHS%202025-26%20Basketball%20Court%20Diagram.pdf
- NCAA full court diagram: https://ncaaorg.s3.amazonaws.com/championships/sports/basketball/rules/men/PRXBB_CourtDiagram.pdf
- NCAA rule book: https://ncaaorg.s3.amazonaws.com/championships/sports/basketball/rules/men/PRMBB_RulesBook.pdf
- NBA court diagram: https://ak-static.cms.nba.com/wp-content/uploads/sites/4/2018/10/NBA_Court.png
- FIBA 2026 rules, Article 2 (link above).

## 6. Failure modes for one cheap fixed camera

Each degrades to unknown or to a flagged low confidence, not to a confident wrong answer.

| Failure | What happens | Degrades to |
|---|---|---|
| Ball hidden by players or the net at the rim | Gap inside the critical box | `unknown` (`ball_lost_at_rim`), or a 0.6 make if real detections exist above and below |
| Ball behind the backboard | Ball vanishes in the zone | `unknown` |
| Far rim is tiny | `W < 20 px` | Attempt logged, `result = unknown` (`rim_small`) |
| No rim calibrated, or the camera moved | No rim, or drift check fires | `unknown`, or `result_conf` capped at 0.3 |
| Air ball on the camera's line of sight | Looks like a pass through the rim box | Depth check gives `unknown`. **Weakest rule in v1; it can produce a wrong make** |
| Ghost ball detections | Speed gate and ghost list | Dropped |
| Clip ends mid-flight | Flight never reaches the zone | No attempt emitted; counted in `n_flights_lost` |
| Shooter's feet hidden or cut off | Bad feet point | `points` empty (`feet_cut_off`) |
| Feet near the line | Within 1 ft | `points` best guess, `points_conf = 0.3` |
| No homography | | `points` empty |
| Crowded release | Votes split | `shooter_conf = 0.3`, or shooter empty |

Known misses that v1 does not handle: dunks and very flat layups (the apex rule fails, so the attempt is missed), lob passes near the rim (counted as attempts), and passes crossing the rim region at a different depth (may be counted as attempts).

## 7. Open-source survey

Nothing below is copied into this repo. All v1 code is original and numpy/OpenCV only.

| Project | Licence | Verdict |
|---|---|---|
| roboflow/sports | MIT (read on the repo page) | Usable **with a specific carve-out, now confirmed**. Its soccer example **does import Ultralytics (AGPL-3.0)**: `examples/soccer/main.py` has `from ultralytics import YOLO` and uses `sv.Detections.from_ultralytics`, and `examples/soccer/requirements.txt` lists `ultralytics`. **Borrow only the library modules** (`sports.common.team`, `sports.common.view`, `sports.common.ball`, `sports.configs`) and **never the example scripts**. Whether the library modules themselves import Ultralytics was *not* checked — verify before adopting even those |
| avishah3/AI-Basketball-Shot-Detection-Tracker | **No licence file** | Do not copy. No licence means all rights reserved. Its README says it uses the **Ultralytics YOLO implementation (AGPL-3.0)**. Ideas only: clean the ball and hoop points, fit a line, test whether it passes through the hoop |
| chonyy/AI-basketball-analysis | **CMU OpenPose licence: non-commercial research only** (read in its LICENSE file; the page also shows a misleading MIT badge) | Avoid |
| anishcd/basketball-shot-detection | No licence visible | Do not copy. Faster R-CNN in PyTorch, ball and rim classes |
| nitinhemaraj/Basketball-shot-detection | Not verified | Avoid. Derived from avishah3 and imports Ultralytics |
| aarnavshah12/shot-tracker (Roboflow blog) | Not verified | Researcher to check before any use |

No AGPL or GPL code is adopted. The two Ultralytics-based projects are the ones to keep out, given the earlier AGPL scare.

## 8. Evaluation plan and file formats

### 8.1 Label file (written by `scripts/label_shots.py`, read by `eval/score_shots.py`)

Path: `data/labels/<video_stem>_shots.csv`. Header and example:

```
frame,time_s,result,points,notes
2,0.084,make,3,
26,1.086,make,2,
```

- `frame`: 0-based frame on screen when the key was pressed.
- `time_s`: `frame / fps`, with fps read from the video.
- `result`: `make` or `miss`.
- `points`: the **shot value**, 1, 2 or 3, on every row including misses. **A scorer must not read `points` as points scored.** Points scored is `points` if `result == "make"`, else 0.
- `notes`: empty from the tool; for hand edits.

Labelling instruction for the owner: press `m` or `x` when the ball reaches the rim and the result is clear, not at release.

Known label noise:
- Key presses during playback are about 0.2-0.4 s late. This bias is one-directional, so it does not cancel out; it eats into the ±1 s tolerance on one side.
- An untyped shot is saved as 2 points. A mis-keyed free throw or three is recorded as a two and will count against 2/3 accuracy.

### 8.2 Predicted file `shots.csv` (written by `src/shots/detect.py`)

The first five columns have the same names and meaning as the label file. Columns, in order:

```
frame,time_s,result,points,notes,
shot_id,release_frame,release_time_s,shooter_track_id,
ball_release_x_px,ball_release_y_px,feet_x_px,feet_y_px,feet_x_ft,feet_y_ft,
rim_id,level,margin_ft,
attempt_conf,release_conf,result_conf,shooter_conf,points_conf
```

- `frame`: the outcome frame (section 4.4), 0-based. `time_s = frame / fps`, 3 decimals.
- `result`: `make`, `miss` or `unknown`.
- `points`: shot value 1, 2, 3, or empty when unknown. Same meaning as the label file.
- `notes`: reason codes joined by `;`. The codes are: `no_rim`, `rim_small`, `rim_drift`, `ball_lost_at_rim`, `clip_end`, `bridged_crossing`, `depth_check_failed`, `no_evidence`, `no_homography`, `feet_off_court`, `feet_cut_off`, `near_arc`, `no_shooter`, `ft_rule`, **`provisional_geometry`**, **`no_court_length`**.
  - **`provisional_geometry`** (added after the step 5 review): a court constant used by the 2/3 decision has `verified = False`, so `points_conf` is capped at 0.6. Fires for **NCAA and NBA only** — verified by running the check across all four levels; NFHS (the default) and FIBA are sourced and uncapped. Without this code the spec table's honesty never reached the CSV, and `--level ncaa` would have published `points_conf 0.9` off a basket offset the code itself labels an assumption.
  - **`no_court_length`** (added after the step 5 review): rim B has no `court_length_ft`, so the rim-B frame conversion is impossible. Previously this emitted `no_homography`, which was misleading when a homography did exist.
  - **`no_homography`** should now be read as "no homography, **or** the three-point decision could not be made".
- `shot_id`: integer from 0 in frame order.
- Empty cells mean unknown. `feet_*_ft` are in the attacked rim's frame.
- Rows are sorted by `frame`.

### 8.3 Metrics (`eval/score_shots.py`)

**Matching.** Build all pairs with `|label.time_s - pred.time_s| <= 1.0`. Sort by absolute difference and accept greedily, each row used once. Every predicted row is an attempt, including `unknown`.

| Metric | Definition |
|---|---|
| Attempt precision | matched / predicted rows |
| Attempt recall | matched / label rows |
| Make/miss accuracy | Over matched pairs where the prediction is `make` or `miss`: fraction equal to the label |
| Make/miss coverage | Fraction of matched pairs where the prediction is not `unknown` |
| Make/miss strict accuracy | Same as accuracy, with `unknown` counted as wrong |
| 2/3 accuracy | Over matched pairs where both `points` values are 2 or 3: fraction equal. **Misses are included** |
| 2/3 coverage | Fraction of matched pairs with label 2 or 3 where the prediction has a value |
| Free-throw precision and recall | Treating `points == 1` as the positive class, over matched pairs |
| Points error per game | Sum of `points` over predicted makes, minus the same over label makes. Report signed and absolute. Computed over all rows, not only matched ones |
| Makes with unknown value | Count of predicted makes with empty `points` (they add 0 to the total) |
| Median time offset | Median of `label.time_s - pred.time_s` over matched pairs |

The median offset is a check on the labelling convention. About +0.3 s is expected. A value near +1 s or negative means labels were pressed at a different moment than assumed, and attempt recall is being understated.

A zero denominator reports `n/a`, never 0 and never an exception. Empty files must not crash.

No number from this scorer is an accuracy claim until labelled real footage exists (Q3).

## 9. Notes for the build and tests

- Tests must use a non-30 fps (23.951) and a second rate (for example 60) to prove nothing is hard-coded.
- A y-direction test: the clean-make trajectory flipped vertically must not be a make.
- A feet test: a shooter whose box centre is inside the arc but whose feet are outside must be a 3.
- A homography direction test: a known pixel corner must map to `(-25, 0)`.
- Edge cases for 2/3 at NFHS: feet at `(20.5, 3.0)` is a 3 (corner); `(0, 5.25 + 19.0)` is a 2; `(0, 5.25 + 20.9)` is a 3 at 0.9; `(0, 5.25 + 20.0)` is a 3 at 0.3.
- Footage of minors: the calibration and label files hold numbers only. No tool here writes frames or crops to disk.

## 10. Post-build review

Reviewed 2026-11-17 (step 6) against the built code: `src/shots/config.py`, `src/court/geometry.py`, `src/shots/trajectory.py`, `src/shots/detect.py`, `scripts/detect_shots.py`, `eval/score_shots.py` and `tests/test_shots.py`. I checked the self-reported deviations Q22-Q27 against the code instead of taking them on trust. Nothing in this section has been measured on footage.

### 10.1 Mismatches against the design

**Accepted as built:**

| Item | Where | Verdict |
|---|---|---|
| Q22: `verified` flags per value; NCAA all `False`; NBA arc `True` and the rest `False` | `geometry.py:74-126` | Accept. Matches section 5 as revised. |
| Q23: held frame must be within 0.5 s of the first free frame (`max_release_gap_s`, CHOSEN) | `detect.py:403` | Accept. It stops an old holder from being credited with a shot. |
| Q24.1: the first point past the zone counts as evidence | `detect.py:233` | Accept. Without it, a fast drop below the rim has no "below" point at 24 fps. |
| Q24.2: an observed below-and-outside point before the loss gives a miss, real points only, capped at 0.6 | `detect.py:285-294` | Accept. This is consistent with "degrade to unknown, not to wrong". |
| Q24.3: the two-rim tie-break uses distance in rim widths | `detect.py:418` | Accept. It is a better reading of the design. |
| Q26: seed picks the highest total confidence first | `trajectory.py:100-105` | Accept. The test showed that lowest cost alone picks a motionless ghost. |
| Q26: an untracked box can vote but cannot win | `detect.py:564` | Accept. |
| Q27: rim-label check raises an error; the median fallback refuses to run when a homography exists | `detect.py:138-158`, `333-336` | Accept. It closes the mirrored-coordinates path. |
| Q27: `provisional_geometry` cap | `detect.py:530-535` | Accept. |
| Q27: the scorer drops and counts bad rows, prints a banner and exits with code 2 | `eval/score_shots.py:43-60`, `205-206` | Accept. |
| Make rule: "y does not decrease by more than 0.1 W" is checked per step, not across the whole a-to-b stretch | `detect.py:258` | Accept. The 0.30 s limit bounds the total rise. Document it as per-step. |

**Should change:**

1. **`gap_n` is now redundant in loss detection.** Remove it there and document why; do not leave it as a dead threshold.
   - `_build` fills every gap inside a segment (`trajectory.py:156-169`), and any gap longer than `max_gap_s` ends the segment (`trajectory.py:110`). So inside a trajectory, `F[k+1] - F[k] - 1 > gap_n` can only be true where `seg` also changes.
   - Remove the gap term from the loss test (`detect.py:278`) and the bridged test (`detect.py:270`), and keep only the segment-change test.
   - Add a comment: "a segment break is the only loss signal; segments end at `trajectory.py:110` (gap) and `:135` (re-anchor)." Then `max_gap_s` has exactly one meaning, "longest gap we interpolate", and whoever tunes it later won't think it does two jobs.
   - The `gap_n` in the `clip_end` test (`detect.py:283`) has a different meaning and stays.
   - The same redundancy appears in `detect.py:395` (`frame[j+1] == frame[j]+1`) and `detect.py:438` (`frame[s] - frame[r_idx] > 1`). These are harmless; add a comment or remove them.
2. **`camera_static` is read and never used** (`geometry.py:274`; no reference in `detect.py`). It is the assumption the whole design rests on.
   - If it is `false`, `detect_shots` should raise `ConfigError`, or at least set every `result_conf` to 0 with note `rim_drift`.
   - See risk 1 for the drift check that should replace the current one.
3. **Flights that end because the ball is "held" before reaching the zone are not counted anywhere.** `n_flights_lost` only counts flights that end by ball loss (`detect.py:408`, `421-424`). Add a debug counter `n_flights_held_before_zone` for flights that pass rules 1-3 and are then cut off by `_held` (risk 3).
4. **Report `n_inliers` and the ghost count in the debug JSON.** Q26 already notes that `n_inliers` is stored but not reported; `ghosts` is local to `trajectory.py:82`. Both are needed for the first footage run.
5. **Free-throw confidence expression** (`detect.py:526`): `min(cap, pts_conf if pts_conf > 0.3 else cap)` raises a 0.3 to 0.6. It can't fire at the free-throw spot under NFHS or FIBA geometry, but it reads like a bug. Replace it with `min(cfg.ft_conf_cap, pts_conf)`.
6. **The speed gate scales with the nearer rim's W** (`detect.py:369-374`). The comment is honest, but the unit is wrong; see 10.3.

**Verified, no mismatch:** confidence levels for attempt and release (`detect.py:435-441`), the make outcome frame (`:271`), the order of the miss rules (`:296-311`), feet at take-off as the bottom-centre (`:487-488`), `is_three` and the line margin, including the corner (`geometry.py:295-320`), and the scorer's metric definitions against 8.3. No scipy, no new dependency, no image writes in either script.

### 10.2 Top three risks for the first real-footage test

**Risk 1: the static rim box is wrong for much of the game, and the result is confident wrong answers, not unknowns.**
- **How it fails:**
  - The camera moves. XbotGo-type mounts usually track the play, and the rim detections in the existing clip span x = 30 to 1250 (Q2), which fits a camera that moves.
  - Or the box is clicked badly on the first try, for example including the flange or the net, or clicked on a frame taken before a bump.
  - Every threshold is in W, so the attempt zone, the critical box and the make window all move together. Makes and misses are then decided against empty air at 0.9.
- **Why the current check can miss it:** the drift check (`detect.py:161-170`) takes one median over the whole clip on each half of the frame. It can't see a camera that sits still most of the time and pans for some stretches.
- **How to detect it early:**
  - Before labelling a game, compute the fraction of rim detections within 0.5 W of the calibrated centre in each 10 s window.
  - Any window below about 50% (**T**), where the rim is detected at all, means the box doesn't hold there.
  - Make this per-window check the drift check: cap `result_conf` per window, not per rim.
  - Film the first game on a tripod with tracking off. Record `frame_width` and `frame_height` in the calibration file.

**Risk 2: make recall collapses, because three separate gates each turn real makes into `unknown`.** This is the metric most likely to look weak first.
- **(a) The depth ratio window `[0.35, 0.75]`.**
  - In the XbotGo clip the ball box is about 22 px wide (median across f680-f709), and the detected rim box is about 71 px. That ratio is 0.31.
  - The ball isn't at the rim's depth in those frames, so this isn't a measurement, but it puts the expected ratio at or below the lower bound.
  - If it holds, almost every make becomes `depth_check_failed`.
- **(b) `make_max_dt_s = 0.30` against the net.** The net slows the ball and hides it, and the gap from "above the rim top" to "seen below the rim bottom" on a real make may be longer than 0.30 s.
- **(c) The 3-detection reseed (Q25).** After the net, a make only counts if three consecutive detections below the rim reseed the chain within that same 0.30 s. This partly cancels the bridged-make rule (Q16).
- **How it fails:** make/miss coverage is low and is made up mostly of misses. Misses don't depend on the depth check, so make/miss accuracy can look fine while points scored are badly low.
- **How to detect it early:**
  - Log, for each attempt, the ball/W ratio at `a`, the a-to-b time and the reason codes, even when the make passes.
  - On the first 10 makes the owner can see, check the ratio distribution and the a-to-b times before tuning anything.
  - In the scorer output, compare the counts of `depth_check_failed` and `ball_lost_at_rim` with label makes. If more than about a third of label makes come back `unknown`, this risk has happened.

**Risk 3: attempts disappear silently, with no counter and no row.** This is worse than a wrong result because nothing shows it.
- **How it fails:**
  - **(a) The "held" test is pure 2D overlap** (`detect.py:194-200`), with all player boxes padded 25% on top. A player standing between the camera and the rim can have a padded box that covers the rim. The ball is then "held" in the zone, the flight ends before reaching it, and nothing is counted (change 3 above).
  - **(b) The speed gate is 36 rim widths per second**, measured at the rim's depth. A ball much nearer the camera than the rim moves more pixels per second at the same physical speed. Fast passes and releases near the camera then fail the gate, and the chain breaks at the moment of release.
  - **(c) The ghost list never expires** (`trajectory.py:82-93`). Over a full game, not a 30 s clip, spots where the real ball rests while the chain is following something else can become permanent ghosts and block reseeding there.
- **How to detect it early:**
  - Watch attempt recall against the debug counters. Low recall with low `n_flights_lost` points here.
  - Add `n_flights_held_before_zone`, re-anchor and segment counts per minute, and the ghost count.
  - More than about 10 ghosts in a game (**T**) means the list needs to expire.

**Covered by the above, not ranked separately:** no threshold in `ShotConfig` has seen a real shot. That's true, but it doesn't help prioritise on its own. The three risks above are the specific thresholds most likely to fail first, and the debug fields named above are how to find the rest.

### 10.3 What I would now change in the design

1. **Gate ball speed in ball widths, not rim widths.** Use a running median of the accepted ball box widths, so the gate scales with the ball's own depth. This replaces `ref_w` (`detect.py:371`). Rim-relative units stay for the rim tests, where they are correct.
2. **Admit that the depth check cannot catch the air ball it was written for.** With the camera about 40 ft away, an air ball 3 ft in front of the rim changes the ball/W ratio by under 10%, well inside the window. It only rejects gross detector errors.
   - Replace it in v1.1 with a check on horizontal speed after the crossing. A make drops almost straight down in the net (|vx| falls sharply); an air ball keeps its horizontal speed.
   - Measure the ratio and the vx drop on the first footage before choosing either threshold.
3. **Make "held" require the same player for at least 2 frames, or require the ball box to overlap the player's upper body.** Do not use one padded-box hit from any player. This is still easy to explain in a presentation and closes risk 3(a).
4. **Expire ghosts:** drop a ghost that has had no hit in the last 10 s (**T**).
5. **Per-window drift check and enforce `camera_static`**, as in risk 1.
6. **Section 4.2 should say explicitly** that a segment break is the only loss signal and `max_gap_s` only sets interpolation (change 1 above).

None of these blocks the first footage test. Change 1 in 10.1 and change 3 in 10.3 are each under 20 lines. Items 1 and 2 in 10.3 should wait until the ratio and vx logs from the first 10 makes are in.

### 10.4 Main-session verification of this section

The architect's resumed session used only two tool calls, so its key line references were spot-checked against the code before this section was saved, rather than taken on trust. All four held exactly:

- `camera_static` is parsed into the `Calibration` dataclass (`geometry.py:232`, `:274`) and referenced **nowhere** in `detect.py` or `scripts/detect_shots.py`.
- `detect.py:526` is `min(cfg.ft_conf_cap, pts_conf if pts_conf > 0.3 else cfg.ft_conf_cap)`, which does lift a 0.3 to 0.6.
- `ghosts` is local to `trajectory.py:82`; neither `ghost` nor `n_inliers` appears in `detect.py`'s debug output.
- The ball/rim ratio: ball box median **22.2 px** over f680-f709 (n=28) against the 71.1 px rim gives **0.31**, below the depth window's 0.35 lower bound — reproduced exactly.

The 10.1 changes and 10.3 redesigns were **not implemented** in this run: step 6 asks for the review, not the changes, and implementing them would expand the task beyond its brief. They are recorded as recommended next work.

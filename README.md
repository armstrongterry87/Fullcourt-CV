# Fullcourt CV

**Basketball stats from one cheap camera.** Fullcourt CV is a computer-vision pipeline that turns a single video of a youth or high school basketball game into a stat sheet: shot attempts, makes and misses, 2s, 3s and free throws, the score, shot charts and per-player stats.

> **Current step (November 2026):** hand-charting a full game, shot by shot (make/miss, 2/3/FT), to build the answer key the model will be tested against. The first quarter is done: 39 shots, 14 makes. Next up is the first end-to-end scoring test on real game footage.

---

## Why

Youth and lower-funded high school programs rarely have real stats. Existing tools assume a budget, a trained operator, or special camera rigs. Fullcourt CV targets what those programs already have: one phone on a tripod, or an auto-panning camera like XbotGo or Veo. The intended customers are organizers (AAU programs, youth leagues, church leagues), priced per league rather than per coach.

## How it works

```
game video
  ├─► player / referee / ball / rim detection   (RF-DETR, fine-tuned)
  │     └─► multi-object tracking               (ByteTrack / BoT-SORT presets)
  │           └─► one role per track            (confidence-weighted vote)
  ├─► camera motion compensation                (background homographies → one fixed "virtual camera")
  │     └─► court calibration on a gym mosaic   (court lines + rim, clicked once)
  └─► shot engine
        ├─ attempt detection from the ball trajectory
        ├─ make / miss relative to the rim
        ├─ shooter attribution (nearest player at release)
        └─ 2 vs 3 vs FT from the shooter's feet in court coordinates
              └─► stats, score and shot chart
```

## Progress so far

| Stage | Status | Result |
|---|---|---|
| Detection | ✅ Fine-tuned RF-DETR Small (ball, player, referee, rim) | Spectators and bench players counted as players dropped from 57–61 IDs to 43–45 on the test clip |
| Tracking | ✅ ByteTrack and BoT-SORT presets, parameter sweeps | **IDF1 0.619** against a hand-corrected answer key (40 frames, 9 players). This is the first real accuracy baseline. |
| Roles and filtering | ✅ One role per track, separate person and ball thresholds, a court filter that uses the feet | No more double-counting players whose label flickers between player and referee |
| Shot engine | ✅ Attempt, make/miss, shooter, 2/3/FT, with a court model for high school (NFHS) and FIBA dimensions from official sources (NCAA and NBA still being verified) | Validated on synthetic trajectories |
| Camera motion compensation | ✅ Maps panning footage to one fixed view | Pose found for 99.9% of frames across an ~85° pan; a stationary person's position spread dropped from 110 px to 10 px |
| Evaluation tooling | ✅ IDF1 scorer, shot scorer, keyboard shot-labeling tool | 128 automated tests passing |
| **Real-game scoring test** | 🔄 **In progress:** hand-charting a full game | First quarter charted (39 shots) |
| Detector retraining on game footage | ⏭️ Next | Targets ball detection, the main weakness so far |
| Stats and shot-chart dashboard | ⏭️ Planned | Streamlit app for coaches |

## Engineering highlights

- **Measure against ground truth, not proxies.** An early "improvement" (fewer track IDs) turned out to be the tracker merging different people. Since then, every tracking change is scored with IDF1 against a hand-corrected answer key, and every scoring change against hand-charted shots.
- **Detection cache.** The detector runs once at the lowest threshold, and every higher threshold is filtered from the cache, so a full parameter sweep costs one detection pass.
- **Virtual static camera.** Auto-panning cameras rotate but don't zoom, so motion between frames is a homography. Features are matched only on the background (player and ball boxes are masked out), with keyframes that limit drift. All coordinates are mapped into one reference view, so the court is calibrated once for the whole game.
- **"Unknown" beats confidently wrong.** When the ball is lost near the rim or the camera pose is unreliable, the shot engine returns `unknown` instead of guessing.
- **Commercially clean stack.** Only permissive licences (Apache-2.0, MIT, BSD). No AGPL dependencies.

## Tech stack

Python · PyTorch · RF-DETR · Roboflow `trackers` and `supervision` · OpenCV · NumPy and pandas · py-motmetrics · pytest · CVAT (annotation) · Roboflow (training)

## Repo layout

```
src/         pipeline: detection → tracking → camera → court → shots → stats → dashboard
scripts/     CLIs: run tracking, stabilize video, calibrate court, detect and label shots
eval/        scorers and dated evaluation reports (eval/reports/)
tests/       unit tests, including synthetic shot and camera-motion cases
docs/        design docs, task briefs, progress log
data/        labels and answer keys only (video and frames are never committed)
```

## Setup (Windows)

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python -m pytest tests -q
```

Model weights and game footage aren't included. Game footage often shows minors, so it never goes into this repo.

## Author

Terry Armstrong. This is a personal project and an early-stage startup idea.

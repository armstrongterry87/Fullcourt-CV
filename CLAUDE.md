# Basketball Shot & Stat Tracker: Project Guide for Claude

This file is loaded at the start of every Claude Code session in this folder. It explains the project, the agent team, and the rules everyone (human and AI) works by.

## What we're building

A computer-vision system that takes video of a youth/high-school basketball game from **one phone or fixed camera** and produces:
- shot attempts, makes and misses
- a shot chart (where on the court each shot came from)
- basic per-player stats

This is a **two-semester senior design project** and a possible startup (target: NBA Launchpad, fall 2027 cycle, "Future of Youth Basketball" priority area). Success is measured by comparing the system's stats to **hand-charted stats from the same games**.

## Pipeline

```
video ─► ingest (frames) ─► player detection ─► tracking (IDs) ─► court mapping (homography)
                         └► ball detection ─► shot detection (attempt → make/miss) ─► shot→player attribution ─► stats + shot chart ─► dashboard
```

| Stage | Folder | Starting approach |
|---|---|---|
| Ingest | `src/ingest/` | OpenCV frame extraction, fixed FPS |
| Player detection | `src/detection/` | RF-DETR (Small/Medium, Apache-2.0) fine-tuned on our labeled frames |
| Tracking | `src/tracking/` | `ByteTrackTracker` from Roboflow's `trackers` package (it moved out of `supervision`); `BoTSORTTracker` if ID switches are bad |
| Court mapping | `src/court/` | Court keypoints → homography → court coordinates in feet |
| Shots | `src/shots/` | Ball detection + rim region + trajectory → attempt, make/miss |
| Stats | `src/stats/` | Aggregate shots/players into box-score style stats |
| Dashboard | `src/dashboard/` | Simple Streamlit app a coach can actually use |

Reference implementation to borrow from: Roboflow's open-source `sports` repo (built for soccer; the detection, tracking, team clustering and homography pieces transfer).

## Repo layout

```
.claude/agents/   agent team definitions (see below)
src/              pipeline code, one package per stage
scripts/          one-off CLI scripts (frame extraction, label conversion, training runs)
data/raw/         original game video         (NEVER committed; see constraints)
data/frames/      extracted frames             (NEVER committed)
data/labels/      annotations (YOLO format)
data/ground_truth/ hand-charted stats per game (CSV), the answer key
models/           trained weights (not committed)
eval/             evaluation scripts, owned by the evaluator agent
eval/reports/     dated accuracy reports
notebooks/        exploration only; real code lives in src/
docs/             project plan, progress log, AI-use log, report drafts
```

## The agent team

The main session is the **project manager**: it plans, breaks work into tasks, delegates, and integrates results. It runs on **Sonnet** by default (set in `.claude/settings.json`). Switch with `/model fable` when you want Fable running the whole session.

| Agent | Model | Owns | Use it for |
|---|---|---|---|
| `architect` | **Fable** | Design decisions | Hard problems only: architecture, choosing between approaches, bugs that survived 2 fix attempts |
| `researcher` | Sonnet | Nothing (read-only) | Papers, repos, datasets, license checks |
| `model-builder` | Sonnet | `src/`, `scripts/train_*` | Detection, tracking, court mapping, shot logic, training/inference code |
| `data-engineer` | Haiku | `scripts/`, `data/labels/` | Frame extraction, label format conversion, dataset splits |
| `evaluator` | Sonnet | `eval/` | Comparing output against `data/ground_truth/`, accuracy reports |
| `reviewer` | Opus | Nothing (read-only) | Review changes before every commit |
| `report-writer` | Haiku | `docs/` | Progress log, AI-use log, senior design report drafts |

### Delegation rules (main session, follow these)

1. **Don't do implementation work yourself when an agent owns it.** Delegate coding to `model-builder`, scripts to `data-engineer`, and so on. Keep the main context for planning and integration.
2. **Fable is expensive. Call `architect` only when** a decision is hard to reverse, approaches genuinely compete, or a bug has survived two attempts. Give it a tight brief: the problem, what's been tried, the constraints.
3. **The builder never grades its own work.** Accuracy numbers come only from `evaluator`. Never edit `eval/` or `data/ground_truth/` from any other agent.
4. **Run `reviewer` before every commit.**
5. **After any meaningful session, have `report-writer` update `docs/progress-log.md` and `docs/ai-use-log.md`.**
6. Agents can work in parallel when tasks don't touch the same files (e.g. `data-engineer` converting labels while `researcher` reads papers).

### Standard loop for a feature

plan (main) → [architect if hard] → build (model-builder / data-engineer) → review (reviewer) → measure (evaluator) → log (report-writer) → commit

## Hard constraints

- **Footage of minors.** Raw video and extracted frames never go into git or any public place. Only film with written permission from the school/program and parent consent. Check whether university IRB/ethics approval is required *before* recording.
- **Licenses matter (this may become a company).** Use RF-DETR Nano/Small/Medium/Large (Apache-2.0). Its XLarge/2XLarge and Atto/Femto/Pico models are PML 1.0 and need a paid license to self-host commercially. Don't add Ultralytics YOLO (AGPL-3.0). `supervision` and `trackers` are open source; the researcher agent must report the license of anything we adopt.
- **AI-use disclosure.** Every substantial AI contribution is logged in `docs/ai-use-log.md`. The student must be able to explain and defend every part of the system. When an agent writes non-trivial code, it ends with a short plain-English explanation of how it works and why.
- **Don't set the `CLAUDE_CODE_SUBAGENT_MODEL` environment variable.** It can override the `model:` field in the agent files and silently run agents on the wrong model.

## Coding conventions

- Python 3.11+, type hints, small functions, docstrings on public functions.
- Coordinates: pixel space is `(x, y)` from top-left; court space is **feet**, origin at the center of the baseline under the rim being attacked. Always label which space a variable is in (`ball_px`, `ball_ft`).
- Frame indices are 0-based and always paired with the video's FPS. Never assume 30fps.
- Paths come from a config or CLI args, never hardcoded to one machine.
- Split datasets **by game**, never by random frame (adjacent frames leak between train and test).

## Current milestone

Semester 1, Milestone 1: an accurate shot chart for one full game. See `docs/project-plan.md`.

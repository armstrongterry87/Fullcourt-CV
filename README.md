# Basketball Shot & Stat Tracker

Single-camera computer vision for youth and high-school basketball. Game video in; shot attempts, makes and misses, shot charts and player stats out.

Senior design project (two semesters) and a possible startup. See `docs/project-plan.md` for the plan and `CLAUDE.md` for how the AI agent team works.

## Setup (Windows)

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

You need a GPU for training: Google Colab, a university cluster, or a lab machine. Inference on short clips can run locally.

## Using the agent team in VS Code

1. Install the **Claude Code** extension in VS Code, and sign in.
2. Open this folder in VS Code.
3. For heavy multi-agent work, open the built-in terminal (Ctrl+`) and run `claude` there; the output is cleaner than the side panel.
4. Run `/agents` to confirm the seven agents loaded.

The main session runs on Sonnet and acts as project manager. Fable is reserved for the `architect` agent. To run the whole session on Fable instead: `/model fable`.

Example prompts:
- "Plan Milestone 1 and break it into tasks for the team."
- "Have the data-engineer write a script that extracts frames at 10fps from a video."
- "Have the researcher compare TrackNet-style ball tracking vs YOLO for a phone camera."
- "Ask the architect how we should detect makes vs misses from one side-angle camera."
- "Have the reviewer check my changes, then commit."
- "Have the report-writer update the progress log and AI-use log for today."

## Layout

```
src/        pipeline: ingest → detection → tracking → court → shots → stats → dashboard
scripts/    CLI scripts (frame extraction, label conversion, training)
data/       raw video, frames, labels, ground truth (video/frames never committed)
models/     trained weights (not committed)
eval/       evaluation code + dated reports
docs/       plan, progress log, AI-use log
```

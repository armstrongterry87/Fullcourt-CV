# Task 003: Tracking answer key and real accuracy score (IDF1)

**Status:** Ready after task 002. Tooling can be built now; the hand-labeling is done by the student.
**Owner (main session):** project manager. This task is mostly data-engineer + evaluator work, so no architect is needed unless something is unclear.

## Why

All tracking numbers so far measure self-consistency, not accuracy. "Fewer IDs" can mean better tracking or wrongly merged players (task 001 found two such merges). To score tracking for real, we need a small answer key: who is who, frame by frame, for a short stretch of video. Then we can compute **IDF1** and **ID switches**, the standard tracking accuracy measures.

## Plan

1. **Pick the clip.** First practice run: a 10-second stretch of `data/raw/xbotgo.mp4` where the camera is fairly still and players cross. Labeling the test clip for *evaluation* is fine; it is never used for *training*. Repeat on the student's own fixed-camera footage when it exists. That becomes the real answer key.

2. **Pre-fill the labels (data-engineer, Haiku).** Write `scripts/export_tracks_for_labeling.py`:
   - takes a run's tracks CSV, a start time and a duration
   - keeps every 6th frame (about 4 per second), players and referees only
   - exports in **MOT 1.1 format**, which CVAT imports, alongside those frames as images
   - pre-filling with the tracker's output means the student only fixes boxes and IDs instead of drawing everything

3. **Label (the student).** Import into CVAT (free, browser-based):
   - fix every wrong ID so each real person keeps one ID for the whole stretch
   - delete boxes on spectators; add missed players
   - export back as MOT 1.1 to `data/ground_truth/xbotgo_10s_mot.txt`

   Expect about an hour for 10 seconds.

4. **Score (evaluator, Sonnet).** Write `eval/score_tracking.py` using **py-motmetrics** (`pip install motmetrics`, MIT license). Inputs: the ground-truth MOT file and any run's tracks CSV. Outputs: **IDF1, ID switches, precision, recall, mostly tracked / lost**, scored only on the labeled frames. Then score the committed default (A0) and the task 002 recommendation, and add the results to `eval/reports/`.

5. **Log (report-writer, Haiku).** Progress log and AI-use log. The AI-use log should say the ground-truth IDs were corrected by hand by the student.

## Done when

- [ ] Export script works and imports cleanly into CVAT
- [ ] `data/ground_truth/xbotgo_10s_mot.txt` exists, hand-corrected by the student
- [ ] `eval/score_tracking.py` reports IDF1 and ID switches for at least two tracker configs
- [ ] Evaluation outputs are reproducible from a single command documented in the report

## Notes

- Ground truth is the answer key: no agent may edit `data/ground_truth/` after the student exports it (CLAUDE.md rule).
- Keep the 10-second window and frame stride fixed, so scores stay comparable across future tasks.

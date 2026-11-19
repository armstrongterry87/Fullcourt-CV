"""Helper for A3: write raw frames with track-id overlays (and debug-video frames) to runs/a3_frames/.

Footage shows minors: output goes under runs/ only (git-ignored).
Usage: python eval/motion_frames.py --video V --tracks T --out runs/a3_frames --frames 10 100 ... [--debug-video D]
"""
from __future__ import annotations
import argparse
from pathlib import Path
import cv2
import pandas as pd


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--tracks", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--frames", type=int, nargs="+", required=True)
    ap.add_argument("--debug-video")
    a = ap.parse_args()
    out = Path(a.out).resolve()
    if "runs" not in out.parts:
        raise SystemExit("output must be under runs/")
    out.mkdir(parents=True, exist_ok=True)
    t = pd.read_csv(a.tracks)
    for src, tag, draw in [(a.video, "raw", True), (a.debug_video, "dbg", False)]:
        if not src:
            continue
        cap = cv2.VideoCapture(src)
        want = set(a.frames)
        i = 0
        while True:
            ok, fr = cap.read()
            if not ok or i > max(want):
                break
            if i in want:
                if draw:
                    for r in t[t.frame == i].itertuples():
                        x1, y1, x2, y2 = map(int, (r.x1_px, r.y1_px, r.x2_px, r.y2_px))
                        col = (0, 255, 255) if r._3 == "player" else (255, 0, 255)
                        cv2.rectangle(fr, (x1, y1), (x2, y2), col, 1)
                        cv2.putText(fr, f"{r.track_id}", (x1, max(y1 - 3, 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                cv2.imwrite(str(out / f"{tag}_{i:04d}.jpg"), fr)
            i += 1
        cap.release()


if __name__ == "__main__":
    main()

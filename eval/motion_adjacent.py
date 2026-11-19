"""A3 helper: jump check. For every pair of consecutive frames (t, t+1) that are both reliable, fit the homography
directly and compare it with the composed one inv(H_{t+1}_to_ref) @ H_t_to_ref (corner error in px of frame t+1).
A jump in the stabilised sequence (for instance where the keyframe changes) shows as a spike here.

Usage: python eval/motion_adjacent.py --video V --det DET --motion MOTION [--out runs/a3_adjacent.csv]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.camera.motion import (MotionConfig, build_mask, extract_features, match_homography, load_motion,  # noqa: E402
                               load_detection_boxes, transform_points, video_frames, _corners)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--det", required=True)
    ap.add_argument("--motion", required=True)
    ap.add_argument("--out", default=str(ROOT / "runs" / "a3_adjacent.csv"))
    a = ap.parse_args()
    cfg = MotionConfig()
    motion = load_motion(Path(a.motion))
    boxes, _ = load_detection_boxes(Path(a.det), motion.n_frames)
    W, H = motion.frame_wh
    corners = _corners(W, H)
    Hf = motion.H_frame_to_ref
    prev_f, rows = None, []
    for t, fr in enumerate(video_frames(Path(a.video))):
        f = extract_features(cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY), build_mask(fr.shape[:2], boxes[t], cfg), cfg)
        if prev_f is not None and np.all(np.isfinite(Hf[t])) and np.all(np.isfinite(Hf[t - 1])):
            fit = match_homography(prev_f, f, (W, H), cfg)
            row = {"t": t, "ok": fit.H_src_to_dst is not None and fit.n_inliers >= cfg.min_inliers,
                   "n_inliers": fit.n_inliers, "kf_prev": int(motion.stats["keyframe"][t - 1]),
                   "kf": int(motion.stats["keyframe"][t]), "q": float(motion.quality[t]),
                   "chained": int(motion.stats["chained"][t])}
            if row["ok"]:
                comp = np.linalg.inv(Hf[t]) @ Hf[t - 1]
                e = np.linalg.norm(transform_points(fit.H_src_to_dst, corners) - transform_points(comp, corners), axis=1)
                row["err_max_corner_px"] = float(np.nanmax(e))
                row["err_mean_corner_px"] = float(np.nanmean(e))
                row["motion_px"] = float(np.nanmean(np.linalg.norm(transform_points(fit.H_src_to_dst, corners) - corners, axis=1)))
            rows.append(row)
        prev_f = f
    pd.DataFrame(rows).to_csv(a.out, index=False)
    print("wrote", a.out, len(rows))


if __name__ == "__main__":
    main()

"""Score a tracker CSV against the 10 s xbotgo MOT answer key (IDF1, IDsw, P, R, MT, ML).

One command (from repo root):
  .venv/Scripts/python.exe eval/score_tracking.py runs/task001_eval/A0/xbotgo_tracks.csv --label A0
Many CSVs: pass several paths. Label defaults to the first ancestor dir that is not a preset name
(runs/x/p0.4_b0.4/A0/f.csv -> p0.4_b0.4); or give --label once per CSV. --role-column role_track scores the voted role.
Frame map: eval/xbotgo_10s_frame_map.json (committed), fallback to the gitignored sidecar. Add --json out.json.

Scores only the 40 original frames in the frame map. Tracker rows with
track_id < 0 (untracked detections) are dropped, matching how the key was built.
IoU distance, threshold 0.5. Modes: players (class==player) and players+referees.
"""
import argparse, json
from pathlib import Path
import numpy as np, pandas as pd, motmetrics as mm

ROOT = Path(__file__).resolve().parent.parent
GT = ROOT / "data/ground_truth/xbotgo_10s_mot.txt"
FRAME_MAP = ROOT / "eval/xbotgo_10s_frame_map.json"  # committed, numbers only
SIDECAR = ROOT / "data/frames/task003_xbotgo_10s/export_sidecar.json"  # fallback (gitignored)
PRESETS = {"A0", "A1", "B0", "B1", "B2", "B3", "B4"}
MODES = {"players": ["player"], "players+referees": ["player", "referee"]}


def load_gt(path):
    """GT MOT file (9 fields, CVAT flavour) -> DataFrame frame,id,x,y,w,h (px, tlwh)."""
    df = pd.read_csv(path, header=None, usecols=range(6), names=["frame", "id", "x", "y", "w", "h"])
    return df


def load_tracks(path, frame_map, classes, role_column="class"):
    """Tracker CSV -> tlwh rows on the labelled frames, mapped to MOT frame numbers."""
    t = pd.read_csv(path)
    if role_column not in t.columns:
        raise SystemExit(f"{path}: no column {role_column!r} (columns: {list(t.columns)})")
    t["class"] = t[role_column]  # the filter below always reads "class"
    o2m = {r["original_frame"]: r["mot_frame"] for r in frame_map}
    n_untracked = int(((t.track_id < 0) & t.frame.isin(o2m) & t["class"].isin(classes)).sum())
    t = t[t.frame.isin(o2m) & t["class"].isin(classes) & (t.track_id >= 0)].copy()
    t["frame"] = t.frame.map(o2m)
    t["x"], t["y"] = t.x1_px, t.y1_px
    t["w"], t["h"] = t.x2_px - t.x1_px, t.y2_px - t.y1_px
    return t.rename(columns={"track_id": "id"})[["frame", "id", "x", "y", "w", "h"]], n_untracked


def iou_dist(a, b, max_dist=0.5):
    """IoU distance (1-IoU) for tlwh arrays; NaN where distance > max_dist.
    Own implementation: motmetrics 1.4.0's iou_matrix calls np.asfarray, removed in NumPy 2."""
    a, b = np.asarray(a, float).reshape(-1, 4), np.asarray(b, float).reshape(-1, 4)
    d = np.full((len(a), len(b)), np.nan)
    for i, (x, y, w, h) in enumerate(a):
        for j, (x2, y2, w2, h2) in enumerate(b):
            iw = min(x + w, x2 + w2) - max(x, x2)
            ih = min(y + h, y2 + h2) - max(y, y2)
            inter = max(iw, 0) * max(ih, 0)
            dist = 1 - inter / (w * h + w2 * h2 - inter)
            if dist <= max_dist:
                d[i, j] = dist
    return d


def score(gt, tr, frames):
    acc = mm.MOTAccumulator(auto_id=False)
    for f in frames:
        g, t = gt[gt.frame == f], tr[tr.frame == f]
        d = iou_dist(g[["x", "y", "w", "h"]].values, t[["x", "y", "w", "h"]].values)
        acc.update(g.id.tolist(), t.id.tolist(), d, frameid=f)
    mh = mm.metrics.create()
    s = mh.compute(acc, metrics=["idf1", "idp", "idr", "num_switches", "precision", "recall",
                                 "mostly_tracked", "mostly_lost", "partially_tracked", "num_false_positives",
                                 "num_misses", "num_matches", "num_objects"], name="x").iloc[0]
    return {k: (float(v) if k in ("idf1", "idp", "idr", "precision", "recall") else int(v)) for k, v in s.items()}


def default_label(p):
    """Label = first ancestor directory that is not a preset name (A0...), else the parent name."""
    parts = Path(p).resolve().parent.parts
    for name in reversed(parts):
        if name not in PRESETS:
            return name
    return Path(p).parent.name


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("tracks", nargs="+", help="tracker CSV(s)")
    ap.add_argument("--label", action="append", help="repeatable; one per CSV, in order. Default: first ancestor "
                    "directory of the CSV that is not a preset name (A0, B1, ...)")
    ap.add_argument("--role-column", default="class", choices=["class", "role_track"],
                    help="which column decides player/referee: per-frame 'class' or voted 'role_track'")
    ap.add_argument("--modes", nargs="+", default=list(MODES), choices=list(MODES))
    ap.add_argument("--gt", default=str(GT))
    ap.add_argument("--frame-map", default=None, help="frame map JSON (default eval/xbotgo_10s_frame_map.json, "
                    "falling back to the gitignored sidecar)")
    ap.add_argument("--json", help="write results here")
    a = ap.parse_args()
    fmp = Path(a.frame_map) if a.frame_map else (FRAME_MAP if FRAME_MAP.exists() else SIDECAR)
    fm = json.load(open(fmp))["frame_map"]
    if a.label and len(a.label) != len(a.tracks):
        ap.error(f"--label given {len(a.label)}x for {len(a.tracks)} CSVs; must match")
    frames = sorted(r["mot_frame"] for r in fm)
    gt = load_gt(a.gt)
    out = []
    for i, p in enumerate(a.tracks):
        label = a.label[i] if a.label else default_label(p)
        full = pd.read_csv(p)
        omap = {r["original_frame"] for r in fm}
        present = sorted(omap & set(full.frame))
        for m in a.modes:
            tr, nun = load_tracks(p, fm, MODES[m], a.role_column)
            r = score(gt, tr, frames)
            r.update(label=label, mode=m, role_column=a.role_column, csv=str(p), tracker_ids_in_window=int(tr.id.nunique()),
                     gt_ids=int(gt.id.nunique()), frames_overlapping=len(present), frames_scored=len(frames),
                     untracked_rows_dropped=nun, tracker_boxes=len(tr))
            out.append(r)
    cols = ["label", "mode", "role_column", "idf1", "idp", "idr", "num_switches", "precision", "recall", "mostly_tracked",
            "mostly_lost", "partially_tracked", "tracker_ids_in_window", "gt_ids", "frames_overlapping",
            "untracked_rows_dropped", "tracker_boxes"]
    print(pd.DataFrame(out)[cols].to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    if a.json:
        json.dump(out, open(a.json, "w"), indent=1)


if __name__ == "__main__":
    main()

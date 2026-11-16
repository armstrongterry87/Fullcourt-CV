"""Tracker self-consistency metrics from baseline_track.py outputs (NOT accuracy; no GT IDs exist).

Usage: python eval/track_consistency.py --dir runs/task001_eval [--det-s-per-frame 0.30] [--out table.md]

Lifetime has two definitions:
  span     = (last_frame - first_frame + 1) / fps   (includes gaps / frames labelled referee)
  presence = number of distinct player-labelled frames / fps
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def preset_metrics(run_dir: Path, det_s: float) -> dict:
    """Metrics for one preset directory containing <stem>_summary.json and <stem>_tracks.csv."""
    sums = sorted(run_dir.glob("*_summary.json"))
    if len(sums) != 1:
        raise FileNotFoundError(f"{run_dir}: expected exactly one *_summary.json, found {len(sums)}")
    csv = sums[0].with_name(sums[0].name.replace("_summary.json", "_tracks.csv"))
    if not csv.exists():
        raise FileNotFoundError(f"{run_dir}: missing {csv.name}")
    summ = json.loads(sums[0].read_text())
    df = pd.read_csv(csv)
    fps = summ["fps"]
    pl = df[(df["class"] == "player") & (df["track_id"] >= 0)]
    g = pl.groupby("track_id")["frame"].agg(["min", "max", "nunique"])
    span_s = (g["max"] - g["min"] + 1) / fps
    pres_s = g["nunique"] / fps
    ref_ids = df[(df["class"] == "referee") & (df["track_id"] >= 0)]["track_id"].unique()
    trk_s = summ["tracker_ms_per_frame"] / 1000
    return {
        "preset": summ["preset"], "tracker": summ["tracker"],
        "unique_player_ids": int(len(g)), "summary_unique_ids": summ["unique_player_ids"],
        "lt2s_span": int((span_s < 2).sum()), "lt2s_presence": int((pres_s < 2).sum()),
        "ge5s_span": int((span_s >= 5).sum()), "ge5s_presence": int((pres_s >= 5).sum()),
        "ids_in_both_player_and_referee": int(len(set(g.index) & set(ref_ids))),
        "tracker_ms_per_frame": summ["tracker_ms_per_frame"],
        "e2e_fps": round(1 / (det_s + trk_s), 2), "ball_seen_pct": summ["ball_seen_pct"],
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", type=Path, required=True)
    ap.add_argument("--det-s-per-frame", type=float, default=0.30)
    ap.add_argument("--out", type=Path, default=None, help="write table as markdown")
    a = ap.parse_args()
    t = pd.DataFrame([preset_metrics(d, a.det_s_per_frame) for d in sorted(a.dir.iterdir()) if d.is_dir()])
    print(t.to_string(index=False))
    if a.out:
        cols = list(t.columns)
        lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
        lines += ["| " + " | ".join(str(v) for v in r) + " |" for r in t.itertuples(index=False)]
        a.out.write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()

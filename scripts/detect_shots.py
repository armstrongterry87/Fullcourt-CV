"""Detect shot attempts, make/miss, shooter and shot value from an existing tracks CSV.

Writes numbers only (shots.csv and a small debug JSON). No image, frame or crop is ever written.

Examples:
  python scripts/detect_shots.py --tracks runs/task001_eval/A0/xbotgo_tracks.csv \
      --calibration data/courts/xbotgo.json --out runs/shots/xbotgo_shots.csv
  python scripts/detect_shots.py --tracks T.csv --calibration C.json --out S.csv --fps 59.94 --level nfhs
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.court.geometry import get_spec, load_calibration  # noqa: E402
from src.court.polygon import ConfigError  # noqa: E402
from src.shots.detect import detect_shots, infer_fps, read_tracks, write_shots_csv  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tracks", type=Path, required=True, help="tracks CSV from baseline_track.py")
    ap.add_argument("--calibration", type=Path, required=True, help="court JSON (rims and homography optional)")
    ap.add_argument("--out", type=Path, required=True, help="shots.csv to write")
    ap.add_argument("--fps", type=float, default=None, help="video fps; default is frame / time_s from the CSV")
    ap.add_argument("--frame-height", type=float, default=None, help="video height in px (feet-cut-off check)")
    ap.add_argument("--level", default=None, help="override the calibration level: nfhs, ncaa, nba, fiba")
    args = ap.parse_args(argv)
    try:
        tracks = read_tracks(args.tracks)
        calib = load_calibration(args.calibration)
        if args.level:
            get_spec(args.level)
            calib = replace(calib, level=args.level.lower())
        fps = args.fps or infer_fps(tracks)
        result = detect_shots(tracks, calib, fps, frame_height_px=args.frame_height)
    except ConfigError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    write_shots_csv(result, args.out)
    dbg_path = args.out.with_suffix(".debug.json")
    dbg_path.write_text(json.dumps(result.debug, indent=2, default=str))
    d = result.debug
    print(f"fps={fps:.3f} attempts={d['n_attempts']} flights={d['n_flights']} "
          f"n_flights_lost={d['n_flights_lost']} segments={d.get('n_segments')} -> {args.out}")
    for w in d["warnings"]:
        print(f"warning: {w}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Export a pre-filled tracking clip for CVAT labeling (task 003, step 2).

Reads a tracker CSV (one row per detection per frame, with ORIGINAL 0-based
frame indices) and the source video, then writes everything into one folder:

  <out-dir>/img1/000001.jpg ...   kept frames, renumbered 1..N
  <out-dir>/gt/gt.txt             MOT 1.1 pre-fill: players and referees only
  <out-dir>/cvat_import.zip       gt/gt.txt + img1/ in one archive (for CVAT MOT 1.1 import)
  <out-dir>/export_sidecar.json   mot_frame -> original_frame map, fps, window, counts

Why the renumbering matters: CVAT pairs the Nth image with MOT frame N, and MOT
frame numbers must be contiguous. The sidecar is the only link back to the
original frame indices, which the evaluator needs to score against the tracker CSV.

The export holds frames of minors, so --out-dir must be under data/frames/ or
runs/ (both gitignored). Any other location is refused.

Window and stride defaults are fixed for task 003 so scores stay comparable.

Examples:
  .venv/Scripts/python.exe scripts/export_tracks_for_labeling.py
  .venv/Scripts/python.exe scripts/export_tracks_for_labeling.py --start-s 12.82 --duration-s 10 --stride 6
"""

from __future__ import annotations

import argparse
import csv
import json
import zipfile
from pathlib import Path

import cv2

REPO_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_TRACKS = REPO_ROOT / "runs" / "task001_eval" / "A0" / "xbotgo_tracks.csv"
DEFAULT_VIDEO = REPO_ROOT / "data" / "raw" / "xbotgo.mp4"
DEFAULT_OUT_DIR = REPO_ROOT / "data" / "frames" / "task003_xbotgo_10s"

# Fixed for task 003 (see docs/tasks/003-tracking-answer-key.md). Do not change without a new task.
DEFAULT_START_S = 12.82
DEFAULT_DURATION_S = 10.0
DEFAULT_STRIDE = 6

KEPT_CLASSES = ("player", "referee")
REQUIRED_COLUMNS = {"frame", "class", "track_id", "x1_px", "y1_px", "x2_px", "y2_px"}
IMAGE_SAFE_ROOTS = (REPO_ROOT / "data" / "frames", REPO_ROOT / "runs")
STAT_KEYS = (
    "exported_player",
    "exported_referee",
    "dropped_negative_id",
    "dropped_degenerate",
    "clamped_to_frame",
    "skipped_other_class",
)


def is_under(path: Path, root: Path) -> bool:
    """Return True if path (resolved) sits inside root (resolved)."""
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False


def get_video_info(video_path: Path) -> tuple[float, int, int, int]:
    """Return (fps, width_px, height_px, frame_count) from the video header."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise SystemExit(f"Could not open video: {video_path}")
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return fps, width, height, count


def window_frames(start_s: float, duration_s: float, fps: float, stride: int) -> tuple[int, int, list[int]]:
    """Return (first frame, last frame inclusive, every-Nth kept frames) in original 0-based indices."""
    start = round(start_s * fps)
    n_window = round(duration_s * fps)
    end = start + n_window - 1
    kept = list(range(start, end + 1, stride))
    return start, end, kept


def read_tracks(csv_path: Path, wanted: set[int]) -> tuple[dict[int, list[dict]], int]:
    """Return rows for the wanted original frames (grouped by frame) and the last frame index in the CSV.

    Extra columns are ignored, so the CSV can grow new columns without breaking this script.
    """
    by_frame: dict[int, list[dict]] = {f: [] for f in wanted}
    last_frame = -1
    with csv_path.open(newline="") as f:
        reader = csv.DictReader(f)
        missing = REQUIRED_COLUMNS - set(reader.fieldnames or [])
        if missing:
            raise SystemExit(f"{csv_path} is missing columns: {sorted(missing)}")
        for row in reader:
            frame = int(row["frame"])
            last_frame = max(last_frame, frame)
            if frame in by_frame:
                by_frame[frame].append(row)
    return by_frame, last_frame


def clamp_box(row: dict, width: int, height: int) -> tuple[float, float, float, float, bool] | None:
    """Clamp a corner box (x1, y1, x2, y2 in pixels) to the frame and convert to MOT (left, top, w, h).

    Returns None for a degenerate box (zero or negative area after clamping).
    The last value says whether any side was moved by the clamp.
    """
    x1, y1 = float(row["x1_px"]), float(row["y1_px"])
    x2, y2 = float(row["x2_px"]), float(row["y2_px"])
    cx1, cy1 = min(max(x1, 0.0), width), min(max(y1, 0.0), height)
    cx2, cy2 = min(max(x2, 0.0), width), min(max(y2, 0.0), height)
    clamped = (cx1, cy1, cx2, cy2) != (x1, y1, x2, y2)
    w, h = cx2 - cx1, cy2 - cy1
    if w <= 0 or h <= 0:
        return None
    return cx1, cy1, w, h, clamped


def build_mot_rows(
    by_frame: dict[int, list[dict]],
    mot_number: dict[int, int],
    width: int,
    height: int,
    stats: dict[str, int],
) -> list[tuple[int, int, float, float, float, float]]:
    """Return MOT rows (mot_frame, track_id, left, top, w, h) for players and referees, updating stats."""
    rows = []
    for orig in sorted(by_frame):
        for row in by_frame[orig]:
            cls = row["class"]
            if cls not in KEPT_CLASSES:
                stats["skipped_other_class"] += 1
                continue
            track_id = int(row["track_id"])
            if track_id < 0:
                stats["dropped_negative_id"] += 1
                continue
            box = clamp_box(row, width, height)
            if box is None:
                stats["dropped_degenerate"] += 1
                continue
            left, top, w, h, clamped = box
            if clamped:
                stats["clamped_to_frame"] += 1
            stats[f"exported_{cls}"] += 1
            rows.append((mot_number[orig], track_id, left, top, w, h))
    return rows


def write_gt(gt_path: Path, rows: list[tuple[int, int, float, float, float, float]]) -> None:
    """Write MOT 1.1 detection lines with conf=1 and x=y=z=-1 (pre-fill, no 3D)."""
    gt_path.parent.mkdir(parents=True, exist_ok=True)
    with gt_path.open("w", newline="") as f:
        for mot_frame, track_id, left, top, w, h in rows:
            f.write(f"{mot_frame},{track_id},{left:.2f},{top:.2f},{w:.2f},{h:.2f},1,-1,-1,-1\n")


def write_images(video_path: Path, mot_number: dict[int, int], last_needed: int, img_dir: Path) -> int:
    """Decode the video front to back and save each kept frame as img1/NNNNNN.jpg. Returns the count written.

    Reads sequentially instead of seeking, so frame indices stay exact.
    Old JPGs in img1/ are removed first so a shorter re-export leaves no stale frames.
    """
    img_dir.mkdir(parents=True, exist_ok=True)
    for old in img_dir.glob("*.jpg"):
        old.unlink()
    cap = cv2.VideoCapture(str(video_path))
    written = 0
    try:
        for orig in range(last_needed + 1):
            ok, frame = cap.read()
            if not ok:
                raise SystemExit(f"Video ended at frame {orig}, before needed frame {last_needed}")
            if orig in mot_number:
                out = img_dir / f"{mot_number[orig]:06d}.jpg"
                if not cv2.imwrite(str(out), frame):
                    raise SystemExit(f"Could not write {out}")
                written += 1
    finally:
        cap.release()
    return written


def write_zip(zip_path: Path, gt_path: Path, img_dir: Path) -> None:
    """Bundle gt/gt.txt and img1/*.jpg into one archive for CVAT's MOT 1.1 importer."""
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as z:
        z.write(gt_path, arcname="gt/gt.txt")
        for img in sorted(img_dir.glob("*.jpg")):
            z.write(img, arcname=f"img1/{img.name}")


def validate(gt_path: Path, img_dir: Path, width: int, height: int) -> dict:
    """Read the written files back and check them against each other. Returns the numbers for the summary."""
    images = {int(p.stem) for p in img_dir.glob("*.jpg")}
    label_frames: set[int] = set()
    seen: set[tuple[int, int]] = set()
    boxes = out_of_bounds = duplicates = 0
    eps = 1e-6
    for line in gt_path.read_text().splitlines():
        parts = line.split(",")
        frame, track_id = int(parts[0]), int(parts[1])
        left, top, w, h = (float(v) for v in parts[2:6])
        boxes += 1
        label_frames.add(frame)
        if (frame, track_id) in seen:
            duplicates += 1
        seen.add((frame, track_id))
        if w <= 0 or h <= 0 or left < -eps or top < -eps or left + w > width + eps or top + h > height + eps:
            out_of_bounds += 1
    return {
        "images": len(images),
        "images_numbered_1_to_n": images == set(range(1, len(images) + 1)),
        "boxes": boxes,
        "distinct_track_ids": len({t for _, t in seen}),
        "label_frames_without_image": len(label_frames - images),
        "image_frames_without_label_rows": len(images - label_frames),
        "out_of_bounds_boxes": out_of_bounds,
        "duplicate_frame_id_pairs": duplicates,
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--tracks", type=Path, default=DEFAULT_TRACKS, help="tracker CSV (default: committed A0 run)")
    p.add_argument("--video", type=Path, default=DEFAULT_VIDEO, help="source video")
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR, help="output folder; must be under data/frames/ or runs/")
    p.add_argument("--start-s", type=float, default=DEFAULT_START_S, help="window start, seconds into the original video")
    p.add_argument("--duration-s", type=float, default=DEFAULT_DURATION_S, help="window length in seconds")
    p.add_argument("--stride", type=int, default=DEFAULT_STRIDE, help="keep every Nth frame")
    args = p.parse_args()

    if not any(is_under(args.out_dir, root) for root in IMAGE_SAFE_ROOTS):
        raise SystemExit(f"Refusing to write frames of minors outside {[str(r) for r in IMAGE_SAFE_ROOTS]}: {args.out_dir}")
    if args.stride < 1:
        raise SystemExit("--stride must be at least 1")

    fps, width, height, frame_count = get_video_info(args.video)
    start, end, kept = window_frames(args.start_s, args.duration_s, fps, args.stride)
    if end >= frame_count:
        raise SystemExit(f"Window ends at frame {end}, but the video has only {frame_count} frames")
    mot_number = {orig: n + 1 for n, orig in enumerate(kept)}

    by_frame, last_frame = read_tracks(args.tracks, set(kept))
    if last_frame < end:
        raise SystemExit(f"Tracker CSV ends at frame {last_frame}, before window end {end}")

    out_dir = args.out_dir
    img_dir = out_dir / "img1"
    gt_path = out_dir / "gt" / "gt.txt"
    zip_path = out_dir / "cvat_import.zip"
    sidecar_path = out_dir / "export_sidecar.json"

    stats = {k: 0 for k in STAT_KEYS}
    mot_rows = build_mot_rows(by_frame, mot_number, width, height, stats)
    write_gt(gt_path, mot_rows)
    n_images = write_images(args.video, mot_number, end, img_dir)
    if n_images != len(kept):
        raise SystemExit(f"Expected {len(kept)} images, wrote {n_images}")
    write_zip(zip_path, gt_path, img_dir)
    checks = validate(gt_path, img_dir, width, height)

    sidecar = {
        "source_tracks_csv": str(args.tracks),
        "source_video": str(args.video),
        "fps": fps,
        "frame_size_px": [width, height],
        "start_s": args.start_s,
        "duration_s": args.duration_s,
        "stride": args.stride,
        "window_first_frame": start,
        "window_last_frame": end,
        "window_frame_count": end - start + 1,
        "classes_kept": list(KEPT_CLASSES),
        "frame_map": [
            {"mot_frame": n + 1, "original_frame": orig, "time_s": round(orig / fps, 4)}
            for n, orig in enumerate(kept)
        ],
        "counts": stats,
        "validation": checks,
    }
    sidecar_path.write_text(json.dumps(sidecar, indent=2) + "\n")

    print(f"Exported {n_images} frames: MOT frames 1..{len(kept)} = original frames {kept[0]}..{kept[-1]}")
    print(f"  window {start}..{end} ({end - start + 1} frames), every {args.stride}th kept, fps {fps:.3f}")
    print(f"Boxes exported: {len(mot_rows)} (player {stats['exported_player']}, referee {stats['exported_referee']})")
    print(f"Dropped: negative track_id {stats['dropped_negative_id']}, degenerate after clamp {stats['dropped_degenerate']}")
    print(f"Kept boxes that were clamped to the frame edge: {stats['clamped_to_frame']}")
    print(f"Ball/rim rows skipped in window: {stats['skipped_other_class']}")
    print(f"Validation: {checks}")
    print(f"Images: {img_dir}\nMOT:    {gt_path}\nZIP:    {zip_path}\nSidecar: {sidecar_path}")


if __name__ == "__main__":
    main()

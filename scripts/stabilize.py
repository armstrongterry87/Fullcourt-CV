"""Estimate camera motion for a video and cache it (a "virtual static camera", design 005).

Writes a motion cache (numbers only). With --mosaic or --debug-video it also writes images or video
of the footage, which may show minors, so those go ONLY under data/frames/ or runs/ (git-ignored);
any other folder is refused.

Examples:
  # detections must exist first (scripts/baseline_track.py --save-detections)
  python scripts/stabilize.py --video data/raw/xbotgo.mp4 --detections runs/cache/xbotgo__wfc2f7c4e__small__thr0.25.npz
  python scripts/stabilize.py --video V --detections D.npz --mosaic --debug-video --max-seconds 30
  python scripts/stabilize.py --video V --detections D.npz --ignore-rect 0,0,300,80   # burned-in scoreboard
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from src.camera.motion import (  # noqa: E402
    MotionConfig, MotionTrack, cache_path, check_detection_meta, ensure_private_output, estimate_motion,
    expected_meta, file_sha256, load_detection_boxes, load_motion, motion_key, render_mosaic, save_motion,
    transform_points, video_fingerprint, video_frames,
)

QUALITY_COLOURS = {0.9: (0, 200, 0), 0.6: (0, 200, 255), 0.0: (0, 0, 255)}  # BGR green / amber / red


def parse_rect(text: str) -> tuple[int, int, int, int]:
    """'x1,y1,x2,y2' -> ints."""
    try:
        x1, y1, x2, y2 = (int(float(v)) for v in text.split(","))
    except ValueError:
        raise argparse.ArgumentTypeError("--ignore-rect needs x1,y1,x2,y2 (pixels)") from None
    return x1, y1, x2, y2


def draw_debug_frame(frame: np.ndarray, motion: MotionTrack, i: int, rims: list[tuple[int, tuple]]) -> np.ndarray:
    """Overlay: border coloured by quality, view-0 outline, a fixed grid of view points, and calibrated rims."""
    out = frame.copy()
    h, w = out.shape[:2]
    q = float(motion.quality[i])
    cv2.rectangle(out, (0, 0), (w - 1, h - 1), QUALITY_COLOURS[min(QUALITY_COLOURS, key=lambda k: abs(k - q))], 8)
    cv2.putText(out, f"f{i} q={q:.1f}", (12, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    H_fv = motion.H_frame_to_view(i, 0) if len(motion.views) else None
    if H_fv is None:
        return out
    H_vf = np.linalg.inv(H_fv)  # view -> frame
    v = motion.views[0]
    outline = transform_points(H_vf, np.array([[0, 0], [v.canvas_w, 0], [v.canvas_w, v.canvas_h], [0, v.canvas_h]], float))
    if np.all(np.isfinite(outline)) and np.abs(outline).max() < 1e5:
        cv2.polylines(out, [outline.astype(np.int32)], True, (255, 0, 255), 2)
    gx, gy = np.meshgrid(np.arange(0, v.canvas_w, 100.0), np.arange(0, v.canvas_h, 100.0))
    pts = transform_points(H_vf, np.column_stack([gx.ravel(), gy.ravel()]))
    for x, y in pts:  # if stabilisation works these dots stay glued to the walls
        if np.isfinite(x) and 0 <= x < w and np.isfinite(y) and 0 <= y < h:
            cv2.circle(out, (int(x), int(y)), 4, (0, 255, 255), -1)
    for view, box in rims:
        Hfv = motion.H_frame_to_view(i, view)
        if Hfv is None:
            continue
        c = transform_points(np.linalg.inv(Hfv), np.array([[(box[0] + box[2]) / 2, (box[1] + box[3]) / 2]]))[0]
        if np.all(np.isfinite(c)) and abs(c).max() < 1e5:
            cv2.circle(out, (int(c[0]), int(c[1])), max(6, int((box[2] - box[0]) / 2)), (255, 128, 0), 2)
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--video", type=Path, required=True)
    ap.add_argument("--detections", type=Path, required=True, help="detection cache .npz (runs/cache/...)")
    ap.add_argument("--out", type=Path, default=None,
                    help="motion cache to write (default runs/cache/<stem>__motion__<key>.npz)")
    ap.add_argument("--mosaic", nargs="?", const="", default=None, metavar="PATH",
                    help="write mosaic JPEGs (default data/frames/<stem>_mosaic.jpg, _v<k> for other views)")
    ap.add_argument("--debug-video", nargs="?", const="", default=None, metavar="PATH",
                    help="write an overlay video (default runs/<stem>_motion_debug.mp4)")
    ap.add_argument("--calibration", type=Path, default=None,
                    help="court JSON: its rims are drawn in the debug video and camera_static is checked")
    ap.add_argument("--ignore-rect", type=parse_rect, action="append", default=[], metavar="X1,Y1,X2,Y2",
                    help="mask a burned-in graphic (repeatable); part of the cache key")
    ap.add_argument("--max-seconds", type=float, default=None, help="only process this many seconds from the start")
    ap.add_argument("--force", action="store_true", help="recompute even if a matching cache exists")
    args = ap.parse_args(argv)

    stem = args.video.stem
    try:  # refuse bad image/video folders BEFORE doing any work
        mosaic_base = debug_path = None
        if args.mosaic is not None:
            mosaic_base = ensure_private_output(
                Path(args.mosaic) if args.mosaic else REPO_ROOT / "data" / "frames" / f"{stem}_mosaic.jpg")
        if args.debug_video is not None:
            debug_path = ensure_private_output(
                Path(args.debug_video) if args.debug_video else REPO_ROOT / "runs" / f"{stem}_motion_debug.mp4")
    except ValueError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2

    cfg = replace(MotionConfig(), ignore_rects_px=tuple(args.ignore_rect))
    fp = video_fingerprint(args.video)
    partial = args.max_seconds is not None
    n_needed = min(fp["n_frames"], int(args.max_seconds * fp["fps"])) if partial else fp["n_frames"]
    try:
        boxes, det_meta = load_detection_boxes(args.detections, n_needed)
        check_detection_meta(det_meta, fp, n_needed, partial=partial)
    except (ValueError, OSError, KeyError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    det_sha = file_sha256(args.detections)
    exp = expected_meta(fp, det_sha, cfg)
    key = motion_key(fp, det_sha, cfg)
    out = args.out or cache_path(args.video, key)
    if partial and args.out is None:  # a clip must never be mistaken for the full video's cache
        out = out.with_name(f"{out.stem}__first{n_needed}.npz")

    motion = None
    if out.exists() and not args.force:
        try:
            motion = load_motion(out, exp)
            if motion.n_frames != n_needed:
                raise ValueError(f"it holds {motion.n_frames} frames, not {n_needed}")
            print(f"reusing {out}")
        except ValueError as e:
            motion = None
            print(f"cache {out} not reusable ({e}); recomputing")
    if motion is None:
        print(f"estimating motion for {n_needed} frames at {fp['fps']:.3f} fps ...")
        motion = estimate_motion(lambda: video_frames(args.video, n_needed), boxes[:n_needed], fp["fps"], cfg,
                                 meta={**exp, "key": key})
        save_motion(out, motion)
        print(f"wrote {out}")

    q = motion.quality
    print(f"frames={motion.n_frames}  reliable(0.9)={np.mean(q >= 0.9):.1%}  usable(>=0.6)={np.mean(q >= 0.6):.1%}  "
          f"unreliable={np.mean(q <= 0):.1%}")
    print(f"keyframes={motion.meta['n_keyframes']}  views={len(motion.views)}  "
          f"max_closure_err_px={motion.meta['max_closure_err_px']:.2f}  "
          f"zoom_suspected={motion.meta['n_zoom_suspected']}  motion_id={motion.motion_id}")
    for w in motion.meta.get("warnings", []):
        print(f"warning: {w}")
    print(f"camera measured {'STATIC: set camera_static true' if motion.is_static else 'MOVING: set camera_static false'} "
          f"in the court JSON (scripts/calibrate_court.py --mosaic does this for you)")

    rims: list[tuple[int, tuple]] = []
    if args.calibration:
        from src.court.geometry import load_calibration
        calib = load_calibration(args.calibration)
        rims = [(r.view, r.box_px) for r in calib.rims]
        if calib.camera_static != motion.is_static:
            print(f"warning: {args.calibration} has camera_static={calib.camera_static} but the video measured "
                  f"{'static' if motion.is_static else 'moving'}")
        if calib.motion_id not in (None, motion.motion_id):
            print("warning: the calibration's motion_id differs from this motion file: recalibrate on the new mosaic")

    if mosaic_base is not None:
        for v in range(len(motion.views)):
            path = mosaic_base if v == 0 else mosaic_base.with_name(f"{mosaic_base.stem}_v{v}{mosaic_base.suffix}")
            img = render_mosaic(lambda: video_frames(args.video, n_needed), boxes[:n_needed], motion, v, cfg)
            path.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(path), img)
            print(f"wrote mosaic view {v}: {path} ({img.shape[1]}x{img.shape[0]})")

    if debug_path is not None:
        debug_path.parent.mkdir(parents=True, exist_ok=True)
        writer = None
        for i, frame in enumerate(video_frames(args.video, n_needed)):
            if writer is None:
                writer = cv2.VideoWriter(str(debug_path), cv2.VideoWriter_fourcc(*"mp4v"), fp["fps"],
                                         (frame.shape[1], frame.shape[0]))
            writer.write(draw_debug_frame(frame, motion, i, rims))
        if writer is not None:
            writer.release()
        print(f"wrote debug video: {debug_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

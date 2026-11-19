"""Click the court polygon on one frame of a video and save it as numbers-only JSON.

The polygon is in pixel space and is only valid for the camera position it was drawn on.
For data/raw/xbotgo.mp4 the camera is essentially locked off at frames 359-503 (15-21 s),
so draw it there and pass --valid-frames 359 503. Elsewhere on that clip the camera pans
and a fixed polygon will be wrong.

Privacy: this script DISPLAYS one frame. It never writes an image file (no thumbnail, no
screenshot, no debug image, under any flag). The only output is the polygon JSON in data/courts/, which holds numbers only.

Keys in the window:
  left click     add a corner point
  u              undo the last point
  s or Enter     save (needs at least 3 points)
  q or Esc       quit without saving

Examples:
  python scripts/calibrate_court.py --video data/raw/xbotgo.mp4 --valid-frames 359 503
  python scripts/calibrate_court.py --video data/raw/xbotgo.mp4 --frame-index 430 --out data/courts/xbotgo.json
  # non-interactive (testing only, no window): give the corners directly, in native pixels
  python scripts/calibrate_court.py --video data/raw/xbotgo.mp4 --points 100,100 1200,100 1200,650 100,650

Mosaic mode (panning camera, design 005 sec. 9.2): click the court and rims ONCE on the mosaic image written by
scripts/stabilize.py --mosaic, instead of on a raw frame. Output is numbers only (court JSON with camera_static
false, the motion_id, and a "view" on every landmark and rim). Repeat with --view 1, 2 ... for the other mosaics.
  python scripts/calibrate_court.py --mosaic data/frames/xbotgo_mosaic.jpg --motion runs/cache/xbotgo__motion__abcd1234.npz
  # non-interactive (testing): rim A box and landmarks "px,py:ft_x,ft_y" (feet in the rim A frame)
  python scripts/calibrate_court.py --mosaic M.jpg --motion X.npz --rim A:100,200,150,215 --landmark 50,300:-25,0 ...
Keys in the mosaic window: l landmark (then type its feet x,y in the console), a / b rim A / B (two clicks:
top-left then bottom-right), u undo, s save, q quit.

Install first (in the project venv):
  .venv/Scripts/python.exe -m pip install opencv-python numpy
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from src.camera.motion import load_motion  # noqa: E402
from src.court.geometry import load_calibration  # noqa: E402
from src.court.polygon import ConfigError, check_resolution, check_video_stem, load_court, save_court  # noqa: E402

DEFAULT_FRAME_INDEX = 430  # inside the locked-off range 359-503 of data/raw/xbotgo.mp4
LOCKED_OFF_RANGE = (359, 503)  # measured on xbotgo.mp4: camera static at 15-21 s
MIN_POINTS = 3
MAX_DISPLAY_WIDTH = 1280  # wider frames are shown scaled down; clicks are mapped back to native pixels
WINDOW_NAME = "Calibrate court: click corners | u=undo | s=save | q=quit"
COLOR_POLY = (0, 255, 255)  # BGR yellow
COLOR_TEXT = (255, 255, 255)


class PolygonBuilder:
    """The clicked corners, in native frame pixels. No OpenCV window needed to use it, so it is testable."""

    def __init__(self, frame_width: int, frame_height: int, scale: float = 1.0) -> None:
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.scale = scale  # display pixels per native pixel (1.0 unless the frame was scaled down)
        self.points: list[list[float]] = []

    def add(self, x_px: float, y_px: float) -> bool:
        """Add a corner given in native pixels. Returns False and adds nothing if it is off the frame."""
        if not (0 <= x_px < self.frame_width and 0 <= y_px < self.frame_height):
            return False
        self.points.append([float(x_px), float(y_px)])
        return True

    def undo(self) -> None:
        """Remove the most recent corner, if any."""
        if self.points:
            self.points.pop()

    def can_save(self) -> bool:
        """True once there are enough corners for a polygon."""
        return len(self.points) >= MIN_POINTS

    def on_mouse(self, event: int, x: int, y: int, flags: int, param: object) -> None:
        """OpenCV mouse callback. Left click adds a corner; x and y are in display pixels."""
        if event == cv2.EVENT_LBUTTONDOWN:
            self.add(round(x / self.scale), round(y / self.scale))


def video_info(video: Path) -> tuple[int, int, int]:
    """Return (width, height, frame_count) from the video header. No frames are decoded."""
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise SystemExit(f"Could not open video: {video}")
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return width, height, count


def read_frame(video: Path, frame_index: int) -> np.ndarray:
    """Return the 0-based frame as BGR. Reads forward from the start so the frame is exact."""
    cap = cv2.VideoCapture(str(video))
    frame = None
    for i in range(frame_index + 1):
        ok, frame = cap.read()
        if not ok:
            cap.release()
            raise SystemExit(f"Could not read frame {frame_index} (stopped at frame {i}).")
    cap.release()
    return frame


def draw_canvas(frame: np.ndarray, builder: PolygonBuilder, hint_lines: list[str]) -> np.ndarray:
    """Return a display copy of the frame with the polygon and hint text. Never saved anywhere."""
    s = builder.scale
    canvas = cv2.resize(frame, None, fx=s, fy=s) if s != 1.0 else frame.copy()
    pts = np.array([[x * s, y * s] for x, y in builder.points], dtype=np.int32)
    if len(pts) >= 2:
        cv2.polylines(canvas, [pts], isClosed=len(pts) >= MIN_POINTS, color=COLOR_POLY, thickness=2)
    for px, py in pts:
        cv2.circle(canvas, (int(px), int(py)), 5, COLOR_POLY, -1)
    for i, line in enumerate(hint_lines):
        y = 26 + 24 * i
        cv2.putText(canvas, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 4)
        cv2.putText(canvas, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR_TEXT, 1)
    return canvas


def interactive_pick(frame: np.ndarray, builder: PolygonBuilder, valid_frames: tuple[int, int] | None) -> bool:
    """Show the frame and let the student click corners. Returns True to save, False to quit."""
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback(WINDOW_NAME, builder.on_mouse)
    while True:
        hint = [
            f"Points: {len(builder.points)} (need {MIN_POINTS}+)   left click = add   u = undo",
            "s or Enter = save   q or Esc = quit without saving",
        ]
        if valid_frames is not None:
            hint.append(f"Polygon will be valid for frames {valid_frames[0]}-{valid_frames[1]} only")
        cv2.imshow(WINDOW_NAME, draw_canvas(frame, builder, hint))
        key = cv2.waitKey(30) & 0xFF
        if key == ord("u"):
            builder.undo()
        elif key in (ord("s"), 13):
            if builder.can_save():
                cv2.destroyAllWindows()
                return True
            print(f"Need at least {MIN_POINTS} points before saving (have {len(builder.points)}).")
        elif key in (ord("q"), 27):
            cv2.destroyAllWindows()
            return False
        if cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
            return False  # window closed with the X button


def parse_points(tokens: list[str]) -> list[tuple[float, float]]:
    """Parse ['x,y', 'x,y', ...] into float pairs."""
    pairs = []
    for tok in tokens:
        x_str, y_str = tok.split(",")
        pairs.append((float(x_str), float(y_str)))
    return pairs


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--video", type=Path, default=None, help="video to calibrate on, e.g. data/raw/xbotgo.mp4 "
                                                            "(not used in --mosaic mode)")
    p.add_argument("--mosaic", type=Path, default=None, help="mosaic image from scripts/stabilize.py --mosaic")
    p.add_argument("--motion", type=Path, default=None, help="motion cache (.npz) the mosaic was made from")
    p.add_argument("--view", type=int, default=0, help="which mosaic view the image is (default 0)")
    p.add_argument("--rim", action="append", default=[], metavar="ID:X1,Y1,X2,Y2",
                   help="mosaic mode, testing: a rim box in mosaic pixels, e.g. A:100,200,150,215")
    p.add_argument("--landmark", action="append", default=[], metavar="PX,PY:FTX,FTY",
                   help="mosaic mode, testing: a court landmark in mosaic pixels and feet (rim A frame)")
    p.add_argument("--level", default="nfhs", help="mosaic mode: court level (nfhs, ncaa, nba, fiba)")
    p.add_argument("--court-length-ft", type=float, default=None, help="mosaic mode: override the court length")
    p.add_argument(
        "--frame-index", type=int, default=DEFAULT_FRAME_INDEX,
        help=f"0-based frame to click on (default {DEFAULT_FRAME_INDEX}, inside the locked-off range "
             f"{LOCKED_OFF_RANGE[0]}-{LOCKED_OFF_RANGE[1]} of xbotgo.mp4). The polygon is only valid "
             "for the camera position in that frame range.",
    )
    p.add_argument(
        "--out", type=Path, default=None,
        help="output JSON (default: data/courts/<video stem>.json under the repo root)",
    )
    p.add_argument(
        "--valid-frames", nargs=2, type=int, metavar=("START", "END"), default=None,
        help="inclusive 0-based frame range the polygon applies to, e.g. '359 503' for xbotgo.mp4. "
             "Outside it every box is kept. Omit only if the camera never moves.",
    )
    p.add_argument(
        "--points", nargs="+", metavar="X,Y", default=None,
        help="testing only: skip the window and save these corners (native pixels), e.g. "
             "--points 100,100 1200,100 1200,650 100,650",
    )
    args = p.parse_args()
    if args.mosaic is None and args.video is None:
        p.error("--video is required (or use --mosaic with --motion)")
    if args.mosaic is not None and args.motion is None:
        p.error("--mosaic needs --motion (the motion cache it was made from)")
    if args.valid_frames is not None and not (0 <= args.valid_frames[0] <= args.valid_frames[1]):
        p.error("--valid-frames needs START <= END, both >= 0")
    if args.points is not None:
        try:
            parse_points(args.points)
        except ValueError:
            p.error("--points needs entries like 100,200 (two numbers separated by a comma)")
    return args


def save_polygon(
    builder: PolygonBuilder, out: Path, video: Path, frame_index: int,
    valid_frames: tuple[int, int] | None,
) -> None:
    """Save the polygon JSON, read it back through load_court, and print the summary and a reminder."""
    if not builder.can_save():
        raise SystemExit(f"Refusing to save: need at least {MIN_POINTS} points, have {len(builder.points)}.")
    existed = out.exists()
    save_court(out, video.stem, frame_index, builder.frame_width, builder.frame_height, builder.points, valid_frames)
    court = load_court(out)
    check_video_stem(court, video.stem)  # round-trip sanity checks, same as baseline_track.py
    check_resolution(court, builder.frame_width, builder.frame_height)

    print(f"{'Overwrote' if existed else 'Saved'} court polygon: {out}")
    print(f"  video: {video.stem}   frame: {frame_index}   size: {builder.frame_width}x{builder.frame_height}")
    print(f"  points ({len(court.polygon_px)}): {court.polygon_px}")
    if valid_frames is not None:
        print(f"  valid frames: {valid_frames[0]}-{valid_frames[1]} (inclusive)")
    else:
        print("  valid frames: none given, so the polygon applies to the WHOLE clip")
    print(
        "REMINDER: this polygon is pixel-space and only valid while the camera stays where it was.\n"
        f"  On xbotgo.mp4 that is frames {LOCKED_OFF_RANGE[0]}-{LOCKED_OFF_RANGE[1]}. "
        "Pass --valid-frames 359 503 for that clip. The camera pans around 8-11 s."
    )


# --------------------------------------------------------------------------- mosaic mode

class MosaicBuilder:
    """Rims and landmarks clicked on one mosaic view, in that view's canvas pixels. No window needed to use it."""

    def __init__(self, canvas_w: int, canvas_h: int, view: int) -> None:
        self.canvas_w, self.canvas_h, self.view = canvas_w, canvas_h, view
        self.rims: dict[str, list[float]] = {}
        self.points: list[dict] = []

    def _inside(self, x: float, y: float) -> bool:
        return 0 <= x < self.canvas_w and 0 <= y < self.canvas_h

    def add_rim(self, rim_id: str, x1: float, y1: float, x2: float, y2: float) -> None:
        if rim_id not in ("A", "B"):
            raise ConfigError("A rim id must be A or B.")
        x1, x2, y1, y2 = min(x1, x2), max(x1, x2), min(y1, y2), max(y1, y2)
        if x2 <= x1 or y2 <= y1 or not (self._inside(x1, y1) and self._inside(x2 - 1e-6, y2 - 1e-6)):
            raise ConfigError(f"Rim {rim_id} box {(x1, y1, x2, y2)} is empty or outside the "
                              f"{self.canvas_w}x{self.canvas_h} mosaic.")
        self.rims[rim_id] = [float(x1), float(y1), float(x2), float(y2)]

    def add_landmark(self, px: float, py: float, ft_x: float, ft_y: float) -> None:
        if not self._inside(px, py):
            raise ConfigError(f"Landmark ({px}, {py}) is outside the {self.canvas_w}x{self.canvas_h} mosaic.")
        self.points.append({"name": f"p{len(self.points)}", "px": [float(px), float(py)],
                            "ft": [float(ft_x), float(ft_y)], "view": self.view})


def merge_mosaic_json(existing: dict | None, builder: MosaicBuilder, *, video_stem: str, frame_w: int, frame_h: int,
                      motion_id: str, level: str, court_length_ft: float | None) -> dict:
    """Court JSON for a panning camera. Entries of other views are kept only if they came from the same motion file.

    A rim is clicked once: a new rim A replaces any older rim A (on any view). Landmarks of this view are replaced.
    """
    same = bool(existing) and existing.get("motion_id") == motion_id and existing.get("camera_static") is False
    d = dict(existing) if same else {}
    new_ids = set(builder.rims)
    rims = [r for r in d.get("rims", []) if r.get("view", 0) != builder.view and r.get("id") not in new_ids]
    rims += [{"id": i, "box_px": b, "source": "manual", "view": builder.view} for i, b in sorted(builder.rims.items())]
    pts = [p for p in d.get("homography_points", []) if p.get("view", 0) != builder.view] + builder.points
    for i, p in enumerate(pts):
        p["name"] = f"p{i}"
    d.update({"video_stem": video_stem, "frame_width": frame_w, "frame_height": frame_h, "level": level,
              "camera_static": False, "motion_id": motion_id, "rims": rims, "homography_points": pts})
    if court_length_ft is not None:
        d["court_length_ft"] = court_length_ft
    return d


def interactive_mosaic(img: np.ndarray, builder: MosaicBuilder) -> bool:
    """Click rims and landmarks on the mosaic. Returns True to save."""
    scale = min(1.0, MAX_DISPLAY_WIDTH / img.shape[1])
    disp = cv2.resize(img, None, fx=scale, fy=scale) if scale != 1.0 else img
    state = {"mode": "l", "clicks": []}
    name = "Mosaic: l=landmark a/b=rim u=undo s=save q=quit"

    def on_mouse(event, x, y, flags, param):
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        px, py = x / scale, y / scale
        try:
            if state["mode"] == "l":
                ans = input(f"feet x,y of the landmark at ({px:.0f}, {py:.0f}) in the rim A frame (blank to skip): ").strip()
                if ans:
                    fx, fy = (float(v) for v in ans.split(","))
                    builder.add_landmark(px, py, fx, fy)
            else:
                state["clicks"].append((px, py))
                if len(state["clicks"]) == 2:
                    (x1, y1), (x2, y2) = state["clicks"]
                    builder.add_rim(state["mode"].upper(), x1, y1, x2, y2)
                    state["clicks"] = []
        except (ValueError, ConfigError) as e:
            print(f"not added: {e}")
            state["clicks"] = []

    cv2.namedWindow(name, cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback(name, on_mouse)
    while True:
        canvas = disp.copy()
        for p in builder.points:
            cv2.circle(canvas, (int(p["px"][0] * scale), int(p["px"][1] * scale)), 5, COLOR_POLY, -1)
        for b in builder.rims.values():
            cv2.rectangle(canvas, (int(b[0] * scale), int(b[1] * scale)), (int(b[2] * scale), int(b[3] * scale)),
                          (255, 128, 0), 2)
        cv2.putText(canvas, f"mode={state['mode']} landmarks={len(builder.points)} rims={sorted(builder.rims)}",
                    (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR_TEXT, 2)
        cv2.imshow(name, canvas)
        key = cv2.waitKey(30) & 0xFF
        if key in (ord("l"), ord("a"), ord("b")):
            state["mode"], state["clicks"] = chr(key), []
        elif key == ord("u"):
            if builder.points:
                builder.points.pop()
        elif key in (ord("s"), 13):
            cv2.destroyAllWindows()
            return True
        elif key in (ord("q"), 27) or cv2.getWindowProperty(name, cv2.WND_PROP_VISIBLE) < 1:
            cv2.destroyAllWindows()
            return False


def mosaic_main(args: argparse.Namespace) -> None:
    """Calibrate on a mosaic: read the image and motion cache, collect clicks, write the court JSON."""
    import json

    motion = load_motion(args.motion)
    if not (0 <= args.view < len(motion.views)):
        raise SystemExit(f"--view {args.view} is not in the motion file (it has {len(motion.views)} views).")
    v = motion.views[args.view]
    img = cv2.imread(str(args.mosaic))
    if img is None:
        raise SystemExit(f"Could not read mosaic image: {args.mosaic}")
    if (img.shape[1], img.shape[0]) != (v.canvas_w, v.canvas_h):
        raise SystemExit(f"{args.mosaic} is {img.shape[1]}x{img.shape[0]} but view {args.view} of the motion file is "
                         f"{v.canvas_w}x{v.canvas_h}: it was made from a different motion file.")
    builder = MosaicBuilder(v.canvas_w, v.canvas_h, args.view)
    try:
        for r in args.rim:
            rid, box = r.split(":")
            builder.add_rim(rid.upper(), *(float(t) for t in box.split(",")))
        for lm in args.landmark:
            a, b = lm.split(":")
            builder.add_landmark(*(float(t) for t in a.split(",")), *(float(t) for t in b.split(",")))
    except (ValueError, ConfigError) as e:
        raise SystemExit(f"error: {e}") from None
    if not (args.rim or args.landmark) and not interactive_mosaic(img, builder):
        print("Quit without saving. No file written.")
        return
    stem = motion.meta["fingerprint"]["video_stem"]
    out = args.out or REPO_ROOT / "data" / "courts" / f"{stem}_mosaic.json"
    existing = json.loads(out.read_text()) if out.exists() else None
    d = merge_mosaic_json(existing, builder, video_stem=stem, frame_w=motion.frame_wh[0], frame_h=motion.frame_wh[1],
                          motion_id=motion.motion_id, level=args.level, court_length_ft=args.court_length_ft)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(d, indent=2))
    print(f"Saved mosaic calibration: {out}  (view {args.view}; {len(d['rims'])} rims, "
          f"{len(d['homography_points'])} landmarks)")
    if len(d["homography_points"]) >= 4:
        try:
            load_calibration(out, motion=motion)
            print("  round-trip check passed: one homography fitted in reference pixels")
        except ConfigError as e:
            print(f"  WARNING: {e}")
    else:
        print("  note: fewer than 4 landmarks so far; add more (any view) before scoring")
    print("  camera_static is false; scoring needs --motion with this motion file (motion_id is stored).")


def main() -> None:
    args = parse_args()
    if args.mosaic is not None:
        mosaic_main(args)
        return
    width, height, total = video_info(args.video)
    if args.frame_index < 0 or (total and args.frame_index >= total):
        raise SystemExit(f"--frame-index {args.frame_index} is outside the video (0-{total - 1}).")
    if args.valid_frames is not None and total and args.valid_frames[1] >= total:
        raise SystemExit(f"--valid-frames end {args.valid_frames[1]} is past the last frame ({total - 1}).")

    valid = tuple(args.valid_frames) if args.valid_frames is not None else None
    out = args.out or REPO_ROOT / "data" / "courts" / f"{args.video.stem}.json"

    if args.points is not None:
        # non-interactive path for tests: no window, no frame decoded
        builder = PolygonBuilder(width, height)
        for x_px, y_px in parse_points(args.points):
            if not builder.add(x_px, y_px):
                raise SystemExit(f"Point ({x_px}, {y_px}) is outside the {width}x{height} frame.")
        save_polygon(builder, out, args.video, args.frame_index, valid)
        return

    frame = read_frame(args.video, args.frame_index)
    scale = min(1.0, MAX_DISPLAY_WIDTH / width)
    builder = PolygonBuilder(width, height, scale)
    if interactive_pick(frame, builder, valid):
        save_polygon(builder, out, args.video, args.frame_index, valid)
    else:
        print("Quit without saving. No file written.")


if __name__ == "__main__":
    main()

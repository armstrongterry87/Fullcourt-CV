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

from src.court.polygon import check_resolution, check_video_stem, load_court, save_court  # noqa: E402

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
    p.add_argument("--video", type=Path, required=True, help="video to calibrate on, e.g. data/raw/xbotgo.mp4")
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


def main() -> None:
    args = parse_args()
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

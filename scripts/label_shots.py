"""Log basketball shots by hand while watching a video. Writes a numbers-only CSV.

Keys in the window:
  Space          pause / resume
  Left / Right   step one frame back / forward (works paused or playing)
  Up / Down      jump 1 second back / forward
  m              mark a make (adds a row)
  x              mark a miss (adds a row)
  2 / 3 / f      shot type: 2-point, 3-point, free throw (1 point)
  u              undo the last row added this session
  q or Esc       save and quit (the file is also saved after every change)

How a row is built (decided here, documented in docs/tasks/004-questions.md, Q5-Q7):
  - m or x creates the row. The row's shot type comes from a type key pressed just
    BEFORE the result (kept as the "next shot type") or just AFTER it (amends that row).
    Each row takes at most one type. A type key after a row has already been typed
    sets the type for the next shot.
  - If no type key is pressed for a shot, its type is 2 points.
  - points = shot VALUE (1 free throw, 2, or 3), not points scored. It is set on misses
    too, so 2/3 accuracy can be scored on misses. Points scored = points if result is
    make, else 0. The scorer does that calculation.
  - undo removes the last row added in this session. Rows already in an existing CSV
    are never removed by undo. If there is nothing to undo, nothing happens.

Which frame a label gets:
  - The frame shown on screen at the moment the key is pressed. Paused or playing.
  - time_s = frame / fps, with fps read from the video (never assumed).
  - Playing means the key lands roughly a reaction time (about 0.2-0.4 s) after the
    shot. Pause on the shot when you can. The scorer matches within +-1 s.

Output: data/labels/<video_stem>_shots.csv with columns frame, time_s, result, points, notes.
  - If the CSV already exists, its rows are loaded and kept, and new rows are added.
    It is never overwritten. A file with a different header stops the script.
  - notes is always empty from this tool. Edit it by hand if needed.

Privacy: the window shows footage of minors. This script writes no image. The only
output is the label CSV, which holds numbers and short text only.

Examples:
  python scripts/label_shots.py --video data/raw/xbotgo.mp4
  python scripts/label_shots.py --video data/raw/xbotgo.mp4 --start-frame 600
  # non-interactive (testing only, no window): replay key tokens, then save
  python scripts/label_shots.py --video data/raw/xbotgo.mp4 --keys right,m,3,right,x,u --out runs/selftest_xbotgo_shots.csv

Install first (in the project venv): .venv/Scripts/python.exe -m pip install opencv-python numpy
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path
from typing import Callable

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

COLUMNS = ["frame", "time_s", "result", "points", "notes"]
RESULTS = ("make", "miss")
DEFAULT_POINTS = 2  # shot value when no type key is pressed
TYPE_POINTS = {"2": 2, "3": 3, "f": 1}  # shot value of each type key (free throw = 1)
STEP_TOKENS = {"left": -1, "right": 1}  # up/down use one second, see LabelSession.handle
ARROW_TOKENS = ("left", "right", "up", "down")
VALID_TOKENS = ("space", "left", "right", "up", "down", "m", "x", "2", "3", "f", "u", "q")

# waitKeyEx codes for the arrow keys: Windows first, then X11/GTK builds of OpenCV
ARROW_CODES = {
    2424832: "left", 2490368: "up", 2555904: "right", 2621440: "down",
    65361: "left", 65362: "up", 65363: "right", 65364: "down",
}

WINDOW_NAME = "Label shots: Space play/pause | m make | x miss | 2/3/f type | u undo | q save+quit"
MAX_DISPLAY_WIDTH = 1280  # wider frames are shown scaled down; the frame index is unaffected
COLOR_TEXT = (255, 255, 255)


class LabelSession:
    """All labelling state: current frame, rows, and the rules above. No window needed, so it is testable."""

    def __init__(self, video_stem: str, fps: float, total_frames: int, out_path: Path, start_frame: int = 0) -> None:
        self.video_stem = video_stem
        self.fps = fps
        self.total_frames = total_frames
        self.out_path = out_path
        self.last_frame = total_frames - 1
        self.frame = min(max(start_frame, 0), self.last_frame)
        self.rows: list[dict] = []
        self.n_loaded = 0  # rows[:n_loaded] came from an existing file and undo never removes them
        self.open_row: dict | None = None  # last row created, while its shot type is still unset
        self.pending_points: int | None = None  # type key pressed before the result key
        self.skipped = 0  # undo presses with nothing to undo
        self.message = ""

    def time_of(self, frame: int) -> float:
        """Seconds for a 0-based frame, using the video's own fps."""
        return round(frame / self.fps, 3)

    def load_rows(self, rows: list[dict]) -> None:
        """Start from rows already in the CSV. They are kept and can't be undone."""
        self.rows = list(rows)
        self.n_loaded = len(rows)

    def handle(self, token: str) -> None:
        """Apply one key token (see VALID_TOKENS). Labels are saved after each change."""
        if token == "m":
            self.mark("make")
        elif token == "x":
            self.mark("miss")
        elif token in TYPE_POINTS:
            self.set_type(TYPE_POINTS[token])
        elif token == "u":
            self.undo()
        elif token in STEP_TOKENS:
            self.step(STEP_TOKENS[token])
        elif token == "up":
            self.step(int(round(self.fps)))
        elif token == "down":
            self.step(-int(round(self.fps)))
        elif token in ("space", "q"):
            self.message = f"'{token}' only does something in the window"
        else:
            raise ValueError(f"unknown key token: {token}")

    def step(self, delta: int) -> None:
        """Move the current frame, clamped to the video."""
        self.frame = min(max(self.frame + delta, 0), self.last_frame)

    def mark(self, result: str) -> None:
        """Add a row for the frame on screen. Its type is the pending type, else the default."""
        if result not in RESULTS:
            raise ValueError(f"result must be make or miss, not {result}")
        points = self.pending_points if self.pending_points is not None else DEFAULT_POINTS
        row = {"frame": self.frame, "time_s": self.time_of(self.frame), "result": result, "points": points, "notes": ""}
        self.rows.append(row)
        self.open_row = row if self.pending_points is None else None
        self.pending_points = None
        typed = "" if self.open_row is None else " (default 2-pt: press 2/3/f to change)"
        self.message = f"{result} at frame {self.frame} = {points}-pt shot{typed}"
        self.save()

    def set_type(self, points: int) -> None:
        """Type the most recent shot if it has no type yet. Otherwise this is the type of the next shot."""
        if self.open_row is not None:
            self.open_row["points"] = points
            self.message = f"{self.open_row['result']} at frame {self.open_row['frame']} set to {points}-pt"
            self.open_row = None
            self.save()
        else:
            self.pending_points = points
            self.message = f"next shot type: {points}-pt"

    def undo(self) -> None:
        """Remove the last row added this session. Does nothing if there is none."""
        if len(self.rows) <= self.n_loaded:
            self.skipped += 1
            self.message = "nothing to undo in this session"
            return
        row = self.rows.pop()
        if row is self.open_row:
            self.open_row = None
        self.message = f"undid {row['result']} at frame {row['frame']}"
        self.save()

    def save(self) -> None:
        """Write the CSV atomically (temp file, then rename), sorted by frame."""
        self.out_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.out_path.parent / (self.out_path.name + ".tmp")
        with tmp.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=COLUMNS)
            writer.writeheader()
            for row in sorted(self.rows, key=lambda r: r["frame"]):
                writer.writerow(row)
        os.replace(tmp, self.out_path)

    def pending_text(self) -> str:
        """Short text for the on-screen status line."""
        if self.pending_points is not None:
            return f"next shot type: {self.pending_points}-pt"
        return "shot type: 2-pt unless you press 2/3/f"


def load_existing(path: Path, total_frames: int) -> list[dict]:
    """Read rows from an existing label CSV. Stops with an error on a bad file, so nothing is overwritten."""
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        if reader.fieldnames != COLUMNS:
            raise SystemExit(f"{path} has columns {reader.fieldnames}, expected {COLUMNS}. Not touching it.")
        rows = []
        for line_no, raw in enumerate(reader, start=2):
            try:
                row = {
                    "frame": int(raw["frame"]),
                    "time_s": float(raw["time_s"]),
                    "result": raw["result"],
                    "points": int(raw["points"]),
                    "notes": raw["notes"],
                }
            except (TypeError, ValueError) as exc:
                raise SystemExit(f"{path} line {line_no}: cannot parse row ({exc}). Not touching it.")
            if row["result"] not in RESULTS:
                raise SystemExit(f"{path} line {line_no}: result must be make or miss. Not touching it.")
            if not (0 <= row["frame"] < total_frames):
                raise SystemExit(f"{path} line {line_no}: frame {row['frame']} is outside the video. Not touching it.")
            rows.append(row)
    return rows


def video_info(video: Path) -> tuple[float, int, int, int]:
    """Return (fps, frame_count, width, height) from the header. Nothing is decoded."""
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise SystemExit(f"Could not open video: {video}")
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    if fps <= 0 or total <= 0:
        raise SystemExit(f"Video header has no usable fps ({fps}) or frame count ({total}): {video}")
    return fps, total, width, height


class FrameReader:
    """Returns frames by 0-based index. Reads forward when it can and seeks only when it has to."""

    def __init__(self, open_fn: Callable[[], object]) -> None:
        self.open_fn = open_fn
        self.cap = open_fn()
        self.next_index = 0  # index the open capture will return next
        self.cached_index: int | None = None
        self.cached_frame: np.ndarray | None = None

    def get(self, index: int) -> np.ndarray | None:
        """Return the BGR frame at index, or None if the video ends before it."""
        if index == self.cached_index:
            return self.cached_frame
        if index != self.next_index:
            ok, frame = self._seek_and_read(index)
        else:
            ok, frame = self.cap.read()
        if not ok:
            return None
        self.next_index = index + 1
        self.cached_index, self.cached_frame = index, frame
        return frame

    def _seek_and_read(self, index: int) -> tuple[bool, np.ndarray | None]:
        """Seek, then check the decoder landed on the right frame. If not, reopen and read forward (slow, exact)."""
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, index)
        ok, frame = self.cap.read()
        landed = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES)) - 1
        if ok and landed == index:
            return ok, frame
        self.cap.release()
        self.cap = self.open_fn()
        ok, frame = False, None
        for _ in range(index + 1):
            ok, frame = self.cap.read()
            if not ok:
                return False, None
        return ok, frame


def key_to_token(key: int) -> str | None:
    """Map a waitKeyEx code to a token from VALID_TOKENS, or None if it is not one of ours."""
    if key < 0:
        return None
    if key in ARROW_CODES:
        return ARROW_CODES[key]
    low = key & 0xFF
    if low == 32:
        return "space"
    if low == 27:
        return "q"
    if 32 <= low < 127:
        ch = chr(low).lower()
        if ch in VALID_TOKENS:
            return ch
    return None


def parse_tokens(text: str) -> list[str]:
    """Parse 'right,m,3' into tokens. Raises ValueError on an unknown token."""
    tokens = [t.strip().lower() for t in text.split(",") if t.strip()]
    for tok in tokens:
        if tok not in VALID_TOKENS:
            raise ValueError(f"unknown key token '{tok}' (valid: {', '.join(VALID_TOKENS)})")
    return tokens


def draw(frame: np.ndarray, session: LabelSession, playing: bool) -> np.ndarray:
    """Return a display copy with the status and key hints. Never saved anywhere."""
    canvas = frame.copy()
    if canvas.shape[1] > MAX_DISPLAY_WIDTH:
        scale = MAX_DISPLAY_WIDTH / canvas.shape[1]
        canvas = cv2.resize(canvas, None, fx=scale, fy=scale)
    makes = sum(1 for r in session.rows if r["result"] == "make")
    misses = len(session.rows) - makes
    state = "PLAYING" if playing else "PAUSED"
    lines = [
        f"frame {session.frame} / {session.last_frame}   t={session.time_of(session.frame):.2f}s   {state}",
        f"rows {len(session.rows)} (make {makes}, miss {misses})   {session.pending_text()}",
        session.message,
        "Space play/pause | Left/Right +-1 frame | Up/Down +-1 s",
        "m make | x miss | 2/3/f type | u undo | q save+quit",
    ]
    for i, line in enumerate(lines):
        y = 26 + 24 * i
        cv2.putText(canvas, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 4)
        cv2.putText(canvas, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR_TEXT, 1)
    return canvas


def read_current(reader: FrameReader, session: LabelSession, playing: bool) -> np.ndarray:
    """Return the frame at session.frame. If the video ends early, move the end back to the last readable frame."""
    frame = reader.get(session.frame)
    if frame is None:
        if session.frame == 0:
            raise SystemExit("The video has no readable frames.")
        session.last_frame = session.frame - 1
        session.frame = session.last_frame
        session.message = f"video ends early: last readable frame is {session.last_frame}"
        frame = reader.get(session.frame)
    return frame


def run_window(session: LabelSession, reader: FrameReader) -> None:
    """Show the video and handle keys until q or the window is closed. Each keypress applies to the frame on screen."""
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_AUTOSIZE)
    playing = False
    delay_ms = max(1, int(round(1000.0 / session.fps)))
    frame = read_current(reader, session, playing)
    cv2.imshow(WINDOW_NAME, draw(frame, session, playing))
    while True:
        key = cv2.waitKeyEx(delay_ms if playing else 30)
        token = key_to_token(key)
        if token == "q":
            break
        if token == "space":
            playing = not playing
        elif token in ARROW_TOKENS:
            playing = False
            session.handle(token)
        elif token is not None:
            session.handle(token)
        if playing:
            if session.frame < session.last_frame:
                session.frame += 1  # the frame shown this loop is the one any key pressed next applies to
            else:
                playing = False
        frame = read_current(reader, session, playing)
        cv2.imshow(WINDOW_NAME, draw(frame, session, playing))
        if cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
            break  # window closed with the X button
    cv2.destroyAllWindows()


def print_summary(session: LabelSession, n_loaded_before: int, fps: float, total: int) -> None:
    """Print what was written and what was skipped."""
    rows = session.rows
    makes = [r for r in rows if r["result"] == "make"]
    misses = [r for r in rows if r["result"] == "miss"]
    by_value = {v: sum(1 for r in rows if r["points"] == v) for v in (1, 2, 3)}
    points_scored = sum(r["points"] for r in makes)
    print(f"Wrote: {session.out_path}")
    print(f"  video: {session.video_stem}   fps: {fps:.3f}   frames: {total}")
    print(f"  rows: {len(rows)} ({len(rows) - session.n_loaded} added this run, {n_loaded_before} were already there)")
    print(f"  makes: {len(makes)}   misses: {len(misses)}")
    print(f"  shot values: 1-pt (free throw) {by_value[1]}   2-pt {by_value[2]}   3-pt {by_value[3]}")
    print(f"  points scored (makes only): {points_scored}")
    print(f"  skipped: {session.skipped} undo press(es) with nothing to undo")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--video", type=Path, required=True, help="video to label, e.g. data/raw/xbotgo.mp4")
    p.add_argument(
        "--out", type=Path, default=None,
        help="label CSV (default: data/labels/<video stem>_shots.csv under the repo root). Existing rows are kept.",
    )
    p.add_argument("--start-frame", type=int, default=0, help="0-based frame to open on (default 0)")
    p.add_argument(
        "--keys", default=None, metavar="TOKENS",
        help="testing only: no window. Comma-separated key tokens, e.g. right,m,3,right,x,u. "
             f"Valid: {', '.join(VALID_TOKENS)}",
    )
    args = p.parse_args()
    if args.start_frame < 0:
        p.error("--start-frame must be >= 0")
    if args.keys is not None:
        try:
            parse_tokens(args.keys)
        except ValueError as exc:
            p.error(str(exc))
    return args


def main() -> None:
    args = parse_args()
    fps, total, _width, _height = video_info(args.video)
    if args.start_frame >= total:
        raise SystemExit(f"--start-frame {args.start_frame} is past the last frame ({total - 1}).")
    out = args.out or REPO_ROOT / "data" / "labels" / f"{args.video.stem}_shots.csv"

    existing = load_existing(out, total)
    session = LabelSession(args.video.stem, fps, total, out, args.start_frame)
    session.load_rows(existing)
    n_loaded_before = len(existing)

    if args.keys is not None:
        for token in parse_tokens(args.keys):
            session.handle(token)
    else:
        reader = FrameReader(lambda: cv2.VideoCapture(str(args.video)))
        run_window(session, reader)

    session.save()
    print_summary(session, n_loaded_before, fps, total)


if __name__ == "__main__":
    main()

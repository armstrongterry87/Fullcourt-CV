"""Tests for scripts/label_shots.py: label logic, CSV round trip, and the window loop with GUI calls mocked.

No window is opened and no image is written. The capture is a fake that returns blank frames.
"""

from __future__ import annotations

import csv
import importlib.util
from pathlib import Path

import cv2
import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("label_shots", REPO_ROOT / "scripts" / "label_shots.py")
label_shots = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(label_shots)

FPS = 23.951
TOTAL = 719
RIGHT_ARROW = 2555904  # waitKeyEx code for the right arrow on Windows


def _session(tmp_path, start=0, total=TOTAL):
    return label_shots.LabelSession("clip", FPS, total, tmp_path / "clip_shots.csv", start)


def _read_csv(path):
    with path.open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


class FakeCapture:
    """Mimics the cv2.VideoCapture calls the script uses. Frames are blank 16x32 images."""

    def __init__(self, n_frames: int = TOTAL) -> None:
        self.n = n_frames
        self.pos = 0  # index of the next frame read() returns

    def isOpened(self) -> bool:
        return True

    def read(self):
        if self.pos >= self.n:
            return False, None
        self.pos += 1
        return True, np.zeros((16, 32, 3), dtype=np.uint8)

    def set(self, prop, value) -> bool:
        if prop == cv2.CAP_PROP_POS_FRAMES:
            self.pos = int(value)
        return True

    def get(self, prop) -> float:
        if prop == cv2.CAP_PROP_POS_FRAMES:
            return float(self.pos)
        if prop == cv2.CAP_PROP_FPS:
            return FPS
        if prop == cv2.CAP_PROP_FRAME_COUNT:
            return float(self.n)
        return 0.0

    def release(self) -> None:
        pass


def _mock_gui(monkeypatch, keys):
    """Replace every GUI call with a stub. keys is an iterator of waitKeyEx codes."""
    monkeypatch.setattr(cv2, "namedWindow", lambda *a, **k: None)
    monkeypatch.setattr(cv2, "destroyAllWindows", lambda: None)
    monkeypatch.setattr(cv2, "imshow", lambda name, img: None)
    monkeypatch.setattr(cv2, "waitKeyEx", lambda delay: next(keys))
    monkeypatch.setattr(cv2, "getWindowProperty", lambda name, prop: 1.0)


def test_make_then_type_after_amends_row(tmp_path):
    s = _session(tmp_path)
    s.handle("m")
    s.handle("3")
    assert s.rows[0]["result"] == "make" and s.rows[0]["points"] == 3


def test_type_before_result_applies_to_next_shot(tmp_path):
    s = _session(tmp_path)
    s.handle("f")
    s.handle("x")
    assert s.rows[0]["result"] == "miss" and s.rows[0]["points"] == 1
    s.handle("m")  # the pending type was used up, so this one takes the default
    assert s.rows[1]["points"] == 2


def test_type_after_typed_row_is_pending_for_next(tmp_path):
    s = _session(tmp_path)
    s.handle("m")
    s.handle("3")
    s.handle("2")  # the row is already typed, so this is the next shot's type
    s.handle("x")
    assert [r["points"] for r in s.rows] == [3, 2]


def test_default_points_when_no_type_pressed(tmp_path):
    s = _session(tmp_path)
    s.handle("x")
    assert s.rows[0]["points"] == 2


def test_time_uses_fps_and_frame_index(tmp_path):
    s = _session(tmp_path, start=697)
    s.handle("m")
    assert s.rows[0]["frame"] == 697
    assert s.rows[0]["time_s"] == round(697 / FPS, 3)


def test_step_is_clamped(tmp_path):
    s = _session(tmp_path, start=0)
    s.handle("left")
    assert s.frame == 0
    s.handle("down")
    assert s.frame == 0
    s.handle("up")
    assert s.frame == round(FPS)
    s.step(10 ** 6)
    assert s.frame == TOTAL - 1


def test_undo_removes_last_row(tmp_path):
    s = _session(tmp_path)
    s.handle("m")
    s.handle("x")
    s.handle("u")
    assert [r["result"] for r in s.rows] == ["make"]


def test_undo_with_zero_rows_does_not_crash(tmp_path):
    s = _session(tmp_path)
    s.handle("u")
    s.handle("u")
    assert s.rows == [] and s.skipped == 2
    assert not (tmp_path / "clip_shots.csv").exists()  # nothing changed, so nothing written


def test_undo_never_removes_loaded_rows(tmp_path):
    s = _session(tmp_path)
    s.load_rows([{"frame": 5, "time_s": 0.209, "result": "make", "points": 2, "notes": ""}])
    s.handle("u")
    assert len(s.rows) == 1 and s.skipped == 1


def test_save_round_trip_appends_to_existing(tmp_path):
    s = _session(tmp_path)
    s.handle("m")
    s.handle("3")  # make at frame 0, 3-pt
    again = _session(tmp_path, start=100)
    again.load_rows(label_shots.load_existing(again.out_path, TOTAL))
    again.handle("x")  # miss at frame 100
    rows = _read_csv(tmp_path / "clip_shots.csv")
    assert list(rows[0].keys()) == ["frame", "time_s", "result", "points", "notes"]
    assert [(r["frame"], r["result"], r["points"]) for r in rows] == [("0", "make", "3"), ("100", "miss", "2")]


def test_existing_file_with_wrong_header_is_not_touched(tmp_path):
    path = tmp_path / "clip_shots.csv"
    path.write_text("a,b\n1,2\n", encoding="utf-8")
    with pytest.raises(SystemExit):
        label_shots.load_existing(path, TOTAL)
    assert path.read_text(encoding="utf-8") == "a,b\n1,2\n"


def test_parse_tokens_rejects_unknown():
    assert label_shots.parse_tokens("Right, m ,3") == ["right", "m", "3"]
    with pytest.raises(ValueError):
        label_shots.parse_tokens("m,k")


def test_key_codes_map_to_tokens():
    assert label_shots.key_to_token(-1) is None
    assert label_shots.key_to_token(ord("m")) == "m"
    assert label_shots.key_to_token(ord("M")) == "m"
    assert label_shots.key_to_token(32) == "space"
    assert label_shots.key_to_token(RIGHT_ARROW) == "right"
    assert label_shots.key_to_token(65362) == "up"
    assert label_shots.key_to_token(ord("k")) is None


def test_window_loop_mocked_label_type_undo_save(tmp_path, monkeypatch):
    """Drive run_window with GUI calls mocked: label, type, undo, undo with nothing left, save."""
    keys = iter([
        ord("m"),         # make at frame 0 (type not set yet)
        ord("3"),         # amend that make to 3-pt
        RIGHT_ARROW,      # frame 1
        RIGHT_ARROW,      # frame 2
        ord("x"),         # miss at frame 2
        ord("u"),         # undo the miss
        RIGHT_ARROW,      # frame 3
        ord("f"),         # pending free throw for the next shot
        ord("m"),         # make at frame 3, 1-pt
        ord("u"),         # undo that make
        ord("u"),         # undo the make at frame 0
        ord("u"),         # nothing left: must not crash
        ord("3"),         # pending 3-pt
        ord("x"),         # miss at frame 3, 3-pt
        ord("q"),         # quit (saves)
    ])
    _mock_gui(monkeypatch, keys)
    s = _session(tmp_path)
    label_shots.run_window(s, label_shots.FrameReader(FakeCapture))
    s.save()

    rows = _read_csv(tmp_path / "clip_shots.csv")
    assert [(r["frame"], r["result"], r["points"]) for r in rows] == [("3", "miss", "3")]
    assert s.skipped == 1


def test_window_loop_playing_labels_the_frame_on_screen(tmp_path, monkeypatch):
    """Space starts playback. Each idle loop advances one frame, and a key labels the frame shown then."""
    keys = iter([
        32,               # space: start playing; this loop already shows frame 1
        -1,               # idle loop: frame 2
        -1,               # idle loop: frame 3
        -1,               # idle loop: frame 4
        ord("x"),         # miss on frame 4, the frame on screen when the key arrived
        ord("q"),
    ])
    _mock_gui(monkeypatch, keys)
    s = _session(tmp_path)
    label_shots.run_window(s, label_shots.FrameReader(FakeCapture))
    s.save()
    rows = _read_csv(tmp_path / "clip_shots.csv")
    assert [(r["frame"], r["result"]) for r in rows] == [("4", "miss")]


def test_window_loop_stops_at_end_of_video(tmp_path, monkeypatch):
    """If the video ends before the header frame count, the session end moves back and nothing crashes."""
    keys = iter([RIGHT_ARROW, RIGHT_ARROW, RIGHT_ARROW, ord("m"), ord("q")])
    _mock_gui(monkeypatch, keys)
    s = _session(tmp_path, start=0, total=10)
    label_shots.run_window(s, label_shots.FrameReader(lambda: FakeCapture(n_frames=3)))
    assert s.last_frame == 2 and s.frame == 2
    assert [(r["frame"], r["result"]) for r in s.rows] == [(2, "make")]

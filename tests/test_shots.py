"""Tests for task 004 step 2: court geometry, trajectory cleaning, shot detection.

All data is synthetic (no real footage, no image is written). Scene tests run at two frame
rates (23.951 and 60) so nothing can hide a hard-coded 30 fps.

Camera model used by the scenes: a sideline camera. Court feet (x across the court, y from
the baseline) map to pixels through `H_FT2PX` (with a little perspective). Rim A sits at the
box RIM_BOX. Image y grows downward, so "above the rim" is a smaller y.
"""

from __future__ import annotations

import json
import math

import numpy as np
import pandas as pd
import pytest

from src.court.geometry import (
    Rim, fit_homography, get_spec, is_three, load_calibration, px_to_ft, three_margin_ft, to_rim_frame,
)
from src.court.polygon import ConfigError, load_court, save_court
from src.shots.config import ShotConfig
from src.shots.detect import COLUMNS, detect_shots, judge_outcome, write_shots_csv
from src.shots.detect import _in_zone  # noqa: PLC2701
from src.shots.trajectory import Trajectory, clean_ball_track

FPS_LIST = [23.951, 60.0]
H_FT2PX = np.array([[0.0, 20.0, 100.0], [-12.0, 0.0, 500.0], [0.0004, 0.0008, 1.0]])
RIM_BOX = (170.0, 380.0, 240.0, 404.0)   # W = 70, top 380, bottom 404, centre (205, 392)
RIM = Rim("A", RIM_BOX, "manual")
BALL_W = 38.0
CFG = ShotConfig()


def ft_to_px(x_ft: float, y_ft: float) -> tuple[float, float]:
    p = H_FT2PX @ np.array([x_ft, y_ft, 1.0])
    return float(p[0] / p[2]), float(p[1] / p[2])


def calib_dict(level: str = "nfhs") -> dict:
    lm = [(-25, 0), (25, 0), (-25, 42), (25, 42), (-25, 21), (25, 21)]
    return {
        "level": level, "court_length_ft": 84.0, "camera_static": True,
        "frame_width": 1920, "frame_height": 1080,
        "homography_points": [{"name": f"p{i}", "px": list(ft_to_px(*p)), "ft": [float(p[0]), float(p[1])]}
                              for i, p in enumerate(lm)],
        "rims": [{"id": "A", "box_px": list(RIM_BOX), "source": "manual"}],
    }


def write_calib(tmp_path, d: dict | None = None):
    path = tmp_path / "court.json"
    path.write_text(json.dumps(d or calib_dict()))
    return load_calibration(path)


# --------------------------------------------------------------------------- scene builder

def ball_flight(t: float, x0: float, y0: float, tx: float, kind: str) -> tuple[float, float]:
    """Ball centre at `t` seconds after release. Make-like parabola reaching the rim centre height at T."""
    T, y_ap, yc = 1.1, 200.0, 392.0
    ta = T / (1 + math.sqrt((yc - y_ap) / (y0 - y_ap)))
    a = (y0 - y_ap) / ta ** 2
    if kind == "rimout" and t > T:
        if t <= T + 0.5:
            u = (t - T) / 0.5
            return tx + 110 * u, yc - 4 * 70 * u * (1 - u)
        s = t - T - 0.5
        return tx + 110 + 100 * s, yc + 300 * s + 505 * s * s
    return x0 + (tx - x0) * t / T, y_ap + a * (t - ta) ** 2


def build_scene(fps: float, shooter_ft=(-21.0, 8.0), kind="make", jump=False, others=(), others_move=False,
                ball_w=BALL_W, flip_ball=False, role_col=False):
    """Tracks DataFrame for one shot. Returns (df, release frame index of the first flight sample)."""
    fx, fy = ft_to_px(*shooter_ft)
    n_hold = round(1.2 * fps)
    tx = 205.0 - (200.0 if kind == "air" else 0.0)
    ball, frames_ball_end = {}, 0
    k = 0
    while True:
        t = (k - n_hold) / fps
        off = 30 * math.sin(math.pi * (t + 0.3) / 0.7) if (jump and -0.3 <= t <= 0.4) else 0.0
        if t < 0:
            ball[k] = (fx, fy - off - 110)
        else:
            x, y = ball_flight(t, fx, fy - 27 * (1 if jump else 0) - 110, tx, kind)
            if kind == "lost" and t > 1.1 - 0.06:
                break
            if y > 560 and t > 1.1:
                break
            if t > 4.0:
                break
            ball[k] = (x, y)
        k += 1
    n_frames = k + round(2.5 * fps)
    rows = []
    for f in range(n_frames):
        t = (f - n_hold) / fps
        off = 30 * math.sin(math.pi * (t + 0.3) / 0.7) if (jump and -0.3 <= t <= 0.4) else 0.0
        boxes = [(7, (fx - 30, fy - 140 - off, fx + 30, fy - off))]
        for oid, (ox, oy) in enumerate(others, start=1):
            mx = 4.0 * (f / fps) if others_move else 0.0
            px, py = ft_to_px(ox + mx, oy)
            boxes.append((oid, (px - 30, py - 140, px + 30, py)))
        for tid, b in boxes:
            r = dict(frame=f, time_s=round(f / fps, 3), **{"class": "player"}, track_id=tid, conf=0.9,
                     x1_px=b[0], y1_px=b[1], x2_px=b[2], y2_px=b[3])
            if role_col:
                r["role_track"] = "player"
            rows.append(r)
        if f in ball:
            bx, by = ball[f]
            if flip_ball:
                by = 2 * 392.0 - by
            r = dict(frame=f, time_s=round(f / fps, 3), **{"class": "ball"}, track_id=-1, conf=0.8,
                     x1_px=bx - ball_w / 2, y1_px=by - ball_w / 2, x2_px=bx + ball_w / 2, y2_px=by + ball_w / 2)
            if role_col:
                r["role_track"] = ""
            rows.append(r)
    return pd.DataFrame(rows)


def run(tmp_path, fps, **kw):
    calib = write_calib(tmp_path, kw.pop("calib", None))
    df = build_scene(fps, **kw)
    return detect_shots(df, calib, fps)


def only(res):
    assert len(res.shots) == 1, f"expected one attempt, got {len(res.shots)}: {res.debug}"
    return res.shots[0]


# --------------------------------------------------------------------------- shot outcomes

@pytest.mark.parametrize("fps", FPS_LIST)
def test_clean_make(tmp_path, fps):
    s = only(run(tmp_path, fps))
    assert s.result == "make" and s.result_conf == 0.9
    assert s.rim_id == "A" and s.shooter_track_id == 7
    assert s.frame > s.release_frame
    assert abs(s.time_s - s.frame / fps) < 1e-9


@pytest.mark.parametrize("fps", FPS_LIST)
def test_rim_out_is_a_miss(tmp_path, fps):
    s = only(run(tmp_path, fps, kind="rimout"))
    assert s.result == "miss" and s.result_conf in (0.9, 0.6)


@pytest.mark.parametrize("fps", FPS_LIST)
def test_air_ball_is_a_miss(tmp_path, fps):
    s = only(run(tmp_path, fps, kind="air"))
    assert s.result == "miss"


@pytest.mark.parametrize("fps", FPS_LIST)
def test_ball_lost_behind_backboard_is_unknown(tmp_path, fps):
    s = only(run(tmp_path, fps, kind="lost"))
    assert s.result == "unknown" and "ball_lost_at_rim" in s.notes
    assert s.result_conf == 0.0
    assert s.points == 3  # the shot value does not depend on the outcome


@pytest.mark.parametrize("fps", FPS_LIST)
def test_vertically_flipped_make_is_not_a_make(tmp_path, fps):
    # the whole pipeline: a flipped ball never starts in a player's hands, so there is no make
    res = run(tmp_path, fps, flip_ball=True)
    assert not any(r.result == "make" for r in res.shots)
    # the make rule itself, with the same path fed straight to judge_outcome
    def outcome(flip: bool) -> str:
        fx, fy = ft_to_px(-21.0, 8.0)
        ts = np.arange(0, 1.6, 1 / fps)
        pts = [ball_flight(t, fx, fy - 110, 205.0, "make") for t in ts]
        pts = [(x, y) for x, y in pts if y <= 560]
        xs = np.array([p[0] for p in pts]); ys = np.array([p[1] for p in pts])
        if flip:
            ys = 2 * 392.0 - ys
        tr = clean_ball_track(np.arange(len(xs)), xs, ys, np.full(len(xs), 0.8), np.full(len(xs), BALL_W),
                              fps, RIM.w_px, CFG)
        z = next(k for k in range(len(tr)) if _in_zone(tr.x_px[k], tr.y_px[k], RIM, CFG))
        return judge_outcome(tr, np.zeros(len(tr), bool), z, RIM, CFG, len(xs)).result
    assert outcome(False) == "make"
    assert outcome(True) != "make"


def _line_traj(fps, times, x, y_of_t, w=BALL_W):
    fr = np.array([round(t * fps) for t in times])
    ys = np.array([y_of_t(t) for t in times])
    return clean_ball_track(fr, np.full(len(fr), float(x)), ys, np.full(len(fr), 0.8), np.full(len(fr), w),
                            fps, RIM.w_px, CFG)


@pytest.mark.parametrize("fps", FPS_LIST)
def test_bridged_crossing_is_a_weaker_make(fps):
    f = lambda t: 300 + 600 * t  # noqa: E731
    times = list(np.arange(0, 0.101, 1 / fps)) + list(np.arange(0.30, 0.5, 1 / fps))
    tr = _line_traj(fps, times, 205.0, f)
    assert tr.seg.max() == 1  # the gap is longer than max_gap_s, so two segments
    out = judge_outcome(tr, np.zeros(len(tr), bool), 0, RIM, CFG, int(tr.frame[-1]))
    assert out.result == "make" and out.conf == 0.6 and "bridged_crossing" in out.notes
    full = _line_traj(fps, list(np.arange(0, 0.5, 1 / fps)), 205.0, f)
    out = judge_outcome(full, np.zeros(len(full), bool), 0, RIM, CFG, int(full.frame[-1]))
    assert out.result == "make" and out.conf == 0.9 and not out.notes


@pytest.mark.parametrize("fps", FPS_LIST)
def test_depth_check_turns_a_wrong_sized_make_into_unknown(fps):
    f = lambda t: 300 + 600 * t  # noqa: E731
    tr = _line_traj(fps, list(np.arange(0, 0.5, 1 / fps)), 205.0, f, w=100.0)  # 1.43 W: not at rim depth
    out = judge_outcome(tr, np.zeros(len(tr), bool), 0, RIM, CFG, int(tr.frame[-1]))
    assert out.result == "unknown" and "depth_check_failed" in out.notes


def test_small_rim_is_unknown():
    small = Rim("A", (200.0, 380.0, 212.0, 384.0), "manual")  # W = 12 px
    fps = 23.951
    tr = _line_traj(fps, list(np.arange(0, 0.5, 1 / fps)), 206.0, lambda t: 300 + 600 * t, w=6.0)
    out = judge_outcome(tr, np.zeros(len(tr), bool), 0, small, CFG, int(tr.frame[-1]))
    assert out.result == "unknown" and out.notes == ["rim_small"]


# --------------------------------------------------------------------------- shot value

@pytest.mark.parametrize("fps", FPS_LIST)
def test_corner_three(tmp_path, fps):
    s = only(run(tmp_path, fps, shooter_ft=(-21.0, 8.0)))
    assert s.points == 3 and s.points_conf == 0.9
    assert s.margin_ft == pytest.approx(math.hypot(21.0, 8.0 - 5.25) - 19.75, abs=0.02)  # nearest part is the arc
    assert s.feet_x_ft == pytest.approx(-21.0, abs=0.01) and s.feet_y_ft == pytest.approx(8.0, abs=0.01)


@pytest.mark.parametrize("fps", FPS_LIST)
def test_long_two_just_inside_the_arc(tmp_path, fps):
    s = only(run(tmp_path, fps, shooter_ft=(0.0, 5.25 + 19.2)))
    assert s.points == 2 and "near_arc" in s.notes and s.points_conf == 0.3


@pytest.mark.parametrize("fps", FPS_LIST)
def test_feet_not_box_centre_decide_two_or_three(tmp_path, fps):
    calib = write_calib(tmp_path)
    fx, fy = ft_to_px(-21.0, 8.0)
    centre_ft = px_to_ft(calib.homography.H_px_to_ft, np.array([[fx, fy - 70.0]]))[0]
    assert not is_three(tuple(centre_ft), "nfhs")      # box centre would say two
    assert is_three((-21.0, 8.0), "nfhs")              # feet say three
    s = only(detect_shots(build_scene(fps, shooter_ft=(-21.0, 8.0)), calib, fps))
    assert s.points == 3


@pytest.mark.parametrize("fps", FPS_LIST)
def test_feet_taken_at_takeoff_not_at_release(tmp_path, fps):
    # airborne at release the feet are 27+ px higher in the image: that point is a two
    s = only(run(tmp_path, fps, shooter_ft=(-21.0, 8.0), jump=True))
    assert s.feet_x_ft == pytest.approx(-21.0, abs=0.05)
    assert s.points == 3


FT_OTHERS = [(-8.0, 10.0), (8.0, 10.0), (-8.0, 14.0)]


@pytest.mark.parametrize("fps", FPS_LIST)
def test_free_throw(tmp_path, fps):
    s = only(run(tmp_path, fps, shooter_ft=(0.0, 19.5), others=FT_OTHERS))
    assert s.points == 1 and "ft_rule" in s.notes and s.points_conf <= 0.6


@pytest.mark.parametrize("fps", FPS_LIST)
def test_free_throw_position_alone_is_a_two(tmp_path, fps):
    s = only(run(tmp_path, fps, shooter_ft=(0.0, 19.5), others=FT_OTHERS, others_move=True))
    assert s.points == 2 and "ft_rule" not in s.notes
    s = only(run(tmp_path, fps, shooter_ft=(0.0, 19.5)))  # nobody else on screen: not enough evidence
    assert s.points == 2


def test_no_homography_leaves_points_empty(tmp_path):
    d = calib_dict()
    del d["homography_points"]
    s = only(run(tmp_path, 23.951, calib=d))
    assert s.points is None and "no_homography" in s.notes and s.feet_x_ft is None


def test_feet_cut_off_leaves_points_empty(tmp_path):
    d = calib_dict()
    d["frame_height"] = 800  # the shooter's feet are at y of about 752: use a frame that ends there
    fx, fy = ft_to_px(-21.0, 8.0)
    d["frame_height"] = int(fy + 1)
    s = only(run(tmp_path, 23.951, calib=d))
    assert s.points is None and "feet_cut_off" in s.notes


def test_role_track_column_is_used_and_optional(tmp_path):
    calib = write_calib(tmp_path)
    fps = 23.951
    df = build_scene(fps, role_col=True)
    assert len(detect_shots(df, calib, fps).shots) == 1
    ref = df.copy()
    ref.loc[ref["class"] == "player", "role_track"] = "referee"  # same boxes, but not players
    assert detect_shots(ref, calib, fps).shots == []


# --------------------------------------------------------------------------- court geometry

def test_homography_maps_pixels_to_feet(tmp_path):
    calib = write_calib(tmp_path)
    corner_px = np.array([ft_to_px(-25.0, 0.0)])
    out = px_to_ft(calib.homography.H_px_to_ft, corner_px)[0]
    assert out == pytest.approx([-25.0, 0.0], abs=1e-6)
    assert px_to_ft(calib.homography.H_px_to_ft, np.array([ft_to_px(25.0, 42.0)]))[0] == pytest.approx([25.0, 42.0], abs=1e-4)
    assert calib.homography.rms_ft < 1e-6
    # the pixel->feet direction: feeding feet as pixels must NOT give the corner back
    assert not np.allclose(px_to_ft(calib.homography.H_px_to_ft, np.array([[-25.0, 0.0]]))[0], [-25.0, 0.0], atol=1.0)


def test_bad_homography_raises_config_error():
    px = np.array([[0, 0], [100, 0], [100, 100], [0, 100], [50, 50]], float)
    ft = np.array([[0, 0], [10, 0], [10, 10], [0, 10], [5, 5]], float)
    assert fit_homography(px, ft).rms_ft < 1e-6
    bad = ft.copy()
    bad[4] = [5.0, 0.0]  # one landmark clicked wrongly: the least-squares fit cannot satisfy it
    with pytest.raises(ConfigError):
        fit_homography(px, bad)
    with pytest.raises(ConfigError):
        fit_homography(px[:3], ft[:3])  # fewer than 4 pairs


def test_rim_b_frame():
    out = to_rim_frame(np.array([[10.0, 20.0]]), "B", 84.0)[0]
    assert out == pytest.approx([-10.0, 64.0])
    assert to_rim_frame(np.array([[10.0, 20.0]]), "A", None)[0] == pytest.approx([10.0, 20.0])
    with pytest.raises(ConfigError):
        to_rim_frame(np.array([[10.0, 20.0]]), "B", None)


def test_old_polygon_file_loads_with_no_rims_or_homography(tmp_path):
    path = tmp_path / "old.json"
    save_court(path, "clip", 10, 1280, 720, [[0, 0], [10, 0], [10, 10]])
    c = load_calibration(path)
    assert c.rims == () and c.homography is None and c.level == "nfhs"
    assert load_court(path).video_stem == "clip"  # polygon.py still reads the same file


@pytest.mark.parametrize("xy,expected,margin_lo,margin_hi", [
    ((20.5, 3.0), True, 0.5, 0.8),                 # corner three, 0.75 ft outside the line
    ((0.0, 5.25 + 19.0), False, 0.7, 0.8),         # two, 0.75 ft inside
    ((0.0, 5.25 + 20.9), True, 1.1, 1.2),          # three, clear of the line
    ((0.0, 5.25 + 20.0), True, 0.2, 0.3),          # three, inside the uncertainty band
    ((19.75, 3.0), False, 0.0, 1e-9),              # on the line counts as a two
])
def test_nfhs_two_or_three(xy, expected, margin_lo, margin_hi):
    assert is_three(xy, "nfhs") is expected
    assert margin_lo <= three_margin_ft(xy, "nfhs") <= margin_hi


def test_three_point_radius_override_and_levels():
    assert is_three((0.0, 5.25 + 20.5), "nfhs") is True
    assert is_three((0.0, 5.25 + 20.5), "nfhs", three_radius_ft=22.146) is False
    assert is_three((21.0, 3.0), "nba") is False and is_three((22.5, 3.0), "nba") is True  # c = 22 ft
    assert is_three((0.0, 5.167 + 22.0), "fiba") is False and is_three((0.0, 5.167 + 22.3), "fiba") is True


def test_spec_table_carries_sources_and_honest_verified_flags():
    for lvl in ("nfhs", "ncaa", "nba", "fiba"):
        for name, v in get_spec(lvl).all_values().items():
            assert v.source and len(v.source) > 20, (lvl, name)
            assert v.status in ("V", "S", "X", "D", "U", "PROVISIONAL")
            if v.status in ("X", "U", "PROVISIONAL"):
                assert not v.verified, (lvl, name)  # never present a weak value as verified
            if v.status == "U":
                assert v.value is None, (lvl, name)  # unavailable, not guessed
    nfhs, ncaa, nba, fiba = (get_spec(x) for x in ("nfhs", "ncaa", "nba", "fiba"))
    assert nfhs.basket_offset_ft.value == 5.25 and nfhs.basket_offset_ft.verified
    for s in (ncaa, nba):  # the US basket offset is only an assumption outside NFHS
        assert s.basket_offset_ft.value == 5.25 and not s.basket_offset_ft.verified
        assert s.ft_line_ft.value == 19.0 and not s.ft_line_ft.verified
    assert ncaa.court_length_ft.value is None and ncaa.court_width_ft.value is None
    assert fiba.three_radius_ft.value == pytest.approx(6.75 / 0.3048)  # metres converted once
    assert fiba.basket_offset_ft.value == pytest.approx(1.575 / 0.3048)
    assert fiba.ft_line_ft.value == pytest.approx(5.80 / 0.3048)


# --------------------------------------------------------------------------- trajectory cleaning

def _steady(fps, n=40, step_px=20.0):
    fr = np.arange(n)
    return fr, 100.0 + step_px * fr, np.full(n, 400.0) - 5 * fr


def test_outliers_are_rejected_by_continuity_not_confidence():
    fps = 23.951
    fr, x, y = _steady(fps)
    conf = np.full(len(fr), 0.45)
    x, y = x.copy(), y.copy()
    for k in (20, 22):  # false ball 336 px away, same confidence as the real ones
        x[k], y[k] = x[k] - 330.0, y[k] + 60.0
    tr = clean_ball_track(fr, x, y, conf, np.full(len(fr), 30.0), fps, 71.0, CFG)
    assert np.all(np.abs(tr.y_px - (400.0 - 5 * tr.frame)) < 1e-6)  # every kept point is on the true path
    assert 19 in tr.frame and 21 in tr.frame and 23 in tr.frame
    # the 2 missing frames are filled (<= 3 frames at 23.951 fps) and flagged
    assert tr.interp[list(tr.frame).index(20)] and tr.interp[list(tr.frame).index(22)]
    assert np.allclose(tr.x_px[list(tr.frame).index(20)], 100 + 20 * 20)


def test_gap_fill_depends_on_seconds_not_frames():
    def gap_frames_kept(fps, gap):
        fr = np.array([k for k in range(40) if not (15 <= k < 15 + gap)])
        step = 20.0 * 23.951 / fps
        tr = clean_ball_track(fr, 100 + step * fr, np.full(len(fr), 300.0), np.full(len(fr), 0.8),
                              np.full(len(fr), 30.0), fps, 71.0, CFG)
        return tr
    assert gap_frames_kept(23.951, 3).seg.max() == 0 and gap_frames_kept(23.951, 3).interp.sum() == 3
    assert gap_frames_kept(23.951, 4).seg.max() == 1 and gap_frames_kept(23.951, 4).interp.sum() == 0
    assert gap_frames_kept(60.0, 7).seg.max() == 0  # 7 frames = 0.117 s fits at 60 fps
    assert gap_frames_kept(60.0, 9).seg.max() == 1  # 9 frames = 0.15 s does not


def test_ghost_detection_that_repeats_is_dropped():
    fps = 23.951
    fr, x, y = _steady(fps, n=40)
    ghost = (480.0, 100.0)  # a still false ball; the true path is far from it in y
    frames = list(fr) + [f for f in range(40)]
    xs = list(x) + [ghost[0]] * 40
    ys = list(y) + [ghost[1]] * 40
    conf = [0.8] * 40 + [0.45] * 40
    tr = clean_ball_track(np.array(frames), np.array(xs), np.array(ys), np.array(conf), np.full(80, 30.0),
                          fps, 71.0, CFG)
    assert len(tr) == 40 and np.all(tr.y_px != 100.0) and tr.seg.max() == 0


def test_velocity_sign_y_down():
    fps = 23.951
    fr = np.arange(30)
    y = 500.0 - 20.0 * fr  # rising up the image: y decreases
    tr = clean_ball_track(fr, np.full(30, 100.0), y, np.full(30, 0.8), np.full(30, 30.0), fps, 71.0, CFG)
    assert np.all(tr.vy < 0) and tr.vy[10] == pytest.approx(-20.0 * fps, rel=0.01)


def test_empty_input():
    tr = clean_ball_track(np.array([], int), np.array([]), np.array([]), np.array([]), np.array([]), 23.951, 70.0)
    assert len(tr) == 0


# --------------------------------------------------------------------------- output

def test_csv_columns_and_empty_cells(tmp_path):
    res = run(tmp_path, 23.951, kind="lost")
    out = tmp_path / "shots.csv"
    write_shots_csv(res, out)
    df = pd.read_csv(out)
    assert list(df.columns) == COLUMNS
    assert df.loc[0, "result"] == "unknown" and df.loc[0, "notes"] == "ball_lost_at_rim"
    empty = tmp_path / "e.csv"
    write_shots_csv(type(res)([], {}), empty)
    assert list(pd.read_csv(empty).columns) == COLUMNS


def test_lockout_suppresses_a_second_attempt_right_after_the_first(tmp_path):
    fps = 23.951
    calib = write_calib(tmp_path)
    one = build_scene(fps)
    # a second identical scene starting about 0.4 s after the first outcome: inside the 0.75 s lockout
    first = detect_shots(one, calib, fps).shots[0]
    shift = first.frame + round(0.4 * fps) - round(1.2 * fps) + 0  # next release soon after the outcome
    two = one.copy()
    two["track_id"] = two["track_id"].where(two["track_id"] < 0, two["track_id"] + 100)  # distinct players
    two["frame"] = two["frame"] + shift
    two["time_s"] = (two["frame"] / fps).round(3)
    both = pd.concat([one, two]).sort_values(["frame", "class"]).reset_index(drop=True)
    res = detect_shots(both, calib, fps)
    assert len(res.shots) == 1 and res.debug["n_lockout_suppressed"] == 1


@pytest.mark.parametrize("fps", FPS_LIST)
def test_a_ball_released_downward_is_not_a_shot(tmp_path, fps):
    # a player holds the ball above rim height, then it drops through the rim: no rise, so no attempt
    calib = write_calib(tmp_path)
    rows = []
    n_hold = round(0.5 * fps)
    for f in range(n_hold + round(1.5 * fps)):
        rows.append(dict(frame=f, time_s=round(f / fps, 3), **{"class": "player"}, track_id=3, conf=0.9,
                         x1_px=100.0, y1_px=270.0, x2_px=160.0, y2_px=500.0))
        t = (f - n_hold) / fps
        x, y = (165.0, 300.0) if t < 0 else (165.0 + 200.0 * t, 300.0 + 700.0 * t * t)
        if y < 560:
            rows.append(dict(frame=f, time_s=round(f / fps, 3), **{"class": "ball"}, track_id=-1, conf=0.8,
                             x1_px=x - 19, y1_px=y - 19, x2_px=x + 19, y2_px=y + 19))
    res = detect_shots(pd.DataFrame(rows), calib, fps)
    assert res.debug["n_flights"] >= 1 and res.shots == []


# --------------------------------------------------------------------------- step 5 review fixes

def _manual_traj(fps, frames, xs, ys, segs, interp=None):
    n = len(frames)
    return Trajectory(fps, np.array(frames), np.array(xs, float), np.array(ys, float), np.full(n, 30.0),
                      np.full(n, 0.8), np.array(interp if interp is not None else [False] * n),
                      np.array(segs), np.zeros(n), np.zeros(n))


def test_gap_length_is_in_seconds_so_a_short_break_is_a_loss_not_a_miss():
    # point 3 is at the rim; after a 5-frame gap the ball shows up below and outside it.
    # At 23.951 fps the gap (0.21 s) exceeds max_gap_s, so the ball was lost: unknown.
    # A gap limit pinned to a 60 fps frame count would treat it as continuous and publish a miss.
    fps = 23.951
    tr = _manual_traj(fps, [0, 1, 2, 3, 9, 10, 11], [205, 205, 205, 205, 330, 330, 330],
                      [350, 365, 380, 392, 450, 455, 460], [0, 0, 0, 0, 1, 1, 1])
    out = judge_outcome(tr, np.zeros(len(tr), bool), 0, RIM, CFG, int(tr.frame[-1]))
    assert out.result == "unknown" and out.notes == ["ball_lost_at_rim"]


def test_frame_adjacent_chain_break_is_a_loss():
    # a re-anchor can start a new segment on the very next frame: zero frame gap, but still a break
    fps = 23.951
    frames = [0, 1, 2, 3, 4, 5, 6]
    xs = [205, 205, 205, 205, 330, 330, 330]
    ys = [350, 365, 380, 392, 450, 455, 460]
    brk = judge_outcome(_manual_traj(fps, frames, xs, ys, [0, 0, 0, 0, 1, 1, 1]),
                        np.zeros(7, bool), 0, RIM, CFG, 6)
    assert brk.result == "unknown" and brk.notes == ["ball_lost_at_rim"]
    one = judge_outcome(_manual_traj(fps, frames, xs, ys, [0] * 7), np.zeros(7, bool), 0, RIM, CFG, 6)
    assert one.result == "miss"  # the same points as ONE chain are real evidence of a miss


def test_miss_after_loss_needs_observed_points_and_is_capped():
    fps = 23.951
    fr = [0, 1, 2, 3, 12]
    xs, ys = [205, 205, 205, 330, 205], [350, 380, 392, 450, 470]
    seg = [0, 0, 0, 0, 1]
    seen = judge_outcome(_manual_traj(fps, fr, xs, ys, seg), np.zeros(5, bool), 0, RIM, CFG, 12)
    assert seen.result == "miss" and seen.conf == 0.6
    guessed = judge_outcome(_manual_traj(fps, fr, xs, ys, seg, [False, False, False, True, False]),
                            np.zeros(5, bool), 0, RIM, CFG, 12)
    assert guessed.result == "unknown"


def _two_rim_calib(tmp_path, rim_ids, with_h=True):
    d = calib_dict()
    if not with_h:
        d.pop("homography_points")
    d["rims"] = [{"id": rim_ids[0], "box_px": list(RIM_BOX), "source": "manual"}]
    return write_calib(tmp_path, d)


def test_mislabelled_rim_is_refused(tmp_path):
    df = build_scene(23.951)
    detect_shots(df, _two_rim_calib(tmp_path, "A"), 23.951)  # RIM_BOX is at the A basket: fine
    with pytest.raises(ConfigError, match="nearer the other basket"):
        detect_shots(df, _two_rim_calib(tmp_path, "B"), 23.951)


def test_median_rim_fallback_is_refused_with_a_homography(tmp_path):
    df = build_scene(23.951)
    rims = pd.DataFrame([dict(frame=f, time_s=f / 23.951, **{"class": "rim"}, track_id=-1, conf=0.9,
                              x1_px=RIM_BOX[0], y1_px=RIM_BOX[1], x2_px=RIM_BOX[2], y2_px=RIM_BOX[3])
                         for f in range(0, 60)])
    d = calib_dict()
    d.pop("rims")
    with pytest.raises(ConfigError, match="median rim fallback"):
        detect_shots(pd.concat([df, rims]), write_calib(tmp_path, d), 23.951)
    d.pop("homography_points")  # without a homography nothing can be mirrored: the fallback is allowed
    assert detect_shots(pd.concat([df, rims]), write_calib(tmp_path, d), 23.951).debug["rims"]


def test_provisional_court_constants_cap_points_confidence(tmp_path):
    nfhs = only(run(tmp_path, 23.951, shooter_ft=(-21.0, 30.0)))
    assert "provisional_geometry" not in nfhs.notes and nfhs.points_conf == 0.9
    nba = only(run(tmp_path, 23.951, shooter_ft=(-21.0, 30.0), calib=calib_dict("nba")))
    assert "provisional_geometry" in nba.notes and nba.points_conf <= 0.6


def test_rim_b_without_court_length_says_no_court_length(tmp_path):
    d = calib_dict("ncaa")
    d.pop("court_length_ft")
    d["rims"][0]["id"] = "B"
    d["rims"][0]["box_px"] = list(RIM_BOX)
    # NCAA has no default length, so the label check is skipped and the frame change is impossible
    s = only(run(tmp_path, 23.951, calib=d))
    assert "no_court_length" in s.notes and s.points is None

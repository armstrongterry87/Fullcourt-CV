"""Shot attempts, make/miss, shooter attribution and shot value from a tracks CSV (design 004).

Input: the tracks CSV written by `scripts/baseline_track.py` (`frame,time_s,class,track_id,conf,
x1_px,y1_px,x2_px,y2_px`, plus `role_track` when task 002 ran) and a calibration file.
Output: `shots.csv`, one row per attempt (columns in `COLUMNS`, design section 8.2).

Spaces: pixels are (x, y) from the top-left and y grows downward, so "above the rim" means
a SMALLER y. Court space is feet in the frame of the rim being attacked. The ball position is
the ball box centre; the shooter position is the box bottom-centre `((x1+x2)/2, y2)`.

Every rule fails to `unknown` (or an empty field) when its evidence is missing. Nothing in
this module writes an image or a frame to disk.
"""

from __future__ import annotations

import csv
import math
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field, replace
from pathlib import Path

import numpy as np
import pandas as pd

from src.camera.motion import MotionTrack, transform_tracks
from src.court.geometry import (
    Calibration, Rim, check_motion_id, feet_off_court, is_three, px_to_ft, three_margin_ft, to_rim_frame,
)
from src.court.polygon import ConfigError
from src.shots.config import ShotConfig
from src.shots.trajectory import Trajectory, clean_ball_track

__all__ = ["COLUMNS", "ShotConfig", "ShotRow", "DetectionResult", "Outcome", "detect_shots",
           "judge_outcome", "estimate_rims", "infer_fps", "read_tracks", "write_shots_csv"]

MOTION_COLUMNS = ["view_id", "camera_quality"]  # appended only when stabilised scoring was used (design 005 sec. 9.3)
_DRIFT_WINDOW_S = 10.0   # CHOSEN: window for the stabilised rim-drift check (design 005 sec. 9.3)
_DRIFT_MIN_DETS = 10     # CHOSEN: fewest rim detections for a window to be judged
_DRIFT_NEAR_W = 2.0      # rim detections farther than this many W from the calibrated rim are other objects

COLUMNS = [
    "frame", "time_s", "result", "points", "notes",
    "shot_id", "release_frame", "release_time_s", "shooter_track_id",
    "ball_release_x_px", "ball_release_y_px", "feet_x_px", "feet_y_px", "feet_x_ft", "feet_y_ft",
    "rim_id", "level", "margin_ft",
    "attempt_conf", "release_conf", "result_conf", "shooter_conf", "points_conf",
]


@dataclass
class ShotRow:
    """One attempt. Empty (None) fields mean unknown."""

    frame: int
    time_s: float
    result: str
    points: int | None
    notes: list[str]
    release_frame: int
    release_time_s: float
    shooter_track_id: int | None
    ball_release_x_px: float
    ball_release_y_px: float
    feet_x_px: float | None
    feet_y_px: float | None
    feet_x_ft: float | None
    feet_y_ft: float | None
    rim_id: str
    level: str
    margin_ft: float | None
    attempt_conf: float
    release_conf: float
    result_conf: float
    shooter_conf: float
    points_conf: float
    shot_id: int = -1
    view_id: int | None = None           # stabilised scoring only
    camera_quality: float | None = None  # lowest motion quality between release and outcome


@dataclass
class DetectionResult:
    """Shot rows plus debug totals (e.g. `n_flights_lost`)."""

    shots: list[ShotRow]
    debug: dict = field(default_factory=dict)


@dataclass
class Outcome:
    """Result of the make/miss decision for one attempt."""

    result: str                 # make, miss or unknown
    notes: list[str]
    conf: float
    frame: int                  # outcome frame, 0-based


# --------------------------------------------------------------------------- reading

def read_tracks(path: Path) -> pd.DataFrame:
    """Read a tracks CSV. The task-002 columns (`label_frame`, `role_track`) are optional."""
    df = pd.read_csv(path)
    need = {"frame", "time_s", "class", "track_id", "conf", "x1_px", "y1_px", "x2_px", "y2_px"}
    missing = need - set(df.columns)
    if missing:
        raise ConfigError(f"{path}: tracks CSV is missing columns {sorted(missing)}.")
    return df


def infer_fps(df: pd.DataFrame) -> float:
    """fps from `frame / time_s` at the last frame (time_s is rounded to 3 decimals)."""
    d = df[df["time_s"] > 0]
    if d.empty:
        raise ConfigError("Cannot infer fps from the tracks CSV (no row with time_s > 0); pass fps.")
    last = d.loc[d["frame"].idxmax()]
    return float(last["frame"] / last["time_s"])


# --------------------------------------------------------------------------- rims

def estimate_rims(rim_dets: pd.DataFrame, frame_width_px: float, cfg: ShotConfig) -> list[Rim]:
    """Median rim box per side of the frame (design 4.1 fallback). Left is 'A', right is 'B'.

    A side is accepted only with enough detections and a tight centre (MAD <= 25% of width).
    """
    out: list[Rim] = []
    if rim_dets.empty:
        return out
    cx = (rim_dets["x1_px"] + rim_dets["x2_px"]) / 2
    for rid, sel in (("A", cx < frame_width_px / 2), ("B", cx >= frame_width_px / 2)):
        d = rim_dets[sel]
        if len(d) < cfg.rim_median_min_detections:
            continue
        box = d[["x1_px", "y1_px", "x2_px", "y2_px"]].median().to_numpy()
        w = box[2] - box[0]
        c = (d["x1_px"] + d["x2_px"]) / 2
        mad = float(np.median(np.abs(c - c.median())))
        if w <= 0 or mad > cfg.rim_median_max_mad_frac * w:
            continue
        out.append(Rim(rid, tuple(float(v) for v in box), "median"))
    return out


def _check_rim_labels(rims: list[Rim], calib: Calibration) -> None:
    """Raise ConfigError if a rim's label disagrees with the homography (design 4.1).

    Each rim box centre is mapped to feet. Rim A must be nearer the A basket (0, b) than the B basket
    (0, L - b), and rim B the reverse. The rim centre is above the floor, so the mapped point is only
    approximate; the test is "nearer to its own basket", which survives that error. Skipped (no check
    possible) without a homography or without a court length.
    """
    if calib.homography is None:
        return
    length, b = calib.effective_length_ft(), calib.spec().basket_offset_ft.value
    if length is None or b is None:
        return
    a_ft, b_ft = np.array([0.0, b]), np.array([0.0, length - b])
    for r in rims:
        p_ft = px_to_ft(calib.homography.H_px_to_ft, np.array([[r.cx_px, r.cy_px]]))[0]
        own, other = (a_ft, b_ft) if r.id == "A" else (b_ft, a_ft)
        if np.linalg.norm(p_ft - own) > np.linalg.norm(p_ft - other):
            raise ConfigError(
                f"Rim {r.id} maps to ({p_ft[0]:.1f}, {p_ft[1]:.1f}) ft, nearer the other basket; the homography "
                f"origin is at rim A. Swap the rim labels or re-click the landmarks.")


def _drifted(rim: Rim, rim_dets: pd.DataFrame, frame_width_px: float, cfg: ShotConfig) -> bool:
    """Drift check: the median detected centre on this rim's side is > 0.5 W from the calibrated one."""
    cx = (rim_dets["x1_px"] + rim_dets["x2_px"]) / 2
    left = rim.cx_px < frame_width_px / 2
    d = rim_dets[(cx < frame_width_px / 2) == left]
    if len(d) < cfg.rim_median_min_detections:
        return False
    mx = float(((d["x1_px"] + d["x2_px"]) / 2).median())
    my = float(((d["y1_px"] + d["y2_px"]) / 2).median())
    return math.hypot(mx - rim.cx_px, my - rim.cy_px) > cfg.rim_drift_w * rim.w_px


def _drifted_windows(rim: Rim, rim_dets: pd.DataFrame, fps: float, cfg: ShotConfig) -> set[int]:
    """Windows (10 s each) whose median detected rim centre is > `rim_drift_w` W from the calibrated one."""
    if rim_dets.empty:
        return set()
    cx = ((rim_dets["x1_px"] + rim_dets["x2_px"]) / 2).to_numpy()
    cy = ((rim_dets["y1_px"] + rim_dets["y2_px"]) / 2).to_numpy()
    d = np.hypot(cx - rim.cx_px, cy - rim.cy_px)
    near = d <= _DRIFT_NEAR_W * rim.w_px
    win = (rim_dets["frame"].to_numpy() / fps // _DRIFT_WINDOW_S).astype(int)
    out = set()
    for w in np.unique(win[near]):
        sel = near & (win == w)
        if sel.sum() >= _DRIFT_MIN_DETS and float(np.median(d[sel])) > cfg.rim_drift_w * rim.w_px:
            out.add(int(w))
    return out


# --------------------------------------------------------------------------- helpers

def _rect_dist(px: float, py: float, box: tuple) -> float:
    """Distance from a point to a rectangle (zero inside)."""
    dx = max(box[0] - px, 0.0, px - box[2])
    dy = max(box[1] - py, 0.0, py - box[3])
    return math.hypot(dx, dy)


def _in_zone(x: float, y: float, rim: Rim, cfg: ShotConfig) -> bool:
    W = rim.w_px
    return (abs(x - rim.cx_px) <= cfg.zone_x_w * W
            and rim.top_px - cfg.zone_above_w * W <= y <= rim.bottom_px + cfg.zone_below_w * W)


def _in_crit(x: float, y: float, rim: Rim, cfg: ShotConfig) -> bool:
    W = rim.w_px
    return (abs(x - rim.cx_px) <= cfg.crit_x_w * W
            and rim.top_px - cfg.crit_y_w * W <= y <= rim.bottom_px + cfg.crit_y_w * W)


def _held(x: float, y: float, boxes: list[tuple], cfg: ShotConfig) -> bool:
    """Ball centre inside a player's box padded 25% of width each side and 25% of height on top."""
    for x1, y1, x2, y2 in boxes:
        px, py = cfg.held_pad_x_frac * (x2 - x1), cfg.held_pad_top_frac * (y2 - y1)
        if x1 - px <= x <= x2 + px and y1 - py <= y <= y2:
            return True
    return False


# --------------------------------------------------------------------------- make / miss

def judge_outcome(tr: Trajectory, held: np.ndarray, zone_start: int, rim: Rim, cfg: ShotConfig,
                  clip_last_frame: int) -> Outcome:
    """Decide make / miss / unknown for the flight whose first zone frame is trajectory index `zone_start`.

    Follows design 4.4 in order: rim checks, make, depth check, ball lost at the rim, miss, otherwise
    unknown. Image y grows downward, so "above the rim" is y < rim top.
    """
    fps, W = tr.fps, rim.w_px
    gap_n = max(1, round(cfg.max_gap_s * fps))
    win_n = cfg.outcome_window_s * fps
    n = len(tr)
    x_l, x_r = rim.box_px[0], rim.box_px[2]
    top, bot, yc = rim.top_px, rim.bottom_px, rim.cy_px

    # outcome window: first zone frame until held, left the zone, or 1.5 s
    idx = [zone_start]
    ended, exit_k = "data_end", None
    for k in range(zone_start + 1, n):
        if held[k]:
            ended = "held"
        elif tr.frame[k] - tr.frame[zone_start] > win_n:
            ended = "timeout"
        elif not _in_zone(tr.x_px[k], tr.y_px[k], rim, cfg):
            ended, exit_k = "left_zone", k
        else:
            idx.append(k)
            continue
        break
    pts = idx + ([exit_k] if exit_k is not None else [])  # the first point past the zone is evidence too
    X, Y, F = tr.x_px[pts], tr.y_px[pts], tr.frame[pts]
    real = ~tr.interp[pts]

    def closest_frame() -> int:
        d = np.hypot(X - rim.cx_px, Y - rim.cy_px)
        return int(F[int(np.argmin(d))])

    def unknown(note: str) -> Outcome:
        return Outcome("unknown", [note], 0.0, closest_frame())

    if rim.w_px < cfg.rim_small_px:
        return unknown("rim_small")

    # --- make
    within_span = lambda xx: x_l - cfg.below_x_margin_w * W <= xx <= x_r + cfg.below_x_margin_w * W  # noqa: E731
    for a in range(len(pts)):
        if not real[a] or Y[a] >= top:
            continue
        for b in range(a + 1, len(pts)):
            if (F[b] - F[a]) / fps > cfg.make_max_dt_s + 1e-9:
                break
            if not real[b] or Y[b] <= bot or not within_span(X[b]):
                continue
            ys = Y[a:b + 1]
            if np.any(ys[1:] < ys[:-1] - cfg.make_max_rise_w * W):
                continue
            cross = next((k for k in range(a, b) if Y[k] < yc <= Y[k + 1]), None)
            if cross is None:
                continue
            t = (yc - Y[cross]) / (Y[cross + 1] - Y[cross])
            xc = X[cross] + t * (X[cross + 1] - X[cross])
            if not (x_l + cfg.make_x_margin_w * W <= xc <= x_r - cfg.make_x_margin_w * W):
                continue
            ball_w = tr.w_px[pts[a]] / W
            if not (cfg.depth_ratio_min <= ball_w <= cfg.depth_ratio_max):
                return unknown("depth_check_failed")
            bridged = any(F[k + 1] - F[k] - 1 > gap_n or tr.seg[pts[k + 1]] != tr.seg[pts[k]] for k in range(a, b))
            first_below = next(k for k in range(a, b + 1) if Y[k] >= yc)
            return Outcome("make", ["bridged_crossing"] if bridged else [], 0.6 if bridged else 0.9,
                           int(F[first_below]))

    # --- ball lost at the rim
    # a chain break (new segment) counts as a loss even when the frames are adjacent (re-anchor)
    loss = next((j for j in range(len(pts) - 1)
                 if (F[j + 1] - F[j] - 1 > gap_n or tr.seg[pts[j + 1]] != tr.seg[pts[j]])
                 and _in_crit(X[j], Y[j], rim, cfg)), None)
    loss_note = "ball_lost_at_rim"
    if loss is None and ended == "data_end":
        loss = len(pts) - 1
        loss_note = "clip_end" if clip_last_frame - F[-1] <= gap_n else "ball_lost_at_rim"

    def below_outside(upto: int, observed_only: bool = False) -> bool:
        return any(Y[k] > bot and not within_span(X[k]) and (real[k] or not observed_only)
                   for k in range(upto + 1))

    if loss is not None:
        # a ball OBSERVED below the rim and outside it before it was lost cannot go in afterwards.
        # The ball was lost, so not all evidence is observed: confidence is capped at 0.6.
        if below_outside(loss, observed_only=True):
            return Outcome("miss", [], 0.6, closest_frame())
        return unknown(loss_note)

    # --- miss
    c_idx = [k for k in range(len(pts)) if _in_crit(X[k], Y[k], rim, cfg)]
    bounce = False
    if c_idx:
        ymax = -math.inf
        for k in range(c_idx[0], len(pts)):
            ymax = max(ymax, Y[k])
            if Y[k] < ymax - cfg.bounce_rise_w * W:
                bounce = True
                break
    sideways = (ended == "left_zone" and abs(X[-1] - rim.cx_px) > cfg.zone_x_w * W
                and not any(Y[k] > top and x_l <= X[k] <= x_r for k in range(len(pts))))
    if below_outside(len(pts) - 1) or bounce or sideways:
        interp_in_c = any(tr.interp[pts[k]] for k in c_idx)
        return Outcome("miss", [], 0.6 if interp_in_c else 0.9, closest_frame())
    return unknown("no_evidence")


# --------------------------------------------------------------------------- main entry

def _detect_core(tracks: pd.DataFrame, calib: Calibration, fps: float, cfg: ShotConfig | None = None,
                 frame_height_px: float | None = None, feet_cut: set[tuple[int, int]] | None = None,
                 stabilised: bool = False) -> DetectionResult:
    """The static-camera engine. With `stabilised` the tracks are already in view pixels."""
    cfg = cfg or ShotConfig()
    debug: dict = {"fps": fps, "n_flights": 0, "n_flights_lost": 0, "n_lockout_suppressed": 0,
                   "n_attempts": 0, "warnings": []}
    frame_height_px = frame_height_px or calib.frame_height
    clip_last = int(tracks["frame"].max()) if len(tracks) else 0
    cls = tracks["class"]
    frame_w = float(calib.frame_width or (tracks["x2_px"].max() if len(tracks) else 0.0))

    # rims: calibration first, else the median fallback; drift check on calibrated rims
    rim_dets = tracks[cls == "rim"]
    rims = list(calib.rims)
    drift: set[str] = set()
    drift_win: dict[str, set[int]] = {}
    if stabilised:
        # design 005 9.3: rims must be manual; drift is judged per 10 s window on stabilised detections
        if not rims:
            raise ConfigError("Stabilised scoring needs manual rims in the calibration file.")
        _check_rim_labels(rims, calib)
        for r in rims:
            drift_win[r.id] = _drifted_windows(r, rim_dets, fps, cfg)
            if drift_win[r.id]:
                debug["warnings"].append(f"rim {r.id}: stabilised rim detections drifted > {cfg.rim_drift_w} W "
                                         f"in windows {sorted(drift_win[r.id])}")
    elif not rims:
        rims = estimate_rims(rim_dets, frame_w, cfg)
        if rims and calib.homography is not None:
            # median labels (left=A, right=B) are arbitrary; the homography fixes which rim is A
            raise ConfigError("No rims in the calibration file but a homography exists: the median rim fallback "
                              "could label A and B the wrong way round. Add 'rims' to the calibration file.")
        if rims:
            debug["warnings"].append("no rims in the calibration file; used the median of rim detections")
    else:
        _check_rim_labels(rims, calib)
        for r in rims:
            if r.source == "manual" and _drifted(r, rim_dets, frame_w, cfg):
                drift.add(r.id)
                debug["warnings"].append(f"rim {r.id}: detected rim centre drifted > {cfg.rim_drift_w} W")
    debug["rims"] = [(r.id, r.box_px, r.source) for r in rims]
    if _provisional_geometry(calib.spec(), calib):
        debug["warnings"].append(f"level {calib.level!r}: 2/3 decision uses unverified court constants; "
                                 "points_conf is capped at 0.6 (note provisional_geometry)")

    # players (role_track == "player" when the column exists) and ball
    # design 4.3: when role_track exists it decides who is a player (not the per-frame class, which can flicker)
    if "role_track" in tracks.columns:
        pl = tracks[tracks["role_track"] == "player"]
    else:
        pl = tracks[cls == "player"]
    players: dict[int, list[tuple[int, tuple]]] = defaultdict(list)  # frame -> [(track_id, box)]
    by_id: dict[int, dict[int, tuple]] = defaultdict(dict)           # track_id -> frame -> box
    for f, tid, x1, y1, x2, y2 in zip(pl["frame"], pl["track_id"], pl["x1_px"], pl["y1_px"], pl["x2_px"], pl["y2_px"]):
        box = (float(x1), float(y1), float(x2), float(y2))
        players[int(f)].append((int(tid), box))
        if tid >= 0:
            by_id[int(tid)][int(f)] = box
    ball = tracks[cls == "ball"]
    debug["n_ball_detections"] = len(ball)
    if not rims or ball.empty:
        debug["warnings"].append("no usable rim" if not rims else "no ball detections")
        return DetectionResult([], debug)

    # the speed gate scales with the NEARER (wider) rim's width, so thresholds at the far rim are looser than
    # 'W' there implies; accepted because one gate serves the whole ball track.
    ref_w = max(r.w_px for r in rims)
    bx, by = (ball["x1_px"] + ball["x2_px"]) / 2, (ball["y1_px"] + ball["y2_px"]) / 2
    tr = clean_ball_track(ball["frame"].to_numpy(), bx.to_numpy(), by.to_numpy(), ball["conf"].to_numpy(),
                          (ball["x2_px"] - ball["x1_px"]).to_numpy(), fps, ref_w, cfg)
    n = len(tr)
    debug["n_trajectory_points"] = n
    debug["n_segments"] = int(tr.seg.max()) + 1 if n else 0
    if n == 0:
        return DetectionResult([], debug)
    held = np.array([_held(tr.x_px[i], tr.y_px[i], [b for _, b in players.get(int(tr.frame[i]), [])], cfg)
                     for i in range(n)])

    sec = lambda s: max(1, round(s * fps))  # noqa: E731  seconds -> frames
    min_flight = sec(cfg.min_flight_s)
    rows: list[ShotRow] = []
    last_outcome_frame = -10**9

    # flights: maximal runs of free frames inside one segment
    i = 0
    while i < n:
        if held[i]:
            i += 1
            continue
        j = i
        while j + 1 < n and not held[j + 1] and tr.seg[j + 1] == tr.seg[i] and tr.frame[j + 1] == tr.frame[j] + 1:
            j += 1
        s, e = i, j
        i = j + 1
        if e - s + 1 < min_flight:
            continue
        debug["n_flights"] += 1
        # rule 1: starts from a held state (recent), rule 2: rises on release
        if s == 0 or not held[s - 1] or tr.frame[s] - tr.frame[s - 1] > sec(cfg.max_release_gap_s):
            continue
        rise = slice(s, s + int(np.searchsorted(tr.frame[s:e + 1], tr.frame[s] + sec(cfg.rise_window_s), side="right")))
        if not np.mean(tr.vy[rise]) < 0:
            continue
        ended_by_loss = not (e + 1 < n and tr.seg[e + 1] == tr.seg[e])
        # rules 3 and 4 per rim; the rim with the smaller closest approach wins
        a_idx = s + int(np.argmin(tr.y_px[s:e + 1]))
        best = None
        for rim in rims:
            if tr.y_px[a_idx] > rim.top_px - cfg.apex_above_rim_w * rim.w_px:
                continue
            zone = next((k for k in range(a_idx, e + 1) if _in_zone(tr.x_px[k], tr.y_px[k], rim, cfg)), None)
            if zone is None:
                continue
            d = float(np.min(np.hypot(tr.x_px[a_idx:e + 1] - rim.cx_px, tr.y_px[a_idx:e + 1] - rim.cy_px))) / rim.w_px
            if best is None or d < best[0]:
                best = (d, rim, zone)
        if best is None:
            if ended_by_loss:
                debug["n_flights_lost"] += 1
            continue
        _, rim, zone = best
        r_idx = s - 1
        release_frame = int(tr.frame[r_idx])
        if release_frame < last_outcome_frame + cfg.attempt_lockout_s * fps:
            debug["n_lockout_suppressed"] += 1
            continue

        out = judge_outcome(tr, held, zone, rim, cfg, clip_last)
        last_outcome_frame = out.frame
        notes = list(out.notes)
        attempt_conf = 0.6 if (tr.interp[a_idx] or tr.interp[zone]) else 0.9
        if tr.seg[r_idx] != tr.seg[s]:
            release_conf = 0.3
        elif tr.interp[r_idx] or tr.interp[s] or tr.frame[s] - tr.frame[r_idx] > 1:
            release_conf = 0.6
        else:
            release_conf = 0.9
        result_conf = out.conf
        if rim.id in drift or int(release_frame / fps // _DRIFT_WINDOW_S) in drift_win.get(rim.id, ()):
            notes.append("rim_drift")
            result_conf = min(result_conf, cfg.rim_drift_conf_cap)

        held_ball = {int(tr.frame[k]): (float(tr.x_px[k]), float(tr.y_px[k]))
                     for k in range(r_idx, -1, -1) if held[k] and tr.frame[k] >= release_frame - sec(cfg.shooter_vote_s)}
        row = _shooter_and_points(release_frame, held_ball, players, by_id, calib, rim, fps, cfg,
                                  frame_height_px, notes, feet_cut)
        row.update(frame=out.frame, time_s=out.frame / fps, result=out.result, release_frame=release_frame,
                   release_time_s=release_frame / fps, ball_release_x_px=float(tr.x_px[r_idx]),
                   ball_release_y_px=float(tr.y_px[r_idx]), rim_id=rim.id, level=calib.level,
                   attempt_conf=attempt_conf, release_conf=release_conf, result_conf=result_conf)
        rows.append(ShotRow(notes=notes, **row))
        debug["n_attempts"] += 1

    rows.sort(key=lambda r: (r.frame, r.release_frame))
    for k, r in enumerate(rows):
        r.shot_id = k
    return DetectionResult(rows, debug)


def detect_shots(tracks: pd.DataFrame, calib: Calibration, fps: float, cfg: ShotConfig | None = None,
                 frame_height_px: float | None = None, motion: MotionTrack | None = None,
                 feet_cut: set[tuple[int, int]] | None = None) -> DetectionResult:
    """Detect attempts and score them. `fps` is the video's frame rate (never assumed).

    `camera_static` and `motion` decide the path (design 005 section 10):
      - no motion file, static camera: the original engine, unchanged
      - no motion file, camera_static false: ConfigError (run scripts/stabilize.py)
      - static camera, motion measured static: the original engine (identity transform)
      - static camera, motion measured moving: original engine, warning, confidences capped at 0.3
      - camera_static false with a motion file whose id matches the calibration: stabilised scoring
    """
    cfg = cfg or ShotConfig()
    if motion is not None:
        if abs(float(motion.meta["fps"]) - fps) > 1e-3:
            raise ConfigError(f"The motion file was made at {motion.meta['fps']} fps but shots are scored at {fps} fps: "
                              f"it belongs to a different video or frame rate.")
        if len(tracks) and int(tracks["frame"].max()) >= motion.n_frames:
            raise ConfigError(f"The tracks reach frame {int(tracks['frame'].max())} but the motion file covers only "
                              f"{motion.n_frames} frames (0-{motion.n_frames - 1}); re-run scripts/stabilize.py "
                              f"on the same clip.")
    if motion is None:
        if not calib.camera_static:
            raise ConfigError("The calibration says camera_static is false (a panning camera), so a motion file is "
                              "required: run scripts/stabilize.py on the video and pass --motion.")
        return _detect_core(tracks, calib, fps, cfg, frame_height_px, feet_cut)
    if calib.camera_static:
        res = _detect_core(tracks, calib, fps, cfg, frame_height_px, feet_cut)
        if not motion.is_static:
            msg = ("the calibration says camera_static but the motion file shows the camera MOVED; the rim box and "
                   "court homography are wrong on most frames. Results are capped at confidence 0.3. Recalibrate "
                   "on the mosaic (scripts/calibrate_court.py --mosaic).")
            print(f"WARNING: {msg}", file=sys.stderr)
            res.debug["warnings"].append(msg)
            for r in res.shots:
                r.notes.append("camera_moved")
                r.result_conf = min(r.result_conf, 0.3)
                r.points_conf = min(r.points_conf, 0.3)
        return res
    check_motion_id(calib, motion)
    return _detect_stabilised(tracks, calib, fps, cfg, frame_height_px, motion, feet_cut)


def _zero_runs(q: np.ndarray) -> list[tuple[int, int]]:
    """Inclusive (start, end) frame runs where quality is 0."""
    z = np.concatenate([[False], q <= 0, [False]]).astype(np.int8)
    d = np.diff(z)
    return list(zip(np.flatnonzero(d == 1).tolist(), (np.flatnonzero(d == -1) - 1).tolist()))


def _detect_stabilised(tracks: pd.DataFrame, calib: Calibration, fps: float, cfg: ShotConfig,
                       frame_height_px: float | None, motion: MotionTrack,
                       feet_cut: set[tuple[int, int]] | None) -> DetectionResult:
    """Run the engine once per view that holds a rim, on tracks moved into that view (design 005 9.3)."""
    fh = frame_height_px or calib.frame_height or motion.meta.get("frame_height")
    if feet_cut is None:  # computed in RAW pixels, before any transform
        people = tracks[~tracks["class"].isin(["ball", "rim"])]
        cut = people["y2_px"] >= (fh - cfg.feet_edge_px) if fh else people["y2_px"] < -1
        feet_cut = {(int(t), int(f)) for t, f in zip(people.loc[cut, "track_id"], people.loc[cut, "frame"])}
    if not calib.rims:
        raise ConfigError("Stabilised scoring needs manual rims (clicked on the mosaic) in the calibration file.")
    gap_n = max(1, round(cfg.max_gap_s * fps))
    q_all = motion.quality
    debug: dict = {"fps": fps, "n_flights": 0, "n_flights_lost": 0, "n_lockout_suppressed": 0, "n_attempts": 0,
                   "warnings": [], "n_ball_detections": 0, "n_trajectory_points": 0, "n_segments": 0, "rims": [],
                   "coordinate_space": "view", "n_views": len({r.view for r in calib.rims}),
                   "n_unreliable_frames": int((q_all <= 0).sum()),
                   "n_unreliable_runs_over_gap": sum(1 for a, b in _zero_runs(q_all) if b - a + 1 > gap_n),
                   "max_closure_err_px": float(motion.meta.get("max_closure_err_px", 0.0))}
    rows: list[ShotRow] = []
    for v in sorted({r.view for r in calib.rims}):
        if v >= len(motion.views):
            raise ConfigError(f"A rim is assigned to view {v}, but the motion file has {len(motion.views)} views.")
        hom = calib.homography
        if hom is not None:  # `hom.H_px_to_ft` holds H_ref_to_ft for a mosaic calibration
            H_view_to_ft = hom.H_px_to_ft @ np.linalg.inv(motion.views[v].H_ref_to_view)
            hom = replace(hom, H_px_to_ft=H_view_to_ft)
        calib_v = replace(calib, rims=tuple(r for r in calib.rims if r.view == v), homography=hom)
        res = _detect_core(transform_tracks(tracks, motion, v), calib_v, fps, cfg, None, feet_cut, stabilised=True)
        q_v = motion.view_quality(v)
        runs_over = [(a, b) for a, b in _zero_runs(q_v) if b - a + 1 > gap_n]
        take = max(1, round(cfg.takeoff_window_s * fps))
        for r in res.shots:
            lo, hi = r.release_frame, max(r.frame, r.release_frame)
            span = q_v[lo:hi + 1]
            r.view_id = v
            r.camera_quality = float(span.min()) if len(span) else 0.0
            if any(a <= hi and b >= lo for a, b in runs_over):
                r.result, r.result_conf = "unknown", 0.0
                r.notes.append("camera_unreliable")
            elif len(span) and span.min() < 0.9:
                r.result_conf = min(r.result_conf, 0.6)
                r.points_conf = min(r.points_conf, 0.6)
                r.notes.append("camera_degraded")
            if "no_shooter" in r.notes and (q_v[max(0, lo - take):lo + 1] <= 0).any():
                r.notes.append("camera_unreliable")
        rows.extend(res.shots)
        for k in ("n_flights", "n_flights_lost", "n_lockout_suppressed", "n_attempts", "n_ball_detections",
                  "n_trajectory_points", "n_segments"):
            debug[k] += res.debug.get(k, 0)
        debug["warnings"] += res.debug["warnings"]
        debug["rims"] += res.debug.get("rims", [])
    rows.sort(key=lambda r: (r.frame, r.release_frame))
    for k, r in enumerate(rows):
        r.shot_id = k
    return DetectionResult(rows, debug)


# --------------------------------------------------------------------------- shooter, feet, points

def _shooter_and_points(release_frame: int, held_ball: dict, players: dict, by_id: dict, calib: Calibration,
                        rim: Rim, fps: float, cfg: ShotConfig, frame_height_px: float | None,
                        notes: list[str], feet_cut: set[tuple[int, int]] | None = None) -> dict:
    """Shooter vote (4.5), feet at take-off, and the shot value (4.6, 4.7). Returns ShotRow kwargs."""
    res = dict(shooter_track_id=None, feet_x_px=None, feet_y_px=None, feet_x_ft=None, feet_y_ft=None,
               margin_ft=None, shooter_conf=0.0, points=None, points_conf=0.0)
    vote = _vote_shooter(release_frame, held_ball, players, fps, cfg)
    if vote is None:
        notes.append("no_shooter")
        return res
    tid, conf = vote
    res.update(shooter_track_id=tid, shooter_conf=conf)

    # feet at take-off: the frame in the 0.5 s before release where the shooter's y2 is largest
    boxes = by_id.get(tid, {})
    lo = release_frame - max(1, round(cfg.takeoff_window_s * fps))
    cand = [(f, b) for f, b in boxes.items() if lo <= f <= release_frame]
    if not cand:
        notes.append("no_shooter")
        res.update(shooter_track_id=None, shooter_conf=0.0)
        return res
    f_off, box = max(cand, key=lambda fb: (fb[1][3], fb[0]))
    feet_px = ((box[0] + box[2]) / 2, box[3])
    res.update(feet_x_px=feet_px[0], feet_y_px=feet_px[1])
    # feet_cut (stabilised scoring) was computed from RAW pixels before the transform; otherwise test the box here
    cut = ((tid, f_off) in feet_cut) if feet_cut is not None else (
        frame_height_px is not None and box[3] >= frame_height_px - cfg.feet_edge_px)
    if cut:
        notes.append("feet_cut_off")
        return res
    if calib.homography is None:
        notes.append("no_homography")
        return res
    length = calib.effective_length_ft()
    try:
        feet_ft = to_rim_frame(px_to_ft(calib.homography.H_px_to_ft, np.array([feet_px])), rim.id, length)[0]
    except ConfigError:
        notes.append("no_court_length")
        return res
    res.update(feet_x_ft=float(feet_ft[0]), feet_y_ft=float(feet_ft[1]))
    spec = calib.spec()
    if feet_off_court(feet_ft, spec, length, cfg.off_court_ft):
        notes.append("feet_off_court")
        return res
    radius = calib.three_point_radius_ft
    try:
        three = is_three(feet_ft, calib.level, radius)
        margin = three_margin_ft(feet_ft, calib.level, radius)
    except ConfigError:
        notes.append("no_homography")
        return res
    res["margin_ft"] = margin
    pts_conf = 0.9 if margin >= cfg.near_arc_ft else 0.3
    if margin < cfg.near_arc_ft:
        notes.append("near_arc")
    if calib.homography.rms_ft > cfg.homography_conf_cap_rms_ft:
        pts_conf = min(pts_conf, 0.6)
    if _provisional_geometry(spec, calib):
        notes.append("provisional_geometry")
        pts_conf = min(pts_conf, 0.6)
    res.update(points=3 if three else 2, points_conf=pts_conf)
    if _free_throw(feet_ft, tid, release_frame, by_id, calib, rim, fps, cfg, length):
        notes.append("ft_rule")
        res.update(points=1, points_conf=min(cfg.ft_conf_cap, pts_conf if pts_conf > 0.3 else cfg.ft_conf_cap))
    return res


def _provisional_geometry(spec, calib: Calibration) -> bool:
    """True if a constant used by the 2/3 decision is unverified (design 5 flag), i.e. not a reading."""
    used = [spec.basket_offset_ft, spec.corner_half_width_ft]
    if calib.three_point_radius_ft is None:
        used.append(spec.three_radius_ft)
    return any(not v.verified for v in used)


def _vote_shooter(release_frame: int, held_ball: dict[int, tuple[float, float]], players: dict, fps: float,
                  cfg: ShotConfig) -> tuple[int, float] | None:
    """Design 4.5: the track nearest the ball over the last 0.25 s of held frames wins the vote.

    Returns (track_id, shooter_conf) or None (no_shooter) when nobody voted, the winner is an
    untracked box, or the winner is farther from the ball than 0.5 of its box height at release.
    """
    lo = release_frame - max(1, round(cfg.shooter_vote_s * fps))
    votes: Counter = Counter()
    dist_sum: dict[int, float] = defaultdict(float)
    last_dist: dict[int, float] = {}
    last_h: dict[int, float] = {}
    n_voted = 0
    for f in sorted(held_ball):
        if not (lo < f <= release_frame) or not players.get(f):
            continue
        bx, by = held_ball[f]
        tid, box = min(players[f], key=lambda tb: _rect_dist(bx, by, tb[1]))
        d = _rect_dist(bx, by, box)
        n_voted += 1
        votes[tid] += 1
        dist_sum[tid] += d
        last_dist[tid], last_h[tid] = d, box[3] - box[1]
    if not votes:
        return None
    tid = min(votes, key=lambda t: (-votes[t], dist_sum[t]))
    if tid < 0 or last_dist[tid] > cfg.shooter_max_dist_h * last_h[tid]:
        return None
    share = votes[tid] / n_voted
    conf = 0.9 if share >= cfg.shooter_conf_hi else 0.6 if share >= cfg.shooter_conf_mid else 0.3
    return tid, conf


def _free_throw(feet_ft: np.ndarray, tid: int, release_frame: int, by_id: dict, calib: Calibration, rim: Rim,
                fps: float, cfg: ShotConfig, length: float | None) -> bool:
    """Design 4.7: at the line, still, nobody within 4 ft, and most others still. All thresholds are T."""
    ft_line = calib.spec().ft_line_ft.value
    if ft_line is None:
        return False
    x, y = float(feet_ft[0]), float(feet_ft[1])
    if abs(x) > cfg.ft_axis_ft or not (ft_line - cfg.ft_in_front_ft <= y <= ft_line + cfg.ft_behind_ft):
        return False
    H = calib.homography.H_px_to_ft
    lo = release_frame - max(1, round(cfg.ft_still_s * fps))

    def series_ft(boxes: dict) -> np.ndarray:
        fp = np.array([[(b[0] + b[2]) / 2, b[3]] for f, b in boxes.items() if lo <= f <= release_frame])
        return to_rim_frame(px_to_ft(H, fp), rim.id, length) if len(fp) else np.zeros((0, 2))

    def spread(p: np.ndarray) -> float:
        if len(p) < 2:
            return math.inf
        return float(max(np.hypot(*(p[i] - p[j])) for i in range(len(p)) for j in range(i + 1, len(p))))

    if spread(series_ft(by_id[tid])) >= cfg.ft_still_ft:
        return False
    others = [o for o in by_id if o != tid and release_frame in by_id[o]]
    if not others:
        return False
    still = 0
    for o in others:
        p_rel = to_rim_frame(px_to_ft(H, np.array([[(by_id[o][release_frame][0] + by_id[o][release_frame][2]) / 2,
                                                    by_id[o][release_frame][3]]])), rim.id, length)[0]
        if math.hypot(*(p_rel - feet_ft)) < cfg.ft_clear_ft:
            return False
        if spread(series_ft(by_id[o])) < cfg.ft_others_still_ft:
            still += 1
    return still / len(others) >= cfg.ft_others_still_frac


# --------------------------------------------------------------------------- output

def write_shots_csv(result: DetectionResult, path: Path) -> None:
    """Write `shots.csv` with the columns of design 8.2. Empty cells mean unknown."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    def cell(v, nd=None):
        if v is None:
            return ""
        return round(float(v), nd) if nd is not None else v

    with path.open("w", newline="") as fh:
        w = csv.writer(fh)
        extra = result.debug.get("coordinate_space") == "view"  # stabilised scoring: two columns at the end
        w.writerow(COLUMNS + (MOTION_COLUMNS if extra else []))
        for r in result.shots:
            w.writerow([
                r.frame, f"{r.time_s:.3f}", r.result, "" if r.points is None else r.points,
                ";".join(dict.fromkeys(r.notes)),
                r.shot_id, r.release_frame, f"{r.release_time_s:.3f}", "" if r.shooter_track_id is None else r.shooter_track_id,
                cell(r.ball_release_x_px, 1), cell(r.ball_release_y_px, 1), cell(r.feet_x_px, 1), cell(r.feet_y_px, 1),
                cell(r.feet_x_ft, 2), cell(r.feet_y_ft, 2),
                r.rim_id, r.level, cell(r.margin_ft, 2),
                r.attempt_conf, r.release_conf, r.result_conf, r.shooter_conf, r.points_conf,
            ] + ([r.view_id, cell(r.camera_quality, 2)] if extra else []))

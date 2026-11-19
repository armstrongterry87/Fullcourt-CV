"""Tests for task 005 step A2: camera motion compensation (src/camera/motion.py) and its use in shot scoring.

All data is synthetic. A big textured planar "world" image is warped by known homographies
(a rotating camera: P = S @ K R K^-1), so the true frame-to-frame motion is known exactly.
Nothing is written outside pytest's tmp_path. Scene tests run at two frame rates (23.951 and 60).
"""

from __future__ import annotations

import dataclasses
import json
import sys
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import pytest

from src.camera.motion import (
    MotionConfig, MotionTrack, View, build_mask, cache_path, ensure_private_output, estimate_motion,
    expected_meta, load_motion, motion_key, save_motion, stretch_at, transform_points, transform_tracks,
    _motion_id,
)
from src.court.polygon import ConfigError

W, H = 640, 360
F_PX = 600.0
K = np.array([[F_PX, 0, W / 2], [0, F_PX, H / 2], [0, 0, 1.0]])
WORLD_W, WORLD_H = 3000, 1600
S_WORLD = np.array([[1.0, 0, 1180], [0, 1.0, 620], [0, 0, 1.0]])  # identity camera sits at (1180, 620) in the world
FPS_LIST = [23.951, 60.0]
CFG = MotionConfig()


@lru_cache(maxsize=1)
def world_image() -> np.ndarray:
    """A textured planar scene: noise blobs, lines, rectangles, circles (random but seeded)."""
    rng = np.random.default_rng(7)
    img = (rng.random((WORLD_H // 8, WORLD_W // 8, 3)) * 255).astype(np.uint8)
    img = cv2.resize(img, (WORLD_W, WORLD_H), interpolation=cv2.INTER_CUBIC)
    img = cv2.GaussianBlur(img, (0, 0), 2.0)
    for _ in range(900):
        x, y = int(rng.integers(0, WORLD_W)), int(rng.integers(0, WORLD_H))
        c = tuple(int(v) for v in rng.integers(0, 255, 3))
        kind = rng.integers(0, 3)
        if kind == 0:
            cv2.rectangle(img, (x, y), (x + int(rng.integers(8, 40)), y + int(rng.integers(8, 40))), c, -1)
        elif kind == 1:
            cv2.circle(img, (x, y), int(rng.integers(4, 20)), c, -1)
        else:
            cv2.line(img, (x, y), (x + int(rng.integers(-60, 60)), y + int(rng.integers(-60, 60))), c, 2)
    return img


def rot(pan_deg: float, tilt_deg: float = 0.0, roll_deg: float = 0.0) -> np.ndarray:
    a, b, c = np.radians([pan_deg, tilt_deg, roll_deg])
    Ry = np.array([[np.cos(a), 0, np.sin(a)], [0, 1, 0], [-np.sin(a), 0, np.cos(a)]])
    Rx = np.array([[1, 0, 0], [0, np.cos(b), -np.sin(b)], [0, np.sin(b), np.cos(b)]])
    Rz = np.array([[np.cos(c), -np.sin(c), 0], [np.sin(c), np.cos(c), 0], [0, 0, 1]])
    return Rz @ Rx @ Ry


def pose(pan: float, tilt: float = 0.0, roll: float = 0.0) -> np.ndarray:
    """True H_frame_to_world for a camera rotated by (pan, tilt, roll) degrees."""
    return S_WORLD @ K @ rot(pan, tilt, roll) @ np.linalg.inv(K)


def render(P: np.ndarray) -> np.ndarray:
    return cv2.warpPerspective(world_image(), P, (W, H), flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP)


def corner_err(Ha: np.ndarray, Hb: np.ndarray) -> float:
    """Largest corner disagreement (px) between two frame->X homographies."""
    c = np.array([[0, 0], [W, 0], [W, H], [0, H]], float)
    return float(np.max(np.linalg.norm(transform_points(Ha, c) - transform_points(Hb, c), axis=1)))


def rel_err(motion: MotionTrack, poses: list[np.ndarray], i: int, j: int) -> float:
    """Error of the relative motion i -> j (in frame j pixels) against the truth. Independent of the anchor."""
    est = np.linalg.inv(motion.H_frame_to_ref[j]) @ motion.H_frame_to_ref[i]
    true = np.linalg.inv(poses[j]) @ poses[i]
    return corner_err(est, true)


def run(frames: list[np.ndarray], fps: float = 30.0, boxes=None, cfg: MotionConfig = CFG) -> MotionTrack:
    boxes = boxes if boxes is not None else [np.zeros((0, 4)) for _ in frames]
    return estimate_motion(lambda: iter(frames), boxes, fps, cfg)


# --------------------------------------------------------------------------- homography maths

def test_direction_and_composition_order():
    """A known frame point lands at a known world point; composing in the wrong order would not."""
    poses = [pose(0.0, 0.0, 0.0), pose(10.0, 5.0, -4.0), pose(18.0, -3.0, 6.0)]
    P = poses[2]
    pt = np.array([[100.0, 50.0]])
    expect = transform_points(P, pt)
    assert not np.allclose(expect, pt, atol=20)  # a real, non-identity move
    Hab = np.linalg.inv(poses[1]) @ poses[0]
    Hbc = np.linalg.inv(poses[2]) @ poses[1]
    Hac = np.linalg.inv(poses[2]) @ poses[0]
    # right-most applied first: a -> b then b -> c
    assert np.allclose(Hbc @ Hab / (Hbc @ Hab)[2, 2], Hac / Hac[2, 2], atol=1e-6)
    wrong = Hab @ Hbc
    assert not np.allclose(wrong / wrong[2, 2], Hac / Hac[2, 2], atol=1e-3)


def test_stretch_at_known_pan():
    """A pure pan of 35 degrees stretches by 1/cos^2 at the image edge of the identity view."""
    Hm = K @ rot(35.0) @ np.linalg.inv(K)
    s0 = stretch_at(Hm, (W / 2, H / 2))
    assert 1.0 < s0 < 1.5
    assert stretch_at(np.eye(3), (10, 10)) == pytest.approx(1.0)
    assert stretch_at(np.array([[1.0, 0, 0], [0, 1, 0], [0.01, 0, -1]]), (10, 10)) == np.inf  # behind the plane


# --------------------------------------------------------------------------- recovered pans

@pytest.mark.parametrize("fps", FPS_LIST)
def test_recovers_known_pan_tilt_roll(fps):
    """Mixed pan + tilt + roll: relative motion between all frame pairs matches the truth within 1.5 px."""
    seq = [(-10 + 2.5 * i, 0.4 * i - 1.6, 0.5 * i - 2) for i in range(9)]
    poses = [pose(*s) for s in seq]
    m = run([render(P) for P in poses], fps)
    assert (m.quality > 0).all()
    assert not m.is_static
    assert np.abs(poses[0] - poses[-1]).max() > 50  # the pan is not trivial
    for i, j in [(0, 8), (3, 7), (8, 1), (0, 1), (5, 6)]:
        assert rel_err(m, poses, i, j) < 1.5, (i, j)


def test_frame_point_lands_at_reference_point():
    """Direction check on estimated output: a frame-j point maps (via H_frame_to_ref) to the same ref
    point as the matching frame-i point of the same world location."""
    poses = [pose(-8, 2, 2), pose(0, 0, 0), pose(9, -2, -3)]
    m = run([render(P) for P in poses], 5.0)  # frames 0.2 s apart: a big jump between them is legitimate
    world_pt = np.array([[1300.0, 700.0]])
    p_i = transform_points(np.linalg.inv(poses[0]), world_pt)
    p_j = transform_points(np.linalg.inv(poses[2]), world_pt)
    ref_i = transform_points(m.H_frame_to_ref[0], p_i)
    ref_j = transform_points(m.H_frame_to_ref[2], p_j)
    assert np.linalg.norm(ref_i - ref_j) < 2.0
    assert np.linalg.norm(p_i - p_j) > 30  # the raw positions differ a lot


# --------------------------------------------------------------------------- masking

def blob_frames(poses, with_boxes: bool):
    """A big, high-contrast, textured blob that stays put in the frame (so it votes for "no motion") and owns most features."""
    rng = np.random.default_rng(3)
    blob = (rng.random((100, 130, 3)) * 255).astype(np.uint8)
    blob = cv2.resize(blob, (260, 200), interpolation=cv2.INTER_NEAREST)
    frames, boxes = [], []
    for i, P in enumerate(poses):
        f = cv2.convertScaleAbs(render(P), alpha=0.4, beta=80)  # a low-contrast wall behind a high-contrast object
        x, y = 20 + (i % 2), 30  # a textured object that moves with the camera (a close-up person)
        f[y:y + 200, x:x + 260] = blob
        frames.append(f)
        boxes.append(np.array([[x, y, x + 260, y + 200]], float) if with_boxes else np.zeros((0, 4)))
    return frames, boxes


def test_masking_ignores_a_moving_blob():
    poses = [pose(-6 + 4 * i, 0, 0) for i in range(7)]
    frames, boxes = blob_frames(poses, True)
    masked = run(frames, 30.0, boxes)
    unmasked = run(frames, 30.0, [np.zeros((0, 4)) for _ in frames])
    err_masked = max(rel_err(masked, poses, 0, j) for j in range(1, 7) if masked.quality[j] > 0)
    assert (masked.quality > 0).all()
    assert err_masked < 2.0
    # without the mask the blob (which moves with the camera) drags the fit: wrong by more than 10 px,
    # or the frames are thrown away as unreliable
    bad = [rel_err(unmasked, poses, 0, j) if unmasked.quality[j] > 0 and unmasked.quality[0] > 0 else np.inf
           for j in range(1, 7)]
    assert max(bad) > 10.0


def test_mask_for_frame_t_uses_boxes_of_frame_t():
    boxes = [np.array([[10, 10, 50, 50]], float), np.array([[300, 200, 340, 260]], float)]
    m0 = build_mask((H, W), boxes[0], CFG)
    m1 = build_mask((H, W), boxes[1], CFG)
    assert m0[30, 30] == 0 and m0[230, 320] == 255   # frame 0 mask holds frame 0's box only
    assert m1[230, 320] == 0 and m1[30, 30] == 255
    # border, dilation (0.15 * 40 + 8 = 14 px) and ignore rects
    assert m0[0, 100] == 0 and m0[30, 50 + 13] == 0 and m0[30, 50 + 16] == 255
    cfg = dataclasses.replace(CFG, ignore_rects_px=((200, 100, 260, 160),))
    assert build_mask((H, W), np.zeros((0, 4)), cfg)[130, 230] == 0

    # estimate_motion must apply boxes[t] to frame t (same-frame mask): record what it asks for
    import src.camera.motion as mo
    seen = []
    orig = mo.build_mask

    def spy(hw, b, cfg):
        seen.append(np.asarray(b).copy())
        return orig(hw, b, cfg)

    frames = [render(pose(4 * i)) for i in range(3)]
    marks = [np.array([[10.0 + 100 * i, 10, 40.0 + 100 * i, 40]]) for i in range(3)]
    mo.build_mask = spy
    try:
        estimate_motion(lambda: iter(frames), marks, 30.0, dataclasses.replace(CFG, pass1_hz=30.0))
    finally:
        mo.build_mask = orig
    assert len(seen) >= 3
    for b in seen:  # every mask call got exactly one frame's box, never a mix
        assert len(b) == 1 and any(np.allclose(b, mk) for mk in marks)
    assert [b[0, 0] for b in seen[:3]] == [10.0, 110.0, 210.0]  # pass 1 order: frame 0, 1, 2 -> boxes 0, 1, 2


# --------------------------------------------------------------------------- drift

def test_long_sweep_drift_is_bounded():
    """Out and back twice over +-28 degrees: relative error vs the truth stays small, and no worse at the end."""
    sweep = list(np.arange(-28, 28.1, 2.0)) + list(np.arange(26, -28.1, -2.0))
    pans = sweep + sweep
    poses = [pose(p, 0.1 * p, 0) for p in pans]
    m = run([render(P) for P in poses], 30.0)
    assert (m.quality > 0).all(), f"unreliable frames: {np.flatnonzero(m.quality == 0)}"
    assert len(m.stats["key_frame_idx"]) >= 3  # several keyframes, so the graph is exercised
    n1 = len(sweep)
    mid = 14  # pan 0: errors are measured in this frame's pixels, so wide-angle corner stretch does not dominate
    assert pans[mid] == 0.0
    err = [rel_err(m, poses, j, mid) for j in range(2 * n1)]
    assert max(err) <= 4.0, f"worst frame {int(np.argmax(err))}: {max(err):.1f} px"
    # the second cycle revisits the same poses: drift must not accumulate
    assert max(err[n1:]) <= max(err[:n1]) + 1.0
    assert np.mean(err[n1:]) <= np.mean(err[:n1]) + 0.5
    assert m.meta["max_closure_err_px"] <= 5.0


# --------------------------------------------------------------------------- quality

def test_textureless_frame_is_unreliable():
    poses = [pose(3 * i) for i in range(7)]
    frames = [render(P) for P in poses]
    frames[3] = np.full((H, W, 3), 128, np.uint8)
    m = run(frames)
    assert m.quality[3] == 0.0 and np.all(np.isnan(m.H_frame_to_ref[3]))
    assert all(m.quality[j] > 0 for j in (0, 1, 2, 4, 5, 6))
    assert not m.usable(3)
    assert m.H_frame_to_view(3) is None


def test_blurred_frame_is_not_trusted_as_good():
    """A heavily blurred frame must never be reported at quality 0.9 with a wrong pose."""
    poses = [pose(3 * i) for i in range(6)]
    frames = [render(P) for P in poses]
    frames[2] = cv2.GaussianBlur(frames[2], (0, 0), 25)
    m = run(frames)
    if m.quality[2] > 0:
        assert rel_err(m, poses, 0, 2) < 3.0
    assert m.quality[2] < 0.9


# --------------------------------------------------------------------------- static video

def static_frames(n=12):
    base = render(pose(0))
    rng = np.random.default_rng(1)
    out = []
    for i in range(n):
        f = base.copy()
        f[100:160, 50 + 10 * i:110 + 10 * i] = rng.integers(0, 255, (60, 60, 3), dtype=np.uint8)  # a moving person
        out.append(f)
    return out


def test_static_video_gives_identity():
    frames = static_frames()
    boxes = [np.array([[50 + 10 * i, 100, 110 + 10 * i, 160]], float) for i in range(len(frames))]
    m = run(frames, 30.0, boxes)
    assert m.is_static
    assert all(np.array_equal(m.H_frame_to_ref[i], np.eye(3)) for i in range(len(frames)))
    assert (m.quality == 0.9).all()
    assert len(m.views) == 1 and (m.views[0].canvas_w, m.views[0].canvas_h) == (W, H)
    assert m.usable(5)
    pts = np.array([[10.0, 20.0], [300.0, 200.0]])
    assert np.allclose(m.to_view(pts, np.array([0, 5])), pts)


def test_moving_video_is_not_static():
    m = run([render(pose(4 * i)) for i in range(6)])
    assert not m.is_static
    assert max(corner_err(m.H_frame_to_ref[i], np.eye(3)) for i in range(6) if m.quality[i] > 0) > 20


# --------------------------------------------------------------------------- views and transform_tracks

def test_views_split_when_pan_is_wide():
    """A 68 degree sweep with a tight stretch cap (1.1) cannot sit on one flat canvas: more than one view,
    and every usable frame has a stretch of at most the frame cap in the view that holds its pixels."""
    pans = list(np.arange(-34, 34.1, 4.0))
    poses = [pose(p) for p in pans]
    cfg = dataclasses.replace(CFG, view_max_stretch=1.1, frame_max_stretch=1.2)
    m = run([render(P) for P in poses], 30.0, cfg=cfg)
    assert (m.quality > 0).all()
    assert len(m.views) >= 2
    for v in range(len(m.views)):
        for f in range(len(pans)):
            if m.usable(f, v):
                assert stretch_at(m.H_frame_to_view(f, v), (W / 2, H / 2)) <= 1.2 + 1e-9
    assert any(m.usable(f, 0) for f in range(len(pans)))
    assert not all(m.usable(f, 0) for f in range(len(pans)))  # the far frames are NaN in view 0
    p = m.to_view(np.array([[W / 2, H / 2]] * len(pans)), np.arange(len(pans)), 0)
    assert np.isnan(p).any() and np.isfinite(p).any()


def test_transform_tracks_keeps_feet_and_centre_exactly():
    poses = [pose(-6, 2, 2), pose(0, 0, 0), pose(7, -2, -2)]
    m = run([render(P) for P in poses])
    rows = []
    for f in range(3):
        rows.append(dict(frame=f, time_s=0.0, **{"class": "player"}, track_id=1, conf=0.9,
                         x1_px=200.0, y1_px=100.0, x2_px=260.0, y2_px=300.0))
        rows.append(dict(frame=f, time_s=0.0, **{"class": "ball"}, track_id=-1, conf=0.9,
                         x1_px=400.0, y1_px=60.0, x2_px=420.0, y2_px=80.0))
    df = pd.DataFrame(rows)
    out = transform_tracks(df, m, 0)
    assert len(out) == 6
    for _, r in out.iterrows():
        f = int(r["frame"])
        if r["class"] == "player":
            feet = m.to_view(np.array([[230.0, 300.0]]), np.array([f]))[0]
            assert np.allclose([(r.x1_px + r.x2_px) / 2, r.y2_px], feet, atol=1e-6)
            assert r.y1_px < r.y2_px
        else:
            c = m.to_view(np.array([[410.0, 70.0]]), np.array([f]))[0]
            assert np.allclose([(r.x1_px + r.x2_px) / 2, (r.y1_px + r.y2_px) / 2], c, atol=1e-6)
    # unreliable frames are dropped
    bad = dataclasses.replace(m, H_frame_to_ref=m.H_frame_to_ref.copy(), _cache={})
    bad.H_frame_to_ref[1] = np.nan
    assert set(transform_tracks(df, bad, 0)["frame"]) == {0, 2}


# --------------------------------------------------------------------------- cache key and files

FP = {"video_stem": "v", "size_bytes": 100, "head_sha256": "ab", "n_frames": 50, "fps": 23.951,
      "frame_width": 640, "frame_height": 360}


def _changed(value):
    if isinstance(value, bool):
        return not value
    if isinstance(value, int):
        return value + 1
    if isinstance(value, float):
        return value * 1.5 + 0.25
    if isinstance(value, tuple):
        return value + ((1, 2, 3, 4),)
    raise AssertionError(type(value))


def test_cache_key_changes_for_every_setting_and_input():
    base = motion_key(FP, "d" * 64, CFG)
    assert motion_key(dict(FP), "d" * 64, MotionConfig()) == base  # deterministic
    for f in dataclasses.fields(MotionConfig):
        cfg = dataclasses.replace(CFG, **{f.name: _changed(getattr(CFG, f.name))})
        assert motion_key(FP, "d" * 64, cfg) != base, f"setting {f.name} is not in the cache key"
    for k in FP:
        fp = dict(FP)
        fp[k] = fp[k] + 1 if not isinstance(fp[k], str) else fp[k] + "x"
        assert motion_key(fp, "d" * 64, CFG) != base, f"fingerprint field {k} is not in the cache key"
    assert motion_key(FP, "e" * 64, CFG) != base  # detections hash
    assert cache_path(Path("x/v.mp4"), base, Path("c")).name == f"v__motion__{base[:8]}.npz"


def test_save_load_roundtrip_and_mismatch_names_the_field(tmp_path):
    m = run([render(pose(4 * i - 8)) for i in range(5)], 23.951)
    exp = expected_meta(FP, "d" * 64, CFG)
    m = dataclasses.replace(m, meta={**m.meta, **exp})
    p = tmp_path / "m.npz"
    save_motion(p, m)
    back = load_motion(p, exp)
    assert np.array_equal(back.H_frame_to_ref, m.H_frame_to_ref, equal_nan=True)
    assert back.motion_id == m.motion_id and len(back.views) == len(m.views)
    assert np.allclose(back.views[0].H_ref_to_view, m.views[0].H_ref_to_view)
    for field_path, mutate in [("fingerprint.fps", lambda e: e["fingerprint"].__setitem__("fps", 60.0)),
                               ("settings.ratio", lambda e: e["settings"].__setitem__("ratio", 0.9)),
                               ("detections_sha256", lambda e: e.__setitem__("detections_sha256", "z")),
                               ("algo_version", lambda e: e.__setitem__("algo_version", 99))]:
        bad = json.loads(json.dumps(exp))
        mutate(bad)
        with pytest.raises(ValueError, match=field_path.replace(".", r"\.")):
            load_motion(p, bad)
    # tampered matrices are caught by motion_id
    z = dict(np.load(p, allow_pickle=False))
    z["H_frame_to_ref"] = z["H_frame_to_ref"] * 1.001
    np.savez_compressed(tmp_path / "t.npz", **z)
    with pytest.raises(ValueError, match="motion_id"):
        load_motion(tmp_path / "t.npz")


def test_private_output_guard(tmp_path):
    repo = tmp_path / "repo"
    (repo / "runs").mkdir(parents=True)
    assert ensure_private_output(repo / "runs" / "a" / "m.jpg", repo)
    assert ensure_private_output(repo / "data" / "frames" / "m.jpg", repo)
    for bad in (repo / "docs" / "m.jpg", repo / "data" / "raw" / "m.jpg", tmp_path / "m.jpg", repo / "runs_evil" / "m.jpg"):
        with pytest.raises(ValueError):
            ensure_private_output(bad, repo)


# --------------------------------------------------------------------------- shot scoring with motion

sys.path.insert(0, str(Path(__file__).parent))
import test_shots as ts  # noqa: E402
from src.court.geometry import load_calibration  # noqa: E402
from src.shots.detect import detect_shots, write_shots_csv  # noqa: E402


def static_motion(n: int, fps: float) -> MotionTrack:
    ident = np.tile(np.eye(3), (n, 1, 1))
    vH = np.eye(3)[None]
    mid = _motion_id(ident, vH)
    stats = {"key_frame_idx": np.array([0]), "key_view": np.array([0])}
    meta = {"frame_width": 1920, "frame_height": 1080, "frame_max_stretch": 3.0, "is_static": True,
            "motion_id": mid, "max_closure_err_px": 0.0, "n_frames": n, "fps": fps}
    return MotionTrack(meta, ident, np.full(n, 0.9, np.float32), stats, (View(0, 0, np.eye(3), 1920, 1080),), True, mid)


@pytest.mark.parametrize("fps", FPS_LIST)
def test_static_motion_leaves_scoring_unchanged(fps, tmp_path):
    tracks = ts.build_scene(fps, shooter_ft=(-21.0, 8.0))
    calib = ts.write_calib(tmp_path)
    plain = detect_shots(tracks, calib, fps)
    n = int(tracks["frame"].max()) + 1
    with_motion = detect_shots(tracks, calib, fps, motion=static_motion(n, fps))
    assert len(plain.shots) >= 1
    assert plain.shots == with_motion.shots
    a, b = tmp_path / "a.csv", tmp_path / "b.csv"
    write_shots_csv(plain, a)
    write_shots_csv(with_motion, b)
    assert a.read_bytes() == b.read_bytes()


def test_camera_static_semantics(tmp_path):
    fps = 30.0
    tracks = ts.build_scene(fps, shooter_ft=(-21.0, 8.0))
    n = int(tracks["frame"].max()) + 1
    calib = ts.write_calib(tmp_path)
    # false + no motion file: refuse and name the tool
    from dataclasses import replace
    with pytest.raises(ConfigError, match="stabilize"):
        detect_shots(tracks, replace(calib, camera_static=False), fps)
    # false + motion whose id differs from the calibration: refuse
    mo = static_motion(n, fps)
    with pytest.raises(ConfigError, match="recalibrate"):
        detect_shots(tracks, replace(calib, camera_static=False, motion_id="other"), fps, motion=mo)
    with pytest.raises(ConfigError, match="recalibrate"):
        detect_shots(tracks, replace(calib, camera_static=False, motion_id=None), fps, motion=mo)
    # true + measured-moving motion: warn loudly and cap every confidence at 0.3
    moving = dataclasses.replace(mo, is_static=False, meta={**mo.meta, "is_static": False})
    res = detect_shots(tracks, calib, fps, motion=moving)
    assert res.shots and res.debug["warnings"]
    for s in res.shots:
        assert "camera_moved" in s.notes and s.result_conf <= 0.3 and s.points_conf <= 0.3
    # false + matching id: the stabilised path runs, extra columns appear, and (identity view) rows agree
    ok = replace(calib, camera_static=False, motion_id=mo.motion_id)
    stab = detect_shots(tracks, ok, fps, motion=mo)
    plain = detect_shots(tracks, calib, fps)
    assert [s.result for s in stab.shots] == [s.result for s in plain.shots]
    assert stab.debug["coordinate_space"] == "view" and stab.debug["n_views"] == 1
    out = tmp_path / "s.csv"
    write_shots_csv(stab, out)
    header = out.read_text().splitlines()[0].split(",")
    assert header[-2:] == ["view_id", "camera_quality"]


def test_unreliable_frames_make_shots_unknown(tmp_path):
    """A run of unreliable frames longer than max_gap_s across the shot -> unknown, result_conf 0."""
    fps = 30.0
    tracks = ts.build_scene(fps, shooter_ft=(-21.0, 8.0))
    n = int(tracks["frame"].max()) + 1
    calib = ts.write_calib(tmp_path)
    mo = static_motion(n, fps)
    from dataclasses import replace
    ok = replace(calib, camera_static=False, motion_id=mo.motion_id)
    base = detect_shots(tracks, ok, fps, motion=mo).shots[0]
    assert base.result in ("make", "miss")
    lo, hi = base.release_frame, base.frame
    # degraded (0.6) over the span: still scored, confidence capped
    q = mo.quality.copy()
    q[lo:hi + 1] = 0.6
    deg = dataclasses.replace(mo, quality=q)
    s = detect_shots(tracks, ok, fps, motion=deg).shots[0]
    assert s.result == base.result and s.result_conf <= 0.6 and "camera_degraded" in s.notes
    # an unreliable run well over max_gap_s covering the span (quality 0; the pose is kept so the engine still
    # sees the flight and it is the camera rule, not a lost ball, that makes the shot unknown)
    q2 = mo.quality.copy()
    q2[lo:hi + 1] = 0.0
    bad = dataclasses.replace(mo, quality=q2, _cache={})
    res = detect_shots(tracks, ok, fps, motion=bad)
    assert res.shots
    for s in res.shots:
        assert s.result == "unknown" and s.result_conf == 0.0 and "camera_unreliable" in s.notes
    assert res.debug["n_unreliable_runs_over_gap"] >= 1


@pytest.mark.parametrize("fps", FPS_LIST)
def test_panning_camera_scores_like_the_static_scene(fps, tmp_path):
    """The same shot filmed by a panning camera (raw pixels shifted by a different amount every frame) scores
    identically once tracks and calibration go through the stabilised path."""
    scene = ts.build_scene(fps, shooter_ft=(-21.0, 8.0))
    n = int(scene["frame"].max()) + 1
    shift = np.array([-300.0 + 4.0 * np.arange(n), 0.3 * np.arange(n)]).T  # raw = scene - shift
    H_ref = np.tile(np.eye(3), (n, 1, 1))
    H_ref[:, 0, 2], H_ref[:, 1, 2] = shift[:, 0], shift[:, 1]  # H_frame_to_ref: ref = raw + shift
    raw = scene.copy()
    f = raw["frame"].to_numpy()
    for c in ("x1_px", "x2_px"):
        raw[c] = raw[c] - shift[f, 0]
    for c in ("y1_px", "y2_px"):
        raw[c] = raw[c] - shift[f, 1]
    assert np.abs(raw["x1_px"] - scene["x1_px"]).max() > 100  # the raw positions really moved
    vH = np.eye(3)[None]
    mid = _motion_id(H_ref, vH)
    mo = MotionTrack({"frame_width": 1920, "frame_height": 1080, "frame_max_stretch": 3.0, "is_static": False,
                      "motion_id": mid, "max_closure_err_px": 0.0, "n_frames": n, "fps": fps},
                     H_ref, np.full(n, 0.9, np.float32), {"key_frame_idx": np.array([0]), "key_view": np.array([0])},
                     (View(0, 0, np.eye(3), 4000, 2000),), False, mid)
    d = ts.calib_dict()
    d.update(camera_static=False, motion_id=mid)
    path = tmp_path / "court.json"
    path.write_text(json.dumps(d))
    calib = load_calibration(path, motion=mo)
    assert calib.motion_id == mid and not calib.camera_static
    expected = detect_shots(scene, ts.write_calib(tmp_path), fps)
    got = detect_shots(raw, calib, fps, motion=mo)
    assert len(expected.shots) == 1 and len(got.shots) == 1
    e, g = expected.shots[0], got.shots[0]
    assert (g.result, g.points, g.shooter_track_id, g.frame) == (e.result, e.points, e.shooter_track_id, e.frame)
    assert g.feet_x_ft == pytest.approx(e.feet_x_ft, abs=1e-6) and g.feet_y_ft == pytest.approx(e.feet_y_ft, abs=1e-6)
    assert g.view_id == 0 and g.camera_quality == pytest.approx(0.9)
    # the SAME raw tracks scored as if the camera were static is wrong (this is what the feature fixes)
    wrong = detect_shots(raw, ts.write_calib(tmp_path), fps)
    assert [s.result for s in wrong.shots] != [e.result] or not wrong.shots or wrong.shots[0].feet_x_ft != e.feet_x_ft


def test_load_calibration_rejects_stale_or_stretched_landmarks(tmp_path):
    fps = 30.0
    n = 5
    H_ref = np.tile(np.eye(3), (n, 1, 1))
    mid = _motion_id(H_ref, np.eye(3)[None])
    mo = MotionTrack({"frame_width": 1920, "frame_height": 1080, "frame_max_stretch": 3.0, "is_static": True,
                      "motion_id": mid, "n_frames": n, "fps": fps, "settings": {"calib_max_stretch": 4.0}},
                     H_ref, np.full(n, 0.9, np.float32), {}, (View(0, 0, np.eye(3), 1920, 1080),), True, mid)
    d = ts.calib_dict()
    d.update(camera_static=False, motion_id="stale")
    path = tmp_path / "c.json"
    path.write_text(json.dumps(d))
    with pytest.raises(ConfigError, match="recalibrate"):
        load_calibration(path, motion=mo)
    d["motion_id"] = mid
    d["homography_points"][0]["view"] = 3  # no such view
    path.write_text(json.dumps(d))
    with pytest.raises(ConfigError, match="view 3"):
        load_calibration(path, motion=mo)


# --------------------------------------------------------------------------- scripts

import importlib.util  # noqa: E402
import shutil  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_script(name: str):
    spec = importlib.util.spec_from_file_location(name, REPO_ROOT / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture()
def synthetic_video(tmp_path):
    """A 40-frame panning mp4 and a matching (empty-box) detection cache, both under tmp_path."""
    fps = 23.951
    poses = [pose(-15 + 0.75 * i) for i in range(40)]
    vid = tmp_path / "pan.mp4"
    wr = cv2.VideoWriter(str(vid), cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))
    for P in poses:
        wr.write(render(P))
    wr.release()
    det = tmp_path / "pan_det.npz"
    meta = {"video_stem": "pan", "n_frames": 40, "fps": float(cv2.VideoCapture(str(vid)).get(cv2.CAP_PROP_FPS)),
            "threshold": 0.25}
    np.savez_compressed(det, frame_idx=np.zeros(0, np.int64), xyxy=np.zeros((0, 4), np.float32),
                        confidence=np.zeros(0, np.float32), class_id=np.zeros(0, np.int64), meta=np.array(json.dumps(meta)))
    return vid, det, poses


def test_stabilize_script_refuses_bad_folders_and_writes_cache(synthetic_video, tmp_path):
    stab = _load_script("stabilize")
    vid, det, poses = synthetic_video
    out = tmp_path / "pan_motion.npz"
    # image/video outputs outside data/frames/ or runs/ are refused before any work happens
    assert stab.main(["--video", str(vid), "--detections", str(det), "--out", str(out), "--mosaic", str(tmp_path / "m.jpg")]) == 2
    assert stab.main(["--video", str(vid), "--detections", str(det), "--out", str(out), "--debug-video", str(tmp_path / "d.mp4")]) == 2
    assert not out.exists() and not (tmp_path / "m.jpg").exists()
    scratch = REPO_ROOT / "runs" / "pytest_motion_scratch"
    try:
        mosaic = scratch / "pan_mosaic.jpg"
        assert stab.main(["--video", str(vid), "--detections", str(det), "--out", str(out), "--mosaic", str(mosaic),
                          "--debug-video", str(scratch / "dbg.mp4")]) == 0
        assert out.exists() and mosaic.exists() and (scratch / "dbg.mp4").exists()
        m = load_motion(out)
        assert m.n_frames == 40 and (m.quality > 0).all() and not m.is_static
        assert m.meta["fingerprint"]["fps"] == pytest.approx(23.951, abs=0.01)
        img = cv2.imread(str(mosaic))
        assert (img.shape[1], img.shape[0]) == (m.views[0].canvas_w, m.views[0].canvas_h)
        assert img.shape[1] > W  # the mosaic is wider than one frame
        assert np.count_nonzero(img.sum(axis=2)) > 0.5 * img.shape[0] * img.shape[1] * 0.4
        # second run reuses the cache
        assert stab.main(["--video", str(vid), "--detections", str(det), "--out", str(out)]) == 0
        # calibrate on the mosaic (non-interactive), then it round-trips through load_calibration
        cal = _load_script("calibrate_court")
        court = tmp_path / "court.json"
        import sys as _sys
        argv = ["calibrate_court.py", "--mosaic", str(mosaic), "--motion", str(out), "--out", str(court),
                "--rim", "A:100,100,140,115", "--landmark", "50,50:-25,0", "--landmark", "400,60:25,0",
                "--landmark", "60,300:-25,40", "--landmark", "420,310:25,40"]
        old = _sys.argv
        _sys.argv = argv
        try:
            cal.main()
        finally:
            _sys.argv = old
        d = json.loads(court.read_text())
        assert d["camera_static"] is False and d["motion_id"] == m.motion_id and d["rims"][0]["view"] == 0
        c = load_calibration(court, motion=m)
        assert c.homography is not None and c.rims[0].view == 0
        # a calibration from a different motion file is refused
        d["motion_id"] = "other"
        court.write_text(json.dumps(d))
        with pytest.raises(ConfigError, match="recalibrate"):
            load_calibration(court, motion=m)
    finally:
        shutil.rmtree(scratch, ignore_errors=True)


def test_stabilize_rejects_mismatched_detection_cache(synthetic_video, tmp_path):
    stab = _load_script("stabilize")
    vid, det, _ = synthetic_video
    z = dict(np.load(det, allow_pickle=False))
    meta = json.loads(str(z["meta"]))
    meta["fps"] = 60.0
    z["meta"] = np.array(json.dumps(meta))
    bad = tmp_path / "bad_det.npz"
    np.savez_compressed(bad, **z)
    assert stab.main(["--video", str(vid), "--detections", str(bad), "--out", str(tmp_path / "o.npz")]) == 2


def test_mosaic_json_merge_keeps_one_rim_per_id():
    cal = _load_script("calibrate_court")
    b0 = cal.MosaicBuilder(1000, 500, 0)
    b0.add_rim("A", 10, 10, 50, 30)
    b0.add_landmark(20, 20, -25, 0)
    d0 = cal.merge_mosaic_json(None, b0, video_stem="v", frame_w=640, frame_h=360, motion_id="m1", level="nfhs",
                               court_length_ft=None)
    b1 = cal.MosaicBuilder(800, 500, 1)
    b1.add_rim("B", 10, 10, 50, 30)
    b1.add_rim("A", 100, 100, 140, 120)  # re-clicking A on another view replaces the old one
    d1 = cal.merge_mosaic_json(d0, b1, video_stem="v", frame_w=640, frame_h=360, motion_id="m1", level="nfhs",
                               court_length_ft=None)
    assert sorted((r["id"], r["view"]) for r in d1["rims"]) == [("A", 1), ("B", 1)]
    assert len(d1["homography_points"]) == 1  # view 0 landmark kept
    d2 = cal.merge_mosaic_json(d1, b1, video_stem="v", frame_w=640, frame_h=360, motion_id="m2", level="nfhs",
                               court_length_ft=None)  # new motion file: old entries are dropped
    assert d2["homography_points"] == [] and d2["motion_id"] == "m2"
    with pytest.raises(ConfigError):
        b1.add_rim("A", 700, 10, 900, 30)  # off the 800 px canvas


# --------------------------------------------------------------------------- reviewer fixes: rescue edges, recovery, fps

def blur_ramp_frames(sigma_max: float):
    """A steady pan where the blur grows frame by frame: neighbours still match each other, but a sharp
    keyframe stops matching, so pass 1 must go into rescue mode and chain."""
    pans = [-14 + 1.6 * i for i in range(36)]
    poses = [pose(p) for p in pans]
    frames = []
    for i, P in enumerate(poses):
        f = render(P)
        sg = min(sigma_max, 0.5 * i)
        frames.append(cv2.GaussianBlur(f, (0, 0), sg) if sg > 0.3 else f)
    return poses, frames


def test_rescue_born_keyframe_is_chained_and_capped_at_0_6():
    poses, frames = blur_ramp_frames(7.0)
    cfg = dataclasses.replace(CFG, key_min_feat_frac=0.0)  # let a blurred frame become a keyframe
    m = run(frames, 30.0, cfg=cfg)
    chain = m.stats["key_chain_len"]
    assert len(chain) >= 2 and chain[1] >= 1 and chain[0] == 0, "the second keyframe must hang on a chained edge"
    on_chain_key = np.flatnonzero(m.stats["keyframe"] == 1)
    assert len(on_chain_key) > 3
    assert (m.quality[on_chain_key] <= 0.6).all()          # matched to a chain-born keyframe: never 0.9
    assert m.quality[int(m.stats["key_frame_idx"][1])] <= 0.6  # the keyframe's own frame too
    good = [i for i in range(36) if m.quality[i] > 0]
    assert len(good) >= 30
    # bounded error (blurred features are less exact, so the bound is looser than for sharp frames)
    assert max(rel_err(m, poses, j, 18) for j in good if 8 <= j <= 28) <= 8.0
    assert any("chained edge" in w for w in m.meta["warnings"])
    # the chain limit is expressed in seconds (converted with the video's fps), never in frames
    names = {f.name for f in dataclasses.fields(MotionConfig)}
    assert "pass1_chain_limit_s" in names and "pass1_chain_limit" not in names
    assert chain[1] <= max(1, round(cfg.pass1_chain_limit_s * 30.0))


def test_pass2_recovers_on_first_textured_frame_after_a_jump_between_keyframes():
    """A textureless run during which the true pose moves to another keyframe's region: the last good pose is
    stale, so recovery must try every keyframe, not only the ones that overlap the stale pose."""
    fwd = [-28 + 2.0 * i for i in range(29)]                # -28 .. +28: the two ends share no view (FOV is 56 degrees)
    back = [-28 + 1.0 * i for i in range(6)]               # after the gap the camera is back at the start
    poses = [pose(p) for p in fwd] + [pose(fwd[-1])] * 30 + [pose(p) for p in back]
    frames = [render(P) for P in poses[:29]] + [np.full((H, W, 3), 128, np.uint8)] * 30 + [render(pose(p)) for p in back]
    m = run(frames, 30.0)
    assert len(m.stats["key_frame_idx"]) >= 3
    assert (m.quality[29:59] == 0).all()
    first = 59
    assert m.quality[first] > 0, "no recovery on the first textured frame"
    assert (m.quality[first:] > 0).all()
    # the recovered pose is right: frame 31 shows the same view as frame 0
    assert rel_err(m, poses, first, 0) < 3.0


def test_detect_shots_rejects_motion_with_wrong_fps_or_too_few_frames(tmp_path):
    fps = 30.0
    tracks = ts.build_scene(fps, shooter_ft=(-21.0, 8.0))
    n = int(tracks["frame"].max()) + 1
    calib = ts.write_calib(tmp_path)
    mo = static_motion(n, fps)
    assert detect_shots(tracks, calib, fps, motion=mo).shots  # matching motion is fine
    with pytest.raises(ConfigError, match="fps"):
        detect_shots(tracks, calib, 23.951, motion=mo)
    short = static_motion(n - 5, fps)
    with pytest.raises(ConfigError, match="frames"):
        detect_shots(tracks, calib, fps, motion=short)

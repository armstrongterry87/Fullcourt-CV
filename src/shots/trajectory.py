"""Ball trajectory cleaning and gap filling (design 004 section 4.2). numpy only.

Input is the raw ball detections (pixel space, several per frame allowed). Output is one
`Trajectory`: at most one point per frame, split into segments. Positions are the raw
detection centres (never smoothed). Velocities come from a 3-frame moving average and are
used only for their sign and size.

Frame indices are 0-based and every duration is converted with the video's fps.
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass

import numpy as np

from src.shots.config import ShotConfig


@dataclass
class Trajectory:
    """Cleaned ball track. All arrays have one entry per trajectory frame, sorted by frame."""

    fps: float
    frame: np.ndarray    # int, 0-based
    x_px: np.ndarray     # ball box centre
    y_px: np.ndarray     # y grows downward
    w_px: np.ndarray     # ball box width
    conf: np.ndarray     # detector confidence, 0 for interpolated points
    interp: np.ndarray   # bool, True for gap-filled points
    seg: np.ndarray      # int segment id
    vx: np.ndarray       # px per second, from the smoothed track
    vy: np.ndarray       # px per second; negative means moving up the image

    def __len__(self) -> int:
        return len(self.frame)


def _empty(fps: float) -> Trajectory:
    z = np.zeros(0)
    return Trajectory(fps, z.astype(int), z, z, z, z, z.astype(bool), z.astype(int), z, z)


def _dist(a: tuple, b: tuple) -> float:
    return math.hypot(a[1] - b[1], a[2] - b[2])


# a point is (frame, x, y, conf, w)
def clean_ball_track(frame: np.ndarray, x_px: np.ndarray, y_px: np.ndarray, conf: np.ndarray,
                     w_px: np.ndarray, fps: float, rim_w_px: float,
                     cfg: ShotConfig | None = None) -> Trajectory:
    """Clean raw ball detections into a `Trajectory`.

    Steps: speed gate, one candidate per frame, 3-detection seeding, re-anchoring, ghost list,
    gap fill (<= max_gap_s), and a velocity-only 3-frame smooth. `rim_w_px` is the rim box
    width W that scales every pixel threshold.
    """
    cfg = cfg or ShotConfig()
    frame = np.asarray(frame, dtype=int)
    if len(frame) == 0:
        return _empty(fps)
    by_frame: dict[int, list[tuple]] = {}
    for f, x, y, c, w in zip(frame, x_px, y_px, conf, w_px):
        by_frame.setdefault(int(f), []).append((int(f), float(x), float(y), float(c), float(w)))

    gap_n = max(1, round(cfg.max_gap_s * fps))
    W = float(rim_w_px)

    def gate(df: int) -> float:
        return cfg.max_speed_w_per_s * W * df / fps

    def passes(a: tuple, b: tuple) -> bool:
        df = b[0] - a[0]
        return 0 < df <= gap_n + 1 and _dist(a, b) <= gate(df)

    segments: list[list[tuple]] = []
    cur: list[tuple] = []
    seek_buf: list[tuple[int, list[tuple]]] = []
    rej: list[tuple] = []
    ghosts: list[list[float]] = []  # [x, y, hits]

    def near_ghost(p: tuple) -> bool:
        return any(g[2] >= cfg.ghost_hits and math.hypot(p[1] - g[0], p[2] - g[1]) <= cfg.ghost_drop_w * W
                   for g in ghosts)

    def record_rejected(p: tuple) -> None:
        for g in ghosts:
            if math.hypot(p[1] - g[0], p[2] - g[1]) <= cfg.ghost_hit_w * W:
                g[2] += 1
                return
        ghosts.append([p[1], p[2], 1])

    def try_seed() -> list[tuple] | None:
        if len(seek_buf) < cfg.seed_len:
            return None
        tail = seek_buf[-cfg.seed_len:]
        best, best_key = None, None
        for combo in itertools.product(*[c for _, c in tail]):
            if all(passes(combo[i], combo[i + 1]) for i in range(len(combo) - 1)):
                # higher total confidence first (a still ghost has zero step cost, so cost alone would pick it)
                key = (-sum(c[3] for c in combo), sum(_dist(combo[i], combo[i + 1]) for i in range(len(combo) - 1)))
                if best_key is None or key < best_key:
                    best, best_key = list(combo), key
        return best

    for f in sorted(by_frame):
        cs = by_frame[f]
        if cur and f - cur[-1][0] - 1 > gap_n:  # gap too long: the segment ends, nothing is invented
            segments.append(cur)
            cur, rej, seek_buf = [], [], []
        if cur:
            last = cur[-1]
            df = f - last[0]
            if len(cur) >= 2:
                prev = cur[-2]
                k = df / (last[0] - prev[0])
                pred = (last[1] + (last[1] - prev[1]) * k, last[2] + (last[2] - prev[2]) * k)
            else:
                pred = (last[1], last[2])
            ok = [c for c in cs if _dist(last, c) <= gate(df)]
            if ok:
                best = min(ok, key=lambda c: (round(math.hypot(c[1] - pred[0], c[2] - pred[1]), 6), -c[3]))
                cur.append(best)
                rej = []
                continue
            kept = [c for c in cs if not near_ghost(c)]
            for c in kept:
                record_rejected(c)
            if kept:
                rej.append(max(kept, key=lambda c: c[3]))
                rej = rej[-cfg.reanchor_len:]
                if len(rej) == cfg.reanchor_len and all(passes(rej[i], rej[i + 1]) for i in range(len(rej) - 1)):
                    segments.append(cur)  # re-anchor: the chain switches to the consistent detections
                    cur, rej, seek_buf = list(rej), [], []
            continue
        kept = [c for c in cs if not near_ghost(c)]
        if not kept:
            continue
        if seek_buf and f - seek_buf[-1][0] > gap_n + 1:
            seek_buf = []
        seek_buf.append((f, kept))
        seek_buf = seek_buf[-(cfg.seed_len + 3):]
        seed = try_seed()
        if seed is not None:
            cur, seek_buf = seed, []
    if cur:
        segments.append(cur)
    return _build(segments, fps)


def _build(segments: list[list[tuple]], fps: float) -> Trajectory:
    """Fill gaps by linear interpolation and compute smoothed velocities per segment."""
    cols: dict[str, list] = {k: [] for k in ("frame", "x", "y", "w", "conf", "interp", "seg", "vx", "vy")}
    for sid, pts in enumerate(segments):
        fr, xs, ys, ws, cf, ip = [], [], [], [], [], []
        for i, p in enumerate(pts):
            if i > 0:
                q = pts[i - 1]
                for g in range(q[0] + 1, p[0]):
                    t = (g - q[0]) / (p[0] - q[0])
                    fr.append(g)
                    xs.append(q[1] + t * (p[1] - q[1]))
                    ys.append(q[2] + t * (p[2] - q[2]))
                    ws.append(q[4] + t * (p[4] - q[4]))
                    cf.append(0.0)
                    ip.append(True)
            fr.append(p[0]); xs.append(p[1]); ys.append(p[2]); ws.append(p[4]); cf.append(p[3]); ip.append(False)
        xa, ya = np.array(xs), np.array(ys)
        vx, vy = _velocity(np.array(fr), xa, ya, fps)
        for k, v in (("frame", fr), ("x", xs), ("y", ys), ("w", ws), ("conf", cf), ("interp", ip),
                     ("seg", [sid] * len(fr)), ("vx", vx), ("vy", vy)):
            cols[k].extend(list(v))
    if not cols["frame"]:
        return _empty(fps)
    return Trajectory(fps, np.array(cols["frame"], dtype=int), np.array(cols["x"]), np.array(cols["y"]),
                      np.array(cols["w"]), np.array(cols["conf"]), np.array(cols["interp"], dtype=bool),
                      np.array(cols["seg"], dtype=int), np.array(cols["vx"]), np.array(cols["vy"]))


def _smooth3(a: np.ndarray) -> np.ndarray:
    """Centred 3-frame moving average; the two ends use the neighbours they have."""
    if len(a) < 3:
        return a.copy()
    out = a.copy()
    out[1:-1] = (a[:-2] + a[1:-1] + a[2:]) / 3.0
    out[0] = (a[0] + a[1]) / 2.0
    out[-1] = (a[-1] + a[-2]) / 2.0
    return out


def _velocity(frame: np.ndarray, x: np.ndarray, y: np.ndarray, fps: float) -> tuple[np.ndarray, np.ndarray]:
    """Velocity in px/s from the smoothed positions (contiguous frames within a segment)."""
    if len(frame) < 2:
        return np.zeros(len(frame)), np.zeros(len(frame))
    t = frame / fps
    return np.gradient(_smooth3(x), t), np.gradient(_smooth3(y), t)

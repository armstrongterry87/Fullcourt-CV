"""Camera motion compensation: a "virtual static camera" (design docs/design/005-motion-compensation.md).

Every frame gets a homography into one fixed reference plane, estimated from masked SIFT
features matched to a few keyframes. Pixel work then happens in one or more "views" (a
keyframe's image plane extended to a canvas), so the static-camera shot engine can run unchanged.

Naming (design section 3): `H_a_to_b` maps a point in `a` to `b`, `p_b ~ H_a_to_b @ p_a`.
The rightmost matrix of a product is applied first: `H_a_to_c = H_b_to_c @ H_a_to_b`.
Spaces: `_px` raw frame pixels, `_ref` reference pixels (the anchor keyframe's plane, unbounded),
`_view` canvas pixels of one view, `_ft` court feet. Frame indices are 0-based and paired with fps.
Every stored matrix has unit Frobenius norm and a positive determinant (see `_normalize`).

Nothing here writes an image; `render_mosaic` only returns an array and the scripts decide where
it goes (see `ensure_private_output`).
"""

from __future__ import annotations

import hashlib
import json
import math
import warnings
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable, Iterator, Sequence

import cv2
import numpy as np
import pandas as pd

ALGO_VERSION = 3  # bump by hand whenever the algorithm changes (part of the cache key)
REPO_ROOT = Path(__file__).resolve().parents[2]
CACHE_DIR = REPO_ROOT / "runs" / "cache"
PRIVATE_OUTPUT_DIRS = (REPO_ROOT / "data" / "frames", REPO_ROOT / "runs")
FLAG_ZOOM = 1  # `flags` bit: zoom_suspected


@dataclass(frozen=True)
class MotionConfig:
    """Every tuning parameter (marked T in design 005). None was fitted to data."""

    # masking (5.1)
    mask_dilate_frac: float = 0.15
    mask_dilate_px: float = 8.0
    border_px: int = 8
    ignore_rects_px: tuple[tuple[int, int, int, int], ...] = ()  # burned-in graphics, (x1, y1, x2, y2)
    # features (5.2)
    sift_contrast: float = 0.03
    sift_edge: float = 10.0
    sift_sigma: float = 1.6
    grid_cols: int = 8
    grid_rows: int = 5
    per_cell: int = 60
    work_width: int = 1280
    # matching and RANSAC (5.3, 5.4)
    ratio: float = 0.75
    min_matches: int = 20
    ransac_thr_px: float = 2.0
    ransac_iters: int = 2000
    ransac_conf: float = 0.995
    # keyframes (6.1)
    pass1_hz: float = 5.0
    min_key_features: int = 300
    min_overlap: float = 0.6
    key_max_masked_frac: float = 0.5
    key_extra_links: int = 2
    key_link_min_overlap: float = 0.3
    key_min_feat_frac: float = 0.6   # a new keyframe needs this share of the first keyframe's features (sharp frame)
    pass1_chain_limit_s: float = 12.0  # longest chain, in SECONDS (converted with the video's fps), in rescue mode
    # views and mosaic (6.3)
    view_max_stretch: float = 1.5
    frame_max_stretch: float = 3.0
    canvas_max_w: int = 6000
    canvas_max_h: int = 2500
    mosaic_max_frames: int = 25
    # pass 2 (6.4)
    chain_limit: int = 10
    # quality (7)
    min_inliers: int = 40
    min_inlier_ratio: float = 0.25
    max_rms_px: float = 1.5
    min_spread_cells: int = 5
    good_inliers: int = 100
    good_inlier_ratio: float = 0.5
    good_rms_px: float = 1.0
    good_spread_cells: int = 7
    max_area_change: float = 4.0
    jump_frac_per_30hz: float = 0.12
    jump_skip_w: float = 2.0         # allowed jump this many frame widths or more: the check cannot tell anything
    zoom_tol: float = 0.05
    zoom_move_frac: float = 0.05
    # drift and static (6.5, 6.6)
    closure_warn_px: float = 5.0
    static_px: float = 3.0
    # calibration on a mosaic (9.2)
    calib_max_stretch: float = 4.0


@dataclass(frozen=True)
class Features:
    pts_px: np.ndarray  # (N, 2) float32, native pixels
    desc: np.ndarray    # (N, 128) float32


@dataclass(frozen=True)
class PairFit:
    H_src_to_dst: np.ndarray | None
    n_matches: int
    n_inliers: int
    inlier_ratio: float
    reproj_rms_px: float  # in working-size pixels (frames wider than work_width are downscaled)
    spread_cells: int


@dataclass(frozen=True)
class View:
    id: int
    keyframe: int
    H_ref_to_view: np.ndarray
    canvas_w: int
    canvas_h: int


@dataclass(frozen=True)
class MotionTrack:
    """Per-frame homographies into the reference, quality, and the views."""

    meta: dict
    H_frame_to_ref: np.ndarray   # (n, 3, 3), NaN where unreliable
    quality: np.ndarray          # (n,) 0.0 / 0.6 / 0.9
    stats: dict[str, np.ndarray]
    views: tuple[View, ...]
    is_static: bool
    motion_id: str
    _cache: dict = field(default_factory=dict, repr=False, compare=False)

    @property
    def n_frames(self) -> int:
        return len(self.quality)

    @property
    def frame_wh(self) -> tuple[int, int]:
        return int(self.meta["frame_width"]), int(self.meta["frame_height"])

    def _usable_array(self, view: int) -> np.ndarray:
        if view not in self._cache:
            w, h = self.frame_wh
            ok = np.zeros(self.n_frames, dtype=bool)
            V = self.views[view].H_ref_to_view
            for f in range(self.n_frames):
                H = self.H_frame_to_ref[f]
                if np.all(np.isfinite(H)):
                    ok[f] = stretch_at(V @ H, (w / 2, h / 2)) <= self.meta["frame_max_stretch"]
            self._cache[view] = ok
        return self._cache[view]

    def H_frame_to_view(self, frame: int, view: int = 0) -> np.ndarray | None:
        """`H_frame_to_view = H_ref_to_view @ H_frame_to_ref`, or None when the frame is unreliable."""
        if not (0 <= frame < self.n_frames) or not np.all(np.isfinite(self.H_frame_to_ref[frame])):
            return None
        return self.views[view].H_ref_to_view @ self.H_frame_to_ref[frame]

    def usable(self, frame: int, view: int = 0) -> bool:
        """True if the frame is reliable and not too stretched in this view."""
        return 0 <= frame < self.n_frames and bool(self._usable_array(view)[frame])

    def view_quality(self, view: int = 0) -> np.ndarray:
        """Quality per frame as seen by this view: 0.0 where the frame is not usable in it."""
        return np.where(self._usable_array(view), self.quality, 0.0).astype(np.float32)

    def to_view(self, pts_px: np.ndarray, frames: np.ndarray, view: int = 0) -> np.ndarray:
        """Raw pixels (N, 2) plus frame indices (N,) -> view pixels (N, 2); NaN where the frame is not usable."""
        pts = np.asarray(pts_px, dtype=np.float64).reshape(-1, 2)
        frames = np.asarray(frames, dtype=np.int64).reshape(-1)
        out = np.full(pts.shape, np.nan)
        for f in np.unique(frames):
            if not self.usable(int(f), view):
                continue
            sel = frames == f
            out[sel] = transform_points(self.H_frame_to_view(int(f), view), pts[sel])
        return out


# --------------------------------------------------------------------------- geometry helpers

def _normalize(H: np.ndarray) -> np.ndarray:
    """Scale to unit Frobenius norm with a positive determinant (a canonical sign)."""
    H = np.asarray(H, dtype=np.float64)
    if not np.all(np.isfinite(H)):
        return H
    H = H / np.linalg.norm(H)
    return -H if np.linalg.det(H) < 0 else H


def transform_points(H_a_to_b: np.ndarray, pts_a: np.ndarray) -> np.ndarray:
    """Map points (N, 2) from space a to space b. NaN for a non-finite H or points that map behind the plane."""
    p = np.asarray(pts_a, dtype=np.float64).reshape(-1, 2)
    H = np.asarray(H_a_to_b, dtype=np.float64)
    if not np.all(np.isfinite(H)):
        return np.full(p.shape, np.nan)
    q = np.column_stack([p, np.ones(len(p))]) @ H.T
    w = q[:, 2]
    out = np.full(p.shape, np.nan)
    ok = w > 1e-12
    out[ok] = q[ok, :2] / w[ok, None]
    return out


def stretch_at(H: np.ndarray, pt: tuple[float, float]) -> float:
    """Largest singular value of the 2x2 Jacobian of H at `pt` (design 6.3). inf if the point maps behind the plane."""
    H = np.asarray(H, dtype=np.float64)
    p = H @ np.array([pt[0], pt[1], 1.0])
    w = p[2]
    if not np.isfinite(w) or w <= 1e-12:
        return math.inf
    xp, yp = p[0] / w, p[1] / w
    J = np.array([[H[0, 0] - xp * H[2, 0], H[0, 1] - xp * H[2, 1]],
                  [H[1, 0] - yp * H[2, 0], H[1, 1] - yp * H[2, 1]]]) / w
    return float(np.linalg.svd(J, compute_uv=False)[0])


def _corners(w: float, h: float) -> np.ndarray:
    return np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype=np.float64)


def _is_convex(poly: np.ndarray) -> bool:
    if not np.all(np.isfinite(poly)):
        return False
    n = len(poly)
    s = []
    for i in range(n):
        a, b, c = poly[i], poly[(i + 1) % n], poly[(i + 2) % n]
        s.append((b[0] - a[0]) * (c[1] - b[1]) - (b[1] - a[1]) * (c[0] - b[0]))
    s = np.array(s)
    return bool(np.all(s > 0) or np.all(s < 0))


def _area(poly: np.ndarray) -> float:
    x, y = poly[:, 0], poly[:, 1]
    return float(0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))


def _overlap(H_frame_to_key: np.ndarray, frame_wh: tuple[int, int]) -> float:
    """Fraction of the projected frame outline that lies inside the keyframe's image rectangle."""
    w, h = frame_wh
    poly = transform_points(H_frame_to_key, _corners(w, h))
    if not _is_convex(poly):
        return 0.0
    a = _area(poly)
    if a < 1e-6:
        return 0.0
    poly32 = poly.astype(np.float32)
    rect32 = _corners(w, h).astype(np.float32)
    if cv2.contourArea(poly32, oriented=True) < 0:
        poly32 = poly32[::-1].copy()
    if cv2.contourArea(rect32, oriented=True) < 0:
        rect32 = rect32[::-1].copy()
    inter, _ = cv2.intersectConvexConvex(poly32, rect32)
    return float(inter / a)


def _outline_sane(H_frame_to_other: np.ndarray, frame_wh: tuple[int, int], cfg: MotionConfig) -> bool:
    """Design 7 sanity check: projected outline is convex and its area changed by at most `max_area_change`."""
    w, h = frame_wh
    poly = transform_points(H_frame_to_other, _corners(w, h))
    if not _is_convex(poly):
        return False
    ratio = _area(poly) / (w * h)
    return 1.0 / cfg.max_area_change <= ratio <= cfg.max_area_change


def _jump_ok(H_cur_to_ref: np.ndarray, H_prev_to_ref: np.ndarray, gap_frames: int, fps: float,
             frame_wh: tuple[int, int], cfg: MotionConfig) -> bool:
    """Design 7 jump check, in pixels of the current frame: how far the previous frame's corners move."""
    w, h = frame_wh
    allowed = cfg.jump_frac_per_30hz * w * max(gap_frames, 1) * 30.0 / fps
    if allowed >= cfg.jump_skip_w * w:  # long gap: any pose is plausible (and far corners map behind the plane)
        return True
    try:
        M = np.linalg.inv(H_cur_to_ref) @ H_prev_to_ref  # prev frame px -> current frame px
    except np.linalg.LinAlgError:
        return False
    c = _corners(w, h)
    moved = transform_points(M, c)
    if not np.all(np.isfinite(moved)):
        return False
    d = float(np.max(np.linalg.norm(moved - c, axis=1)))
    return d <= allowed


# --------------------------------------------------------------------------- mask, features, matching

def build_mask(frame_hw: tuple[int, int], boxes_xyxy_px: np.ndarray, cfg: MotionConfig) -> np.ndarray:
    """uint8 mask, 255 = usable background. `boxes_xyxy_px` must be the SAME frame's detections.

    Every box (any class, any confidence) is dilated by `0.15 * max(w, h) + 8` px on each side.
    The frame border and `cfg.ignore_rects_px` are masked too.
    """
    h, w = frame_hw
    m = np.full((h, w), 255, dtype=np.uint8)
    b = int(cfg.border_px)
    if b > 0:
        m[:b, :] = 0
        m[h - b:, :] = 0
        m[:, :b] = 0
        m[:, w - b:] = 0
    for x1, y1, x2, y2 in cfg.ignore_rects_px:
        m[max(0, y1):max(0, y2), max(0, x1):max(0, x2)] = 0
    boxes = np.asarray(boxes_xyxy_px, dtype=np.float64).reshape(-1, 4)
    for x1, y1, x2, y2 in boxes:
        d = cfg.mask_dilate_frac * max(x2 - x1, y2 - y1) + cfg.mask_dilate_px
        xa, ya = max(0, int(math.floor(x1 - d))), max(0, int(math.floor(y1 - d)))
        xb, yb = min(w, int(math.ceil(x2 + d))), min(h, int(math.ceil(y2 + d)))
        if xb > xa and yb > ya:
            m[ya:yb, xa:xb] = 0
    return m


def _empty_features() -> Features:
    return Features(np.zeros((0, 2), np.float32), np.zeros((0, 128), np.float32))


def extract_features(gray: np.ndarray, mask: np.ndarray, cfg: MotionConfig) -> Features:
    """Masked SIFT on a grayscale frame, bucketed per grid cell. Points are returned in native pixels."""
    h, w = gray.shape[:2]
    s = min(1.0, cfg.work_width / w)
    g, m = gray, mask
    if s < 1.0:
        g = cv2.resize(gray, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
        m = cv2.resize(mask, (g.shape[1], g.shape[0]), interpolation=cv2.INTER_NEAREST)
    sift = cv2.SIFT_create(nfeatures=0, contrastThreshold=cfg.sift_contrast,
                           edgeThreshold=cfg.sift_edge, sigma=cfg.sift_sigma)
    kps, desc = sift.detectAndCompute(g, m)
    if desc is None or len(kps) == 0:
        return _empty_features()
    pts = np.array([k.pt for k in kps], dtype=np.float64)
    resp = np.array([k.response for k in kps])
    gh, gw = g.shape[:2]
    cell = (np.minimum(pts[:, 0] / gw * cfg.grid_cols, cfg.grid_cols - 1).astype(int) * cfg.grid_rows
            + np.minimum(pts[:, 1] / gh * cfg.grid_rows, cfg.grid_rows - 1).astype(int))
    keep = []
    for c in np.unique(cell):
        idx = np.flatnonzero(cell == c)
        keep.extend(idx[np.argsort(-resp[idx])[:cfg.per_cell]].tolist())
    keep = np.array(sorted(keep), dtype=int)
    return Features((pts[keep] / s).astype(np.float32), desc[keep].astype(np.float32))


def _no_fit(n_matches: int = 0) -> PairFit:
    return PairFit(None, n_matches, 0, 0.0, math.inf, 0)


def match_homography(src: Features, dst: Features, frame_wh: tuple[int, int], cfg: MotionConfig) -> PairFit:
    """Fit `H_src_to_dst` from ratio-tested, mutually-best SIFT matches with RANSAC (native-pixel matrix)."""
    if len(src.pts_px) < 2 or len(dst.pts_px) < 2:
        return _no_fit()
    bf = cv2.BFMatcher(cv2.NORM_L2)
    fwd = bf.knnMatch(src.desc, dst.desc, k=2)
    back = bf.match(dst.desc, src.desc)
    rev = np.array([m.trainIdx for m in back])  # for each dst feature, its best src feature
    q, t = [], []
    for pair in fwd:
        if len(pair) < 2:
            continue
        a, b = pair
        if a.distance < cfg.ratio * b.distance and rev[a.trainIdx] == a.queryIdx:
            q.append(a.queryIdx)
            t.append(a.trainIdx)
    if len(q) < cfg.min_matches:
        return _no_fit(len(q))
    w, h = frame_wh
    s = min(1.0, cfg.work_width / w)
    a_w = src.pts_px[q].astype(np.float64) * s
    b_w = dst.pts_px[t].astype(np.float64) * s
    cv2.setRNGSeed(0)
    H_w, mask = cv2.findHomography(a_w, b_w, cv2.RANSAC, cfg.ransac_thr_px,
                                   maxIters=cfg.ransac_iters, confidence=cfg.ransac_conf)
    if H_w is None or mask is None or not np.all(np.isfinite(H_w)):
        return _no_fit(len(q))
    inl = mask.ravel().astype(bool)
    n_in = int(inl.sum())
    if n_in < 4:
        return _no_fit(len(q))
    err = np.linalg.norm(transform_points(H_w, a_w[inl]) - b_w[inl], axis=1)
    rms = float(math.sqrt(np.mean(err ** 2))) if np.all(np.isfinite(err)) else math.inf
    S = np.diag([s, s, 1.0])
    H = _normalize(np.linalg.inv(S) @ H_w @ S)
    p = src.pts_px[q][inl]
    cells = {(min(int(x / w * 4), 3), min(int(y / h * 3), 2)) for x, y in p}
    return PairFit(H, len(q), n_in, n_in / len(q), rms, len(cells))


def score_quality(fit: PairFit, chained: int, cfg: MotionConfig) -> float:
    """0.0 unreliable, 0.6 degraded, 0.9 reliable (design 7). `chained` is the chain length (0 = direct to a keyframe)."""
    if (fit.H_src_to_dst is None or fit.n_inliers < cfg.min_inliers or fit.inlier_ratio < cfg.min_inlier_ratio
            or fit.reproj_rms_px > cfg.max_rms_px or fit.spread_cells < cfg.min_spread_cells
            or chained > cfg.chain_limit):
        return 0.0
    if (chained == 0 and fit.n_inliers >= cfg.good_inliers and fit.inlier_ratio >= cfg.good_inlier_ratio
            and fit.reproj_rms_px <= cfg.good_rms_px and fit.spread_cells >= cfg.good_spread_cells):
        return 0.9
    return 0.6


# --------------------------------------------------------------------------- keyframe graph

@dataclass
class _Key:
    idx: int            # frame index
    feat: Features
    masked_frac: float


def _gray(frame: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame


def _frame_features(frame: np.ndarray, boxes: np.ndarray, cfg: MotionConfig) -> tuple[Features, float]:
    g = _gray(frame)
    mask = build_mask(g.shape[:2], boxes, cfg)
    return extract_features(g, mask, cfg), float(np.mean(mask == 0))


def _pass1(frame_source: Callable[[], Iterator[np.ndarray]], boxes: Sequence[np.ndarray], fps: float,
           cfg: MotionConfig, frame_wh: tuple[int, int] | None = None
           ) -> tuple[list[_Key], dict[tuple[int, int], tuple[np.ndarray, int]]]:
    """Build keyframes and the edges between them (design 6.1). edges[(a, b)] = (H_a_to_b, n_inliers), both directions.

    Normally every `fps / 5`-th frame is looked at. When a sample cannot be matched to the current keyframe
    (a fast, blurred pan), the pass switches to "rescue" mode: every frame is chained to the previous localised
    frame (consecutive frames overlap strongly) until a sharp frame can become a new keyframe, or an existing
    keyframe fits again. The chain gives the new keyframe's edge; extra links to other keyframes add loops.
    """
    n = len(boxes)
    step = max(1, round(fps / cfg.pass1_hz))
    keys: list[_Key] = []
    edges: dict[tuple[int, int], tuple[np.ndarray, int]] = {}
    Hk0: list[np.ndarray] = []  # provisional key -> key 0, used only to predict overlaps
    st = {"cur": -1, "prev": None, "rescue": False}  # prev: idx, feat, P (= H_frame_to_key0), chained
    wh = frame_wh
    buf: dict[int, np.ndarray] = {}

    chain_limit_frames = max(1, round(cfg.pass1_chain_limit_s * fps))

    def add_edge(a: int, b: int, H_a_to_b: np.ndarray, n_in: int, chain_len: int = 0) -> None:
        """chain_len = 0 for a direct fit; otherwise how many chained frames the edge was composed from."""
        edges[(a, b)] = (H_a_to_b, n_in, chain_len)
        edges[(b, a)] = (_normalize(np.linalg.inv(H_a_to_b)), n_in, chain_len)

    def good(fit: PairFit) -> bool:
        return score_quality(fit, 0, cfg) > 0 and _outline_sane(fit.H_src_to_dst, wh, cfg)

    def sharp(feat: Features) -> bool:
        # compared with keyframe 0's feature count: key 0 is the first frame with >= min_key_features, taken as
        # the video's "sharp" benchmark (resolution- and texture-dependent, so it is relative on purpose)
        return len(feat.pts_px) >= max(cfg.min_key_features, cfg.key_min_feat_frac * len(keys[0].feat.pts_px))

    def add_key(i: int, feat: Features, masked: float, H_new_to_cur: np.ndarray, n_in: int,
                chain_len: int = 0) -> None:
        """New keyframe. `H_new_to_cur` may come from a chain of frames (chain_len > 0). Every existing keyframe
        predicted to overlap (including the current one) is matched DIRECTLY; a good direct fit is an edge. The
        chained edge to the current keyframe is kept only when no direct fit exists at all, and then it records
        chain_len, so quality along it can be capped."""
        cur = st["cur"]
        new = len(keys)
        keys.append(_Key(i, feat, masked))
        Hk0.append(Hk0[cur] @ H_new_to_cur)  # H_new_to_key0 = H_cur_to_key0 @ H_new_to_cur
        n_direct = 0
        if chain_len == 0:
            add_edge(new, cur, _normalize(H_new_to_cur), n_in)
            n_direct += 1
        linked = [(-_overlap(np.linalg.inv(Hk0[k]) @ Hk0[new], wh), k) for k in range(new)]
        for nov, k in sorted(linked):
            if -nov < cfg.key_link_min_overlap and not (k == cur and chain_len > 0):
                continue
            if chain_len == 0 and k == cur:
                continue
            f3 = match_homography(feat, keys[k].feat, wh, cfg)
            if good(f3):
                add_edge(new, k, f3.H_src_to_dst, f3.n_inliers)
                n_direct += 1
        if n_direct == 0:
            add_edge(new, cur, _normalize(H_new_to_cur), n_in, chain_len)
        st["cur"], st["prev"], st["rescue"] = new, {"idx": i, "feat": feat, "P": Hk0[new], "chained": 0}, False

    def localise(i: int, frame: np.ndarray) -> bool:
        """Place frame i relative to the keyframes. False if it could not be placed at all."""
        feat, masked = _frame_features(frame, boxes[i], cfg)
        cur, prev = st["cur"], st["prev"]
        direct = match_homography(feat, keys[cur].feat, wh, cfg)
        cand = None
        if good(direct):
            if _overlap(direct.H_src_to_dst, wh) >= cfg.min_overlap:
                st["prev"] = {"idx": i, "feat": feat, "P": Hk0[cur] @ direct.H_src_to_dst, "chained": 0}
                st["rescue"] = False
                return True
            cand = direct
        # re-anchor: an existing keyframe predicted to overlap enough (from the last localised pose). When the
        # last localised frame is stale (rescue mode and it was not the previous frame), the prediction means
        # nothing, so every other keyframe is tried, best predicted overlap first.
        stale = st["rescue"] and prev["idx"] != i - 1
        order = []
        for k in range(len(keys)):
            if k != cur:
                ov = _overlap(np.linalg.inv(Hk0[k]) @ prev["P"], wh)
                if ov >= cfg.min_overlap or stale:
                    order.append((-ov, abs(keys[k].idx - i), k))
        for _, _, k in sorted(order):
            f2 = match_homography(feat, keys[k].feat, wh, cfg)
            if good(f2) and _overlap(f2.H_src_to_dst, wh) >= cfg.min_overlap:
                st["cur"] = k
                st["prev"] = {"idx": i, "feat": feat, "P": Hk0[k] @ f2.H_src_to_dst, "chained": 0}
                st["rescue"] = False
                return True
        if cand is not None and score_quality(cand, 0, cfg) >= 0.9 and masked < cfg.key_max_masked_frac:
            add_key(i, feat, masked, cand.H_src_to_dst, cand.n_inliers)
            return True
        # chain from the previous localised frame
        if prev["chained"] >= chain_limit_frames:
            st["rescue"] = True
            return False
        fc = match_homography(feat, prev["feat"], wh, cfg)
        if score_quality(fc, 1, cfg) == 0 or not _outline_sane(fc.H_src_to_dst, wh, cfg):
            st["rescue"] = True
            return False
        P = _normalize(prev["P"] @ fc.H_src_to_dst)  # H_f_to_key0 = H_prev_to_key0 @ H_f_to_prev
        H_to_cur = np.linalg.inv(Hk0[cur]) @ P
        if _overlap(H_to_cur, wh) < cfg.min_overlap and sharp(feat) and masked < cfg.key_max_masked_frac:
            add_key(i, feat, masked, H_to_cur, fc.n_inliers, prev["chained"] + 1)
            return True
        st["prev"] = {"idx": i, "feat": feat, "P": P, "chained": prev["chained"] + 1}
        st["rescue"] = True
        return True

    for i, frame in enumerate(frame_source()):
        if i >= n:
            break
        if wh is None:
            wh = (frame.shape[1], frame.shape[0])
        buf[i] = frame
        buf.pop(i - step - 1, None)
        if not keys:
            if i % step == 0:
                feat, masked = _frame_features(frame, boxes[i], cfg)
                if len(feat.pts_px) >= cfg.min_key_features:
                    keys.append(_Key(i, feat, masked))
                    Hk0.append(np.eye(3))
                    st["cur"], st["prev"] = 0, {"idx": i, "feat": feat, "P": np.eye(3), "chained": 0}
            continue
        if st["rescue"]:
            localise(i, frame)
        elif i % step == 0 and not localise(i, frame):
            # a sample failed: replay the frames since the last localised one, one by one
            for j in range(st["prev"]["idx"] + 1, i):
                if j in buf:
                    localise(j, buf[j])
    return keys, edges


def _reference_graph(K: int, edges: dict) -> dict:
    """Anchor (middle of the graph), BFS tree, H_key_to_ref, hop counts and closure error (6.2, 6.5)."""
    adj: dict[int, list[int]] = {k: [] for k in range(K)}
    for (a, b) in edges:
        adj[a].append(b)

    def bfs(src: int) -> dict[int, int]:
        dist, q = {src: 0}, [src]
        for u in q:
            for v in sorted(adj[u]):
                if v not in dist:
                    dist[v] = dist[u] + 1
                    q.append(v)
        return dist

    ecc = [max(bfs(k).values()) for k in range(K)]
    anchor = min(range(K), key=lambda k: (ecc[k], k))
    dist = bfs(anchor)
    parent = [-1] * K
    H_to_ref = [np.eye(3) for _ in range(K)]
    for u in sorted(dist, key=lambda k: (dist[k], k)):
        if u == anchor:
            continue
        cands = [v for v in adj[u] if dist.get(v) == dist[u] - 1]
        p = max(sorted(cands), key=lambda v: (edges[(u, v)][2] == 0, edges[(u, v)][1]))  # direct edges first
        parent[u] = p
        H_to_ref[u] = _normalize(H_to_ref[p] @ edges[(u, p)][0])  # H_u_to_ref = H_p_to_ref @ H_u_to_p
    chain_len = [0 if parent[k] < 0 else edges[(k, parent[k])][2] for k in range(K)]
    path_chained = [False] * K  # True if the tree path from the keyframe to the anchor uses a chained edge
    for u in sorted(dist, key=lambda k: (dist[k], k)):
        if parent[u] >= 0:
            path_chained[u] = chain_len[u] > 0 or path_chained[parent[u]]
    return {"anchor": anchor, "parent": parent, "hops": [dist[k] for k in range(K)], "H_to_ref": H_to_ref,
            "chain_len": chain_len, "path_chained": path_chained}


def _closure_error(edges: dict, parent: list[int], H_to_ref: list[np.ndarray], frame_wh: tuple[int, int]) -> float:
    """Largest corner disagreement (reference px) between a keyframe's tree path and a non-tree edge path."""
    w, h = frame_wh
    err = 0.0
    for (a, b), (H_ab, _n, _c) in edges.items():
        if a > b or parent[a] == b or parent[b] == a:
            continue
        via_edge = transform_points(H_to_ref[b] @ H_ab, _corners(w, h))
        via_tree = transform_points(H_to_ref[a], _corners(w, h))
        d = np.linalg.norm(via_edge - via_tree, axis=1)
        if np.all(np.isfinite(d)):
            err = max(err, float(d.max()))
    return err


# --------------------------------------------------------------------------- views

def _build_views(H_to_ref: list[np.ndarray], order: list[int], frame_wh: tuple[int, int],
                 cfg: MotionConfig) -> tuple[list[View], list[int], list[str]]:
    """Group keyframes into views whose stretch stays <= `view_max_stretch` (design 6.3)."""
    w, h = frame_wh
    centre = (w / 2, h / 2)
    vkeys: list[int] = []
    member_view: dict[int, int] = {}
    for k in order:
        best = None
        for vi, vk in enumerate(vkeys):
            s = stretch_at(np.linalg.inv(H_to_ref[vk]) @ H_to_ref[k], centre)
            if best is None or s < best[0]:
                best = (s, vi)
        if best is not None and best[0] <= cfg.view_max_stretch:
            member_view[k] = best[1]
        else:
            vkeys.append(k)
            member_view[k] = len(vkeys) - 1
    views, warns = [], []
    for vi, vk in enumerate(vkeys):
        Hv = np.linalg.inv(H_to_ref[vk])  # ref -> view key's own plane
        pts = []
        for k, v in member_view.items():
            if v == vi:
                c = transform_points(Hv @ H_to_ref[k], _corners(w, h))
                pts.append(c[np.all(np.isfinite(c), axis=1)])
        pts = np.vstack(pts)
        x0, y0 = pts.min(axis=0)
        x1, y1 = pts.max(axis=0)
        if x1 - x0 > cfg.canvas_max_w or y1 - y0 > cfg.canvas_max_h:
            warns.append(f"view {vi}: canvas {x1 - x0:.0f}x{y1 - y0:.0f} px exceeds the cap; clipped")
            cx, cy = w / 2, h / 2
            if x1 - x0 > cfg.canvas_max_w:
                x0 = max(x0, cx - cfg.canvas_max_w / 2)
                x1 = min(x1, x0 + cfg.canvas_max_w)
            if y1 - y0 > cfg.canvas_max_h:
                y0 = max(y0, cy - cfg.canvas_max_h / 2)
                y1 = min(y1, y0 + cfg.canvas_max_h)
        x0, y0 = math.floor(x0), math.floor(y0)
        cw, ch = int(math.ceil(x1) - x0), int(math.ceil(y1) - y0)
        T = np.array([[1.0, 0, -x0], [0, 1.0, -y0], [0, 0, 1.0]])
        views.append(View(vi, vk, _normalize(T @ Hv), cw, ch))
    return views, [member_view[k] for k in range(len(H_to_ref))], warns


# --------------------------------------------------------------------------- estimate

def _motion_id(H_frame_to_ref: np.ndarray, view_H: np.ndarray) -> str:
    h = hashlib.sha256()
    h.update(np.ascontiguousarray(H_frame_to_ref, dtype=np.float64).tobytes())
    h.update(np.ascontiguousarray(view_H, dtype=np.float64).tobytes())
    return h.hexdigest()[:16]


def estimate_motion(frame_source: Callable[[], Iterator[np.ndarray]], boxes_per_frame: Sequence[np.ndarray],
                    fps: float, cfg: MotionConfig | None = None, meta: dict | None = None) -> MotionTrack:
    """Estimate `H_frame_to_ref` for every frame (two passes over `frame_source`, each from frame 0).

    `boxes_per_frame[t]` are frame t's cached detections (xyxy, native pixels); they are used only
    to mask frame t. Raises ValueError if the source yields fewer frames than `boxes_per_frame`.
    """
    cfg = cfg or MotionConfig()
    n = len(boxes_per_frame)
    first = next(iter(frame_source()), None)
    if first is None or n == 0:
        raise ValueError("estimate_motion: no frames to process.")
    wh = (first.shape[1], first.shape[0])
    w, h = wh
    keys, edges = _pass1(frame_source, boxes_per_frame, fps, cfg, wh)
    K = len(keys)
    warns: list[str] = []
    if K:
        g = _reference_graph(K, edges)
        H_to_ref, hops, parent, anchor = g["H_to_ref"], g["hops"], g["parent"], g["anchor"]
        key_chain_len, path_chained = g["chain_len"], g["path_chained"]
        if any(path_chained):
            warns.append(f"{sum(path_chained)} keyframe(s) are tied to the anchor through a chained edge; frames "
                         f"matched to them are capped at quality 0.6")
        closure = _closure_error(edges, parent, H_to_ref, wh)
        if closure > cfg.closure_warn_px:
            warns.append(f"closure error {closure:.1f} px exceeds {cfg.closure_warn_px} px")
    else:
        H_to_ref, hops, parent, anchor, closure = [], [], [], -1, 0.0
        key_chain_len, path_chained = [], []
        warns.append("no keyframe found (not enough background features)")
    key_idx = {k.idx: j for j, k in enumerate(keys)}

    # ---- pass 2
    H_all = np.full((n, 3, 3), np.nan)
    quality = np.zeros(n, np.float32)
    st = {"n_matches": np.zeros(n, np.int32), "n_inliers": np.zeros(n, np.int32),
          "inlier_ratio": np.zeros(n, np.float32), "reproj_rms_px": np.zeros(n, np.float32),
          "spread_cells": np.zeros(n, np.int16), "keyframe": np.full(n, -1, np.int32),
          "chained": np.zeros(n, np.int16), "flags": np.zeros(n, np.int16)}
    last = None  # dict(H, feat, idx, chained)
    centre = np.array([w / 2, h / 2])
    seen = 0
    for i, frame in enumerate(frame_source() if K else iter(())):
        if i >= n:
            break
        seen += 1
        if i in key_idx:
            j = key_idx[i]
            feat = keys[j].feat
            H_all[i] = H_to_ref[j]
            quality[i] = 0.6 if path_chained[j] else 0.9
            # NOTE: n_matches / n_inliers / inlier_ratio / spread_cells of a keyframe's own frame are FILLER
            # (the frame is matched to itself), not measurements
            st["keyframe"][i] = j
            st["n_matches"][i] = st["n_inliers"][i] = len(feat.pts_px)
            st["inlier_ratio"][i] = 1.0
            st["spread_cells"][i] = 12
            last = {"H": H_all[i].copy(), "feat": feat, "idx": i, "chained": 0, "q": float(quality[i])}
            continue
        feat, _ = _frame_features(frame, boxes_per_frame[i], cfg)
        # candidates: keyframes ranked by predicted overlap with the last good pose (never a stale guess at start)
        # recovery (last pose stale: not the previous frame, or the chain is exhausted): try EVERY keyframe,
        # best predicted overlap first, then nearest in time. Otherwise only the best two with some overlap.
        recovering = last is None or last["idx"] != i - 1 or last["chained"] >= cfg.chain_limit
        if last is not None:
            ranked = sorted(((-_overlap(np.linalg.inv(H_to_ref[k]) @ last["H"], wh), abs(keys[k].idx - i), k)
                             for k in range(K)))
            cand_keys = [k for ov, _, k in ranked] if recovering else \
                [k for ov, _, k in ranked if -ov >= cfg.key_link_min_overlap][:2]
        else:
            cand_keys = sorted(range(K), key=lambda k: abs(keys[k].idx - i))
        done = False
        for k in cand_keys:
            fit = match_homography(feat, keys[k].feat, wh, cfg)
            q = score_quality(fit, 0, cfg)
            if q == 0 or not _outline_sane(fit.H_src_to_dst, wh, cfg):
                continue
            Hr = _normalize(H_to_ref[k] @ fit.H_src_to_dst)  # H_frame_to_ref = H_key_to_ref @ H_frame_to_key
            if last is not None and not _jump_ok(Hr, last["H"], i - last["idx"], fps, wh, cfg):
                continue
            q = min(q, 0.6) if path_chained[k] else q  # tied to the anchor through a chained keyframe edge
            _store(H_all, quality, st, i, Hr, q, fit, k, 0, centre, wh, cfg)
            last = {"H": Hr, "feat": feat, "idx": i, "chained": 0}
            done = True
            break
        if done or last is None or last["chained"] >= cfg.chain_limit:
            continue
        fit = match_homography(feat, last["feat"], wh, cfg)
        q = score_quality(fit, last["chained"] + 1, cfg)
        if q == 0 or not _outline_sane(fit.H_src_to_dst, wh, cfg):
            continue
        Hr = _normalize(last["H"] @ fit.H_src_to_dst)  # H_t_to_ref = H_prev_to_ref @ H_t_to_prev
        if not _jump_ok(Hr, last["H"], i - last["idx"], fps, wh, cfg):
            continue
        _store(H_all, quality, st, i, Hr, min(q, 0.6), fit, -1, last["chained"] + 1, centre, wh, cfg)
        last = {"H": Hr, "feat": feat, "idx": i, "chained": last["chained"] + 1}
    if K and seen < n:
        raise ValueError(f"estimate_motion: frame source gave {seen} frames but {n} detection rows were given.")

    # ---- static snap (6.6)
    is_static = False
    fitted = np.flatnonzero(np.all(np.isfinite(H_all), axis=(1, 2)))
    if K == 1 and len(fitted):
        c = _corners(w, h)
        disp = [np.max(np.linalg.norm(transform_points(H_all[f], c) - c, axis=1)) for f in fitted]
        if np.percentile(disp, 99) < cfg.static_px:
            is_static = True
            H_all[:] = np.eye(3)
            quality[:] = 0.9
            st["flags"][:] = 0
    n_zoom = int(np.sum((st["flags"] & FLAG_ZOOM) > 0))

    # ---- views
    if K:
        order = sorted(range(K), key=lambda k: (hops[k], k))
        views, key_view, vwarn = _build_views(H_to_ref, order, wh, cfg)
        if is_static:
            views = [View(0, 0, np.eye(3), w, h)]
        warns += vwarn
    else:
        views, key_view = [], []
    view_H = np.array([v.H_ref_to_view for v in views]).reshape(-1, 3, 3)
    stats = dict(st)
    stats.update(
        key_frame_idx=np.array([k.idx for k in keys], np.int32), key_parent=np.array(parent, np.int32),
        key_hops=np.array(hops, np.int32), key_view=np.array(key_view, np.int32),
        key_chain_len=np.array(key_chain_len, np.int32),
        key_H_to_ref=np.array(H_to_ref, np.float64).reshape(-1, 3, 3),
        view_key=np.array([v.keyframe for v in views], np.int32), view_H_ref_to_view=view_H,
        view_canvas_wh=np.array([[v.canvas_w, v.canvas_h] for v in views], np.int32).reshape(-1, 2))
    mid = _motion_id(H_all, view_H)
    m = dict(meta or {})
    m.update(frame_width=w, frame_height=h, fps=float(fps), n_frames=n, frame_max_stretch=cfg.frame_max_stretch,
             is_static=is_static, max_closure_err_px=float(closure), motion_id=mid, n_zoom_suspected=n_zoom,
             n_keyframes=K, anchor_key=int(anchor), warnings=warns, settings=_settings_dict(cfg),
             algo_version=ALGO_VERSION, cv2_version=cv2.__version__)
    return MotionTrack(m, H_all, quality, stats, tuple(views), is_static, mid)


def _store(H_all, quality, st, i, Hr, q, fit, k, chained, centre, wh, cfg) -> None:
    """Record one fitted frame, including the zoom_suspected flag (7)."""
    H_all[i] = Hr
    quality[i] = q
    st["n_matches"][i], st["n_inliers"][i] = fit.n_matches, fit.n_inliers
    st["inlier_ratio"][i], st["reproj_rms_px"][i] = fit.inlier_ratio, fit.reproj_rms_px
    st["spread_cells"][i], st["keyframe"][i], st["chained"][i] = fit.spread_cells, k, chained
    moved = np.linalg.norm(transform_points(fit.H_src_to_dst, centre[None])[0] - centre) / wh[0]
    if abs(stretch_at(fit.H_src_to_dst, tuple(centre)) - 1.0) > cfg.zoom_tol and moved < cfg.zoom_move_frac:
        st["flags"][i] |= FLAG_ZOOM


# --------------------------------------------------------------------------- tracks

_POINT_CLASSES = ("ball", "rim")


def transform_tracks(tracks: pd.DataFrame, motion: MotionTrack, view: int) -> pd.DataFrame:
    """Move tracks CSV boxes into view pixels (design 9.1). Points are transformed, never box corners.

    Players and referees: feet (bottom-centre), head (top-centre) and left/right mid-height give the
    new box (feet exactly preserved). Ball and rim: centre and the four edge mid-points (centre
    preserved). Rows from frames not usable in the view are dropped.
    """
    df = tracks.reset_index(drop=True).copy()
    if df.empty:
        return df
    x1, y1, x2, y2 = (df[c].to_numpy(float) for c in ("x1_px", "y1_px", "x2_px", "y2_px"))
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    frames = df["frame"].to_numpy(np.int64)
    pointlike = df["class"].isin(_POINT_CLASSES).to_numpy()
    # five points per row: 0 anchor (feet or centre), 1 top, 2 bottom-or-feet, 3 left, 4 right
    ptsets = [np.column_stack([cx, np.where(pointlike, cy, y2)]),           # 0: feet / centre
              np.column_stack([cx, y1]),                                    # 1: head / top
              np.column_stack([cx, y2]),                                    # 2: bottom
              np.column_stack([x1, cy]),                                    # 3: left
              np.column_stack([x2, cy])]                                    # 4: right
    T = [motion.to_view(p, frames, view) for p in ptsets]
    a, top, bot, left, right = T
    w_new = np.linalg.norm(right - left, axis=1)
    h_new = np.linalg.norm(bot - top, axis=1)
    nx1, nx2 = a[:, 0] - w_new / 2, a[:, 0] + w_new / 2
    ny1 = np.where(pointlike, a[:, 1] - h_new / 2, np.minimum(top[:, 1], a[:, 1] - 1.0))
    ny2 = np.where(pointlike, a[:, 1] + h_new / 2, a[:, 1])
    ok = np.all(np.isfinite(np.column_stack([nx1, ny1, nx2, ny2])), axis=1)
    df["x1_px"], df["y1_px"], df["x2_px"], df["y2_px"] = nx1, ny1, nx2, ny2
    return df[ok].reset_index(drop=True)


# --------------------------------------------------------------------------- mosaic

def render_mosaic(frame_source: Callable[[], Iterator[np.ndarray]], boxes_per_frame: Sequence[np.ndarray],
                  motion: MotionTrack, view: int, cfg: MotionConfig | None = None) -> np.ndarray:
    """Per-pixel median of keyframes plus evenly sampled quality-0.9 frames warped into `view` (BGR uint8).

    Masked pixels (people, ball, border) are left out of the median; pixels nobody covers are 0.
    The returned array is only ever written by scripts that call `ensure_private_output`.
    """
    cfg = cfg or MotionConfig()
    v = motion.views[view]
    usable = motion._usable_array(view)
    keys = [int(f) for f, kv in zip(motion.stats["key_frame_idx"], motion.stats["key_view"]) if kv == view]
    keys = [f for f in keys if usable[f]][:cfg.mosaic_max_frames]
    extra = np.flatnonzero(usable & (motion.quality >= 0.9))
    extra = [int(f) for f in extra if int(f) not in keys]
    room = cfg.mosaic_max_frames - len(keys)
    if room > 0 and extra:
        extra = [extra[int(j)] for j in np.unique(np.linspace(0, len(extra) - 1, min(room, len(extra))).astype(int))]
    else:
        extra = []
    chosen = set(keys) | set(extra)
    imgs: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
    for i, frame in enumerate(frame_source()):
        if i >= motion.n_frames or (chosen and i > max(chosen)):
            break
        if i in chosen:
            mask = build_mask(frame.shape[:2], boxes_per_frame[i], cfg)
            imgs.append((frame, mask, motion.H_frame_to_view(i, view)))
    out = np.zeros((v.canvas_h, v.canvas_w, 3), np.uint8)
    strip = 64
    for y0 in range(0, v.canvas_h, strip):
        h = min(strip, v.canvas_h - y0)
        Tshift = np.array([[1.0, 0, 0], [0, 1.0, -y0], [0, 0, 1.0]])
        stack = np.full((len(imgs), h, v.canvas_w, 3), np.nan, np.float32)
        for j, (frame, mask, Hfv) in enumerate(imgs):
            M = Tshift @ Hfv  # frame -> strip; no WARP_INVERSE_MAP
            warped = cv2.warpPerspective(frame, M, (v.canvas_w, h), flags=cv2.INTER_LINEAR)
            cover = cv2.warpPerspective(mask, M, (v.canvas_w, h), flags=cv2.INTER_NEAREST) > 0
            stack[j][cover] = warped[cover]
        if len(imgs):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                med = np.nanmedian(stack, axis=0)
            out[y0:y0 + h] = np.nan_to_num(med, nan=0.0).astype(np.uint8)
    return out


# --------------------------------------------------------------------------- cache

def _settings_dict(cfg: MotionConfig) -> dict:
    return json.loads(json.dumps(asdict(cfg)))  # tuples become lists, so a reloaded value compares equal


def video_fingerprint(video: Path) -> dict:
    """Stem, size, SHA-256 of the first 8 MiB, and header frame count / fps / size."""
    video = Path(video)
    with video.open("rb") as fh:
        head = fh.read(8 * 1024 * 1024)
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video}")
    fp = {"video_stem": video.stem, "size_bytes": video.stat().st_size, "head_sha256": hashlib.sha256(head).hexdigest(),
          "n_frames": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), "fps": float(cap.get(cv2.CAP_PROP_FPS)),
          "frame_width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), "frame_height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}
    cap.release()
    return fp


def expected_meta(fingerprint: dict, detections_sha256: str, cfg: MotionConfig) -> dict:
    """Every field that goes into the cache key (and is compared on load)."""
    return {"fingerprint": dict(fingerprint), "detections_sha256": detections_sha256,
            "settings": _settings_dict(cfg), "algo_version": ALGO_VERSION, "cv2_version": cv2.__version__}


def motion_key(fingerprint: dict, detections_sha256: str, cfg: MotionConfig) -> str:
    """SHA-256 hex of the canonical JSON of `expected_meta`. The first 8 characters go in the file name."""
    blob = json.dumps(expected_meta(fingerprint, detections_sha256, cfg), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode()).hexdigest()


def cache_path(video: Path, key: str, cache_dir: Path = CACHE_DIR) -> Path:
    return Path(cache_dir) / f"{Path(video).stem}__motion__{key[:8]}.npz"


_ARRAY_KEYS = ("n_matches", "n_inliers", "inlier_ratio", "reproj_rms_px", "spread_cells", "keyframe", "chained",
               "flags", "key_frame_idx", "key_parent", "key_hops", "key_view", "key_chain_len", "key_H_to_ref", "view_key",
               "view_H_ref_to_view", "view_canvas_wh")


def save_motion(path: Path, motion: MotionTrack) -> None:
    """Write the motion cache (.npz, no pickle)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, H_frame_to_ref=motion.H_frame_to_ref, quality=motion.quality,
                        meta=np.array(json.dumps(motion.meta, default=str)),
                        **{k: motion.stats[k] for k in _ARRAY_KEYS})


def _compare(a, b, name: str, path: Path) -> None:
    if isinstance(a, dict) and isinstance(b, dict):
        for k in sorted(set(a) | set(b)):
            _compare(a.get(k), b.get(k), f"{name}.{k}", path)
    elif a != b:
        raise ValueError(f"Motion cache {path} does not match this run: {name} is {a!r} in the cache but {b!r} now.")


def load_motion(path: Path, expected: dict | None = None) -> MotionTrack:
    """Read a motion cache. With `expected` (from `expected_meta`) every key field must match exactly,
    otherwise ValueError names the field. The stored `motion_id` is re-computed and must match."""
    path = Path(path)
    z = np.load(path, allow_pickle=False)
    meta = json.loads(str(z["meta"]))
    if expected is not None:
        for k, v in expected.items():
            _compare(meta.get(k), v, k, path)
    stats = {k: z[k] for k in _ARRAY_KEYS}
    H, q = z["H_frame_to_ref"], z["quality"]
    views = tuple(View(i, int(stats["view_key"][i]), stats["view_H_ref_to_view"][i],
                       int(stats["view_canvas_wh"][i][0]), int(stats["view_canvas_wh"][i][1]))
                  for i in range(len(stats["view_key"])))
    mid = _motion_id(H, stats["view_H_ref_to_view"])
    if mid != meta.get("motion_id"):
        raise ValueError(f"Motion cache {path} is corrupt: motion_id does not match its matrices.")
    return MotionTrack(meta, H, q, stats, views, bool(meta["is_static"]), mid)


# --------------------------------------------------------------------------- video and detection helpers

def video_frames(video: Path, max_frames: int | None = None) -> Iterator[np.ndarray]:
    """Yield BGR frames sequentially from frame 0 (never seeks, so indices match the detection cache)."""
    cap = cv2.VideoCapture(str(video))
    try:
        i = 0
        while max_frames is None or i < max_frames:
            ok, frame = cap.read()
            if not ok:
                break
            yield frame
            i += 1
    finally:
        cap.release()


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_detection_boxes(path: Path, n_frames: int) -> tuple[list[np.ndarray], dict]:
    """Per-frame xyxy boxes (every class, every confidence) from a detection cache, plus its meta record."""
    z = np.load(path, allow_pickle=False)
    meta = json.loads(str(z["meta"]))
    frame_idx, xyxy = z["frame_idx"], z["xyxy"]
    bounds = np.searchsorted(frame_idx, np.arange(n_frames + 1))
    return [xyxy[a:b].astype(np.float64) for a, b in zip(bounds[:-1], bounds[1:])], meta


def check_detection_meta(det_meta: dict, video_fp: dict, n_needed: int, partial: bool) -> None:
    """Raise ValueError if the detection cache disagrees with the video (fps, frame count, stem)."""
    if det_meta.get("video_stem") != video_fp["video_stem"]:
        raise ValueError(f"Detection cache is for video {det_meta.get('video_stem')!r}, not {video_fp['video_stem']!r}.")
    if abs(float(det_meta["fps"]) - video_fp["fps"]) > 1e-3:
        raise ValueError(f"Detection cache fps {det_meta['fps']} differs from the video's {video_fp['fps']}.")
    if int(det_meta["n_frames"]) < n_needed:
        raise ValueError(f"Detection cache holds {det_meta['n_frames']} frames but {n_needed} are needed.")
    if not partial and abs(int(det_meta["n_frames"]) - video_fp["n_frames"]) > 1:
        raise ValueError(f"Detection cache has {det_meta['n_frames']} frames but the video has {video_fp['n_frames']}.")


def ensure_private_output(path: Path, repo_root: Path = REPO_ROOT) -> Path:
    """Return the resolved path, or raise ValueError unless it is under <repo>/data/frames or <repo>/runs.

    Footage shows minors: images and videos may only go to the git-ignored folders.
    """
    p = Path(path).resolve()
    allowed = (Path(repo_root).resolve() / "data" / "frames", Path(repo_root).resolve() / "runs")
    if not any(p == a or a in p.parents for a in allowed):
        raise ValueError(f"Refusing to write footage-derived output to {p}: use a folder under data/frames/ or runs/.")
    return p


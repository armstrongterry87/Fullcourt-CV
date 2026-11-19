"""Court geometry: specs per rule level, the pixel->feet homography, and the 2-or-3 test.

Spaces: pixels are `(x, y)` from the top-left (y grows downward); court space is FEET.
Rim A frame: origin at the centre of the baseline under rim A, x along the baseline,
y toward mid-court. The rim B frame is `x_B = -x_A`, `y_B = court_length_ft - y_A`.

Every court constant carries a `source` string and a `verified` flag that match the status
table in docs/design/004-shot-scoring.md section 5. Values the design marks X (secondary
source) or PROVISIONAL are `verified=False`. Values the design marks U (not found) are
unavailable (`value=None`); nothing here is filled in from memory.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import cv2
import numpy as np

from src.camera.motion import stretch_at, transform_points
from src.court.polygon import ConfigError
from src.shots.config import ShotConfig

if TYPE_CHECKING:
    from src.camera.motion import MotionTrack

FT_PER_M = 1.0 / 0.3048  # 1 ft = 0.3048 m (international foot). The only metre->feet conversion.
DEFAULT_LEVEL = "nfhs"


def _m(metres: float) -> float:
    """Convert metres to feet. FIBA numbers go through here once, in the spec table."""
    return metres * FT_PER_M


@dataclass(frozen=True)
class SpecValue:
    """One court constant (feet). `value=None` means unavailable (design status U)."""

    value: float | None
    source: str
    verified: bool
    status: str  # design status: V, S, X, D, U or PROVISIONAL


@dataclass(frozen=True)
class CourtSpec:
    """Court dimensions for one rule level. All lengths in feet."""

    level: str
    basket_offset_ft: SpecValue      # basket centre from the baseline (b)
    three_radius_ft: SpecValue       # arc radius from the basket centre (R)
    corner_half_width_ft: SpecValue  # |x| of the straight three-point segments (c)
    ft_line_ft: SpecValue            # free-throw line from the baseline
    court_width_ft: SpecValue
    court_length_ft: SpecValue

    def all_values(self) -> dict[str, SpecValue]:
        """Every constant by field name (used to audit source/verified)."""
        return {k: v for k, v in self.__dict__.items() if isinstance(v, SpecValue)}


_NFHS_SRC = "NFHS Rule 1, quoted verbatim in a Draper gymnasium design guide (secondary publisher; not the paywalled NFHS rule book): "
_PROV_B = ("ASSUMPTION, not a reading: taken equal to the NFHS 63 in (5.25 ft). The NCAA/NBA court diagrams are "
           "image-only and were not read. PROVISIONAL.")
_PROV_FT = ("Assumed equal to NFHS: 15 ft from the backboard face plus a 4 ft backboard offset (19.0 ft). "
            "Not read from NCAA/NBA text. PROVISIONAL.")
_U = "Design 004 sec.5: U (not found in any source read). Unavailable; supply it in the calibration file."


def _u() -> SpecValue:
    return SpecValue(None, _U, False, "U")


_SPECS: dict[str, CourtSpec] = {
    "nfhs": CourtSpec(
        "nfhs",
        SpecValue(5.25, _NFHS_SRC + "Rule 1 Sec.4 Art.1 straight segments 63 in from the end line; confirmed by "
                  "Sec.8 Art.1 (backboard 4 ft) + Sec.11 Art.1 (6 in) + Sec.10 Art.1 (9 in ring radius) = 63 in",
                  True, "S"),
        SpecValue(19.75, _NFHS_SRC + "Rule 1 Sec.4 Art.1, 19 ft 9 in from the point below the basket centre",
                  True, "S"),
        SpecValue(19.75, "Derived: NFHS has no separate corner distance, so c = R (Rule 1 Sec.4 Art.1)", True, "D"),
        SpecValue(19.0, _NFHS_SRC + "Rule 1 Sec.6 (15 ft from the backboard face) + Sec.8 Art.1 (backboard 4 ft "
                  "from the end line) = 19.0 ft", True, "S"),
        SpecValue(50.0, _NFHS_SRC + "Rule 1 Sec.1, court not greater than 94 x 50 ft", True, "S"),
        SpecValue(84.0, _NFHS_SRC + "Rule 1 Sec.1, ideal high school court 84 x 50 ft (up to 94 ft; override "
                  "court_length_ft in the calibration file)", True, "S"),
    ),
    "ncaa": CourtSpec(
        "ncaa",
        SpecValue(5.25, _PROV_B, False, "PROVISIONAL"),
        SpecValue(22.0 + 1.75 / 12.0, "NCAA release 5 Jun 2019: 22 ft 1 3/4 in, seen as a search excerpt only",
                  False, "S"),
        SpecValue(21.0 + 7.875 / 12.0, "NCAA.com article 26 Sep 2019: 21 ft 7 7/8 in, seen as a search "
                  "excerpt only (two secondary sites give 7 3/4 in and 8 in)", False, "S"),
        SpecValue(19.0, _PROV_FT, False, "PROVISIONAL"),
        _u(),
        _u(),
    ),
    "nba": CourtSpec(
        "nba",
        SpecValue(5.25, _PROV_B, False, "PROVISIONAL"),
        SpecValue(23.75, "NBA Rule No. 1, Section I: 23 ft 9 in from the middle of the basket", True, "V"),
        SpecValue(22.0, "Derived: 25 ft minus the 3 ft sideline offset (NBA Rule 1 Sec. I, V); depends on "
                  "the 50 ft width, which is X", False, "D"),
        SpecValue(19.0, _PROV_FT, False, "PROVISIONAL"),
        SpecValue(50.0, "Design 004 sec.5: court size 94 x 50 ft, secondary source (the rule text refers to "
                  "an image diagram)", False, "X"),
        SpecValue(94.0, "Design 004 sec.5: court size 94 x 50 ft, secondary source", False, "X"),
    ),
    "fiba": CourtSpec(
        "fiba",
        SpecValue(_m(1.575), "FIBA Official Basketball Rules 2024 v1.0a Art. 2.4.5: basket centre 1.575 m from "
                  "the inner edge of the mid-point of the endline (primary PDF)", True, "V"),
        SpecValue(_m(6.75), "FIBA Official Basketball Rules 2024 v1.0a Art. 2.4.5: arc radius 6.75 m from the "
                  "point beneath the basket centre, to the outer edge (primary PDF)", True, "V"),
        SpecValue(_m(6.60), "Derived: 7.50 m minus 0.90 m (Art. 2.4.5 straight segments 0.90 m from the sideline, "
                  "both V)", True, "D"),
        SpecValue(_m(5.80), "FIBA Official Basketball Rules 2024 v1.0a Art. 2.4.3: free-throw line furthest edge "
                  "5.80 m from the inner edge of the endline (primary PDF)", True, "V"),
        SpecValue(_m(15.0), "FIBA Official Basketball Rules 2024 v1.0a Art. 2.1: 28 m x 15 m, from the inner edge "
                  "of the boundary line", True, "V"),
        SpecValue(_m(28.0), "FIBA Official Basketball Rules 2024 v1.0a Art. 2.1: 28 m x 15 m, from the inner edge "
                  "of the boundary line", True, "V"),
    ),
}

LEVELS = tuple(_SPECS)


def get_spec(level: str) -> CourtSpec:
    """Return the spec for `level` ('nfhs', 'ncaa', 'nba' or 'fiba'); raise ConfigError if unknown."""
    try:
        return _SPECS[level.lower()]
    except (KeyError, AttributeError):
        raise ConfigError(f"Unknown court level {level!r}; use one of {', '.join(LEVELS)}.") from None


# --------------------------------------------------------------------------- homography

@dataclass(frozen=True)
class Homography:
    """`H_px_to_ft` maps pixels to feet in the rim A frame. `rms_ft` is the reprojection RMS."""

    H_px_to_ft: np.ndarray
    rms_ft: float
    n_points: int
    n_inliers: int


def fit_homography(pts_px: np.ndarray, pts_ft: np.ndarray, cfg: ShotConfig | None = None) -> Homography:
    """Fit pixels -> feet. Least squares for 4 or 5 pairs, RANSAC (0.5 ft) for 6 or more.

    Raises ConfigError for fewer than 4 pairs, a degenerate fit, or RMS above the limit.
    The RMS is taken over the inliers (all points when least squares is used).
    """
    cfg = cfg or ShotConfig()
    pts_px = np.asarray(pts_px, dtype=np.float64).reshape(-1, 2)
    pts_ft = np.asarray(pts_ft, dtype=np.float64).reshape(-1, 2)
    n = len(pts_px)
    if n < 4 or len(pts_ft) != n:
        raise ConfigError(f"A homography needs at least 4 matching px/ft pairs, found {n}.")
    method = cv2.RANSAC if n >= 6 else 0
    H, mask = cv2.findHomography(pts_px, pts_ft, method, cfg.homography_ransac_ft)
    if H is None or not np.all(np.isfinite(H)):
        raise ConfigError("The homography could not be fitted (points may be collinear).")
    inl = np.ones(n, dtype=bool) if (method == 0 or mask is None) else mask.ravel().astype(bool)
    err = np.linalg.norm(px_to_ft(H, pts_px) - pts_ft, axis=1)
    rms = float(math.sqrt(np.mean(err[inl] ** 2))) if inl.any() else float("inf")
    if rms > cfg.homography_max_rms_ft:
        raise ConfigError(f"Homography reprojection RMS is {rms:.2f} ft (limit {cfg.homography_max_rms_ft} ft); "
                          f"re-click the landmarks.")
    return Homography(H, rms, n, int(inl.sum()))


def px_to_ft(H_px_to_ft: np.ndarray, pts_px: np.ndarray) -> np.ndarray:
    """Map pixel points (N,2) to rim-A-frame feet (N,2) with `cv2.perspectiveTransform`."""
    p = np.asarray(pts_px, dtype=np.float64).reshape(-1, 1, 2)
    return cv2.perspectiveTransform(p, np.asarray(H_px_to_ft, dtype=np.float64)).reshape(-1, 2)


def to_rim_frame(xy_ft: np.ndarray, rim_id: str, court_length_ft: float | None) -> np.ndarray:
    """Convert rim-A-frame feet to the frame of the rim being attacked (identity for rim A)."""
    xy = np.asarray(xy_ft, dtype=np.float64).reshape(-1, 2)
    if rim_id == "A":
        return xy.copy()
    if court_length_ft is None:
        raise ConfigError("court_length_ft is needed to express feet in the rim B frame.")
    return np.column_stack([-xy[:, 0], court_length_ft - xy[:, 1]])


# --------------------------------------------------------------------------- calibration file

@dataclass(frozen=True)
class Rim:
    """A rim box in pixels. `box_px` = (x1, y1, x2, y2); `source` is 'manual' or 'median'.

    `view` is the motion-compensation view (mosaic canvas) whose pixels `box_px` is in; 0 for a static camera.
    """

    id: str
    box_px: tuple[float, float, float, float]
    source: str
    view: int = 0

    @property
    def w_px(self) -> float:
        return self.box_px[2] - self.box_px[0]

    @property
    def cx_px(self) -> float:
        return (self.box_px[0] + self.box_px[2]) / 2

    @property
    def cy_px(self) -> float:
        return (self.box_px[1] + self.box_px[3]) / 2

    @property
    def top_px(self) -> float:
        return self.box_px[1]

    @property
    def bottom_px(self) -> float:
        return self.box_px[3]


@dataclass(frozen=True)
class Calibration:
    """Optional shot-engine keys of the court JSON. Old polygon-only files load with no rims/homography."""

    level: str
    court_length_ft: float | None
    three_point_radius_ft: float | None
    homography: Homography | None
    rims: tuple[Rim, ...]
    camera_static: bool
    frame_width: int | None
    frame_height: int | None
    # design 005: set when the court was calibrated on a mosaic (camera_static is then false). For such a
    # calibration `homography.H_px_to_ft` holds H_ref_to_ft (reference pixels to feet), not raw pixels.
    motion_id: str | None = None
    camera_static_explicit: bool = True  # False when the JSON had no camera_static key (assumed static)

    def spec(self) -> CourtSpec:
        return get_spec(self.level)

    def effective_length_ft(self) -> float | None:
        """Court length: the calibration value, else the level's default (None if unavailable)."""
        return self.court_length_ft if self.court_length_ft is not None else self.spec().court_length_ft.value


def check_motion_id(calib: Calibration, motion: "MotionTrack") -> None:
    """Raise ConfigError unless the calibration was clicked on the mosaic of THIS motion file."""
    if calib.motion_id is None or calib.motion_id != motion.motion_id:
        raise ConfigError(f"The calibration's motion_id ({calib.motion_id!r}) does not match the motion file "
                          f"({motion.motion_id!r}): recalibrate on the current mosaic "
                          f"(scripts/calibrate_court.py --mosaic).")


def load_calibration(path: Path, cfg: ShotConfig | None = None, motion: "MotionTrack | None" = None) -> Calibration:
    """Read the shot-engine keys from the court JSON (same file `load_court` reads).

    With `motion` and `camera_static: false`, landmark points are in view (mosaic) pixels: each is mapped to
    reference pixels and ONE `H_ref_to_ft` is fitted. A landmark too stretched in the reference is rejected.
    """
    path = Path(path)
    try:
        d = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as e:
        raise ConfigError(f"{path}: cannot read calibration ({e}).") from e
    level = str(d.get("level", DEFAULT_LEVEL)).lower()
    get_spec(level)
    hp = d.get("homography_points")
    hom = None
    static = bool(d.get("camera_static", True))
    mapped = motion is not None and not static
    if mapped:
        check_motion_id(Calibration(level, None, None, None, (), static, None, None, d.get("motion_id")), motion)
    if hp:
        try:
            px = np.array([p["px"] for p in hp], dtype=np.float64)
            ft = np.array([p["ft"] for p in hp], dtype=np.float64)
        except (KeyError, TypeError, ValueError) as e:
            raise ConfigError(f"{path}: each homography point needs 'px' and 'ft' ({e}).") from e
        if px.ndim != 2 or px.shape[1] != 2 or ft.shape != px.shape:
            raise ConfigError(f"{path}: homography 'px' and 'ft' must be [x, y] pairs.")
        if mapped:
            px = _view_to_ref(px, [int(p.get("view", 0)) for p in hp], motion, path)
        hom = fit_homography(px, ft, cfg)
    rims = []
    for r in d.get("rims") or []:
        box = r.get("box_px")
        if r.get("id") not in ("A", "B") or not box or len(box) != 4 or box[2] <= box[0] or box[3] <= box[1]:
            raise ConfigError(f"{path}: a rim needs id 'A' or 'B' and box_px [x1, y1, x2, y2] with x2>x1, y2>y1.")
        rims.append(Rim(r["id"], tuple(float(v) for v in box), str(r.get("source", "manual")), int(r.get("view", 0))))
    length = d.get("court_length_ft")
    radius = d.get("three_point_radius_ft")
    return Calibration(
        level, None if length is None else float(length), None if radius is None else float(radius),
        hom, tuple(rims), static,
        d.get("frame_width"), d.get("frame_height"), d.get("motion_id"), "camera_static" in d,
    )


def _view_to_ref(px_view: np.ndarray, views: list[int], motion: "MotionTrack", path: Path) -> np.ndarray:
    """Map landmark pixels clicked on mosaic views to reference pixels; reject over-stretched landmarks."""
    out = np.zeros_like(px_view)
    limit = float(motion.meta.get("settings", {}).get("calib_max_stretch", 4.0))
    for i, (p, v) in enumerate(zip(px_view, views)):
        if not (0 <= v < len(motion.views)):
            raise ConfigError(f"{path}: landmark {i} is on view {v}, but the motion file has {len(motion.views)} views.")
        H_view_to_ref = np.linalg.inv(motion.views[v].H_ref_to_view)
        s = stretch_at(H_view_to_ref, (float(p[0]), float(p[1])))
        if not (s <= limit):
            raise ConfigError(f"{path}: landmark {i} (view {v}) is stretched {s:.1f}x in the reference (limit {limit}); "
                              f"pick a landmark nearer the centre of its mosaic.")
        out[i] = transform_points(H_view_to_ref, p[None])[0]
    return out


# --------------------------------------------------------------------------- two or three

def _line_geometry(level: str, three_radius_ft: float | None) -> tuple[float, float, float, float]:
    """(b, R, c, y_join) for the level. y_join is where the straight segments meet the arc."""
    s = get_spec(level)
    b = s.basket_offset_ft.value
    R = three_radius_ft if three_radius_ft is not None else s.three_radius_ft.value
    c = s.corner_half_width_ft.value
    if three_radius_ft is not None and level.lower() == "nfhs":
        c = three_radius_ft  # NFHS has no separate corner distance: c = R
    if b is None or R is None or c is None:
        raise ConfigError(f"Level {level!r} has an unavailable three-point dimension.")
    c = min(c, R)
    return b, R, c, b + math.sqrt(max(R * R - c * c, 0.0))


def is_three(court_xy_ft: tuple[float, float], level: str = DEFAULT_LEVEL,
             three_radius_ft: float | None = None) -> bool:
    """True if feet at `court_xy_ft` (attacked-rim frame) are a three. A foot on the line is a two."""
    b, R, c, _ = _line_geometry(level, three_radius_ft)
    x, y = float(court_xy_ft[0]), float(court_xy_ft[1])
    two = abs(x) <= c and (y <= b or math.hypot(x, y - b) <= R)
    return not two


def three_margin_ft(court_xy_ft: tuple[float, float], level: str = DEFAULT_LEVEL,
                    three_radius_ft: float | None = None) -> float:
    """Distance in feet from `court_xy_ft` to the nearest part of the three-point line."""
    b, R, c, y_join = _line_geometry(level, three_radius_ft)
    x, y = float(court_xy_ft[0]), float(court_xy_ft[1])
    best = math.inf
    for sx in (-c, c):  # straight segments, y in [0, y_join]
        best = min(best, math.hypot(x - sx, y - min(max(y, 0.0), y_join)))
    dx, dy = x, y - b
    d = math.hypot(dx, dy)
    if d > 1e-9:
        qx, qy = R * dx / d, b + R * dy / d  # radial projection onto the circle
        if abs(qx) <= c and qy >= y_join - 1e-9:
            best = min(best, abs(d - R))
    for sx in (-c, c):  # arc end points
        best = min(best, math.hypot(x - sx, y - y_join))
    return best


def feet_off_court(xy_ft: tuple[float, float], spec: CourtSpec, length_ft: float | None,
                   tol_ft: float) -> bool:
    """True if the feet are more than `tol_ft` outside the court. Unavailable dimensions are skipped."""
    x, y = float(xy_ft[0]), float(xy_ft[1])
    w = spec.court_width_ft.value
    if w is not None and abs(x) > w / 2 + tol_ft:
        return True
    if y < -tol_ft:
        return True
    return length_ft is not None and y > length_ft + tol_ft

"""Court polygon (pixel space) used to drop people who are off the court.

The polygon is clicked by hand on one frame and saved as numbers-only JSON
(no image is ever written here, because footage may contain minors).

NOTE: this is a pixel-space test and is perspective/camera dependent. Once a
homography exists it should move to court feet (`_ft`) and this module goes away.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


class ConfigError(ValueError):
    """A court file does not match the video or is malformed (a user-facing setup error)."""


@dataclass(frozen=True)
class Court:
    """Court polygon in pixels, valid for one resolution and (optionally) a frame range."""

    video_stem: str
    frame_index: int
    frame_width: int
    frame_height: int
    polygon_px: list[list[float]]
    valid_frames: tuple[int, int] | None  # inclusive [start, end], or None = whole clip


def save_court(
    path: Path, video_stem: str, frame_index: int, frame_width: int, frame_height: int,
    polygon_px: list[list[float]], valid_frames: tuple[int, int] | None = None,
) -> None:
    """Write the polygon JSON (numbers only). `frame_index` is the 0-based frame clicked on."""
    if len(polygon_px) < 3:
        raise ValueError("A court polygon needs at least 3 points.")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({
        "video_stem": video_stem,
        "frame_index": int(frame_index),
        "frame_width": int(frame_width),
        "frame_height": int(frame_height),
        "polygon_px": [[float(x), float(y)] for x, y in polygon_px],
        "valid_frames": [int(valid_frames[0]), int(valid_frames[1])] if valid_frames else None,
    }, indent=2))


def load_court(path: Path) -> Court:
    """Read a polygon JSON written by `save_court`."""
    d = json.loads(Path(path).read_text())
    if len(d["polygon_px"]) < 3:
        raise ConfigError(f"{path}: a court polygon needs at least 3 points, found {len(d['polygon_px'])}.")
    vf = d.get("valid_frames")
    return Court(d["video_stem"], d["frame_index"], d["frame_width"], d["frame_height"],
                 d["polygon_px"], (int(vf[0]), int(vf[1])) if vf else None)


def check_video_stem(court: Court, video_stem: str) -> None:
    """Raise ValueError if the polygon was drawn on a different video (same size is not enough)."""
    if court.video_stem != video_stem:
        raise ConfigError(
            f"Court polygon was drawn on video '{court.video_stem}' but you are running '{video_stem}'. "
            f"Pass --allow-court-stem-mismatch only if both are the same camera position."
        )


def check_resolution(court: Court, frame_width: int, frame_height: int) -> None:
    """Raise ValueError if the video size differs from the one the polygon was drawn on."""
    if (court.frame_width, court.frame_height) != (frame_width, frame_height):
        raise ConfigError(
            f"Court polygon was drawn at {court.frame_width}x{court.frame_height} but the video is "
            f"{frame_width}x{frame_height}."
        )


def court_keep_mask(xyxy_px: np.ndarray, court: Court, frame_index: int, margin_px: float = 20.0) -> np.ndarray:
    """Boolean mask of boxes to keep. Feet = bottom-centre ((x1+x2)/2, y2), in pixels.

    A box is kept if its feet are inside the polygon or within `margin_px` outside it.
    Outside `court.valid_frames` every box is kept.
    """
    n = len(xyxy_px)
    if court.valid_frames is not None and not (court.valid_frames[0] <= frame_index <= court.valid_frames[1]):
        return np.ones(n, dtype=bool)
    poly = np.asarray(court.polygon_px, dtype=np.float32).reshape(-1, 1, 2)
    keep = np.zeros(n, dtype=bool)
    for k, (x1, _y1, x2, y2) in enumerate(np.asarray(xyxy_px, dtype=np.float64)):
        feet_px = (float((x1 + x2) / 2), float(y2))
        keep[k] = cv2.pointPolygonTest(poly, feet_px, True) >= -margin_px
    return keep

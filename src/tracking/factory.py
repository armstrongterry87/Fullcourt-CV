"""Tracker construction: one place for every tracker constant.

`PRESETS` is the grid of settings we compare. `PlayerTracker` wraps either
ByteTrack or BoT-SORT from Roboflow's `trackers` package (Apache-2.0) behind one
`update(detections, frame_bgr)` call.

Units: `lost_seconds` is real seconds. The library wants `lost_track_buffer` as a
count of frames *at 30 fps* (it divides by 30 and rescales with `frame_rate`), so
we pass round(lost_seconds * 30) no matter what the video fps is.
"""

from __future__ import annotations

import inspect
from dataclasses import dataclass
from typing import Any

import numpy as np
import supervision as sv


@dataclass(frozen=True)
class TrackerSettings:
    """Settings for one tracker run."""

    name: str  # "bytetrack" or "botsort"
    lost_seconds: float = 1.0
    track_activation_threshold: float = 0.7
    high_conf_det_threshold: float = 0.6
    extra: tuple[tuple[str, Any], ...] = ()  # tracker-specific (key, value) kwargs, passed through as-is; tuple keeps presets immutable


PRESETS: dict[str, TrackerSettings] = {
    "A0": TrackerSettings("bytetrack"),
    "A1": TrackerSettings("bytetrack", lost_seconds=3.0),
    "B0": TrackerSettings("botsort", extra=(("enable_cmc", False),)),
    "B1": TrackerSettings("botsort"),
    "B2": TrackerSettings("botsort", lost_seconds=3.0),
    "B3": TrackerSettings("botsort", lost_seconds=3.0, extra=(("cmc_method", "orb"),)),
    "B4": TrackerSettings(
        "botsort", lost_seconds=3.0, track_activation_threshold=0.5, high_conf_det_threshold=0.5
    ),
}

DEFAULT_PRESET = {"bytetrack": "A0", "botsort": "B1"}


def build_kwargs(settings: TrackerSettings, fps: float) -> dict[str, Any]:
    """Constructor kwargs for the library tracker (also saved in the run summary)."""
    return {
        "lost_track_buffer": round(settings.lost_seconds * 30),
        "frame_rate": fps,
        "track_activation_threshold": settings.track_activation_threshold,
        "high_conf_det_threshold": settings.high_conf_det_threshold,
        **dict(settings.extra),
    }


class PlayerTracker:
    """ByteTrack or BoT-SORT behind one interface."""

    def __init__(self, settings: TrackerSettings, fps: float) -> None:
        from trackers import BoTSORTTracker, ByteTrackTracker

        classes = {"bytetrack": ByteTrackTracker, "botsort": BoTSORTTracker}
        if settings.name not in classes:
            raise ValueError(f"Unknown tracker {settings.name!r}; choose from {sorted(classes)}")
        self.settings = settings
        self.kwargs = build_kwargs(settings, fps)
        cls = classes[settings.name]
        self._tracker = cls(**self.kwargs)  # bad extra key -> TypeError
        # effective config: library defaults overlaid with what we passed (for run summaries)
        defaults = {
            k: v.default for k, v in inspect.signature(cls).parameters.items()
            if v.default is not inspect.Parameter.empty
        }
        self.resolved_kwargs: dict[str, Any] = {
            k: (v if isinstance(v, (int, float, str, bool, type(None))) else getattr(v, "__name__", repr(v)))
            for k, v in {**defaults, **self.kwargs}.items()
        }

    def update(self, detections: sv.Detections, frame_bgr: np.ndarray) -> sv.Detections:
        """Track one frame. The BGR frame is used only by BoT-SORT (camera motion compensation)."""
        if self.settings.name == "botsort":
            return self._tracker.update(detections, frame=frame_bgr)
        return self._tracker.update(detections)  # ByteTrack warns if given a frame

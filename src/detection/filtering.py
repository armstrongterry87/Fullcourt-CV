"""Per-class confidence thresholds applied to raw detections (pure, no I/O)."""

from __future__ import annotations

import numpy as np
import supervision as sv

# Class names are matched by meaning, so the same code works for COCO and fine-tuned models.
PLAYER_NAMES = {"person", "player"}
REFEREE_NAMES = {"referee"}
BALL_NAMES = {"sports ball", "ball"}
RIM_NAMES = {"rim"}
PEOPLE_NAMES = PLAYER_NAMES | REFEREE_NAMES


def apply_class_thresholds(
    detections: sv.Detections,
    names: np.ndarray,
    person_threshold: float,
    ball_threshold: float,
    rim_threshold: float,
) -> tuple[sv.Detections, np.ndarray]:
    """Keep detections whose confidence is strictly above their class's threshold.

    People (player/referee/person) use `person_threshold`, the ball uses
    `ball_threshold`, everything else (rim, other classes) uses `rim_threshold`.
    Returns the filtered detections and the matching filtered `names`.
    The comparison is strict (`conf > thr`); the replay gate against the committed
    A0 run is what confirms this matches rfdetr's own cut.
    """
    if len(detections) == 0 or detections.confidence is None:
        return detections, names
    thr = np.full(len(detections), rim_threshold, dtype=np.float64)
    thr[np.isin(names, list(PEOPLE_NAMES))] = person_threshold
    thr[np.isin(names, list(BALL_NAMES))] = ball_threshold
    conf = np.asarray(detections.confidence, dtype=np.float32)  # the cache stores float32; coerce so live == replay
    keep = conf > thr.astype(np.float32)  # compare in float32: float32(0.4) > float64(0.4) would wrongly keep a 0.40 box
    return detections[keep], names[keep]

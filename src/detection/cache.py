"""Save and replay raw detector output so tracker comparisons share identical detections.

One .npz per (video, weights content, model size, threshold). It stores the model's raw
output for every class, before any class filtering, plus a JSON meta record.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import supervision as sv

REPO_ROOT = Path(__file__).resolve().parents[2]
CACHE_DIR = REPO_ROOT / "runs" / "cache"  # anchored to the repo, not the process cwd

CHECKED_KEYS = ("video_stem", "fps", "weights_name", "weights_sha256", "model_size", "threshold", "classes")


def weights_sha256(weights: Path | None) -> str | None:
    """SHA-256 of the weights file content (None for the COCO baseline)."""
    return hashlib.sha256(weights.read_bytes()).hexdigest() if weights is not None else None


def cache_path(video: Path, sha256: str | None, model_size: str, threshold: float, cache_dir: Path = CACHE_DIR) -> Path:
    """Default cache location; different checkpoints get different files."""
    w = f"w{sha256[:8]}" if sha256 else "coco"
    return cache_dir / f"{video.stem}__{w}__{model_size}__thr{threshold:.2f}.npz"


def make_meta(
    video: Path, n_frames: int, fps: float, weights: Path | None, model_size: str, threshold: float,
    classes: list[str] | None, sha256: str | None = None,
) -> dict:
    """Describe what produced the detections (checked again on load)."""
    return {
        "video_stem": video.stem,
        "n_frames": int(n_frames),
        "fps": float(fps),
        "weights_name": weights.name if weights is not None else None,
        "weights_sha256": sha256 if sha256 is not None else weights_sha256(weights),
        "model_size": model_size,
        "threshold": float(threshold),
        "classes": classes,
    }


def save_detections(path: Path, per_frame: list[sv.Detections], meta: dict) -> None:
    """Write raw per-frame detections (index = 0-based frame number) to one .npz."""
    path.parent.mkdir(parents=True, exist_ok=True)
    idx, xyxy, conf, cid = [], [], [], []
    for i, d in enumerate(per_frame):
        n = len(d)
        if n == 0:
            continue
        idx.append(np.full(n, i, dtype=np.int64))
        xyxy.append(np.asarray(d.xyxy, dtype=np.float32))
        conf.append(np.asarray(d.confidence, dtype=np.float32))
        cid.append(np.asarray(d.class_id, dtype=np.int64))
    cat = lambda parts, shape, dt: np.concatenate(parts) if parts else np.zeros(shape, dtype=dt)  # noqa: E731
    np.savez_compressed(
        path,
        frame_idx=cat(idx, (0,), np.int64),
        xyxy=cat(xyxy, (0, 4), np.float32),
        confidence=cat(conf, (0,), np.float32),
        class_id=cat(cid, (0,), np.int64),
        meta=np.array(json.dumps(meta)),
    )


def read_meta(path: Path) -> dict:
    """Meta record of a cache file."""
    return json.loads(str(np.load(path, allow_pickle=False)["meta"]))


def load_detections(path: Path, expected: dict, n_frames_needed: int) -> list[sv.Detections]:
    """Load cached detections for the first `n_frames_needed` frames.

    Raises ValueError if the cache was made from a different video, fps, weights
    content, model size, threshold or class list, or holds fewer frames than needed.
    """
    z = np.load(path, allow_pickle=False)
    meta = json.loads(str(z["meta"]))
    for key in CHECKED_KEYS:
        if meta.get(key) != expected.get(key):
            raise ValueError(f"Detection cache {path} does not match this run: {key} is {meta.get(key)!r} in the cache but {expected.get(key)!r} now.")
    if meta["n_frames"] < n_frames_needed:
        raise ValueError(f"Detection cache {path} holds {meta['n_frames']} frames but {n_frames_needed} are needed.")
    # read each array once (NpzFile decompresses on every lookup)
    frame_idx, xyxy, conf, cid = z["frame_idx"], z["xyxy"], z["confidence"], z["class_id"]
    bounds = np.searchsorted(frame_idx, np.arange(n_frames_needed + 1))  # frame_idx is sorted
    return [
        sv.Detections(xyxy=xyxy[a:b].copy(), confidence=conf[a:b].copy(), class_id=cid[a:b].copy())
        for a, b in zip(bounds[:-1], bounds[1:])
    ]

"""Detect and track players, referees, ball and rim in a game video.

Two modes:
  - Baseline (no --weights): RF-DETR pretrained on COCO, using its "person" and
    "sports ball" classes. Shows what off-the-shelf models get right.
  - Fine-tuned (--weights): your own trained RF-DETR checkpoint with the classes
    player / referee / ball / rim.

Tracking uses BoT-SORT (default) or ByteTrack from Roboflow's `trackers` package;
settings live in src/tracking/factory.py (PRESETS).

Outputs:
  - an annotated video (boxes, track IDs, short motion trails)
  - a CSV with one row per detection per frame

Examples:
  python scripts/baseline_track.py --video data/raw/game1.mp4 --max-seconds 120
  python scripts/baseline_track.py --video data/raw/game1.mp4 --show   # watch it live
  python scripts/baseline_track.py --video data/raw/game1.mp4 --weights models/weights.pt --show
  # detect once, then compare trackers on identical detections:
  python scripts/baseline_track.py --video V --weights W --preset A0 --save-detections --no-video
  python scripts/baseline_track.py --video V --weights W --preset B2 --detections runs/cache/<file>.npz --no-video

Install first:
  pip install rfdetr trackers supervision opencv-python
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import supervision as sv

# imports below must follow the sys.path bootstrap above (script is run as a file, not a package)
from src.detection.cache import cache_path, load_detections, make_meta, read_meta, save_detections, weights_sha256  # noqa: E402
from src.tracking.factory import DEFAULT_PRESET, PRESETS, PlayerTracker, TrackerSettings  # noqa: E402

# Class names are matched by meaning, so the same code works for COCO and fine-tuned models.
PLAYER_NAMES = {"person", "player"}
REFEREE_NAMES = {"referee"}
BALL_NAMES = {"sports ball", "ball"}
RIM_NAMES = {"rim"}


def load_model(size: str, weights: Path | None = None, trust_checkpoint: bool = False):
    """Load RF-DETR: COCO-pretrained by size, or a fine-tuned checkpoint from --weights.

    Roboflow's downloaded weights.pt does not record its model size or class names,
    so the size comes from --model and the names from --classes.
    Nano/Small/Medium/Large are Apache-2.0.
    """
    import rfdetr

    classes = {
        "nano": rfdetr.RFDETRNano,
        "small": rfdetr.RFDETRSmall,
        "medium": rfdetr.RFDETRMedium,
        "large": rfdetr.RFDETRLarge,
    }
    if weights is None:
        return classes[size]()

    weights = weights.resolve()  # rfdetr treats bare file names as its own download cache
    if not weights.exists():
        raise SystemExit(f"Weights file not found: {weights}")
    try:
        return classes[size](
            pretrain_weights=str(weights),
            num_classes=checkpoint_num_classes(weights),
            trust_checkpoint=trust_checkpoint,
        )
    except Exception as e:
        raise SystemExit(
            f"Could not load {weights} as RF-DETR {size}: {e}\n"
            "Check --model matches the size you trained (Small by default). If the file came from "
            "your own Roboflow account and safe loading failed, rerun with --trust-checkpoint."
        )


def checkpoint_num_classes(weights: Path) -> int:
    """Number of trained classes, read from the classifier head (it has one extra row)."""
    import torch

    state = torch.load(str(weights), map_location="cpu", weights_only=True)
    head = state.get("model", state)["class_embed.weight"]
    return int(head.shape[0]) - 1


def make_namer(model, class_list: list[str] | None):
    """Return a function mapping a detection's class_id to its class name.

    Fine-tuned Roboflow checkpoints number classes from 1 in alphabetical order
    (1 = ball, 2 = player, 3 = referee, 4 = rim), so id k is class_list[k - 1].
    The COCO model's predictions use original COCO ids (person = 1).
    """
    if class_list is not None:
        return lambda cid: class_list[cid - 1] if 1 <= cid <= len(class_list) else str(cid)

    from rfdetr.assets.coco_classes import COCO_CLASSES

    return lambda cid: COCO_CLASSES.get(cid, str(cid))


def class_names(detections: sv.Detections, namer) -> list[str]:
    """Class name for each detection, from its class id.

    (rfdetr attaches COCO names to every prediction, which are wrong for fine-tuned
    models, so names always come from the namer.)
    """
    return [namer(int(c)) for c in detections.class_id]


def pick(detections: sv.Detections, names: np.ndarray, wanted: set[str]) -> sv.Detections:
    """Keep only detections whose class name is in `wanted`."""
    if len(detections) == 0:
        return detections
    return detections[np.isin(names, list(wanted))]


def run(
    video_path: Path,
    out_video: Path,
    out_csv: Path,
    model,
    namer,
    threshold: float,
    max_seconds: float | None,
    show: bool = False,
    settings: TrackerSettings = PRESETS[DEFAULT_PRESET["botsort"]],
    cached: list[sv.Detections] | None = None,
    write_video: bool = True,
    save_cache_meta: tuple[Path, dict] | None = None,
) -> dict:
    """Detect, track and annotate every frame. Returns a short summary.

    With show=True, a window plays the annotated frames live (press q to stop early).
    """
    info = sv.VideoInfo.from_video_path(str(video_path))
    fps = float(info.fps)
    max_frames = int(max_seconds * fps) if max_seconds else info.total_frames
    if cached is not None:
        max_frames = min(max_frames, len(cached))

    tracker = PlayerTracker(settings, fps)
    raw_per_frame: list[sv.Detections] = []
    track_secs = 0.0
    box = sv.BoxAnnotator(thickness=2)
    ref_box = sv.BoxAnnotator(thickness=2, color=sv.Color.YELLOW)
    rim_box = sv.BoxAnnotator(thickness=2, color=sv.Color.GREEN)
    label = sv.LabelAnnotator(text_scale=0.5, text_padding=3)
    trace = sv.TraceAnnotator(trace_length=int(fps))  # ~1 second trail
    ball_marker = sv.TriangleAnnotator(base=18, height=16)

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    frames_done, ball_frames, player_ids = 0, 0, set()
    sink_cm = sv.VideoSink(str(out_video), info) if write_video else contextlib.nullcontext()

    with sink_cm as sink, open(out_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["frame", "time_s", "class", "track_id", "conf", "x1_px", "y1_px", "x2_px", "y2_px"])

        for i, frame in enumerate(sv.get_video_frames_generator(str(video_path))):
            if i >= max_frames:
                break
            if cached is not None:
                detections = cached[i]
            else:
                # frames from OpenCV are BGR; RF-DETR expects RGB
                detections = model.predict(frame[:, :, ::-1].copy(), threshold=threshold)
            if save_cache_meta is not None:
                raw_per_frame.append(detections)
            names = np.array(class_names(detections, namer))

            # players and referees are tracked together (both are people moving on court)
            people = pick(detections, names, PLAYER_NAMES | REFEREE_NAMES)
            people_names = names[np.isin(names, list(PLAYER_NAMES | REFEREE_NAMES))] if len(detections) else names
            ball = pick(detections, names, BALL_NAMES)
            rims = pick(detections, names, RIM_NAMES)
            if len(ball) > 1:  # keep only the most confident ball
                ball = ball[[int(np.argmax(ball.confidence))]]

            if len(people):
                people.data["class_name"] = people_names
            t0 = time.perf_counter()
            people = tracker.update(people, frame)
            track_secs += time.perf_counter() - t0
            is_ref = (
                np.isin(people.data["class_name"], list(REFEREE_NAMES))
                if len(people) and "class_name" in people.data
                else np.zeros(len(people), dtype=bool)
            )
            players, refs = people[~is_ref], people[is_ref]
            player_ids.update(int(t) for t in players.tracker_id if t is not None and t >= 0)
            ball_frames += int(len(ball) > 0)

            t = i / fps
            for d_class, dets in (("player", players), ("referee", refs), ("ball", ball), ("rim", rims)):
                for k in range(len(dets)):
                    x1, y1, x2, y2 = dets.xyxy[k]
                    tid = int(dets.tracker_id[k]) if dets.tracker_id is not None else -1
                    conf = float(dets.confidence[k]) if dets.confidence is not None else ""
                    writer.writerow([i, f"{t:.3f}", d_class, tid, conf, f"{x1:.1f}", f"{y1:.1f}", f"{x2:.1f}", f"{y2:.1f}"])

            def id_labels(dets: sv.Detections, prefix: str = "") -> list[str]:
                if dets.tracker_id is None:
                    return [prefix.strip()] * len(dets)
                return [f"{prefix}#{int(t)}" if t >= 0 else prefix.strip() for t in dets.tracker_id]

            if sink is not None or show:
                annotated = trace.annotate(frame.copy(), players)
                annotated = box.annotate(annotated, players)
                annotated = label.annotate(annotated, players, id_labels(players))
                annotated = ref_box.annotate(annotated, refs)
                annotated = label.annotate(annotated, refs, id_labels(refs, "REF "))
                annotated = rim_box.annotate(annotated, rims)
                annotated = ball_marker.annotate(annotated, ball)
            if sink is not None:
                sink.write_frame(annotated)
            frames_done += 1
            print(f"  frame {frames_done}/{max_frames}", end="\r", flush=True)

            if show:
                import cv2

                cv2.imshow("Basketball CV - press q to stop", annotated)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

    if show:
        import cv2

        cv2.destroyAllWindows()
    print()

    if save_cache_meta is not None:
        path, meta = save_cache_meta
        save_detections(path, raw_per_frame, {**meta, "n_frames": frames_done, "fps": fps})

    return {
        "frames": frames_done,
        "fps": fps,
        "unique_player_ids": len(player_ids),
        "ball_seen_pct": round(100 * ball_frames / max(frames_done, 1), 1),
        "tracker_ms_per_frame": round(1000 * track_secs / max(frames_done, 1), 3),
        "tracker_kwargs": tracker.kwargs,
        "tracker_resolved_config": tracker.resolved_kwargs,
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--video", type=Path, required=True, help="input game video")
    p.add_argument("--out-dir", type=Path, default=None, help="where outputs go (default: runs/baseline or runs/finetuned)")
    p.add_argument("--model", choices=["nano", "small", "medium", "large"], default="small", help="model size (with --weights: the size you trained)")
    p.add_argument("--weights", type=Path, default=None, help="fine-tuned RF-DETR checkpoint, e.g. models/weights.pt")
    p.add_argument("--classes", default="ball,player,referee,rim", help="fine-tuned class names in alphabetical order, comma-separated")
    p.add_argument("--trust-checkpoint", action="store_true", help="allow full loading of a checkpoint you trust (your own Roboflow download)")
    p.add_argument("--threshold", type=float, default=0.4, help="detection confidence threshold")
    p.add_argument("--max-seconds", type=float, default=None, help="only process the first N seconds")
    p.add_argument("--tracker", choices=["bytetrack", "botsort"], default="botsort", help="tracker; picks its default preset (bytetrack=A0, botsort=B1)")
    p.add_argument("--preset", choices=sorted(PRESETS), default=None, help="explicit tracker grid row (see src/tracking/factory.py); overrides --tracker")
    p.add_argument("--save-detections", nargs="?", const="auto", default=None, metavar="PATH", help="save raw detections to a .npz (no PATH: runs/cache/<video>__w<sha8 of weights>__<model>__thr<X>.npz)")
    p.add_argument("--detections", type=Path, default=None, metavar="PATH", help="replay detections from a cache .npz instead of running the model")
    p.add_argument("--no-video", action="store_true", help="skip writing the annotated video (CSV and summary only)")
    p.add_argument("--show", action="store_true", help="open a window and watch the tracking live (q to stop)")
    args = p.parse_args()

    fine_tuned = args.weights is not None
    preset_id = args.preset or DEFAULT_PRESET[args.tracker]
    settings = PRESETS[preset_id]
    # one folder per preset so a grid loop never overwrites an earlier run
    out_dir = (args.out_dir or Path("runs/finetuned" if fine_tuned else "runs/baseline")) / preset_id
    stem = args.video.stem
    out_video = out_dir / f"{stem}_tracked.mp4"
    out_csv = out_dir / f"{stem}_tracks.csv"

    info = sv.VideoInfo.from_video_path(str(args.video))
    fps = float(info.fps)
    max_frames = int(args.max_seconds * fps) if args.max_seconds else info.total_frames
    class_list = [c.strip() for c in args.classes.split(",")] if fine_tuned else None
    sha = None
    if args.detections is not None and (args.weights is None or not args.weights.exists()):
        # replay without the checkpoint: trust the hash recorded in the cache
        sha = read_meta(args.detections).get("weights_sha256")
        print("Weights file not found; replaying without checking the checkpoint hash.")
    elif args.detections is not None:
        sha = weights_sha256(args.weights)  # checkpoint present: verify it matches the cache
    meta = make_meta(args.video, max_frames, fps, args.weights, args.model, args.threshold, class_list, sha)

    cached = None
    model = None
    if args.detections is not None:
        try:
            # (header truncation is the same for every preset, so presets always see equal frame counts)
            # container frame counts can over-report; a cache of the actual decoded frames is complete
            max_frames = min(max_frames, read_meta(args.detections)["n_frames"])
            cached = load_detections(args.detections, meta, max_frames)
        except ValueError as e:
            raise SystemExit(str(e))
        namer = make_namer(None, class_list)
        print(f"Replaying detections from {args.detections}")
    else:
        model = load_model(args.model, args.weights, args.trust_checkpoint)
        namer = make_namer(model, class_list)
        if fine_tuned:
            print(f"Loaded {args.weights} (RF-DETR {args.model}) with classes: {class_list}")

    save_meta = None
    if args.save_detections is not None and cached is None:
        save_path = cache_path(args.video, meta["weights_sha256"], args.model, args.threshold) if args.save_detections == "auto" else Path(args.save_detections)
        save_meta = (save_path, meta)

    print(f"Tracker preset {preset_id}: {settings.name}")
    summary = run(
        args.video, out_video, out_csv, model, namer, args.threshold, args.max_seconds, args.show,
        settings=settings, cached=cached, write_video=not args.no_video, save_cache_meta=save_meta,
    )

    out_json = out_csv.with_name(f"{stem}_summary.json")
    out_json.write_text(json.dumps({"preset": preset_id, "tracker": settings.name, **summary}, indent=2))

    print(f"Processed {summary['frames']} frames at {summary['fps']:.3f} fps")
    print(f"Unique player track IDs: {summary['unique_player_ids']}  (about 10 is ideal; far more = ID switches)")
    print(f"Ball detected in {summary['ball_seen_pct']}% of frames")
    print(f"Tracker time: {summary['tracker_ms_per_frame']} ms/frame")
    if save_meta is not None:
        print(f"Detections cache: {save_meta[0]}")
    print(f"Video: {'(skipped)' if args.no_video else out_video}\nCSV:   {out_csv}\nSummary: {out_json}")


if __name__ == "__main__":
    main()

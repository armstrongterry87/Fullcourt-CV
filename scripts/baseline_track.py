"""Detect and track players, referees, ball and rim in a game video.

Two modes:
  - Baseline (no --weights): RF-DETR pretrained on COCO, using its "person" and
    "sports ball" classes. Shows what off-the-shelf models get right.
  - Fine-tuned (--weights): your own trained RF-DETR checkpoint with the classes
    player / referee / ball / rim.

Tracking uses ByteTrack (default, preset A0) or BoT-SORT from Roboflow's `trackers` package;
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
from src.court.polygon import ConfigError, check_resolution, check_video_stem, court_keep_mask, load_court  # noqa: E402
from src.detection.filtering import BALL_NAMES, PLAYER_NAMES, REFEREE_NAMES, RIM_NAMES, apply_class_thresholds  # noqa: E402
from src.tracking.factory import DEFAULT_PRESET, PRESETS, PlayerTracker, TrackerSettings  # noqa: E402
from src.tracking.roles import RoleTally, rewrite_with_roles  # noqa: E402

CSV_HEADER = ["frame", "time_s", "class", "track_id", "conf", "x1_px", "y1_px", "x2_px", "y2_px"]


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


def id_labels(dets: sv.Detections, prefix: str = "") -> list[str]:
    """Text label per detection: '#<track id>' (with optional prefix), or just the prefix."""
    if dets.tracker_id is None:
        return [prefix.strip()] * len(dets)
    return [f"{prefix}#{int(t)}" if t >= 0 else prefix.strip() for t in dets.tracker_id]


class Annotator:
    """Draws players, referees, rim and ball onto a frame."""

    def __init__(self, fps: float) -> None:
        self.box = sv.BoxAnnotator(thickness=2)
        self.ref_box = sv.BoxAnnotator(thickness=2, color=sv.Color.YELLOW)
        self.rim_box = sv.BoxAnnotator(thickness=2, color=sv.Color.GREEN)
        self.label = sv.LabelAnnotator(text_scale=0.5, text_padding=3)
        self.trace = sv.TraceAnnotator(trace_length=int(fps))  # ~1 second trail
        self.ball_marker = sv.TriangleAnnotator(base=18, height=16)

    def draw(self, frame: np.ndarray, players, refs, ball, rims) -> np.ndarray:
        """Return an annotated copy of `frame`."""
        out = self.trace.annotate(frame.copy(), players)
        out = self.box.annotate(out, players)
        out = self.label.annotate(out, players, id_labels(players))
        out = self.ref_box.annotate(out, refs)
        out = self.label.annotate(out, refs, id_labels(refs, "REF "))
        out = self.rim_box.annotate(out, rims)
        return self.ball_marker.annotate(out, ball)


def render_video(video_path: Path, final_csv: Path, out_video: Path, max_frames: int | None = None) -> None:
    """Second pass: draw the finished CSV, so the video shows exactly what the eval counts.

    Players and referees are drawn by `role_track` (one role per track).
    """
    info = sv.VideoInfo.from_video_path(str(video_path))
    by_frame: dict[int, list[dict]] = {}
    with open(final_csv, newline="") as f:
        for r in csv.DictReader(f):
            by_frame.setdefault(int(r["frame"]), []).append(r)
    if by_frame and "role_track" not in next(iter(by_frame.values()))[0]:
        raise SystemExit(f"{final_csv} has no role_track column (made before task 002); re-run to regenerate it.")
    last = (max(by_frame) if by_frame else -1) if max_frames is None else max_frames - 1
    ann = Annotator(float(info.fps))

    def dets(rows: list[dict]) -> sv.Detections:
        if not rows:
            return sv.Detections.empty()
        return sv.Detections(
            xyxy=np.array([[r["x1_px"], r["y1_px"], r["x2_px"], r["y2_px"]] for r in rows], dtype=np.float32),
            class_id=np.zeros(len(rows), dtype=int),
            tracker_id=np.array([int(r["track_id"]) for r in rows], dtype=int),
        )

    out_video.parent.mkdir(parents=True, exist_ok=True)
    with sv.VideoSink(str(out_video), info) as sink:
        for i, frame in enumerate(sv.get_video_frames_generator(str(video_path))):
            if i > last:
                break
            rows = by_frame.get(i, [])
            role = lambda want: dets([r for r in rows if r["class"] in ("player", "referee") and r["role_track"] == want])  # noqa: E731
            cls = lambda want: dets([r for r in rows if r["class"] == want])  # noqa: E731
            sink.write_frame(ann.draw(frame, role("player"), role("referee"), cls("ball"), cls("rim")))


def run(
    video_path: Path,
    out_csv: Path,
    model,
    namer,
    threshold: float,
    max_seconds: float | None,
    show: bool = False,
    settings: TrackerSettings = PRESETS[DEFAULT_PRESET["bytetrack"]],
    cached: list[sv.Detections] | None = None,
    save_cache_meta: tuple[Path, dict] | None = None,
    person_threshold: float | None = None,
    ball_threshold: float | None = None,
    rim_threshold: float | None = None,
    court=None,
    court_margin_px: float = 20.0,
    allow_court_stem_mismatch: bool = False,
) -> dict:
    """Detect (or replay), threshold per class, filter, track, and write the CSV.

    `threshold` is the detection floor and the rim threshold; person/ball thresholds default
    to it. Per-class thresholds are applied on both live and replay paths, so the cache stays
    raw. The annotated video is made afterwards by `render_video`. With show=True a window
    plays a per-frame-label live preview (press q to stop early).
    """
    person_threshold = threshold if person_threshold is None else person_threshold
    ball_threshold = threshold if ball_threshold is None else ball_threshold
    rim_threshold = threshold if rim_threshold is None else rim_threshold
    info = sv.VideoInfo.from_video_path(str(video_path))
    fps = float(info.fps)
    max_frames = int(max_seconds * fps) if max_seconds else info.total_frames
    if cached is not None:
        max_frames = min(max_frames, len(cached))
    if court is not None:
        check_resolution(court, info.resolution_wh[0], info.resolution_wh[1])
        if not allow_court_stem_mismatch:
            check_video_stem(court, video_path.stem)

    tracker = PlayerTracker(settings, fps)
    raw_per_frame: list[sv.Detections] = []
    track_secs = 0.0
    ann = Annotator(fps) if show else None

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    tmp_csv = out_csv.with_name(out_csv.name + ".tmp")
    frames_done, ball_frames, ball_multi = 0, 0, 0
    tally = RoleTally()
    frame_label_player_ids: set[int] = set()

    with open(tmp_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(CSV_HEADER)

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
            detections, names = apply_class_thresholds(detections, names, person_threshold, ball_threshold, rim_threshold)

            # players and referees are tracked together (both are people moving on court)
            people = pick(detections, names, PLAYER_NAMES | REFEREE_NAMES)
            people_names = names[np.isin(names, list(PLAYER_NAMES | REFEREE_NAMES))] if len(detections) else names
            ball = pick(detections, names, BALL_NAMES)
            rims = pick(detections, names, RIM_NAMES)
            ball_multi += int(len(ball) > 1)  # 2+ candidates, after thresholds, before the argmax dedup
            if len(ball) > 1:  # keep only the most confident ball
                ball = ball[[int(np.argmax(ball.confidence))]]

            if court is not None and len(people):
                # people only, before the tracker, so off-court spectators never start tracks
                keep = court_keep_mask(people.xyxy, court, i, court_margin_px)
                people, people_names = people[keep], people_names[keep]

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
            frame_label_player_ids.update(int(t) for t in players.tracker_id if t is not None and t >= 0)
            ball_frames += int(len(ball) > 0)

            t = i / fps
            for d_class, dets in (("player", players), ("referee", refs), ("ball", ball), ("rim", rims)):
                for k in range(len(dets)):
                    x1, y1, x2, y2 = dets.xyxy[k]
                    tid = int(dets.tracker_id[k]) if dets.tracker_id is not None else -1
                    conf = float(dets.confidence[k]) if dets.confidence is not None else ""
                    writer.writerow([i, f"{t:.3f}", d_class, tid, conf, f"{x1:.1f}", f"{y1:.1f}", f"{x2:.1f}", f"{y2:.1f}"])
                    tally.add(tid, d_class, conf if isinstance(conf, float) else 0.0)

            frames_done += 1
            print(f"  frame {frames_done}/{max_frames}", end="\r", flush=True)

            if show:
                import cv2

                cv2.imshow("Basketball CV - press q to stop (per-frame labels)", ann.draw(frame, players, refs, ball, rims))
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

    if show:
        import cv2

        cv2.destroyAllWindows()
    print()

    roles = tally.roles()
    rewrite_with_roles(tmp_csv, out_csv, roles)
    tmp_csv.unlink()

    if save_cache_meta is not None:
        path, meta = save_cache_meta
        save_detections(path, raw_per_frame, {**meta, "n_frames": frames_done, "fps": fps})

    return {
        "frames": frames_done,
        "fps": fps,
        "unique_player_ids": len(frame_label_player_ids),
        "unique_player_ids_by_role": sum(1 for r in roles.values() if r == "player"),
        "tracks_with_both_labels": tally.tracks_with_both_labels(),
        "ball_seen_pct": round(100 * ball_frames / max(frames_done, 1), 1),
        "ball_multi_candidate_frames": ball_multi,
        "person_threshold": person_threshold,
        "ball_threshold": ball_threshold,
        "rim_threshold": rim_threshold,
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
    # Future work: a separate --rim-threshold and --floor flag (today --threshold is both the rim threshold and the floor fallback).
    p.add_argument("--threshold", type=float, default=0.4, help="rim threshold, and the default for --person-threshold / --ball-threshold")
    p.add_argument("--person-threshold", type=float, default=None, help="min confidence for players/referees (default: --threshold)")
    p.add_argument("--ball-threshold", type=float, default=None, help="min confidence for the ball (default: --threshold)")
    p.add_argument("--court", type=Path, default=None, help="court polygon JSON (see src/court/polygon.py); drops people whose feet are outside it")
    p.add_argument("--allow-court-stem-mismatch", action="store_true", help="allow a court polygon drawn on a different video file (same camera position)")
    p.add_argument("--court-margin-px", type=float, default=20.0, help="pixels outside the court polygon still kept (native pixels)")
    p.add_argument("--max-seconds", type=float, default=None, help="only process the first N seconds")
    p.add_argument("--tracker", choices=["bytetrack", "botsort"], default="bytetrack", help="tracker; picks its default preset (bytetrack=A0 [default], botsort=B1)")
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
    # detection floor = lowest threshold in use; the cache and the model run at this floor
    floor = min(x for x in (args.person_threshold, args.ball_threshold, args.threshold) if x is not None)
    meta = make_meta(args.video, max_frames, fps, args.weights, args.model, floor, class_list, sha)
    court = load_court(args.court) if args.court else None

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
        save_path = cache_path(args.video, meta["weights_sha256"], args.model, floor) if args.save_detections == "auto" else Path(args.save_detections)
        save_meta = (save_path, meta)

    print(f"Tracker preset {preset_id}: {settings.name}")
    try:
        summary = run(
            args.video, out_csv, model, namer, floor, args.max_seconds, args.show,
            settings=settings, cached=cached, save_cache_meta=save_meta,
            person_threshold=args.person_threshold if args.person_threshold is not None else args.threshold,
            ball_threshold=args.ball_threshold if args.ball_threshold is not None else args.threshold,
            rim_threshold=args.threshold, court=court, court_margin_px=args.court_margin_px,
            allow_court_stem_mismatch=args.allow_court_stem_mismatch,
        )
    except ConfigError as e:  # only expected setup errors; other ValueErrors keep their traceback
        raise SystemExit(str(e))
    if not args.no_video:
        print("Rendering video from the final CSV ...")
        render_video(args.video, out_csv, out_video, summary["frames"])

    out_json = out_csv.with_name(f"{stem}_summary.json")
    out_json.write_text(json.dumps({"preset": preset_id, "tracker": settings.name, **summary}, indent=2))

    print(f"Processed {summary['frames']} frames at {summary['fps']:.3f} fps")
    print(f"Unique player track IDs: {summary['unique_player_ids']} by per-frame label, {summary['unique_player_ids_by_role']} by role vote  (about 10 is ideal)")
    print(f"Ball detected in {summary['ball_seen_pct']}% of frames")
    print(f"Tracker time: {summary['tracker_ms_per_frame']} ms/frame")
    if save_meta is not None:
        print(f"Detections cache: {save_meta[0]}")
    print(f"Video: {'(skipped)' if args.no_video else out_video}\nCSV:   {out_csv}\nSummary: {out_json}")


if __name__ == "__main__":
    main()

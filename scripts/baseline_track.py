"""Baseline: detect and track players + ball in a game video, with NO training.

Uses RF-DETR pretrained on COCO ("person" and "sports ball" classes) and
ByteTrack from Roboflow's `trackers` package. This shows what off-the-shelf
models already get right on your footage, before any fine-tuning.

Outputs:
  - an annotated video (boxes, track IDs, short motion trails)
  - a CSV with one row per detection per frame

Example:
  python scripts/baseline_track.py --video data/raw/game1.mp4 --max-seconds 120
  python scripts/baseline_track.py --video data/raw/game1.mp4 --show   # watch it live

Install first:
  pip install rfdetr trackers supervision opencv-python
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import supervision as sv

PERSON = "person"
BALL = "sports ball"


def load_model(size: str):
    """Load a COCO-pretrained RF-DETR model. Nano/Small/Medium/Large are Apache-2.0."""
    import rfdetr

    classes = {
        "nano": rfdetr.RFDETRNano,
        "small": rfdetr.RFDETRSmall,
        "medium": rfdetr.RFDETRMedium,
        "large": rfdetr.RFDETRLarge,
    }
    return classes[size]()


def class_names(detections: sv.Detections) -> list[str]:
    """Class name for each detection: from the checkpoint if present, else COCO."""
    names = detections.data.get("class_name") if detections.data else None
    if names is not None and len(names) == len(detections):
        return [str(n) for n in names]
    from rfdetr.assets.coco_classes import COCO_CLASSES

    return [COCO_CLASSES.get(int(c), str(c)) for c in detections.class_id]


def run(
    video_path: Path,
    out_video: Path,
    out_csv: Path,
    model,
    threshold: float,
    max_seconds: float | None,
    show: bool = False,
) -> dict:
    """Detect, track and annotate every frame. Returns a short summary.

    With show=True, a window plays the annotated frames live (press q to stop early).
    """
    from trackers import ByteTrackTracker

    info = sv.VideoInfo.from_video_path(str(video_path))
    fps = float(info.fps)
    max_frames = int(max_seconds * fps) if max_seconds else info.total_frames

    tracker = ByteTrackTracker(frame_rate=fps)
    box = sv.BoxAnnotator(thickness=2)
    label = sv.LabelAnnotator(text_scale=0.5, text_padding=3)
    trace = sv.TraceAnnotator(trace_length=int(fps))  # ~1 second trail
    ball_marker = sv.TriangleAnnotator(base=18, height=16)

    out_video.parent.mkdir(parents=True, exist_ok=True)
    frames_done, ball_frames, track_ids = 0, 0, set()

    with sv.VideoSink(str(out_video), info) as sink, open(out_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["frame", "time_s", "class", "track_id", "conf", "x1_px", "y1_px", "x2_px", "y2_px"])

        for i, frame in enumerate(sv.get_video_frames_generator(str(video_path))):
            if i >= max_frames:
                break
            # frames from OpenCV are BGR; RF-DETR expects RGB
            detections = model.predict(frame[:, :, ::-1].copy(), threshold=threshold)
            names = np.array(class_names(detections))

            players = detections[names == PERSON] if len(detections) else detections
            ball = detections[names == BALL] if len(detections) else detections
            if len(ball) > 1:  # keep only the most confident ball
                ball = ball[[int(np.argmax(ball.confidence))]]

            players = tracker.update(players)
            track_ids.update(int(t) for t in players.tracker_id if t is not None and t >= 0)
            ball_frames += int(len(ball) > 0)

            t = i / fps
            for d_class, dets in ((PERSON, players), (BALL, ball)):
                for k in range(len(dets)):
                    x1, y1, x2, y2 = dets.xyxy[k]
                    tid = int(dets.tracker_id[k]) if dets.tracker_id is not None else -1
                    conf = float(dets.confidence[k]) if dets.confidence is not None else ""
                    writer.writerow([i, f"{t:.3f}", d_class, tid, conf, f"{x1:.1f}", f"{y1:.1f}", f"{x2:.1f}", f"{y2:.1f}"])

            labels = [f"#{int(t)}" if t >= 0 else "" for t in players.tracker_id] if players.tracker_id is not None else []
            annotated = trace.annotate(frame.copy(), players)
            annotated = box.annotate(annotated, players)
            annotated = label.annotate(annotated, players, labels)
            annotated = ball_marker.annotate(annotated, ball)
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

    return {
        "frames": frames_done,
        "fps": fps,
        "unique_player_ids": len(track_ids),
        "ball_seen_pct": round(100 * ball_frames / max(frames_done, 1), 1),
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--video", type=Path, required=True, help="input game video")
    p.add_argument("--out-dir", type=Path, default=Path("runs/baseline"), help="where outputs go (not committed)")
    p.add_argument("--model", choices=["nano", "small", "medium", "large"], default="small")
    p.add_argument("--threshold", type=float, default=0.4, help="detection confidence threshold")
    p.add_argument("--max-seconds", type=float, default=None, help="only process the first N seconds")
    p.add_argument("--show", action="store_true", help="open a window and watch the tracking live (q to stop)")
    args = p.parse_args()

    stem = args.video.stem
    out_video = args.out_dir / f"{stem}_tracked.mp4"
    out_csv = args.out_dir / f"{stem}_tracks.csv"

    summary = run(args.video, out_video, out_csv, load_model(args.model), args.threshold, args.max_seconds, args.show)

    print(f"Processed {summary['frames']} frames at {summary['fps']:.1f} fps")
    print(f"Unique player track IDs: {summary['unique_player_ids']}  (10 players + refs is ideal; far more = ID switches)")
    print(f"Ball detected in {summary['ball_seen_pct']}% of frames")
    print(f"Video: {out_video}\nCSV:   {out_csv}")


if __name__ == "__main__":
    main()

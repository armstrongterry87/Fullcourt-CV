"""One role per track: confidence-weighted vote over a track's per-frame labels.

Per-frame labels flicker between "player" and "referee". After the run, each track
(track_id >= 0) gets the label with the larger summed confidence; ties go to the label
with more frames, then to "player". A track with no player-labelled frames therefore
comes out as a referee. Rows with track_id == -1, and ball/rim rows, are never pooled:
their role_track equals their per-frame label.
"""

from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

VOTE_LABELS = ("player", "referee")


class RoleTally:
    """Accumulates per-track confidence sums and frame counts by label."""

    def __init__(self) -> None:
        self.conf: dict[int, dict[str, float]] = defaultdict(lambda: defaultdict(float))
        self.frames: dict[int, dict[str, int]] = defaultdict(lambda: defaultdict(int))

    def add(self, track_id: int, label: str, conf: float) -> None:
        """Record one detection row. Ignores track_id < 0 and non player/referee labels."""
        if track_id < 0 or label not in VOTE_LABELS:
            return
        self.conf[track_id][label] += conf
        self.frames[track_id][label] += 1

    def roles(self) -> dict[int, str]:
        """Final role for every tallied track."""
        out = {}
        for tid in self.conf:
            c, n = self.conf[tid], self.frames[tid]
            key = lambda lab: (c.get(lab, 0.0), n.get(lab, 0))  # noqa: E731
            p, r = key("player"), key("referee")
            out[tid] = "referee" if r > p else "player"  # tie -> player
        return out

    def tracks_with_both_labels(self) -> int:
        """Tracks that had at least one frame of each label."""
        return sum(1 for tid in self.frames if all(self.frames[tid].get(l, 0) for l in VOTE_LABELS))


def rewrite_with_roles(tmp_csv: Path, out_csv: Path, roles: dict[int, str]) -> None:
    """Copy tmp_csv to out_csv, appending label_frame (= class) and role_track columns.

    The nine original columns are copied verbatim; `class` keeps its per-frame meaning.
    `class` and `label_frame` are the same per-frame label (kept as two columns by request);
    `role_track` is the per-track voted role. Use `role_track` for per-player stats.
    """
    with open(tmp_csv, newline="") as fin, open(out_csv, "w", newline="") as fout:
        reader, writer = csv.reader(fin), csv.writer(fout)
        header = next(reader)
        writer.writerow(header + ["label_frame", "role_track"])
        for row in reader:
            label, tid = row[2], int(row[3])
            writer.writerow(row + [label, roles.get(tid, label) if tid >= 0 and label in VOTE_LABELS else label])

"""Unit tests for task 002: cache floor guard, class thresholds, role vote, court filter."""

import json
from pathlib import Path

import numpy as np
import pytest
import supervision as sv

from src.court.polygon import check_resolution, court_keep_mask, load_court, save_court
from src.detection.cache import load_detections, make_meta, save_detections
from src.detection.filtering import apply_class_thresholds
from src.tracking.roles import RoleTally, rewrite_with_roles


def _cache(tmp_path, threshold=0.25, sha="abc"):
    meta = make_meta(Path("v.mp4"), 2, 30.0, None, "small", threshold, ["ball", "player"], sha)
    d = sv.Detections(xyxy=np.array([[0, 0, 1, 1]], dtype=np.float32), confidence=np.array([0.5], dtype=np.float32), class_id=np.array([2]))
    path = tmp_path / "c.npz"
    save_detections(path, [d, d], {**meta, "n_frames": 2})
    return path


def test_cache_floor_lower_ok_and_higher_rejected(tmp_path):
    p = _cache(tmp_path, 0.25)
    ok = make_meta(Path("v.mp4"), 2, 30.0, None, "small", 0.40, ["ball", "player"], "abc")
    assert len(load_detections(p, ok, 2)) == 2
    p40 = _cache(tmp_path, 0.40)
    req = make_meta(Path("v.mp4"), 2, 30.0, None, "small", 0.25, ["ball", "player"], "abc")
    with pytest.raises(ValueError, match="floor"):
        load_detections(p40, req, 2)


def test_cache_sha_mismatch_rejected(tmp_path):
    p = _cache(tmp_path, 0.25, sha="abc")
    req = make_meta(Path("v.mp4"), 2, 30.0, None, "small", 0.25, ["ball", "player"], "zzz")
    with pytest.raises(ValueError, match="weights_sha256"):
        load_detections(p, req, 2)


def test_class_thresholds_strict_and_per_class():
    d = sv.Detections(
        xyxy=np.zeros((5, 4), dtype=np.float32),
        confidence=np.array([0.40, 0.41, 0.30, 0.50, 0.40], dtype=np.float32),
        class_id=np.arange(5),
    )
    names = np.array(["player", "player", "ball", "rim", "ball"])
    out, n = apply_class_thresholds(d, names, 0.40, 0.25, 0.45)
    assert list(n) == ["player", "ball", "rim", "ball"]  # 0.40 player dropped (strict), 0.41 kept


def _tally(rows):
    t = RoleTally()
    for tid, lab, c in rows:
        t.add(tid, lab, c)
    return t


def test_vote_confidence_sum_beats_frame_count():
    t = _tally([(1, "player", 0.9), (1, "referee", 0.5), (1, "referee", 0.5), (1, "referee", 0.5)])
    assert t.roles()[1] == "referee"  # 1.5 > 0.9
    assert t.tracks_with_both_labels() == 1


def test_vote_tie_goes_to_player_and_zero_player_is_referee():
    t = _tally([(1, "player", 0.5), (1, "referee", 0.5), (2, "referee", 0.6), (-1, "referee", 0.9)])
    r = t.roles()
    assert r[1] == "player" and r[2] == "referee" and -1 not in r


def test_rewrite_keeps_columns_and_minus_one_rows(tmp_path):
    tmp, out = tmp_path / "a.csv", tmp_path / "b.csv"
    tmp.write_text("frame,time_s,class,track_id,conf,x1_px,y1_px,x2_px,y2_px\n0,0.0,player,1,0.9,1,2,3,4\n0,0.0,referee,-1,0.9,1,2,3,4\n0,0.0,ball,-1,0.9,1,2,3,4\n1,0.0,referee,1,0.5,1,2,3,4\n")
    rewrite_with_roles(tmp, out, {1: "player"})
    lines = out.read_text().splitlines()
    assert lines[0].endswith(",label_frame,role_track")
    rows = [l.split(",") for l in lines[1:]]
    assert [r[-2:] for r in rows] == [["player", "player"], ["referee", "referee"], ["ball", "ball"], ["referee", "player"]]
    assert rows[3][:9] == ["1", "0.0", "referee", "1", "0.5", "1", "2", "3", "4"]


def _court(valid=None):
    return load_court_from(valid)


def load_court_from(valid):
    from src.court.polygon import Court
    return Court("v", 0, 100, 100, [[0, 0], [100, 0], [100, 50], [0, 50]], valid)


def test_court_feet_vs_centre_and_margin():
    c = _court()
    # tall box: centre (y=40) inside, feet (y=90) 40 px below the polygon edge at y=50
    box = np.array([[40, -10, 60, 90]], dtype=float)
    assert not court_keep_mask(box, c, 0, margin_px=20)[0]  # feet outside beyond margin
    centre_box = np.array([[40, -10, 60, 90]], dtype=float)
    assert court_keep_mask(centre_box, c, 0, margin_px=45)[0]  # margin widens it
    near = np.array([[40, 0, 60, 60]], dtype=float)  # feet 10 px outside
    assert court_keep_mask(near, c, 0, margin_px=20)[0]
    assert not court_keep_mask(near, c, 0, margin_px=5)[0]


def test_court_valid_frames_and_resolution(tmp_path):
    c = _court((10, 20))
    far = np.array([[40, 0, 60, 99]], dtype=float)
    assert not court_keep_mask(far, c, 15, 0)[0]
    assert court_keep_mask(far, c, 25, 0)[0]  # outside valid frames: pass through
    for f, kept in ((9, True), (10, False), (20, False), (21, True)):  # inclusive [10, 20]
        assert court_keep_mask(far, c, f, 0)[0] == kept, f
    with pytest.raises(ValueError, match="1920x1080"):
        check_resolution(c, 1920, 1080)
    p = tmp_path / "c.json"
    save_court(p, "v", 0, 100, 100, [[0, 0], [1, 0], [1, 1]], (3, 4))
    assert load_court(p).valid_frames == (3, 4)
    assert set(json.loads(p.read_text())) == {"video_stem", "frame_index", "frame_width", "frame_height", "polygon_px", "valid_frames"}


def test_court_stem_mismatch_and_short_polygon(tmp_path):
    from src.court.polygon import ConfigError, check_video_stem
    c = _court()  # video_stem "v"
    check_video_stem(c, "v")
    with pytest.raises(ConfigError, match="gym_b"):
        check_video_stem(c, "gym_b")
    p = tmp_path / "bad.json"
    p.write_text(json.dumps({"video_stem": "v", "frame_index": 0, "frame_width": 1, "frame_height": 1,
                             "polygon_px": [[0, 0], [1, 1]], "valid_frames": None}))
    with pytest.raises(ValueError, match="at least 3"):
        load_court(p)

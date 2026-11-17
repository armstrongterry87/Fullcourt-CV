"""Synthetic tests for eval/score_shots.py (design 004 section 8.3)."""
import importlib.util
from pathlib import Path

_p = Path(__file__).resolve().parents[1] / "eval" / "score_shots.py"
_spec = importlib.util.spec_from_file_location("score_shots", _p)
ss = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ss)


def R(t, result, pts, notes=""):
    return {"time_s": t, "result": result, "points": pts, "notes": notes}


def test_perfect():
    rows = [R(1, "make", 3), R(5, "miss", 2), R(9, "make", 1)]
    lab = [R(1.3, "make", 3), R(5.3, "miss", 2), R(9.3, "make", 1)]
    s = ss.score(rows, lab)
    for k in ("attempt_precision", "attempt_recall", "makemiss_accuracy", "makemiss_coverage",
              "makemiss_strict_accuracy", "twothree_accuracy", "twothree_coverage",
              "ft_precision", "ft_recall"):
        assert s[k] == 1.0, k
    assert s["points_error_signed"] == 0
    assert abs(s["median_time_offset_s"] - 0.3) < 1e-9


def test_missed_three_counts_for_23_not_points():
    s = ss.score([R(1, "miss", 3)], [R(1, "miss", 3)])
    assert s["twothree_accuracy"] == 1.0
    assert s["points_pred"] == 0 and s["points_label"] == 0


def test_unknown():
    pred = [R(1, "unknown", None), R(5, "make", 2)]
    lab = [R(1, "make", 2), R(5, "make", 2)]
    s = ss.score(pred, lab)
    assert s["attempt_precision"] == 1.0
    assert s["makemiss_accuracy"] == 1.0
    assert s["makemiss_coverage"] == 0.5
    assert s["makemiss_strict_accuracy"] == 0.5
    assert s["twothree_coverage"] == 0.5


def test_tolerance():
    s = ss.score([R(0.9, "make", 2)], [R(0, "make", 2)])
    assert s["n_matched"] == 1
    s = ss.score([R(1.1, "make", 2)], [R(0, "make", 2)])
    assert s["n_matched"] == 0
    assert s["attempt_precision"] == 0.0 and s["attempt_recall"] == 0.0


def test_competition_nearer_wins():
    s = ss.score([R(0.8, "make", 2), R(0.1, "miss", 2)], [R(0, "miss", 2)])
    assert s["n_matched"] == 1
    assert s["makemiss_accuracy"] == 1.0
    assert s["attempt_precision"] == 0.5


def test_empty():
    one = [R(1, "make", 2)]
    for p, l in (([], one), (one, []), ([], [])):
        s = ss.score(p, l)
        assert s["makemiss_accuracy"] == "n/a"
        assert s["twothree_accuracy"] == "n/a"
        assert s["median_time_offset_s"] == "n/a"
        ss.format_report(s)
    assert ss.score([], one)["attempt_recall"] == 0.0
    assert ss.score([], one)["attempt_precision"] == "n/a"
    assert ss.score(one, [])["attempt_recall"] == "n/a"


def test_empty_files(tmp_path):
    f = tmp_path / "e.csv"
    f.write_text("")
    g = tmp_path / "h.csv"
    g.write_text("frame,time_s,result,points,notes\n")
    assert ss.read_rows(f) == [] and ss.read_rows(g) == []
    assert ss.read_rows(tmp_path / "missing.csv") == []


def test_points_error_and_unknown_value():
    s = ss.score([R(1, "make", 2)], [R(1, "make", 3), R(9, "make", 3)])
    assert s["points_error_signed"] == -4 and s["points_error_abs"] == 4
    pred = [R(1, "make", 3), R(5, "make", None)]
    lab = [R(1, "make", 2), R(5, "make", 3)]
    s = ss.score(pred, lab)
    assert s["makes_with_unknown_value"] == 1
    assert s["points_error_signed"] == -2


def test_ft_and_codes():
    pred = [R(1, "make", 1, "ft_rule;near_arc"), R(5, "make", 2, "near_arc")]
    lab = [R(1, "make", 1), R(5, "make", 1)]
    s = ss.score(pred, lab)
    assert s["ft_precision"] == 1.0 and s["ft_recall"] == 0.5
    assert s["note_codes"]["near_arc"] == 2


# ---- input validation: invalid rows are dropped AND counted, never absorbed ----
HDR = "time_s,result,points,notes\n"


def _w(tmp_path, name, body):
    f = tmp_path / name
    f.write_text(HDR + body, encoding="utf-8")
    return f


def test_blank_time_counted(tmp_path):
    rows = ss.read_rows(_w(tmp_path, "l.csv", ",make,3,\n5,miss,2,\n"))
    assert len(rows) == 1 and len(rows.invalid) == 1 and "blank" in rows.invalid[0][1]


def test_unparseable_time_counted(tmp_path):
    rows = ss.read_rows(_w(tmp_path, "l.csv", "1:05,make,3,\n5,miss,2,\n"))
    assert len(rows) == 1 and len(rows.invalid) == 1 and "unparseable" in rows.invalid[0][1]


def test_typo_result_counted_not_nonmake(tmp_path):
    rows = ss.read_rows(_w(tmp_path, "l.csv", "1,made,3,\n5,miss,2,\n"))
    assert [r["time_s"] for r in rows] == [5.0] and len(rows.invalid) == 1


def test_unknown_allowed_in_pred_only(tmp_path):
    f = _w(tmp_path, "p.csv", "1,unknown,\n")
    assert len(ss.read_rows(f, "pred")) == 1
    assert len(ss.read_rows(f, "label")) == 0


def test_bad_points_counted(tmp_path):
    rows = ss.read_rows(_w(tmp_path, "l.csv", "1,make,two,\n2,make,4,\n3,make,,\n4,miss,3,\n"))
    assert len(rows) == 2 and len(rows.invalid) == 2
    assert [r["points"] for r in rows] == [None, 3]


def test_mixed_file_metrics_over_valid_rows_and_header(tmp_path):
    lab = ss.read_rows(_w(tmp_path, "l.csv", "1,make,3,\n,miss,2,\n9,made,2,\n20,miss,2,\n"))
    pred = ss.read_rows(_w(tmp_path, "p.csv", "1,make,3,\nabc,miss,2,\n"), "pred")
    s = ss.score(pred, lab)
    assert (s["n_label"], s["n_label_invalid"]) == (2, 2)
    assert (s["n_pred"], s["n_pred_invalid"]) == (1, 1)
    assert s["attempt_recall"] == 0.5 and s["attempt_precision"] == 1.0
    out = ss.format_report(s)
    assert "INPUT ERRORS" in out and "2 label row(s)" in out and "1 predicted row(s)" in out


def test_clean_file_header_is_quiet(tmp_path):
    f = _w(tmp_path, "l.csv", "1,make,3,\n")
    out = ss.format_report(ss.score(ss.read_rows(f, "pred"), ss.read_rows(f)))
    assert "INPUT ERRORS" not in out and "0 invalid" in out

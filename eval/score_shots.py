"""Score a predicted shots.csv against a hand-label CSV (design 004, section 8.3).

Usage: python eval/score_shots.py PRED.csv LABELS.csv [--tol 1.0]

IMPORTANT: `points` is the SHOT VALUE (1/2/3) on every row, misses included.
Points scored = points if result == "make" else 0.
Zero denominators report "n/a". No number here is an accuracy claim until
labelled real footage exists (Q3).
"""
from __future__ import annotations

import argparse
import csv
import statistics
from collections import Counter
from pathlib import Path

NA = "n/a"


def _num(s):
    s = (s or "").strip()
    if not s:
        return None
    try:
        return float(s)
    except ValueError:
        return None


LABEL_RESULTS = ("make", "miss")
PRED_RESULTS = ("make", "miss", "unknown")


class Rows(list):
    """List of valid rows; `.invalid` holds (line_no, reason) for every row dropped."""

    def __init__(self, *a):
        super().__init__(*a)
        self.invalid: list[tuple[int, str]] = []


def _parse_row(r, allowed):
    """Return (row, None) or (None, reason). Never guesses: bad input is rejected."""
    raw_t = (r.get("time_s") or "").strip()
    t = _num(raw_t)
    if t is None or t != t or t in (float("inf"), float("-inf")):
        return None, f"time_s {'blank' if not raw_t else 'unparseable: ' + repr(raw_t)}"
    res = (r.get("result") or "").strip().lower()
    if res not in allowed:
        return None, f"result {res!r} not in {'/'.join(allowed)}"
    raw_p = (r.get("points") or "").strip()
    pts = None
    if raw_p:
        v = _num(raw_p)
        if v is None or v not in (1.0, 2.0, 3.0):
            return None, f"points {raw_p!r} not in 1/2/3/empty"
        pts = int(v)
    return {"time_s": t, "result": res, "points": pts,
            "notes": (r.get("notes") or "").strip()}, None


def read_rows(path, kind: str = "label") -> Rows:
    """Read a shots CSV into valid rows (time_s, result, points int|None, notes).

    kind="label" allows result make/miss; kind="pred" also allows unknown.
    Invalid rows are DROPPED AND COUNTED (rows.invalid), never absorbed.
    A missing or empty file yields no rows and no invalid rows.
    """
    allowed = PRED_RESULTS if kind == "pred" else LABEL_RESULTS
    out = Rows()
    p = Path(path)
    if not p.exists() or p.stat().st_size == 0:
        return out
    with open(p, newline="", encoding="utf-8") as f:
        rd = csv.DictReader(f)
        for r in rd:
            row, why = _parse_row(r, allowed)
            if row is None:
                out.invalid.append((rd.line_num, why))
            else:
                out.append(row)
    return out


def match(pred: list[dict], label: list[dict], tol: float = 1.0) -> list[tuple[int, int]]:
    """Greedy one-to-one matching by smallest |dt|. Returns (pred_idx, label_idx)."""
    pairs = sorted(
        (abs(l["time_s"] - p["time_s"]), i, j)
        for i, p in enumerate(pred) for j, l in enumerate(label)
        if abs(l["time_s"] - p["time_s"]) <= tol
    )
    up, ul, out = set(), set(), []
    for _, i, j in pairs:
        if i in up or j in ul:
            continue
        up.add(i)
        ul.add(j)
        out.append((i, j))
    return sorted(out)


def _ratio(num, den):
    return num / den if den else NA


def _scored(row) -> int:
    """Points scored: shot value if make, else 0 (empty value counts 0)."""
    return (row["points"] or 0) if row["result"] == "make" else 0


def score(pred: list[dict], label: list[dict], tol: float = 1.0) -> dict:
    """Compute every design 8.3 metric over the rows given. Zero denominators give 'n/a'.

    Invalid-row counts/details are taken from `.invalid` when the inputs came from read_rows.
    """
    pred_bad = list(getattr(pred, "invalid", []))
    label_bad = list(getattr(label, "invalid", []))
    m = match(pred, label, tol)
    pairs = [(pred[i], label[j]) for i, j in m]
    known = [(p, l) for p, l in pairs if p["result"] in ("make", "miss")]
    mm_ok = sum(p["result"] == l["result"] for p, l in known)
    both23 = [(p, l) for p, l in pairs
              if p["points"] in (2, 3) and l["points"] in (2, 3)]
    lab23 = [(p, l) for p, l in pairs if l["points"] in (2, 3)]
    ft_tp = sum(p["points"] == 1 and l["points"] == 1 for p, l in pairs)
    ft_pred = sum(p["points"] == 1 for p, l in pairs)
    ft_lab = sum(l["points"] == 1 for p, l in pairs)
    pred_pts = sum(_scored(p) for p in pred)
    lab_pts = sum(_scored(l) for l in label)
    offs = [l["time_s"] - p["time_s"] for p, l in pairs]
    codes = Counter(c for p in pred for c in p["notes"].split(";") if c)
    return {
        "n_pred": len(pred), "n_label": len(label), "n_matched": len(m),
        "attempt_precision": _ratio(len(m), len(pred)),
        "attempt_recall": _ratio(len(m), len(label)),
        "attempt_f1": (NA if not m else 2 * len(m) / (len(pred) + len(label))),
        "makemiss_accuracy": _ratio(mm_ok, len(known)),
        "makemiss_coverage": _ratio(len(known), len(pairs)),
        "makemiss_strict_accuracy": _ratio(mm_ok, len(pairs)),
        "twothree_accuracy": _ratio(sum(p["points"] == l["points"] for p, l in both23), len(both23)),
        "twothree_coverage": _ratio(sum(p["points"] is not None for p, l in lab23), len(lab23)),
        "ft_precision": _ratio(ft_tp, ft_pred),
        "ft_recall": _ratio(ft_tp, ft_lab),
        "points_pred": pred_pts, "points_label": lab_pts,
        "points_error_signed": pred_pts - lab_pts,
        "points_error_abs": abs(pred_pts - lab_pts),
        "makes_with_unknown_value": sum(p["result"] == "make" and p["points"] is None for p in pred),
        "median_time_offset_s": (statistics.median(offs) if offs else NA),
        "note_codes": dict(codes.most_common()),
        "n_pred_invalid": len(pred_bad), "n_label_invalid": len(label_bad),
        "invalid_detail": {"pred": pred_bad, "label": label_bad},
    }


def format_report(s: dict) -> str:
    """Render metrics as text, with the time-offset check first."""
    def f(v):
        return v if isinstance(v, str) else (f"{v:.3f}" if isinstance(v, float) else str(v))
    lines = []
    npb, nlb = s.get("n_pred_invalid", 0), s.get("n_label_invalid", 0)
    if npb or nlb:
        lines += [
            "!" * 78,
            f"INPUT ERRORS - INCOMPLETE READ: {npb} predicted row(s) and {nlb} label row(s) were "
            "INVALID and DROPPED.",
            "Every metric below is over the VALID rows only. Dropped labels RAISE recall; "
            "fix the CSV and re-run before quoting any number.",
        ]
        for side, bad in s["invalid_detail"].items():
            for ln, why in bad[:10]:
                lines.append(f"  {side} line {ln}: {why}")
            if len(bad) > 10:
                lines.append(f"  {side}: ... and {len(bad) - 10} more")
        lines.append("!" * 78)
    else:
        lines.append("Input rows: 0 invalid (pred 0, labels 0).")
    lines += [
        "CHECK FIRST - median time offset (label - pred, s): " + f(s["median_time_offset_s"])
        + "  (expected ~ +0.3; near +1 or negative => labelling convention differs,"
        " recall understated, see Q13)",
        f"Rows: predicted={s['n_pred']} labels={s['n_label']} matched={s['n_matched']}",
        "| Metric | Value |", "|---|---|",
    ]
    for k, v in s.items():
        if k in ("note_codes", "n_pred", "n_label", "n_matched",
                 "n_pred_invalid", "n_label_invalid", "invalid_detail"):
            continue
        lines.append(f"| {k} | {f(v)} |")
    codes = ", ".join(f"{k}={v}" for k, v in s["note_codes"].items()) or "none"
    lines.append("Reason codes: " + codes)
    if s["n_label"] < 30:
        lines.append("WARNING: small sample (<30 labelled shots); not an accuracy claim.")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description="Score shots.csv against hand labels")
    ap.add_argument("pred")
    ap.add_argument("labels")
    ap.add_argument("--tol", type=float, default=1.0)
    a = ap.parse_args()
    s = score(read_rows(a.pred, "pred"), read_rows(a.labels, "label"), a.tol)
    print(format_report(s))
    if s["n_pred_invalid"] or s["n_label_invalid"]:
        raise SystemExit(2)  # drop-and-count, but a script can still see the partial read


if __name__ == "__main__":
    main()

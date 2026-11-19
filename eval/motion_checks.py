"""A3 physical checks for camera motion compensation (design 005 section 12.1). Evaluator-owned.

Reads: video, detection cache, motion cache, tracks CSV. Writes a JSON of results (default runs/a3_motion_checks.json).
Nothing here edits src/. No footage is written.

Usage:
  python eval/motion_checks.py --video data/raw/xbotgo.mp4 --det runs/cache/<det>.npz \
      --motion runs/cache/<motion>.npz --tracks runs/finetuned/xbotgo_tracks.csv --out runs/a3_motion_checks.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.camera.motion import (MotionConfig, build_mask, extract_features, match_homography, load_motion,  # noqa: E402
                               load_detection_boxes, transform_points, stretch_at, video_frames, _overlap, _corners)

RIM, BALL, PLAYER, REF = 4, 1, 2, 3   # class ids in the detection cache (checked against the tracks CSV counts)
# Hand-picked BEFORE any stabilised number was computed (see report). Standing/seated spectators by track id,
# controls = long-running on-court players.
STATIC_TRACKS = {"track34_standing_spectator": 34, "track35_standing_spectator": 35}
CONTROL_TRACKS = {"control_track6_player": 6, "control_track19_player": 19}
# Bench seats: (seed frame, seed feet x, seed feet y) read by eye from raw frames, followed through raw detections.
# gate = raw feet x range, y2 range; window = frames. A is seen through the slow pan back (x drifts 198 -> 245 in raw).
BENCH_GATES = {"bench_A": ((190, 250, 474, 488), (372, 640)), "bench_B": ((160, 178, 474, 488), (372, 540)),
               "bench_C": ((295, 310, 472, 482), (372, 540))}
# Scoreboards, centre of the LED panel read by eye from raw frames (+-5 px). A = frames 10/120 (and 250 left), B = 400+.
# Re-read on 3x zoomed crops (runs/a3_frames/scoreboards*.jpg) using the board's RIGHT EDGE, because the frame-250
# left board is cut off by the frame edge (a centre reading there is wrong). Frames 500 and 600 are edge = centre + 36.
SCOREBOARD = {"A": {10: (897, 255), 120: (928, 255), 250: (55, 248)},
              "B": {250: (997, 247), 400: (426, 262), 500: (417, 261), 600: (464, 263), 705: (554, 266)}}


def mad_p95(d: np.ndarray) -> dict:
    d = np.asarray(d, float)
    d = d[np.isfinite(d)]
    if len(d) == 0:
        return {"n": 0}
    return {"n": int(len(d)), "mad": float(np.median(d)), "p95": float(np.percentile(d, 95)), "max": float(d.max())}


def spread(pts: np.ndarray) -> dict:
    """Distance of each point to the median point: its median and 95th percentile."""
    pts = pts[np.all(np.isfinite(pts), axis=1)]
    if len(pts) == 0:
        return {"n": 0}
    return mad_p95(np.linalg.norm(pts - np.median(pts, axis=0), axis=1))


def rim_check(det, motion):
    fi, xyxy, conf, cid = det
    s = (cid == RIM) & (conf >= 0.4)
    f, b = fi[s], xyxy[s].astype(float)
    cen = np.column_stack([(b[:, 0] + b[:, 2]) / 2, (b[:, 1] + b[:, 3]) / 2])
    left = np.column_stack([b[:, 0], cen[:, 1]])
    right = np.column_stack([b[:, 2], cen[:, 1]])
    trunc = (b[:, 0] <= 2) | (b[:, 2] >= motion.frame_wh[0] - 2)
    W_raw = float(np.median(b[:, 2] - b[:, 0]))
    out = {"n_dets": int(s.sum()), "n_truncated_at_frame_edge": int(trunc.sum()), "W_raw_px": W_raw,
           "raw_x_range": [float(cen[:, 0].min()), float(cen[:, 0].max())], "views": {}}
    lab = DBSCAN(eps=W_raw, min_samples=10).fit_predict(cen)
    lab_raw = lab.copy()
    out["raw_clusters"] = [{"label": int(l), "n": int((lab == l).sum()), "x_med": float(np.median(cen[lab == l, 0])),
                            "y_med": float(np.median(cen[lab == l, 1])), "raw_spread": spread(cen[lab == l]),
                            "raw_spread_excl_trunc": spread(cen[(lab == l) & ~trunc])} for l in sorted(set(lab)) if l >= 0]
    out["raw_noise_points"] = int((lab < 0).sum())
    out["raw_all_spread_one_cluster"] = spread(cen)
    for v in range(len(motion.views)):
        c = motion.to_view(cen, f, v)
        l_ = motion.to_view(left, f, v)
        r_ = motion.to_view(right, f, v)
        ok = np.all(np.isfinite(c), axis=1)
        if ok.sum() == 0:
            out["views"][v] = {"n_usable": 0}
            continue
        Wv = float(np.median(np.linalg.norm(r_[ok] - l_[ok], axis=1)))
        lab = np.full(len(cen), -1)
        lab[ok] = DBSCAN(eps=Wv, min_samples=10).fit_predict(c[ok])
        cl = []
        for k in sorted(set(lab[ok])):
            if k < 0:
                continue
            m = lab == k
            if m.sum() < 20:
                continue
            med = np.median(c[m], axis=0)
            d = np.linalg.norm(c[m] - med, axis=1)
            d_nt = np.linalg.norm(c[m & ~trunc] - med, axis=1)
            st = [stretch_at(motion.views[v].H_ref_to_view @ motion.H_frame_to_ref[ff], tuple(cc))
                  for ff, cc in zip(f[m], cen[m])]
            raw_in = spread(cen[m])
            cl.append({"cluster": int(k), "n": int(m.sum()), "n_truncated": int((m & trunc).sum()),
                       "centre_view_px": med.round(1).tolist(),
                       "raw_x_range_of_members": [float(cen[m, 0].min()), float(cen[m, 0].max())],
                       "W_view_px": Wv, "stab": mad_p95(d), "stab_excl_trunc": mad_p95(d_nt), "raw_same_dets": raw_in,
                       "mad_in_W": float(np.median(d) / Wv), "p95_in_W": float(np.percentile(d, 95) / Wv),
                       "p95_ratio_vs_raw": float(np.percentile(d, 95) / raw_in["p95"]) if raw_in["p95"] > 0 else None,
                       "max_stretch_at_rim": float(np.max(st)), "median_stretch_at_rim": float(np.median(st)),
                       "frames": [int(f[m].min()), int(f[m].max())]})
        out["views"][v] = {"n_usable": int(ok.sum()), "n_noise": int((lab[ok] < 0).sum()), "W_view_px": Wv, "clusters": cl}
    # one rim or two? map each raw cluster into view 0 and view 1 WITHOUT the usability gate (stretch may be large)
    one_two = []
    for c_ in out["raw_clusters"]:
        m = lab_raw == c_["label"]
        row = {"raw_cluster": c_["label"], "frames": [int(f[m].min()), int(f[m].max())]}
        for v in range(len(motion.views)):
            pts = []
            for ff_, cc_ in zip(f[m], cen[m]):
                H = motion.H_frame_to_ref[ff_]
                if np.all(np.isfinite(H)):
                    pts.append(transform_points(motion.views[v].H_ref_to_view @ H, cc_[None])[0])
            pts = np.array(pts)
            row[f"view{v}_median_px"] = np.nanmedian(pts, axis=0).round(1).tolist() if len(pts) else None
        one_two.append(row)
    out["rim_clusters_in_each_view_no_usability_gate"] = one_two
    order = np.argsort(f, kind="stable")
    ff, cc = f[order], cen[order]
    cons = np.diff(ff) == 1
    dr = np.linalg.norm(np.diff(cc, axis=0), axis=1)[cons]
    out["noise_floor_raw_consecutive_px"] = {"n": int(cons.sum()), "median": float(np.median(dr)) if len(dr) else None,
                                            "p95": float(np.percentile(dr, 95)) if len(dr) else None}
    # same, in view 0, between consecutive detections (camera motion removed)
    c0 = motion.to_view(cc, ff, 0)
    d0 = np.linalg.norm(np.diff(c0, axis=0), axis=1)[cons]
    d0 = d0[np.isfinite(d0)]
    out["noise_floor_view0_consecutive_px"] = {"n": int(len(d0)), "median": float(np.median(d0)) if len(d0) else None,
                                              "p95": float(np.percentile(d0, 95)) if len(d0) else None}
    return out


def follow(det, gate, window):
    """Seated-person detections by a RAW-pixel gate read by eye from the raw frames and the raw detection list.

    gate = (x_lo, x_hi, y2_lo, y2_hi) on the feet point. (A first attempt that followed a seed by nearest neighbour
    made seats A and B collapse onto the same detections, so it was replaced by fixed gates.)
    """
    fi, xyxy, conf, cid = det
    feet = np.column_stack([(xyxy[:, 0] + xyxy[:, 2]) / 2, xyxy[:, 3]])
    x_lo, x_hi, y_lo, y_hi = gate
    ok = (((cid == PLAYER) | (cid == REF)) & ((xyxy[:, 3] - xyxy[:, 1]) < 100) & (conf >= 0.25)
          & (fi >= window[0]) & (fi <= window[1]) & (feet[:, 0] >= x_lo) & (feet[:, 0] <= x_hi)
          & (feet[:, 1] >= y_lo) & (feet[:, 1] <= y_hi))
    res = {}
    for i in np.where(ok)[0]:
        res.setdefault(int(fi[i]), feet[i])   # one detection per frame (first)
    return res


def people_check(det, motion, tracks):
    people = {}
    for name, tid in {**STATIC_TRACKS, **CONTROL_TRACKS}.items():
        t = tracks[(tracks.track_id == tid) & tracks["class"].isin(["player", "referee"])]
        people[name] = (t.frame.to_numpy(), np.column_stack([(t.x1_px + t.x2_px) / 2, t.y2_px]).astype(float), tid)
    for name, (gate, win) in BENCH_GATES.items():
        r = follow(det, gate, win)
        fr = np.array(sorted(r))
        people[name] = (fr, np.array([r[k] for k in fr]) if len(fr) else np.zeros((0, 2)), None)
    out = {}
    for name, (fr, feet, tid) in people.items():
        if len(fr) == 0:
            out[name] = {"n_frames": 0}
            continue
        d = {"track_id": tid, "n_frames": int(len(fr)), "frames": [int(fr.min()), int(fr.max())],
             "raw": spread(feet), "views": {}}
        for v in range(len(motion.views)):
            p = motion.to_view(feet, fr, v)
            if np.isfinite(p[:, 0]).sum():
                okp = np.isfinite(p[:, 0])
                sts = [stretch_at(motion.views[v].H_ref_to_view @ motion.H_frame_to_ref[int(f_)], tuple(pt))
                       for f_, pt in zip(fr[okp], feet[okp])]
                d["views"][v] = {"n": int(okp.sum()), "stab": spread(p), "median_stretch": float(np.median(sts)),
                                 "stab_p95_raw_equiv_px": spread(p).get("p95", float("nan")) / float(np.median(sts))}
        ok0 = np.isfinite(motion.to_view(feet, fr, 0)[:, 0])
        d["raw_same_frames_as_view0"] = spread(feet[ok0])
        d["raw_range_px"] = (feet.max(0) - feet.min(0)).round(0).tolist()
        out[name] = d
    return out


def independent_drift(video, det_boxes, motion, n_pairs=200, seed=0, min_overlap=0.5):
    cfg = MotionConfig()
    rng = np.random.default_rng(seed)
    W, Hh = motion.frame_wh
    n = len(motion.quality)
    Hf = motion.H_frame_to_ref
    good = np.where(np.all(np.isfinite(Hf.reshape(n, -1)), axis=1))[0]
    par, chain = motion.stats["key_parent"], motion.stats["key_chain_len"]

    def on_chain(k):
        while par[k] >= 0:
            if chain[k] > 0:
                return True
            k = par[k]
        return False

    kc = np.array([on_chain(k) for k in range(len(par))])
    kf = motion.stats["keyframe"]
    frame_chained = np.array([(kf[i] >= 0 and kc[kf[i]]) for i in range(n)])
    pairs, tries = [], 0
    while len(pairs) < n_pairs and tries < 200000:
        tries += 1
        a, b = rng.choice(good, 2, replace=False)
        if abs(int(a) - int(b)) < 2:
            continue
        if _overlap(np.linalg.inv(Hf[b]) @ Hf[a], (W, Hh)) >= min_overlap:
            pairs.append((int(a), int(b)))
    need = sorted({x for p in pairs for x in p})
    feats = {}
    for i, fr in enumerate(video_frames(video, max(need) + 1)):
        if i in need:
            m = build_mask(fr.shape[:2], det_boxes[i], cfg)
            feats[i] = extract_features(cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY), m, cfg)
    corners = _corners(W, Hh)
    cp = np.array([[W / 2, Hh / 2]])
    rows = []
    for a, b in pairs:
        fit = match_homography(feats[a], feats[b], (W, Hh), cfg)
        ok = fit.H_src_to_dst is not None and fit.n_inliers >= cfg.min_inliers
        row = {"a": a, "b": b, "gap": abs(a - b), "fit": bool(ok), "n_inliers": int(fit.n_inliers),
               "a_chain": bool(frame_chained[a]), "b_chain": bool(frame_chained[b]),
               "q_a": float(motion.quality[a]), "q_b": float(motion.quality[b])}
        if ok:
            comp = np.linalg.inv(Hf[b]) @ Hf[a]
            e = np.linalg.norm(transform_points(fit.H_src_to_dst, corners) - transform_points(comp, corners), axis=1)
            row["err_max_corner_px"] = float(np.nanmax(e))
            row["err_mean_corner_px"] = float(np.nanmean(e))
            row["err_centre_px"] = float(np.linalg.norm(transform_points(fit.H_src_to_dst, cp) - transform_points(comp, cp)))
        rows.append(row)
    df = pd.DataFrame(rows)

    def summ(d):
        d = d[d.fit]
        if len(d) == 0:
            return {"n_matched": 0}
        return {"n_matched": int(len(d)), "median_max_corner_px": float(d.err_max_corner_px.median()),
                "p95_max_corner_px": float(d.err_max_corner_px.quantile(0.95)),
                "max_max_corner_px": float(d.err_max_corner_px.max()),
                "median_mean_corner_px": float(d.err_mean_corner_px.median()),
                "p95_mean_corner_px": float(d.err_mean_corner_px.quantile(0.95)),
                "median_centre_px": float(d.err_centre_px.median()), "p95_centre_px": float(d.err_centre_px.quantile(0.95))}

    chained = df.a_chain | df.b_chain
    both = df.a_chain & df.b_chain
    out = {"n_pairs_requested": n_pairs, "n_pairs_sampled": len(pairs), "n_direct_fit_failed": int((~df.fit).sum()),
           "median_gap_frames": float(df.gap.median()), "frac_gap_over_100": float((df.gap > 100).mean()),
           "all": summ(df), "neither_on_chained_subtree": summ(df[~chained]),
           "one_on_chained_subtree": summ(df[chained & ~both]), "both_on_chained_subtree": summ(df[both]),
           "any_on_chained_subtree": summ(df[chained]),
           "gap_le_30": summ(df[df.gap <= 30]), "gap_gt_100": summ(df[df.gap > 100]),
           "frames_on_chained_subtree": int(frame_chained[good].sum()), "n_good_frames": int(len(good)),
           "n_failed_by_group": {"neither": int(((~df.fit) & ~chained).sum()), "any_chained": int(((~df.fit) & chained).sum())}}
    df.to_csv(ROOT / "runs" / "a3_drift_pairs.csv", index=False)
    worst = df[df.fit].sort_values("err_max_corner_px", ascending=False).head(8)
    out["worst_pairs"] = worst[["a", "b", "gap", "a_chain", "b_chain", "n_inliers", "err_max_corner_px",
                                "err_centre_px"]].round(2).to_dict("records")
    return out


def coverage(motion, fps, max_gap_s=0.125):
    q = motion.quality
    n = len(q)
    unrel = q == 0
    runs, i = [], 0
    while i < n:
        if unrel[i]:
            j = i
            while j + 1 < n and unrel[j + 1]:
                j += 1
            runs.append((i, j - i + 1))
            i = j + 1
        else:
            i += 1
    gap_n = max(1, round(max_gap_s * fps))
    ch, kf, s = motion.stats["chained"], motion.stats["keyframe"], motion.stats
    d6 = (q == 0.6) & (ch == 0)
    return {"n_frames": n, "share_0.9": float((q >= 0.9).mean()), "share_ge_0.6": float((q >= 0.6).mean()),
            "share_unreliable": float(unrel.mean()), "n_unreliable": int(unrel.sum()),
            "unreliable_runs": [{"start": a, "len": b} for a, b in runs],
            "longest_unreliable_frames": max([b for _, b in runs], default=0),
            "longest_unreliable_s": max([b for _, b in runs], default=0) / fps,
            "runs_longer_than_max_gap": int(sum(b > gap_n for _, b in runs)), "max_gap_frames": gap_n,
            "n_chained_frames_pass2": int((ch > 0).sum()),
            "q06_and_chained": int(((q == 0.6) & (ch > 0)).sum()), "q06_and_direct": int(d6.sum()),
            "q06_direct_reasons_(overlapping)": {"inliers<100": int((d6 & (s["n_inliers"] < 100)).sum()),
                                                  "ratio<0.5": int((d6 & (s["inlier_ratio"] < 0.5)).sum()),
                                                  "rms>1.0": int((d6 & (s["reproj_rms_px"] > 1.0)).sum()),
                                                  "spread<7": int((d6 & (s["spread_cells"] < 7)).sum())},
            "q06_by_keyframe": {int(k): int(((q == 0.6) & (kf == k)).sum()) for k in np.unique(kf)},
            "q09_by_keyframe": {int(k): int(((q == 0.9) & (kf == k)).sum()) for k in np.unique(kf)},
            "per_100_frames_[start,share0.9,share_unreliable]": [
                [int(a), round(float((q[a:a + 100] >= 0.9).mean()), 2), round(float((q[a:a + 100] == 0).mean()), 2)]
                for a in range(0, n, 100)]}


def structure(motion):
    s = motion.stats
    W, H = motion.frame_wh
    st = {"keyframes": s["key_frame_idx"].tolist(), "key_parent": s["key_parent"].tolist(),
          "key_hops": s["key_hops"].tolist(), "key_view": s["key_view"].tolist(),
          "key_chain_len": s["key_chain_len"].tolist(), "n_views": len(motion.views),
          "view_canvas_wh": s["view_canvas_wh"].tolist(), "max_hops": int(s["key_hops"].max()),
          "max_closure_err_px": motion.meta["max_closure_err_px"], "is_static": motion.is_static,
          "zoom_suspected_meta": motion.meta["n_zoom_suspected"], "zoom_suspected_flag_frames": int((s["flags"] & 1).sum()),
          "warnings": motion.meta["warnings"], "frames_per_view_usable": {}, "frame_centre_stretch": {}}
    for v in range(len(motion.views)):
        u = motion._usable_array(v)
        st["frames_per_view_usable"][v] = int(u.sum())
        sv = [stretch_at(motion.views[v].H_ref_to_view @ motion.H_frame_to_ref[f], (W / 2, H / 2)) for f in np.where(u)[0]]
        st["frame_centre_stretch"][v] = {"max": float(np.max(sv)), "median": float(np.median(sv))}
    u0, u1 = motion._usable_array(0), motion._usable_array(1) if len(motion.views) > 1 else np.zeros(len(s["keyframe"]), bool)
    st["frames_usable_in_both_views"] = int((u0 & u1).sum())
    st["frames_usable_in_no_view"] = int((~(u0 | u1)).sum())
    st["frames_reliable_but_in_no_view"] = int(((motion.quality > 0) & ~(u0 | u1)).sum())
    return st


def scoreboard_check(motion):
    out = {}
    for name, d in SCOREBOARD.items():
        pts = {}
        for f, p in d.items():
            for v in range(len(motion.views)):
                q = motion.to_view(np.array([p], float), np.array([f]), v)[0]
                if np.isfinite(q[0]):
                    pts.setdefault(v, {})[f] = q.round(1).tolist()
        out[name] = pts
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--det", required=True)
    ap.add_argument("--motion", required=True)
    ap.add_argument("--tracks", required=True)
    ap.add_argument("--out", default=str(ROOT / "runs" / "a3_motion_checks.json"))
    a = ap.parse_args()
    motion = load_motion(Path(a.motion))
    z = np.load(a.det, allow_pickle=False)
    det = (z["frame_idx"], z["xyxy"], z["confidence"], z["class_id"])
    boxes, _ = load_detection_boxes(Path(a.det), motion.n_frames)
    tracks = pd.read_csv(a.tracks)
    fps = float(motion.meta["fps"])
    res = {"structure": structure(motion), "coverage": coverage(motion, fps)}
    res["rim"] = rim_check(det, motion)
    res["people"] = people_check(det, motion, tracks)
    res["scoreboard"] = scoreboard_check(motion)
    res["drift"] = independent_drift(Path(a.video), boxes, motion)
    Path(a.out).write_text(json.dumps(res, indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o)))
    print("wrote", a.out)


if __name__ == "__main__":
    main()

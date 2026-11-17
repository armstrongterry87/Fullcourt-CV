"""Every tuning parameter (marked T in docs/design/004-shot-scoring.md) in one place.

Durations are in seconds and lengths along the image are in rim widths (W), so nothing
here depends on the frame rate or resolution. None of these values was fitted to data
(there is no labelled footage yet); the design doc is the source for each default.
Values the design does not give are marked CHOSEN in the comment.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ShotConfig:
    """Tuning parameters for trajectory cleaning, attempt, make/miss, shooter and 2/3."""

    # --- 4.1 rim calibration ---
    rim_median_min_detections: int = 30          # per side, median fallback
    rim_median_max_mad_frac: float = 0.25        # MAD of centre / median width
    rim_drift_w: float = 0.5                     # drift threshold, in W
    rim_small_px: float = 20.0                   # W below this: result is always unknown
    rim_drift_conf_cap: float = 0.3              # design: result_conf capped at 0.3 on drift

    # --- 4.2 trajectory cleaning ---
    max_speed_w_per_s: float = 36.0              # speed gate, W per second
    seed_len: int = 3                            # consecutive detections to start a chain (design: 3)
    reanchor_len: int = 5                        # rejected-but-consistent detections to re-anchor
    ghost_hit_w: float = 0.1                     # same location within this, hit twice => ghost
    ghost_hits: int = 2
    ghost_drop_w: float = 0.25                   # detections this close to a ghost are dropped
    max_gap_s: float = 0.125                     # gap filled by linear interpolation

    # --- 4.3 shot attempt ---
    held_pad_x_frac: float = 0.25                # of player box width, each side
    held_pad_top_frac: float = 0.25              # of player box height, top only
    min_flight_s: float = 0.15
    rise_window_s: float = 0.25                  # window for the mean vy < 0 test
    apex_above_rim_w: float = 0.5                # apex at least this far above rim top
    zone_x_w: float = 3.0
    zone_above_w: float = 4.0
    zone_below_w: float = 1.0
    attempt_lockout_s: float = 0.75
    max_release_gap_s: float = 0.5               # CHOSEN: the held frame must be this recent

    # --- 4.4 make / miss ---
    outcome_window_s: float = 1.5
    crit_x_w: float = 1.5
    crit_y_w: float = 1.5
    make_max_dt_s: float = 0.30
    make_max_rise_w: float = 0.1                 # y may not decrease by more than this between a and b
    make_x_margin_w: float = 0.1                 # crossing must be inside the span shrunk by this
    below_x_margin_w: float = 0.25               # b / "outside the rim" tolerance
    bounce_rise_w: float = 0.5
    depth_ratio_min: float = 0.35                # ball width / W at a
    depth_ratio_max: float = 0.75

    # --- 4.5 shooter and feet ---
    shooter_vote_s: float = 0.25
    shooter_max_dist_h: float = 0.5              # of box height
    shooter_conf_hi: float = 0.8
    shooter_conf_mid: float = 0.5
    takeoff_window_s: float = 0.5
    feet_edge_px: float = 2.0                    # y2 this close to the frame bottom => cut off

    # --- 4.6 two or three ---
    homography_ransac_ft: float = 0.5
    homography_max_rms_ft: float = 1.0
    homography_conf_cap_rms_ft: float = 0.5
    near_arc_ft: float = 1.0
    off_court_ft: float = 1.0                    # feet farther than this outside the court

    # --- 4.7 free throw ---
    ft_axis_ft: float = 3.0
    ft_in_front_ft: float = 1.0
    ft_behind_ft: float = 3.0
    ft_still_ft: float = 1.0
    ft_still_s: float = 1.0
    ft_clear_ft: float = 4.0
    ft_others_still_ft: float = 1.5
    ft_others_still_frac: float = 0.7
    ft_conf_cap: float = 0.6

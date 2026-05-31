#!/usr/bin/env python3

import argparse
import csv
import math
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib
import numpy as np
import yaml

matplotlib.use("Agg")
import matplotlib.pyplot as plt


DEFAULT_DATA_ROOT = "/home/shin/challenge_ws/nav_cloning_data"
DEFAULT_BIN_EDGES = "-1.2,-0.72,-0.24,0.24,0.72,1.2"


def parse_args():
    ap = argparse.ArgumentParser(
        description=(
            "Compare monocular feature VO output against command-integrated odometry. "
            "The command odometry is a pseudo reference from angular labels and constant speed."
        )
    )
    ap.add_argument("--vo-dir", required=True, help="directory containing trajectory.csv and pairs.csv")
    ap.add_argument("--data-root", default=DEFAULT_DATA_ROOT)
    ap.add_argument("--time", default="", help="dataset time id; inferred from summary.csv when omitted")
    ap.add_argument("--speed", type=float, default=0.0, help="linear speed [m/s]; 0 uses metadata speed")
    ap.add_argument(
        "--action-sign",
        type=float,
        default=1.0,
        choices=(-1.0, 1.0),
        help="multiply angular labels by this sign before command integration",
    )
    ap.add_argument(
        "--dt-mode",
        choices=("duration", "timestamp"),
        default="duration",
        help="duration uses metadata duration per frame; timestamp uses index.csv timestamp deltas",
    )
    ap.add_argument("--bin-edges", default=DEFAULT_BIN_EDGES)
    ap.add_argument("--fit-on", choices=("ok", "all"), default="ok")
    ap.add_argument("--out-dir", default="", help="default: <vo-dir>/command_odom_eval")
    return ap.parse_args()


def read_csv_dicts(path: Path):
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows):
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def load_metadata(video_dir: Path):
    path = video_dir / "metadata.yaml"
    if not path.is_file():
        return {}
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def infer_time(vo_dir: Path, explicit_time: str):
    if explicit_time:
        return explicit_time
    summary_path = vo_dir / "summary.csv"
    if summary_path.is_file():
        rows = read_csv_dicts(summary_path)
        if rows and rows[0].get("dataset"):
            return rows[0]["dataset"]
    raise SystemExit("[ERR] pass --time or keep summary.csv in --vo-dir")


def parse_float(value, default=np.nan):
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def parse_int(value, default=-1):
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def parse_edges(value: str):
    vals = [float(x.strip()) for x in value.split(",") if x.strip()]
    if len(vals) < 2:
        raise SystemExit("[ERR] --bin-edges needs at least two comma-separated numbers")
    vals = np.array(vals, dtype=float)
    if np.any(np.diff(vals) <= 0):
        raise SystemExit("[ERR] --bin-edges must be strictly increasing")
    return vals


def wrap_angle(rad):
    return (rad + math.pi) % (2.0 * math.pi) - math.pi


def load_index(video_dir: Path):
    rows = read_csv_dicts(video_dir / "index.csv")
    out = {}
    for row in rows:
        frame = parse_int(row.get("video_frame"))
        if frame < 0:
            continue
        out[frame] = {
            "timestamp": parse_float(row.get("timestamp")),
            "action": parse_float(row.get("action"), 0.0),
        }
    if not out:
        raise SystemExit(f"[ERR] no usable rows in {video_dir / 'index.csv'}")
    return out


def integrate_command_odom(index_by_frame, max_frame, speed, duration, dt_mode, action_sign):
    poses = {}
    x = 0.0
    y = 0.0
    theta = 0.0
    last_action = 0.0

    first_frame = min(index_by_frame)
    prev_timestamp = index_by_frame.get(first_frame, {}).get("timestamp", np.nan)

    for frame in range(0, max_frame + 1):
        row = index_by_frame.get(frame)
        if row is not None and np.isfinite(row["action"]):
            last_action = action_sign * float(row["action"])

        poses[frame] = {
            "cmd_x": x,
            "cmd_y": y,
            "cmd_theta": theta,
            "cmd_action": last_action,
        }

        next_frame = frame + 1
        if next_frame > max_frame:
            break

        if dt_mode == "timestamp":
            next_ts = index_by_frame.get(next_frame, {}).get("timestamp", np.nan)
            if np.isfinite(prev_timestamp) and np.isfinite(next_ts):
                dt = max(0.0, min(float(next_ts - prev_timestamp), 1.0))
            else:
                dt = duration
            if np.isfinite(next_ts):
                prev_timestamp = next_ts
        else:
            dt = duration

        mid_theta = theta + 0.5 * last_action * dt
        x += speed * dt * math.cos(mid_theta)
        y += speed * dt * math.sin(mid_theta)
        theta = wrap_angle(theta + last_action * dt)

    return poses


def se2_align(source_xy, target_xy):
    if len(source_xy) < 2:
        raise SystemExit("[ERR] need at least two points for SE(2) alignment")
    src_mean = source_xy.mean(axis=0)
    tgt_mean = target_xy.mean(axis=0)
    src = source_xy - src_mean
    tgt = target_xy - tgt_mean
    h_mat = src.T @ tgt
    u_mat, _, vt_mat = np.linalg.svd(h_mat)
    rot = vt_mat.T @ u_mat.T
    if np.linalg.det(rot) < 0:
        vt_mat[-1, :] *= -1.0
        rot = vt_mat.T @ u_mat.T
    trans = tgt_mean - src_mean @ rot.T
    return rot, trans


def apply_se2(points_xy, rot, trans):
    return points_xy @ rot.T + trans


def assign_bin(value, edges):
    idx = int(np.searchsorted(edges, value, side="left") - 1)
    return max(0, min(len(edges) - 2, idx))


def mean_action(index_by_frame, start_frame, end_frame, action_sign):
    vals = []
    for frame in range(start_frame, end_frame + 1):
        row = index_by_frame.get(frame)
        if row is not None and np.isfinite(row["action"]):
            vals.append(action_sign * float(row["action"]))
    if not vals:
        return 0.0
    return float(np.mean(vals))


def summarize_groups(rows, key):
    groups = {}
    for row in rows:
        groups.setdefault(row[key], []).append(row)

    out = []
    for name in sorted(groups):
        vals = groups[name]
        n = len(vals)
        ok = [r for r in vals if r["success"]]
        direction_errors = [
            r["direction_abs_error_deg"]
            for r in vals
            if np.isfinite(r["direction_abs_error_deg"])
        ]
        out.append(
            {
                key: name,
                "pairs": n,
                "ok_pairs": len(ok),
                "success_rate": len(ok) / n if n else 0.0,
                "mean_abs_action": float(np.mean([abs(r["mean_action"]) for r in vals])),
                "mean_matches": float(np.mean([r["matches"] for r in vals])),
                "mean_essential_inliers": float(np.mean([r["essential_inliers"] for r in vals])),
                "mean_pose_inliers": float(np.mean([r["pose_inliers"] for r in vals])),
                "mean_alignment_error": float(np.mean([r["alignment_error"] for r in vals])),
                "mean_direction_abs_error_deg": float(np.mean(direction_errors))
                if direction_errors
                else np.nan,
            }
        )
    return out


def save_overlay_plot(out_dir, cmd_xy, vo_xy, vo_aligned, title):
    fig, ax = plt.subplots(figsize=(9, 9))
    ax.plot(cmd_xy[:, 0], cmd_xy[:, 1], color="#222222", linewidth=1.8, label="command odom")
    ax.plot(vo_aligned[:, 0], vo_aligned[:, 1], color="#1f77b4", linewidth=1.2, label="VO aligned by SE(2)")
    ax.plot(vo_xy[:, 0], vo_xy[:, 1], color="#9ecae1", linewidth=0.9, alpha=0.55, label="VO raw x-z")
    ax.scatter(cmd_xy[:1, 0], cmd_xy[:1, 1], color="#2ca02c", s=55, label="start", zorder=3)
    ax.scatter(cmd_xy[-1:, 0], cmd_xy[-1:, 1], color="#d62728", s=55, label="end", zorder=3)
    ax.set_xlabel("x [m-ish, command odom frame]")
    ax.set_ylabel("y [m-ish, command odom frame]")
    ax.set_title(title)
    ax.axis("equal")
    ax.grid(alpha=0.3)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(out_dir / "command_vs_vo_se2_overlay.png", dpi=180)
    plt.close(fig)


def save_error_plot(out_dir, frames, errors, statuses, title):
    ok = np.array([s == "ok" or s == "start" for s in statuses], dtype=bool)
    fig, ax = plt.subplots(figsize=(12, 4.8))
    ax.plot(frames, errors, color="#4c78a8", linewidth=1.0, label="SE(2) alignment error")
    ax.scatter(frames[~ok], errors[~ok], color="#e45756", s=12, label="VO update failed", zorder=3)
    ax.set_xlabel("video frame")
    ax.set_ylabel("position error [m-ish]")
    ax.set_title(title)
    ax.grid(alpha=0.25)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(out_dir / "alignment_error_by_frame.png", dpi=180)
    plt.close(fig)


def save_bin_plot(out_dir, bin_rows, title):
    labels = [str(row["action_bin"]) for row in bin_rows]
    x_pos = np.arange(len(bin_rows), dtype=float)
    success = np.array([row["success_rate"] for row in bin_rows], dtype=float)
    matches = np.array([row["mean_matches"] for row in bin_rows], dtype=float)
    pose = np.array([row["mean_pose_inliers"] for row in bin_rows], dtype=float)

    fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
    axes[0].bar(x_pos, success, color="#4c78a8", edgecolor="#263238", linewidth=0.7)
    axes[0].set_ylabel("VO success rate")
    axes[0].set_ylim(0, 1.05)
    axes[0].grid(axis="y", alpha=0.25)
    for i, row in enumerate(bin_rows):
        axes[0].text(i, success[i] + 0.02, f'n={row["pairs"]}', ha="center", fontsize=8)

    axes[1].plot(x_pos, matches, marker="o", color="#f58518", label="matches")
    axes[1].plot(x_pos, pose, marker="o", color="#54a24b", label="pose inliers")
    axes[1].set_ylabel("mean count")
    axes[1].set_xlabel("angular velocity bin [rad/s]")
    axes[1].grid(alpha=0.25)
    axes[1].legend(loc="best")
    axes[1].set_xticks(x_pos)
    axes[1].set_xticklabels(labels)

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_dir / "success_by_angular_bin.png", dpi=180)
    plt.close(fig)


def main():
    args = parse_args()
    vo_dir = Path(args.vo_dir).expanduser().resolve()
    if not vo_dir.is_dir():
        raise SystemExit(f"[ERR] missing --vo-dir: {vo_dir}")

    time_id = infer_time(vo_dir, args.time)
    video_dir = Path(args.data_root).expanduser().resolve() / time_id / "dataset" / "video"
    metadata = load_metadata(video_dir)
    index_by_frame = load_index(video_dir)
    duration = float(metadata.get("duration", 0.1))
    speed = float(args.speed) if args.speed > 0 else float(metadata.get("speed", 0.8))
    edges = parse_edges(args.bin_edges)

    traj_rows = read_csv_dicts(vo_dir / "trajectory.csv")
    pair_rows_raw = read_csv_dicts(vo_dir / "pairs.csv")
    if not traj_rows or not pair_rows_raw:
        raise SystemExit("[ERR] trajectory.csv and pairs.csv are required")

    frames = np.array([parse_int(r["frame"]) for r in traj_rows], dtype=int)
    max_frame = int(np.max(frames))
    cmd_poses = integrate_command_odom(
        index_by_frame, max_frame, speed, duration, args.dt_mode, args.action_sign
    )

    vo_xy = np.array([[parse_float(r["x"]), parse_float(r["z"])] for r in traj_rows], dtype=float)
    cmd_xy = np.array(
        [[cmd_poses[int(f)]["cmd_x"], cmd_poses[int(f)]["cmd_y"]] for f in frames],
        dtype=float,
    )
    statuses = np.array([r.get("status", "") for r in traj_rows], dtype=object)

    if args.fit_on == "ok":
        fit_mask = np.array([(s == "ok" or s == "start") for s in statuses], dtype=bool)
    else:
        fit_mask = np.ones(len(statuses), dtype=bool)
    if int(fit_mask.sum()) < 2:
        fit_mask = np.ones(len(statuses), dtype=bool)

    rot, trans = se2_align(vo_xy[fit_mask], cmd_xy[fit_mask])
    vo_aligned = apply_se2(vo_xy, rot, trans)
    align_errors = np.linalg.norm(vo_aligned - cmd_xy, axis=1)

    frame_to_i = {int(frame): i for i, frame in enumerate(frames)}
    per_pair = []
    previous_vo_heading = np.nan
    previous_cmd_heading = np.nan

    for row in pair_rows_raw:
        prev_frame = parse_int(row.get("prev_frame"))
        frame = parse_int(row.get("frame"))
        if prev_frame not in frame_to_i or frame not in frame_to_i:
            continue
        i0 = frame_to_i[prev_frame]
        i1 = frame_to_i[frame]
        mean_w = mean_action(index_by_frame, prev_frame, frame, args.action_sign)
        bin_idx = assign_bin(mean_w, edges)
        bin_label = f"{edges[bin_idx]:.2f} to {edges[bin_idx + 1]:.2f}"
        abs_w = abs(mean_w)
        if abs_w < 0.24:
            motion_class = "straight"
        elif mean_w > 0:
            motion_class = "left_turn"
        else:
            motion_class = "right_turn"

        cmd_delta = cmd_xy[i1] - cmd_xy[i0]
        vo_delta = vo_aligned[i1] - vo_aligned[i0]
        cmd_norm = float(np.linalg.norm(cmd_delta))
        vo_norm = float(np.linalg.norm(vo_delta))
        direction_error = np.nan
        direction_abs_error = np.nan
        vo_heading = np.nan
        cmd_heading = np.nan
        vo_heading_change = np.nan
        cmd_heading_change = np.nan
        if cmd_norm > 1e-9 and vo_norm > 1e-9:
            cmd_heading = math.atan2(cmd_delta[1], cmd_delta[0])
            vo_heading = math.atan2(vo_delta[1], vo_delta[0])
            direction_error = wrap_angle(vo_heading - cmd_heading)
            direction_abs_error = abs(math.degrees(direction_error))
            if np.isfinite(previous_vo_heading) and np.isfinite(previous_cmd_heading):
                vo_heading_change = wrap_angle(vo_heading - previous_vo_heading)
                cmd_heading_change = wrap_angle(cmd_heading - previous_cmd_heading)
            previous_vo_heading = vo_heading
            previous_cmd_heading = cmd_heading

        status = row.get("status", "")
        per_pair.append(
            {
                "prev_frame": prev_frame,
                "frame": frame,
                "status": status,
                "success": status == "ok",
                "mean_action": mean_w,
                "action_bin": bin_label,
                "motion_class": motion_class,
                "matches": parse_float(row.get("matches"), 0.0),
                "essential_inliers": parse_float(row.get("essential_inliers"), 0.0),
                "pose_inliers": parse_float(row.get("pose_inliers"), 0.0),
                "cmd_dx": float(cmd_delta[0]),
                "cmd_dy": float(cmd_delta[1]),
                "vo_aligned_dx": float(vo_delta[0]),
                "vo_aligned_dy": float(vo_delta[1]),
                "direction_error_deg": float(math.degrees(direction_error))
                if np.isfinite(direction_error)
                else np.nan,
                "direction_abs_error_deg": float(direction_abs_error)
                if np.isfinite(direction_abs_error)
                else np.nan,
                "vo_heading_change_deg": float(math.degrees(vo_heading_change))
                if np.isfinite(vo_heading_change)
                else np.nan,
                "cmd_heading_change_deg": float(math.degrees(cmd_heading_change))
                if np.isfinite(cmd_heading_change)
                else np.nan,
                "alignment_error": float(align_errors[i1]),
            }
        )

    aligned_traj = []
    for i, row in enumerate(traj_rows):
        frame = int(frames[i])
        pose = cmd_poses[frame]
        aligned_traj.append(
            {
                "frame": frame,
                "timestamp": row.get("timestamp", ""),
                "action": row.get("action", ""),
                "status": row.get("status", ""),
                "vo_raw_x": float(vo_xy[i, 0]),
                "vo_raw_z": float(vo_xy[i, 1]),
                "vo_se2_x": float(vo_aligned[i, 0]),
                "vo_se2_y": float(vo_aligned[i, 1]),
                "cmd_x": float(pose["cmd_x"]),
                "cmd_y": float(pose["cmd_y"]),
                "cmd_theta": float(pose["cmd_theta"]),
                "alignment_error": float(align_errors[i]),
            }
        )

    out_dir = Path(args.out_dir).expanduser().resolve() if args.out_dir else vo_dir / "command_odom_eval"
    out_dir.mkdir(parents=True, exist_ok=True)

    write_csv(out_dir / "aligned_trajectory.csv", aligned_traj)
    write_csv(out_dir / "pair_eval.csv", per_pair)
    bin_rows = summarize_groups(per_pair, "action_bin")
    motion_rows = summarize_groups(per_pair, "motion_class")
    write_csv(out_dir / "stats_by_angular_bin.csv", bin_rows)
    write_csv(out_dir / "stats_by_motion_class.csv", motion_rows)

    ok_errors = [
        r["alignment_error"]
        for r in per_pair
        if r["success"] and np.isfinite(r["alignment_error"])
    ]
    direction_errors = [
        r["direction_abs_error_deg"]
        for r in per_pair
        if r["success"] and np.isfinite(r["direction_abs_error_deg"])
    ]
    vo_turn = np.array(
        [
            r["vo_heading_change_deg"]
            for r in per_pair
            if r["success"]
            and np.isfinite(r["vo_heading_change_deg"])
            and np.isfinite(r["cmd_heading_change_deg"])
        ],
        dtype=float,
    )
    cmd_turn = np.array(
        [
            r["cmd_heading_change_deg"]
            for r in per_pair
            if r["success"]
            and np.isfinite(r["vo_heading_change_deg"])
            and np.isfinite(r["cmd_heading_change_deg"])
        ],
        dtype=float,
    )
    turn_corr = np.nan
    if len(vo_turn) >= 3 and np.std(vo_turn) > 0 and np.std(cmd_turn) > 0:
        turn_corr = float(np.corrcoef(vo_turn, cmd_turn)[0, 1])

    total_pairs = len(per_pair)
    ok_pairs = sum(1 for r in per_pair if r["success"])
    summary = [
        {
            "dataset": time_id,
            "vo_dir": str(vo_dir),
            "dt_mode": args.dt_mode,
            "speed": speed,
            "duration": duration,
            "action_sign": args.action_sign,
            "fit_on": args.fit_on,
            "selected_frames": len(traj_rows),
            "pairs": total_pairs,
            "ok_pairs": ok_pairs,
            "success_rate": ok_pairs / total_pairs if total_pairs else 0.0,
            "se2_rmse_all": float(np.sqrt(np.mean(align_errors**2))),
            "se2_mean_error_all": float(np.mean(align_errors)),
            "se2_rmse_ok_pairs": float(np.sqrt(np.mean(np.square(ok_errors)))) if ok_errors else np.nan,
            "se2_mean_error_ok_pairs": float(np.mean(ok_errors)) if ok_errors else np.nan,
            "mean_direction_abs_error_deg_ok": float(np.mean(direction_errors))
            if direction_errors
            else np.nan,
            "median_direction_abs_error_deg_ok": float(np.median(direction_errors))
            if direction_errors
            else np.nan,
            "vo_cmd_heading_change_corr": turn_corr,
            "se2_rotation_deg": float(math.degrees(math.atan2(rot[1, 0], rot[0, 0]))),
            "se2_tx": float(trans[0]),
            "se2_ty": float(trans[1]),
        }
    ]
    write_csv(out_dir / "summary_command_eval.csv", summary)

    title = f"{time_id}: command odom vs monocular feature VO"
    save_overlay_plot(out_dir, cmd_xy, vo_xy, vo_aligned, title)
    save_error_plot(out_dir, frames, align_errors, statuses, title)
    save_bin_plot(out_dir, bin_rows, title)

    print(f"[DONE] out_dir                  : {out_dir}")
    print(f"[DONE] summary                 : {out_dir / 'summary_command_eval.csv'}")
    print(f"[DONE] aligned trajectory      : {out_dir / 'aligned_trajectory.csv'}")
    print(f"[DONE] pair eval               : {out_dir / 'pair_eval.csv'}")
    print(f"[DONE] bin stats               : {out_dir / 'stats_by_angular_bin.csv'}")
    print(f"[DONE] motion stats            : {out_dir / 'stats_by_motion_class.csv'}")
    print(f"[DONE] overlay plot            : {out_dir / 'command_vs_vo_se2_overlay.png'}")
    print(f"[DONE] angular bin plot        : {out_dir / 'success_by_angular_bin.png'}")
    print("[NOTE] Command odom is a pseudo reference from labels, not ground truth.")


if __name__ == "__main__":
    main()

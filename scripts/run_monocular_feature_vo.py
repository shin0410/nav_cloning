#!/usr/bin/env python3

import argparse
import csv
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import cv2
import matplotlib
import numpy as np
import yaml

matplotlib.use("Agg")
import matplotlib.pyplot as plt


DEFAULT_DATA_ROOT = "/home/shin/challenge_ws/nav_cloning_data"
DEFAULT_OUT_ROOT = "nav_cloning/analysis/monocular_feature_vo"


def parse_args():
    ap = argparse.ArgumentParser(
        description=(
            "Run a CPU-only monocular feature VO trial on a nav_cloning video. "
            "This estimates relative camera motion from ORB/SIFT matches and an essential matrix."
        )
    )
    ap.add_argument("--data-root", default=DEFAULT_DATA_ROOT)
    ap.add_argument("--time", required=True, help="dataset time id, e.g. 20260426_13_39_33")
    ap.add_argument("--view", default="center", choices=("center", "left", "right"))
    ap.add_argument("--method", default="orb", choices=("orb", "sift"))
    ap.add_argument("--stride", type=int, default=3, help="process every Nth video frame")
    ap.add_argument(
        "--max-selected-frames",
        type=int,
        default=0,
        help="stop after this many processed frames; 0 means all selected frames",
    )
    ap.add_argument("--start-frame", type=int, default=0)
    ap.add_argument("--nfeatures", type=int, default=2500)
    ap.add_argument("--ratio", type=float, default=0.75, help="Lowe ratio-test threshold")
    ap.add_argument("--min-matches", type=int, default=40)
    ap.add_argument("--min-essential-inliers", type=int, default=40)
    ap.add_argument("--min-pose-inliers", type=int, default=40)
    ap.add_argument(
        "--ransac-threshold",
        type=float,
        default=1.5,
        help="essential-matrix RANSAC threshold in pixels",
    )
    ap.add_argument(
        "--focal-px",
        type=float,
        default=0.0,
        help="camera focal length in pixels; 0 uses --focal-scale * video width",
    )
    ap.add_argument("--focal-scale", type=float, default=0.9)
    ap.add_argument(
        "--scale-mode",
        choices=("commanded", "unit"),
        default="commanded",
        help="commanded uses speed*duration*frame_delta as a rough metric scale; unit keeps each step length at 1",
    )
    ap.add_argument("--out-dir", default="")
    return ap.parse_args()


def load_metadata(video_dir: Path):
    path = video_dir / "metadata.yaml"
    if not path.is_file():
        return {}
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def load_index(video_dir: Path):
    path = video_dir / "index.csv"
    rows = {}
    if not path.is_file():
        return rows
    with path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                rows[int(row["video_frame"])] = row
            except (KeyError, TypeError, ValueError):
                continue
    return rows


def make_feature_extractor(method: str, nfeatures: int):
    if method == "sift":
        detector = cv2.SIFT_create(nfeatures=nfeatures)
        matcher = cv2.BFMatcher(cv2.NORM_L2)
    else:
        detector = cv2.ORB_create(
            nfeatures=nfeatures,
            scaleFactor=1.2,
            nlevels=8,
            edgeThreshold=19,
            patchSize=31,
            fastThreshold=7,
        )
        matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
    return detector, matcher


def detect(detector, frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return detector.detectAndCompute(gray, None)


def ratio_matches(matcher, prev_desc, curr_desc, ratio):
    if prev_desc is None or curr_desc is None:
        return []
    if len(prev_desc) < 2 or len(curr_desc) < 2:
        return []
    pairs = matcher.knnMatch(prev_desc, curr_desc, k=2)
    good = []
    for pair in pairs:
        if len(pair) != 2:
            continue
        m, n = pair
        if m.distance < ratio * n.distance:
            good.append(m)
    good.sort(key=lambda m: m.distance)
    return good


def recover_best_pose(essential, pts_prev, pts_curr, k_mat, inlier_mask):
    if essential is None:
        return None

    candidates = []
    if essential.shape == (3, 3):
        candidates = [essential]
    elif essential.ndim == 2 and essential.shape[1] == 3 and essential.shape[0] % 3 == 0:
        candidates = [essential[i : i + 3, :] for i in range(0, essential.shape[0], 3)]

    best = None
    for e_mat in candidates:
        try:
            if inlier_mask is None:
                pose_inliers, rot, trans, pose_mask = cv2.recoverPose(
                    e_mat, pts_prev, pts_curr, k_mat
                )
            else:
                pose_inliers, rot, trans, pose_mask = cv2.recoverPose(
                    e_mat, pts_prev, pts_curr, k_mat, mask=inlier_mask.copy()
                )
        except cv2.error:
            continue
        if best is None or pose_inliers > best[0]:
            best = (int(pose_inliers), rot, trans, pose_mask)
    return best


def read_selected_frames(cap, start_frame, stride, max_selected):
    frame_idx = -1
    selected = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame_idx += 1
        if frame_idx < start_frame:
            continue
        if (frame_idx - start_frame) % stride != 0:
            continue
        yield frame_idx, frame
        selected += 1
        if max_selected > 0 and selected >= max_selected:
            break


def save_csv(path: Path, rows):
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def plot_trajectory(out_dir: Path, traj_rows, title: str):
    xs = np.array([r["x"] for r in traj_rows], dtype=float)
    ys = np.array([r["y"] for r in traj_rows], dtype=float)
    zs = np.array([r["z"] for r in traj_rows], dtype=float)

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.plot(xs, zs, color="#1f77b4", linewidth=1.5)
    ax.scatter(xs[:1], zs[:1], color="#2ca02c", s=55, label="start", zorder=3)
    ax.scatter(xs[-1:], zs[-1:], color="#d62728", s=55, label="end", zorder=3)
    ax.set_xlabel("x, arbitrary camera/world units")
    ax.set_ylabel("z, arbitrary camera/world units")
    ax.set_title(title)
    ax.grid(alpha=0.3)
    ax.axis("equal")
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(out_dir / "trajectory_xz.png", dpi=180)
    plt.close(fig)

    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot(xs, ys, zs, color="#1f77b4", linewidth=1.2)
    ax.scatter(xs[:1], ys[:1], zs[:1], color="#2ca02c", s=45, label="start")
    ax.scatter(xs[-1:], ys[-1:], zs[-1:], color="#d62728", s=45, label="end")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.set_title(title)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(out_dir / "trajectory_3d.png", dpi=180)
    plt.close(fig)


def plot_diagnostics(out_dir: Path, pair_rows):
    if not pair_rows:
        return
    frames = np.array([r["frame"] for r in pair_rows], dtype=int)
    matches = np.array([r["matches"] for r in pair_rows], dtype=float)
    ransac = np.array([r["essential_inliers"] for r in pair_rows], dtype=float)
    pose = np.array([r["pose_inliers"] for r in pair_rows], dtype=float)
    status = np.array([1 if r["status"] == "ok" else 0 for r in pair_rows], dtype=float)

    fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
    axes[0].plot(frames, matches, color="#4c78a8", linewidth=1.0, label="ratio matches")
    axes[0].plot(frames, ransac, color="#f58518", linewidth=1.0, label="essential inliers")
    axes[0].plot(frames, pose, color="#54a24b", linewidth=1.0, label="recoverPose inliers")
    axes[0].set_ylabel("count")
    axes[0].grid(alpha=0.25)
    axes[0].legend(loc="best")

    with np.errstate(divide="ignore", invalid="ignore"):
        inlier_ratio = np.where(matches > 0, ransac / matches, 0.0)
    axes[1].plot(frames, inlier_ratio, color="#e45756", linewidth=1.0)
    axes[1].set_ylabel("E inlier ratio")
    axes[1].set_ylim(0, 1.05)
    axes[1].grid(alpha=0.25)

    axes[2].plot(frames, status, color="#72b7b2", linewidth=0.9)
    axes[2].set_ylabel("success")
    axes[2].set_xlabel("video frame")
    axes[2].set_ylim(-0.05, 1.05)
    axes[2].grid(alpha=0.25)

    fig.suptitle("Feature VO diagnostics")
    fig.tight_layout()
    fig.savefig(out_dir / "diagnostics.png", dpi=180)
    plt.close(fig)


def save_match_debug(out_dir: Path, prev_frame, prev_kp, curr_frame, curr_kp, matches, mask):
    if prev_frame is None or curr_frame is None or not matches:
        return
    keep = []
    if mask is not None:
        flat = mask.reshape(-1)
        for i, m in enumerate(matches):
            if i < len(flat) and flat[i]:
                keep.append(m)
    else:
        keep = matches
    keep = keep[:80]
    if not keep:
        keep = matches[:80]
    img = cv2.drawMatches(
        prev_frame,
        prev_kp,
        curr_frame,
        curr_kp,
        keep,
        None,
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
    )
    cv2.imwrite(str(out_dir / "first_successful_matches.jpg"), img)


def main():
    args = parse_args()
    if args.stride < 1:
        raise SystemExit("--stride must be >= 1")

    dataset_dir = Path(args.data_root).expanduser().resolve() / args.time / "dataset"
    video_dir = dataset_dir / "video"
    video_path = video_dir / f"{args.view}.mp4"
    if not video_path.is_file():
        raise SystemExit(f"[ERR] missing video: {video_path}")

    metadata = load_metadata(video_dir)
    index_rows = load_index(video_dir)

    if args.out_dir:
        out_dir = Path(args.out_dir).expanduser().resolve()
    else:
        suffix = f"{args.time}_{args.view}_{args.method}_stride{args.stride}"
        if args.max_selected_frames > 0:
            suffix += f"_first{args.max_selected_frames}"
        out_dir = (Path(DEFAULT_OUT_ROOT) / suffix).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise SystemExit(f"[ERR] failed to open: {video_path}")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or int(metadata.get("video_width", 640))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or int(metadata.get("video_height", 480))
    focal_px = float(args.focal_px) if args.focal_px > 0 else float(args.focal_scale * width)
    k_mat = np.array(
        [[focal_px, 0.0, width / 2.0], [0.0, focal_px, height / 2.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )

    detector, matcher = make_feature_extractor(args.method, args.nfeatures)
    speed = float(metadata.get("speed", 1.0))
    duration = float(metadata.get("duration", 1.0 / max(float(metadata.get("video_fps", 10.0)), 1.0)))

    prev_frame = None
    prev_frame_idx = None
    prev_kp = None
    prev_desc = None
    first_debug_saved = False

    # Rotation from current camera coordinates to the start-frame world coordinates,
    # and camera center in start-frame world coordinates.
    world_from_camera = np.eye(3, dtype=np.float64)
    camera_center = np.zeros((3, 1), dtype=np.float64)

    traj_rows = []
    pair_rows = []

    for frame_idx, frame in read_selected_frames(
        cap, args.start_frame, args.stride, args.max_selected_frames
    ):
        kp, desc = detect(detector, frame)
        index_row = index_rows.get(frame_idx, {})
        timestamp = index_row.get("timestamp", "")
        action = index_row.get("action", "")

        if prev_frame is None:
            traj_rows.append(
                {
                    "frame": frame_idx,
                    "timestamp": timestamp,
                    "action": action,
                    "x": 0.0,
                    "y": 0.0,
                    "z": 0.0,
                    "status": "start",
                }
            )
            prev_frame = frame
            prev_frame_idx = frame_idx
            prev_kp = kp
            prev_desc = desc
            continue

        matches = ratio_matches(matcher, prev_desc, desc, args.ratio)
        status = "ok"
        essential_inliers = 0
        pose_inliers = 0
        step_scale = 0.0
        e_mask = None

        if len(matches) < args.min_matches:
            status = "few_matches"
        else:
            pts_prev = np.float32([prev_kp[m.queryIdx].pt for m in matches])
            pts_curr = np.float32([kp[m.trainIdx].pt for m in matches])
            essential, e_mask = cv2.findEssentialMat(
                pts_prev,
                pts_curr,
                k_mat,
                method=cv2.RANSAC,
                prob=0.999,
                threshold=args.ransac_threshold,
            )
            if e_mask is not None:
                essential_inliers = int(e_mask.reshape(-1).sum())
            if essential_inliers < args.min_essential_inliers:
                pose = None
                status = "few_essential_inliers"
            else:
                pose = recover_best_pose(essential, pts_prev, pts_curr, k_mat, e_mask)
            if pose is None:
                if status == "ok":
                    status = "pose_failed"
            else:
                pose_inliers, inc_rot, inc_trans, _ = pose
                if pose_inliers < args.min_pose_inliers:
                    status = "few_pose_inliers"
                else:
                    frame_delta = max(1, frame_idx - int(prev_frame_idx))
                    if args.scale_mode == "commanded":
                        step_scale = speed * duration * frame_delta
                    else:
                        step_scale = 1.0

                    # recoverPose gives P2 = K [R|t] in the previous camera frame.
                    # The second camera center in the previous camera frame is -R.T t.
                    delta_prev_camera = -inc_rot.T @ (inc_trans * step_scale)
                    camera_center = camera_center + world_from_camera @ delta_prev_camera
                    world_from_camera = world_from_camera @ inc_rot.T

                    if not first_debug_saved:
                        save_match_debug(
                            out_dir,
                            prev_frame,
                            prev_kp,
                            frame,
                            kp,
                            matches,
                            e_mask,
                        )
                        first_debug_saved = True

        pair_rows.append(
            {
                "prev_frame": int(prev_frame_idx),
                "frame": int(frame_idx),
                "timestamp": timestamp,
                "action": action,
                "status": status,
                "prev_keypoints": int(len(prev_kp) if prev_kp is not None else 0),
                "keypoints": int(len(kp) if kp is not None else 0),
                "matches": int(len(matches)),
                "essential_inliers": int(essential_inliers),
                "pose_inliers": int(pose_inliers),
                "step_scale": float(step_scale),
                "x": float(camera_center[0, 0]),
                "y": float(camera_center[1, 0]),
                "z": float(camera_center[2, 0]),
            }
        )
        traj_rows.append(
            {
                "frame": int(frame_idx),
                "timestamp": timestamp,
                "action": action,
                "x": float(camera_center[0, 0]),
                "y": float(camera_center[1, 0]),
                "z": float(camera_center[2, 0]),
                "status": status,
            }
        )

        prev_frame = frame
        prev_frame_idx = frame_idx
        prev_kp = kp
        prev_desc = desc

    cap.release()

    if len(traj_rows) < 2:
        raise SystemExit("[ERR] fewer than two selected frames were processed")

    save_csv(out_dir / "trajectory.csv", traj_rows)
    save_csv(out_dir / "pairs.csv", pair_rows)

    ok_rows = [r for r in pair_rows if r["status"] == "ok"]
    total_pairs = len(pair_rows)
    success_rate = len(ok_rows) / total_pairs if total_pairs else 0.0
    distances = []
    prev = np.array([traj_rows[0]["x"], traj_rows[0]["y"], traj_rows[0]["z"]], dtype=float)
    for row in traj_rows[1:]:
        cur = np.array([row["x"], row["y"], row["z"]], dtype=float)
        distances.append(float(np.linalg.norm(cur - prev)))
        prev = cur

    summary = [
        {
            "dataset": args.time,
            "view": args.view,
            "method": args.method,
            "video": str(video_path),
            "selected_frames": len(traj_rows),
            "pairs": total_pairs,
            "successful_pairs": len(ok_rows),
            "success_rate": success_rate,
            "stride": args.stride,
            "start_frame": args.start_frame,
            "max_selected_frames": args.max_selected_frames,
            "min_essential_inliers": args.min_essential_inliers,
            "min_pose_inliers": args.min_pose_inliers,
            "width": width,
            "height": height,
            "focal_px": focal_px,
            "scale_mode": args.scale_mode,
            "speed": speed,
            "duration": duration,
            "mean_matches": float(np.mean([r["matches"] for r in pair_rows])) if pair_rows else 0.0,
            "mean_essential_inliers": float(np.mean([r["essential_inliers"] for r in pair_rows]))
            if pair_rows
            else 0.0,
            "mean_pose_inliers": float(np.mean([r["pose_inliers"] for r in pair_rows]))
            if pair_rows
            else 0.0,
            "path_length": float(np.sum(distances)),
            "final_x": float(traj_rows[-1]["x"]),
            "final_y": float(traj_rows[-1]["y"]),
            "final_z": float(traj_rows[-1]["z"]),
        }
    ]
    save_csv(out_dir / "summary.csv", summary)

    title = (
        f"{args.time} {args.view} {args.method.upper()} feature VO "
        f"(stride={args.stride}, success={success_rate:.1%})"
    )
    plot_trajectory(out_dir, traj_rows, title)
    plot_diagnostics(out_dir, pair_rows)

    print(f"[DONE] out_dir              : {out_dir}")
    print(f"[DONE] trajectory_csv      : {out_dir / 'trajectory.csv'}")
    print(f"[DONE] pairs_csv           : {out_dir / 'pairs.csv'}")
    print(f"[DONE] summary_csv         : {out_dir / 'summary.csv'}")
    print(f"[DONE] trajectory_plot     : {out_dir / 'trajectory_xz.png'}")
    print(f"[DONE] diagnostics_plot    : {out_dir / 'diagnostics.png'}")
    if (out_dir / "first_successful_matches.jpg").is_file():
        print(f"[DONE] match_debug         : {out_dir / 'first_successful_matches.jpg'}")
    print(
        "[NOTE] Monocular VO has arbitrary scale; commanded speed/duration is only a rough scale prior."
    )


if __name__ == "__main__":
    main()

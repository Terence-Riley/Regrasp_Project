#!/usr/bin/env python3
"""Dry-run regrasp planning from pickup workspace to place workspace.

中文说明：基于校准好的杯子散落区和放置区进行重抓取 dry run。脚本只做感知、
聚类、候选杯子排序和放置点规划，不连接 Kortex，不移动机械臂。

This script is perception-only and does not move the Kinova arm. It detects cup
candidates in the pickup workspace, uses each cleaned cluster center as the
first-pass pick point, generates placement slots along the centerline of the
placement rectangle, and prints/saves the planned sequence.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import open3d as o3d
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from perception.aruco_detector import (  # noqa: E402
    MARKER_LENGTH,
    TARGET_MARKER_ID,
    create_aruco_detector,
    detect_target_marker_pose,
)
from perception.realsense_camera import RealSenseCamera  # noqa: E402
from utils.transform_utils import invert_T  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=15)
    parser.add_argument(
        "--camera-stream-order",
        choices=["color_first", "depth_first", "both"],
        default="color_first",
    )
    parser.add_argument("--camera-start-attempts", type=int, default=3)
    parser.add_argument("--camera-retry-delay", type=float, default=2.0)
    parser.add_argument("--camera-hardware-reset-on-fail", action="store_true")
    parser.add_argument(
        "--calibration",
        type=Path,
        default=PROJECT_ROOT / "configs" / "real_calibration.yaml",
    )
    parser.add_argument(
        "--workspaces",
        type=Path,
        default=PROJECT_ROOT / "configs" / "real_workspaces.yaml",
    )

    parser.add_argument("--depth-min", type=float, default=0.05)
    parser.add_argument("--depth-trunc", type=float, default=1.20)
    parser.add_argument("--voxel-size", type=float, default=0.004)
    parser.add_argument("--table-distance-threshold", type=float, default=0.008)
    parser.add_argument("--near-table-clearance", type=float, default=0.006)
    parser.add_argument("--disable-near-table-filter", action="store_true")
    parser.add_argument("--cluster-eps", type=float, default=0.02)
    parser.add_argument("--cluster-min-points", type=int, default=50)
    parser.add_argument("--min-raw-points", type=int, default=200)
    parser.add_argument("--min-clean-points", type=int, default=120)

    parser.add_argument("--disable-depth-edge-filter", action="store_true")
    parser.add_argument("--depth-edge-threshold", type=float, default=0.02)
    parser.add_argument("--depth-edge-kernel", type=int, default=5)
    parser.add_argument("--disable-statistical-filter", action="store_true")
    parser.add_argument("--stat-nb-neighbors", type=int, default=40)
    parser.add_argument("--stat-std-ratio", type=float, default=1.0)
    parser.add_argument("--disable-radius-filter", action="store_true")
    parser.add_argument("--radius-nb-points", type=int, default=10)
    parser.add_argument("--radius", type=float, default=0.01)
    parser.add_argument("--post-clean-cluster-eps", type=float, default=0.018)
    parser.add_argument("--post-clean-cluster-min-points", type=int, default=20)

    parser.add_argument(
        "--pick-sort",
        choices=["nearest", "largest", "x", "y"],
        default="nearest",
        help="Order used for the dry-run pickup sequence.",
    )
    parser.add_argument(
        "--place-slot-spacing",
        type=float,
        default=None,
        help="Override placement slot spacing in meters.",
    )
    parser.add_argument(
        "--place-slot-margin",
        type=float,
        default=None,
        help="Override placement slot margin in meters.",
    )
    parser.add_argument(
        "--place-z-offset",
        type=float,
        default=0.04,
        help="Dry-run place point z = place table_z_mean + this offset.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "debug_pointclouds" / "regrasp_dry_run",
    )
    parser.add_argument("--save-debug-pcd", action="store_true")
    return parser.parse_args()


def load_T_base_marker(path):
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    T = np.asarray(data["T_base_marker"], dtype=np.float64)
    if T.shape != (4, 4):
        raise ValueError("T_base_marker must be 4x4.")
    return T


def load_workspaces(path):
    if not path.exists():
        raise FileNotFoundError(
            f"Missing workspace file: {path}\n"
            "Run: python scripts/calibrate_workspaces.py"
        )
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    workspaces = data.get("workspaces", {})
    for name in ("pickup", "place"):
        if name not in workspaces:
            raise KeyError(f"{path} must contain workspaces.{name}.")
        polygon = np.asarray(workspaces[name]["polygon_base_xy"], dtype=np.float64)
        if polygon.ndim != 2 or polygon.shape[1] != 2 or polygon.shape[0] < 3:
            raise ValueError(f"workspaces.{name}.polygon_base_xy must be an Nx2 polygon.")
        workspaces[name]["polygon_base_xy"] = polygon
        workspaces[name]["z_min"] = float(workspaces[name]["z_min"])
        workspaces[name]["z_max"] = float(workspaces[name]["z_max"])
        workspaces[name]["table_z_mean"] = float(workspaces[name]["table_z_mean"])
    placement = data.get("placement", {})
    return workspaces, placement


def make_depth_valid_mask(depth_z16, depth_scale_m, args):
    depth_m = depth_z16.astype(np.float32) * float(depth_scale_m)
    valid = np.isfinite(depth_m) & (depth_m >= args.depth_min) & (depth_m <= args.depth_trunc)

    edge_mask = np.zeros(depth_m.shape, dtype=bool)
    if not args.disable_depth_edge_filter:
        k = max(3, int(args.depth_edge_kernel))
        if k % 2 == 0:
            k += 1
        depth_for_max = np.where(valid, depth_m, 0.0).astype(np.float32)
        depth_for_min = np.where(valid, depth_m, float(args.depth_trunc) + 1.0).astype(np.float32)
        kernel = np.ones((k, k), dtype=np.uint8)
        local_range = cv2.dilate(depth_for_max, kernel) - cv2.erode(depth_for_min, kernel)
        edge_mask = valid & np.isfinite(local_range) & (local_range > float(args.depth_edge_threshold))
        valid = valid & ~edge_mask
    return valid, edge_mask


def build_camera_point_cloud(frames, valid_mask, args):
    depth_m = frames.depth_z16.astype(np.float64) * float(frames.depth_scale_m)
    h, w = depth_m.shape
    us, vs = np.meshgrid(np.arange(w), np.arange(h))

    z = depth_m[valid_mask]
    x = (us[valid_mask].astype(np.float64) - frames.camera_matrix[0, 2]) * z / frames.camera_matrix[0, 0]
    y = (vs[valid_mask].astype(np.float64) - frames.camera_matrix[1, 2]) * z / frames.camera_matrix[1, 1]
    points = np.stack([x, y, z], axis=1)
    colors = cv2.cvtColor(frames.color_bgr, cv2.COLOR_BGR2RGB)[valid_mask].astype(np.float64) / 255.0

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    if args.voxel_size > 0:
        pcd = pcd.voxel_down_sample(args.voxel_size)
    return pcd


def transform_pcd(pcd, T_A_B):
    out = o3d.geometry.PointCloud(pcd)
    out.transform(np.asarray(T_A_B, dtype=np.float64).reshape(4, 4))
    return out


def points_in_polygon_xy(points_xy, polygon_xy):
    polygon = np.asarray(polygon_xy, dtype=np.float32).reshape(-1, 1, 2)
    return np.array(
        [cv2.pointPolygonTest(polygon, (float(p[0]), float(p[1])), False) >= 0 for p in points_xy],
        dtype=bool,
    )


def crop_pcd_by_workspace(pcd, workspace):
    points = np.asarray(pcd.points)
    colors = np.asarray(pcd.colors)
    mask = (
        points_in_polygon_xy(points[:, :2], workspace["polygon_base_xy"])
        & (points[:, 2] >= workspace["z_min"])
        & (points[:, 2] <= workspace["z_max"])
    )
    out = o3d.geometry.PointCloud()
    out.points = o3d.utility.Vector3dVector(points[mask])
    if colors.shape[0] == points.shape[0]:
        out.colors = o3d.utility.Vector3dVector(colors[mask])
    return out


def remove_table_plane(pcd, args):
    if len(pcd.points) < 100:
        return pcd, None, []
    try:
        plane_model, inliers = pcd.segment_plane(
            distance_threshold=args.table_distance_threshold,
            ransac_n=3,
            num_iterations=1000,
        )
        return pcd.select_by_index(inliers, invert=True), plane_model, inliers
    except RuntimeError:
        return pcd, None, []


def remove_near_table_residuals(pcd, plane_model, args):
    if args.disable_near_table_filter or plane_model is None or args.near_table_clearance <= 0:
        return pcd, 0
    points = np.asarray(pcd.points)
    colors = np.asarray(pcd.colors)
    a, b, c, d = [float(v) for v in plane_model]
    normal = np.array([a, b, c], dtype=np.float64)
    distances = np.abs(points @ normal + d) / max(float(np.linalg.norm(normal)), 1e-12)
    keep = distances > float(args.near_table_clearance)
    out = o3d.geometry.PointCloud()
    out.points = o3d.utility.Vector3dVector(points[keep])
    if colors.shape[0] == points.shape[0]:
        out.colors = o3d.utility.Vector3dVector(colors[keep])
    return out, int(np.count_nonzero(~keep))


def clean_cluster_pcd(cluster_pcd, args):
    cleaned = o3d.geometry.PointCloud(cluster_pcd)
    report = {
        "raw_points": len(cleaned.points),
        "after_statistical_points": None,
        "after_radius_points": None,
        "post_clean_component_count": None,
        "clean_points": None,
    }
    if not args.disable_statistical_filter and len(cleaned.points) >= args.stat_nb_neighbors + 1:
        cleaned, _ = cleaned.remove_statistical_outlier(args.stat_nb_neighbors, args.stat_std_ratio)
        report["after_statistical_points"] = len(cleaned.points)
    if not args.disable_radius_filter and len(cleaned.points) >= args.radius_nb_points:
        cleaned, _ = cleaned.remove_radius_outlier(args.radius_nb_points, args.radius)
        report["after_radius_points"] = len(cleaned.points)
    if args.post_clean_cluster_eps > 0 and len(cleaned.points) >= args.post_clean_cluster_min_points:
        labels = np.asarray(
            cleaned.cluster_dbscan(
                eps=args.post_clean_cluster_eps,
                min_points=args.post_clean_cluster_min_points,
                print_progress=False,
            )
        )
        valid_labels = [label for label in sorted(set(labels.tolist())) if label >= 0]
        report["post_clean_component_count"] = len(valid_labels)
        if valid_labels:
            largest_label = max(valid_labels, key=lambda label: int(np.count_nonzero(labels == label)))
            cleaned = cleaned.select_by_index(np.where(labels == largest_label)[0])
    report["clean_points"] = len(cleaned.points)
    return cleaned, report


def extent(points):
    if len(points) == 0:
        return np.zeros(3, dtype=np.float64)
    return np.ptp(points, axis=0)


def make_candidates(pcd_no_table, args):
    labels = np.asarray(
        pcd_no_table.cluster_dbscan(
            eps=args.cluster_eps,
            min_points=args.cluster_min_points,
            print_progress=False,
        )
    )
    if labels.size == 0 or labels.max() < 0:
        return [], {}, {"total": int(labels.size), "noise": int(np.count_nonzero(labels < 0)), "clusters": 0}

    candidates = []
    cluster_pcds = {}
    cluster_labels = [int(label) for label in sorted(set(labels.tolist())) if label >= 0]
    for label in cluster_labels:
        idx = np.where(labels == label)[0]
        raw_pcd = pcd_no_table.select_by_index(idx)
        raw_points = np.asarray(raw_pcd.points)
        raw_count = int(len(raw_points))
        if raw_count < args.min_raw_points:
            continue

        clean_pcd, clean_report = clean_cluster_pcd(raw_pcd, args)
        clean_points = np.asarray(clean_pcd.points)
        clean_count = int(len(clean_points))
        if clean_count < args.min_clean_points:
            continue

        min_bound = np.min(clean_points, axis=0)
        max_bound = np.max(clean_points, axis=0)
        center = (min_bound + max_bound) / 2.0
        candidate = {
            "label": int(label),
            "raw_points": raw_count,
            "clean_points": clean_count,
            "center_base": center,
            "extent": extent(clean_points),
            "distance_xy": float(np.linalg.norm(center[:2])),
            "clean_report": clean_report,
        }
        candidates.append(candidate)
        cluster_pcds[int(label)] = {"raw": raw_pcd, "clean": clean_pcd}

    summary = {
        "total": int(labels.size),
        "noise": int(np.count_nonzero(labels < 0)),
        "clusters": len(cluster_labels),
    }
    return candidates, cluster_pcds, summary


def sort_candidates(candidates, strategy):
    if strategy == "largest":
        return sorted(candidates, key=lambda c: c["clean_points"], reverse=True)
    if strategy == "x":
        return sorted(candidates, key=lambda c: float(c["center_base"][0]))
    if strategy == "y":
        return sorted(candidates, key=lambda c: float(c["center_base"][1]))
    return sorted(candidates, key=lambda c: c["distance_xy"])


def place_centerline_slots(place_ws, placement_cfg, args, count):
    polygon = np.asarray(place_ws["polygon_base_xy"], dtype=np.float64)
    center = np.mean(polygon, axis=0)

    edges = []
    for i in range(len(polygon)):
        p0 = polygon[i]
        p1 = polygon[(i + 1) % len(polygon)]
        vec = p1 - p0
        length = float(np.linalg.norm(vec))
        if length > 1e-9:
            edges.append((length, vec / length))
    if not edges:
        raise ValueError("Place workspace polygon is degenerate.")

    line_length, line_dir = max(edges, key=lambda item: item[0])
    spacing = float(args.place_slot_spacing or placement_cfg.get("slot_spacing", 0.09))
    margin = float(args.place_slot_margin or placement_cfg.get("slot_margin", 0.045))
    usable = max(0.0, line_length - 2.0 * margin)
    max_slots = max(1, int(np.floor(usable / spacing)) + 1)
    slot_count = min(count, max_slots)

    if slot_count <= 1:
        offsets = [0.0]
    else:
        total_span = min(usable, spacing * (slot_count - 1))
        offsets = np.linspace(-total_span / 2.0, total_span / 2.0, slot_count)

    z = float(place_ws["table_z_mean"]) + float(args.place_z_offset)
    slots = []
    for idx, offset in enumerate(offsets):
        xy = center + float(offset) * line_dir
        slots.append(
            {
                "slot": idx,
                "place_point_base": [float(xy[0]), float(xy[1]), z],
                "line_offset": float(offset),
            }
        )
    return slots, {
        "line_center_xy": [float(center[0]), float(center[1])],
        "line_dir_xy": [float(line_dir[0]), float(line_dir[1])],
        "line_length": line_length,
        "slot_spacing": spacing,
        "slot_margin": margin,
        "max_slots": max_slots,
    }


def save_debug_outputs(output_dir, ts, pcd_pickup_crop, pcd_no_table, cluster_pcds, plan):
    output_dir.mkdir(parents=True, exist_ok=True)
    o3d.io.write_point_cloud(str(output_dir / f"{ts}_pickup_crop_workspace.ply"), pcd_pickup_crop)
    o3d.io.write_point_cloud(str(output_dir / f"{ts}_pickup_no_table.ply"), pcd_no_table)
    for label, pair in cluster_pcds.items():
        o3d.io.write_point_cloud(str(output_dir / f"{ts}_cluster_label_{label:02d}_raw.ply"), pair["raw"])
        o3d.io.write_point_cloud(str(output_dir / f"{ts}_cluster_label_{label:02d}_clean.ply"), pair["clean"])
    with open(output_dir / f"{ts}_dry_run_plan.json", "w", encoding="utf-8") as f:
        json.dump(plan, f, indent=2)


def detect_and_plan(args):
    T_base_marker = load_T_base_marker(args.calibration)
    workspaces, placement_cfg = load_workspaces(args.workspaces)
    pickup_ws = workspaces["pickup"]
    place_ws = workspaces["place"]

    detector_bundle = create_aruco_detector()
    camera = RealSenseCamera(
        width=args.width,
        height=args.height,
        fps=args.fps,
        stream_order=args.camera_stream_order,
    )

    print("\n========== Regrasp Dry Run Planner ==========")
    print("Perception/planning only: no Kortex connection and no robot motion.")
    print(f"MARKER_LENGTH={MARKER_LENGTH:.3f} m, target marker ID={TARGET_MARKER_ID}")
    print(f"Using workspaces: {args.workspaces}")

    try:
        camera.start(
            start_attempts=args.camera_start_attempts,
            retry_delay_s=args.camera_retry_delay,
            hardware_reset_on_fail=args.camera_hardware_reset_on_fail,
        )
        frames = None
        marker_detection = None
        for attempt in range(120):
            frames = camera.get_aligned_frames()
            if frames is None:
                continue
            marker_detection, corners, ids, rejected = detect_target_marker_pose(
                frames.color_bgr,
                frames.camera_matrix,
                frames.dist_coeffs,
                detector_bundle=detector_bundle,
            )
            if marker_detection is not None:
                break
            if attempt % 20 == 0:
                print(f"Waiting for marker ID {TARGET_MARKER_ID}... attempt {attempt}")

        if frames is None or marker_detection is None:
            raise RuntimeError(f"Could not detect ArUco marker ID {TARGET_MARKER_ID}.")

        valid_mask, edge_mask = make_depth_valid_mask(frames.depth_z16, frames.depth_scale_m, args)
        T_base_camera = T_base_marker @ invert_T(marker_detection.T_camera_marker)

        pcd_camera = build_camera_point_cloud(frames, valid_mask, args)
        pcd_base_full = transform_pcd(pcd_camera, T_base_camera)
        pcd_pickup_crop = crop_pcd_by_workspace(pcd_base_full, pickup_ws)
        pcd_no_table, plane_model, inliers = remove_table_plane(pcd_pickup_crop, args)
        pcd_no_table, near_table_removed = remove_near_table_residuals(pcd_no_table, plane_model, args)

        candidates, cluster_pcds, dbscan_summary = make_candidates(pcd_no_table, args)
        candidates = sort_candidates(candidates, args.pick_sort)
        slots, slot_info = place_centerline_slots(place_ws, placement_cfg, args, len(candidates))

        sequence = []
        for idx, candidate in enumerate(candidates[: len(slots)]):
            pick = candidate["center_base"]
            sequence.append(
                {
                    "step": idx,
                    "label": candidate["label"],
                    "pick_point_base": [float(v) for v in pick],
                    "place_point_base": slots[idx]["place_point_base"],
                    "raw_points": candidate["raw_points"],
                    "clean_points": candidate["clean_points"],
                    "extent": [float(v) for v in candidate["extent"]],
                }
            )

        ts = time.strftime("%Y%m%d_%H%M%S")
        plan = {
            "timestamp": ts,
            "mode": "dry_run_no_robot_motion",
            "workspaces": str(args.workspaces),
            "pickup_workspace_points": len(pcd_pickup_crop.points),
            "pickup_non_table_points": len(pcd_no_table.points),
            "near_table_removed": near_table_removed,
            "depth_edge_rejected_pixels": int(np.count_nonzero(edge_mask)),
            "dbscan": dbscan_summary,
            "slot_info": slot_info,
            "candidates": [
                {
                    "label": c["label"],
                    "center_base": [float(v) for v in c["center_base"]],
                    "raw_points": c["raw_points"],
                    "clean_points": c["clean_points"],
                    "extent": [float(v) for v in c["extent"]],
                }
                for c in candidates
            ],
            "sequence": sequence,
        }

        print("\nPickup candidates:")
        if not candidates:
            print("  none")
        for c in candidates:
            print(
                f"  label={c['label']:02d}, center=[{c['center_base'][0]:.4f}, {c['center_base'][1]:.4f}, {c['center_base'][2]:.4f}], "
                f"raw={c['raw_points']}, clean={c['clean_points']}, extent={[round(float(v), 4) for v in c['extent']]}"
            )

        print("\nPlacement centerline:")
        print(f"  center_xy={slot_info['line_center_xy']}")
        print(f"  dir_xy={slot_info['line_dir_xy']}  (along long edge, perpendicular to a short edge)")
        print(f"  max_slots={slot_info['max_slots']}, requested={len(candidates)}, planned={len(sequence)}")

        print("\nPlanned sequence:")
        if not sequence:
            print("  no pick/place steps")
        for step in sequence:
            p = step["pick_point_base"]
            q = step["place_point_base"]
            print(
                f"  step={step['step']}: label={step['label']:02d} "
                f"pick=[{p[0]:.4f}, {p[1]:.4f}, {p[2]:.4f}] -> "
                f"place=[{q[0]:.4f}, {q[1]:.4f}, {q[2]:.4f}]"
            )

        args.output_dir.mkdir(parents=True, exist_ok=True)
        plan_path = args.output_dir / f"{ts}_dry_run_plan.json"
        with open(plan_path, "w", encoding="utf-8") as f:
            json.dump(plan, f, indent=2)
        print(f"\nSaved dry-run plan: {plan_path}")

        if args.save_debug_pcd:
            save_debug_outputs(args.output_dir, ts, pcd_pickup_crop, pcd_no_table, cluster_pcds, plan)
            print(f"Saved debug point clouds to: {args.output_dir}")

        return plan

    finally:
        camera.stop()


def main():
    args = parse_args()
    detect_and_plan(args)


if __name__ == "__main__":
    main()

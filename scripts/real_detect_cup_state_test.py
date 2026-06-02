#!/usr/bin/env python3
"""RealSense + ArUco point-cloud cup-state detection test.

中文说明：早期杯子状态检测脚本，使用 RealSense 点云、ArUco 动态标定、
桌面去除、DBSCAN 和 PCA/几何规则判断 upright/lying/unknown。该脚本只做感知。

This script is perception-only and does not move the Kinova arm.

Pipeline:
1. RealSense gets aligned color + depth in the color camera frame.
2. OpenCV ArUco detects marker ID 23 and estimates T_camera_marker.
3. Load T_base_marker from configs/real_calibration.yaml.
4. Compute T_base_camera = T_base_marker @ inv(T_camera_marker).
5. Convert aligned RGB-D to a point cloud, transform it into the Kinova base frame.
6. Crop workspace, remove table plane, run DBSCAN clustering.
7. Print each cluster's bounds and first-pass cup state: upright / lying / unknown.
8. Optionally save debug point clouds.
"""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass
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


@dataclass
class ClusterCandidate:
    label: int
    num_points: int
    center_base: np.ndarray
    min_bound: np.ndarray
    max_bound: np.ndarray
    extent: np.ndarray
    obb_extent: np.ndarray | None
    height: float
    max_xy: float
    state: str
    distance_xy: float
    isolation_xy: float


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)

    # Accepted for command compatibility with the real pick-place scripts.
    parser.add_argument("--ip", type=str, default="192.168.1.10", help="Unused; this perception-only script does not connect to Kortex.")

    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=15)
    parser.add_argument(
        "--camera-stream-order",
        choices=["color_first", "depth_first", "both"],
        default="color_first",
        help="Order used when enabling RealSense color/depth streams.",
    )
    parser.add_argument("--camera-start-attempts", type=int, default=3, help="Retry RealSense pipeline startup this many times.")
    parser.add_argument("--camera-retry-delay", type=float, default=2.0, help="Seconds to wait between RealSense startup attempts.")
    parser.add_argument(
        "--camera-hardware-reset-on-fail",
        action="store_true",
        help="After the first failed startup attempt, request a RealSense hardware reset before retrying.",
    )
    parser.add_argument(
        "--calibration",
        type=Path,
        default=PROJECT_ROOT / "configs" / "real_calibration.yaml",
        help="YAML file containing T_base_marker.",
    )
    parser.add_argument(
        "--table-workspace",
        type=Path,
        default=PROJECT_ROOT / "configs" / "table_workspace.yaml",
        help="Optional YAML from scripts/calibrate_table_workspace.py. Used if the file exists.",
    )
    parser.add_argument(
        "--ignore-table-workspace",
        action="store_true",
        help="Ignore --table-workspace and use rectangular crop-x/y/z bounds only.",
    )

    parser.add_argument("--depth-trunc", type=float, default=1.20, help="Ignore depth points farther than this camera z distance.")
    parser.add_argument("--voxel-size", type=float, default=0.004, help="Voxel size for point cloud downsampling in meters.")
    parser.add_argument("--table-distance-threshold", type=float, default=0.008, help="RANSAC table plane distance threshold in meters.")
    parser.add_argument("--cluster-eps", type=float, default=0.03, help="DBSCAN eps in meters.")
    parser.add_argument("--cluster-min-points", type=int, default=50, help="DBSCAN minimum points.")
    parser.add_argument("--min-points", type=int, default=80, help="Minimum points for a reported target candidate.")
    parser.add_argument(
        "--select-strategy",
        choices=["largest", "nearest", "isolated"],
        default="largest",
        help="Which reported cluster to select.",
    )

    parser.add_argument("--crop-x-min", type=float, default=0.10)
    parser.add_argument("--crop-x-max", type=float, default=0.75)
    parser.add_argument("--crop-y-min", type=float, default=-0.50)
    parser.add_argument("--crop-y-max", type=float, default=0.50)
    parser.add_argument("--crop-z-min", type=float, default=-0.08)
    parser.add_argument("--crop-z-max", type=float, default=0.35)

    parser.add_argument("--upright-ratio", type=float, default=1.25)
    parser.add_argument("--lying-ratio", type=float, default=1.40)

    parser.add_argument("--save-debug-pcd", action="store_true", help="Save debug point clouds under debug_pointclouds/cup_state/.")
    return parser.parse_args()


def load_T_base_marker(path: Path) -> np.ndarray:
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    T_base_marker = np.asarray(data["T_base_marker"], dtype=np.float64)
    if T_base_marker.shape != (4, 4):
        raise ValueError("T_base_marker must be 4x4.")
    return T_base_marker


def load_table_workspace(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    workspace = data.get("table_workspace", data)
    polygon = np.asarray(workspace["polygon_base_xy"], dtype=np.float64)
    if polygon.ndim != 2 or polygon.shape[1] != 2 or polygon.shape[0] < 3:
        raise ValueError("table workspace polygon_base_xy must be an Nx2 polygon with at least three points.")
    z_min = float(workspace["z_min"])
    z_max = float(workspace["z_max"])
    return {
        "polygon_base_xy": polygon,
        "z_min": z_min,
        "z_max": z_max,
        "path": path,
    }


def points_in_polygon_xy(points_xy, polygon_xy):
    polygon = np.asarray(polygon_xy, dtype=np.float32).reshape(-1, 1, 2)
    return np.array(
        [cv2.pointPolygonTest(polygon, (float(p[0]), float(p[1])), False) >= 0 for p in points_xy],
        dtype=bool,
    )


def build_camera_point_cloud(frames, args):
    """Create an Open3D point cloud in the RealSense/OpenCV color camera frame."""
    depth_m = frames.depth_z16.astype(np.float64) * float(frames.depth_scale_m)
    h, w = depth_m.shape

    us, vs = np.meshgrid(np.arange(w), np.arange(h))
    valid = np.isfinite(depth_m) & (depth_m > 0.05) & (depth_m < args.depth_trunc)

    z = depth_m[valid]
    x = (us[valid].astype(np.float64) - frames.camera_matrix[0, 2]) * z / frames.camera_matrix[0, 0]
    y = (vs[valid].astype(np.float64) - frames.camera_matrix[1, 2]) * z / frames.camera_matrix[1, 1]
    points = np.stack([x, y, z], axis=1)

    color_rgb = cv2.cvtColor(frames.color_bgr, cv2.COLOR_BGR2RGB)
    colors = color_rgb[valid].astype(np.float64) / 255.0

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


def crop_pcd_base(pcd_base_full, args):
    points_full = np.asarray(pcd_base_full.points)
    colors_full = np.asarray(pcd_base_full.colors)
    workspace = getattr(args, "table_workspace_data", None)
    if workspace is not None:
        polygon_mask = points_in_polygon_xy(points_full[:, :2], workspace["polygon_base_xy"])
        mask = (
            polygon_mask
            & (points_full[:, 2] >= workspace["z_min"])
            & (points_full[:, 2] <= workspace["z_max"])
        )
    else:
        mask = (
            (points_full[:, 0] >= args.crop_x_min)
            & (points_full[:, 0] <= args.crop_x_max)
            & (points_full[:, 1] >= args.crop_y_min)
            & (points_full[:, 1] <= args.crop_y_max)
            & (points_full[:, 2] >= args.crop_z_min)
            & (points_full[:, 2] <= args.crop_z_max)
        )

    pcd_base_crop = o3d.geometry.PointCloud()
    pcd_base_crop.points = o3d.utility.Vector3dVector(points_full[mask])
    if colors_full.shape[0] == points_full.shape[0]:
        pcd_base_crop.colors = o3d.utility.Vector3dVector(colors_full[mask])
    return pcd_base_crop


def remove_table_plane(pcd_base_crop, args):
    if len(pcd_base_crop.points) < 100:
        return pcd_base_crop, None, []
    try:
        plane_model, inliers = pcd_base_crop.segment_plane(
            distance_threshold=args.table_distance_threshold,
            ransac_n=3,
            num_iterations=1000,
        )
        return pcd_base_crop.select_by_index(inliers, invert=True), plane_model, inliers
    except RuntimeError:
        return pcd_base_crop, None, []


def classify_cup_state(extent, args):
    height = float(extent[2])
    max_xy = float(max(extent[0], extent[1]))

    if max_xy <= 1e-9 or height <= 1e-9:
        return "unknown", height, max_xy
    if height > args.upright_ratio * max_xy:
        return "upright", height, max_xy
    if max_xy > args.lying_ratio * height:
        return "lying", height, max_xy
    return "unknown", height, max_xy


def safe_obb_extent(cluster_pcd):
    if len(cluster_pcd.points) < 4:
        return None
    try:
        obb = cluster_pcd.get_oriented_bounding_box()
        return np.asarray(obb.extent, dtype=np.float64)
    except RuntimeError:
        return None


def compute_isolation_xy(centers):
    if len(centers) <= 1:
        return [float("inf")] * len(centers)

    isolations = []
    for idx, center in enumerate(centers):
        distances = [
            float(np.linalg.norm(center[:2] - other[:2]))
            for other_idx, other in enumerate(centers)
            if other_idx != idx
        ]
        isolations.append(min(distances))
    return isolations


def make_cluster_candidate(label, idx, pcd_no_table, args):
    cluster_pcd = pcd_no_table.select_by_index(idx)
    cluster_points = np.asarray(cluster_pcd.points)
    min_bound = np.min(cluster_points, axis=0)
    max_bound = np.max(cluster_points, axis=0)
    extent = max_bound - min_bound
    center_base = (min_bound + max_bound) / 2.0
    state, height, max_xy = classify_cup_state(extent, args)

    return ClusterCandidate(
        label=int(label),
        num_points=int(len(idx)),
        center_base=center_base,
        min_bound=min_bound,
        max_bound=max_bound,
        extent=extent,
        obb_extent=safe_obb_extent(cluster_pcd),
        height=height,
        max_xy=max_xy,
        state=state,
        distance_xy=float(np.linalg.norm(center_base[:2])),
        isolation_xy=0.0,
    )


def select_candidate(candidates, args):
    if not candidates:
        return None
    if args.select_strategy == "nearest":
        return min(candidates, key=lambda c: c.distance_xy)
    if args.select_strategy == "isolated":
        return max(candidates, key=lambda c: c.isolation_xy)
    return max(candidates, key=lambda c: c.num_points)


def fmt_vec(vec):
    if vec is None:
        return "None"
    return "[" + ", ".join(f"{float(v):.3f}" for v in vec) + "]"


def print_candidate(candidate):
    print(f"label={candidate.label}")
    print(f"num_points={candidate.num_points}")
    print(f"center_base={fmt_vec(candidate.center_base)}")
    print(f"extent={fmt_vec(candidate.extent)}")
    print(f"obb_extent={fmt_vec(candidate.obb_extent)}")
    print(f"height={candidate.height:.3f}")
    print(f"max_xy={candidate.max_xy:.3f}")
    print(f"isolation_xy={candidate.isolation_xy:.3f}")
    print(f"state={candidate.state}")
    print("")


def save_debug_point_clouds(
    pcd_base_full,
    pcd_base_crop,
    pcd_no_table,
    pcd_clusters,
    candidates,
):
    debug_dir = PROJECT_ROOT / "debug_pointclouds" / "cup_state"
    debug_dir.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")

    o3d.io.write_point_cloud(str(debug_dir / f"{ts}_pcd_base_full.ply"), pcd_base_full)
    o3d.io.write_point_cloud(str(debug_dir / f"{ts}_pcd_base_crop.ply"), pcd_base_crop)
    o3d.io.write_point_cloud(str(debug_dir / f"{ts}_pcd_base_no_table.ply"), pcd_no_table)

    for candidate in candidates:
        cluster_pcd = pcd_clusters[candidate.label]
        filename = f"{ts}_cluster_label_{candidate.label:02d}_{candidate.state}.ply"
        o3d.io.write_point_cloud(str(debug_dir / filename), cluster_pcd)

    print(f"Saved debug point clouds to: {debug_dir}")


def detect_cup_state(args, T_base_marker):
    detector_bundle = create_aruco_detector()
    camera = RealSenseCamera(
        width=args.width,
        height=args.height,
        fps=args.fps,
        stream_order=args.camera_stream_order,
    )

    print("\n========== Real Cup State Detection Test ==========")
    print("Perception only: no Kortex connection and no robot motion.")
    print(f"MARKER_LENGTH={MARKER_LENGTH:.3f} m, target marker ID={TARGET_MARKER_ID}")
    workspace = getattr(args, "table_workspace_data", None)
    if workspace is not None:
        print(f"Using table workspace: {workspace['path']}")
        print(f"workspace z: [{workspace['z_min']:.3f}, {workspace['z_max']:.3f}]")
    print("\nT_base_marker:")
    print(T_base_marker)
    print("\nStarting RealSense and looking for ArUco marker...")

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

        print(f"Detected marker ID {TARGET_MARKER_ID}.")
        print("T_camera_marker:")
        print(marker_detection.T_camera_marker)

        T_base_camera = T_base_marker @ invert_T(marker_detection.T_camera_marker)
        print("T_base_camera:")
        print(T_base_camera)

        pcd_camera = build_camera_point_cloud(frames, args)
        pcd_base_full = transform_pcd(pcd_camera, T_base_camera)
        pcd_base_crop = crop_pcd_base(pcd_base_full, args)
        pcd_no_table, plane_model, inliers = remove_table_plane(pcd_base_crop, args)

        print(f"\nFull base-frame points: {len(pcd_base_full.points)}")
        print(f"Cropped workspace points: {len(pcd_base_crop.points)}")
        print(f"Table plane model: {plane_model}")
        print(f"Table inliers: {len(inliers)}")
        print(f"Non-table points: {len(pcd_no_table.points)}")

        if len(pcd_no_table.points) < args.cluster_min_points:
            raise RuntimeError("Not enough non-table points after plane removal.")

        labels = np.asarray(
            pcd_no_table.cluster_dbscan(
                eps=args.cluster_eps,
                min_points=args.cluster_min_points,
                print_progress=False,
            )
        )
        if labels.size == 0 or labels.max() < 0:
            raise RuntimeError("DBSCAN found no clusters. Try increasing --cluster-eps or reducing --cluster-min-points.")

        candidates_all = []
        pcd_clusters = {}
        for label in sorted(set(labels.tolist())):
            if label < 0:
                continue
            idx = np.where(labels == label)[0]
            cluster_pcd = pcd_no_table.select_by_index(idx)
            pcd_clusters[int(label)] = cluster_pcd
            candidates_all.append(make_cluster_candidate(label, idx, pcd_no_table, args))

        isolations = compute_isolation_xy([c.center_base for c in candidates_all])
        for candidate, isolation_xy in zip(candidates_all, isolations):
            candidate.isolation_xy = isolation_xy

        candidates_reported = [c for c in candidates_all if c.num_points >= args.min_points]

        print("\nCluster candidates:")
        if not candidates_reported:
            print(f"No cluster has at least --min-points={args.min_points}.")
        for candidate in candidates_reported:
            print_candidate(candidate)

        selected = select_candidate(candidates_reported, args)
        if selected is None:
            raise RuntimeError("No reportable cluster found. Try reducing --min-points or tuning crop/DBSCAN parameters.")

        print("Selected target:")
        print(f"label={selected.label}")
        print(f"state={selected.state}")
        print(f"center_base={fmt_vec(selected.center_base)}")
        print(f"extent={fmt_vec(selected.extent)}")
        print(f"num_points={selected.num_points}")

        if args.save_debug_pcd:
            save_debug_point_clouds(
                pcd_base_full,
                pcd_base_crop,
                pcd_no_table,
                pcd_clusters,
                candidates_reported,
            )

        return selected, candidates_reported

    finally:
        camera.stop()


def main():
    args = parse_args()
    args.table_workspace_data = None
    if not args.ignore_table_workspace and args.table_workspace.exists():
        args.table_workspace_data = load_table_workspace(args.table_workspace)
    elif not args.ignore_table_workspace:
        print(f"Table workspace file not found, using rectangular crop bounds: {args.table_workspace}")
    T_base_marker = load_T_base_marker(args.calibration)
    detect_cup_state(args, T_base_marker)


if __name__ == "__main__":
    main()

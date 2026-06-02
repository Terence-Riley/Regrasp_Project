#!/usr/bin/env python3
"""Real automatic pick-place with point-cloud cup-center detection.

中文说明：早期自动 cluster 检测版本 pick-place 脚本。它用 RealSense 点云、
桌面去除和 DBSCAN 自动找杯子中心，然后执行抓取和放置。该脚本会移动真实
机械臂和夹爪。

This script is based on the working real_pick_place_auto.py pipeline, but replaces
manual clicking with an automatic RGB-D point-cloud detector:

1. RealSense gets aligned color + depth.
2. OpenCV ArUco detects marker ID 23 and estimates T_camera_marker.
3. Use T_base_marker from configs/real_calibration.yaml to compute T_base_camera.
4. Convert aligned depth to a point cloud, transform it into the Kinova base frame.
5. Crop workspace, remove the table plane, run DBSCAN clustering.
6. Select one cup-like cluster and use its 3D center as cup_pos_base.
7. Execute pick-place using the proven Kortex path:
       ComputeInverseKinematics -> reach_joint_angles

Keep E-stop ready. First run with --dry-run and inspect the printed candidate clusters.
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

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Reuse the proven perception + Kortex IK/joint-action utilities.
from scripts.real_move_above_cup_test import (  # noqa: E402
    BaseClient,
    BaseCyclicClient,
    Base_pb2,
    KServerException,
    compute_ik_official,
    get_tool_pose_from_feedback,
    joint_angles_to_list,
    load_T_base_marker,
    move_to_joint_angles_official,
    normalize_joint_angles_0_360,
    shortest_angle_delta_deg,
    utilities,
)
from perception.aruco_detector import (  # noqa: E402
    MARKER_LENGTH,
    TARGET_MARKER_ID,
    create_aruco_detector,
    detect_target_marker_pose,
)
from perception.realsense_camera import RealSenseCamera  # noqa: E402
from utils.transform_utils import invert_T, make_T  # noqa: E402


WINDOW_NAME = "Auto Cluster Pick-Place - RealSense RGB-D"
TIMEOUT_DURATION = 30


@dataclass
class ClusterCandidate:
    label: int
    num_points: int
    center_base: np.ndarray
    min_bound: np.ndarray
    max_bound: np.ndarray
    extent: np.ndarray
    distance_xy: float


# ============================================================
# Argument parsing
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)

    # Keep these names compatible with Kinova official utilities.DeviceConnection.
    parser.add_argument("--ip", type=str, default="192.168.1.10")
    parser.add_argument("-u", "--username", type=str, default="admin")
    parser.add_argument("-p", "--password", type=str, default="admin")

    # RealSense / perception parameters
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=15)
    parser.add_argument(
        "--calibration",
        type=Path,
        default=PROJECT_ROOT / "configs" / "real_calibration.yaml",
        help="YAML file containing T_base_marker.",
    )

    # Automatic point-cloud detector parameters
    parser.add_argument(
        "--perception-mode",
        choices=["cluster"],
        default="cluster",
        help="Currently uses automatic point-cloud clustering only.",
    )
    parser.add_argument("--depth-trunc", type=float, default=1.20, help="Ignore depth points farther than this camera z distance.")
    parser.add_argument("--voxel-size", type=float, default=0.004, help="Voxel size for point cloud downsampling in meters.")
    parser.add_argument("--plane-threshold", type=float, default=0.008, help="RANSAC table plane distance threshold in meters.")
    parser.add_argument("--cluster-eps", type=float, default=0.025, help="DBSCAN eps in meters.")
    parser.add_argument("--cluster-min-points", type=int, default=80, help="DBSCAN minimum points.")
    parser.add_argument(
        "--select-strategy",
        choices=["largest", "nearest"],
        default="largest",
        help="Which valid cup cluster to pick.",
    )
    parser.add_argument("--marker-exclusion-radius", type=float, default=0.13, help="Ignore clusters close to the ArUco marker center in XY.")

    # Workspace crop in Kinova base frame. Tune these if needed.
    parser.add_argument("--crop-x-min", type=float, default=0.12)
    parser.add_argument("--crop-x-max", type=float, default=0.75)
    parser.add_argument("--crop-y-min", type=float, default=-0.50)
    parser.add_argument("--crop-y-max", type=float, default=0.50)
    parser.add_argument("--crop-z-min", type=float, default=-0.08)
    parser.add_argument("--crop-z-max", type=float, default=0.35)

    # Cup cluster filters in base frame.
    parser.add_argument("--cup-min-height", type=float, default=0.025)
    parser.add_argument("--cup-max-height", type=float, default=0.20)
    parser.add_argument("--cup-min-xy-extent", type=float, default=0.025)
    parser.add_argument("--cup-max-xy-extent", type=float, default=0.16)

    parser.add_argument("--save-debug-pcd", action="store_true", help="Save debug point clouds under debug_pointclouds/.")

    # Motion heights relative to estimated cup center.
    parser.add_argument("--pre-grasp-height", type=float, default=0.20)
    parser.add_argument(
        "--grasp-height",
        type=float,
        default=0.03,
        help="Z offset above estimated cup center for final grasp approach. Use 0.00 if cup center is correct.",
    )
    parser.add_argument("--max-z-step", type=float, default=0.02)

    # Gripper parameters. Convention: 0=open, 1=closed.
    parser.add_argument("--open-value", type=float, default=0.0)
    parser.add_argument("--close-value", type=float, default=0.8, help="Start with 0.35~0.60; 1.0 is fully closed.")
    parser.add_argument("--gripper-step", type=float, default=0.05)
    parser.add_argument("--gripper-settle-time", type=float, default=0.20)
    parser.add_argument("--hold-after-lift", type=float, default=2.0)

    # Place target. x/y are fixed base-frame coordinates.
    parser.add_argument("--place-x", type=float, default=0.30)
    parser.add_argument("--place-y", type=float, default=-0.25)
    parser.add_argument(
        "--place-z",
        type=float,
        default=None,
        help="Base-frame z reference for place target. If omitted, uses detected cup_pos_base[2].",
    )
    parser.add_argument("--pre-place-height", type=float, default=0.20)
    parser.add_argument("--place-height", type=float, default=None, help="If omitted, uses --grasp-height.")
    parser.add_argument("--retreat-height", type=float, default=None, help="If omitted, uses --pre-place-height.")
    parser.add_argument("--skip-place", action="store_true", help="Only pick/lift; do not place automatically.")
    parser.add_argument("--release-after-lift", action="store_true", help="If --skip-place is used, ask whether to release after lift.")

    parser.add_argument("--dry-run", action="store_true", help="Perception only; do not move robot.")
    parser.add_argument("--auto", action="store_true", help="Do not pause before every motion segment.")
    return parser.parse_args()


# ============================================================
# Motion utilities
# ============================================================

def make_z_segment_positions(start_pos, end_pos, max_z_step):
    start_pos = np.asarray(start_pos, dtype=np.float64)
    end_pos = np.asarray(end_pos, dtype=np.float64)
    dz = float(end_pos[2] - start_pos[2])
    steps = max(1, int(np.ceil(abs(dz) / float(max_z_step))))
    return [start_pos + (end_pos - start_pos) * (idx / steps) for idx in range(1, steps + 1)]


def assert_valid_position(name, pos):
    pos = np.asarray(pos, dtype=np.float64)
    if not np.all(np.isfinite(pos)):
        raise ValueError(f"{name} contains NaN or Inf: {pos}")


def execute_ik_joint_pose(base, base_cyclic, target_pose, label):
    """Compute IK for target tool pose and execute with official reach_joint_angles."""
    print(f"\n========== {label} ==========")
    print("Target tool pose [x,y,z,theta_x,theta_y,theta_z]:")
    print(target_pose)

    try:
        target_joints = compute_ik_official(base, target_pose)
    except KServerException as ex:
        print("Unable to compute inverse kinematics.")
        print("Error_code:{} , Sub_error_code:{} ".format(ex.get_error_code(), ex.get_error_sub_code()))
        print("KServerException:", ex)
        return False

    print("IK target joint angles raw [deg]:")
    print(joint_angles_to_list(target_joints))
    target_joints = normalize_joint_angles_0_360(base, target_joints)
    print("IK target joint angles normalized to [0, 360) [deg]:")
    print(joint_angles_to_list(target_joints))

    joints_before = base.GetMeasuredJointAngles()
    pose_before = get_tool_pose_from_feedback(base_cyclic)

    ok = move_to_joint_angles_official(base, target_joints)

    joints_after = base.GetMeasuredJointAngles()
    pose_after = get_tool_pose_from_feedback(base_cyclic)
    before_values = joint_angles_to_list(joints_before)
    after_values = joint_angles_to_list(joints_after)
    deltas = [shortest_angle_delta_deg(after, before) for before, after in zip(before_values, after_values)]

    print("Measured joint deltas shortest-path [deg]:")
    print(deltas)
    print("Tool pose after [x,y,z,theta_x,theta_y,theta_z]:")
    print(pose_after)
    print("Measured tool position delta [m]:")
    print(pose_after[:3] - pose_before[:3])
    return ok


# ============================================================
# Gripper utilities
# ============================================================

def clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def get_measured_gripper_position(base):
    request = Base_pb2.GripperRequest()
    request.mode = Base_pb2.GRIPPER_POSITION
    measure = base.GetMeasuredGripperMovement(request)
    if len(measure.finger) == 0:
        return None
    return float(measure.finger[0].value)


def send_gripper_position(base, position: float, finger_id: int = 1):
    command = Base_pb2.GripperCommand()
    command.mode = Base_pb2.GRIPPER_POSITION
    finger = command.gripper.finger.add()
    finger.finger_identifier = finger_id
    finger.value = clamp01(position)
    base.SendGripperCommand(command)


def ramp_gripper_to(base, target: float, step: float, settle_time: float):
    target = clamp01(target)
    step = abs(float(step))
    if step <= 0:
        raise ValueError("--gripper-step must be > 0")

    current = get_measured_gripper_position(base)
    if current is None:
        print("Measured gripper position unavailable; sending target directly.")
        send_gripper_position(base, target)
        time.sleep(settle_time)
        return

    print(f"Measured current gripper position: {current:.3f}")
    print(f"Target gripper position: {target:.3f}")

    if abs(target - current) < 1e-3:
        print("Gripper already near target.")
        return

    direction = 1.0 if target > current else -1.0
    value = current
    while True:
        value = min(target, value + step) if direction > 0 else max(target, value - step)
        print(f"Sending gripper position: {value:.3f}")
        send_gripper_position(base, value)
        time.sleep(settle_time)
        if abs(value - target) < 1e-6:
            break


def open_gripper(base, args):
    print("\n========== Open gripper ==========")
    ramp_gripper_to(base, args.open_value, args.gripper_step, args.gripper_settle_time)


def close_gripper(base, args):
    print("\n========== Close gripper ==========")
    ramp_gripper_to(base, args.close_value, args.gripper_step, args.gripper_settle_time)


# ============================================================
# Point cloud object detection
# ============================================================

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

    # BGR -> RGB, normalize
    color_rgb = cv2.cvtColor(frames.color_bgr, cv2.COLOR_BGR2RGB)
    colors = color_rgb[valid].astype(np.float64) / 255.0

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    if args.voxel_size > 0:
        pcd = pcd.voxel_down_sample(args.voxel_size)
    return pcd


def transform_pcd(pcd, T):
    out = o3d.geometry.PointCloud(pcd)
    out.transform(np.asarray(T, dtype=np.float64).reshape(4, 4))
    return out


def crop_points_base(points, args):
    mask = (
        (points[:, 0] >= args.crop_x_min)
        & (points[:, 0] <= args.crop_x_max)
        & (points[:, 1] >= args.crop_y_min)
        & (points[:, 1] <= args.crop_y_max)
        & (points[:, 2] >= args.crop_z_min)
        & (points[:, 2] <= args.crop_z_max)
    )
    return mask


def remove_table_plane(pcd_base, args):
    if len(pcd_base.points) < 100:
        return pcd_base, None, []
    try:
        plane_model, inliers = pcd_base.segment_plane(
            distance_threshold=args.plane_threshold,
            ransac_n=3,
            num_iterations=1000,
        )
        pcd_no_table = pcd_base.select_by_index(inliers, invert=True)
        return pcd_no_table, plane_model, inliers
    except RuntimeError:
        return pcd_base, None, []


def is_valid_cup_candidate(candidate: ClusterCandidate, T_base_marker, args):
    ex, ey, ez = candidate.extent
    xy_extent = max(ex, ey)

    if candidate.num_points < args.cluster_min_points:
        return False
    if not (args.cup_min_height <= ez <= args.cup_max_height):
        return False
    if not (args.cup_min_xy_extent <= xy_extent <= args.cup_max_xy_extent):
        return False

    marker_xy = np.asarray(T_base_marker[:2, 3], dtype=np.float64)
    dist_to_marker = np.linalg.norm(candidate.center_base[:2] - marker_xy)
    if dist_to_marker < args.marker_exclusion_radius:
        return False

    return True


def select_candidate(candidates, args):
    if not candidates:
        return None
    if args.select_strategy == "nearest":
        return min(candidates, key=lambda c: c.distance_xy)
    return max(candidates, key=lambda c: c.num_points)


def detect_cup_center_cluster(args, T_base_marker):
    """Return (T_base_object, cup_pos_base) using automatic point-cloud clustering."""
    detector_bundle = create_aruco_detector()
    camera = RealSenseCamera(width=args.width, height=args.height, fps=args.fps)

    print("\n========== Automatic point-cloud cup detection ==========")
    print("Starting RealSense and looking for ArUco marker...")

    try:
        camera.start()
        marker_detection = None
        frames = None

        # Wait for a frame where the marker is visible.
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

        pcd_camera = build_camera_point_cloud(frames, args)
        pcd_base_full = transform_pcd(pcd_camera, T_base_camera)

        points_full = np.asarray(pcd_base_full.points)
        colors_full = np.asarray(pcd_base_full.colors)
        crop_mask = crop_points_base(points_full, args)
        pcd_base_crop = o3d.geometry.PointCloud()
        pcd_base_crop.points = o3d.utility.Vector3dVector(points_full[crop_mask])
        if colors_full.shape[0] == points_full.shape[0]:
            pcd_base_crop.colors = o3d.utility.Vector3dVector(colors_full[crop_mask])

        print(f"Full base-frame points: {len(points_full)}")
        print(f"Cropped workspace points: {len(pcd_base_crop.points)}")

        pcd_no_table, plane_model, inliers = remove_table_plane(pcd_base_crop, args)
        print(f"Table plane model: {plane_model}")
        print(f"Non-table points: {len(pcd_no_table.points)}")

        if len(pcd_no_table.points) < args.cluster_min_points:
            raise RuntimeError("Not enough non-table points after plane removal. Try increasing crop_z_max or reducing plane-threshold.")

        labels = np.asarray(pcd_no_table.cluster_dbscan(eps=args.cluster_eps, min_points=args.cluster_min_points, print_progress=False))
        if labels.size == 0 or labels.max() < 0:
            raise RuntimeError("DBSCAN found no clusters. Try increasing --cluster-eps or reducing --cluster-min-points.")

        points = np.asarray(pcd_no_table.points)
        candidates_all = []
        candidates_valid = []
        for label in sorted(set(labels.tolist())):
            if label < 0:
                continue
            idx = np.where(labels == label)[0]
            cluster_points = points[idx]
            min_bound = np.min(cluster_points, axis=0)
            max_bound = np.max(cluster_points, axis=0)
            center = (min_bound + max_bound) / 2.0
            extent = max_bound - min_bound
            candidate = ClusterCandidate(
                label=int(label),
                num_points=int(len(idx)),
                center_base=center,
                min_bound=min_bound,
                max_bound=max_bound,
                extent=extent,
                distance_xy=float(np.linalg.norm(center[:2])),
            )
            candidates_all.append(candidate)
            if is_valid_cup_candidate(candidate, T_base_marker, args):
                candidates_valid.append(candidate)

        print("\nCluster candidates:")
        for c in candidates_all:
            valid = "VALID" if c in candidates_valid else "reject"
            print(
                f"  label={c.label:2d} points={c.num_points:5d} {valid:6s} "
                f"center=[{c.center_base[0]:.3f},{c.center_base[1]:.3f},{c.center_base[2]:.3f}] "
                f"extent=[{c.extent[0]:.3f},{c.extent[1]:.3f},{c.extent[2]:.3f}]"
            )

        selected = select_candidate(candidates_valid, args)
        if selected is None:
            raise RuntimeError(
                "No valid cup-like cluster found. Try tuning crop ranges, cluster params, or marker-exclusion-radius."
            )

        cup_pos_base = selected.center_base.copy()
        T_base_object = make_T(translation=cup_pos_base)

        print("\nSelected cup candidate:")
        print(
            f"label={selected.label}, points={selected.num_points}, "
            f"center={cup_pos_base}, extent={selected.extent}"
        )
        print("T_base_object:")
        print(T_base_object)

        if args.save_debug_pcd:
            debug_dir = PROJECT_ROOT / "debug_pointclouds"
            debug_dir.mkdir(parents=True, exist_ok=True)
            ts = time.strftime("%Y%m%d_%H%M%S")
            o3d.io.write_point_cloud(str(debug_dir / f"{ts}_pcd_camera.ply"), pcd_camera)
            o3d.io.write_point_cloud(str(debug_dir / f"{ts}_pcd_base_full.ply"), pcd_base_full)
            o3d.io.write_point_cloud(str(debug_dir / f"{ts}_pcd_base_crop.ply"), pcd_base_crop)
            o3d.io.write_point_cloud(str(debug_dir / f"{ts}_pcd_base_no_table.ply"), pcd_no_table)
            print(f"Saved debug point clouds to: {debug_dir}")

        # Show debug image with marker axis and selected target text.
        vis = frames.color_bgr.copy()
        cv2.drawFrameAxes(
            vis,
            frames.camera_matrix,
            frames.dist_coeffs,
            marker_detection.rvec,
            marker_detection.tvec,
            MARKER_LENGTH * 0.5,
        )
        cv2.putText(
            vis,
            f"selected cup base=[{cup_pos_base[0]:.3f},{cup_pos_base[1]:.3f},{cup_pos_base[2]:.3f}]",
            (20, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )
        depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(frames.depth_z16, alpha=0.03), cv2.COLORMAP_JET)
        cv2.imshow(WINDOW_NAME, np.hstack((vis, depth_colormap)))
        cv2.waitKey(1000)
        cv2.destroyWindow(WINDOW_NAME)

        return T_base_object, cup_pos_base

    finally:
        camera.stop()
        try:
            cv2.destroyWindow(WINDOW_NAME)
        except cv2.error:
            pass


# ============================================================
# Main
# ============================================================

def main():
    args = parse_args()

    if args.grasp_height >= args.pre_grasp_height:
        raise ValueError("--grasp-height must be lower than --pre-grasp-height.")

    # Fill optional place parameters before comparing them.
    if args.place_height is None:
        args.place_height = args.grasp_height
    if args.retreat_height is None:
        args.retreat_height = args.pre_place_height

    if args.place_height >= args.pre_place_height:
        raise ValueError("--place-height must be lower than --pre-place-height.")

    do_place = not args.skip_place
    T_base_marker = load_T_base_marker(args.calibration)

    print("========== Real Pick-Place Auto Cluster Test ==========")
    print("Default behavior: automatic point-cloud detection, pick, lift, place, release, and retreat.")
    print("Control path: RealSense/ArUco/point-cloud -> T_base_object -> Kortex IK -> reach_joint_angles.")
    print(f"MARKER_LENGTH={MARKER_LENGTH:.3f} m, target marker ID={TARGET_MARKER_ID}")
    print(f"pre_grasp_height={args.pre_grasp_height:.3f} m")
    print(f"grasp_height={args.grasp_height:.3f} m")
    print(f"max_z_step={args.max_z_step:.3f} m")
    print(f"open_value={args.open_value:.3f}, close_value={args.close_value:.3f}")
    print(f"place_x={args.place_x:.3f}, place_y={args.place_y:.3f}, place_z={args.place_z}")
    print(f"pre_place_height={args.pre_place_height:.3f}, place_height={args.place_height:.3f}, retreat_height={args.retreat_height:.3f}")
    print("\nT_base_marker:")
    print(T_base_marker)

    T_base_object, cup_pos_base = detect_cup_center_cluster(args, T_base_marker)

    pre_grasp_pos = cup_pos_base + np.array([0.0, 0.0, args.pre_grasp_height], dtype=np.float64)
    grasp_pos = cup_pos_base + np.array([0.0, 0.0, args.grasp_height], dtype=np.float64)
    lift_pos = pre_grasp_pos.copy()

    print("\n========== Accepted Automatic Target ==========")
    print("T_base_object:")
    print(T_base_object)
    print("cup_pos_base [m]:", cup_pos_base)
    print("pre_grasp_pos [m]:", pre_grasp_pos)
    print("grasp_pos [m]:", grasp_pos)
    print("lift_pos [m]:", lift_pos)

    # If user does not specify --place-z, use the detected cup center z.
    if args.place_z is None:
        args.place_z = float(cup_pos_base[2])
        print(f"\n--place-z not provided. Using cup_pos_base[2] as place_z: {args.place_z:.6f} m")

    if do_place:
        place_base = np.array([args.place_x, args.place_y, args.place_z], dtype=np.float64)
        pre_place_pos = place_base + np.array([0.0, 0.0, args.pre_place_height], dtype=np.float64)
        place_pos = place_base + np.array([0.0, 0.0, args.place_height], dtype=np.float64)
        retreat_pos = place_base + np.array([0.0, 0.0, args.retreat_height], dtype=np.float64)

        assert_valid_position("place_base", place_base)
        assert_valid_position("pre_place_pos", pre_place_pos)
        assert_valid_position("place_pos", place_pos)
        assert_valid_position("retreat_pos", retreat_pos)

        print("\nPlace mode enabled.")
        print("place_base [m]:", place_base)
        print("pre_place_pos [m]:", pre_place_pos)
        print("place_pos [m]:", place_pos)
        print("retreat_pos [m]:", retreat_pos)
    else:
        place_base = pre_place_pos = place_pos = retreat_pos = None
        print("\nPlace mode disabled. Script will stop after lift with gripper closed.")

    if args.dry_run:
        print("Dry run enabled. Exiting without robot motion.")
        return 0

    print("\nSafety checklist:")
    print(" - Keep E-stop ready.")
    print(" - First use --dry-run to verify the selected cluster center.")
    print(" - Use a light, non-fragile cup for the first test.")
    print(" - Make sure the path above the cup and place target is clear.")
    print(" - Start with conservative --grasp-height, --place-height, and --close-value.")
    answer = input("\nType PICKPLACE to connect and run the automatic pick-place test: ").strip()
    if answer != "PICKPLACE":
        print("Confirmation not received. Exiting.")
        return 1

    with utilities.DeviceConnection.createTcpConnection(args) as router:
        base = BaseClient(router)
        base_cyclic = BaseCyclicClient(router)

        current_pose = get_tool_pose_from_feedback(base_cyclic)
        print("\nCurrent tool pose [x,y,z,theta_x,theta_y,theta_z]:")
        print(current_pose)

        pre_grasp_pose = current_pose.copy(); pre_grasp_pose[:3] = pre_grasp_pos
        grasp_pose = current_pose.copy(); grasp_pose[:3] = grasp_pos
        lift_pose = current_pose.copy(); lift_pose[:3] = lift_pos

        # 1. Open gripper.
        if not args.auto:
            input("\nPress Enter to open gripper...")
        open_gripper(base, args)

        # 2. Move to pre-grasp.
        if not args.auto:
            input("\nPress Enter to move to pre-grasp...")
        if not execute_ik_joint_pose(base, base_cyclic, pre_grasp_pose, "Move to pre-grasp"):
            print("Failed to reach pre-grasp. Stopping.")
            return 2

        # 3. Descend in z segments.
        descent_positions = make_z_segment_positions(pre_grasp_pos, grasp_pos, args.max_z_step)
        for idx, pos in enumerate(descent_positions, start=1):
            target_pose = current_pose.copy()
            target_pose[:3] = pos
            print(f"\nNext descent segment {idx}/{len(descent_positions)} target z={pos[2]:.3f}")
            if not args.auto:
                input("Press Enter to execute this descent segment...")
            if not execute_ik_joint_pose(base, base_cyclic, target_pose, f"Descend segment {idx}"):
                print("Descent segment failed. Stopping before gripper close.")
                return 3

        # 4. Close gripper.
        if not args.auto:
            input("\nPress Enter to close gripper around the cup...")
        close_gripper(base, args)
        time.sleep(0.5)

        # 5. Lift in z segments.
        lift_positions = make_z_segment_positions(grasp_pos, lift_pos, args.max_z_step)
        for idx, pos in enumerate(lift_positions, start=1):
            target_pose = current_pose.copy()
            target_pose[:3] = pos
            print(f"\nNext lift segment {idx}/{len(lift_positions)} target z={pos[2]:.3f}")
            if not args.auto:
                input("Press Enter to execute this lift segment...")
            if not execute_ik_joint_pose(base, base_cyclic, target_pose, f"Lift segment {idx}"):
                print("Lift segment failed.")
                return 4

        print(f"\nLift completed. Holding for {args.hold_after_lift:.1f} s...")
        time.sleep(args.hold_after_lift)

        # Optional place sequence.
        if do_place:
            pre_place_pose = current_pose.copy(); pre_place_pose[:3] = pre_place_pos
            place_pose = current_pose.copy(); place_pose[:3] = place_pos
            retreat_pose = current_pose.copy(); retreat_pose[:3] = retreat_pos

            if not args.auto:
                input("\nPress Enter to move to pre-place...")
            if not execute_ik_joint_pose(base, base_cyclic, pre_place_pose, "Move to pre-place"):
                print("Failed to reach pre-place. Keeping gripper closed.")
                return 5

            place_positions = make_z_segment_positions(pre_place_pos, place_pos, args.max_z_step)
            for idx, pos in enumerate(place_positions, start=1):
                target_pose = current_pose.copy()
                target_pose[:3] = pos
                print(f"\nNext place descent segment {idx}/{len(place_positions)} target z={pos[2]:.3f}")
                if not args.auto:
                    input("Press Enter to execute this place descent segment...")
                if not execute_ik_joint_pose(base, base_cyclic, target_pose, f"Place descent segment {idx}"):
                    print("Place descent failed. Keeping gripper closed.")
                    return 6

            if not args.auto:
                input("\nPress Enter to open gripper and release cup...")
            open_gripper(base, args)

            if not args.auto:
                input("\nPress Enter to retreat upward...")
            if not execute_ik_joint_pose(base, base_cyclic, retreat_pose, "Retreat after place"):
                print("Retreat failed.")
                return 7

            print("\nAutomatic cluster pick-place sequence completed.")
            return 0

        # No place mode.
        if args.release_after_lift:
            answer = input("\nCup should now be lifted. Type RELEASE to open gripper now: ").strip()
            if answer == "RELEASE":
                open_gripper(base, args)
            else:
                print("Release not confirmed. Gripper remains closed.")
        else:
            print("\nPick/lift sequence completed. Gripper remains closed by default.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

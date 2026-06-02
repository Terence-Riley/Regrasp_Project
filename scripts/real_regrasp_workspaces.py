#!/usr/bin/env python3
"""Real regrasp from calibrated pickup workspace to calibrated place workspace.

中文说明：在校准好的杯子散落区和放置区之间执行真实重抓取循环。该脚本会
连接 RealSense 和 Kortex，检测杯子候选，规划抓取/放置点，并移动真实机械臂。
运行前必须确认工作空间安全，首次使用建议先运行 dry-run 脚本。

This script moves the Kinova arm. Keep E-stop ready.

It intentionally keeps the first version simple:
1. Detect cup candidates only inside the pickup workspace.
2. Use the cleaned cluster center as the pick point.
3. Use a fixed tool orientation, by default the current orientation at startup.
4. Place cups along the calibrated place-workspace centerline.
5. Re-detect after each successful pick-place cycle.

Control path:
    RealSense/ArUco/Open3D -> Kortex ComputeInverseKinematics -> reach_joint_angles
"""

from __future__ import annotations

import argparse
import json
import sys
import threading
import time
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from plan_regrasp_dry_run import (  # noqa: E402
    build_camera_point_cloud,
    crop_pcd_by_workspace,
    load_T_base_marker,
    load_workspaces,
    make_candidates,
    make_depth_valid_mask,
    place_centerline_slots,
    remove_near_table_residuals,
    remove_table_plane,
    sort_candidates,
    transform_pcd,
)
from perception.aruco_detector import (  # noqa: E402
    MARKER_LENGTH,
    TARGET_MARKER_ID,
    create_aruco_detector,
    detect_target_marker_pose,
)
from perception.realsense_camera import RealSenseCamera  # noqa: E402
from scripts.real_pick_place_auto_cluster import (  # noqa: E402
    BaseClient,
    BaseCyclicClient,
    Base_pb2,
    KServerException,
    assert_valid_position,
    close_gripper,
    compute_ik_official,
    execute_ik_joint_pose,
    joint_angles_to_list,
    make_z_segment_positions,
    move_to_joint_angles_official,
    normalize_joint_angles_0_360,
    open_gripper,
    shortest_angle_delta_deg,
    utilities,
)
from scripts.real_move_above_cup_test import move_to_cartesian_pose_official  # noqa: E402
from scripts.real_move_above_cup_test import check_for_end_or_abort  # noqa: E402
from utils.transform_utils import invert_T  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)

    # Kortex connection arguments expected by utilities.DeviceConnection.
    parser.add_argument("--ip", type=str, default="192.168.1.10")
    parser.add_argument("-u", "--username", type=str, default="admin")
    parser.add_argument("-p", "--password", type=str, default="admin")

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

    # Perception parameters, matching plan_regrasp_dry_run.py.
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
    parser.add_argument(
        "--pick-center-z-min",
        type=float,
        default=None,
        help="Reject pickup candidates whose center z is below this base-frame value.",
    )
    parser.add_argument(
        "--pick-center-z-max",
        type=float,
        default=None,
        help="Reject pickup candidates whose center z is above this base-frame value. Defaults to pickup table_z_mean + --pick-center-max-above-table.",
    )
    parser.add_argument(
        "--pick-center-max-above-table",
        type=float,
        default=0.14,
        help="Default max pickup candidate center height above pickup table_z_mean.",
    )
    parser.add_argument("--pick-max-extent-z", type=float, default=0.22)
    parser.add_argument("--pick-max-extent-xy", type=float, default=0.22)
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
    )

    # Placement slots.
    parser.add_argument("--place-slot-spacing", type=float, default=None)
    parser.add_argument("--place-slot-margin", type=float, default=None)
    parser.add_argument(
        "--place-z-offset",
        type=float,
        default=0.04,
        help="place base z = place table_z_mean + this offset.",
    )
    parser.add_argument(
        "--max-cups",
        type=int,
        default=1,
        help="Maximum cups to move in this run. Increase after one-cup testing.",
    )

    # Motion heights relative to pick/place base points.
    parser.add_argument("--pre-grasp-height", type=float, default=0.20)
    parser.add_argument("--grasp-height", type=float, default=0.03)
    parser.add_argument("--pre-place-height", type=float, default=0.20)
    parser.add_argument(
        "--pre-place-extra-heights",
        type=str,
        default="0,0.05,0.10",
        help="Comma-separated extra z offsets tried for pre-place IK, in meters.",
    )
    parser.add_argument(
        "--place-height",
        type=float,
        default=0.03,
        help="Release height above place base point.",
    )
    parser.add_argument("--retreat-height", type=float, default=None)
    parser.add_argument("--max-z-step", type=float, default=0.02)
    parser.add_argument(
        "--segmented-z-motion",
        action="store_true",
        help="Use old conservative segmented z descent/lift/place moves. Default is one IK action per vertical move.",
    )
    parser.add_argument(
        "--motion-mode",
        choices=["ik_joint", "cartesian", "ik_then_cartesian"],
        default="ik_joint",
        help="Motion backend: Kortex IK+reach_joint_angles, direct reach_pose, or Cartesian fallback after IK failure.",
    )
    parser.add_argument(
        "--retract-before-run",
        action="store_true",
        help="Execute the robot's built-in Retract action before starting pick-place cycles.",
    )
    parser.add_argument(
        "--home-before-run",
        action="store_true",
        help="Deprecated alias for --retract-before-run.",
    )
    parser.add_argument(
        "--retract-on-ik-failure",
        action="store_true",
        default=True,
        help="When IK fails, execute Retract/ready posture once, then retry the same IK target.",
    )
    parser.add_argument(
        "--disable-retract-on-ik-failure",
        action="store_true",
        help="Disable the Retract/ready retry fallback after IK failure.",
    )
    parser.add_argument("--recovery-action-name", type=str, default="Retract")
    parser.add_argument("--recovery-timeout", type=float, default=60.0)
    parser.add_argument(
        "--ready-joints",
        type=str,
        default=None,
        help="Optional comma-separated joint angles in degrees used if built-in Retract action is unavailable.",
    )
    parser.add_argument(
        "--max-joint-delta",
        type=float,
        default=120.0,
        help="Reject IK solutions requiring any joint to move more than this shortest-path angle in degrees. Use <=0 to disable.",
    )
    parser.add_argument(
        "--disable-shortest-ik-selection",
        action="store_true",
        help="Use the first successful IK solution instead of the shortest joint-distance solution.",
    )

    # Gripper parameters. Convention: 0=open, 1=closed.
    parser.add_argument("--open-value", type=float, default=0.0)
    parser.add_argument("--close-value", type=float, default=0.8)
    parser.add_argument("--gripper-step", type=float, default=0.05)
    parser.add_argument("--gripper-settle-time", type=float, default=0.20)
    parser.add_argument("--hold-after-lift", type=float, default=1.0)

    # Tool orientation. Kortex Cartesian pose uses theta_x/theta_y/theta_z in degrees.
    parser.add_argument(
        "--tool-orientation-preset",
        choices=["current", "vertical_down", "manual"],
        default="vertical_down",
        help=(
            "current: use measured startup orientation; "
            "vertical_down: use --vertical-down-theta-*; "
            "manual: use --tool-theta-*."
        ),
    )
    parser.add_argument("--tool-theta-x", type=float, default=None)
    parser.add_argument("--tool-theta-y", type=float, default=None)
    parser.add_argument("--tool-theta-z", type=float, default=None)
    parser.add_argument(
        "--vertical-down-theta-x",
        type=float,
        default=180.0,
        help="Default theta_x for a vertical-down gripper pose, in Kortex degrees.",
    )
    parser.add_argument(
        "--vertical-down-theta-y",
        type=float,
        default=0.0,
        help="Default theta_y for a vertical-down gripper pose, in Kortex degrees.",
    )
    parser.add_argument(
        "--vertical-down-theta-z",
        type=float,
        default=90.0,
        help="Default theta_z/yaw for a vertical-down gripper pose, in Kortex degrees.",
    )
    parser.add_argument(
        "--orientation-search",
        action="store_true",
        help="Before each cycle, try yaw variants and top-down presets, then use the first IK-reachable orientation.",
    )
    parser.add_argument(
        "--orientation-yaw-step",
        type=float,
        default=30.0,
        help="Yaw step in degrees for --orientation-search.",
    )
    parser.add_argument(
        "--orientation-yaw-range",
        type=float,
        default=180.0,
        help="Yaw search range around the initial theta_z in degrees for --orientation-search.",
    )

    parser.add_argument("--dry-run", action="store_true", help="Detect and print plan, but do not connect/move.")
    parser.add_argument("--auto", action="store_true", help="Do not pause before every motion segment.")
    parser.add_argument(
        "--yes",
        action="store_true",
        help="Skip the initial REALREGRASP confirmation prompt. Use only after dry-run verification.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "debug_pointclouds" / "real_regrasp",
    )
    return parser.parse_args()


def validate_args(args):
    if args.grasp_height >= args.pre_grasp_height:
        raise ValueError("--grasp-height must be lower than --pre-grasp-height.")
    if args.retreat_height is None:
        args.retreat_height = args.pre_place_height
    if args.place_height >= args.pre_place_height:
        raise ValueError("--place-height must be lower than --pre-place-height.")
    if args.max_cups < 1:
        raise ValueError("--max-cups must be >= 1.")
    if args.home_before_run:
        args.retract_before_run = True
    if args.disable_retract_on_ik_failure:
        args.retract_on_ik_failure = False
    if args.tool_orientation_preset == "manual" and None in (
        args.tool_theta_x,
        args.tool_theta_y,
        args.tool_theta_z,
    ):
        raise ValueError(
            "--tool-orientation-preset manual requires --tool-theta-x, --tool-theta-y, and --tool-theta-z."
        )


def resolve_tool_thetas(args, current_tool_pose):
    if args.tool_orientation_preset == "manual":
        return np.array([args.tool_theta_x, args.tool_theta_y, args.tool_theta_z], dtype=np.float64)
    if args.tool_orientation_preset == "vertical_down":
        return np.array(
            [
                args.vertical_down_theta_x,
                args.vertical_down_theta_y,
                args.vertical_down_theta_z,
            ],
            dtype=np.float64,
        )
    return np.asarray(current_tool_pose[3:], dtype=np.float64).copy()


def filter_pick_candidates(candidates, args, pickup_ws):
    z_min = args.pick_center_z_min
    if z_min is None:
        z_min = float(pickup_ws["table_z_mean"]) - 0.02
    z_max = args.pick_center_z_max
    if z_max is None:
        z_max = float(pickup_ws["table_z_mean"]) + float(args.pick_center_max_above_table)

    kept = []
    for candidate in candidates:
        center = np.asarray(candidate["center_base"], dtype=np.float64)
        ext = np.asarray(candidate["extent"], dtype=np.float64)
        max_xy = float(max(ext[0], ext[1]))
        reasons = []
        if center[2] < z_min:
            reasons.append(f"center_z<{z_min:.3f}")
        if center[2] > z_max:
            reasons.append(f"center_z>{z_max:.3f}")
        if ext[2] > args.pick_max_extent_z:
            reasons.append(f"extent_z>{args.pick_max_extent_z:.3f}")
        if max_xy > args.pick_max_extent_xy:
            reasons.append(f"extent_xy>{args.pick_max_extent_xy:.3f}")

        if reasons:
            print(
                f"Reject candidate label={candidate['label']:02d}: "
                f"center_z={center[2]:.3f}, extent={[round(float(v), 3) for v in ext]}, "
                f"reason={','.join(reasons)}"
            )
            continue
        kept.append(candidate)
    return kept


def capture_pickup_candidates(args, T_base_marker, pickup_ws):
    detector_bundle = create_aruco_detector()
    camera = RealSenseCamera(
        width=args.width,
        height=args.height,
        fps=args.fps,
        stream_order=args.camera_stream_order,
    )

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
        candidates = filter_pick_candidates(candidates, args, pickup_ws)

        summary = {
            "pickup_workspace_points": len(pcd_pickup_crop.points),
            "pickup_non_table_points": len(pcd_no_table.points),
            "near_table_removed": near_table_removed,
            "depth_edge_rejected_pixels": int(np.count_nonzero(edge_mask)),
            "dbscan": dbscan_summary,
        }
        return candidates, summary

    finally:
        camera.stop()


def candidate_to_printable(candidate):
    return {
        "label": int(candidate["label"]),
        "center_base": [float(v) for v in candidate["center_base"]],
        "raw_points": int(candidate["raw_points"]),
        "clean_points": int(candidate["clean_points"]),
        "extent": [float(v) for v in candidate["extent"]],
    }


def make_pose(position, tool_thetas):
    pose = np.zeros(6, dtype=np.float64)
    pose[:3] = np.asarray(position, dtype=np.float64)
    pose[3:] = np.asarray(tool_thetas, dtype=np.float64)
    return pose


def maybe_pause(args, prompt):
    if not args.auto:
        input(prompt)


def parse_ready_joints(text):
    if text is None:
        return None
    values = [float(part.strip()) for part in text.split(",") if part.strip()]
    if not values:
        return None
    return values


def parse_float_list(text):
    if text is None:
        return []
    return [float(part.strip()) for part in str(text).split(",") if part.strip()]


def execute_recovery_action(base, action_name="Retract", timeout_s=60.0):
    print(f"\n========== Move to {action_name} action ==========")
    action_type = Base_pb2.RequestedActionType()
    action_type.action_type = Base_pb2.REACH_JOINT_ANGLES
    action_list = base.ReadAllActions(action_type)

    action_handle = None
    available = []
    for action in action_list.action_list:
        available.append(action.name)
        if action.name == action_name:
            action_handle = action.handle
            break

    if action_handle is None:
        print(f"Could not find action named {action_name!r}. Available actions: {available}")
        return False

    done_event = threading.Event()
    result = {"event": None, "abort_details": None}
    notification_handle = base.OnNotificationActionTopic(
        check_for_end_or_abort(done_event, result),
        Base_pb2.NotificationOptions(),
    )

    print(f"Executing action from reference: {action_name}")
    base.ExecuteActionFromReference(action_handle)
    finished = done_event.wait(float(timeout_s))
    base.Unsubscribe(notification_handle)

    if not finished:
        print(f"Timeout waiting for {action_name} action notification.")
        return False
    if result["event"] == "ACTION_ABORT":
        print(f"{action_name} action aborted:", result["abort_details"])
        return False
    return result["event"] == "ACTION_END"


def execute_ready_posture(base, args):
    if execute_recovery_action(base, args.recovery_action_name, args.recovery_timeout):
        return True

    ready_values = parse_ready_joints(args.ready_joints)
    if ready_values is None:
        print("No --ready-joints provided; cannot use ready joint fallback.")
        return False

    print("\n========== Move to ready joints fallback ==========")
    print("ready_joints [deg]:", ready_values)
    ready = make_guess_from_values(ready_values)
    return move_to_joint_angles_official(base, ready, timeout_s=args.recovery_timeout)


def make_guess_from_values(values):
    guess = Base_pb2.JointAngles()
    for idx, value in enumerate(values):
        joint_angle = guess.joint_angles.add()
        joint_angle.joint_identifier = idx
        joint_angle.value = float(value)
    return guess


def compute_ik_with_explicit_guess(base, target_pose, guess_joint_angles):
    ik_data = Base_pb2.IKData()
    ik_data.cartesian_pose.x = float(target_pose[0])
    ik_data.cartesian_pose.y = float(target_pose[1])
    ik_data.cartesian_pose.z = float(target_pose[2])
    ik_data.cartesian_pose.theta_x = float(target_pose[3])
    ik_data.cartesian_pose.theta_y = float(target_pose[4])
    ik_data.cartesian_pose.theta_z = float(target_pose[5])
    ik_data.guess.CopyFrom(guess_joint_angles)
    return base.ComputeInverseKinematics(ik_data)


def ik_solution_score(base, joint_angles):
    measured = base.GetMeasuredJointAngles()
    target = normalize_joint_angles_0_360(base, joint_angles)
    current_values = joint_angles_to_list(measured)
    target_values = joint_angles_to_list(target)
    deltas = [
        abs(shortest_angle_delta_deg(target_value, current_value))
        for current_value, target_value in zip(current_values, target_values)
    ]
    max_delta = max(deltas) if deltas else 0.0
    total_delta = sum(deltas)
    return max_delta, total_delta, deltas, target


def compute_ik_multi_guess(base, target_pose, args=None):
    measured = base.GetMeasuredJointAngles()
    measured_values = joint_angles_to_list(measured)
    actuator_count = base.GetActuatorCount().count
    measured_values = measured_values[:actuator_count]

    guesses = [("official_measured_minus_1", None)]
    guesses.append(("measured_raw", make_guess_from_values(measured_values)))
    guesses.append(("measured_0_360", normalize_joint_angles_0_360(base, measured)))

    offsets = [
        [0.0, 0.0, 0.0, 10.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, -10.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 10.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, -10.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 10.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, -10.0],
    ]
    for idx, offset in enumerate(offsets, start=1):
        values = measured_values.copy()
        for joint_idx, delta in enumerate(offset[: len(values)]):
            values[joint_idx] = (values[joint_idx] + delta) % 360.0
        guesses.append((f"measured_offset_{idx}", make_guess_from_values(values)))

    errors = []
    successes = []
    for name, guess in guesses:
        try:
            if guess is None:
                result = compute_ik_official(base, target_pose)
            else:
                result = compute_ik_with_explicit_guess(base, target_pose, guess)
            max_delta, total_delta, deltas, normalized = ik_solution_score(base, result)
            print(
                f"IK candidate succeeded: {name}, "
                f"max_delta={max_delta:.2f} deg, total_delta={total_delta:.2f} deg, "
                f"deltas={[round(float(v), 2) for v in deltas]}"
            )
            successes.append((max_delta, total_delta, name, normalized, deltas))
        except KServerException as exc:
            errors.append(f"{name}: error={exc.get_error_code()} sub={exc.get_error_sub_code()} {exc}")
        except Exception as exc:
            errors.append(f"{name}: {exc}")

    if successes:
        if args is not None and args.disable_shortest_ik_selection:
            chosen = successes[0]
        else:
            chosen = min(successes, key=lambda item: (item[0], item[1]))

        max_delta, total_delta, name, normalized, deltas = chosen
        max_allowed = None if args is None else float(args.max_joint_delta)
        if max_allowed is not None and max_allowed > 0 and max_delta > max_allowed:
            print(
                f"Rejecting shortest IK solution {name}: max_delta={max_delta:.2f} deg "
                f"> --max-joint-delta={max_allowed:.2f} deg"
            )
        else:
            print(
                f"Selected IK solution: {name}, max_delta={max_delta:.2f} deg, "
                f"total_delta={total_delta:.2f} deg"
            )
            return normalized

    print("All IK guesses failed:")
    for error in errors:
        print("  " + error)
    if successes:
        print("IK successes existed, but all were rejected by joint-delta limits.")
    raise RuntimeError("All IK guesses failed.")


def ik_reachable(base, target_pose, args=None):
    try:
        compute_ik_multi_guess(base, target_pose, args=args)
        return True
    except Exception:
        return False


def unique_orientations(orientations, decimals=3):
    seen = set()
    out = []
    for orientation in orientations:
        arr = np.asarray(orientation, dtype=np.float64).reshape(3)
        key = tuple(np.round(arr, decimals=decimals).tolist())
        if key in seen:
            continue
        seen.add(key)
        out.append(arr)
    return out


def build_orientation_candidates(base_orientation, args):
    base_orientation = np.asarray(base_orientation, dtype=np.float64).reshape(3)
    candidates = [base_orientation]

    if args.orientation_search:
        step = abs(float(args.orientation_yaw_step))
        yaw_range = abs(float(args.orientation_yaw_range))
        if step <= 0:
            step = 30.0
        yaw_offsets = np.arange(-yaw_range, yaw_range + 0.5 * step, step)
        for offset in yaw_offsets:
            candidate = base_orientation.copy()
            candidate[2] = base_orientation[2] + float(offset)
            candidates.append(candidate)

        # Common Gen3 Lite top-down-like Euler presets. The controller will
        # reject invalid ones; this list simply gives IK more posture choices.
        for tx, ty in [
            (0.0, 180.0),
            (0.0, -180.0),
            (180.0, 0.0),
            (-180.0, 0.0),
            (90.0, 0.0),
            (-90.0, 0.0),
        ]:
            for yaw in [base_orientation[2], 0.0, 45.0, 90.0, 135.0, 180.0, -45.0, -90.0, -135.0]:
                candidates.append(np.array([tx, ty, yaw], dtype=np.float64))

    return unique_orientations(candidates)


def choose_reachable_orientation(base, pre_grasp_pos, base_orientation, args, cycle_idx):
    candidates = build_orientation_candidates(base_orientation, args)
    print(f"\nChecking {len(candidates)} orientation candidate(s) for cycle {cycle_idx} pre-grasp IK...")
    for idx, orientation in enumerate(candidates, start=1):
        target_pose = make_pose(pre_grasp_pos, orientation)
        print(f"  orientation {idx}/{len(candidates)}: {orientation}")
        if ik_reachable(base, target_pose, args=args):
            print(f"Selected reachable orientation for cycle {cycle_idx}: {orientation}")
            return orientation
    print("No orientation candidate produced a reachable pre-grasp IK.")
    return base_orientation


def choose_reachable_orientation_for_target(base, target_pos, base_orientation, args, label):
    candidates = build_orientation_candidates(base_orientation, args)
    print(f"\nChecking {len(candidates)} orientation candidate(s) for {label} IK...")
    for idx, orientation in enumerate(candidates, start=1):
        target_pose = make_pose(target_pos, orientation)
        print(f"  orientation {idx}/{len(candidates)}: {orientation}")
        if ik_reachable(base, target_pose, args=args):
            print(f"Selected reachable orientation for {label}: {orientation}")
            return orientation
    print(f"No orientation candidate produced a reachable IK for {label}.")
    return base_orientation


def execute_ik_joint_pose_multi_guess(base, base_cyclic, target_pose, label, args):
    print(f"\n========== {label} ==========")
    print("Target tool pose [x,y,z,theta_x,theta_y,theta_z]:")
    print(target_pose)

    try:
        target_joints = compute_ik_multi_guess(base, target_pose, args=args)
    except Exception:
        return False

    print("IK target joint angles raw [deg]:")
    print(joint_angles_to_list(target_joints))
    target_joints = normalize_joint_angles_0_360(base, target_joints)
    print("IK target joint angles normalized to [0, 360) [deg]:")
    print(joint_angles_to_list(target_joints))

    joints_before = base.GetMeasuredJointAngles()
    pose_before = np.array(
        [
            base_cyclic.RefreshFeedback().base.tool_pose_x,
            base_cyclic.RefreshFeedback().base.tool_pose_y,
            base_cyclic.RefreshFeedback().base.tool_pose_z,
        ],
        dtype=np.float64,
    )
    ok = move_to_joint_angles_official(base, target_joints)
    joints_after = base.GetMeasuredJointAngles()
    before_values = joint_angles_to_list(joints_before)
    after_values = joint_angles_to_list(joints_after)
    deltas = [shortest_angle_delta_deg(after, before) for before, after in zip(before_values, after_values)]
    print("Measured joint deltas shortest-path [deg]:")
    print(deltas)
    pose_after_feedback = base_cyclic.RefreshFeedback().base
    pose_after = np.array(
        [
            pose_after_feedback.tool_pose_x,
            pose_after_feedback.tool_pose_y,
            pose_after_feedback.tool_pose_z,
        ],
        dtype=np.float64,
    )
    print("Measured tool position delta [m]:")
    print(pose_after - pose_before)
    return ok


def execute_tool_pose(base, base_cyclic, target_pose, label, args, allow_home_retry=True):
    if args.motion_mode == "cartesian":
        print(f"\n========== {label} ==========")
        print("Target tool pose [x,y,z,theta_x,theta_y,theta_z]:")
        print(target_pose)
        return move_to_cartesian_pose_official(base, target_pose)

    if args.motion_mode == "ik_joint":
        ok = execute_ik_joint_pose_multi_guess(base, base_cyclic, target_pose, label, args)
    else:
        ok = execute_ik_joint_pose_multi_guess(base, base_cyclic, target_pose, label, args)
    if ok:
        return True

    if allow_home_retry and args.retract_on_ik_failure:
        print(f"\nIK failed for {label}. Moving to Retract/ready posture, then retrying IK once...")
        if execute_ready_posture(base, args):
            retry_ok = execute_tool_pose(
                base,
                base_cyclic,
                target_pose,
                label + " after Retract/ready",
                args,
                allow_home_retry=False,
            )
            if retry_ok:
                return True
        else:
            print("Retract/ready recovery move failed or unavailable.")

    if args.motion_mode != "ik_then_cartesian":
        return ok

    print(f"\nIK failed for {label}. Trying Cartesian reach_pose fallback...")
    return move_to_cartesian_pose_official(base, target_pose)


def execute_pick_place_cycle(base, base_cyclic, pick_base, place_base, tool_thetas, args, cycle_idx):
    pick_base = np.asarray(pick_base, dtype=np.float64)
    place_base = np.asarray(place_base, dtype=np.float64)

    pre_grasp_pos = pick_base + np.array([0.0, 0.0, args.pre_grasp_height], dtype=np.float64)
    grasp_pos = pick_base + np.array([0.0, 0.0, args.grasp_height], dtype=np.float64)
    lift_pos = pre_grasp_pos.copy()

    pre_place_pos = place_base + np.array([0.0, 0.0, args.pre_place_height], dtype=np.float64)
    place_pos = place_base + np.array([0.0, 0.0, args.place_height], dtype=np.float64)
    retreat_pos = place_base + np.array([0.0, 0.0, args.retreat_height], dtype=np.float64)

    for name, pos in [
        ("pick_base", pick_base),
        ("place_base", place_base),
        ("pre_grasp_pos", pre_grasp_pos),
        ("grasp_pos", grasp_pos),
        ("lift_pos", lift_pos),
        ("pre_place_pos", pre_place_pos),
        ("place_pos", place_pos),
        ("retreat_pos", retreat_pos),
    ]:
        assert_valid_position(name, pos)

    print(f"\n========== Cycle {cycle_idx} Motion Plan ==========")
    print("pick_base [m]:", pick_base)
    print("pre_grasp_pos [m]:", pre_grasp_pos)
    print("grasp_pos [m]:", grasp_pos)
    print("place_base [m]:", place_base)
    print("pre_place_pos [m]:", pre_place_pos)
    print("place_pos [m]:", place_pos)
    print("retreat_pos [m]:", retreat_pos)
    print("tool_thetas [deg]:", tool_thetas)

    maybe_pause(args, "\nPress Enter to open gripper...")
    open_gripper(base, args)

    maybe_pause(args, "\nPress Enter to move to pre-grasp...")
    if not execute_tool_pose(base, base_cyclic, make_pose(pre_grasp_pos, tool_thetas), f"Cycle {cycle_idx}: move to pre-grasp", args):
        return False

    descent_positions = (
        make_z_segment_positions(pre_grasp_pos, grasp_pos, args.max_z_step)
        if args.segmented_z_motion
        else [grasp_pos]
    )
    for idx, pos in enumerate(descent_positions, start=1):
        maybe_pause(args, f"Press Enter for descent segment {idx}/{len(descent_positions)}...")
        if not execute_tool_pose(base, base_cyclic, make_pose(pos, tool_thetas), f"Cycle {cycle_idx}: descent {idx}", args):
            return False

    maybe_pause(args, "\nPress Enter to close gripper...")
    close_gripper(base, args)
    time.sleep(0.5)

    lift_positions = (
        make_z_segment_positions(grasp_pos, lift_pos, args.max_z_step)
        if args.segmented_z_motion
        else [lift_pos]
    )
    for idx, pos in enumerate(lift_positions, start=1):
        maybe_pause(args, f"Press Enter for lift segment {idx}/{len(lift_positions)}...")
        if not execute_tool_pose(base, base_cyclic, make_pose(pos, tool_thetas), f"Cycle {cycle_idx}: lift {idx}", args):
            return False

    print(f"\nLift completed. Holding for {args.hold_after_lift:.1f} s...")
    time.sleep(args.hold_after_lift)

    place_tool_thetas = tool_thetas
    selected_pre_place_pos = pre_place_pos.copy()
    selected_extra_height = 0.0
    if args.orientation_search:
        extra_heights = parse_float_list(args.pre_place_extra_heights)
        if not extra_heights:
            extra_heights = [0.0]
        for extra_height in extra_heights:
            candidate_pre_place_pos = pre_place_pos + np.array([0.0, 0.0, float(extra_height)], dtype=np.float64)
            candidate_tool_thetas = choose_reachable_orientation_for_target(
                base,
                candidate_pre_place_pos,
                tool_thetas,
                args,
                f"cycle {cycle_idx} pre-place extra_z={extra_height:.3f}",
            )
            if ik_reachable(base, make_pose(candidate_pre_place_pos, candidate_tool_thetas), args=args):
                selected_pre_place_pos = candidate_pre_place_pos
                selected_extra_height = float(extra_height)
                place_tool_thetas = candidate_tool_thetas
                break
        if selected_extra_height != 0.0:
            print(f"Using raised pre-place position with extra_z={selected_extra_height:.3f} m:", selected_pre_place_pos)

    maybe_pause(args, "\nPress Enter to move to pre-place...")
    if not execute_tool_pose(base, base_cyclic, make_pose(selected_pre_place_pos, place_tool_thetas), f"Cycle {cycle_idx}: move to pre-place", args):
        print("Failed to reach pre-place. Keeping gripper closed.")
        return False

    place_positions = (
        make_z_segment_positions(selected_pre_place_pos, place_pos, args.max_z_step)
        if args.segmented_z_motion
        else [place_pos]
    )
    for idx, pos in enumerate(place_positions, start=1):
        maybe_pause(args, f"Press Enter for place descent segment {idx}/{len(place_positions)}...")
        if not execute_tool_pose(base, base_cyclic, make_pose(pos, place_tool_thetas), f"Cycle {cycle_idx}: place descent {idx}", args):
            print("Place descent failed. Keeping gripper closed.")
            return False

    maybe_pause(args, "\nPress Enter to open gripper and release...")
    open_gripper(base, args)

    maybe_pause(args, "\nPress Enter to retreat upward...")
    if not execute_tool_pose(base, base_cyclic, make_pose(retreat_pos, place_tool_thetas), f"Cycle {cycle_idx}: retreat", args):
        return False

    return True


def main():
    args = parse_args()
    validate_args(args)

    T_base_marker = load_T_base_marker(args.calibration)
    workspaces, placement_cfg = load_workspaces(args.workspaces)
    pickup_ws = workspaces["pickup"]
    place_ws = workspaces["place"]
    slots, slot_info = place_centerline_slots(place_ws, placement_cfg, args, args.max_cups)

    print("========== Real Workspace Regrasp ==========")
    print("This script WILL move the robot unless --dry-run is used.")
    print("Control path: perception -> ComputeInverseKinematics -> reach_joint_angles.")
    print(f"MARKER_LENGTH={MARKER_LENGTH:.3f} m, target marker ID={TARGET_MARKER_ID}")
    print(f"workspaces: {args.workspaces}")
    print(f"max_cups={args.max_cups}, generated place slots={len(slots)}")
    print(f"slot_info={slot_info}")
    print(f"pre_grasp_height={args.pre_grasp_height:.3f}, grasp_height={args.grasp_height:.3f}")
    print(f"pre_place_height={args.pre_place_height:.3f}, place_height={args.place_height:.3f}, retreat_height={args.retreat_height:.3f}")
    print(f"open_value={args.open_value:.3f}, close_value={args.close_value:.3f}")
    print(f"motion_mode={args.motion_mode}")
    print(f"retract_before_run={args.retract_before_run}, retract_on_ik_failure={args.retract_on_ik_failure}")
    print(f"max_joint_delta={args.max_joint_delta:.1f} deg")

    first_candidates, first_summary = capture_pickup_candidates(args, T_base_marker, pickup_ws)
    print("\nInitial pickup candidates:")
    if not first_candidates:
        print("  none")
    for c in first_candidates:
        center = c["center_base"]
        print(
            f"  label={c['label']:02d}, center=[{center[0]:.4f}, {center[1]:.4f}, {center[2]:.4f}], "
            f"raw={c['raw_points']}, clean={c['clean_points']}, extent={[round(float(v), 4) for v in c['extent']]}"
        )

    initial_plan = []
    for idx, c in enumerate(first_candidates[: len(slots)]):
        initial_plan.append(
            {
                "step": idx,
                "label": int(c["label"]),
                "pick_point_base": [float(v) for v in c["center_base"]],
                "place_point_base": slots[idx]["place_point_base"],
                "raw_points": int(c["raw_points"]),
                "clean_points": int(c["clean_points"]),
            }
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    plan_path = args.output_dir / f"{ts}_initial_real_regrasp_plan.json"
    with open(plan_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "mode": "real_motion" if not args.dry_run else "dry_run_no_robot_motion",
                "initial_summary": first_summary,
                "slot_info": slot_info,
                "initial_plan": initial_plan,
            },
            f,
            indent=2,
        )
    print(f"\nSaved initial plan: {plan_path}")

    if args.dry_run:
        print("Dry run enabled. Exiting without robot motion.")
        return 0

    if not first_candidates:
        print("No pickup candidates. Exiting.")
        return 1

    print("\nSafety checklist:")
    print(" - Keep E-stop ready.")
    print(" - First test with --max-cups 1 and without --auto.")
    print(" - Make sure pickup/place vertical paths are clear.")
    print(" - Tool orientation is controlled by --tool-orientation-preset and theta parameters.")
    if not args.yes:
        answer = input("\nType REALREGRASP to connect and run real workspace regrasp: ").strip()
        if answer != "REALREGRASP":
            print("Confirmation not received. Exiting.")
            return 1
    else:
        print("\n--yes enabled: skipping initial REALREGRASP confirmation.")

    completed = []
    with utilities.DeviceConnection.createTcpConnection(args) as router:
        base = BaseClient(router)
        base_cyclic = BaseCyclicClient(router)

        current_pose = base_cyclic.RefreshFeedback().base
        current_tool_pose = np.array(
            [
                current_pose.tool_pose_x,
                current_pose.tool_pose_y,
                current_pose.tool_pose_z,
                current_pose.tool_pose_theta_x,
                current_pose.tool_pose_theta_y,
                current_pose.tool_pose_theta_z,
            ],
            dtype=np.float64,
        )
        print("\nCurrent tool pose [x,y,z,theta_x,theta_y,theta_z]:")
        print(current_tool_pose)

        if args.retract_before_run:
            if not execute_ready_posture(base, args):
                print("Requested --retract-before-run but Retract/ready move failed. Exiting before grasping.")
                return 2
            current_pose = base_cyclic.RefreshFeedback().base
            current_tool_pose = np.array(
                [
                    current_pose.tool_pose_x,
                    current_pose.tool_pose_y,
                    current_pose.tool_pose_z,
                    current_pose.tool_pose_theta_x,
                    current_pose.tool_pose_theta_y,
                    current_pose.tool_pose_theta_z,
                ],
                dtype=np.float64,
            )
            print("Tool pose after Retract/ready [x,y,z,theta_x,theta_y,theta_z]:")
            print(current_tool_pose)

        tool_thetas = resolve_tool_thetas(args, current_tool_pose)
        print(f"Tool orientation preset: {args.tool_orientation_preset}")
        print("Using tool orientation [theta_x, theta_y, theta_z] deg:", tool_thetas)

        for cycle_idx in range(1, min(args.max_cups, len(slots)) + 1):
            print(f"\n========== Detection for cycle {cycle_idx} ==========")
            if cycle_idx == 1:
                candidates = first_candidates
                summary = first_summary
            else:
                candidates, summary = capture_pickup_candidates(args, T_base_marker, pickup_ws)

            if not candidates:
                print("No more pickup candidates. Stopping.")
                break

            candidate = candidates[0]
            slot = slots[cycle_idx - 1]
            pick_base = np.asarray(candidate["center_base"], dtype=np.float64)
            place_base = np.asarray(slot["place_point_base"], dtype=np.float64)

            print(
                f"Selected label={candidate['label']:02d}, "
                f"pick=[{pick_base[0]:.4f}, {pick_base[1]:.4f}, {pick_base[2]:.4f}], "
                f"place=[{place_base[0]:.4f}, {place_base[1]:.4f}, {place_base[2]:.4f}]"
            )

            cycle_tool_thetas = tool_thetas
            if args.orientation_search:
                pre_grasp_pos = pick_base + np.array([0.0, 0.0, args.pre_grasp_height], dtype=np.float64)
                cycle_tool_thetas = choose_reachable_orientation(
                    base,
                    pre_grasp_pos,
                    tool_thetas,
                    args,
                    cycle_idx,
                )

            ok = execute_pick_place_cycle(base, base_cyclic, pick_base, place_base, cycle_tool_thetas, args, cycle_idx)
            completed.append(
                {
                    "cycle": cycle_idx,
                    "success": bool(ok),
                    "label": int(candidate["label"]),
                    "pick_point_base": [float(v) for v in pick_base],
                    "place_point_base": [float(v) for v in place_base],
                    "summary": summary,
                }
            )
            if not ok:
                print("Cycle failed. Stopping real regrasp run.")
                break

    result_path = args.output_dir / f"{ts}_real_regrasp_result.json"
    with open(result_path, "w", encoding="utf-8") as f:
        json.dump({"completed": completed}, f, indent=2)
    print(f"\nSaved result: {result_path}")
    print(f"Completed successful cycles: {sum(1 for item in completed if item['success'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Move Kinova Gen3 lite above a clicked cup/object.

Safety policy for this step:
- Perception first, robot motion second.
- Move only to pre-grasp above the object.
- No descent.
- No gripper command.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from control.kortex_controller import CartesianPose, KortexController  # noqa: E402
from perception.aruco_detector import (  # noqa: E402
    MARKER_LENGTH,
    TARGET_MARKER_ID,
    create_aruco_detector,
    detect_target_marker_pose,
)
from perception.object_locator import locate_clicked_object  # noqa: E402
from perception.realsense_camera import RealSenseCamera  # noqa: E402
from utils.transform_utils import transform_camera_object_to_base  # noqa: E402


WINDOW_NAME = "Move Above Cup Test - click cup center"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Estimate object pose and move Kinova above it. No descent/grasp."
    )
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=15)
    parser.add_argument(
        "--calibration",
        type=Path,
        default=PROJECT_ROOT / "configs" / "real_calibration.yaml",
        help="YAML file containing T_base_marker.",
    )
    parser.add_argument(
        "--robot-config",
        type=Path,
        default=PROJECT_ROOT / "configs" / "robot_config.yaml",
        help="YAML file containing Kortex connection and safety settings.",
    )
    parser.add_argument("--base-marker-npy", type=Path, default=None)
    parser.add_argument("--depth-window", type=int, default=9)
    parser.add_argument(
        "--move-above-height",
        type=float,
        default=None,
        help="Override height above object center in meters. Default from config or 0.20.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run perception and target generation, but do not connect or move the robot.",
    )
    return parser.parse_args()


def validate_T(T, source: str) -> np.ndarray:
    T = np.asarray(T, dtype=np.float64)
    if T.shape != (4, 4):
        raise ValueError(f"T_base_marker from {source} must be a 4x4 matrix.")
    if not np.allclose(T[3], np.array([0.0, 0.0, 0.0, 1.0]), atol=1e-9):
        raise ValueError(f"T_base_marker from {source} has an invalid last row.")
    return T


def load_T_base_marker(args) -> np.ndarray:
    if args.base_marker_npy is not None:
        return validate_T(np.load(args.base_marker_npy), str(args.base_marker_npy))

    if not args.calibration.exists():
        raise FileNotFoundError(
            f"Missing calibration file: {args.calibration}\n"
            "Create configs/real_calibration.yaml first."
        )

    with open(args.calibration, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if "T_base_marker" not in data:
        raise KeyError(f"{args.calibration} must contain T_base_marker.")
    return validate_T(np.asarray(data["T_base_marker"], dtype=np.float64), str(args.calibration))


def load_robot_config(path: Path) -> dict:
    default = {
        "kortex": {
            "ip": "192.168.1.10",
            "username": "admin",
            "password": "admin",
            "port": 10000,
        },
        "safety": {
            "move_above_height_m": 0.20,
            "via_z_m": 0.25,
            "max_xy_jump_m": 0.40,
            "min_target_z_m": 0.05,
            "action_timeout_s": 60.0,
            "translation_speed_m_s": 0.03,
            "orientation_speed_deg_s": 10.0,
            "motion_method": "trajectory",
            "max_cartesian_step_m": 0.05,
        },
    }
    if not path.exists():
        return default

    with open(path, "r", encoding="utf-8") as f:
        loaded = yaml.safe_load(f) or {}

    for section, values in default.items():
        loaded.setdefault(section, {})
        for key, value in values.items():
            loaded[section].setdefault(key, value)
    return loaded


def draw_text(vis, lines, origin=(20, 28), line_height=25):
    x, y = origin
    for idx, line in enumerate(lines):
        cv2.putText(
            vis,
            line,
            (x, y + idx * line_height),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.62,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )


def estimate_object_pose_interactive(args, T_base_marker):
    clicked_uv = {"value": None}
    latest_result = {"value": None}

    def on_mouse(event, x, y, flags, userdata):
        if event == cv2.EVENT_LBUTTONDOWN:
            clicked_uv["value"] = (int(x), int(y))
            print(f"\nClicked object pixel: u={x}, v={y}")

    detector_bundle = create_aruco_detector()
    camera = RealSenseCamera(width=args.width, height=args.height, fps=args.fps)

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback(WINDOW_NAME, on_mouse)

    print("\nStarting perception. This stage does not move the robot.")
    print("Controls: left click object center, press m to accept target, q/Esc to quit.")

    try:
        camera.start()
        while True:
            frames = camera.get_aligned_frames()
            if frames is None:
                continue

            marker_detection, corners, ids, rejected = detect_target_marker_pose(
                frames.color_bgr,
                frames.camera_matrix,
                frames.dist_coeffs,
                detector_bundle=detector_bundle,
            )

            vis = frames.color_bgr.copy()
            if ids is not None:
                cv2.aruco.drawDetectedMarkers(vis, corners, ids)

            status = []
            if marker_detection is None:
                status.append(f"marker ID {TARGET_MARKER_ID}: not detected")
            else:
                cv2.drawFrameAxes(
                    vis,
                    frames.camera_matrix,
                    frames.dist_coeffs,
                    marker_detection.rvec,
                    marker_detection.tvec,
                    MARKER_LENGTH * 0.5,
                )
                tx, ty, tz = marker_detection.tvec.reshape(3)
                status.append(f"marker: [{tx:.3f},{ty:.3f},{tz:.3f}] m")

            if clicked_uv["value"] is None:
                status.append("click cup/object center")
            else:
                u, v = clicked_uv["value"]
                cv2.circle(vis, (u, v), 5, (0, 0, 255), -1)
                object_detection = locate_clicked_object(
                    frames.camera_matrix,
                    frames.depth_z16,
                    frames.depth_scale_m,
                    clicked_uv["value"],
                    window=args.depth_window,
                )
                if object_detection is None:
                    status.append(f"object click ({u},{v}): invalid depth")
                elif marker_detection is None:
                    status.append("object depth ok; waiting for marker")
                else:
                    T_base_object = transform_camera_object_to_base(
                        T_base_marker,
                        marker_detection.T_camera_marker,
                        object_detection.T_camera_object,
                    )
                    cup_pos_base = T_base_object[:3, 3].copy()
                    latest_result["value"] = (T_base_object, cup_pos_base)
                    status.append(
                        f"object base: [{cup_pos_base[0]:.3f},{cup_pos_base[1]:.3f},{cup_pos_base[2]:.3f}] m"
                    )
                    status.append("press m to accept and close camera")

            draw_text(vis, status)
            depth_colormap = cv2.applyColorMap(
                cv2.convertScaleAbs(frames.depth_z16, alpha=0.03),
                cv2.COLORMAP_JET,
            )
            cv2.imshow(WINDOW_NAME, np.hstack((vis, depth_colormap)))

            key = cv2.waitKey(1) & 0xFF
            if key in [ord("q"), 27]:
                return None, None
            if key == ord("m"):
                if latest_result["value"] is None:
                    print("\nNo valid object pose yet. Need marker + clicked depth.")
                else:
                    return latest_result["value"]
    finally:
        camera.stop()
        cv2.destroyAllWindows()


def safety_check_target(pre_grasp_pos, current_pose, safety):
    min_target_z_m = float(safety["min_target_z_m"])
    max_xy_jump_m = float(safety["max_xy_jump_m"])

    if pre_grasp_pos[2] < min_target_z_m:
        raise ValueError(
            f"Target z={pre_grasp_pos[2]:.3f} m is below min_target_z_m={min_target_z_m:.3f} m."
        )

    current_xy = np.array([current_pose.x, current_pose.y], dtype=np.float64)
    target_xy = pre_grasp_pos[:2]
    xy_jump = float(np.linalg.norm(target_xy - current_xy))
    if xy_jump > max_xy_jump_m:
        raise ValueError(
            f"XY move distance {xy_jump:.3f} m exceeds max_xy_jump_m={max_xy_jump_m:.3f} m."
        )


def build_segmented_move_targets(current_pose, pre_grasp_pos, safety):
    max_step = float(safety["max_cartesian_step_m"])
    via_z = max(
        float(safety["via_z_m"]),
        float(pre_grasp_pos[2]),
        float(current_pose.z),
    )

    orientation = {
        "theta_x": current_pose.theta_x,
        "theta_y": current_pose.theta_y,
        "theta_z": current_pose.theta_z,
    }

    waypoints = [
        np.array([current_pose.x, current_pose.y, current_pose.z], dtype=np.float64),
        np.array([current_pose.x, current_pose.y, via_z], dtype=np.float64),
        np.array([pre_grasp_pos[0], pre_grasp_pos[1], via_z], dtype=np.float64),
        np.array([pre_grasp_pos[0], pre_grasp_pos[1], pre_grasp_pos[2]], dtype=np.float64),
    ]

    targets = []
    for start, end in zip(waypoints[:-1], waypoints[1:]):
        delta = end - start
        distance = float(np.linalg.norm(delta))
        if distance < 0.005:
            continue
        steps = max(1, int(np.ceil(distance / max_step)))
        for idx in range(1, steps + 1):
            pos = start + delta * (idx / steps)
            targets.append(
                CartesianPose(
                    x=float(pos[0]),
                    y=float(pos[1]),
                    z=float(pos[2]),
                    **orientation,
                )
            )
    return targets


def main():
    args = parse_args()
    T_base_marker = load_T_base_marker(args)
    robot_config = load_robot_config(args.robot_config)
    safety = robot_config["safety"]
    kortex = robot_config["kortex"]

    move_above_height = (
        float(args.move_above_height)
        if args.move_above_height is not None
        else float(safety["move_above_height_m"])
    )

    print("========== Real Move Above Cup Test ==========")
    print("This script moves only to a safe pre-grasp position above the object.")
    print("It will not descend and will not close the gripper.")
    print("\nT_base_marker:")
    print(T_base_marker)

    T_base_object, cup_pos_base = estimate_object_pose_interactive(args, T_base_marker)
    if T_base_object is None:
        print("No target accepted. Exiting without robot motion.")
        return

    pre_grasp_pos = cup_pos_base + np.array([0.0, 0.0, move_above_height], dtype=np.float64)

    print("\n========== Accepted Target ==========")
    print("T_base_object:")
    print(T_base_object)
    print("cup_pos_base [m]:", cup_pos_base)
    print(f"move_above_height [m]: {move_above_height:.3f}")
    print("pre_grasp_pos [m]:", pre_grasp_pos)

    if args.dry_run:
        print("\nDry run enabled. Exiting without robot motion.")
        return

    answer = input(
        "\nAbout to connect to Kinova and move above the object. "
        "Type MOVE to continue: "
    ).strip()
    if answer != "MOVE":
        print("Confirmation not received. Exiting without robot motion.")
        return

    with KortexController(
        ip=str(kortex["ip"]),
        username=str(kortex["username"]),
        password=str(kortex["password"]),
        port=int(kortex["port"]),
    ) as robot:
        robot.clear_faults()
        robot.set_single_level_servoing()
        current_pose = robot.get_measured_cartesian_pose()
        print("\nCurrent tool pose:")
        print(current_pose)
        print("Arm state:", robot.get_arm_state())
        print("Servoing mode:", robot.get_servoing_mode())

        safety_check_target(pre_grasp_pos, current_pose, safety)

        targets = build_segmented_move_targets(current_pose, pre_grasp_pos, safety)

        print("\nSegmented target tool poses:")
        for idx, target_pose in enumerate(targets, start=1):
            print(f"[{idx}] {target_pose}")

        ok = True
        for idx, target_pose in enumerate(targets, start=1):
            input(f"\nPress Enter to execute segment {idx}/{len(targets)}...")
            ok = robot.move_to_cartesian_pose(
                target_pose,
                name=f"move_above_{idx}",
                timeout_s=float(safety["action_timeout_s"]),
                translation_speed_m_s=float(safety["translation_speed_m_s"]),
                orientation_speed_deg_s=float(safety["orientation_speed_deg_s"]),
                method=str(safety["motion_method"]),
            )
            if not ok:
                print(f"Segment {idx} aborted. Stopping remaining motion.")
                break

        print("\nKortex action result:", "success" if ok else "aborted")


if __name__ == "__main__":
    main()

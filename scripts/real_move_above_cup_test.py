#!/usr/bin/env python3
"""Real move-above-cup test using Kinova's official Kortex example style.

Pipeline:
1. RealSense + ArUco perception estimates T_base_object.
2. Generate pre_grasp_pos = cup_pos + [0, 0, move_above_height].
3. Execute one official-style Kortex reach_pose action.

No descent. No gripper command.
"""

from __future__ import annotations

import argparse
import os
import sys
import threading
from pathlib import Path

import cv2
import numpy as np
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

OFFICIAL_EXAMPLES_DIR = (
    PROJECT_ROOT / "third_party" / "Kinova-kortex2_Gen3_G3L" / "api_python" / "examples"
)
if not OFFICIAL_EXAMPLES_DIR.exists():
    OFFICIAL_EXAMPLES_DIR = PROJECT_ROOT / "official_kortex_examples"
if str(OFFICIAL_EXAMPLES_DIR) not in sys.path:
    sys.path.insert(0, str(OFFICIAL_EXAMPLES_DIR))

import utilities  # noqa: E402
from kortex_api.autogen.client_stubs.BaseClientRpc import BaseClient  # noqa: E402
from kortex_api.autogen.client_stubs.BaseCyclicClientRpc import BaseCyclicClient  # noqa: E402
from kortex_api.autogen.messages import Base_pb2  # noqa: E402
from kortex_api.Exceptions.KServerException import KServerException  # noqa: E402

from perception.aruco_detector import (  # noqa: E402
    MARKER_LENGTH,
    TARGET_MARKER_ID,
    create_aruco_detector,
    detect_target_marker_pose,
)
from perception.object_locator import locate_clicked_object  # noqa: E402
from perception.realsense_camera import RealSenseCamera  # noqa: E402
from utils.transform_utils import transform_camera_object_to_base  # noqa: E402


WINDOW_NAME = "Move Above Cup - official Kortex style"
TIMEOUT_DURATION = 30


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ip", type=str, default="192.168.1.10")
    parser.add_argument("-u", "--username", type=str, default="admin")
    parser.add_argument("-p", "--password", type=str, default="admin")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=15)
    parser.add_argument("--depth-window", type=int, default=9)
    parser.add_argument("--move-above-height", type=float, default=0.20)
    parser.add_argument(
        "--motion-mode",
        choices=["ik_joint", "cartesian"],
        default="ik_joint",
        help="Use Kortex IK + reach_joint_angles, or direct reach_pose.",
    )
    parser.add_argument(
        "--calibration",
        type=Path,
        default=PROJECT_ROOT / "configs" / "real_calibration.yaml",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def load_T_base_marker(path: Path) -> np.ndarray:
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    T = np.asarray(data["T_base_marker"], dtype=np.float64)
    if T.shape != (4, 4):
        raise ValueError("T_base_marker must be 4x4.")
    return T


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

    print("\nPerception stage: click object center, press m to accept, q/Esc to quit.")

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
                    status.append("press m to accept")

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
                    print("\nNo valid object pose yet.")
                else:
                    return latest_result["value"]
    finally:
        camera.stop()
        cv2.destroyAllWindows()


def check_for_end_or_abort(done_event, result):
    def check(notification, done_event=done_event, result=result):
        event_name = Base_pb2.ActionEvent.Name(notification.action_event)
        print("EVENT :", event_name)
        if notification.action_event == Base_pb2.ACTION_END:
            result["event"] = "ACTION_END"
            done_event.set()
        elif notification.action_event == Base_pb2.ACTION_ABORT:
            result["event"] = "ACTION_ABORT"
            result["abort_details"] = Base_pb2.SubErrorCodes.Name(notification.abort_details)
            done_event.set()

    return check


def move_to_cartesian_pose_official(base, target_pose, timeout_s=TIMEOUT_DURATION):
    base_servo_mode = Base_pb2.ServoingModeInformation()
    base_servo_mode.servoing_mode = Base_pb2.SINGLE_LEVEL_SERVOING
    base.SetServoingMode(base_servo_mode)

    action = Base_pb2.Action()
    action.name = "move above cup"
    action.application_data = ""

    cartesian_pose = action.reach_pose.target_pose
    cartesian_pose.x = float(target_pose[0])
    cartesian_pose.y = float(target_pose[1])
    cartesian_pose.z = float(target_pose[2])
    cartesian_pose.theta_x = float(target_pose[3])
    cartesian_pose.theta_y = float(target_pose[4])
    cartesian_pose.theta_z = float(target_pose[5])

    done_event = threading.Event()
    result = {"event": None, "abort_details": None}
    notification_handle = base.OnNotificationActionTopic(
        check_for_end_or_abort(done_event, result),
        Base_pb2.NotificationOptions(),
    )

    print("Executing official reach_pose action...")
    base.ExecuteAction(action)
    finished = done_event.wait(timeout_s)
    base.Unsubscribe(notification_handle)

    if not finished:
        print("Timeout waiting for action notification.")
        return False
    if result["event"] == "ACTION_ABORT":
        print("Cartesian action aborted:", result["abort_details"])
        return False
    return result["event"] == "ACTION_END"


def compute_ik_official(base, target_pose):
    """Follow Kinova's 111-kinematics example style."""
    input_joint_angles = base.GetMeasuredJointAngles()

    ik_data = Base_pb2.IKData()
    ik_data.cartesian_pose.x = float(target_pose[0])
    ik_data.cartesian_pose.y = float(target_pose[1])
    ik_data.cartesian_pose.z = float(target_pose[2])
    ik_data.cartesian_pose.theta_x = float(target_pose[3])
    ik_data.cartesian_pose.theta_y = float(target_pose[4])
    ik_data.cartesian_pose.theta_z = float(target_pose[5])

    # Official example only fills values and nudges them by -1 degree.
    # Keep that style because it is known to work with this API family.
    for joint_angle in input_joint_angles.joint_angles:
        guess = ik_data.guess.joint_angles.add()
        guess.value = joint_angle.value - 1.0

    return base.ComputeInverseKinematics(ik_data)


def move_to_joint_angles_official(base, joint_angles, timeout_s=TIMEOUT_DURATION):
    base_servo_mode = Base_pb2.ServoingModeInformation()
    base_servo_mode.servoing_mode = Base_pb2.SINGLE_LEVEL_SERVOING
    base.SetServoingMode(base_servo_mode)

    action = Base_pb2.Action()
    action.name = "move above joints"
    action.application_data = ""

    target_joint_angles = action.reach_joint_angles.joint_angles.joint_angles
    for idx, joint_angle_in in enumerate(joint_angles.joint_angles):
        joint_angle = target_joint_angles.add()
        joint_angle.joint_identifier = idx
        joint_angle.value = joint_angle_in.value

    done_event = threading.Event()
    result = {"event": None, "abort_details": None}
    notification_handle = base.OnNotificationActionTopic(
        check_for_end_or_abort(done_event, result),
        Base_pb2.NotificationOptions(),
    )

    print("Executing official reach_joint_angles action...")
    base.ExecuteAction(action)
    finished = done_event.wait(timeout_s)
    base.Unsubscribe(notification_handle)

    if not finished:
        print("Timeout waiting for action notification.")
        return False
    if result["event"] == "ACTION_ABORT":
        print("Angular action aborted:", result["abort_details"])
        return False
    return result["event"] == "ACTION_END"


def wrap_angle_0_360(angle_deg):
    return float(angle_deg) % 360.0


def joint_angles_to_list(joint_angles):
    return [float(j.value) for j in joint_angles.joint_angles]


def normalize_joint_angles_0_360(base, joint_angles):
    normalized = Base_pb2.JointAngles()
    actuator_count = base.GetActuatorCount().count
    values = joint_angles_to_list(joint_angles)
    for idx in range(min(actuator_count, len(values))):
        joint_angle = normalized.joint_angles.add()
        joint_angle.joint_identifier = idx
        joint_angle.value = wrap_angle_0_360(values[idx])
    return normalized


def get_tool_pose_from_feedback(base_cyclic):
    feedback = base_cyclic.RefreshFeedback()
    return np.array(
        [
            feedback.base.tool_pose_x,
            feedback.base.tool_pose_y,
            feedback.base.tool_pose_z,
            feedback.base.tool_pose_theta_x,
            feedback.base.tool_pose_theta_y,
            feedback.base.tool_pose_theta_z,
        ],
        dtype=np.float64,
    )


def shortest_angle_delta_deg(after, before):
    return ((float(after) - float(before) + 180.0) % 360.0) - 180.0


def main():
    args = parse_args()
    T_base_marker = load_T_base_marker(args.calibration)

    print("========== Real Move Above Cup Test ==========")
    print(f"Kortex control path: official examples style, motion_mode={args.motion_mode}.")
    print("No descent. No gripper command.")
    print("\nT_base_marker:")
    print(T_base_marker)

    T_base_object, cup_pos_base = estimate_object_pose_interactive(args, T_base_marker)
    if T_base_object is None:
        print("No target accepted. Exiting.")
        return 1

    pre_grasp_pos = cup_pos_base + np.array([0.0, 0.0, args.move_above_height], dtype=np.float64)

    print("\n========== Accepted Target ==========")
    print("T_base_object:")
    print(T_base_object)
    print("cup_pos_base [m]:", cup_pos_base)
    print("pre_grasp_pos [m]:", pre_grasp_pos)

    if args.dry_run:
        print("Dry run enabled. Exiting without robot motion.")
        return 0

    answer = input("\nType MOVE to connect and move above the object: ").strip()
    if answer != "MOVE":
        print("Confirmation not received. Exiting.")
        return 1

    with utilities.DeviceConnection.createTcpConnection(args) as router:
        base = BaseClient(router)
        base_cyclic = BaseCyclicClient(router)
        feedback = base_cyclic.RefreshFeedback()

        current_pose = np.array(
            [
                feedback.base.tool_pose_x,
                feedback.base.tool_pose_y,
                feedback.base.tool_pose_z,
                feedback.base.tool_pose_theta_x,
                feedback.base.tool_pose_theta_y,
                feedback.base.tool_pose_theta_z,
            ],
            dtype=np.float64,
        )

        target_pose = current_pose.copy()
        target_pose[:3] = pre_grasp_pos

        print("\nCurrent tool pose [x,y,z,theta_x,theta_y,theta_z]:")
        print(current_pose)
        print("Target tool pose [x,y,z,theta_x,theta_y,theta_z]:")
        print(target_pose)

        if args.motion_mode == "cartesian":
            ok = move_to_cartesian_pose_official(base, target_pose)
        else:
            try:
                print("\nComputing IK using official Kortex style...")
                target_joints = compute_ik_official(base, target_pose)
            except KServerException as ex:
                print("Unable to compute inverse kinematics.")
                print("Error_code:{} , Sub_error_code:{} ".format(ex.get_error_code(), ex.get_error_sub_code()))
                print("KServerException:", ex)
                return 3

            print("IK target joint angles raw [deg]:")
            print(joint_angles_to_list(target_joints))

            target_joints = normalize_joint_angles_0_360(base, target_joints)
            print("IK target joint angles normalized to [0, 360) [deg]:")
            print(joint_angles_to_list(target_joints))
            confirm = input("Type JOINTMOVE to execute IK joint target: ").strip()
            if confirm != "JOINTMOVE":
                print("Confirmation not received. Exiting.")
                return 1

            joints_before = base.GetMeasuredJointAngles()
            pose_before = get_tool_pose_from_feedback(base_cyclic)
            print("Measured joint angles before [deg]:")
            print(joint_angles_to_list(joints_before))

            ok = move_to_joint_angles_official(base, target_joints)

            joints_after = base.GetMeasuredJointAngles()
            pose_after = get_tool_pose_from_feedback(base_cyclic)
            before_values = joint_angles_to_list(joints_before)
            after_values = joint_angles_to_list(joints_after)
            deltas = [
                shortest_angle_delta_deg(after, before)
                for before, after in zip(before_values, after_values)
            ]
            print("Measured joint angles after [deg]:")
            print(after_values)
            print("Measured joint deltas shortest-path [deg]:")
            print(deltas)
            print("Tool pose before [x,y,z,theta_x,theta_y,theta_z]:")
            print(pose_before)
            print("Tool pose after [x,y,z,theta_x,theta_y,theta_z]:")
            print(pose_after)
            print("Measured tool position delta [m]:")
            print(pose_after[:3] - pose_before[:3])

        print("Move-above result:", "success" if ok else "failed")
        return 0 if ok else 2


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Real approach test.

中文说明：真实机械臂接近动作测试脚本。它会点击/估计目标点，移动到目标上方，
再下降到 approach 高度，不闭合夹爪，然后抬回 pre-grasp。该脚本会移动真机。

This script reuses the proven perception + official Kortex IK/joint-action path:
1. Click object center and estimate T_base_object.
2. Move to pre-grasp above the object.
3. Move down in small z steps to an approach height.
4. Do not close the gripper.
5. Move back up to pre-grasp.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.real_move_above_cup_test import (  # noqa: E402
    BaseClient,
    BaseCyclicClient,
    KServerException,
    compute_ik_official,
    estimate_object_pose_interactive,
    get_tool_pose_from_feedback,
    joint_angles_to_list,
    load_T_base_marker,
    move_to_joint_angles_official,
    normalize_joint_angles_0_360,
    shortest_angle_delta_deg,
    utilities,
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ip", type=str, default="192.168.1.10")
    parser.add_argument("-u", "--username", type=str, default="admin")
    parser.add_argument("-p", "--password", type=str, default="admin")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=15)
    parser.add_argument("--depth-window", type=int, default=9)
    parser.add_argument("--pre-grasp-height", type=float, default=0.20)
    parser.add_argument("--approach-height", type=float, default=0.10)
    parser.add_argument("--max-z-step", type=float, default=0.03)
    parser.add_argument(
        "--calibration",
        type=Path,
        default=PROJECT_ROOT / "configs" / "real_calibration.yaml",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--auto",
        action="store_true",
        help="Do not ask for Enter before each descent/lift segment.",
    )
    return parser.parse_args()


def make_z_segment_positions(start_pos, end_pos, max_z_step):
    start_pos = np.asarray(start_pos, dtype=np.float64)
    end_pos = np.asarray(end_pos, dtype=np.float64)
    dz = float(end_pos[2] - start_pos[2])
    steps = max(1, int(np.ceil(abs(dz) / float(max_z_step))))
    return [start_pos + (end_pos - start_pos) * (idx / steps) for idx in range(1, steps + 1)]


def execute_ik_joint_pose(base, base_cyclic, target_pose, label):
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
    deltas = [
        shortest_angle_delta_deg(after, before)
        for before, after in zip(before_values, after_values)
    ]

    print("Measured joint deltas shortest-path [deg]:")
    print(deltas)
    print("Tool pose after [x,y,z,theta_x,theta_y,theta_z]:")
    print(pose_after)
    print("Measured tool position delta [m]:")
    print(pose_after[:3] - pose_before[:3])
    return ok


def main():
    args = parse_args()
    if args.approach_height >= args.pre_grasp_height:
        raise ValueError("--approach-height must be lower than --pre-grasp-height.")

    T_base_marker = load_T_base_marker(args.calibration)

    print("========== Real Approach Test ==========")
    print("No gripper command. Descend only, then lift back.")
    print(f"pre_grasp_height={args.pre_grasp_height:.3f} m")
    print(f"approach_height={args.approach_height:.3f} m")
    print(f"max_z_step={args.max_z_step:.3f} m")
    print("\nT_base_marker:")
    print(T_base_marker)

    T_base_object, cup_pos_base = estimate_object_pose_interactive(args, T_base_marker)
    if T_base_object is None:
        print("No target accepted. Exiting.")
        return 1

    pre_grasp_pos = cup_pos_base + np.array([0.0, 0.0, args.pre_grasp_height], dtype=np.float64)
    approach_pos = cup_pos_base + np.array([0.0, 0.0, args.approach_height], dtype=np.float64)

    print("\n========== Accepted Target ==========")
    print("cup_pos_base [m]:", cup_pos_base)
    print("pre_grasp_pos [m]:", pre_grasp_pos)
    print("approach_pos [m]:", approach_pos)

    if args.dry_run:
        print("Dry run enabled. Exiting without robot motion.")
        return 0

    answer = input("\nType APPROACH to connect and run approach test: ").strip()
    if answer != "APPROACH":
        print("Confirmation not received. Exiting.")
        return 1

    with utilities.DeviceConnection.createTcpConnection(args) as router:
        base = BaseClient(router)
        base_cyclic = BaseCyclicClient(router)

        current_pose = get_tool_pose_from_feedback(base_cyclic)
        pre_grasp_pose = current_pose.copy()
        pre_grasp_pose[:3] = pre_grasp_pos
        approach_pose = current_pose.copy()
        approach_pose[:3] = approach_pos

        print("\nCurrent tool pose [x,y,z,theta_x,theta_y,theta_z]:")
        print(current_pose)

        if not execute_ik_joint_pose(base, base_cyclic, pre_grasp_pose, "Move to pre-grasp"):
            print("Failed to reach pre-grasp. Stopping.")
            return 2

        descent_positions = make_z_segment_positions(pre_grasp_pos, approach_pos, args.max_z_step)
        for idx, pos in enumerate(descent_positions, start=1):
            target_pose = pre_grasp_pose.copy()
            target_pose[:3] = pos
            print(f"\nNext descent segment {idx}/{len(descent_positions)} target z={pos[2]:.3f}")
            if not args.auto:
                input("Press Enter to execute this descent segment...")
            if not execute_ik_joint_pose(base, base_cyclic, target_pose, f"Descend segment {idx}"):
                print("Descent segment failed. Stopping before lower motion.")
                return 3

        print("\nReached approach height. No gripper command was sent.")
        if not args.auto:
            input("Press Enter to lift back to pre-grasp...")

        lift_positions = make_z_segment_positions(approach_pos, pre_grasp_pos, args.max_z_step)
        for idx, pos in enumerate(lift_positions, start=1):
            target_pose = pre_grasp_pose.copy()
            target_pose[:3] = pos
            print(f"\nNext lift segment {idx}/{len(lift_positions)} target z={pos[2]:.3f}")
            if not args.auto:
                input("Press Enter to execute this lift segment...")
            if not execute_ik_joint_pose(base, base_cyclic, target_pose, f"Lift segment {idx}"):
                print("Lift segment failed.")
                return 4

    print("\nApproach test completed successfully.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

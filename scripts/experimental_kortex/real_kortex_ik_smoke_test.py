#!/usr/bin/env python3
"""Kortex IK + joint trajectory smoke test."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from control.kortex_controller import CartesianPose, KortexController  # noqa: E402
from scripts.real_move_above_cup_test import load_robot_config  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(description="Move current tool pose by dz using Kortex IK + joint trajectory.")
    parser.add_argument(
        "--robot-config",
        type=Path,
        default=PROJECT_ROOT / "configs" / "robot_config.yaml",
    )
    parser.add_argument("--dz", type=float, default=0.02)
    parser.add_argument("--duration", type=float, default=8.0)
    parser.add_argument("--timeout", type=float, default=30.0)
    return parser.parse_args()


def joint_values_deg(joint_angles):
    return [float(j.value) for j in joint_angles.joint_angles]


def build_ik_guesses(robot, current_joints):
    current = joint_values_deg(current_joints)
    guesses = [
        ("current", current_joints),
        ("current_normalized", robot.normalized_joint_guess(current_joints)),
    ]

    # A few broad, gentle seeds for Gen3 lite branch selection.
    seed_values = [
        [0.0, 20.0, 160.0, 270.0, 330.0, 270.0],
        [0.0, 20.0, 160.0, -90.0, -30.0, -90.0],
        [0.0, 45.0, 120.0, 270.0, 330.0, 270.0],
        [0.0, 0.0, 180.0, 270.0, 330.0, 270.0],
        current[:],
    ]
    for idx, values in enumerate(seed_values):
        guesses.append((f"seed_{idx}", robot.make_joint_angles(values)))
    return guesses


def main():
    args = parse_args()
    cfg = load_robot_config(args.robot_config)
    kortex = cfg["kortex"]

    print("========== Kortex IK Smoke Test ==========")
    print(f"This will compute IK for current pose + dz={args.dz:.3f} m, then play a joint trajectory.")

    answer = input("Type IK to execute this small IK joint move: ").strip()
    if answer != "IK":
        print("Confirmation not received. Exiting.")
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
        current_joints = robot.get_measured_joint_angles()

        print("\nCurrent pose:")
        print(current_pose)
        print("Current joint angles [deg]:")
        print(joint_values_deg(current_joints))
        print("Arm state:", robot.get_arm_state())
        print("Servoing mode:", robot.get_servoing_mode())

        target_pose = CartesianPose(
            x=current_pose.x,
            y=current_pose.y,
            z=current_pose.z + args.dz,
            theta_x=current_pose.theta_x,
            theta_y=current_pose.theta_y,
            theta_z=current_pose.theta_z,
        )
        print("\nTarget pose:")
        print(target_pose)

        guess_name, target_joints = robot.compute_inverse_kinematics_with_guesses(
            target_pose,
            build_ik_guesses(robot, current_joints),
        )
        print(f"IK succeeded with guess: {guess_name}")
        print("IK joint target [deg]:")
        print(joint_values_deg(target_joints))

        ok = robot.move_to_joint_angles(
            target_joints,
            name="ik_smoke",
            timeout_s=args.timeout,
            duration_s=args.duration,
        )

        after_pose = robot.get_measured_cartesian_pose()
        after_joints = robot.get_measured_joint_angles()
        print("\nAfter pose:")
        print(after_pose)
        print("After joint angles [deg]:")
        print(joint_values_deg(after_joints))
        print(f"Measured dz: {after_pose.z - current_pose.z:.4f} m")
        print("IK smoke result:", "success" if ok else "aborted")


if __name__ == "__main__":
    main()

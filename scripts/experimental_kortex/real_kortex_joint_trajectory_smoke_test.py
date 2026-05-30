#!/usr/bin/env python3
"""Tiny Kortex joint trajectory smoke test."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from control.kortex_controller import KortexController  # noqa: E402
from scripts.real_move_above_cup_test import load_robot_config  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(description="Move one joint by a tiny delta using PlayJointTrajectory.")
    parser.add_argument(
        "--robot-config",
        type=Path,
        default=PROJECT_ROOT / "configs" / "robot_config.yaml",
    )
    parser.add_argument("--joint", type=int, default=0, help="Zero-based joint identifier.")
    parser.add_argument("--delta", type=float, default=1.0, help="Joint angle delta in deg.")
    parser.add_argument("--duration", type=float, default=8.0)
    parser.add_argument("--timeout", type=float, default=30.0)
    return parser.parse_args()


def joint_values_deg(joint_angles):
    return [float(j.value) for j in joint_angles.joint_angles]


def main():
    args = parse_args()
    cfg = load_robot_config(args.robot_config)
    kortex = cfg["kortex"]

    print("========== Kortex Joint Trajectory Smoke Test ==========")
    print(f"Joint: {args.joint}, delta: {args.delta:.3f} deg, duration: {args.duration:.3f} s")

    answer = input("Type JTRAJ to execute this small joint trajectory: ").strip()
    if answer != "JTRAJ":
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

        before = robot.get_measured_joint_angles()
        target_values = joint_values_deg(before)
        print("\nBefore joint angles [deg]:")
        print(target_values)
        print("Arm state:", robot.get_arm_state())
        print("Servoing mode:", robot.get_servoing_mode())

        if not 0 <= args.joint < len(target_values):
            raise ValueError(f"Joint index {args.joint} is out of range for {len(target_values)} joints.")

        target_values[args.joint] += args.delta
        target_joints = robot.make_joint_angles(target_values)

        print("\nTarget joint angles [deg]:")
        print(target_values)

        ok = robot.move_to_joint_angles(
            target_joints,
            name="joint_traj_smoke",
            timeout_s=args.timeout,
            duration_s=args.duration,
        )

        after = robot.get_measured_joint_angles()
        after_values = joint_values_deg(after)
        print("\nAfter joint angles [deg]:")
        print(after_values)
        print(f"Measured joint {args.joint} delta: {after_values[args.joint] - joint_values_deg(before)[args.joint]:.4f} deg")
        print("Joint trajectory smoke result:", "success" if ok else "aborted")


if __name__ == "__main__":
    main()

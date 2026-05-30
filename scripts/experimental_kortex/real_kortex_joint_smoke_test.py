#!/usr/bin/env python3
"""Tiny Kortex joint-speed smoke test."""

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
    parser = argparse.ArgumentParser(description="Move one joint slowly for a short time.")
    parser.add_argument(
        "--robot-config",
        type=Path,
        default=PROJECT_ROOT / "configs" / "robot_config.yaml",
    )
    parser.add_argument("--joint", type=int, default=0, help="Zero-based joint identifier.")
    parser.add_argument("--speed", type=float, default=2.0, help="Joint speed in deg/s.")
    parser.add_argument("--duration", type=float, default=1.0, help="Command duration in seconds.")
    parser.add_argument("--joystick", action="store_true", help="Use joystick joint-speed RPC.")
    return parser.parse_args()


def joint_values_deg(joint_angles):
    return [float(j.value) for j in joint_angles.joint_angles]


def main():
    args = parse_args()
    cfg = load_robot_config(args.robot_config)
    kortex = cfg["kortex"]

    print("========== Kortex Joint Smoke Test ==========")
    print(f"Joint: {args.joint}, speed: {args.speed:.3f} deg/s, duration: {args.duration:.3f} s")
    print("This test does not use perception.")

    answer = input("Type JOINT to execute this small joint-speed command: ").strip()
    if answer != "JOINT":
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
        print("\nBefore joint angles [deg]:")
        print(joint_values_deg(before))
        print("Arm state:", robot.get_arm_state())
        print("Servoing mode:", robot.get_servoing_mode())

        robot.move_joint_with_speed(
            joint_identifier=args.joint,
            speed_deg_s=args.speed,
            duration_s=args.duration,
            use_joystick=args.joystick,
        )

        after = robot.get_measured_joint_angles()
        before_values = joint_values_deg(before)
        after_values = joint_values_deg(after)
        print("\nAfter joint angles [deg]:")
        print(after_values)
        if 0 <= args.joint < len(after_values):
            print(f"Measured joint {args.joint} delta: {after_values[args.joint] - before_values[args.joint]:.4f} deg")


if __name__ == "__main__":
    main()

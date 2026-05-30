#!/usr/bin/env python3
"""Tiny Kortex Cartesian motion smoke test.

This does not use perception. It only asks the arm to raise the current tool
pose by a small z offset, keeping the current orientation.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from control.kortex_controller import CartesianPose, KortexController  # noqa: E402
from scripts.real_move_above_cup_test import load_robot_config  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(description="Move the Kinova tool up by a tiny z offset.")
    parser.add_argument(
        "--robot-config",
        type=Path,
        default=PROJECT_ROOT / "configs" / "robot_config.yaml",
    )
    parser.add_argument("--dz", type=float, default=0.02, help="Relative z lift in meters.")
    parser.add_argument("--speed", type=float, default=0.02, help="Cartesian speed in m/s.")
    parser.add_argument("--timeout", type=float, default=30.0)
    parser.add_argument(
        "--method",
        choices=["twist", "twist_joystick", "trajectory", "action"],
        default="twist",
        help="Motion method to test.",
    )
    parser.add_argument(
        "--frame",
        choices=["base", "tool", "mixed"],
        default="base",
        help="Reference frame for twist methods.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = load_robot_config(args.robot_config)
    kortex = cfg["kortex"]

    print("========== Kortex Smoke Test ==========")
    print("This will move only current tool z by a small positive offset.")

    answer = input(f"Type LIFT to move +{args.dz:.3f} m in z: ").strip()
    if answer != "LIFT":
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
        current = robot.get_measured_cartesian_pose()
        print("\nCurrent pose:")
        print(current)
        print("Arm state:", robot.get_arm_state())
        print("Servoing mode:", robot.get_servoing_mode())

        target = CartesianPose(
            x=current.x,
            y=current.y,
            z=current.z + args.dz,
            theta_x=current.theta_x,
            theta_y=current.theta_y,
            theta_z=current.theta_z,
        )
        print("\nTarget pose:")
        print(target)

        if args.method in ["twist", "twist_joystick"]:
            robot.move_relative_z_with_twist(
                dz_m=args.dz,
                speed_m_s=args.speed,
                use_joystick=args.method == "twist_joystick",
                reference_frame=args.frame,
            )
            after = robot.get_measured_cartesian_pose()
            print("\nPose after twist:")
            print(after)
            print(f"Measured dz: {after.z - current.z:.4f} m")
            ok = True
        else:
            ok = robot.move_to_cartesian_pose(
                target,
                name="smoke_lift",
                timeout_s=args.timeout,
                translation_speed_m_s=args.speed,
                orientation_speed_deg_s=10.0,
                method=args.method,
            )
        print("\nSmoke test result:", "success" if ok else "aborted")


if __name__ == "__main__":
    main()

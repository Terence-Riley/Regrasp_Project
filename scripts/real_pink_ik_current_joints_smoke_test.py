#!/usr/bin/env python3
"""Read real Kinova joints, then run Pink IK offline from that posture.

中文说明：该脚本只连接 Kortex 读取当前真实关节角，然后把这些角度作为
Pink/Pinocchio 的初始姿态做离线 IK 验证。它不会发送运动命令，不会移动真机。

This script connects to Kortex only to read the current measured joint angles.
It does not send any motion command to the robot.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from control.kortex_controller import KortexController  # noqa: E402
from scripts.test_pink_ik_gen3_lite import solve_pink_ik  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--robot-config",
        type=Path,
        default=PROJECT_ROOT / "configs" / "robot_config.yaml",
        help="Robot config containing Kortex ip/username/password/port.",
    )

    parser.add_argument(
        "--urdf",
        type=Path,
        default=PROJECT_ROOT / "kortex_description" / "robots" / "gen3_lite.urdf",
        help="Gen3 Lite URDF path.",
    )
    parser.add_argument(
        "--frame",
        default="tool_frame",
        help="Robot frame to control. Common options: tool_frame, end_effector_link, gripper_base_link.",
    )
    parser.add_argument(
        "--target-offset",
        type=float,
        nargs=3,
        default=[0.02, 0.0, 0.0],
        metavar=("DX", "DY", "DZ"),
        help="Target translation offset from the current URDF frame pose, in meters.",
    )
    parser.add_argument(
        "--target-position",
        type=float,
        nargs=3,
        default=None,
        metavar=("X", "Y", "Z"),
        help="Absolute target frame position in URDF/world coordinates. If omitted, current position + --target-offset is used.",
    )
    parser.add_argument(
        "--target-rpy-deg",
        type=float,
        nargs=3,
        default=None,
        metavar=("ROLL", "PITCH", "YAW"),
        help="Absolute target frame orientation as RPY degrees. If omitted, current orientation is kept.",
    )
    parser.add_argument("--dt", type=float, default=0.05)
    parser.add_argument("--iterations", type=int, default=200)
    parser.add_argument("--position-cost", type=float, default=60.0)
    parser.add_argument("--orientation-cost", type=float, default=1.0)
    parser.add_argument("--posture-cost", type=float, default=0.05)
    parser.add_argument("--position-tolerance", type=float, default=0.002)
    parser.add_argument("--orientation-tolerance", type=float, default=0.03)
    parser.add_argument("--solver", default=None)
    parser.add_argument("--list-frames", action="store_true")
    parser.add_argument(
        "--no-wrap-current-joints",
        action="store_true",
        help="Do not wrap Kortex 0..360 style joint angles to [-180, 180].",
    )
    parser.set_defaults(current_joints_deg=None)
    return parser.parse_args()


def joint_values_deg(joint_angles):
    return [float(j.value) for j in joint_angles.joint_angles]


def load_robot_config(path: Path):
    if not path.exists():
        raise FileNotFoundError(
            f"Missing robot config: {path}\n"
            "Copy configs/robot_config.example.yaml to configs/robot_config.yaml and edit it."
        )
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if "kortex" not in data:
        raise KeyError(f"{path} must contain a 'kortex' section.")
    return data


def main():
    args = parse_args()

    if args.list_frames:
        solve_pink_ik(args)
        return

    cfg = load_robot_config(args.robot_config)
    kortex = cfg["kortex"]

    print("\n========== Real Pink IK Current-Joints Smoke Test ==========")
    print("This script only reads Kortex measured joints. It does not move the robot.")
    print(f"Kortex IP: {kortex['ip']}")

    with KortexController(
        ip=str(kortex["ip"]),
        username=str(kortex["username"]),
        password=str(kortex["password"]),
        port=int(kortex["port"]),
    ) as robot:
        current_joints = robot.get_measured_joint_angles()
        values = joint_values_deg(current_joints)

    print("\nMeasured Kortex joint angles [deg], raw:")
    print("  " + ", ".join(f"{v:.3f}" for v in values))

    result = solve_pink_ik(args, current_joints_deg=values)
    if result is None:
        return

    print("\nNo motion command was sent. Use this result only as an IK feasibility check.")


if __name__ == "__main__":
    main()

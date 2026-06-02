#!/usr/bin/env python3
"""Real-joint Pink IK absolute-position smoke test.

中文说明：读取真实 Kinova 当前关节角，以当前姿态为初值，用 Pink/Pinocchio
求解指定的绝对 tool_frame 目标位置。默认不移动真机，只验证绝对目标点 IK。
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
    parser.add_argument("--robot-config", type=Path, default=PROJECT_ROOT / "configs" / "robot_config.yaml")
    parser.add_argument("--urdf", type=Path, default=PROJECT_ROOT / "kortex_description" / "robots" / "gen3_lite.urdf")
    parser.add_argument("--frame", default="tool_frame")
    parser.add_argument("--target-position", type=float, nargs=3, required=True, metavar=("X", "Y", "Z"))
    parser.add_argument(
        "--target-rpy-deg",
        type=float,
        nargs=3,
        default=None,
        metavar=("ROLL", "PITCH", "YAW"),
        help="Absolute target orientation. Omit to keep current orientation.",
    )
    parser.add_argument("--target-offset", type=float, nargs=3, default=[0.0, 0.0, 0.0])
    parser.add_argument("--dt", type=float, default=0.05)
    parser.add_argument("--iterations", type=int, default=200)
    parser.add_argument("--position-cost", type=float, default=60.0)
    parser.add_argument("--orientation-cost", type=float, default=1.0)
    parser.add_argument("--posture-cost", type=float, default=0.05)
    parser.add_argument("--position-tolerance", type=float, default=0.002)
    parser.add_argument("--orientation-tolerance", type=float, default=0.03)
    parser.add_argument("--solver", default=None)
    parser.add_argument("--list-frames", action="store_true")
    parser.add_argument("--no-wrap-current-joints", action="store_true")
    parser.add_argument("--clamp-current-to-limits", action="store_true")
    parser.add_argument("--limit-margin-rad", type=float, default=1e-4)
    parser.set_defaults(current_joints_deg=None)
    parser.add_argument("--max-joint-delta", type=float, default=15.0)
    parser.add_argument("--max-position-error", type=float, default=0.004)
    parser.add_argument("--max-orientation-error", type=float, default=0.08)
    parser.add_argument("--duration", type=float, default=8.0)
    parser.add_argument("--timeout", type=float, default=30.0)
    parser.add_argument("--execute", action="store_true", help="Actually move with Kortex PlayJointTrajectory.")
    parser.add_argument("--yes", action="store_true", help="Skip MOVEABS confirmation when --execute is set.")
    return parser.parse_args()


def load_robot_config(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if "kortex" not in data:
        raise KeyError(f"{path} must contain a 'kortex' section.")
    return data


def joint_values_deg(joint_angles):
    return [float(j.value) for j in joint_angles.joint_angles]


def shortest_delta_deg(target, current):
    return [((float(t) - float(c) + 180.0) % 360.0) - 180.0 for t, c in zip(target, current)]


def safety_check(result, current_raw_deg, args):
    if result is None or not result["converged"]:
        print("Safety stop: Pink did not converge.")
        return False
    if result["position_error_m"] > args.max_position_error:
        print(f"Safety stop: position error {result['position_error_m']:.6f} m is too high.")
        return False
    if result["orientation_error_rad"] > args.max_orientation_error:
        print(f"Safety stop: orientation error {result['orientation_error_rad']:.6f} rad is too high.")
        return False

    delta = shortest_delta_deg(result["solved_joints_deg"], current_raw_deg)
    max_delta = max(abs(v) for v in delta)
    print("\nShortest deltas from measured raw Kortex joints [deg]:")
    print("  " + ", ".join(f"{v:.3f}" for v in delta))
    print(f"Max shortest joint delta: {max_delta:.3f} deg")
    if max_delta > args.max_joint_delta:
        print(f"Safety stop: max joint delta exceeds --max-joint-delta={args.max_joint_delta:.3f} deg.")
        return False
    return True


def main():
    args = parse_args()
    if args.list_frames:
        solve_pink_ik(args)
        return

    cfg = load_robot_config(args.robot_config)
    kortex = cfg["kortex"]

    print("\n========== Real Pink Absolute Position Smoke Test ==========")
    print("Default mode solves IK only. Add --execute to move with PlayJointTrajectory.")
    print(f"Target position: {args.target_position}")

    with KortexController(
        ip=str(kortex["ip"]),
        username=str(kortex["username"]),
        password=str(kortex["password"]),
        port=int(kortex["port"]),
    ) as robot:
        current_joints = robot.get_measured_joint_angles()
        values = joint_values_deg(current_joints)

        print("Measured Kortex joint angles [deg], raw:")
        print("  " + ", ".join(f"{v:.3f}" for v in values))
        result = solve_pink_ik(args, current_joints_deg=values)
        if not safety_check(result, values, args):
            print("No motion command sent.")
            return

        if not args.execute:
            print("IK is feasible. No motion command sent because --execute was not set.")
            return

        print("\nThis will execute PlayJointTrajectory to the absolute-position IK target.")
        if not args.yes:
            answer = input("Type MOVEABS to execute this absolute target move: ").strip()
            if answer != "MOVEABS":
                print("Confirmation not received. No motion command sent.")
                return

        robot.clear_faults()
        robot.set_single_level_servoing()
        target = robot.make_joint_angles(result["solved_joints_deg"])
        ok = robot.move_to_joint_angles(
            target,
            name="pink_absolute_target",
            timeout_s=float(args.timeout),
            duration_s=float(args.duration),
        )
        after = joint_values_deg(robot.get_measured_joint_angles())
        print("\nAfter Kortex joint angles [deg], raw:")
        print("  " + ", ".join(f"{v:.3f}" for v in after))
        print(f"Move result: {'success' if ok else 'failed'}")


if __name__ == "__main__":
    main()

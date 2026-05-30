#!/usr/bin/env python3
"""First real pick / lift test for Kinova Gen3 lite.

This is the first conservative real-robot pick-place script.
Default behavior is intentionally only PICK + LIFT:

1. RealSense + ArUco estimates T_base_object by clicking the cup center.
2. Open gripper.
3. Move to pre-grasp above the cup.
4. Descend in small z steps to grasp height.
5. Close gripper.
6. Lift back to pre-grasp.
7. Keep gripper closed by default.

Optional place behavior is available if --place-x --place-y --place-z are all provided.
The control path matches the working approach script:
    Kortex ComputeInverseKinematics -> reach_joint_angles

Do NOT use this as a high-speed autonomous script. Keep E-stop ready.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Reuse the proven perception + Kortex IK/joint-action utilities.
from scripts.real_move_above_cup_test import (  # noqa: E402
    BaseClient,
    BaseCyclicClient,
    Base_pb2,
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


TIMEOUT_DURATION = 30


# ============================================================
# Argument parsing
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)

    # Keep these argument names compatible with Kinova official utilities.DeviceConnection.
    parser.add_argument("--ip", type=str, default="192.168.1.10")
    parser.add_argument("-u", "--username", type=str, default="admin")
    parser.add_argument("-p", "--password", type=str, default="admin")

    # RealSense / perception parameters
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=15)
    parser.add_argument("--depth-window", type=int, default=9)
    parser.add_argument(
        "--calibration",
        type=Path,
        default=PROJECT_ROOT / "configs" / "real_calibration.yaml",
        help="YAML file containing T_base_marker.",
    )

    # Motion heights relative to clicked/estimated object position.
    parser.add_argument(
        "--pre-grasp-height",
        type=float,
        default=0.20,
        help="Height above estimated object point for pre-grasp/lift, in meters.",
    )
    parser.add_argument(
        "--grasp-height",
        type=float,
        default=0.00,
        help="Height above estimated object point for final grasp approach, in meters. Start conservative.",
    )
    parser.add_argument(
        "--max-z-step",
        type=float,
        default=0.02,
        help="Maximum vertical step between IK waypoints, in meters.",
    )

    # Gripper parameters. Convention: 0=open, 1=closed.
    parser.add_argument("--open-value", type=float, default=0.0)
    parser.add_argument(
        "--close-value",
        type=float,
        default=0.80,
        help="Normalized close value. Start with 0.35~0.60; 1.0 is fully closed.",
    )
    parser.add_argument(
        "--gripper-step",
        type=float,
        default=0.05,
        help="Small gripper command increment for ramping.",
    )
    parser.add_argument(
        "--gripper-settle-time",
        type=float,
        default=0.20,
        help="Seconds to wait after each gripper ramp command.",
    )
    parser.add_argument(
        "--hold-after-lift",
        type=float,
        default=2.0,
        help="Seconds to hold after lifting the cup.",
    )

    # Optional place target. If all three are provided, script performs place after lift.
    parser.add_argument("--place-x", type=float, default=None)
    parser.add_argument("--place-y", type=float, default=None)
    parser.add_argument("--place-z", type=float, default=None)
    parser.add_argument(
        "--pre-place-height",
        type=float,
        default=0.20,
        help="Height above place position for pre-place, in meters.",
    )
    parser.add_argument(
        "--place-height",
        type=float,
        default=0.08,
        help="Height above place position for release, in meters.",
    )

    parser.add_argument("--dry-run", action="store_true", help="Perception only; do not move robot.")
    parser.add_argument("--auto", action="store_true", help="Do not pause before every motion segment.")
    parser.add_argument(
        "--release-after-lift",
        action="store_true",
        help="After lift, ask for confirmation and open gripper. Without place target this may drop the cup.",
    )
    return parser.parse_args()


# ============================================================
# Motion utilities
# ============================================================

def make_z_segment_positions(start_pos, end_pos, max_z_step):
    start_pos = np.asarray(start_pos, dtype=np.float64)
    end_pos = np.asarray(end_pos, dtype=np.float64)
    dz = float(end_pos[2] - start_pos[2])
    steps = max(1, int(np.ceil(abs(dz) / float(max_z_step))))
    return [start_pos + (end_pos - start_pos) * (idx / steps) for idx in range(1, steps + 1)]


def execute_ik_joint_pose(base, base_cyclic, target_pose, label):
    """Compute IK for target tool pose and execute with official reach_joint_angles."""
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


# ============================================================
# Gripper utilities
# ============================================================

def clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def get_measured_gripper_position(base):
    """Return measured normalized gripper position, or None if unavailable."""
    request = Base_pb2.GripperRequest()
    request.mode = Base_pb2.GRIPPER_POSITION
    measure = base.GetMeasuredGripperMovement(request)
    if len(measure.finger) == 0:
        return None
    return float(measure.finger[0].value)


def send_gripper_position(base, position: float, finger_id: int = 1):
    """Send one normalized gripper position command. Convention: 0=open, 1=closed."""
    command = Base_pb2.GripperCommand()
    command.mode = Base_pb2.GRIPPER_POSITION

    finger = command.gripper.finger.add()
    finger.finger_identifier = finger_id
    finger.value = clamp01(position)

    base.SendGripperCommand(command)


def ramp_gripper_to(base, target: float, step: float, settle_time: float):
    """Ramp from measured/current gripper position to target in small steps."""
    target = clamp01(target)
    step = abs(float(step))
    if step <= 0:
        raise ValueError("--gripper-step must be > 0")

    current = get_measured_gripper_position(base)
    if current is None:
        print("Measured gripper position unavailable; sending target directly.")
        send_gripper_position(base, target)
        time.sleep(settle_time)
        return

    print(f"Measured current gripper position: {current:.3f}")
    print(f"Target gripper position: {target:.3f}")

    if abs(target - current) < 1e-3:
        print("Gripper already near target.")
        return

    direction = 1.0 if target > current else -1.0
    value = current
    while True:
        if direction > 0:
            value = min(target, value + step)
        else:
            value = max(target, value - step)

        print(f"Sending gripper position: {value:.3f}")
        send_gripper_position(base, value)
        time.sleep(settle_time)

        if abs(value - target) < 1e-6:
            break


def open_gripper(base, args):
    print("\n========== Open gripper ==========")
    ramp_gripper_to(base, args.open_value, args.gripper_step, args.gripper_settle_time)


def close_gripper(base, args):
    print("\n========== Close gripper ==========")
    ramp_gripper_to(base, args.close_value, args.gripper_step, args.gripper_settle_time)


# ============================================================
# Main
# ============================================================

def main():
    args = parse_args()

    if args.grasp_height >= args.pre_grasp_height:
        raise ValueError("--grasp-height must be lower than --pre-grasp-height.")
    if args.place_height >= args.pre_place_height:
        raise ValueError("--place-height must be lower than --pre-place-height.")

    do_place = all(v is not None for v in [args.place_x, args.place_y, args.place_z])
    partially_specified_place = any(v is not None for v in [args.place_x, args.place_y, args.place_z]) and not do_place
    if partially_specified_place:
        raise ValueError("To enable place, provide all three: --place-x --place-y --place-z")

    T_base_marker = load_T_base_marker(args.calibration)

    print("========== Real Pick / Lift Test ==========")
    print("Default behavior: pick and lift only; gripper stays closed after lift.")
    print("Control path: RealSense/ArUco -> T_base_object -> Kortex IK -> reach_joint_angles.")
    print(f"pre_grasp_height={args.pre_grasp_height:.3f} m")
    print(f"grasp_height={args.grasp_height:.3f} m")
    print(f"max_z_step={args.max_z_step:.3f} m")
    print(f"open_value={args.open_value:.3f}, close_value={args.close_value:.3f}")
    print("\nT_base_marker:")
    print(T_base_marker)

    T_base_object, cup_pos_base = estimate_object_pose_interactive(args, T_base_marker)
    if T_base_object is None:
        print("No target accepted. Exiting.")
        return 1

    pre_grasp_pos = cup_pos_base + np.array([0.0, 0.0, args.pre_grasp_height], dtype=np.float64)
    grasp_pos = cup_pos_base + np.array([0.0, 0.0, args.grasp_height], dtype=np.float64)
    lift_pos = pre_grasp_pos.copy()

    print("\n========== Accepted Target ==========")
    print("T_base_object:")
    print(T_base_object)
    print("cup_pos_base [m]:", cup_pos_base)
    print("pre_grasp_pos [m]:", pre_grasp_pos)
    print("grasp_pos [m]:", grasp_pos)
    print("lift_pos [m]:", lift_pos)

    if do_place:
        place_base = np.array([args.place_x, args.place_y, args.place_z], dtype=np.float64)
        pre_place_pos = place_base + np.array([0.0, 0.0, args.pre_place_height], dtype=np.float64)
        place_pos = place_base + np.array([0.0, 0.0, args.place_height], dtype=np.float64)
        retreat_pos = pre_place_pos.copy()
        print("\nPlace mode enabled.")
        print("place_base [m]:", place_base)
        print("pre_place_pos [m]:", pre_place_pos)
        print("place_pos [m]:", place_pos)
        print("retreat_pos [m]:", retreat_pos)
    else:
        place_base = pre_place_pos = place_pos = retreat_pos = None
        print("\nPlace mode disabled. Script will stop after lift with gripper closed.")

    if args.dry_run:
        print("Dry run enabled. Exiting without robot motion.")
        return 0

    print("\nSafety checklist:")
    print(" - Keep E-stop ready.")
    print(" - Use a light, non-fragile cup for the first test.")
    print(" - Make sure the path above the cup is clear.")
    print(" - Start with conservative --grasp-height and --close-value.")
    answer = input("\nType PICK to connect and run the pick/lift test: ").strip()
    if answer != "PICK":
        print("Confirmation not received. Exiting.")
        return 1

    with utilities.DeviceConnection.createTcpConnection(args) as router:
        base = BaseClient(router)
        base_cyclic = BaseCyclicClient(router)

        current_pose = get_tool_pose_from_feedback(base_cyclic)
        print("\nCurrent tool pose [x,y,z,theta_x,theta_y,theta_z]:")
        print(current_pose)

        pre_grasp_pose = current_pose.copy()
        pre_grasp_pose[:3] = pre_grasp_pos
        grasp_pose = current_pose.copy()
        grasp_pose[:3] = grasp_pos
        lift_pose = current_pose.copy()
        lift_pose[:3] = lift_pos

        # 1. Open gripper.
        if not args.auto:
            input("\nPress Enter to open gripper...")
        open_gripper(base, args)

        # 2. Move to pre-grasp.
        if not args.auto:
            input("\nPress Enter to move to pre-grasp...")
        if not execute_ik_joint_pose(base, base_cyclic, pre_grasp_pose, "Move to pre-grasp"):
            print("Failed to reach pre-grasp. Stopping.")
            return 2

        # 3. Descend in z segments.
        descent_positions = make_z_segment_positions(pre_grasp_pos, grasp_pos, args.max_z_step)
        for idx, pos in enumerate(descent_positions, start=1):
            target_pose = current_pose.copy()
            target_pose[:3] = pos
            print(f"\nNext descent segment {idx}/{len(descent_positions)} target z={pos[2]:.3f}")
            if not args.auto:
                input("Press Enter to execute this descent segment...")
            if not execute_ik_joint_pose(base, base_cyclic, target_pose, f"Descend segment {idx}"):
                print("Descent segment failed. Stopping before gripper close.")
                return 3

        # 4. Close gripper.
        if not args.auto:
            input("\nPress Enter to close gripper around the cup...")
        close_gripper(base, args)
        time.sleep(0.5)

        # 5. Lift in z segments.
        lift_positions = make_z_segment_positions(grasp_pos, lift_pos, args.max_z_step)
        for idx, pos in enumerate(lift_positions, start=1):
            target_pose = current_pose.copy()
            target_pose[:3] = pos
            print(f"\nNext lift segment {idx}/{len(lift_positions)} target z={pos[2]:.3f}")
            if not args.auto:
                input("Press Enter to execute this lift segment...")
            if not execute_ik_joint_pose(base, base_cyclic, target_pose, f"Lift segment {idx}"):
                print("Lift segment failed.")
                return 4

        print(f"\nLift completed. Holding for {args.hold_after_lift:.1f} s...")
        time.sleep(args.hold_after_lift)

        # Optional place sequence.
        if do_place:
            pre_place_pose = current_pose.copy()
            pre_place_pose[:3] = pre_place_pos
            place_pose = current_pose.copy()
            place_pose[:3] = place_pos
            retreat_pose = current_pose.copy()
            retreat_pose[:3] = retreat_pos

            if not args.auto:
                input("\nPress Enter to move to pre-place...")
            if not execute_ik_joint_pose(base, base_cyclic, pre_place_pose, "Move to pre-place"):
                print("Failed to reach pre-place. Keeping gripper closed.")
                return 5

            place_positions = make_z_segment_positions(pre_place_pos, place_pos, args.max_z_step)
            for idx, pos in enumerate(place_positions, start=1):
                target_pose = current_pose.copy()
                target_pose[:3] = pos
                print(f"\nNext place descent segment {idx}/{len(place_positions)} target z={pos[2]:.3f}")
                if not args.auto:
                    input("Press Enter to execute this place descent segment...")
                if not execute_ik_joint_pose(base, base_cyclic, target_pose, f"Place descent segment {idx}"):
                    print("Place descent failed. Keeping gripper closed.")
                    return 6

            if not args.auto:
                input("\nPress Enter to open gripper and release cup...")
            open_gripper(base, args)

            if not args.auto:
                input("\nPress Enter to retreat upward...")
            if not execute_ik_joint_pose(base, base_cyclic, retreat_pose, "Retreat after place"):
                print("Retreat failed.")
                return 7

            print("\nPick-place sequence completed.")
            return 0

        # No place mode: do not drop the cup automatically.
        if args.release_after_lift:
            answer = input(
                "\nCup should now be lifted. Type RELEASE to open gripper now "
                "(only if safe / over table): "
            ).strip()
            if answer == "RELEASE":
                open_gripper(base, args)
            else:
                print("Release not confirmed. Gripper remains closed.")
        else:
            print("\nPick/lift sequence completed. Gripper remains closed by default.")
            print("Move the robot/place the cup safely, or run real_gripper_test.py --mode open when ready.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

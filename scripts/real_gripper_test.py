#!/usr/bin/env python3
"""Real gripper test for Kinova Gen3 lite.

This script follows the official Kortex API style used by the current project:
- add third_party/.../api_python/examples to sys.path
- use utilities.DeviceConnection.createTcpConnection(args)
- use BaseClient + Base_pb2.GripperCommand

Default safe sequence:
    open -> partial close -> open

The gripper position command is normalized:
    0.0 = open
    1.0 = fully closed

For the first test, keep --close-value conservative, e.g. 0.35~0.60.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path


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
from kortex_api.autogen.messages import Base_pb2  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)

    # Keep names compatible with Kinova official utilities.DeviceConnection.
    parser.add_argument("--ip", type=str, default="192.168.1.10")
    parser.add_argument("-u", "--username", type=str, default="admin")
    parser.add_argument("-p", "--password", type=str, default="admin")

    parser.add_argument(
        "--mode",
        choices=["sequence", "open", "close", "read"],
        default="sequence",
        help="sequence=open -> close -> open; open=open only; close=close only; read=read current gripper position only.",
    )
    parser.add_argument(
        "--open-value",
        type=float,
        default=0.0,
        help="Normalized gripper open position. Usually 0.0.",
    )
    parser.add_argument(
        "--close-value",
        type=float,
        default=0.55,
        help="Normalized gripper close position. Use 0.35~0.60 first; 1.0 is fully closed.",
    )
    parser.add_argument(
        "--step",
        type=float,
        default=0.05,
        help="Position increment for ramped movement. Smaller is gentler.",
    )
    parser.add_argument(
        "--settle-time",
        type=float,
        default=0.25,
        help="Seconds to wait after each gripper command step.",
    )
    parser.add_argument(
        "--final-wait",
        type=float,
        default=1.0,
        help="Seconds to wait after reaching final command.",
    )
    parser.add_argument(
        "--no-confirm",
        action="store_true",
        help="Skip typed confirmation prompt.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print measured gripper movement after each step when available.",
    )
    return parser.parse_args()


def clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def get_measured_gripper_position(base: BaseClient):
    """Return measured normalized gripper position, or None if unavailable."""
    request = Base_pb2.GripperRequest()
    request.mode = Base_pb2.GRIPPER_POSITION
    measure = base.GetMeasuredGripperMovement(request)
    if len(measure.finger) == 0:
        return None
    return float(measure.finger[0].value)


def send_gripper_position(base: BaseClient, position: float, finger_id: int = 1):
    """Send one gripper position command.

    Kortex position mode uses a normalized position command.
    Convention used by Kinova official example:
        0.0 -> open
        1.0 -> closed
    """
    position = clamp01(position)

    command = Base_pb2.GripperCommand()
    command.mode = Base_pb2.GRIPPER_POSITION

    finger = command.gripper.finger.add()
    finger.finger_identifier = finger_id
    finger.value = position

    base.SendGripperCommand(command)


def ramp_gripper_to(base: BaseClient, target: float, step: float, settle_time: float, verbose: bool):
    """Ramp from measured/current position to target using small position steps."""
    target = clamp01(target)
    step = abs(float(step))
    if step <= 0:
        raise ValueError("--step must be > 0")

    current = get_measured_gripper_position(base)
    if current is None:
        # If no measurement is available, send the target once.
        print("Measured gripper position unavailable; sending target directly.")
        send_gripper_position(base, target)
        time.sleep(settle_time)
        return

    print(f"Measured current gripper position: {current:.3f}")
    print(f"Target gripper position: {target:.3f}")

    if abs(target - current) < 1e-3:
        print("Already near target.")
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

        if verbose:
            measured = get_measured_gripper_position(base)
            if measured is not None:
                print(f"  measured: {measured:.3f}")

        if abs(value - target) < 1e-6:
            break


def run_sequence(base: BaseClient, args):
    open_value = clamp01(args.open_value)
    close_value = clamp01(args.close_value)

    print("\n========== Gripper test sequence ==========")
    print(f"open_value  = {open_value:.3f}")
    print(f"close_value = {close_value:.3f}")
    print("Sequence: open -> partial close -> open")

    print("\n[1] Opening gripper")
    ramp_gripper_to(base, open_value, args.step, args.settle_time, args.verbose)
    time.sleep(args.final_wait)

    print("\n[2] Closing gripper partially")
    ramp_gripper_to(base, close_value, args.step, args.settle_time, args.verbose)
    time.sleep(args.final_wait)

    print("\n[3] Opening gripper again")
    ramp_gripper_to(base, open_value, args.step, args.settle_time, args.verbose)
    time.sleep(args.final_wait)

    final_pos = get_measured_gripper_position(base)
    print("\n========== Result ==========")
    if final_pos is None:
        print("Final measured gripper position unavailable, but commands were sent.")
    else:
        print(f"Final measured gripper position: {final_pos:.3f}")
    print("Gripper test completed.")


def main():
    args = parse_args()

    print("========== Real Gripper Test ==========")
    print(f"Robot IP: {args.ip}")
    print("Position convention: 0.0=open, 1.0=closed")
    print("For the first test, keep the gripper empty and clear of objects.")

    if not args.no_confirm and args.mode != "read":
        answer = input("\nType GRIPPER to connect and run gripper command: ").strip()
        if answer != "GRIPPER":
            print("Confirmation not received. Exiting.")
            return 1

    with utilities.DeviceConnection.createTcpConnection(args) as router:
        base = BaseClient(router)

        measured = get_measured_gripper_position(base)
        if measured is None:
            print("Initial measured gripper position: unavailable")
        else:
            print(f"Initial measured gripper position: {measured:.3f}")

        if args.mode == "read":
            return 0

        if args.mode == "sequence":
            run_sequence(base, args)
        elif args.mode == "open":
            print("\nOpening gripper...")
            ramp_gripper_to(base, args.open_value, args.step, args.settle_time, args.verbose)
            time.sleep(args.final_wait)
        elif args.mode == "close":
            print("\nClosing gripper...")
            ramp_gripper_to(base, args.close_value, args.step, args.settle_time, args.verbose)
            time.sleep(args.final_wait)

        measured_after = get_measured_gripper_position(base)
        if measured_after is not None:
            print(f"Measured gripper position after command: {measured_after:.3f}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Official-style Kortex example: ExecuteAction with reach_joint_angles.

This is intentionally tiny and conservative: it reads current joint angles,
adds a small delta to one joint, and sends a high-level angular action.
"""

from __future__ import annotations

import threading

import utilities
from kortex_api.autogen.client_stubs.BaseClientRpc import BaseClient
from kortex_api.autogen.messages import Base_pb2


def check_for_end_or_abort(event):
    def check(notification, event=event):
        if notification.action_event == Base_pb2.ACTION_END:
            print("EVENT: ACTION_END")
            event.set()
        elif notification.action_event == Base_pb2.ACTION_ABORT:
            print("EVENT: ACTION_ABORT")
            print("abort_details:", Base_pb2.SubErrorCodes.Name(notification.abort_details))
            event.set()

    return check


def main():
    parser = utilities.parse_connection_arguments(__doc__)
    parser.add_argument("--joint", type=int, default=0, help="Zero-based joint id.")
    parser.add_argument("--delta", type=float, default=1.0, help="Delta angle in degrees.")
    parser.add_argument("--timeout", type=float, default=30.0)
    args = parser.parse_args()

    print("This example will use ExecuteAction(reach_joint_angles).")
    print(f"Requested: joint {args.joint} += {args.delta:.3f} deg")
    answer = input("Type OFFICIAL to execute: ").strip()
    if answer != "OFFICIAL":
        print("Confirmation not received. Exiting.")
        return

    with utilities.DeviceConnection.create_tcp_connection(args) as router:
        base = BaseClient(router)
        base.ClearFaults()

        current = base.GetMeasuredJointAngles()
        current_values = [float(j.value) for j in current.joint_angles]
        print("Current joint angles [deg]:")
        print(current_values)

        if not 0 <= args.joint < len(current_values):
            raise ValueError(f"Joint {args.joint} is out of range for {len(current_values)} joints.")

        target_values = current_values[:]
        target_values[args.joint] += args.delta

        action = Base_pb2.Action()
        action.name = "small angular action"
        action.application_data = ""
        joint_angles = action.reach_joint_angles.joint_angles.joint_angles
        for joint_id, value in enumerate(target_values):
            joint_angle = joint_angles.add()
            joint_angle.joint_identifier = joint_id
            joint_angle.value = value

        print("Target joint angles [deg]:")
        print(target_values)

        event = threading.Event()
        notification_handle = base.OnNotificationActionTopic(
            check_for_end_or_abort(event),
            Base_pb2.NotificationOptions(),
        )

        print("Executing action...")
        base.ExecuteAction(action)
        finished = event.wait(args.timeout)
        base.Unsubscribe(notification_handle)

        after = base.GetMeasuredJointAngles()
        after_values = [float(j.value) for j in after.joint_angles]
        print("After joint angles [deg]:")
        print(after_values)
        print("Finished:", finished)


if __name__ == "__main__":
    main()

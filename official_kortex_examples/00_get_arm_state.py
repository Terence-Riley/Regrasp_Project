#!/usr/bin/env python3
"""Official-style Kortex example: connect and read arm state."""

from __future__ import annotations

import utilities
from kortex_api.autogen.client_stubs.BaseClientRpc import BaseClient


def main():
    parser = utilities.parse_connection_arguments(__doc__)
    args = parser.parse_args()

    with utilities.DeviceConnection.create_tcp_connection(args) as router:
        base = BaseClient(router)
        print("Arm state:")
        print(base.GetArmState())
        print("\nMeasured Cartesian pose:")
        print(base.GetMeasuredCartesianPose())
        print("\nMeasured joint angles:")
        print(base.GetMeasuredJointAngles())


if __name__ == "__main__":
    main()

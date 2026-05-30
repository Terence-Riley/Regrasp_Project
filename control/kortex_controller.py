"""Minimal Kinova Kortex controller wrapper.

The Kortex Python API is distributed by Kinova as a wheel inside their release
package, not as a normal PyPI package. Install that wheel before using this
module on the real robot.
"""

from __future__ import annotations

import threading
import collections
import collections.abc
import time
from dataclasses import dataclass


KORTEX_INSTALL_HINT = """
Kinova Kortex Python API is not installed in this Python environment.

Kinova distributes it as a .whl file in the official Kortex release package.
Install the wheel first, for example:

    python -m pip install /path/to/kortex_api-*.whl

Then rerun this script.
"""


@dataclass(frozen=True)
class CartesianPose:
    x: float
    y: float
    z: float
    theta_x: float
    theta_y: float
    theta_z: float

    @classmethod
    def from_kortex(cls, pose):
        return cls(
            x=float(pose.x),
            y=float(pose.y),
            z=float(pose.z),
            theta_x=float(pose.theta_x),
            theta_y=float(pose.theta_y),
            theta_z=float(pose.theta_z),
        )


class KortexController:
    """Small context-manager wrapper around Kortex BaseClient."""

    def __init__(
        self,
        ip: str = "192.168.1.10",
        username: str = "admin",
        password: str = "admin",
        port: int = 10000,
    ):
        self.ip = ip
        self.username = username
        self.password = password
        self.port = port
        self.transport = None
        self.router = None
        self.session_manager = None
        self.base = None
        self._imports = None

    def __enter__(self):
        self.connect()
        return self

    def __exit__(self, exc_type, exc, tb):
        self.disconnect()

    def _load_kortex_api(self):
        # Kinova's older Kortex API wheels can bundle protobuf 3.5.x, which still
        # imports ABCs from collections. Python 3.10 moved them to collections.abc.
        for name in ["Mapping", "MutableMapping", "Sequence", "MutableSequence", "Iterable"]:
            if not hasattr(collections, name):
                setattr(collections, name, getattr(collections.abc, name))

        try:
            from kortex_api.TCPTransport import TCPTransport
            from kortex_api.RouterClient import RouterClient
            from kortex_api.SessionManager import SessionManager
            from kortex_api.autogen.client_stubs.BaseClientRpc import BaseClient
            from kortex_api.autogen.messages import Base_pb2, Session_pb2
        except ImportError as exc:
            raise RuntimeError(KORTEX_INSTALL_HINT) from exc

        return TCPTransport, RouterClient, SessionManager, BaseClient, Base_pb2, Session_pb2

    def connect(self):
        TCPTransport, RouterClient, SessionManager, BaseClient, Base_pb2, Session_pb2 = (
            self._load_kortex_api()
        )
        self._imports = (TCPTransport, RouterClient, SessionManager, BaseClient, Base_pb2, Session_pb2)

        self.transport = TCPTransport()
        self.router = RouterClient(self.transport, RouterClient.basicErrorCallback)
        self.transport.connect(self.ip, self.port)

        session_info = Session_pb2.CreateSessionInfo()
        session_info.username = self.username
        session_info.password = self.password
        session_info.session_inactivity_timeout = 60000
        session_info.connection_inactivity_timeout = 2000

        self.session_manager = SessionManager(self.router)
        self.session_manager.CreateSession(session_info)

        self.base = BaseClient(self.router)

    def disconnect(self):
        if self.session_manager is not None:
            try:
                self.session_manager.CloseSession()
            finally:
                self.session_manager = None

        if self.transport is not None:
            self.transport.disconnect()
            self.transport = None

        self.router = None
        self.base = None

    def get_measured_cartesian_pose(self) -> CartesianPose:
        return CartesianPose.from_kortex(self.base.GetMeasuredCartesianPose())

    def get_measured_joint_angles(self):
        return self.base.GetMeasuredJointAngles()

    def make_joint_angles(self, values_deg):
        Base_pb2 = self._imports[4]
        joint_angles = Base_pb2.JointAngles()
        for idx, value in enumerate(values_deg):
            joint_angle = joint_angles.joint_angles.add()
            joint_angle.joint_identifier = int(idx)
            joint_angle.value = float(value)
        return joint_angles

    def joint_angles_to_list(self, joint_angles):
        return [float(j.value) for j in joint_angles.joint_angles]

    def normalized_joint_guess(self, joint_angles):
        values = self.joint_angles_to_list(joint_angles)
        normalized = [((value + 180.0) % 360.0) - 180.0 for value in values]
        return self.make_joint_angles(normalized)

    def compute_inverse_kinematics(self, target_pose: CartesianPose, guess_joint_angles=None):
        Base_pb2 = self._imports[4]
        ik_data = Base_pb2.IKData()
        ik_data.cartesian_pose.x = float(target_pose.x)
        ik_data.cartesian_pose.y = float(target_pose.y)
        ik_data.cartesian_pose.z = float(target_pose.z)
        ik_data.cartesian_pose.theta_x = float(target_pose.theta_x)
        ik_data.cartesian_pose.theta_y = float(target_pose.theta_y)
        ik_data.cartesian_pose.theta_z = float(target_pose.theta_z)

        if guess_joint_angles is not None:
            ik_data.guess.CopyFrom(guess_joint_angles)

        return self.base.ComputeInverseKinematics(ik_data)

    def compute_inverse_kinematics_with_guesses(self, target_pose: CartesianPose, guesses):
        errors = []
        for name, guess in guesses:
            try:
                return name, self.compute_inverse_kinematics(target_pose, guess_joint_angles=guess)
            except Exception as exc:
                errors.append(f"{name}: {exc}")
        raise RuntimeError("All IK guesses failed:\n" + "\n".join(errors))

    def clear_faults(self):
        self.base.ClearFaults()

    def get_arm_state(self):
        return self.base.GetArmState()

    def get_controller_state(self):
        return self.base.GetControllerState()

    def get_servoing_mode(self):
        return self.base.GetServoingMode()

    def set_single_level_servoing(self):
        Base_pb2 = self._imports[4]
        servoing_mode = Base_pb2.ServoingModeInformation()
        servoing_mode.servoing_mode = Base_pb2.SINGLE_LEVEL_SERVOING
        self.base.SetServoingMode(servoing_mode)

    def action_event_name(self, event_value: int) -> str:
        Base_pb2 = self._imports[4]
        return Base_pb2.ActionEvent.Name(event_value)

    def sub_error_name(self, error_value: int) -> str:
        Base_pb2 = self._imports[4]
        return Base_pb2.SubErrorCodes.Name(error_value)

    def move_to_cartesian_pose(
        self,
        target_pose: CartesianPose,
        name: str = "move_above_cup",
        timeout_s: float = 60.0,
        translation_speed_m_s: float = 0.03,
        orientation_speed_deg_s: float = 10.0,
        method: str = "trajectory",
    ) -> bool:
        """Move to an absolute Cartesian pose.

        method="trajectory" uses Base.PlayCartesianTrajectory directly.
        method="action" wraps the same target in an ExecuteAction reach_pose.
        """
        Base_pb2 = self._imports[4]

        constrained_pose = Base_pb2.ConstrainedPose()
        target = constrained_pose.target_pose
        target.x = float(target_pose.x)
        target.y = float(target_pose.y)
        target.z = float(target_pose.z)
        target.theta_x = float(target_pose.theta_x)
        target.theta_y = float(target_pose.theta_y)
        target.theta_z = float(target_pose.theta_z)
        constrained_pose.constraint.speed.translation = float(translation_speed_m_s)
        constrained_pose.constraint.speed.orientation = float(orientation_speed_deg_s)

        if method == "action":
            action = Base_pb2.Action()
            action.name = name[:20]
            action.application_data = ""
            action.reach_pose.CopyFrom(constrained_pose)
            return self._execute_with_action_notification(
                lambda: self.base.ExecuteAction(action),
                name=name,
                timeout_s=timeout_s,
            )

        if method == "trajectory":
            return self._execute_with_action_notification(
                lambda: self.base.PlayCartesianTrajectory(constrained_pose),
                name=name,
                timeout_s=timeout_s,
            )

        raise ValueError(f"Unknown Cartesian motion method: {method}")

    def move_to_joint_angles(
        self,
        joint_angles,
        name: str = "joint_move",
        timeout_s: float = 60.0,
        duration_s: float = 8.0,
    ) -> bool:
        Base_pb2 = self._imports[4]
        constrained = Base_pb2.ConstrainedJointAngles()
        constrained.joint_angles.CopyFrom(joint_angles)
        constrained.constraint.type = Base_pb2.JOINT_CONSTRAINT_DURATION
        constrained.constraint.value = float(duration_s)

        return self._execute_with_action_notification(
            lambda: self.base.PlayJointTrajectory(constrained),
            name=name,
            timeout_s=timeout_s,
        )

    def _execute_with_action_notification(self, command, name: str, timeout_s: float) -> bool:
        Base_pb2 = self._imports[4]

        finished = threading.Event()
        result = {"ok": False, "event": None, "abort_details": None}

        def on_notification(notification):
            result["event"] = int(notification.action_event)
            if notification.action_event == Base_pb2.ACTION_END:
                result["ok"] = True
                finished.set()
            elif notification.action_event == Base_pb2.ACTION_ABORT:
                result["ok"] = False
                result["abort_details"] = int(notification.abort_details)
                finished.set()

        notification_handle = self.base.OnNotificationActionTopic(
            on_notification,
            Base_pb2.NotificationOptions(),
        )

        try:
            command()
            if not finished.wait(timeout_s):
                raise TimeoutError(f"Timed out waiting for Kortex action after {timeout_s:.1f} s.")
            if not result["ok"]:
                event = result["event"]
                details = result["abort_details"]
                event_name = self.action_event_name(event) if event is not None else "UNKNOWN"
                detail_name = self.sub_error_name(details) if details is not None else "UNKNOWN"
                print(f"Kortex action {name!r} ended with {event_name}, abort_details={detail_name}.")
            return bool(result["ok"])
        finally:
            self.base.Unsubscribe(notification_handle)

    def send_twist_base(
        self,
        linear_x: float = 0.0,
        linear_y: float = 0.0,
        linear_z: float = 0.0,
        angular_x: float = 0.0,
        angular_y: float = 0.0,
        angular_z: float = 0.0,
        duration_s: float = 0.2,
        use_joystick: bool = False,
        reference_frame: str = "base",
    ):
        """Send one Cartesian twist command.

        Linear units are m/s. Angular units follow Kortex Pose/Twist convention
        and are degrees/s.
        """
        Base_pb2 = self._imports[4]
        command = Base_pb2.TwistCommand()
        if reference_frame == "tool":
            command.reference_frame = Base_pb2.CARTESIAN_REFERENCE_FRAME_TOOL
        elif reference_frame == "mixed":
            command.reference_frame = Base_pb2.CARTESIAN_REFERENCE_FRAME_MIXED
        else:
            command.reference_frame = Base_pb2.CARTESIAN_REFERENCE_FRAME_BASE
        command.duration = int(max(1, duration_s * 1000.0))
        command.twist.linear_x = float(linear_x)
        command.twist.linear_y = float(linear_y)
        command.twist.linear_z = float(linear_z)
        command.twist.angular_x = float(angular_x)
        command.twist.angular_y = float(angular_y)
        command.twist.angular_z = float(angular_z)
        if use_joystick:
            self.base.SendTwistJoystickCommand(command)
        else:
            self.base.SendTwistCommand(command)

    def stop(self):
        self.base.Stop()

    def move_relative_z_with_twist(
        self,
        dz_m: float,
        speed_m_s: float = 0.02,
        command_period_s: float = 0.1,
        use_joystick: bool = False,
        reference_frame: str = "base",
    ):
        """Move approximately along base z using repeated twist commands."""
        if dz_m == 0.0:
            return

        direction = 1.0 if dz_m > 0.0 else -1.0
        speed = abs(float(speed_m_s)) * direction
        total_time_s = abs(float(dz_m)) / abs(float(speed_m_s))
        steps = max(1, int(total_time_s / command_period_s))

        try:
            for _ in range(steps):
                self.send_twist_base(
                    linear_z=speed,
                    duration_s=command_period_s,
                    use_joystick=use_joystick,
                    reference_frame=reference_frame,
                )
                time.sleep(command_period_s)
        finally:
            self.send_twist_base(
                duration_s=0.05,
                use_joystick=use_joystick,
                reference_frame=reference_frame,
            )
            self.stop()

    def send_joint_speed(
        self,
        joint_identifier: int,
        speed_deg_s: float,
        duration_s: float = 0.2,
        use_joystick: bool = False,
    ):
        """Send one joint speed command. Joint identifiers are zero-based."""
        Base_pb2 = self._imports[4]
        command = Base_pb2.JointSpeeds()
        command.duration = int(max(1, duration_s * 1000.0))
        joint_speed = command.joint_speeds.add()
        joint_speed.joint_identifier = int(joint_identifier)
        joint_speed.value = float(speed_deg_s)
        joint_speed.duration = int(max(1, duration_s * 1000.0))
        if use_joystick:
            self.base.SendJointSpeedsJoystickCommand(command)
        else:
            self.base.SendJointSpeedsCommand(command)

    def move_joint_with_speed(
        self,
        joint_identifier: int,
        speed_deg_s: float = 2.0,
        duration_s: float = 1.0,
        command_period_s: float = 0.1,
        use_joystick: bool = False,
    ):
        steps = max(1, int(duration_s / command_period_s))
        try:
            for _ in range(steps):
                self.send_joint_speed(
                    joint_identifier=joint_identifier,
                    speed_deg_s=speed_deg_s,
                    duration_s=command_period_s,
                    use_joystick=use_joystick,
                )
                time.sleep(command_period_s)
        finally:
            self.send_joint_speed(
                joint_identifier=joint_identifier,
                speed_deg_s=0.0,
                duration_s=0.05,
                use_joystick=use_joystick,
            )
            self.stop()

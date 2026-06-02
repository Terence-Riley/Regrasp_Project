#!/usr/bin/env python3
"""Smoke-test Pink/Pinocchio IK on the Kinova Gen3 Lite URDF.

中文说明：该脚本只做离线 Pink/Pinocchio 逆运动学测试，不连接真机、
不发送任何运动命令。它用于验证 Gen3 Lite URDF、tool frame、QP solver
和外部 IK 求解流程是否可用。

This script is offline only. It does not connect to Kortex and it does not move
the real robot. It verifies whether Pink can solve a small target pose from a
given starting joint configuration, then prints the resulting six arm joint
angles that can later be passed to Kortex reach_joint_angles/PlayJointTrajectory.
"""

from __future__ import annotations

import argparse
import math
import shutil
import sys
import tempfile
import uuid
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]


ARM_JOINT_NAMES = [f"joint_{idx}" for idx in range(1, 7)]


def is_ascii_path(path: Path) -> bool:
    try:
        str(path).encode("ascii")
        return True
    except UnicodeEncodeError:
        return False


def make_ascii_urdf_copy(urdf: Path) -> Path:
    """Copy URDF to an ASCII temp path for Pinocchio/urdfdom on Windows.

    urdfdom can mangle non-ASCII Windows paths. While copying, resolve
    package://kortex_description mesh URLs to absolute file paths so the copied
    URDF remains loadable from the temp directory.
    """

    temp_dir = Path(tempfile.gettempdir()) / f"regrasp_pin_{uuid.uuid4().hex}"
    temp_dir.mkdir(parents=True, exist_ok=True)
    temp_package = temp_dir / "kortex_description"
    shutil.copytree(PROJECT_ROOT / "kortex_description", temp_package)

    text = urdf.read_text(encoding="utf-8")
    text = text.replace(
        "package://kortex_description",
        temp_package.as_posix(),
    )
    temp_urdf = temp_dir / urdf.name
    temp_urdf.write_text(text, encoding="utf-8")
    return temp_urdf


def cleanup_ascii_urdf_copy(temp_urdf: Path, original_urdf: Path):
    if temp_urdf == original_urdf:
        return
    try:
        shutil.rmtree(temp_urdf.parent)
    except OSError:
        pass


def solve_pink_ik(args, current_joints_deg=None):
    pin, pink, solve_ik, FrameTask, PostureTask = require_pink_pinocchio()

    urdf = args.urdf if args.urdf.is_absolute() else PROJECT_ROOT / args.urdf
    if not urdf.exists():
        raise FileNotFoundError(f"Missing URDF: {urdf}")

    urdf_for_pin = make_ascii_urdf_copy(urdf) if not is_ascii_path(urdf) else urdf
    try:
        model = pin.buildModelFromUrdf(str(urdf_for_pin))
        data = model.createData()

        if args.list_frames:
            print("Frames:")
            for frame in model.frames:
                print(f"  {frame.name}")
            return None

        if model.getFrameId(args.frame) >= len(model.frames):
            raise KeyError(f"Frame not found: {args.frame}. Run with --list-frames.")

        q0 = pin.neutral(model)
        if current_joints_deg is None:
            current_joints_deg = args.current_joints_deg
        if current_joints_deg is not None:
            if not args.no_wrap_current_joints:
                current_joints_deg = wrap_degrees_180(current_joints_deg)
            q0 = set_arm_joints_deg(model, q0, current_joints_deg)
            if getattr(args, "clamp_current_to_limits", False):
                q0 = clamp_configuration_to_limits(
                    model,
                    q0,
                    margin_rad=float(getattr(args, "limit_margin_rad", 1e-4)),
                    verbose=True,
                )

        configuration = pink.Configuration(model, data, q0)
        current_pose = configuration.get_transform_frame_to_world(args.frame)

        target_translation = np.array(current_pose.translation, dtype=np.float64)
        if args.target_position is None:
            target_translation = target_translation + np.asarray(args.target_offset, dtype=np.float64)
        else:
            target_translation = np.asarray(args.target_position, dtype=np.float64)

        target_rotation = np.array(current_pose.rotation, dtype=np.float64)
        if getattr(args, "target_rotation_matrix", None) is not None:
            target_rotation = np.asarray(args.target_rotation_matrix, dtype=np.float64).reshape(3, 3)
        elif args.target_rpy_deg is not None:
            target_rotation = rpy_deg_to_matrix(pin, args.target_rpy_deg)

        target_pose = pin.SE3(target_rotation, target_translation)

        frame_task = FrameTask(
            args.frame,
            position_cost=float(args.position_cost),
            orientation_cost=float(args.orientation_cost),
            lm_damping=1e-4,
        )
        frame_task.set_target(target_pose)

        posture_task = PostureTask(cost=float(args.posture_cost))
        posture_task.set_target(q0)
        tasks = [frame_task, posture_task]

        solver = choose_solver(args.solver)

        print("\n========== Pink IK Smoke Test ==========")
        print("Offline only: no Kortex motion command will be sent.")
        print(f"URDF: {urdf}")
        if urdf_for_pin != urdf:
            print(f"Pinocchio loaded temporary ASCII URDF: {urdf_for_pin}")
        print(f"solver: {solver}")
        print_model_summary(model, args.frame)
        print("\nInitial six arm joints [deg]:")
        print("  " + ", ".join(f"{v:.3f}" for v in get_arm_joints_deg(model, q0)))
        print("\nCurrent frame position:")
        print("  " + ", ".join(f"{v:.5f}" for v in current_pose.translation))
        current_rpy_deg = matrix_to_rpy_deg(pin, current_pose.rotation)
        print("Current frame RPY [deg]:")
        print("  " + ", ".join(f"{v:.3f}" for v in current_rpy_deg))
        print("Target frame position:")
        print("  " + ", ".join(f"{v:.5f}" for v in target_pose.translation))
        target_rpy_deg = matrix_to_rpy_deg(pin, target_pose.rotation)
        print("Target frame RPY [deg]:")
        print("  " + ", ".join(f"{v:.3f}" for v in target_rpy_deg))

        converged = False
        iteration = 0
        for iteration in range(1, int(args.iterations) + 1):
            velocity = solve_ik(configuration, tasks, dt=float(args.dt), solver=solver)
            configuration.integrate_inplace(velocity, float(args.dt))
            pose = configuration.get_transform_frame_to_world(args.frame)
            pos_error, orient_error = pose_errors(pin, pose, target_pose)
            if pos_error <= args.position_tolerance and orient_error <= args.orientation_tolerance:
                converged = True
                break

        q_sol = np.array(configuration.q, dtype=np.float64)
        final_pose = configuration.get_transform_frame_to_world(args.frame)
        pos_error, orient_error = pose_errors(pin, final_pose, target_pose)
        final_rpy_deg = matrix_to_rpy_deg(pin, final_pose.rotation)
        arm_start_deg = get_arm_joints_deg(model, q0)
        arm_sol_deg = get_arm_joints_deg(model, q_sol)
        delta_deg = np.asarray(arm_sol_deg) - np.asarray(arm_start_deg)

        print("\nResult:")
        print(f"  converged: {converged}")
        print(f"  iterations: {iteration}")
        print(f"  position_error_m: {pos_error:.6f}")
        print(f"  orientation_error_rad: {orient_error:.6f}")
        print("  final_frame_rpy_deg: " + ", ".join(f"{v:.3f}" for v in final_rpy_deg))
        print("\nSolved six arm joints [deg], Kortex order joint_1..joint_6:")
        print("  " + ", ".join(f"{v:.3f}" for v in arm_sol_deg))
        print("Joint deltas from start [deg]:")
        print("  " + ", ".join(f"{v:.3f}" for v in delta_deg))

        if not converged:
            print("\nNot converged. Try increasing --iterations, lowering orientation cost, or using a smaller target offset.")

        return {
            "converged": converged,
            "iterations": iteration,
            "position_error_m": pos_error,
            "orientation_error_rad": orient_error,
            "start_joints_deg": arm_start_deg,
            "solved_joints_deg": arm_sol_deg,
            "delta_deg": delta_deg.tolist(),
        }
    finally:
        cleanup_ascii_urdf_copy(urdf_for_pin, urdf)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
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
        "--current-joints-deg",
        type=float,
        nargs=6,
        default=None,
        help="Current six arm joint angles in degrees, in Kortex order joint_1..joint_6.",
    )
    parser.add_argument(
        "--no-wrap-current-joints",
        action="store_true",
        help="Do not wrap --current-joints-deg from Kortex 0..360 style to [-180, 180].",
    )
    parser.add_argument(
        "--clamp-current-to-limits",
        action="store_true",
        help="Clamp the initial joint seed inside URDF limits. Useful when Kortex reports a joint a tiny bit outside URDF limits.",
    )
    parser.add_argument(
        "--limit-margin-rad",
        type=float,
        default=1e-4,
        help="Small margin used by --clamp-current-to-limits.",
    )
    parser.add_argument(
        "--target-offset",
        type=float,
        nargs=3,
        default=[0.0, 0.0, 0.02],
        metavar=("DX", "DY", "DZ"),
        help="Target translation offset from the current frame pose, in meters.",
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
    parser.add_argument("--dt", type=float, default=0.05, help="Differential IK time step in seconds.")
    parser.add_argument("--iterations", type=int, default=200, help="Maximum Pink IK iterations.")
    parser.add_argument("--position-cost", type=float, default=60.0)
    parser.add_argument("--orientation-cost", type=float, default=8.0)
    parser.add_argument("--posture-cost", type=float, default=0.05)
    parser.add_argument("--position-tolerance", type=float, default=0.002, help="Meters.")
    parser.add_argument("--orientation-tolerance", type=float, default=0.03, help="Radians.")
    parser.add_argument(
        "--solver",
        default=None,
        help="QP solver name. If omitted, choose the first available among quadprog/osqp/proxqp/clarabel/daqp.",
    )
    parser.add_argument(
        "--list-frames",
        action="store_true",
        help="List URDF frame names and exit.",
    )
    return parser.parse_args()


def require_pink_pinocchio():
    try:
        import pinocchio as pin
        import pink
        from pink import solve_ik
        from pink.tasks import FrameTask, PostureTask
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "Pink/Pinocchio is not installed. Try one of:\n"
            "  conda install -c conda-forge pink\n"
            "  python -m pip install pin-pink qpsolvers osqp\n"
            "Then rerun this script in the same environment."
        ) from exc
    return pin, pink, solve_ik, FrameTask, PostureTask


def choose_solver(requested):
    if requested:
        return requested
    try:
        from qpsolvers import available_solvers

        available = list(available_solvers)
    except Exception:
        available = []
    for candidate in ("quadprog", "osqp", "proxqp", "clarabel", "daqp"):
        if candidate in available:
            return candidate
    return "osqp"


def joint_index(model, joint_name):
    jid = model.getJointId(joint_name)
    if jid == 0 or jid >= len(model.joints):
        raise KeyError(f"Joint not found in URDF model: {joint_name}")
    return int(model.idx_qs[jid])


def set_arm_joints_deg(model, q, joints_deg):
    q = np.array(q, dtype=np.float64).copy()
    for name, value_deg in zip(ARM_JOINT_NAMES, joints_deg):
        q[joint_index(model, name)] = math.radians(float(value_deg))
    return q


def wrap_degrees_180(values_deg):
    return [((float(value) + 180.0) % 360.0) - 180.0 for value in values_deg]


def get_arm_joints_deg(model, q):
    values = []
    for name in ARM_JOINT_NAMES:
        values.append(math.degrees(float(q[joint_index(model, name)])))
    return values


def clamp_configuration_to_limits(model, q, margin_rad=1e-4, verbose=False):
    clipped = np.array(q, dtype=np.float64).copy()
    lower = np.asarray(model.lowerPositionLimit, dtype=np.float64)
    upper = np.asarray(model.upperPositionLimit, dtype=np.float64)
    changes = []
    for idx in range(model.nq):
        lo = float(lower[idx])
        hi = float(upper[idx])
        if not np.isfinite(lo) or not np.isfinite(hi) or lo >= hi:
            continue
        safe_lo = lo + float(margin_rad)
        safe_hi = hi - float(margin_rad)
        before = float(clipped[idx])
        after = min(max(before, safe_lo), safe_hi)
        if abs(after - before) > 1e-12:
            clipped[idx] = after
            changes.append((idx, before, after, lo, hi))

    if verbose and changes:
        print("\nClamped initial q to URDF limits:")
        for idx, before, after, lo, hi in changes:
            print(
                f"  q[{idx}]: {math.degrees(before):.3f} deg -> {math.degrees(after):.3f} deg "
                f"(limit {math.degrees(lo):.3f}..{math.degrees(hi):.3f} deg)"
            )
    return clipped


def rpy_deg_to_matrix(pin, rpy_deg):
    rpy_rad = np.radians(np.asarray(rpy_deg, dtype=np.float64))
    return pin.rpy.rpyToMatrix(float(rpy_rad[0]), float(rpy_rad[1]), float(rpy_rad[2]))


def matrix_to_rpy_deg(pin, rotation):
    return np.degrees(pin.rpy.matrixToRpy(np.asarray(rotation, dtype=np.float64)))


def pose_errors(pin, current, target):
    pos_error = float(np.linalg.norm(target.translation - current.translation))
    local_error = pin.log(current.inverse() * target).vector
    orient_error = float(np.linalg.norm(local_error[3:]))
    return pos_error, orient_error


def print_model_summary(model, frame_name):
    print("\nModel summary:")
    print(f"  nq={model.nq}, nv={model.nv}")
    print(f"  controlled frame: {frame_name}")
    print("  arm joint q indices:")
    for name in ARM_JOINT_NAMES:
        print(f"    {name}: q[{joint_index(model, name)}]")


def main():
    args = parse_args()
    solve_pink_ik(args)


if __name__ == "__main__":
    main()

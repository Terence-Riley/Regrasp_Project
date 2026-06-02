#!/usr/bin/env python3
"""Pick one detected cup with Pink IK and Kortex joint trajectories.

中文说明：单杯真实抓取 smoke test。脚本用 RealSense + mesh pose 检测一个杯子，
取 mesh center 的 x/y/z 作为抓取点，固定 top-down/vertical-down 工具姿态，
然后执行 open -> pre-grasp -> grasp -> close -> lift。该脚本会移动真实机械臂
和夹爪，运行前必须确认急停和工作空间安全。
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path

import numpy as np
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from control.kortex_controller import KortexController  # noqa: E402
from scripts.real_gripper_test import ramp_gripper_to  # noqa: E402
from scripts.real_mesh_cup_pose_test import (  # noqa: E402
    BASE_Z,
    capture_cluster_pcds,
    fit_cluster_with_meshes,
    load_offline_pcds,
    load_mesh_models,
    load_workspaces,
    make_transform,
    rotation_align_vectors,
)
from scripts.test_pink_ik_gen3_lite import solve_pink_ik  # noqa: E402
from scripts.test_pink_ik_gen3_lite import (  # noqa: E402
    cleanup_ascii_urdf_copy,
    is_ascii_path,
    make_ascii_urdf_copy,
    require_pink_pinocchio,
    set_arm_joints_deg,
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument("--robot-config", type=Path, default=PROJECT_ROOT / "configs" / "robot_config.yaml")
    parser.add_argument("--urdf", type=Path, default=PROJECT_ROOT / "kortex_description" / "robots" / "gen3_lite.urdf")
    parser.add_argument("--frame", default="tool_frame")
    parser.add_argument("--mesh", action="append", required=True)
    parser.add_argument("--mesh-unit", choices=["m", "mm"], default="m")
    parser.add_argument("--mesh-scale", type=float, default=1.0)
    parser.add_argument("--mesh-axis", choices=["x", "y", "z", "-x", "-y", "-z"], default="y")
    parser.add_argument("--mesh-origin", choices=["bbox_center", "bbox_bottom_center", "mesh_origin"], default="bbox_center")
    parser.add_argument("--model-sample-points", type=int, default=5000)
    parser.add_argument("--model-voxel-size", type=float, default=0.004)

    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=15)
    parser.add_argument("--camera-stream-order", choices=["color_first", "depth_first", "both"], default="color_first")
    parser.add_argument("--camera-start-attempts", type=int, default=3)
    parser.add_argument("--camera-retry-delay", type=float, default=2.0)
    parser.add_argument("--camera-hardware-reset-on-fail", action="store_true")
    parser.add_argument("--calibration", type=Path, default=PROJECT_ROOT / "configs" / "real_calibration.yaml")
    parser.add_argument("--workspaces", type=Path, default=PROJECT_ROOT / "configs" / "real_workspaces.yaml")
    parser.add_argument("--workspace-name", choices=["pickup", "place"], default="pickup")

    # Perception parameters mirrored from real_mesh_cup_pose_test.
    parser.add_argument("--depth-min", type=float, default=0.05)
    parser.add_argument("--depth-trunc", type=float, default=1.20)
    parser.add_argument("--voxel-size", type=float, default=0.004)
    parser.add_argument("--table-distance-threshold", type=float, default=0.008)
    parser.add_argument("--near-table-clearance", type=float, default=0.006)
    parser.add_argument("--disable-near-table-filter", action="store_true")
    parser.add_argument("--cluster-eps", type=float, default=0.02)
    parser.add_argument("--cluster-min-points", type=int, default=50)
    parser.add_argument("--min-raw-points", type=int, default=200)
    parser.add_argument("--min-clean-points", type=int, default=120)
    parser.add_argument("--disable-depth-edge-filter", action="store_true")
    parser.add_argument("--depth-edge-threshold", type=float, default=0.02)
    parser.add_argument("--depth-edge-kernel", type=int, default=5)
    parser.add_argument("--disable-statistical-filter", action="store_true")
    parser.add_argument("--stat-nb-neighbors", type=int, default=40)
    parser.add_argument("--stat-std-ratio", type=float, default=1.0)
    parser.add_argument("--disable-radius-filter", action="store_true")
    parser.add_argument("--radius-nb-points", type=int, default=10)
    parser.add_argument("--radius", type=float, default=0.01)
    parser.add_argument("--post-clean-cluster-eps", type=float, default=0.018)
    parser.add_argument("--post-clean-cluster-min-points", type=int, default=20)
    parser.add_argument("--enable-dark-object-filter", action="store_true")
    parser.add_argument("--dark-value-max", type=float, default=0.38)
    parser.add_argument("--dark-saturation-min", type=float, default=0.0)
    parser.add_argument("--dark-rgb-max", type=float, default=None)

    parser.add_argument("--pick-sort", choices=["nearest", "largest", "x", "y"], default="nearest")
    parser.add_argument("--target-label", type=int, default=None, help="Pick this DBSCAN label. Default picks first sorted candidate.")
    parser.add_argument("--force-model-set", default=None)
    parser.add_argument("--fit-states", choices=["both", "upright", "lying"], default="both")
    parser.add_argument("--yaw-samples", type=int, default=12)
    parser.add_argument("--roll-samples", type=int, default=8)
    parser.add_argument("--icp-max-distance", type=float, default=0.025)
    parser.add_argument("--icp-iterations", type=int, default=50)
    parser.add_argument("--coverage-distance", type=float, default=None)
    parser.add_argument("--min-cluster-coverage", type=float, default=0.25)
    parser.add_argument("--extent-weight", type=float, default=1.2)
    parser.add_argument("--coverage-weight", type=float, default=0.9)
    parser.add_argument("--fitness-weight", type=float, default=0.4)
    parser.add_argument("--observed-size-prior-weight", type=float, default=1.4)
    parser.add_argument("--big-observed-diameter-min", type=float, default=0.058)
    parser.add_argument("--small-observed-diameter-max", type=float, default=0.055)
    parser.add_argument("--big-model-name", default="big")
    parser.add_argument("--small-model-name", default="small")
    parser.add_argument("--axis-report", choices=["model", "unoriented", "table_up"], default="table_up")
    parser.add_argument("--allow-unknown", action="store_true")
    parser.set_defaults(detect_inverted_upright=False)
    parser.add_argument("--detect-inverted-upright", dest="detect_inverted_upright", action="store_true")
    parser.add_argument("--disable-detect-inverted-upright", dest="detect_inverted_upright", action="store_false")
    parser.add_argument("--upright-axis-z-threshold", type=float, default=0.75)
    parser.add_argument("--lying-axis-z-threshold", type=float, default=0.45)
    parser.add_argument("--cup-height-min", type=float, default=0.065)
    parser.add_argument("--cup-height-max", type=float, default=0.085)
    parser.add_argument("--height-prior-weight", type=float, default=0.55)
    parser.add_argument("--table-z", type=float, default=None)
    parser.add_argument("--max-support-gap", type=float, default=0.035)
    parser.add_argument("--support-prior-weight", type=float, default=0.75)
    parser.add_argument("--upright-z-extent-min", type=float, default=0.045)
    parser.add_argument("--lying-z-extent-max", type=float, default=0.060)
    parser.add_argument("--state-prior-weight", type=float, default=0.35)
    parser.add_argument("--axis-snap-prior-weight", type=float, default=0.9)
    parser.add_argument("--upright-axis-z-ideal-min", type=float, default=0.92)
    parser.add_argument("--lying-axis-z-ideal-max", type=float, default=0.18)
    parser.add_argument("--hard-axis-snap", action="store_true")
    parser.add_argument("--upright-rim-prior-weight", type=float, default=0.0)
    parser.add_argument("--upright-rim-absolute-weight", type=float, default=0.0)
    parser.add_argument("--rim-absolute-scale", type=float, default=0.01)
    parser.add_argument("--rim-band-fraction", type=float, default=0.18)
    parser.add_argument("--rim-radius-quantile", type=float, default=0.80)
    parser.add_argument("--min-rim-points", type=int, default=20)
    parser.add_argument("--distinct-rim-ratio", type=float, default=1.12)
    parser.add_argument("--min-fitness", type=float, default=0.08)
    parser.add_argument("--max-rmse", type=float, default=0.035)
    parser.add_argument("--input-pcd", action="append", type=Path, default=[])
    parser.add_argument("--input-as-scene", action="store_true")
    parser.add_argument("--input-frame", choices=["base", "camera"], default="base")
    parser.add_argument("--skip-input-cleaning", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "debug_pointclouds" / "pink_pick_one")
    parser.add_argument("--save-debug-pcd", action="store_true")
    parser.add_argument("--visualize", action="store_true")

    # Pink and motion parameters.
    parser.add_argument(
        "--grasp-frame-mode",
        choices=["center_vertical", "mesh_y_aligned"],
        default="center_vertical",
        help="center_vertical uses the current center/top-down logic. mesh_y_aligned aligns tool origin/+Y with mesh origin/+Y.",
    )
    parser.add_argument(
        "--mesh-y-roll-samples",
        type=int,
        default=12,
        help="Number of roll candidates around mesh +Y when --grasp-frame-mode mesh_y_aligned.",
    )
    parser.add_argument("--tool-rpy-deg", type=float, nargs=3, default=[180.0, 0.0, 0.0], help="Fixed vertical-down tool RPY in URDF/world frame.")
    parser.add_argument(
        "--lying-align-gripper",
        action="store_true",
        default=True,
        help="For lying cups, yaw the vertical-down gripper so the closing direction is perpendicular to cup axis.",
    )
    parser.add_argument(
        "--no-lying-align-gripper",
        dest="lying_align_gripper",
        action="store_false",
        help="Disable lying-cup yaw alignment and use --tool-rpy-deg directly.",
    )
    parser.add_argument(
        "--gripper-closing-axis",
        choices=["x", "y"],
        default="y",
        help="Tool-frame horizontal axis that represents gripper closing direction. Switch to x if the jaw direction is rotated 90 deg.",
    )
    parser.add_argument(
        "--search-both-gripper-axes",
        action="store_true",
        help="Try both tool_x and tool_y as the gripper closing axis during full-chain grasp planning.",
    )
    parser.add_argument(
        "--force-lying-vertical-down",
        action="store_true",
        default=True,
        help="For lying cups, only use vertical-down tool poses so the open gripper approaches from above.",
    )
    parser.add_argument(
        "--no-force-lying-vertical-down",
        dest="force_lying_vertical_down",
        action="store_false",
        help="Allow non-vertical tool poses for lying cups. This is more likely to collide with the cup.",
    )
    parser.add_argument(
        "--max-lying-tool-z-tilt-deg",
        type=float,
        default=8.0,
        help="Maximum angle between tool +Z and base -Z for lying-cup grasp candidates.",
    )
    parser.add_argument(
        "--grasp-yaw-offsets-deg",
        type=float,
        nargs="*",
        default=[-45.0, -30.0, -15.0, 0.0, 15.0, 30.0, 45.0],
        help="Horizontal closing-direction offsets around the perpendicular-to-axis grasp direction.",
    )
    parser.add_argument(
        "--lying-axis-min-xy-norm",
        type=float,
        default=0.20,
        help="Minimum XY norm of reported cup axis required for lying-cup yaw alignment.",
    )
    parser.add_argument("--pre-grasp-height", type=float, default=0.10)
    parser.add_argument("--lift-height", type=float, default=0.12)
    parser.add_argument("--place-workspace-name", default="place")
    parser.add_argument("--place-bottom-clearance", type=float, default=0.012)
    parser.add_argument("--place-yaw-samples", type=int, default=12)
    parser.add_argument(
        "--rotate-stage-xy",
        choices=["lift", "place"],
        default="place",
        help="XY location used for upright rotation staging. 'place' is usually easier than rotating at the pickup lift point.",
    )
    parser.add_argument(
        "--rotate-stage-z",
        type=float,
        default=0.26,
        help="Absolute base-frame z used for pre-rotate and rotate-upright staging.",
    )
    parser.add_argument(
        "--rotate-stage-z-step",
        type=float,
        default=0.04,
        help="Extra z step used while automatically searching a feasible rotate staging pose.",
    )
    parser.add_argument(
        "--rotate-stage-z-samples",
        type=int,
        default=3,
        help="Number of z levels to try for rotate staging.",
    )
    parser.add_argument("--pre-place-height", type=float, default=0.12)
    parser.add_argument("--retreat-height", type=float, default=0.12)
    parser.add_argument("--pick-only", action="store_true", help="Stop after lift and keep the gripper closed.")
    parser.add_argument(
        "--grasp-z-offset",
        type=float,
        default=0.03,
        help="Added to detected mesh-center z before grasping. Positive values keep the tool higher above the cup/table.",
    )
    parser.add_argument(
        "--min-grasp-z-above-table",
        type=float,
        default=0.035,
        help="Clamp grasp z to at least table_z + this clearance when table_z is available.",
    )
    parser.add_argument(
        "--collision-table-z",
        type=float,
        default=None,
        help="Base-frame table collision plane z. Default uses detected/workspace table_z; fallback is -0.06.",
    )
    parser.add_argument(
        "--tool-frame-table-clearance",
        type=float,
        default=0.08,
        help="Require tool_frame z to stay at least table_z + this clearance during planned motions.",
    )
    parser.add_argument(
        "--gripper-table-clearance",
        type=float,
        default=0.015,
        help="Require approximated open-gripper collision points to stay at least table_z + this clearance.",
    )
    parser.add_argument(
        "--gripper-forward-axis",
        choices=["x", "y", "z", "-x", "-y", "-z"],
        default="z",
        help="Tool-frame axis from tool_frame toward the gripper fingertips. Use -z if your model points the other way.",
    )
    parser.add_argument(
        "--gripper-finger-length",
        type=float,
        default=0.115,
        help="Approximate distance from tool_frame to the open fingertip collision points, in meters.",
    )
    parser.add_argument(
        "--gripper-open-half-width",
        type=float,
        default=0.055,
        help="Approximate half distance between the two open fingertips, in meters.",
    )
    parser.add_argument(
        "--gripper-side-axis",
        choices=["x", "y", "z", "-x", "-y", "-z"],
        default="y",
        help="Tool-frame axis across the two open fingertips.",
    )
    parser.add_argument(
        "--disable-gripper-table-collision-check",
        action="store_true",
        help="Only check tool_frame height, not approximate open-gripper fingertip points.",
    )
    parser.add_argument(
        "--path-check-samples",
        type=int,
        default=16,
        help="Joint-space interpolation samples per segment for table collision checking.",
    )
    parser.add_argument("--disable-table-collision-check", action="store_true")
    parser.add_argument("--dt", type=float, default=0.05)
    parser.add_argument("--iterations", type=int, default=250)
    parser.add_argument("--position-cost", type=float, default=70.0)
    parser.add_argument("--orientation-cost", type=float, default=4.0)
    parser.add_argument("--posture-cost", type=float, default=0.08)
    parser.add_argument("--position-tolerance", type=float, default=0.003)
    parser.add_argument("--orientation-tolerance", type=float, default=0.08)
    parser.add_argument("--solver", default=None)
    parser.add_argument("--list-frames", action="store_true")
    parser.add_argument("--no-wrap-current-joints", action="store_true")
    parser.add_argument("--clamp-current-to-limits", action="store_true", default=True)
    parser.add_argument("--no-clamp-current-to-limits", dest="clamp_current_to_limits", action="store_false")
    parser.add_argument("--limit-margin-rad", type=float, default=1e-4)
    parser.set_defaults(current_joints_deg=None)
    parser.add_argument("--max-joint-delta", type=float, default=180)
    parser.add_argument("--duration", type=float, default=8.0)
    parser.add_argument("--timeout", type=float, default=35.0)
    parser.add_argument("--open-value", type=float, default=0.0)
    parser.add_argument("--close-value", type=float, default=0.9)
    parser.add_argument("--gripper-step", type=float, default=0.05)
    parser.add_argument("--gripper-settle-time", type=float, default=0.2)
    parser.add_argument("--yes", action="store_true")
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


def vertical_down_rotation_from_closing_direction(direction_xy, closing_axis="x"):
    direction = np.asarray([direction_xy[0], direction_xy[1], 0.0], dtype=np.float64)
    norm = float(np.linalg.norm(direction[:2]))
    if norm < 1e-9:
        raise ValueError("Closing direction has near-zero XY norm.")
    closing_world = direction / norm
    z_world = np.array([0.0, 0.0, -1.0], dtype=np.float64)

    if closing_axis == "x":
        x_world = closing_world
        y_world = np.cross(z_world, x_world)
        y_world /= max(float(np.linalg.norm(y_world)), 1e-9)
    else:
        y_world = closing_world
        x_world = np.cross(y_world, z_world)
        x_world /= max(float(np.linalg.norm(x_world)), 1e-9)

    return np.column_stack([x_world, y_world, z_world])


def rotation_matrix_to_rpy_deg(R):
    # Matches the common R = Rz(yaw) * Ry(pitch) * Rx(roll) convention.
    sy = math.sqrt(float(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0]))
    singular = sy < 1e-9
    if not singular:
        roll = math.atan2(float(R[2, 1]), float(R[2, 2]))
        pitch = math.atan2(float(-R[2, 0]), sy)
        yaw = math.atan2(float(R[1, 0]), float(R[0, 0]))
    else:
        roll = math.atan2(float(-R[1, 2]), float(R[1, 1]))
        pitch = math.atan2(float(-R[2, 0]), sy)
        yaw = 0.0
    return [math.degrees(roll), math.degrees(pitch), math.degrees(yaw)]


def rpy_deg_to_rotation_matrix(rpy_deg):
    roll, pitch, yaw = np.radians(np.asarray(rpy_deg, dtype=np.float64))
    cr, sr = math.cos(float(roll)), math.sin(float(roll))
    cp, sp = math.cos(float(pitch)), math.sin(float(pitch))
    cy, sy = math.cos(float(yaw)), math.sin(float(yaw))
    rx = np.array([[1.0, 0.0, 0.0], [0.0, cr, -sr], [0.0, sr, cr]], dtype=np.float64)
    ry = np.array([[cp, 0.0, sp], [0.0, 1.0, 0.0], [-sp, 0.0, cp]], dtype=np.float64)
    rz = np.array([[cy, -sy, 0.0], [sy, cy, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)
    return rz @ ry @ rx


def rotation_matrix_about_z(angle_rad):
    c = math.cos(float(angle_rad))
    s = math.sin(float(angle_rad))
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)


def rotation_matrix_about_axis(axis, angle_rad):
    axis = np.asarray(axis, dtype=np.float64).reshape(3)
    axis = axis / max(float(np.linalg.norm(axis)), 1e-9)
    x, y, z = axis
    c = math.cos(float(angle_rad))
    s = math.sin(float(angle_rad))
    C = 1.0 - c
    return np.array(
        [
            [c + x * x * C, x * y * C - z * s, x * z * C + y * s],
            [y * x * C + z * s, c + y * y * C, y * z * C - x * s],
            [z * x * C - y * s, z * y * C + x * s, c + z * z * C],
        ],
        dtype=np.float64,
    )


def rotation_from_tool_y_axis(tool_y_base, roll_rad=0.0):
    y_axis = np.asarray(tool_y_base, dtype=np.float64).reshape(3)
    y_axis = y_axis / max(float(np.linalg.norm(y_axis)), 1e-9)
    helper = np.array([0.0, 0.0, -1.0], dtype=np.float64)
    if abs(float(np.dot(helper, y_axis))) > 0.92:
        helper = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    x_axis = np.cross(y_axis, helper)
    x_axis /= max(float(np.linalg.norm(x_axis)), 1e-9)
    z_axis = np.cross(x_axis, y_axis)
    z_axis /= max(float(np.linalg.norm(z_axis)), 1e-9)
    R0 = np.column_stack([x_axis, y_axis, z_axis])
    return rotation_matrix_about_axis(y_axis, roll_rad) @ R0


def invert_transform(T):
    T = np.asarray(T, dtype=np.float64).reshape(4, 4)
    out = np.eye(4, dtype=np.float64)
    out[:3, :3] = T[:3, :3].T
    out[:3, 3] = -out[:3, :3] @ T[:3, 3]
    return out


def print_transform(name, T):
    T = np.asarray(T, dtype=np.float64).reshape(4, 4)
    print(f"\n{name}:")
    for row in T:
        print("  " + " ".join(f"{float(v): .6f}" for v in row))


def tool_z_down_tilt_deg(rotation_matrix):
    R = np.asarray(rotation_matrix, dtype=np.float64).reshape(3, 3)
    tool_z_base = R[:, 2]
    down = np.array([0.0, 0.0, -1.0], dtype=np.float64)
    cos_angle = float(np.dot(tool_z_base, down) / max(float(np.linalg.norm(tool_z_base)), 1e-9))
    cos_angle = max(-1.0, min(1.0, cos_angle))
    return math.degrees(math.acos(cos_angle))


def enforce_lying_vertical_down_candidates(args, selected, candidates):
    state = str(selected.get("state", ""))
    if state != "lying" or not bool(getattr(args, "force_lying_vertical_down", True)):
        return candidates

    max_tilt = float(getattr(args, "max_lying_tool_z_tilt_deg", 8.0))
    kept = []
    rejected = []
    for candidate in candidates:
        tilt = tool_z_down_tilt_deg(candidate["rotation_matrix"])
        candidate["tool_z_down_tilt_deg"] = tilt
        if tilt <= max_tilt:
            kept.append(candidate)
        else:
            rejected.append((candidate["name"], tilt))

    if rejected:
        print("\nRejected lying-cup grasp candidates that are not vertical-down enough:")
        for name, tilt in rejected:
            print(f"  {name}: tool_z_down_tilt={tilt:.3f} deg > {max_tilt:.3f} deg")
    if not kept:
        raise RuntimeError(
            "No lying-cup grasp candidate satisfies vertical-down approach. "
            "Increase --max-lying-tool-z-tilt-deg only if the gripper is safely above the cup."
        )
    return kept


def choose_tool_orientation_candidates(args, selected):
    state = str(selected.get("state", ""))
    force_lying_vertical = (
        bool(getattr(args, "force_lying_vertical_down", True))
        and state == "lying"
    )

    if args.grasp_frame_mode == "mesh_y_aligned" and not force_lying_vertical:
        T_base_model = np.asarray(selected["T_base_model"], dtype=np.float64).reshape(4, 4)
        tool_y_base = T_base_model[:3, :3] @ np.array([0.0, 1.0, 0.0], dtype=np.float64)
        tool_y_base /= max(float(np.linalg.norm(tool_y_base)), 1e-9)
        samples = max(1, int(args.mesh_y_roll_samples))
        candidates = []
        for idx in range(samples):
            roll_rad = 2.0 * math.pi * float(idx) / float(samples)
            R = rotation_from_tool_y_axis(tool_y_base, roll_rad)
            rpy = rotation_matrix_to_rpy_deg(R)
            candidates.append(
                {
                    "name": f"mesh_y_aligned_roll_{math.degrees(roll_rad):.1f}",
                    "rotation_matrix": R,
                    "rpy_deg": rpy,
                    "tool_y_base": tool_y_base.copy(),
                    "roll_about_y_deg": math.degrees(roll_rad),
                }
            )
        print("\nMesh-Y aligned grasp candidates:")
        print(f"  mesh_y_base=[{tool_y_base[0]:.4f}, {tool_y_base[1]:.4f}, {tool_y_base[2]:.4f}]")
        for item in candidates:
            rpy = item["rpy_deg"]
            print(f"  {item['name']}: rpy=[{rpy[0]:.3f}, {rpy[1]:.3f}, {rpy[2]:.3f}]")
        return enforce_lying_vertical_down_candidates(args, selected, candidates)

    axis = np.asarray(selected.get("axis_report_base", [0.0, 0.0, 0.0]), dtype=np.float64)
    axis_xy = axis[:2]
    axis_xy_norm = float(np.linalg.norm(axis_xy))

    if args.grasp_frame_mode == "mesh_y_aligned" and force_lying_vertical:
        print("\nLying cup detected: ignoring mesh_y_aligned grasp orientation and forcing vertical-down approach.")

    if args.lying_align_gripper and state == "lying" and axis_xy_norm >= float(args.lying_axis_min_xy_norm):
        cup_axis_xy = axis_xy / axis_xy_norm
        base_closing_direction_xy = np.array([-cup_axis_xy[1], cup_axis_xy[0]], dtype=np.float64)
        candidates = []
        closing_axes = ["x", "y"] if args.search_both_gripper_axes else [args.gripper_closing_axis]
        offsets = [float(v) for v in args.grasp_yaw_offsets_deg]
        for closing_axis in closing_axes:
            for sign in (1.0, -1.0):
                signed_base = sign * base_closing_direction_xy
                for offset_deg in offsets:
                    R_offset = rotation_matrix_about_z(math.radians(offset_deg))
                    closing_direction_xy = (R_offset @ np.array([signed_base[0], signed_base[1], 0.0]))[:2]
                    R = vertical_down_rotation_from_closing_direction(closing_direction_xy, closing_axis)
                    rpy = rotation_matrix_to_rpy_deg(R)
                    dot = float(abs(np.dot(cup_axis_xy, closing_direction_xy / max(float(np.linalg.norm(closing_direction_xy)), 1e-9))))
                    candidates.append(
                        {
                            "name": (
                                f"lying_axis_{closing_axis}_{'plus' if sign > 0 else 'minus'}_"
                                f"offset_{offset_deg:+.1f}"
                            ),
                            "rotation_matrix": R,
                            "rpy_deg": rpy,
                            "closing_direction_xy": closing_direction_xy,
                            "dot": dot,
                            "grasp_offset_deg": float(offset_deg),
                            "gripper_closing_axis": closing_axis,
                        }
                    )
        print("\nLying-cup gripper yaw alignment:")
        print(f"  cup_axis_xy=[{cup_axis_xy[0]:.4f}, {cup_axis_xy[1]:.4f}]")
        print(f"  gripper_closing_axis candidates={closing_axes}")
        for item in candidates:
            direction = item["closing_direction_xy"]
            rpy = item["rpy_deg"]
            print(
                f"  {item['name']}: closing_direction_xy=[{direction[0]:.4f}, {direction[1]:.4f}], "
                f"abs(dot)={item['dot']:.6f}, rpy=[{rpy[0]:.3f}, {rpy[1]:.3f}, {rpy[2]:.3f}]"
            )
        return enforce_lying_vertical_down_candidates(args, selected, candidates)

    print("\nUsing fixed tool orientation:")
    print(f"  state={state}, lying_align_gripper={args.lying_align_gripper}, axis_xy_norm={axis_xy_norm:.4f}")
    candidates = [
        {
            "name": "fixed_tool_rpy",
            "rotation_matrix": rpy_deg_to_rotation_matrix(args.tool_rpy_deg),
            "rpy_deg": [float(v) for v in args.tool_rpy_deg],
            "closing_direction_xy": None,
            "dot": None,
        }
    ]
    return enforce_lying_vertical_down_candidates(args, selected, candidates)


def make_ik_args(args, target_position, target_rotation_matrix=None, target_rpy_deg=None):
    ns = argparse.Namespace(**vars(args))
    ns.target_position = [float(v) for v in target_position]
    ns.target_offset = [0.0, 0.0, 0.0]
    ns.target_rotation_matrix = target_rotation_matrix
    ns.target_rpy_deg = [float(v) for v in (target_rpy_deg if target_rpy_deg is not None else args.tool_rpy_deg)]
    return ns


def solve_waypoint(
    args,
    current_joints_deg,
    name,
    target_position,
    target_rotation_matrix=None,
    target_rpy_deg=None,
    enforce_joint_delta=True,
):
    print(f"\n========== Pink IK waypoint: {name} ==========")
    ik_args = make_ik_args(args, target_position, target_rotation_matrix, target_rpy_deg)
    result = solve_pink_ik(ik_args, current_joints_deg=current_joints_deg)
    if result is None or not result["converged"]:
        raise RuntimeError(f"Pink IK failed for {name}.")
    if result["position_error_m"] > args.position_tolerance * 2.0:
        raise RuntimeError(f"Pink IK position error too high for {name}: {result['position_error_m']:.6f} m")
    delta = shortest_delta_deg(result["solved_joints_deg"], current_joints_deg)
    max_delta = max(abs(v) for v in delta)
    print(f"Waypoint {name} max joint delta from current command seed: {max_delta:.3f} deg")
    result["max_joint_delta_from_seed"] = float(max_delta)
    if enforce_joint_delta and max_delta > args.max_joint_delta:
        raise RuntimeError(f"Waypoint {name} exceeds --max-joint-delta={args.max_joint_delta:.3f} deg")
    return result


def solve_pregrasp_with_orientation_candidates(args, current_joints_deg, pre_grasp, orientation_candidates):
    attempts = []
    for candidate in orientation_candidates:
        print(f"\nTrying orientation candidate: {candidate['name']}")
        try:
            result = solve_waypoint(
                args,
                current_joints_deg,
                "pre_grasp",
                pre_grasp,
                target_rotation_matrix=candidate["rotation_matrix"],
                target_rpy_deg=candidate["rpy_deg"],
                enforce_joint_delta=False,
            )
            attempts.append((candidate, result, None))
        except Exception as exc:
            attempts.append((candidate, None, exc))
            print(f"Orientation candidate failed: {candidate['name']}: {exc}")

    valid = [(candidate, result) for candidate, result, exc in attempts if result is not None]
    if not valid:
        errors = "\n".join(f"  {candidate['name']}: {exc}" for candidate, result, exc in attempts)
        raise RuntimeError("All pre-grasp orientation candidates failed:\n" + errors)

    valid.sort(key=lambda item: float(item[1].get("max_joint_delta_from_seed", float("inf"))))
    best_candidate, best_result = valid[0]
    best_delta = float(best_result["max_joint_delta_from_seed"])
    print("\nSelected orientation candidate:")
    print(f"  name={best_candidate['name']}")
    print("  target_tool_rpy_deg=[" + ", ".join(f"{v:.3f}" for v in best_candidate["rpy_deg"]) + "]")
    print(f"  pre_grasp_max_joint_delta={best_delta:.3f} deg")
    if best_delta > args.max_joint_delta:
        raise RuntimeError(
            f"Best pre_grasp orientation still exceeds --max-joint-delta={args.max_joint_delta:.3f} deg "
            f"(best={best_delta:.3f} deg)."
        )
    return best_candidate, best_result


def detect_one_cup(args):
    models = load_mesh_models(args)
    capture_info = load_offline_pcds(args) if args.input_pcd else capture_cluster_pcds(args)
    candidates = capture_info["candidates"]
    cluster_pcds = capture_info["cluster_pcds"]
    if not candidates:
        raise RuntimeError("No cup candidates found.")

    selected = None
    for candidate in candidates:
        if args.target_label is not None and int(candidate["label"]) != int(args.target_label):
            continue
        label = int(candidate["label"])
        fit = fit_cluster_with_meshes(cluster_pcds[label]["clean"], models, args, table_z=capture_info.get("table_z"))
        if fit is None:
            continue
        fit["label"] = label
        fit["candidate"] = candidate
        if selected is None or fit["score"] > selected["score"]:
            selected = fit
        if args.target_label is not None:
            break
    if selected is None:
        raise RuntimeError("No selected cup has a valid mesh fit.")
    center = selected["center_base"]
    print("\nSelected cup:")
    print(
        f"  label={selected['label']:02d}, model={selected['model_name']}, state={selected['state']}, "
        f"score={selected['score']:.4f}, center=[{center[0]:.4f}, {center[1]:.4f}, {center[2]:.4f}]"
    )
    if "top_center_base" in selected and "bottom_center_base" in selected:
        top = selected["top_center_base"]
        bottom = selected["bottom_center_base"]
        signed_axis = selected["cup_axis_signed_base"]
        print(
            f"  top_center=[{top[0]:.4f}, {top[1]:.4f}, {top[2]:.4f}], "
            f"bottom_center=[{bottom[0]:.4f}, {bottom[1]:.4f}, {bottom[2]:.4f}]"
        )
        print(
            f"  cup_axis_signed=[{signed_axis[0]:.4f}, {signed_axis[1]:.4f}, {signed_axis[2]:.4f}], "
            f"signed_orientation={selected.get('signed_orientation')}"
        )
    selected["table_z"] = capture_info.get("table_z")
    return selected


def choose_grasp_position(args, selected):
    if args.grasp_frame_mode == "mesh_y_aligned":
        origin = np.asarray(selected["origin_base"], dtype=np.float64)
        grasp = [float(origin[0]), float(origin[1]), float(origin[2])]
        z_label = "mesh_origin_z"
        raw_z = float(origin[2])
    else:
        center = np.asarray(selected["center_base"], dtype=np.float64)
        grasp = [float(center[0]), float(center[1]), float(center[2] + float(args.grasp_z_offset))]
        z_label = "center_z+offset"
        raw_z = float(center[2])
    print(
        f"\nGrasp frame mode: {args.grasp_frame_mode}, "
        f"target tool origin=[{grasp[0]:.4f}, {grasp[1]:.4f}, {grasp[2]:.4f}]"
    )
    table_z = selected.get("table_z")
    if args.collision_table_z is not None:
        table_z = float(args.collision_table_z)
    if table_z is not None:
        clearance = max(float(args.min_grasp_z_above_table), float(args.tool_frame_table_clearance))
        min_safe_z = float(table_z) + clearance
        if grasp[2] < min_safe_z:
            print("\nGrasp z safety clamp:")
            print(f"  raw_z={raw_z:.4f}")
            print(f"  requested_z={z_label}={grasp[2]:.4f}")
            print(f"  table_z={float(table_z):.4f}, clearance={clearance:.4f}, min_safe_z={min_safe_z:.4f}")
            grasp[2] = min_safe_z
    return grasp


def workspace_center_and_table_z(args, workspace_name):
    workspaces, metadata = load_workspaces(args.workspaces)
    if workspace_name not in workspaces:
        raise KeyError(f"Workspace '{workspace_name}' not found in {args.workspaces}")
    workspace = workspaces[workspace_name]
    polygon = np.asarray(workspace["polygon_base_xy"], dtype=np.float64).reshape(-1, 2)
    center_xy = np.mean(polygon, axis=0)
    table_z = workspace.get("table_z_mean")
    if table_z is None:
        corners = np.asarray(workspace.get("corner_points_base_xyz", []), dtype=np.float64)
        if corners.size == 0:
            raise KeyError(f"Workspace '{workspace_name}' has no table_z_mean or corner_points_base_xyz.")
        table_z = float(np.mean(corners[:, 2]))
    return center_xy, float(table_z), metadata


def make_cup_place_pose(args, selected, yaw_rad=0.0, verbose=True):
    model = selected["model"]
    place_xy, place_table_z, _ = workspace_center_and_table_z(args, args.place_workspace_name)
    R_base_cup_place = rotation_matrix_about_z(yaw_rad) @ rotation_align_vectors(model.axis_local, BASE_Z)
    desired_bottom_center = np.array(
        [
            float(place_xy[0]),
            float(place_xy[1]),
            float(place_table_z) + float(args.place_bottom_clearance),
        ],
        dtype=np.float64,
    )
    t_base_cup_place = desired_bottom_center - R_base_cup_place @ model.bottom_center_local
    T_base_cup_place = make_transform(R_base_cup_place, t_base_cup_place)
    top_center_place = t_base_cup_place + R_base_cup_place @ model.top_center_local
    bottom_center_place = t_base_cup_place + R_base_cup_place @ model.bottom_center_local
    cup_axis_place = top_center_place - bottom_center_place
    cup_axis_place /= max(float(np.linalg.norm(cup_axis_place)), 1e-9)

    if verbose:
        print("\nCup place target:")
        print(f"  workspace={args.place_workspace_name}, center_xy=[{place_xy[0]:.4f}, {place_xy[1]:.4f}]")
        print(f"  place_yaw_deg={math.degrees(float(yaw_rad)):.3f}")
        print(f"  place_table_z={place_table_z:.4f}, place_bottom_clearance={args.place_bottom_clearance:.4f}")
        print(
            f"  bottom_center_place=[{bottom_center_place[0]:.4f}, {bottom_center_place[1]:.4f}, {bottom_center_place[2]:.4f}]"
        )
        print(f"  top_center_place=[{top_center_place[0]:.4f}, {top_center_place[1]:.4f}, {top_center_place[2]:.4f}]")
        print(f"  cup_axis_place=[{cup_axis_place[0]:.4f}, {cup_axis_place[1]:.4f}, {cup_axis_place[2]:.4f}]")
    return T_base_cup_place


def plan_place_tool_poses(
    args,
    selected,
    grasp,
    lift,
    target_rotation_matrix,
    place_yaw_rad=0.0,
    stage_position=None,
    stage_name=None,
    verbose=True,
):
    R_base_tool_grasp = np.asarray(target_rotation_matrix, dtype=np.float64).reshape(3, 3)
    T_base_tool_grasp = make_transform(R_base_tool_grasp, np.asarray(grasp, dtype=np.float64))
    T_base_cup_grasp = np.asarray(selected["T_base_model"], dtype=np.float64).reshape(4, 4)
    T_tool_cup = invert_transform(T_base_tool_grasp) @ T_base_cup_grasp
    place_xy, _, _ = workspace_center_and_table_z(args, args.place_workspace_name)
    T_base_cup_place = make_cup_place_pose(args, selected, yaw_rad=place_yaw_rad, verbose=verbose)
    T_base_tool_place = T_base_cup_place @ invert_transform(T_tool_cup)

    stage_xy = np.asarray(lift[:2], dtype=np.float64)
    stage_z = max(float(args.rotate_stage_z), float(lift[2]))
    if args.rotate_stage_xy == "place":
        stage_xy = np.asarray(place_xy, dtype=np.float64)
    if stage_position is not None:
        stage_position = np.asarray(stage_position, dtype=np.float64).reshape(3)
        stage_xy = stage_position[:2]
        stage_z = float(stage_position[2])
    pre_rotate = make_transform(
        R_base_tool_grasp,
        np.array([float(stage_xy[0]), float(stage_xy[1]), stage_z], dtype=np.float64),
    )
    pre_place = T_base_tool_place.copy()
    pre_place[:3, 3] += np.array([0.0, 0.0, float(args.pre_place_height)], dtype=np.float64)
    rotate_upright = T_base_tool_place.copy()
    rotate_upright[:3, 3] = pre_rotate[:3, 3]
    retreat = T_base_tool_place.copy()
    retreat[:3, 3] += np.array([0.0, 0.0, float(args.retreat_height)], dtype=np.float64)

    if verbose:
        print("\nRotate staging:")
        print(f"  rotate_stage_xy={stage_name or args.rotate_stage_xy}, stage_z={stage_z:.4f}")
        print_transform("T_base_tool_grasp", T_base_tool_grasp)
        print_transform("T_base_cup_grasp", T_base_cup_grasp)
        print_transform("T_tool_cup", T_tool_cup)
        print_transform("T_base_cup_place", T_base_cup_place)
        print_transform("T_base_tool_pre_rotate", pre_rotate)
        print_transform("T_base_tool_place", T_base_tool_place)
    return {
        "T_base_tool_grasp": T_base_tool_grasp,
        "T_base_cup_grasp": T_base_cup_grasp,
        "T_tool_cup": T_tool_cup,
        "T_base_cup_place": T_base_cup_place,
        "T_base_tool_place": T_base_tool_place,
        "stage_name": stage_name or args.rotate_stage_xy,
        "stage_position": pre_rotate[:3, 3].copy(),
        "pre_rotate": pre_rotate,
        "rotate_upright": rotate_upright,
        "pre_place": pre_place,
        "place": T_base_tool_place,
        "retreat": retreat,
    }


def solve_transform_waypoint(args, current_joints_deg, name, T, enforce_joint_delta=True):
    T = np.asarray(T, dtype=np.float64).reshape(4, 4)
    rpy = rotation_matrix_to_rpy_deg(T[:3, :3])
    return solve_waypoint(
        args,
        current_joints_deg,
        name,
        T[:3, 3],
        target_rotation_matrix=T[:3, :3],
        target_rpy_deg=rpy,
        enforce_joint_delta=enforce_joint_delta,
    )


def rotate_stage_candidates(args, lift, place_xy):
    lift = np.asarray(lift, dtype=np.float64).reshape(3)
    place_xy = np.asarray(place_xy, dtype=np.float64).reshape(2)
    midpoint_xy = 0.5 * (lift[:2] + place_xy)
    xy_candidates = [
        ("lift_xy", lift[:2]),
        ("mid_xy", midpoint_xy),
        ("place_xy", place_xy),
    ]
    base_z = max(float(args.rotate_stage_z), float(lift[2]))
    count = max(1, int(args.rotate_stage_z_samples))
    step = max(0.0, float(args.rotate_stage_z_step))
    out = []
    for name, xy in xy_candidates:
        for idx in range(count):
            z = base_z + step * float(idx)
            out.append((f"{name}_z{idx}", np.array([float(xy[0]), float(xy[1]), z], dtype=np.float64)))
    return out


def solve_place_sequence_with_yaw_candidates(args, selected, grasp, lift, target_rotation_matrix, lift_result):
    samples = max(1, int(args.place_yaw_samples))
    place_xy, _, _ = workspace_center_and_table_z(args, args.place_workspace_name)
    stage_candidates = rotate_stage_candidates(args, lift, place_xy)
    attempts = []
    for stage_name, stage_position in stage_candidates:
        print(
            f"\nTrying rotate stage candidate: {stage_name}, "
            f"position=[{stage_position[0]:.4f}, {stage_position[1]:.4f}, {stage_position[2]:.4f}]"
        )
        pre_rotate_result = None
        try:
            pre_rotate_plan = plan_place_tool_poses(
                args,
                selected,
                grasp,
                lift,
                target_rotation_matrix,
                place_yaw_rad=0.0,
                stage_position=stage_position,
                stage_name=stage_name,
                verbose=False,
            )
            pre_rotate_result = solve_transform_waypoint(
                args,
                lift_result["solved_joints_deg"],
                "pre_rotate",
                pre_rotate_plan["pre_rotate"],
                enforce_joint_delta=False,
            )
        except Exception as exc:
            attempts.append({"stage_name": stage_name, "stage_position": stage_position, "yaw_deg": None, "error": exc})
            print(f"Rotate stage candidate {stage_name} failed at pre_rotate: {exc}")
            continue

        for idx in range(samples):
            yaw_rad = 2.0 * math.pi * float(idx) / float(samples)
            yaw_deg = math.degrees(yaw_rad)
            print(f"\nTrying place yaw candidate {idx + 1}/{samples}: stage={stage_name}, yaw={yaw_deg:.1f} deg")
            try:
                place_plan = plan_place_tool_poses(
                    args,
                    selected,
                    grasp,
                    lift,
                    target_rotation_matrix,
                    place_yaw_rad=yaw_rad,
                    stage_position=stage_position,
                    stage_name=stage_name,
                    verbose=False,
                )
                rotate_result = solve_transform_waypoint(
                    args,
                    pre_rotate_result["solved_joints_deg"],
                    "rotate_upright",
                    place_plan["rotate_upright"],
                    enforce_joint_delta=False,
                )
                pre_place_result = solve_transform_waypoint(
                    args,
                    rotate_result["solved_joints_deg"],
                    "pre_place",
                    place_plan["pre_place"],
                    enforce_joint_delta=False,
                )
                place_result = solve_transform_waypoint(
                    args,
                    pre_place_result["solved_joints_deg"],
                    "place",
                    place_plan["place"],
                    enforce_joint_delta=False,
                )
                retreat_result = solve_transform_waypoint(
                    args,
                    place_result["solved_joints_deg"],
                    "retreat",
                    place_plan["retreat"],
                    enforce_joint_delta=False,
                )
                max_delta = max(
                    float(pre_rotate_result["max_joint_delta_from_seed"]),
                    float(rotate_result["max_joint_delta_from_seed"]),
                    float(pre_place_result["max_joint_delta_from_seed"]),
                    float(place_result["max_joint_delta_from_seed"]),
                    float(retreat_result["max_joint_delta_from_seed"]),
                )
                attempts.append(
                    {
                        "stage_name": stage_name,
                        "stage_position": stage_position,
                        "yaw_rad": yaw_rad,
                        "yaw_deg": yaw_deg,
                        "place_plan": place_plan,
                        "pre_rotate_result": pre_rotate_result,
                        "rotate_result": rotate_result,
                        "pre_place_result": pre_place_result,
                        "place_result": place_result,
                        "retreat_result": retreat_result,
                        "max_delta": max_delta,
                        "error": None,
                    }
                )
                print(
                    f"Place candidate solved: stage={stage_name}, yaw={yaw_deg:.1f} deg, "
                    f"sequence max delta={max_delta:.3f} deg"
                )
            except Exception as exc:
                attempts.append(
                    {
                        "stage_name": stage_name,
                        "stage_position": stage_position,
                        "yaw_rad": yaw_rad,
                        "yaw_deg": yaw_deg,
                        "error": exc,
                    }
                )
                print(f"Place candidate failed: stage={stage_name}, yaw={yaw_deg:.1f} deg: {exc}")

    valid = [item for item in attempts if item.get("error") is None]
    if not valid:
        def format_attempt(item):
            yaw = item.get("yaw_deg")
            yaw_text = "pre_rotate" if yaw is None else f"yaw={yaw:.1f}"
            return f"  stage={item.get('stage_name')}, {yaw_text}: {item['error']}"

        errors = "\n".join(format_attempt(item) for item in attempts)
        raise RuntimeError("All rotate stage / place yaw candidates failed:\n" + errors)

    valid.sort(key=lambda item: float(item["max_delta"]))
    best = valid[0]
    if float(best["max_delta"]) > float(args.max_joint_delta):
        raise RuntimeError(
            f"Best place yaw still exceeds --max-joint-delta={args.max_joint_delta:.3f} deg "
            f"(best yaw={best['yaw_deg']:.1f}, max delta={best['max_delta']:.3f} deg)."
        )

    print("\nSelected place yaw candidate:")
    print(f"  stage={best['stage_name']}")
    pos = best["stage_position"]
    print(f"  stage_position=[{pos[0]:.4f}, {pos[1]:.4f}, {pos[2]:.4f}]")
    print(f"  yaw_deg={best['yaw_deg']:.3f}")
    print(f"  sequence_max_joint_delta={best['max_delta']:.3f} deg")
    plan_place_tool_poses(
        args,
        selected,
        grasp,
        lift,
        target_rotation_matrix,
        place_yaw_rad=best["yaw_rad"],
        stage_position=best["stage_position"],
        stage_name=best["stage_name"],
        verbose=True,
    )
    return (
        best["place_plan"],
        best["pre_rotate_result"],
        best["rotate_result"],
        best["pre_place_result"],
        best["place_result"],
        best["retreat_result"],
    )


_FK_CACHE = {}


def resolve_collision_table_z(args, selected):
    if args.collision_table_z is not None:
        return float(args.collision_table_z)
    if selected.get("table_z") is not None:
        return float(selected["table_z"])
    try:
        _, table_z, _ = workspace_center_and_table_z(args, args.workspace_name)
        return float(table_z)
    except Exception:
        return -0.06


def axis_vector(axis_name):
    sign = -1.0 if axis_name.startswith("-") else 1.0
    axis = axis_name[-1]
    if axis == "x":
        vec = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    elif axis == "y":
        vec = np.array([0.0, 1.0, 0.0], dtype=np.float64)
    elif axis == "z":
        vec = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    else:
        raise ValueError(f"Unsupported axis name: {axis_name}")
    return sign * vec


def approximate_gripper_collision_points_tool(args):
    if getattr(args, "disable_gripper_table_collision_check", False):
        return []
    forward = axis_vector(getattr(args, "gripper_forward_axis", "z"))
    side = axis_vector(getattr(args, "gripper_side_axis", "y"))
    length = float(getattr(args, "gripper_finger_length", 0.115))
    half_width = float(getattr(args, "gripper_open_half_width", 0.055))

    points = [
        np.zeros(3, dtype=np.float64),
        forward * length,
        forward * length + side * half_width,
        forward * length - side * half_width,
        forward * (0.5 * length) + side * half_width,
        forward * (0.5 * length) - side * half_width,
    ]
    return points


def fk_tool_transform_for_joints(args, joints_deg):
    urdf = args.urdf if args.urdf.is_absolute() else PROJECT_ROOT / args.urdf
    key = str(urdf.resolve())
    cached = _FK_CACHE.get(key)
    if cached is None:
        pin, *_ = require_pink_pinocchio()
        urdf_for_pin = make_ascii_urdf_copy(urdf) if not is_ascii_path(urdf) else urdf
        model = pin.buildModelFromUrdf(str(urdf_for_pin))
        data = model.createData()
        cached = {"pin": pin, "model": model, "data": data, "urdf_for_pin": urdf_for_pin, "urdf": urdf}
        _FK_CACHE[key] = cached
    pin = cached["pin"]
    model = cached["model"]
    data = cached["data"]
    q = pin.neutral(model)
    q = set_arm_joints_deg(model, q, joints_deg)
    pin.forwardKinematics(model, data, q)
    pin.updateFramePlacements(model, data)
    frame_id = model.getFrameId(args.frame)
    return np.array(data.oMf[frame_id].homogeneous, dtype=np.float64)


def fk_tool_z_for_joints(args, joints_deg):
    return float(fk_tool_transform_for_joints(args, joints_deg)[2, 3])


def interpolate_joints_deg(start, end, samples):
    start = np.asarray(start, dtype=np.float64)
    end = np.asarray(end, dtype=np.float64)
    delta = np.asarray([((e - s + 180.0) % 360.0) - 180.0 for s, e in zip(start, end)], dtype=np.float64)
    for idx in range(max(2, int(samples)) + 1):
        t = float(idx) / float(max(2, int(samples)))
        yield (start + t * delta).tolist()


def validate_table_clearance_for_sequence(args, selected, joint_sequence, sequence_name):
    if args.disable_table_collision_check:
        return
    table_z = resolve_collision_table_z(args, selected)
    min_tool_allowed_z = table_z + float(args.tool_frame_table_clearance)
    min_gripper_allowed_z = table_z + float(getattr(args, "gripper_table_clearance", 0.015))
    gripper_points_tool = approximate_gripper_collision_points_tool(args)
    min_seen_tool_z = float("inf")
    min_seen_gripper_z = float("inf")
    for seg_idx in range(len(joint_sequence) - 1):
        start = joint_sequence[seg_idx]
        end = joint_sequence[seg_idx + 1]
        for sample_idx, joints in enumerate(interpolate_joints_deg(start, end, args.path_check_samples)):
            T_base_tool = fk_tool_transform_for_joints(args, joints)
            tool_z = float(T_base_tool[2, 3])
            if tool_z < min_seen_tool_z:
                min_seen_tool_z = tool_z
            if tool_z < min_tool_allowed_z:
                raise RuntimeError(
                    f"Table collision check failed for {sequence_name}: tool_frame z={tool_z:.4f} m "
                    f"< table_z+clearance={min_tool_allowed_z:.4f} m "
                    f"(table_z={table_z:.4f}, clearance={args.tool_frame_table_clearance:.4f}, "
                    f"segment={seg_idx}, sample={sample_idx})."
                )
            for point_idx, point_tool in enumerate(gripper_points_tool):
                point_base = T_base_tool[:3, :3] @ point_tool + T_base_tool[:3, 3]
                point_z = float(point_base[2])
                if point_z < min_seen_gripper_z:
                    min_seen_gripper_z = point_z
                if point_z < min_gripper_allowed_z:
                    raise RuntimeError(
                        f"Table collision check failed for {sequence_name}: approximate gripper point "
                        f"{point_idx} z={point_z:.4f} m < table_z+clearance={min_gripper_allowed_z:.4f} m "
                        f"(table_z={table_z:.4f}, gripper_clearance={getattr(args, 'gripper_table_clearance', 0.015):.4f}, "
                        f"forward_axis={getattr(args, 'gripper_forward_axis', 'z')}, "
                        f"finger_length={getattr(args, 'gripper_finger_length', 0.115):.4f}, "
                        f"open_half_width={getattr(args, 'gripper_open_half_width', 0.055):.4f}, "
                        f"segment={seg_idx}, sample={sample_idx})."
                    )
    if gripper_points_tool:
        gripper_text = f", min approximate gripper z={min_seen_gripper_z:.4f} m, gripper limit={min_gripper_allowed_z:.4f} m"
    else:
        gripper_text = ""
    print(
        f"Table collision check passed for {sequence_name}: "
        f"min tool_frame z={min_seen_tool_z:.4f} m, tool limit={min_tool_allowed_z:.4f} m"
        f"{gripper_text}"
    )


def cleanup_fk_cache():
    for item in _FK_CACHE.values():
        cleanup_ascii_urdf_copy(item["urdf_for_pin"], item["urdf"])
    _FK_CACHE.clear()


def solve_full_sequence_with_grasp_candidates(args, current_joints, selected, grasp, pre_grasp, lift, orientation_candidates):
    attempts = []
    for idx, candidate in enumerate(orientation_candidates, start=1):
        print(f"\n========== Full-chain grasp candidate {idx}/{len(orientation_candidates)}: {candidate['name']} ==========")
        try:
            pre_result = solve_waypoint(
                args,
                current_joints,
                "pre_grasp",
                pre_grasp,
                target_rotation_matrix=candidate["rotation_matrix"],
                target_rpy_deg=candidate["rpy_deg"],
                enforce_joint_delta=False,
            )
            grasp_result = solve_waypoint(
                args,
                pre_result["solved_joints_deg"],
                "grasp",
                grasp,
                target_rotation_matrix=candidate["rotation_matrix"],
                target_rpy_deg=candidate["rpy_deg"],
                enforce_joint_delta=False,
            )
            lift_result = solve_waypoint(
                args,
                grasp_result["solved_joints_deg"],
                "lift",
                lift,
                target_rotation_matrix=candidate["rotation_matrix"],
                target_rpy_deg=candidate["rpy_deg"],
                enforce_joint_delta=False,
            )

            if args.pick_only:
                place_tuple = (None, None, None, None, None, None)
                max_delta = max(
                    float(pre_result["max_joint_delta_from_seed"]),
                    float(grasp_result["max_joint_delta_from_seed"]),
                    float(lift_result["max_joint_delta_from_seed"]),
                )
            else:
                place_tuple = solve_place_sequence_with_yaw_candidates(
                    args,
                    selected,
                    grasp,
                    lift,
                    candidate["rotation_matrix"],
                    lift_result,
                )
                max_delta = max(
                    float(pre_result["max_joint_delta_from_seed"]),
                    float(grasp_result["max_joint_delta_from_seed"]),
                    float(lift_result["max_joint_delta_from_seed"]),
                    float(place_tuple[1]["max_joint_delta_from_seed"]),
                    float(place_tuple[2]["max_joint_delta_from_seed"]),
                    float(place_tuple[3]["max_joint_delta_from_seed"]),
                    float(place_tuple[4]["max_joint_delta_from_seed"]),
                    float(place_tuple[5]["max_joint_delta_from_seed"]),
                )

            joint_sequence = [
                current_joints,
                pre_result["solved_joints_deg"],
                grasp_result["solved_joints_deg"],
                lift_result["solved_joints_deg"],
            ]
            if not args.pick_only:
                joint_sequence.extend(
                    [
                        place_tuple[1]["solved_joints_deg"],
                        place_tuple[2]["solved_joints_deg"],
                        place_tuple[3]["solved_joints_deg"],
                        place_tuple[4]["solved_joints_deg"],
                        place_tuple[5]["solved_joints_deg"],
                    ]
                )
            validate_table_clearance_for_sequence(args, selected, joint_sequence, candidate["name"])

            attempts.append(
                {
                    "candidate": candidate,
                    "pre_result": pre_result,
                    "grasp_result": grasp_result,
                    "lift_result": lift_result,
                    "place_tuple": place_tuple,
                    "max_delta": max_delta,
                    "error": None,
                }
            )
            print(f"Full-chain candidate solved: {candidate['name']}, max_delta={max_delta:.3f} deg")
        except Exception as exc:
            attempts.append({"candidate": candidate, "error": exc})
            print(f"Full-chain candidate failed: {candidate['name']}: {exc}")

    valid = [item for item in attempts if item.get("error") is None]
    if not valid:
        errors = "\n".join(f"  {item['candidate']['name']}: {item['error']}" for item in attempts)
        raise RuntimeError("All full-chain grasp candidates failed:\n" + errors)

    valid.sort(key=lambda item: float(item["max_delta"]))
    best = valid[0]
    if float(best["max_delta"]) > float(args.max_joint_delta):
        raise RuntimeError(
            f"Best full-chain grasp exceeds --max-joint-delta={args.max_joint_delta:.3f} deg "
            f"(candidate={best['candidate']['name']}, max_delta={best['max_delta']:.3f} deg)."
        )

    print("\nSelected full-chain grasp candidate:")
    print(f"  name={best['candidate']['name']}")
    print("  target_tool_rpy_deg=[" + ", ".join(f"{v:.3f}" for v in best["candidate"]["rpy_deg"]) + "]")
    print(f"  full_chain_max_joint_delta={best['max_delta']:.3f} deg")
    return best


def move_to_joint_result(robot, result, name, args):
    target = robot.make_joint_angles(result["solved_joints_deg"])
    ok = robot.move_to_joint_angles(target, name=name, timeout_s=args.timeout, duration_s=args.duration)
    if not ok:
        raise RuntimeError(f"Kortex motion failed or aborted at {name}.")
    return joint_values_deg(robot.get_measured_joint_angles())


def main():
    args = parse_args()
    if args.list_frames:
        solve_pink_ik(args)
        return

    selected = detect_one_cup(args)
    grasp = choose_grasp_position(args, selected)
    pre_grasp = [grasp[0], grasp[1], grasp[2] + float(args.pre_grasp_height)]
    lift = [grasp[0], grasp[1], grasp[2] + float(args.lift_height)]
    orientation_candidates = choose_tool_orientation_candidates(args, selected)

    print("\nPlanned tool positions:")
    print(f"  pre_grasp={pre_grasp}")
    print(f"  grasp    ={grasp}")
    print(f"  lift     ={lift}")
    print(f"  grasp_z_offset={args.grasp_z_offset:.4f}, min_grasp_z_above_table={args.min_grasp_z_above_table:.4f}")
    print("  target_tool_orientation=selected after current-joint IK check")

    if not args.yes:
        answer = input("Type PINKPICK to solve IK and execute this one-cup pick smoke test: ").strip()
        if answer != "PINKPICK":
            print("Confirmation not received. Exiting before robot motion.")
            return

    cfg = load_robot_config(args.robot_config)
    kortex = cfg["kortex"]
    with KortexController(
        ip=str(kortex["ip"]),
        username=str(kortex["username"]),
        password=str(kortex["password"]),
        port=int(kortex["port"]),
    ) as robot:
        robot.clear_faults()
        robot.set_single_level_servoing()
        current = joint_values_deg(robot.get_measured_joint_angles())
        print("\nMeasured joints before pick [deg]:")
        print("  " + ", ".join(f"{v:.3f}" for v in current))

        plan = solve_full_sequence_with_grasp_candidates(
            args,
            current,
            selected,
            grasp,
            pre_grasp,
            lift,
            orientation_candidates,
        )
        pre_result = plan["pre_result"]
        grasp_result = plan["grasp_result"]
        lift_result = plan["lift_result"]
        (
            place_plan,
            pre_rotate_result,
            rotate_result,
            pre_place_result,
            place_result,
            retreat_result,
        ) = plan["place_tuple"]

        print("\nOpening gripper...")
        ramp_gripper_to(robot.base, args.open_value, args.gripper_step, args.gripper_settle_time, verbose=False)
        current = move_to_joint_result(robot, pre_result, "pink_pre_grasp", args)
        current = move_to_joint_result(robot, grasp_result, "pink_grasp", args)
        print("\nClosing gripper...")
        ramp_gripper_to(robot.base, args.close_value, args.gripper_step, args.gripper_settle_time, verbose=False)
        time.sleep(0.5)
        current = move_to_joint_result(robot, lift_result, "pink_lift", args)

        if args.pick_only:
            print("\nPick smoke test completed. Gripper remains closed.")
        else:
            current = move_to_joint_result(robot, pre_rotate_result, "pink_pre_rotate", args)
            current = move_to_joint_result(robot, rotate_result, "pink_rotate_upright", args)
            current = move_to_joint_result(robot, pre_place_result, "pink_pre_place", args)
            current = move_to_joint_result(robot, place_result, "pink_place", args)
            print("\nOpening gripper at place target...")
            ramp_gripper_to(robot.base, args.open_value, args.gripper_step, args.gripper_settle_time, verbose=False)
            time.sleep(0.5)
            current = move_to_joint_result(robot, retreat_result, "pink_retreat", args)
            print("\nPick-upright-place test completed. Gripper is open.")
        print("Current joints [deg]:")
        print("  " + ", ".join(f"{v:.3f}" for v in current))


if __name__ == "__main__":
    try:
        main()
    finally:
        cleanup_fk_cache()

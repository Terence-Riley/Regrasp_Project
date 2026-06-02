#!/usr/bin/env python3
"""Viser console for real Regrasp perception, mesh pose, and guarded motion.

This script starts a Viser web UI that can:

1. Show the Kinova Gen3 Lite URDF following measured Kortex joints.
2. Capture RealSense/ArUco cup point clouds in the Kinova base frame.
3. Fit cup meshes, replace the segmented point cloud with the fitted mesh.
4. Create a draggable target cup pose.
5. After explicit GUI arming, execute a real pick-and-place to the dragged pose.

The real-motion path uses Pink/Pinocchio IK and Kortex joint trajectories.
It never uses Kortex Cartesian reach_pose.
"""

from __future__ import annotations

import argparse
import math
import sys
import threading
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
    capture_cluster_pcds,
    fit_cluster_with_meshes,
    load_mesh_models,
)
from scripts.real_pink_pick_one_cup_smoke_test import (  # noqa: E402
    choose_grasp_position,
    choose_tool_orientation_candidates,
    validate_table_clearance_for_sequence,
    invert_transform,
    joint_values_deg,
    make_transform,
    move_to_joint_result,
    rpy_deg_to_rotation_matrix,
    solve_pregrasp_with_orientation_candidates,
    solve_transform_waypoint,
    solve_waypoint,
)
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

    # Perception / mesh fit args mirrored from real_mesh_cup_pose_test.py.
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
    parser.add_argument("--target-label", type=int, default=None)
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

    # Motion args.
    parser.add_argument("--frame", default="tool_frame")
    parser.add_argument(
        "--grasp-frame-mode",
        choices=["center_vertical", "mesh_y_aligned"],
        default="center_vertical",
    )
    parser.add_argument("--mesh-y-roll-samples", type=int, default=12)
    parser.add_argument("--tool-rpy-deg", type=float, nargs=3, default=[180.0, 0.0, 0.0])
    parser.add_argument("--lying-align-gripper", action="store_true", default=True)
    parser.add_argument("--no-lying-align-gripper", dest="lying_align_gripper", action="store_false")
    parser.add_argument("--gripper-closing-axis", choices=["x", "y"], default="x")
    parser.add_argument("--lying-axis-min-xy-norm", type=float, default=0.20)
    parser.add_argument("--pre-grasp-height", type=float, default=0.10)
    parser.add_argument("--lift-height", type=float, default=0.16)
    parser.add_argument("--pre-place-height", type=float, default=0.12)
    parser.add_argument("--retreat-height", type=float, default=0.12)
    parser.add_argument("--grasp-z-offset", type=float, default=0.03)
    parser.add_argument("--min-grasp-z-above-table", type=float, default=0.035)
    parser.add_argument("--collision-table-z", type=float, default=None)
    parser.add_argument("--tool-frame-table-clearance", type=float, default=0.08)
    parser.add_argument("--gripper-table-clearance", type=float, default=0.015)
    parser.add_argument("--gripper-forward-axis", choices=["x", "y", "z", "-x", "-y", "-z"], default="z")
    parser.add_argument("--gripper-finger-length", type=float, default=0.115)
    parser.add_argument("--gripper-open-half-width", type=float, default=0.055)
    parser.add_argument("--gripper-side-axis", choices=["x", "y", "z", "-x", "-y", "-z"], default="y")
    parser.add_argument("--disable-gripper-table-collision-check", action="store_true")
    parser.add_argument("--path-check-samples", type=int, default=16)
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

    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--joint-update-hz", type=float, default=5.0)
    parser.add_argument("--point-size", type=float, default=0.004)
    parser.add_argument(
        "--visual-table-z",
        type=float,
        default=None,
        help="Base-frame z height for the visual table/grid. Default reads table_z_mean from --workspaces.",
    )
    parser.add_argument(
        "--visual-table-size",
        type=float,
        default=1.0,
        help="Width/height of the Viser table grid in meters.",
    )
    return parser.parse_args()


def require_viser():
    try:
        import viser
        from viser.extras import ViserUrdf
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "This script needs viser and yourdfpy. Install with:\n"
            "  pip install viser yourdfpy trimesh\n"
            "or update your conda env from requirements.txt."
        ) from exc
    return viser, ViserUrdf


def load_robot_config(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if "kortex" not in data:
        raise KeyError(f"{path} must contain a 'kortex' section.")
    return data["kortex"]


def resolve_visual_table_z(args):
    if args.visual_table_z is not None:
        return float(args.visual_table_z)
    if args.workspaces.exists():
        with open(args.workspaces, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        workspace = (data.get("workspaces") or {}).get(args.workspace_name) or {}
        if "table_z_mean" in workspace:
            return float(workspace["table_z_mean"])
    return -0.06


def wrap_deg_180(values):
    return [((float(v) + 180.0) % 360.0) - 180.0 for v in values]


def rotation_to_wxyz(R):
    R = np.asarray(R, dtype=np.float64).reshape(3, 3)
    trace = float(np.trace(R))
    if trace > 0.0:
        s = math.sqrt(trace + 1.0) * 2.0
        w = 0.25 * s
        x = (R[2, 1] - R[1, 2]) / s
        y = (R[0, 2] - R[2, 0]) / s
        z = (R[1, 0] - R[0, 1]) / s
    else:
        idx = int(np.argmax(np.diag(R)))
        if idx == 0:
            s = math.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2.0
            w = (R[2, 1] - R[1, 2]) / s
            x = 0.25 * s
            y = (R[0, 1] + R[1, 0]) / s
            z = (R[0, 2] + R[2, 0]) / s
        elif idx == 1:
            s = math.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2.0
            w = (R[0, 2] - R[2, 0]) / s
            x = (R[0, 1] + R[1, 0]) / s
            y = 0.25 * s
            z = (R[1, 2] + R[2, 1]) / s
        else:
            s = math.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2.0
            w = (R[1, 0] - R[0, 1]) / s
            x = (R[0, 2] + R[2, 0]) / s
            y = (R[1, 2] + R[2, 1]) / s
            z = 0.25 * s
    q = np.array([w, x, y, z], dtype=np.float64)
    return q / max(float(np.linalg.norm(q)), 1e-12)


def wxyz_to_rotation(q):
    w, x, y, z = np.asarray(q, dtype=np.float64).reshape(4)
    n = max(float(np.linalg.norm([w, x, y, z])), 1e-12)
    w, x, y, z = w / n, x / n, y / n, z / n
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def transform_to_pose(T):
    T = np.asarray(T, dtype=np.float64).reshape(4, 4)
    return T[:3, 3], rotation_to_wxyz(T[:3, :3])


def pose_to_transform(position, wxyz):
    return make_transform(wxyz_to_rotation(wxyz), np.asarray(position, dtype=np.float64))


class ViserRegraspConsole:
    def __init__(self, args):
        self.args = args
        self.viser, self.ViserUrdf = require_viser()
        self.server = self.viser.ViserServer(host=args.host, port=args.port)
        self.models = load_mesh_models(args)
        self.robot = None
        self.urdf_vis = None
        self.selected = None
        self.target_handle = None
        self.target_T_base_cup = None
        self.pin = None
        self.pin_model = None
        self.pin_data = None
        self.pin_urdf = None
        self.busy = threading.Lock()
        self.stop_event = threading.Event()

        self.status = self.server.gui.add_text("Status", "Idle")
        self.arm_motion = self.server.gui.add_checkbox("Arm real motion", initial_value=False)
        self.detect_button = self.server.gui.add_button("Detect cup / update scene")
        self.target_button = self.server.gui.add_button("Reset target to detected cup")
        self.execute_button = self.server.gui.add_button("Execute real pick to dragged target")

        self.detect_button.on_click(lambda _: threading.Thread(target=self.detect_and_render, daemon=True).start())
        self.target_button.on_click(lambda _: self.reset_target_to_detected())
        self.execute_button.on_click(lambda _: threading.Thread(target=self.execute_motion, daemon=True).start())

    def set_status(self, text):
        print(text)
        self.status.value = text

    def connect_robot(self):
        cfg = load_robot_config(self.args.robot_config)
        self.robot = KortexController(
            ip=str(cfg["ip"]),
            username=str(cfg["username"]),
            password=str(cfg["password"]),
            port=int(cfg["port"]),
        )
        self.robot.__enter__()
        self.robot.clear_faults()
        self.robot.set_single_level_servoing()

    def setup_scene(self):
        self.server.scene.set_up_direction("+z")
        self.server.scene.add_frame("/base", axes_length=0.12, axes_radius=0.004)
        table_z = resolve_visual_table_z(self.args)
        self.server.scene.add_frame(
            "/table",
            position=(0.0, 0.0, float(table_z)),
            wxyz=(1.0, 0.0, 0.0, 0.0),
            show_axes=False,
        )
        self.server.scene.add_grid(
            "/table/grid",
            width=float(self.args.visual_table_size),
            height=float(self.args.visual_table_size),
            cell_size=0.05,
        )
        self.server.scene.add_frame(
            "/table/origin",
            position=(0.0, 0.0, 0.0),
            axes_length=0.08,
            axes_radius=0.003,
        )
        self.server.scene.add_frame("/robot", show_axes=False)
        self.urdf_vis = self.ViserUrdf(self.server, urdf_or_path=self.args.urdf, root_node_name="/robot")
        self.target_handle = self.server.scene.add_transform_controls(
            "/target_cup",
            scale=0.12,
            position=(0.35, 0.0, 0.08),
            wxyz=(1.0, 0.0, 0.0, 0.0),
        )

        @self.target_handle.on_update
        def _(_event):
            self.target_T_base_cup = pose_to_transform(self.target_handle.position, self.target_handle.wxyz)

        self.target_T_base_cup = pose_to_transform(self.target_handle.position, self.target_handle.wxyz)
        self.set_status(f"Visual table grid z={table_z:.4f} m in base frame.")

    def setup_pinocchio_tool_frame(self):
        pin, *_ = require_pink_pinocchio()
        urdf = self.args.urdf if self.args.urdf.is_absolute() else PROJECT_ROOT / self.args.urdf
        self.pin_urdf = make_ascii_urdf_copy(urdf) if not is_ascii_path(urdf) else urdf
        self.pin = pin
        self.pin_model = pin.buildModelFromUrdf(str(self.pin_urdf))
        self.pin_data = self.pin_model.createData()
        if self.pin_model.getFrameId(self.args.frame) >= len(self.pin_model.frames):
            raise KeyError(f"Frame not found in URDF: {self.args.frame}")
        self.server.scene.add_frame(
            "/robot/tool_frame_live",
            axes_length=0.10,
            axes_radius=0.004,
        )

    def update_tool_frame(self, wrapped_joints_deg):
        if self.pin is None or self.pin_model is None or self.pin_data is None:
            return
        q = self.pin.neutral(self.pin_model)
        q = set_arm_joints_deg(self.pin_model, q, wrapped_joints_deg)
        self.pin.forwardKinematics(self.pin_model, self.pin_data, q)
        self.pin.updateFramePlacements(self.pin_model, self.pin_data)
        frame_id = self.pin_model.getFrameId(self.args.frame)
        pose = self.pin_data.oMf[frame_id]
        T = make_transform(np.asarray(pose.rotation, dtype=np.float64), np.asarray(pose.translation, dtype=np.float64))
        position, wxyz = transform_to_pose(T)
        self.server.scene.add_frame(
            "/robot/tool_frame_live",
            position=tuple(float(v) for v in position),
            wxyz=tuple(float(v) for v in wxyz),
            axes_length=0.10,
            axes_radius=0.004,
        )

    def start_joint_thread(self):
        def loop():
            limits = None
            try:
                limits = self.urdf_vis.get_actuated_joint_limits()
            except Exception:
                pass
            names = list(limits.keys()) if isinstance(limits, dict) else [f"joint_{i}" for i in range(1, 7)]
            dt = 1.0 / max(float(self.args.joint_update_hz), 0.5)
            while not self.stop_event.is_set():
                try:
                    measured = joint_values_deg(self.robot.get_measured_joint_angles())
                    wrapped = wrap_deg_180(measured)
                    self.update_tool_frame(wrapped)
                    q = []
                    for name in names:
                        if name.startswith("joint_"):
                            idx = int(name.split("_")[1]) - 1
                            q.append(math.radians(wrapped[idx]) if 0 <= idx < 6 else 0.0)
                        else:
                            q.append(0.0)
                    self.urdf_vis.update_cfg(np.asarray(q, dtype=np.float64))
                except Exception as exc:
                    print(f"Robot visualization update failed: {exc}")
                time.sleep(dt)

        threading.Thread(target=loop, daemon=True).start()

    def fit_best_candidate(self, capture_info):
        candidates = capture_info["candidates"]
        cluster_pcds = capture_info["cluster_pcds"]
        selected = None
        for candidate in candidates:
            if self.args.target_label is not None and int(candidate["label"]) != int(self.args.target_label):
                continue
            label = int(candidate["label"])
            fit = fit_cluster_with_meshes(
                cluster_pcds[label]["clean"],
                self.models,
                self.args,
                table_z=capture_info.get("table_z"),
            )
            if fit is None:
                continue
            fit["label"] = label
            fit["candidate"] = candidate
            fit["cluster_pcd"] = cluster_pcds[label]["clean"]
            fit["table_z"] = capture_info.get("table_z")
            if selected is None or fit["score"] > selected["score"]:
                selected = fit
            if self.args.target_label is not None:
                break
        if selected is None:
            raise RuntimeError("No valid mesh fit found.")
        return selected

    def add_mesh_node(self, prefix, model, T, color):
        vertices = np.asarray(model.mesh.vertices, dtype=np.float32)
        faces = np.asarray(model.mesh.triangles, dtype=np.uint32)
        position, wxyz = transform_to_pose(T)
        self.server.scene.add_frame(prefix, position=position, wxyz=wxyz, axes_length=0.08, axes_radius=0.003)
        self.server.scene.add_mesh_simple(
            f"{prefix}/mesh",
            vertices=vertices,
            faces=faces,
            color=color,
            opacity=0.65,
        )

    def add_target_mesh_child(self, model):
        vertices = np.asarray(model.mesh.vertices, dtype=np.float32)
        faces = np.asarray(model.mesh.triangles, dtype=np.uint32)
        self.server.scene.add_mesh_simple(
            "/target_cup/mesh",
            vertices=vertices,
            faces=faces,
            color=(0.1, 0.55, 1.0),
            opacity=0.55,
        )

    def detect_and_render(self):
        if not self.busy.acquire(blocking=False):
            self.set_status("Busy")
            return
        try:
            self.set_status("Capturing RealSense and fitting mesh...")
            capture_info = capture_cluster_pcds(self.args)
            pcd = capture_info["pcd_no_table"]
            pts = np.asarray(pcd.points, dtype=np.float32)
            colors = np.asarray(pcd.colors, dtype=np.float32)
            if pts.size:
                self.server.scene.add_point_cloud(
                    "/perception/no_table_points",
                    points=pts,
                    colors=colors if len(colors) == len(pts) else np.tile(np.array([[0.2, 0.7, 1.0]], dtype=np.float32), (len(pts), 1)),
                    point_size=float(self.args.point_size),
                )

            selected = self.fit_best_candidate(capture_info)
            self.selected = selected
            self.add_mesh_node("/detected_cup", selected["model"], selected["T_base_model"], (0.05, 0.05, 0.05))
            clean = selected["cluster_pcd"]
            cluster_pts = np.asarray(clean.points, dtype=np.float32)
            if len(cluster_pts):
                self.server.scene.add_point_cloud(
                    "/perception/selected_cluster",
                    points=cluster_pts,
                    colors=np.tile(np.array([[1.0, 0.25, 0.05]], dtype=np.float32), (len(cluster_pts), 1)),
                    point_size=float(self.args.point_size) * 1.4,
                )
            self.reset_target_to_detected()
            self.set_status(
                f"Detected label={selected['label']:02d}, model={selected['model_name']}, "
                f"state={selected['state']}, signed={selected.get('signed_orientation')}"
            )
        except Exception as exc:
            self.set_status(f"Detection failed: {exc}")
        finally:
            self.busy.release()

    def reset_target_to_detected(self):
        if self.selected is None:
            self.set_status("No detected cup yet.")
            return
        position, wxyz = transform_to_pose(self.selected["T_base_model"])
        self.target_handle.position = tuple(float(v) for v in position)
        self.target_handle.wxyz = tuple(float(v) for v in wxyz)
        self.target_T_base_cup = np.asarray(self.selected["T_base_model"], dtype=np.float64).copy()
        self.add_target_mesh_child(self.selected["model"])
        self.set_status("Target reset to detected cup pose. Drag /target_cup before executing.")

    def plan_to_dragged_target(self, current_joints):
        selected = self.selected
        if selected is None or self.target_T_base_cup is None:
            raise RuntimeError("Need a detected cup and dragged target before executing.")
        grasp = choose_grasp_position(self.args, selected)
        pre_grasp = [grasp[0], grasp[1], grasp[2] + float(self.args.pre_grasp_height)]
        lift = [grasp[0], grasp[1], grasp[2] + float(self.args.lift_height)]
        candidates = choose_tool_orientation_candidates(self.args, selected)
        orientation, pre_result = solve_pregrasp_with_orientation_candidates(self.args, current_joints, pre_grasp, candidates)
        R_grasp = np.asarray(orientation["rotation_matrix"], dtype=np.float64).reshape(3, 3)
        rpy_grasp = orientation["rpy_deg"]
        grasp_result = solve_waypoint(
            self.args,
            pre_result["solved_joints_deg"],
            "grasp",
            grasp,
            target_rotation_matrix=R_grasp,
            target_rpy_deg=rpy_grasp,
        )
        lift_result = solve_waypoint(
            self.args,
            grasp_result["solved_joints_deg"],
            "lift",
            lift,
            target_rotation_matrix=R_grasp,
            target_rpy_deg=rpy_grasp,
        )

        T_base_tool_grasp = make_transform(R_grasp, np.asarray(grasp, dtype=np.float64))
        T_tool_cup = invert_transform(T_base_tool_grasp) @ np.asarray(selected["T_base_model"], dtype=np.float64)
        T_base_tool_target = np.asarray(self.target_T_base_cup, dtype=np.float64) @ invert_transform(T_tool_cup)
        pre_place = T_base_tool_target.copy()
        pre_place[:3, 3] += np.array([0.0, 0.0, float(self.args.pre_place_height)])
        retreat = T_base_tool_target.copy()
        retreat[:3, 3] += np.array([0.0, 0.0, float(self.args.retreat_height)])

        pre_place_result = solve_transform_waypoint(
            self.args,
            lift_result["solved_joints_deg"],
            "pre_place_dragged",
            pre_place,
        )
        place_result = solve_transform_waypoint(
            self.args,
            pre_place_result["solved_joints_deg"],
            "place_dragged",
            T_base_tool_target,
        )
        retreat_result = solve_transform_waypoint(
            self.args,
            place_result["solved_joints_deg"],
            "retreat_dragged",
            retreat,
        )
        validate_table_clearance_for_sequence(
            self.args,
            selected,
            [
                current_joints,
                pre_result["solved_joints_deg"],
                grasp_result["solved_joints_deg"],
                lift_result["solved_joints_deg"],
                pre_place_result["solved_joints_deg"],
                place_result["solved_joints_deg"],
                retreat_result["solved_joints_deg"],
            ],
            "viser_dragged_target",
        )
        return pre_result, grasp_result, lift_result, pre_place_result, place_result, retreat_result

    def execute_motion(self):
        if not self.busy.acquire(blocking=False):
            self.set_status("Busy")
            return
        try:
            if not self.arm_motion.value:
                self.set_status("Motion blocked. Check 'Arm real motion' first.")
                return
            self.set_status("Solving IK sequence to dragged target...")
            current = joint_values_deg(self.robot.get_measured_joint_angles())
            sequence = self.plan_to_dragged_target(current)
            self.set_status("Executing real robot motion...")
            ramp_gripper_to(self.robot.base, self.args.open_value, self.args.gripper_step, self.args.gripper_settle_time, verbose=False)
            current = move_to_joint_result(self.robot, sequence[0], "viser_pre_grasp", self.args)
            current = move_to_joint_result(self.robot, sequence[1], "viser_grasp", self.args)
            ramp_gripper_to(self.robot.base, self.args.close_value, self.args.gripper_step, self.args.gripper_settle_time, verbose=False)
            time.sleep(0.5)
            current = move_to_joint_result(self.robot, sequence[2], "viser_lift", self.args)
            current = move_to_joint_result(self.robot, sequence[3], "viser_pre_place", self.args)
            current = move_to_joint_result(self.robot, sequence[4], "viser_place", self.args)
            ramp_gripper_to(self.robot.base, self.args.open_value, self.args.gripper_step, self.args.gripper_settle_time, verbose=False)
            time.sleep(0.5)
            current = move_to_joint_result(self.robot, sequence[5], "viser_retreat", self.args)
            self.arm_motion.value = False
            self.set_status("Done. Motion disarmed.")
        except Exception as exc:
            self.arm_motion.value = False
            self.set_status(f"Motion failed: {exc}")
        finally:
            self.busy.release()

    def run(self):
        self.connect_robot()
        self.setup_scene()
        self.setup_pinocchio_tool_frame()
        self.start_joint_thread()
        self.set_status(f"Viser running at http://{self.args.host}:{self.args.port}")
        try:
            while True:
                time.sleep(1.0)
        except KeyboardInterrupt:
            self.stop_event.set()
            if self.pin_urdf is not None:
                cleanup_ascii_urdf_copy(self.pin_urdf, self.args.urdf)
            if self.robot is not None:
                self.robot.__exit__(None, None, None)


def main():
    args = parse_args()
    console = ViserRegraspConsole(args)
    console.run()


if __name__ == "__main__":
    main()

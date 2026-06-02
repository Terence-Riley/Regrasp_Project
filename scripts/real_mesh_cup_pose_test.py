#!/usr/bin/env python3
"""Estimate cup axis pose from RealSense point clouds and cup meshes.

中文说明：该脚本用于基于杯子 CAD/STL mesh 估计杯子的型号、中心、主轴和
状态。它支持直接从 RealSense 在线采集，也支持读取已经保存的 PLY/PCD 点云
离线验证。脚本只做感知，不连接 Kortex，不移动真机。

This script is perception-only. It does not connect to Kortex and it does not
move the robot. It can either capture a fresh RealSense frame or fit meshes to
previously saved PLY/PCD point clouds.

The estimator is designed for rotationally symmetric cups. It searches mesh
alignments whose symmetry axis is upright or lying, runs ICP against each
cleaned cluster, and reports the cup center, axis direction, state, model name,
and match score. Yaw is searched internally only to improve ICP; the reported
pose should not be interpreted as a reliable yaw estimate.
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import sys
import tempfile
import time
import uuid
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import open3d as o3d

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from perception.aruco_detector import (  # noqa: E402
    MARKER_LENGTH,
    TARGET_MARKER_ID,
    create_aruco_detector,
    detect_target_marker_pose,
)
from perception.realsense_camera import RealSenseCamera  # noqa: E402
from scripts.plan_regrasp_dry_run import (  # noqa: E402
    build_camera_point_cloud,
    clean_cluster_pcd,
    crop_pcd_by_workspace,
    load_T_base_marker,
    load_workspaces,
    make_candidates,
    make_depth_valid_mask,
    remove_near_table_residuals,
    remove_table_plane,
    sort_candidates,
    transform_pcd,
)
from utils.transform_utils import invert_T  # noqa: E402


BASE_Z = np.array([0.0, 0.0, 1.0], dtype=np.float64)


def is_ascii_path(path: Path) -> bool:
    try:
        str(path).encode("ascii")
        return True
    except UnicodeEncodeError:
        return False


def copy_to_ascii_temp(path: Path) -> Path:
    suffix = path.suffix or ".tmp"
    dst = Path(tempfile.gettempdir()) / f"regrasp_o3d_{uuid.uuid4().hex}{suffix}"
    shutil.copy2(path, dst)
    return dst


def read_triangle_mesh_compatible(path: Path) -> o3d.geometry.TriangleMesh:
    """Read a mesh on Windows even when the project path contains non-ASCII text."""
    if is_ascii_path(path):
        return o3d.io.read_triangle_mesh(str(path))
    tmp_path = copy_to_ascii_temp(path)
    try:
        return o3d.io.read_triangle_mesh(str(tmp_path))
    finally:
        try:
            tmp_path.unlink()
        except OSError:
            pass


def read_point_cloud_compatible(path: Path) -> o3d.geometry.PointCloud:
    """Read a point cloud on Windows even when the project path contains non-ASCII text."""
    if is_ascii_path(path):
        return o3d.io.read_point_cloud(str(path))
    tmp_path = copy_to_ascii_temp(path)
    try:
        return o3d.io.read_point_cloud(str(tmp_path))
    finally:
        try:
            tmp_path.unlink()
        except OSError:
            pass


def write_point_cloud_compatible(path: Path, pcd: o3d.geometry.PointCloud) -> bool:
    """Write a point cloud on Windows even when the project path contains non-ASCII text."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if is_ascii_path(path):
        return bool(o3d.io.write_point_cloud(str(path), pcd))
    tmp_path = Path(tempfile.gettempdir()) / f"regrasp_o3d_{uuid.uuid4().hex}{path.suffix or '.ply'}"
    try:
        ok = bool(o3d.io.write_point_cloud(str(tmp_path), pcd))
        if ok:
            shutil.copy2(tmp_path, path)
        return ok
    finally:
        try:
            tmp_path.unlink()
        except OSError:
            pass


@dataclass
class MeshModel:
    name: str
    path: Path
    mesh: o3d.geometry.TriangleMesh
    pcd: o3d.geometry.PointCloud
    axis_local: np.ndarray
    bbox_center_local: np.ndarray
    bottom_center_local: np.ndarray
    top_center_local: np.ndarray
    bottom_radius: float
    top_radius: float
    end_radius_ratio: float
    height_axis: float
    extent: np.ndarray


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=15)
    parser.add_argument(
        "--camera-stream-order",
        choices=["color_first", "depth_first", "both"],
        default="color_first",
    )
    parser.add_argument("--camera-start-attempts", type=int, default=3)
    parser.add_argument("--camera-retry-delay", type=float, default=2.0)
    parser.add_argument("--camera-hardware-reset-on-fail", action="store_true")

    parser.add_argument(
        "--calibration",
        type=Path,
        default=PROJECT_ROOT / "configs" / "real_calibration.yaml",
        help="YAML containing T_base_marker.",
    )
    parser.add_argument(
        "--workspaces",
        type=Path,
        default=PROJECT_ROOT / "configs" / "real_workspaces.yaml",
        help="YAML generated by scripts/calibrate_workspaces.py.",
    )
    parser.add_argument(
        "--workspace-name",
        choices=["pickup", "place"],
        default="pickup",
        help="Workspace polygon used for cup search.",
    )
    parser.add_argument(
        "--input-pcd",
        action="append",
        type=Path,
        default=[],
        help="Offline mode: existing PLY/PCD point cloud to fit. Repeat for multiple files.",
    )
    parser.add_argument(
        "--input-as-scene",
        action="store_true",
        help="Treat --input-pcd as scene/no-table point clouds and run DBSCAN before mesh fitting.",
    )
    parser.add_argument(
        "--input-frame",
        choices=["base", "camera"],
        default="base",
        help="Frame of --input-pcd. Use base for files saved by the current debug scripts.",
    )
    parser.add_argument(
        "--skip-input-cleaning",
        action="store_true",
        help="Offline direct-cluster mode only: treat input cluster point clouds as already cleaned.",
    )

    parser.add_argument(
        "--mesh",
        action="append",
        required=True,
        help="Cup mesh spec. Use NAME=path/to/model.stl or just path/to/model.stl. Repeat for multiple models.",
    )
    parser.add_argument(
        "--mesh-unit",
        choices=["m", "mm"],
        default="m",
        help="Unit of mesh vertices. Use mm for most CAD exports.",
    )
    parser.add_argument(
        "--mesh-scale",
        type=float,
        default=1.0,
        help="Extra scale after mesh-unit conversion.",
    )
    parser.add_argument(
        "--mesh-axis",
        choices=["x", "y", "z", "-x", "-y", "-z"],
        default="y",
        help="Cup symmetry/main axis in the mesh local frame.",
    )
    parser.add_argument(
        "--mesh-origin",
        choices=["bbox_center", "bbox_bottom_center", "mesh_origin"],
        default="mesh_origin",
        help="Model origin used for reported T_base_model.",
    )
    parser.add_argument("--model-sample-points", type=int, default=5000)
    parser.add_argument("--model-voxel-size", type=float, default=0.004)

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
    parser.add_argument(
        "--enable-dark-object-filter",
        action="store_true",
        help="Keep only dark RGB points before DBSCAN. Useful for black cups on a white table.",
    )
    parser.add_argument(
        "--dark-value-max",
        type=float,
        default=0.38,
        help="Maximum HSV value/brightness in [0,1] for dark-object filtering.",
    )
    parser.add_argument(
        "--dark-saturation-min",
        type=float,
        default=0.0,
        help="Minimum HSV saturation in [0,1] for dark-object filtering. Keep 0 for black objects.",
    )
    parser.add_argument(
        "--dark-rgb-max",
        type=float,
        default=None,
        help="Optional max of all RGB channels in [0,1]. If set, points with any channel above this are rejected.",
    )

    parser.add_argument(
        "--pick-sort",
        choices=["nearest", "largest", "x", "y"],
        default="nearest",
        help="Order used when printing clusters.",
    )
    parser.add_argument(
        "--fit-states",
        choices=["both", "upright", "lying"],
        default="both",
        help="Which coarse cup states to try during mesh alignment.",
    )
    parser.add_argument(
        "--force-model-set",
        default=None,
        help="Scene-level prior. Example: big,small forces the selected clusters to contain exactly one big and one small model.",
    )
    parser.add_argument("--yaw-samples", type=int, default=12)
    parser.add_argument("--roll-samples", type=int, default=8)
    parser.add_argument("--icp-max-distance", type=float, default=0.025)
    parser.add_argument("--icp-iterations", type=int, default=50)
    parser.add_argument(
        "--coverage-distance",
        type=float,
        default=None,
        help="Distance threshold for observed-cluster coverage. Defaults to --icp-max-distance.",
    )
    parser.add_argument(
        "--min-cluster-coverage",
        type=float,
        default=0.25,
        help="Reject a model if too few observed cluster points are close to the transformed mesh.",
    )
    parser.add_argument(
        "--extent-weight",
        type=float,
        default=1.2,
        help="Penalty weight for transformed mesh bbox size mismatch against the observed cluster bbox.",
    )
    parser.add_argument(
        "--observed-size-prior-weight",
        type=float,
        default=1.4,
        help="Strong upright-only prior using observed cluster XY extent to distinguish big/small cups.",
    )
    parser.add_argument(
        "--big-observed-diameter-min",
        type=float,
        default=0.058,
        help="Upright clusters with max XY extent >= this value are more likely big cups.",
    )
    parser.add_argument(
        "--small-observed-diameter-max",
        type=float,
        default=0.055,
        help="Upright clusters with max XY extent <= this value are more likely small cups.",
    )
    parser.add_argument(
        "--big-model-name",
        default="big",
        help="Mesh name treated as the big cup for observed-size prior.",
    )
    parser.add_argument(
        "--small-model-name",
        default="small",
        help="Mesh name treated as the small cup for observed-size prior.",
    )
    parser.add_argument(
        "--coverage-weight",
        type=float,
        default=0.9,
        help="Score weight for observed-cluster coverage. Increase this if small cups win too often.",
    )
    parser.add_argument(
        "--fitness-weight",
        type=float,
        default=0.4,
        help="Score weight for Open3D ICP model-to-cluster fitness.",
    )
    parser.add_argument(
        "--axis-report",
        choices=["model", "unoriented", "table_up"],
        default="table_up",
        help="How to report axis direction. table_up flips upright-like axes to positive base z.",
    )
    parser.add_argument(
        "--allow-unknown",
        action="store_true",
        help="Allow unknown state. By default the result is forced to upright_like or lying.",
    )
    parser.set_defaults(detect_inverted_upright=False)
    parser.add_argument(
        "--detect-inverted-upright",
        dest="detect_inverted_upright",
        action="store_true",
        help="For upright-like cups, report inverted_upright when the larger rim appears below the smaller end.",
    )
    parser.add_argument(
        "--disable-detect-inverted-upright",
        dest="detect_inverted_upright",
        action="store_false",
        help="Disable inverted_upright reporting.",
    )
    parser.add_argument("--upright-axis-z-threshold", type=float, default=0.75)
    parser.add_argument("--lying-axis-z-threshold", type=float, default=0.45)
    parser.add_argument(
        "--cup-height-min",
        type=float,
        default=0.065,
        help="Expected cup axis height lower bound in meters.",
    )
    parser.add_argument(
        "--cup-height-max",
        type=float,
        default=0.085,
        help="Expected cup axis height upper bound in meters.",
    )
    parser.add_argument(
        "--height-prior-weight",
        type=float,
        default=0.55,
        help="Penalty weight when mesh axis height is outside the expected cup height range.",
    )
    parser.add_argument(
        "--table-z",
        type=float,
        default=None,
        help="Known table z in base frame. If omitted, workspace table_z_mean is used when available.",
    )
    parser.add_argument(
        "--max-support-gap",
        type=float,
        default=0.035,
        help="Maximum allowed gap in meters between cluster min z and table z before it is treated as floating.",
    )
    parser.add_argument(
        "--support-prior-weight",
        type=float,
        default=0.75,
        help="Penalty weight for clusters/poses that appear to float above the table.",
    )
    parser.add_argument(
        "--upright-z-extent-min",
        type=float,
        default=0.045,
        help="Observed z extent should be at least this large for an upright cup.",
    )
    parser.add_argument(
        "--lying-z-extent-max",
        type=float,
        default=0.060,
        help="Observed z extent should usually stay below this value for a lying cup.",
    )
    parser.add_argument(
        "--state-prior-weight",
        type=float,
        default=0.35,
        help="Penalty weight for state-specific observed extent inconsistencies.",
    )
    parser.add_argument(
        "--axis-snap-prior-weight",
        type=float,
        default=0.9,
        help="Penalty weight for fitted cup axes that are neither almost vertical nor almost horizontal.",
    )
    parser.add_argument(
        "--upright-axis-z-ideal-min",
        type=float,
        default=0.92,
        help="Upright fits are penalized when abs(axis_z) is below this value.",
    )
    parser.add_argument(
        "--lying-axis-z-ideal-max",
        type=float,
        default=0.18,
        help="Lying fits are penalized when abs(axis_z) is above this value.",
    )
    parser.add_argument(
        "--hard-axis-snap",
        action="store_true",
        help="Reject fits whose axes are not close enough to vertical/horizontal ideals.",
    )
    parser.add_argument(
        "--upright-rim-prior-weight",
        type=float,
        default=0.0,
        help="Penalty weight for upright cup top/bottom radius-ratio mismatch.",
    )
    parser.add_argument(
        "--upright-rim-absolute-weight",
        type=float,
        default=0.0,
        help="Penalty weight for upright top/bottom absolute radius mismatch.",
    )
    parser.add_argument(
        "--rim-absolute-scale",
        type=float,
        default=0.01,
        help="Radius error scale in meters. 0.01 means 1 cm radius error gives about penalty 1.",
    )
    parser.add_argument(
        "--rim-band-fraction",
        type=float,
        default=0.18,
        help="Fraction of cup axis height used as the top/bottom band for rim radius estimation.",
    )
    parser.add_argument(
        "--rim-radius-quantile",
        type=float,
        default=0.80,
        help="Radius quantile used to estimate each end circle. Higher values are less affected by missing center points.",
    )
    parser.add_argument(
        "--min-rim-points",
        type=int,
        default=20,
        help="Minimum observed points in top/bottom bands before applying the upright rim prior.",
    )
    parser.add_argument(
        "--distinct-rim-ratio",
        type=float,
        default=1.12,
        help="If max(top,bottom)/min(top,bottom) exceeds this, the observed cup is treated as having a distinct mouth/base size.",
    )
    parser.add_argument("--min-fitness", type=float, default=0.08)
    parser.add_argument("--max-rmse", type=float, default=0.035)

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "debug_pointclouds" / "mesh_cup_pose",
    )
    parser.add_argument("--save-debug-pcd", action="store_true")
    parser.add_argument("--visualize", action="store_true")
    return parser.parse_args()


def parse_mesh_spec(spec: str) -> tuple[str, Path]:
    if "=" in spec:
        name, path_text = spec.split("=", 1)
        name = name.strip()
        path = Path(path_text.strip())
    else:
        path = Path(spec.strip())
        name = path.stem
    if not name:
        raise ValueError(f"Invalid mesh spec: {spec}")
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return name, path


def axis_from_arg(text: str) -> np.ndarray:
    sign = -1.0 if text.startswith("-") else 1.0
    key = text[-1]
    axis = {
        "x": np.array([1.0, 0.0, 0.0]),
        "y": np.array([0.0, 1.0, 0.0]),
        "z": np.array([0.0, 0.0, 1.0]),
    }[key]
    return sign * axis


def normalize(v: np.ndarray, fallback: np.ndarray | None = None) -> np.ndarray:
    v = np.asarray(v, dtype=np.float64).reshape(3)
    n = float(np.linalg.norm(v))
    if n < 1e-12:
        if fallback is None:
            raise ValueError("Cannot normalize zero vector.")
        return np.asarray(fallback, dtype=np.float64).reshape(3)
    return v / n


def rotation_matrix_from_axis_angle(axis: np.ndarray, angle_rad: float) -> np.ndarray:
    axis = normalize(axis, BASE_Z)
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


def rotation_align_vectors(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a = normalize(a, BASE_Z)
    b = normalize(b, BASE_Z)
    cross = np.cross(a, b)
    dot = float(np.clip(np.dot(a, b), -1.0, 1.0))
    if np.linalg.norm(cross) < 1e-12:
        if dot > 0.0:
            return np.eye(3)
        helper = np.array([1.0, 0.0, 0.0])
        if abs(float(np.dot(a, helper))) > 0.9:
            helper = np.array([0.0, 1.0, 0.0])
        return rotation_matrix_from_axis_angle(np.cross(a, helper), math.pi)
    axis = normalize(cross)
    angle = math.atan2(float(np.linalg.norm(cross)), dot)
    return rotation_matrix_from_axis_angle(axis, angle)


def make_transform(R: np.ndarray, t: np.ndarray) -> np.ndarray:
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = np.asarray(R, dtype=np.float64).reshape(3, 3)
    T[:3, 3] = np.asarray(t, dtype=np.float64).reshape(3)
    return T


def mesh_origin_offset(mesh: o3d.geometry.TriangleMesh, axis_local: np.ndarray, mode: str) -> np.ndarray:
    vertices = np.asarray(mesh.vertices)
    if mode == "mesh_origin":
        return np.zeros(3, dtype=np.float64)

    min_bound = np.min(vertices, axis=0)
    max_bound = np.max(vertices, axis=0)
    center = (min_bound + max_bound) / 2.0
    if mode == "bbox_center":
        return center

    axis = normalize(axis_local)
    projections = vertices @ axis
    bottom = float(np.min(projections))
    center_on_axis = float(center @ axis)
    return center + (bottom - center_on_axis) * axis


def estimate_end_radii(points: np.ndarray, axis: np.ndarray, band_fraction: float, radius_quantile: float, axis_point=None):
    points = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    axis = normalize(axis)
    if axis_point is None:
        axis_point = np.zeros(3, dtype=np.float64)
    axis_point = np.asarray(axis_point, dtype=np.float64).reshape(3)
    relative = points - axis_point
    projections = relative @ axis
    p_min = float(np.min(projections))
    p_max = float(np.max(projections))
    height = max(p_max - p_min, 1e-9)
    band = max(float(band_fraction) * height, 1e-6)

    radial_vectors = relative - np.outer(projections, axis)
    radial_distances = np.linalg.norm(radial_vectors, axis=1)

    bottom_mask = projections <= p_min + band
    top_mask = projections >= p_max - band

    def band_radius(mask):
        if np.count_nonzero(mask) == 0:
            return 0.0
        return float(np.quantile(radial_distances[mask], float(radius_quantile)))

    bottom_radius = band_radius(bottom_mask)
    top_radius = band_radius(top_mask)
    ratio = max(top_radius, bottom_radius) / max(min(top_radius, bottom_radius), 1e-9)
    return {
        "bottom_radius": float(bottom_radius),
        "top_radius": float(top_radius),
        "end_radius_ratio": float(ratio),
        "bottom_points": int(np.count_nonzero(bottom_mask)),
        "top_points": int(np.count_nonzero(top_mask)),
        "axis_height": float(height),
    }


def estimate_model_end_radii(vertices: np.ndarray, axis_local: np.ndarray, args):
    return estimate_end_radii(
        vertices,
        axis_local,
        band_fraction=float(args.rim_band_fraction),
        radius_quantile=float(args.rim_radius_quantile),
    )


def model_end_centers(vertices: np.ndarray, axis_local: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    axis = normalize(axis_local)
    projections = vertices @ axis
    p_min = float(np.min(projections))
    p_max = float(np.max(projections))
    center = (np.min(vertices, axis=0) + np.max(vertices, axis=0)) / 2.0
    center_projection = float(center @ axis)
    bottom_center = center + (p_min - center_projection) * axis
    top_center = center + (p_max - center_projection) * axis
    return bottom_center, top_center


def load_mesh_model(name: str, path: Path, args) -> MeshModel:
    if not path.exists():
        raise FileNotFoundError(f"Missing mesh file for {name}: {path}")

    mesh = read_triangle_mesh_compatible(path)
    if mesh.is_empty():
        raise ValueError(f"Could not read mesh or mesh is empty: {path}")
    if not mesh.has_vertex_normals():
        mesh.compute_vertex_normals()

    unit_scale = 0.001 if args.mesh_unit == "mm" else 1.0
    scale = float(unit_scale) * float(args.mesh_scale)
    mesh.scale(scale, center=np.zeros(3))

    axis_local = axis_from_arg(args.mesh_axis)
    offset = mesh_origin_offset(mesh, axis_local, args.mesh_origin)
    mesh.translate(-offset)
    mesh.compute_vertex_normals()

    vertices = np.asarray(mesh.vertices)
    projections = vertices @ normalize(axis_local)
    height_axis = float(np.max(projections) - np.min(projections))
    extent = np.ptp(vertices, axis=0)
    end_radii = estimate_model_end_radii(vertices, normalize(axis_local), args)
    bottom_center_local, top_center_local = model_end_centers(vertices, normalize(axis_local))

    pcd = mesh.sample_points_uniformly(number_of_points=int(args.model_sample_points))
    if args.model_voxel_size > 0:
        pcd = pcd.voxel_down_sample(float(args.model_voxel_size))
    pcd.estimate_normals()

    return MeshModel(
        name=name,
        path=path,
        mesh=mesh,
        pcd=pcd,
        axis_local=normalize(axis_local),
        bbox_center_local=(np.min(vertices, axis=0) + np.max(vertices, axis=0)) / 2.0,
        bottom_center_local=bottom_center_local,
        top_center_local=top_center_local,
        bottom_radius=float(end_radii["bottom_radius"]),
        top_radius=float(end_radii["top_radius"]),
        end_radius_ratio=float(end_radii["end_radius_ratio"]),
        height_axis=height_axis,
        extent=extent,
    )


def load_mesh_models(args) -> list[MeshModel]:
    models = []
    for spec in args.mesh:
        name, path = parse_mesh_spec(spec)
        model = load_mesh_model(name, path, args)
        models.append(model)
    return models


def resolve_table_z(args, workspace=None):
    if args.table_z is not None:
        return float(args.table_z)
    if workspace is not None and "table_z_mean" in workspace:
        return float(workspace["table_z_mean"])
    if args.workspaces and Path(args.workspaces).exists():
        try:
            workspaces, _ = load_workspaces(args.workspaces)
            if args.workspace_name in workspaces and "table_z_mean" in workspaces[args.workspace_name]:
                return float(workspaces[args.workspace_name]["table_z_mean"])
        except Exception:
            return None
    return None


def pca_axis(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    points = np.asarray(points, dtype=np.float64)
    centered = points - np.mean(points, axis=0)
    cov = centered.T @ centered / max(len(points) - 1, 1)
    values, vectors = np.linalg.eigh(cov)
    order = np.argsort(values)[::-1]
    return vectors[:, order[0]], values[order]


def cluster_center(points: np.ndarray) -> np.ndarray:
    min_bound = np.min(points, axis=0)
    max_bound = np.max(points, axis=0)
    return (min_bound + max_bound) / 2.0


def candidate_target_axes(cluster_points: np.ndarray, args) -> list[tuple[str, np.ndarray]]:
    axes: list[tuple[str, np.ndarray]] = []
    if args.fit_states in ("both", "upright"):
        axes.append(("upright", BASE_Z))
        axes.append(("upright", -BASE_Z))

    if args.fit_states in ("both", "lying"):
        major_axis, _ = pca_axis(cluster_points)
        horizontal = major_axis.copy()
        horizontal[2] = 0.0
        horizontal = normalize(horizontal, np.array([1.0, 0.0, 0.0]))
        axes.append(("lying", horizontal))
        axes.append(("lying", -horizontal))
    return axes


def make_initial_transforms(model: MeshModel, cluster_points: np.ndarray, args) -> list[tuple[str, np.ndarray]]:
    target_center = cluster_center(cluster_points)
    transforms = []
    for state_hint, target_axis in candidate_target_axes(cluster_points, args):
        align_R = rotation_align_vectors(model.axis_local, target_axis)
        samples = args.yaw_samples if state_hint == "upright" else args.roll_samples
        samples = max(1, int(samples))
        for idx in range(samples):
            angle = 2.0 * math.pi * idx / samples
            spin_R = rotation_matrix_from_axis_angle(target_axis, angle)
            R = spin_R @ align_R
            t = target_center - R @ model.bbox_center_local
            transforms.append((state_hint, make_transform(R, t)))
    return transforms


def run_icp(model: MeshModel, cluster_pcd: o3d.geometry.PointCloud, init_T: np.ndarray, args):
    criteria = o3d.pipelines.registration.ICPConvergenceCriteria(
        max_iteration=int(args.icp_iterations)
    )
    return o3d.pipelines.registration.registration_icp(
        model.pcd,
        cluster_pcd,
        max_correspondence_distance=float(args.icp_max_distance),
        init=init_T,
        estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(),
        criteria=criteria,
    )


def pose_state_from_axis(axis_base: np.ndarray, args) -> str:
    z_abs = abs(float(axis_base[2]))
    if z_abs >= float(args.upright_axis_z_threshold):
        return "upright_like"
    if z_abs <= float(args.lying_axis_z_threshold):
        return "lying"
    if not args.allow_unknown:
        split = 0.5 * (float(args.upright_axis_z_threshold) + float(args.lying_axis_z_threshold))
        return "upright_like" if z_abs >= split else "lying"
    return "unknown"


def height_prior_penalty(model: MeshModel, args) -> float:
    height = float(model.height_axis)
    low = float(args.cup_height_min)
    high = float(args.cup_height_max)
    if low <= height <= high:
        return 0.0
    scale = max(high - low, 1e-6)
    if height < low:
        return float((low - height) / scale)
    return float((height - high) / scale)


def support_gap_from_table(cluster_points: np.ndarray, table_z: float | None) -> float | None:
    if table_z is None or len(cluster_points) == 0:
        return None
    return float(np.min(cluster_points[:, 2]) - float(table_z))


def support_prior_penalty(support_gap: float | None, args) -> float:
    if support_gap is None:
        return 0.0
    gap = max(0.0, float(support_gap))
    allowed = max(float(args.max_support_gap), 1e-6)
    if gap <= allowed:
        return 0.0
    return float((gap - allowed) / allowed)


def state_extent_prior_penalty(state: str, cluster_points: np.ndarray, args) -> float:
    if len(cluster_points) == 0:
        return 0.0
    z_extent = float(np.ptp(cluster_points[:, 2]))
    if state in ("upright_like", "inverted_upright") and z_extent < float(args.upright_z_extent_min):
        return float((float(args.upright_z_extent_min) - z_extent) / max(float(args.upright_z_extent_min), 1e-6))
    if state == "lying" and z_extent > float(args.lying_z_extent_max):
        return float((z_extent - float(args.lying_z_extent_max)) / max(float(args.lying_z_extent_max), 1e-6))
    return 0.0


def observed_size_prior_penalty(model: MeshModel, state: str, cluster_points: np.ndarray, args):
    if state not in ("upright_like", "inverted_upright") or len(cluster_points) == 0:
        return 0.0, None

    extent_xy = np.ptp(cluster_points[:, :2], axis=0)
    observed_diameter = float(np.max(extent_xy))
    model_name = str(model.name)
    big_name = str(args.big_model_name)
    small_name = str(args.small_model_name)
    big_min = float(args.big_observed_diameter_min)
    small_max = float(args.small_observed_diameter_max)
    gap = max(big_min - small_max, 1e-6)

    penalty = 0.0
    reason = "neutral"
    if model_name == small_name and observed_diameter >= big_min:
        penalty = 1.0 + (observed_diameter - big_min) / gap
        reason = "large_observed_cluster_penalizes_small_model"
    elif model_name == big_name and observed_diameter <= small_max:
        penalty = 1.0 + (small_max - observed_diameter) / gap
        reason = "small_observed_cluster_penalizes_big_model"

    details = {
        "observed_xy_extent": [float(v) for v in extent_xy],
        "observed_xy_diameter": observed_diameter,
        "penalty": float(penalty),
        "reason": reason,
        "big_observed_diameter_min": big_min,
        "small_observed_diameter_max": small_max,
        "big_model_name": big_name,
        "small_model_name": small_name,
    }
    return float(penalty), details


def axis_snap_prior_penalty(state: str, axis_base: np.ndarray, args) -> float:
    z_abs = abs(float(normalize(axis_base, BASE_Z)[2]))
    if state in ("upright_like", "inverted_upright"):
        ideal_min = float(args.upright_axis_z_ideal_min)
        if z_abs >= ideal_min:
            return 0.0
        return float((ideal_min - z_abs) / max(1.0 - ideal_min, 1e-6))
    if state == "lying":
        ideal_max = float(args.lying_axis_z_ideal_max)
        if z_abs <= ideal_max:
            return 0.0
        return float((z_abs - ideal_max) / max(ideal_max, 1e-6))
    return 1.0


def upright_rim_prior(model: MeshModel, cluster_points: np.ndarray, T: np.ndarray, state: str, args) -> dict:
    default = {
        "enabled": False,
        "penalty": 0.0,
        "ratio_penalty": 0.0,
        "absolute_penalty": 0.0,
        "direct_absolute_penalty": 0.0,
        "flipped_absolute_penalty": 0.0,
        "direction_penalty": 0.0,
        "distinct_penalty": 0.0,
        "observed_top_radius": None,
        "observed_bottom_radius": None,
        "observed_radius_ratio": None,
        "observed_top_points": 0,
        "observed_bottom_points": 0,
        "model_top_radius": float(model.top_radius),
        "model_bottom_radius": float(model.bottom_radius),
        "model_radius_ratio": float(model.end_radius_ratio),
        "model_world_top_radius": None,
        "model_world_bottom_radius": None,
        "observed_distinct_rims": None,
        "model_distinct_rims": bool(model.end_radius_ratio >= float(args.distinct_rim_ratio)),
    }
    if state not in ("upright_like", "inverted_upright") or len(cluster_points) == 0:
        return default

    axis_base = normalize(np.asarray(T[:3, :3], dtype=np.float64) @ model.axis_local, BASE_Z)
    model_world_top_radius = float(model.top_radius if axis_base[2] >= 0.0 else model.bottom_radius)
    model_world_bottom_radius = float(model.bottom_radius if axis_base[2] >= 0.0 else model.top_radius)
    if axis_base[2] < 0.0:
        axis_base = -axis_base
    center_base = np.asarray(T[:3, 3], dtype=np.float64) + np.asarray(T[:3, :3], dtype=np.float64) @ model.bbox_center_local
    observed = estimate_end_radii(
        cluster_points,
        axis_base,
        band_fraction=float(args.rim_band_fraction),
        radius_quantile=float(args.rim_radius_quantile),
        axis_point=center_base,
    )
    default.update(
        {
            "enabled": True,
            "observed_top_radius": float(observed["top_radius"]),
            "observed_bottom_radius": float(observed["bottom_radius"]),
            "observed_radius_ratio": float(observed["end_radius_ratio"]),
            "observed_top_points": int(observed["top_points"]),
            "observed_bottom_points": int(observed["bottom_points"]),
            "model_world_top_radius": model_world_top_radius,
            "model_world_bottom_radius": model_world_bottom_radius,
            "observed_distinct_rims": bool(observed["end_radius_ratio"] >= float(args.distinct_rim_ratio)),
        }
    )

    if observed["top_points"] < int(args.min_rim_points) or observed["bottom_points"] < int(args.min_rim_points):
        default["enabled"] = False
        return default

    observed_ratio = max(float(observed["end_radius_ratio"]), 1e-9)
    model_ratio = max(float(model.end_radius_ratio), 1e-9)
    ratio_penalty = abs(math.log(observed_ratio / model_ratio))

    observed_distinct = observed_ratio >= float(args.distinct_rim_ratio)
    model_distinct = model_ratio >= float(args.distinct_rim_ratio)
    distinct_penalty = 0.0 if observed_distinct == model_distinct else 0.5
    observed_top = float(observed["top_radius"])
    observed_bottom = float(observed["bottom_radius"])
    model_top = float(model_world_top_radius)
    model_bottom = float(model_world_bottom_radius)
    scale = max(float(args.rim_absolute_scale), 1e-6)
    direct_abs_penalty = 0.5 * (abs(observed_top - model_top) + abs(observed_bottom - model_bottom)) / scale
    flipped_abs_penalty = 0.5 * (abs(observed_top - model_bottom) + abs(observed_bottom - model_top)) / scale
    absolute_penalty = min(direct_abs_penalty, flipped_abs_penalty)

    direction_penalty = 0.0
    if observed_distinct and model_distinct:
        observed_top_larger = observed_top >= observed_bottom
        model_top_larger = model_top >= model_bottom
        direction_penalty = 0.0 if observed_top_larger == model_top_larger else 0.15

    default["ratio_penalty"] = float(ratio_penalty)
    default["absolute_penalty"] = float(absolute_penalty)
    default["direct_absolute_penalty"] = float(direct_abs_penalty)
    default["flipped_absolute_penalty"] = float(flipped_abs_penalty)
    default["distinct_penalty"] = float(distinct_penalty)
    default["direction_penalty"] = float(direction_penalty)
    default["penalty"] = float(ratio_penalty + distinct_penalty + direction_penalty)
    return default


def refine_upright_state_from_rims(state: str, rim_prior: dict, args) -> str:
    if not args.detect_inverted_upright or state != "upright_like":
        return state
    if not rim_prior or not rim_prior.get("enabled"):
        return state
    if not rim_prior.get("observed_distinct_rims"):
        return state
    top = rim_prior.get("observed_top_radius")
    bottom = rim_prior.get("observed_bottom_radius")
    if top is None or bottom is None:
        return state
    return "inverted_upright" if float(bottom) > float(top) else "upright_like"


def report_axis_direction(axis_base: np.ndarray, state: str, args) -> np.ndarray:
    axis = normalize(axis_base, BASE_Z)
    if args.axis_report == "model":
        return axis
    if args.axis_report == "table_up" and state in ("upright_like", "inverted_upright") and axis[2] < 0.0:
        return -axis
    if args.axis_report == "unoriented":
        idx = int(np.argmax(np.abs(axis)))
        if axis[idx] < 0.0:
            return -axis
    return axis


def classify_signed_cup_orientation(cup_axis_signed_base: np.ndarray, args) -> str:
    z = float(cup_axis_signed_base[2])
    upright_threshold = float(args.upright_axis_z_threshold)
    lying_threshold = float(args.lying_axis_z_threshold)
    if z >= upright_threshold:
        return "mouth_up"
    if z <= -upright_threshold:
        return "mouth_down"
    if abs(z) <= lying_threshold:
        return "lying_signed"
    return "tilted_signed"


def mesh_end_pose_fields(model: MeshModel, T: np.ndarray, args) -> dict:
    R = np.asarray(T[:3, :3], dtype=np.float64)
    t = np.asarray(T[:3, 3], dtype=np.float64)
    bottom_center_base = t + R @ model.bottom_center_local
    top_center_base = t + R @ model.top_center_local
    cup_axis_signed_base = normalize(top_center_base - bottom_center_base, R @ model.axis_local)
    return {
        "bottom_center_base": bottom_center_base,
        "top_center_base": top_center_base,
        "cup_axis_signed_base": cup_axis_signed_base,
        "signed_orientation": classify_signed_cup_orientation(cup_axis_signed_base, args),
    }


def point_cloud_from_points(points: np.ndarray) -> o3d.geometry.PointCloud:
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.asarray(points, dtype=np.float64).reshape(-1, 3))
    return pcd


def transformed_model_points(model: MeshModel, T: np.ndarray) -> np.ndarray:
    points = np.asarray(model.pcd.points)
    R = np.asarray(T[:3, :3], dtype=np.float64)
    t = np.asarray(T[:3, 3], dtype=np.float64)
    return points @ R.T + t


def nearest_distance_stats(source_points: np.ndarray, target_pcd: o3d.geometry.PointCloud, threshold: float):
    tree = o3d.geometry.KDTreeFlann(target_pcd)
    distances = []
    close = 0
    for point in np.asarray(source_points, dtype=np.float64):
        count, idx, dist2 = tree.search_knn_vector_3d(point, 1)
        if count > 0:
            dist = math.sqrt(float(dist2[0]))
            distances.append(dist)
            if dist <= threshold:
                close += 1
    if not distances:
        return 0.0, float("inf")
    return float(close) / float(len(distances)), float(np.mean(distances))


def extent_mismatch(cluster_points: np.ndarray, model_points: np.ndarray) -> float:
    cluster_extent = np.maximum(np.ptp(cluster_points, axis=0), 1e-6)
    model_extent = np.maximum(np.ptp(model_points, axis=0), 1e-6)
    cluster_sorted = np.sort(cluster_extent)
    model_sorted = np.sort(model_extent)
    log_ratio = np.log(model_sorted / cluster_sorted)
    return float(np.mean(np.abs(log_ratio)))


def score_registration(result, cluster_pcd, model: MeshModel, state: str, axis_base, args, table_z=None) -> tuple[float, dict]:
    if result.fitness <= 0.0:
        return -1e9, {
            "cluster_coverage": 0.0,
            "cluster_mean_distance": float("inf"),
            "model_coverage": 0.0,
            "model_mean_distance": float("inf"),
            "extent_mismatch": float("inf"),
            "height_penalty": float("inf"),
            "support_gap": None,
            "support_penalty": 0.0,
            "state_extent_penalty": 0.0,
            "observed_size_prior": None,
            "observed_size_penalty": 0.0,
            "axis_snap_penalty": 1.0,
            "upright_rim_prior": None,
        }

    distance_threshold = float(args.coverage_distance or args.icp_max_distance)
    T = np.asarray(result.transformation, dtype=np.float64)
    model_points = transformed_model_points(model, T)
    cluster_points = np.asarray(cluster_pcd.points)
    model_pcd = point_cloud_from_points(model_points)

    cluster_coverage, cluster_mean_distance = nearest_distance_stats(
        cluster_points,
        model_pcd,
        distance_threshold,
    )
    model_coverage, model_mean_distance = nearest_distance_stats(
        model_points,
        cluster_pcd,
        distance_threshold,
    )
    mismatch = extent_mismatch(cluster_points, model_points)
    height_penalty = height_prior_penalty(model, args)
    support_gap = support_gap_from_table(cluster_points, table_z)
    support_penalty = support_prior_penalty(support_gap, args)
    state_penalty = state_extent_prior_penalty(state, cluster_points, args)
    observed_size_penalty, observed_size_details = observed_size_prior_penalty(model, state, cluster_points, args)
    axis_penalty = axis_snap_prior_penalty(state, axis_base, args)
    rim_prior = upright_rim_prior(model, cluster_points, T, state, args)

    rmse_norm = float(result.inlier_rmse) / max(float(args.icp_max_distance), 1e-9)
    score = (
        float(args.fitness_weight) * float(result.fitness)
        + float(args.coverage_weight) * cluster_coverage
        + 0.10 * model_coverage
        - 0.25 * rmse_norm
        - float(args.extent_weight) * mismatch
        - float(args.height_prior_weight) * height_penalty
        - float(args.support_prior_weight) * support_penalty
        - float(args.state_prior_weight) * state_penalty
        - float(args.observed_size_prior_weight) * observed_size_penalty
        - float(args.axis_snap_prior_weight) * axis_penalty
        - float(args.upright_rim_prior_weight) * float(rim_prior["penalty"])
        - float(args.upright_rim_absolute_weight) * float(rim_prior.get("absolute_penalty", 0.0))
    )
    details = {
        "cluster_coverage": float(cluster_coverage),
        "cluster_mean_distance": float(cluster_mean_distance),
        "model_coverage": float(model_coverage),
        "model_mean_distance": float(model_mean_distance),
        "extent_mismatch": float(mismatch),
        "height_penalty": float(height_penalty),
        "support_gap": None if support_gap is None else float(support_gap),
        "support_penalty": float(support_penalty),
        "state_extent_penalty": float(state_penalty),
        "observed_size_prior": observed_size_details,
        "observed_size_penalty": float(observed_size_penalty),
        "axis_snap_penalty": float(axis_penalty),
        "upright_rim_prior": rim_prior,
    }
    return float(score), details


def fit_cluster_with_meshes(cluster_pcd, models: list[MeshModel], args, table_z=None) -> dict | None:
    cluster_points = np.asarray(cluster_pcd.points)
    if len(cluster_points) < args.min_clean_points:
        return None

    best = None
    best_by_model = {}
    for model in models:
        for state_hint, init_T in make_initial_transforms(model, cluster_points, args):
            try:
                result = run_icp(model, cluster_pcd, init_T, args)
            except RuntimeError:
                continue

            T = np.asarray(result.transformation, dtype=np.float64)
            axis_base = normalize(T[:3, :3] @ model.axis_local, BASE_Z)
            state = pose_state_from_axis(axis_base, args)
            score, score_details = score_registration(
                result,
                cluster_pcd,
                model,
                state,
                axis_base,
                args,
                table_z=table_z,
            )
            state = refine_upright_state_from_rims(
                state,
                score_details.get("upright_rim_prior"),
                args,
            )
            axis_report = report_axis_direction(axis_base, state, args)
            origin_base = T[:3, 3].copy()
            center_base = origin_base + T[:3, :3] @ model.bbox_center_local
            end_pose = mesh_end_pose_fields(model, T, args)
            record = {
                "model": model,
                "model_name": model.name,
                "state_hint": state_hint,
                "state": state,
                "T_base_model": T,
                "origin_base": origin_base,
                "center_base": center_base,
                "axis_base": axis_base,
                "axis_report_base": axis_report,
                "top_center_base": end_pose["top_center_base"],
                "bottom_center_base": end_pose["bottom_center_base"],
                "cup_axis_signed_base": end_pose["cup_axis_signed_base"],
                "signed_orientation": end_pose["signed_orientation"],
                "axis_z_abs": abs(float(axis_base[2])),
                "fitness": float(result.fitness),
                "inlier_rmse": float(result.inlier_rmse),
                "score": float(score),
                "score_details": score_details,
                "accepted": bool(
                    result.fitness >= args.min_fitness
                    and score_details["cluster_coverage"] >= args.min_cluster_coverage
                    and result.inlier_rmse <= args.max_rmse
                    and score_details["support_penalty"] <= 1.0
                    and (not args.hard_axis_snap or score_details["axis_snap_penalty"] <= 1.0)
                ),
            }
            model_best = best_by_model.get(model.name)
            if model_best is None or record["score"] > model_best["score"]:
                best_by_model[model.name] = record
            if best is None or record["score"] > best["score"]:
                best = record
    if best is not None:
        best["model_alternatives"] = best_by_model
    return best


def capture_cluster_pcds(args):
    T_base_marker = load_T_base_marker(args.calibration)
    workspaces, _ = load_workspaces(args.workspaces)
    workspace = workspaces[args.workspace_name]
    table_z = resolve_table_z(args, workspace)

    detector_bundle = create_aruco_detector()
    camera = RealSenseCamera(
        width=args.width,
        height=args.height,
        fps=args.fps,
        stream_order=args.camera_stream_order,
    )

    print("\n========== Real Mesh Cup Pose Test ==========")
    print("Perception only: no robot motion.")
    print(f"MARKER_LENGTH={MARKER_LENGTH:.3f} m, target marker ID={TARGET_MARKER_ID}")
    print(f"Using workspace: {args.workspaces} -> {args.workspace_name}")

    try:
        camera.start(
            start_attempts=args.camera_start_attempts,
            retry_delay_s=args.camera_retry_delay,
            hardware_reset_on_fail=args.camera_hardware_reset_on_fail,
        )

        frames = None
        marker_detection = None
        for attempt in range(120):
            frames = camera.get_aligned_frames()
            if frames is None:
                continue
            marker_detection, corners, ids, rejected = detect_target_marker_pose(
                frames.color_bgr,
                frames.camera_matrix,
                frames.dist_coeffs,
                detector_bundle=detector_bundle,
            )
            if marker_detection is not None:
                break
            if attempt % 20 == 0:
                print(f"Waiting for marker ID {TARGET_MARKER_ID}... attempt {attempt}")

        if frames is None or marker_detection is None:
            raise RuntimeError(f"Could not detect ArUco marker ID {TARGET_MARKER_ID}.")

        valid_mask, edge_mask = make_depth_valid_mask(frames.depth_z16, frames.depth_scale_m, args)
        T_base_camera = T_base_marker @ invert_T(marker_detection.T_camera_marker)

        pcd_camera = build_camera_point_cloud(frames, valid_mask, args)
        pcd_base_full = transform_pcd(pcd_camera, T_base_camera)
        pcd_crop = crop_pcd_by_workspace(pcd_base_full, workspace)
        pcd_no_table, plane_model, inliers = remove_table_plane(pcd_crop, args)
        pcd_no_table, near_table_removed = remove_near_table_residuals(pcd_no_table, plane_model, args)
        pcd_no_table, color_removed = filter_dark_object_points(pcd_no_table, args)

        candidates, cluster_pcds, dbscan_summary = make_candidates(pcd_no_table, args)
        candidates = sort_candidates(candidates, args.pick_sort)

        capture_info = {
            "frames": frames,
            "edge_mask": edge_mask,
            "pcd_crop": pcd_crop,
            "pcd_no_table": pcd_no_table,
            "plane_model": plane_model,
            "near_table_removed": int(near_table_removed),
            "dark_color_removed": int(color_removed),
            "table_z": table_z,
            "dbscan_summary": dbscan_summary,
            "candidates": candidates,
            "cluster_pcds": cluster_pcds,
        }
        return capture_info
    finally:
        camera.stop()


def load_offline_pcds(args):
    if args.input_frame != "base":
        raise ValueError(
            "Offline mesh fitting currently expects base-frame point clouds. "
            "Use files saved by debug_pointclouds/*/*_cluster_*_clean.ply or *_no_table.ply."
        )

    input_paths = []
    for path in args.input_pcd:
        resolved = path if path.is_absolute() else PROJECT_ROOT / path
        if not resolved.exists():
            raise FileNotFoundError(f"Missing input point cloud: {resolved}")
        input_paths.append(resolved)

    print("\n========== Offline Mesh Cup Pose Test ==========")
    print("Perception only: no RealSense and no robot motion.")
    print("Input point clouds are assumed to already be in Kinova base frame.")
    table_z = resolve_table_z(args)
    if table_z is None:
        print("No table_z available. Floating-object prior is disabled. Use --table-z to enable it.")
    else:
        print(f"Using table_z={table_z:.4f} m for floating-object prior.")

    if args.input_as_scene:
        all_points = []
        all_colors = []
        has_all_colors = True
        for path in input_paths:
            pcd = read_point_cloud_compatible(path)
            if pcd.is_empty():
                print(f"Warning: empty point cloud skipped: {path}")
                continue
            points = np.asarray(pcd.points)
            colors = np.asarray(pcd.colors)
            all_points.append(points)
            if colors.shape[0] == points.shape[0]:
                all_colors.append(colors)
            else:
                has_all_colors = False

        scene = o3d.geometry.PointCloud()
        if all_points:
            scene.points = o3d.utility.Vector3dVector(np.vstack(all_points))
            if has_all_colors and all_colors:
                scene.colors = o3d.utility.Vector3dVector(np.vstack(all_colors))
        if args.voxel_size > 0:
            scene = scene.voxel_down_sample(float(args.voxel_size))
        scene, color_removed = filter_dark_object_points(scene, args)

        candidates, cluster_pcds, dbscan_summary = make_candidates(scene, args)
        candidates = sort_candidates(candidates, args.pick_sort)
        return {
            "frames": None,
            "edge_mask": np.zeros((1, 1), dtype=bool),
            "pcd_crop": scene,
            "pcd_no_table": scene,
            "plane_model": None,
            "near_table_removed": 0,
            "dark_color_removed": int(color_removed),
            "table_z": table_z,
            "dbscan_summary": dbscan_summary,
            "candidates": candidates,
            "cluster_pcds": cluster_pcds,
        }

    candidates = []
    cluster_pcds = {}
    total_color_removed = 0
    for label, path in enumerate(input_paths):
        raw_pcd = read_point_cloud_compatible(path)
        if raw_pcd.is_empty():
            print(f"Warning: empty point cloud skipped: {path}")
            continue
        raw_pcd, color_removed = filter_dark_object_points(raw_pcd, args)
        total_color_removed += int(color_removed)
        raw_points = np.asarray(raw_pcd.points)
        raw_count = int(len(raw_points))
        if raw_count < args.min_raw_points:
            print(f"Warning: skipped {path.name}, raw_points={raw_count} < {args.min_raw_points}")
            continue

        if args.skip_input_cleaning:
            clean_pcd = o3d.geometry.PointCloud(raw_pcd)
            clean_report = {"raw_points": raw_count, "clean_points": raw_count, "skipped_input_cleaning": True}
        else:
            clean_pcd, clean_report = clean_cluster_pcd(raw_pcd, args)

        clean_points = np.asarray(clean_pcd.points)
        clean_count = int(len(clean_points))
        if clean_count < args.min_clean_points:
            print(f"Warning: skipped {path.name}, clean_points={clean_count} < {args.min_clean_points}")
            continue

        min_bound = np.min(clean_points, axis=0)
        max_bound = np.max(clean_points, axis=0)
        center = (min_bound + max_bound) / 2.0
        candidate = {
            "label": int(label),
            "source_path": str(path),
            "raw_points": raw_count,
            "clean_points": clean_count,
            "center_base": center,
            "extent": np.ptp(clean_points, axis=0),
            "distance_xy": float(np.linalg.norm(center[:2])),
            "clean_report": clean_report,
        }
        candidates.append(candidate)
        cluster_pcds[int(label)] = {"raw": raw_pcd, "clean": clean_pcd}

    candidates = sort_candidates(candidates, args.pick_sort)
    merged = o3d.geometry.PointCloud()
    if cluster_pcds:
        merged.points = o3d.utility.Vector3dVector(
            np.vstack([np.asarray(pair["clean"].points) for pair in cluster_pcds.values()])
        )

    return {
        "frames": None,
        "edge_mask": np.zeros((1, 1), dtype=bool),
        "pcd_crop": merged,
        "pcd_no_table": merged,
        "plane_model": None,
        "near_table_removed": 0,
        "dark_color_removed": int(total_color_removed),
        "table_z": table_z,
        "dbscan_summary": {
            "total": int(sum(c["clean_points"] for c in candidates)),
            "noise": 0,
            "clusters": len(candidates),
            "offline_direct_clusters": True,
        },
        "candidates": candidates,
        "cluster_pcds": cluster_pcds,
    }


def filter_dark_object_points(pcd: o3d.geometry.PointCloud, args):
    if not args.enable_dark_object_filter:
        return pcd, 0
    points = np.asarray(pcd.points)
    colors = np.asarray(pcd.colors)
    if len(points) == 0 or colors.shape[0] != points.shape[0]:
        return pcd, 0

    rgb = np.clip(colors.astype(np.float64), 0.0, 1.0)
    max_channel = np.max(rgb, axis=1)
    min_channel = np.min(rgb, axis=1)
    value = max_channel
    saturation = np.zeros_like(value)
    nonzero = value > 1e-9
    saturation[nonzero] = (max_channel[nonzero] - min_channel[nonzero]) / value[nonzero]

    keep = (value <= float(args.dark_value_max)) & (saturation >= float(args.dark_saturation_min))
    if args.dark_rgb_max is not None:
        keep &= np.all(rgb <= float(args.dark_rgb_max), axis=1)

    filtered = o3d.geometry.PointCloud()
    filtered.points = o3d.utility.Vector3dVector(points[keep])
    filtered.colors = o3d.utility.Vector3dVector(colors[keep])
    return filtered, int(np.count_nonzero(~keep))


def make_axis_lineset(center: np.ndarray, axis: np.ndarray, length: float, color) -> o3d.geometry.LineSet:
    center = np.asarray(center, dtype=np.float64).reshape(3)
    axis = normalize(axis, BASE_Z)
    half = 0.5 * float(length)
    points = np.vstack([center - half * axis, center + half * axis])
    line = o3d.geometry.LineSet()
    line.points = o3d.utility.Vector3dVector(points)
    line.lines = o3d.utility.Vector2iVector([[0, 1]])
    line.colors = o3d.utility.Vector3dVector([color])
    return line


def transform_mesh(mesh: o3d.geometry.TriangleMesh, T: np.ndarray, color=None):
    out = o3d.geometry.TriangleMesh(mesh)
    out.transform(T)
    if color is not None:
        out.paint_uniform_color(color)
    return out


def parse_force_model_set(text: str | None):
    if not text:
        return []
    return [item.strip() for item in text.split(",") if item.strip()]


def apply_forced_model_assignment(fit_records: list[dict], required_models: list[str]):
    if not required_models:
        return fit_records, None
    if len(fit_records) < len(required_models):
        return fit_records, {
            "enabled": True,
            "applied": False,
            "reason": "not_enough_clusters",
            "required_models": required_models,
            "cluster_count": len(fit_records),
        }

    best_assignment = None

    def backtrack(model_idx, used_indices, assigned, total_score):
        nonlocal best_assignment
        if model_idx >= len(required_models):
            if best_assignment is None or total_score > best_assignment["score"]:
                best_assignment = {
                    "score": float(total_score),
                    "assigned": list(assigned),
                }
            return

        model_name = required_models[model_idx]
        for idx, fit in enumerate(fit_records):
            if idx in used_indices:
                continue
            alternative = fit.get("model_alternatives", {}).get(model_name)
            if alternative is None:
                continue
            backtrack(
                model_idx + 1,
                used_indices | {idx},
                assigned + [(idx, model_name, alternative)],
                total_score + float(alternative["score"]),
            )

    backtrack(0, set(), [], 0.0)
    if best_assignment is None:
        return fit_records, {
            "enabled": True,
            "applied": False,
            "reason": "missing_model_alternatives",
            "required_models": required_models,
            "cluster_count": len(fit_records),
        }

    assigned_indices = {idx for idx, _, _ in best_assignment["assigned"]}
    assigned_by_index = {idx: alternative for idx, _, alternative in best_assignment["assigned"]}
    selected = []
    for idx, fit in enumerate(fit_records):
        if idx in assigned_indices:
            forced = dict(assigned_by_index[idx])
            forced["cluster_pcd"] = fit["cluster_pcd"]
            forced["forced_model_assignment"] = True
            selected.append(forced)

    info = {
        "enabled": True,
        "applied": True,
        "required_models": required_models,
        "selected_labels": [int(fit["label"]) for fit in selected],
        "assignment_score": best_assignment["score"],
    }
    return selected, info


def save_outputs(args, ts, capture_info, results):
    args.output_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = args.output_dir / f"{ts}_mesh_pose_results.json"

    serializable = {
        "timestamp": ts,
        "mode": "mesh_pose_perception_only_no_robot_motion",
        "workspaces": str(args.workspaces),
        "workspace_name": args.workspace_name,
        "mesh_specs": args.mesh,
        "mesh_unit": args.mesh_unit,
        "mesh_scale": args.mesh_scale,
        "mesh_axis": args.mesh_axis,
        "mesh_origin": args.mesh_origin,
        "near_table_removed": capture_info["near_table_removed"],
        "dark_color_removed": capture_info.get("dark_color_removed", 0),
        "table_z": capture_info.get("table_z"),
        "depth_edge_rejected_pixels": int(np.count_nonzero(capture_info["edge_mask"])),
        "dbscan": capture_info["dbscan_summary"],
        "parameters": {
            "cluster_eps": args.cluster_eps,
            "cluster_min_points": args.cluster_min_points,
            "min_raw_points": args.min_raw_points,
            "min_clean_points": args.min_clean_points,
            "force_model_set": args.force_model_set,
            "icp_max_distance": args.icp_max_distance,
            "icp_iterations": args.icp_iterations,
            "yaw_samples": args.yaw_samples,
            "roll_samples": args.roll_samples,
            "coverage_distance": args.coverage_distance,
            "min_cluster_coverage": args.min_cluster_coverage,
            "coverage_weight": args.coverage_weight,
            "fitness_weight": args.fitness_weight,
            "extent_weight": args.extent_weight,
            "observed_size_prior_weight": args.observed_size_prior_weight,
            "big_observed_diameter_min": args.big_observed_diameter_min,
            "small_observed_diameter_max": args.small_observed_diameter_max,
            "big_model_name": args.big_model_name,
            "small_model_name": args.small_model_name,
            "cup_height_min": args.cup_height_min,
            "cup_height_max": args.cup_height_max,
            "height_prior_weight": args.height_prior_weight,
            "max_support_gap": args.max_support_gap,
            "support_prior_weight": args.support_prior_weight,
            "state_prior_weight": args.state_prior_weight,
            "axis_snap_prior_weight": args.axis_snap_prior_weight,
            "upright_axis_z_ideal_min": args.upright_axis_z_ideal_min,
            "lying_axis_z_ideal_max": args.lying_axis_z_ideal_max,
            "hard_axis_snap": args.hard_axis_snap,
            "dark_object_filter": args.enable_dark_object_filter,
            "dark_value_max": args.dark_value_max,
            "dark_saturation_min": args.dark_saturation_min,
            "dark_rgb_max": args.dark_rgb_max,
            "detect_inverted_upright": args.detect_inverted_upright,
            "upright_rim_prior_weight": args.upright_rim_prior_weight,
            "upright_rim_absolute_weight": args.upright_rim_absolute_weight,
            "rim_absolute_scale": args.rim_absolute_scale,
            "rim_band_fraction": args.rim_band_fraction,
            "rim_radius_quantile": args.rim_radius_quantile,
            "min_rim_points": args.min_rim_points,
            "distinct_rim_ratio": args.distinct_rim_ratio,
            "axis_report": args.axis_report,
            "min_fitness": args.min_fitness,
            "max_rmse": args.max_rmse,
        },
        "results": results,
    }
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(serializable, f, indent=2)

    if args.save_debug_pcd:
        write_point_cloud_compatible(args.output_dir / f"{ts}_crop_workspace.ply", capture_info["pcd_crop"])
        write_point_cloud_compatible(args.output_dir / f"{ts}_no_table.ply", capture_info["pcd_no_table"])
        for result in results:
            label = result["label"]
            pair = capture_info["cluster_pcds"][label]
            write_point_cloud_compatible(args.output_dir / f"{ts}_cluster_{label:02d}_raw.ply", pair["raw"])
            write_point_cloud_compatible(args.output_dir / f"{ts}_cluster_{label:02d}_clean.ply", pair["clean"])

    print(f"\nSaved results: {metadata_path}")


def visualize_results(capture_info, fit_records):
    geometries = []
    no_table = o3d.geometry.PointCloud(capture_info["pcd_no_table"])
    no_table.paint_uniform_color([0.55, 0.55, 0.55])
    geometries.append(no_table)

    colors = [
        [0.90, 0.15, 0.12],
        [0.10, 0.55, 0.95],
        [0.10, 0.75, 0.25],
        [0.95, 0.65, 0.10],
        [0.65, 0.25, 0.85],
    ]
    for idx, record in enumerate(fit_records):
        if record is None:
            continue
        color = colors[idx % len(colors)]
        cluster = o3d.geometry.PointCloud(record["cluster_pcd"])
        cluster.paint_uniform_color(color)
        mesh = transform_mesh(record["model"].mesh, record["T_base_model"], color=[0.05, 0.05, 0.05])
        mesh.compute_vertex_normals()
        axis_line = make_axis_lineset(
            record["center_base"],
            record["axis_report_base"],
            max(record["model"].height_axis, 0.08),
            color,
        )
        geometries.extend([cluster, mesh, axis_line])

    o3d.visualization.draw_geometries(geometries)


def fit_to_json_record(fit: dict):
    candidate = fit["candidate"]
    return {
        "label": int(fit["label"]),
        "model_name": fit["model_name"],
        "state_hint": fit["state_hint"],
        "state": fit["state"],
        "accepted": fit["accepted"],
        "forced_model_assignment": bool(fit.get("forced_model_assignment", False)),
        "score": float(fit["score"]),
        "fitness": float(fit["fitness"]),
        "inlier_rmse": float(fit["inlier_rmse"]),
        "score_details": fit["score_details"],
        "origin_base": [float(v) for v in fit["origin_base"]],
        "center_base": [float(v) for v in fit["center_base"]],
        "axis_base": [float(v) for v in fit["axis_base"]],
        "axis_report_base": [float(v) for v in fit["axis_report_base"]],
        "top_center_base": [float(v) for v in fit["top_center_base"]],
        "bottom_center_base": [float(v) for v in fit["bottom_center_base"]],
        "cup_axis_signed_base": [float(v) for v in fit["cup_axis_signed_base"]],
        "signed_orientation": fit["signed_orientation"],
        "axis_z_abs": float(fit["axis_z_abs"]),
        "T_base_model": np.asarray(fit["T_base_model"]).tolist(),
        "raw_points": int(candidate["raw_points"]),
        "clean_points": int(candidate["clean_points"]),
        "cluster_center_base": [float(v) for v in candidate["center_base"]],
        "cluster_extent": [float(v) for v in candidate["extent"]],
    }


def print_forced_assignment(fit_records, assignment_info):
    if assignment_info is None or not assignment_info.get("enabled"):
        return
    print("\nForced model assignment:")
    if not assignment_info.get("applied"):
        print(f"  not applied: {assignment_info.get('reason')}")
        print(f"  required_models={assignment_info.get('required_models')}")
        print(f"  cluster_count={assignment_info.get('cluster_count')}")
        return
    print(f"  required_models={assignment_info['required_models']}")
    print(f"  assignment_score={assignment_info['assignment_score']:.4f}")
    for fit in fit_records:
        center = fit["center_base"]
        print(
            f"  label={int(fit['label']):02d}, forced_model={fit['model_name']}, "
            f"state={fit['state']}, score={fit['score']:.4f}, "
            f"fitness={fit['fitness']:.4f}, center=[{center[0]:.4f}, {center[1]:.4f}, {center[2]:.4f}]"
        )


def main():
    args = parse_args()
    models = load_mesh_models(args)

    print("Loaded mesh models:")
    for model in models:
        print(
            f"  {model.name}: {model.path}, sampled={len(model.pcd.points)}, "
            f"axis_height={model.height_axis:.4f} m, extent={[round(float(v), 4) for v in model.extent]}, "
            f"bbox_center_local={[round(float(v), 4) for v in model.bbox_center_local]}, "
            f"top_radius={model.top_radius:.4f}, bottom_radius={model.bottom_radius:.4f}, "
            f"end_radius_ratio={model.end_radius_ratio:.3f}"
        )

    if args.input_pcd:
        capture_info = load_offline_pcds(args)
    else:
        capture_info = capture_cluster_pcds(args)
    candidates = capture_info["candidates"]
    cluster_pcds = capture_info["cluster_pcds"]

    print("\nPerception summary:")
    print(f"  workspace points: {len(capture_info['pcd_crop'].points)}")
    print(f"  no-table points: {len(capture_info['pcd_no_table'].points)}")
    print(f"  near-table removed: {capture_info['near_table_removed']}")
    print(f"  dark-color removed: {capture_info.get('dark_color_removed', 0)}")
    if capture_info.get("table_z") is not None:
        print(f"  table_z: {capture_info['table_z']:.4f}")
    print(f"  DBSCAN: {capture_info['dbscan_summary']}")

    results_json = []
    fit_records = []
    print("\nMesh pose results:")
    if not candidates:
        print("  no clusters passed point-count filters")

    for candidate in candidates:
        label = int(candidate["label"])
        clean_pcd = cluster_pcds[label]["clean"]
        fit = fit_cluster_with_meshes(clean_pcd, models, args, table_z=capture_info.get("table_z"))
        if fit is None:
            print(f"  label={label:02d}: not enough clean points")
            continue

        fit["label"] = label
        fit["candidate"] = candidate
        fit["cluster_pcd"] = clean_pcd
        for alternative in fit.get("model_alternatives", {}).values():
            alternative["label"] = label
            alternative["candidate"] = candidate
            alternative["cluster_pcd"] = clean_pcd
        fit_records.append(fit)

        center = fit["center_base"]
        axis = fit["axis_report_base"]
        extent = [float(v) for v in candidate["extent"]]
        details = fit["score_details"]
        print(
            f"  label={label:02d}, model={fit['model_name']}, state={fit['state']}, "
            f"accepted={fit['accepted']}, score={fit['score']:.4f}, "
            f"fitness={fit['fitness']:.4f}, rmse={fit['inlier_rmse']:.4f}"
        )
        print(
            f"    cluster_coverage={details['cluster_coverage']:.4f}, "
            f"model_coverage={details['model_coverage']:.4f}, "
            f"extent_mismatch={details['extent_mismatch']:.4f}"
        )
        support_gap_text = "None" if details["support_gap"] is None else f"{details['support_gap']:.4f}"
        print(
            f"    height_penalty={details['height_penalty']:.4f}, "
            f"support_gap={support_gap_text}, support_penalty={details['support_penalty']:.4f}, "
            f"state_extent_penalty={details['state_extent_penalty']:.4f}, "
            f"observed_size_penalty={details['observed_size_penalty']:.4f}, "
            f"axis_snap_penalty={details['axis_snap_penalty']:.4f}"
        )
        size_prior = details.get("observed_size_prior")
        if size_prior is not None:
            print(
                f"    observed_size: xy_extent={[round(float(v), 4) for v in size_prior['observed_xy_extent']]}, "
                f"xy_diameter={size_prior['observed_xy_diameter']:.4f}, "
                f"reason={size_prior['reason']}"
            )
        rim = details.get("upright_rim_prior")
        if rim is not None and rim.get("enabled"):
            print(
                f"    upright_rim: penalty={rim['penalty']:.4f}, "
                f"observed_top={rim['observed_top_radius']:.4f}, "
                f"observed_bottom={rim['observed_bottom_radius']:.4f}, "
                f"observed_ratio={rim['observed_radius_ratio']:.3f}, "
                f"model_ratio={rim['model_radius_ratio']:.3f}, "
                f"distinct={rim['observed_distinct_rims']}, "
                f"ratio_penalty={rim['ratio_penalty']:.4f}, "
                f"absolute_penalty={rim['absolute_penalty']:.4f}, "
                f"direction_penalty={rim['direction_penalty']:.4f}"
            )
        elif rim is not None and rim.get("observed_radius_ratio") is not None:
            print(
                f"    upright_rim: disabled, "
                f"top_points={rim['observed_top_points']}, bottom_points={rim['observed_bottom_points']}, "
                f"observed_ratio={rim['observed_radius_ratio']:.3f}, model_ratio={rim['model_radius_ratio']:.3f}"
            )
        print(
            f"    center_base=[{center[0]:.4f}, {center[1]:.4f}, {center[2]:.4f}], "
            f"axis_report_base=[{axis[0]:.4f}, {axis[1]:.4f}, {axis[2]:.4f}], "
            f"axis_z_abs={fit['axis_z_abs']:.4f}"
        )
        top = fit["top_center_base"]
        bottom = fit["bottom_center_base"]
        signed_axis = fit["cup_axis_signed_base"]
        print(
            f"    top_center_base=[{top[0]:.4f}, {top[1]:.4f}, {top[2]:.4f}], "
            f"bottom_center_base=[{bottom[0]:.4f}, {bottom[1]:.4f}, {bottom[2]:.4f}]"
        )
        print(
            f"    cup_axis_signed_base=[{signed_axis[0]:.4f}, {signed_axis[1]:.4f}, {signed_axis[2]:.4f}], "
            f"signed_orientation={fit['signed_orientation']}"
        )
        origin = fit["origin_base"]
        print(f"    origin_base=[{origin[0]:.4f}, {origin[1]:.4f}, {origin[2]:.4f}]")
        print(
            f"    raw_points={candidate['raw_points']}, clean_points={candidate['clean_points']}, "
            f"cluster_extent={[round(v, 4) for v in extent]}"
        )

        results_json.append(fit_to_json_record(fit))

    required_models = parse_force_model_set(args.force_model_set)
    assignment_info = None
    if required_models:
        fit_records, assignment_info = apply_forced_model_assignment(fit_records, required_models)
        print_forced_assignment(fit_records, assignment_info)
        if assignment_info and assignment_info.get("applied"):
            results_json = [fit_to_json_record(fit) for fit in fit_records]

    ts = time.strftime("%Y%m%d_%H%M%S")
    save_outputs(args, ts, capture_info, results_json)

    if args.visualize:
        visualize_results(capture_info, fit_records)


if __name__ == "__main__":
    main()

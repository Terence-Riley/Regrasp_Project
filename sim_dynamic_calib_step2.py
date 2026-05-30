import json
import math
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import open3d as o3d
import pybullet as p
import pybullet_data


# =========================
# User settings
# =========================

ROBOT_URDF_PATH = "kortex_description/robots/gen3_lite.urdf"
SAVE_ROOT = Path("sim_captures")

WIDTH = 640
HEIGHT = 480
FOV_DEG = 60.0
NEAR = 0.01
FAR = 2.0
DEPTH_TRUNC_M = 3.0  # 生成点云时忽略 3 m 以外的背景点

# 仿真里假设 PyBullet world frame = Kinova robot base frame
# marker 和 cup 的位置都在 base/world 坐标系下定义，单位 m
MARKER_SIZE_M = 0.096  # 和你的真实 ArUco marker 边长保持一致：9.6 cm
MARKER_POS_BASE = np.array([0.30, -0.15, 0.002], dtype=np.float64)
MARKER_YAW_DEG = 0.0

CUP_RADIUS_M = 0.035
CUP_HEIGHT_M = 0.10
CUP_POS_BASE = np.array([0.45, 0.10, CUP_HEIGHT_M / 2.0], dtype=np.float64)

# 虚拟相机摆放
CAMERA_EYE = [0.50, -0.55, 0.45]
CAMERA_TARGET = [0.35, 0.00, 0.03]
CAMERA_UP = [0, 0, 1]


# =========================
# Transform utilities
# =========================

def yaw_to_quat(yaw_deg):
    return p.getQuaternionFromEuler([0.0, 0.0, math.radians(yaw_deg)])


def pose_to_T(pos, quat):
    """
    PyBullet position + quaternion -> 4x4 transform.
    T_A_B means frame B expressed in frame A.
    """
    T = np.eye(4, dtype=np.float64)
    R = np.array(p.getMatrixFromQuaternion(quat), dtype=np.float64).reshape(3, 3)
    T[:3, :3] = R
    T[:3, 3] = np.array(pos, dtype=np.float64)
    return T


def get_body_T_base(body_id):
    pos, quat = p.getBasePositionAndOrientation(body_id)
    return pose_to_T(pos, quat)


def invert_T(T):
    return np.linalg.inv(T)


def transform_camera_object_to_base(T_base_marker, T_camera_marker, T_camera_object):
    """
    Dynamic calibration formula:
    T_base_object = T_base_marker @ inv(T_camera_marker) @ T_camera_object
    """
    return T_base_marker @ invert_T(T_camera_marker) @ T_camera_object


# =========================
# Camera utilities
# =========================

def get_camera_intrinsics(width, height, fov_deg):
    """
    Build a simple pinhole camera matrix from vertical FOV.
    PyBullet computeProjectionMatrixFOV uses vertical FOV.
    """
    fov_rad = math.radians(fov_deg)
    fy = height / (2.0 * math.tan(fov_rad / 2.0))
    fx = fy
    cx = width / 2.0
    cy = height / 2.0

    K = np.array(
        [
            [fx, 0.0, cx],
            [0.0, fy, cy],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    return K


def get_virtual_camera(width=WIDTH, height=HEIGHT, fov_deg=FOV_DEG, near=NEAR, far=FAR):
    """
    Return view/projection matrices and camera transforms.

    PyBullet/OpenGL camera convention:
        x right, y up, z backward
    OpenCV/RealSense-style camera convention used in this project:
        x right, y down, z forward
    """
    view_matrix = p.computeViewMatrix(
        cameraEyePosition=CAMERA_EYE,
        cameraTargetPosition=CAMERA_TARGET,
        cameraUpVector=CAMERA_UP,
    )

    projection_matrix = p.computeProjectionMatrixFOV(
        fov=fov_deg,
        aspect=width / height,
        nearVal=near,
        farVal=far,
    )

    # PyBullet view matrix: base/world -> OpenGL camera
    T_gl_base = np.array(view_matrix, dtype=np.float64).reshape(4, 4, order="F")

    # OpenGL camera -> OpenCV camera
    T_cv_gl = np.diag([1.0, -1.0, -1.0, 1.0])

    # base/world -> OpenCV camera
    T_camera_base = T_cv_gl @ T_gl_base

    # OpenCV camera -> base/world
    T_base_camera = invert_T(T_camera_base)

    K = get_camera_intrinsics(width, height, fov_deg)

    return view_matrix, projection_matrix, T_base_camera, T_camera_base, K


def depth_buffer_to_meters(depth_buffer, near=NEAR, far=FAR):
    """
    Convert PyBullet/OpenGL depth buffer to metric depth in meters.
    """
    depth_buffer = np.asarray(depth_buffer, dtype=np.float64)
    return far * near / (far - (far - near) * depth_buffer)


def deproject_pixel_to_camera(K, u, v, depth_m):
    """
    Pixel + depth -> 3D point in OpenCV camera frame.
    x right, y down, z forward.
    """
    fx = K[0, 0]
    fy = K[1, 1]
    cx = K[0, 2]
    cy = K[1, 2]

    z = float(depth_m)
    x = (float(u) - cx) * z / fx
    y = (float(v) - cy) * z / fy
    return np.array([x, y, z], dtype=np.float64)


def project_point_camera_to_pixel(K, point_camera):
    """
    3D point in camera frame -> image pixel.
    """
    x, y, z = point_camera.reshape(3)
    if z <= 0:
        return None

    u = K[0, 0] * x / z + K[0, 2]
    v = K[1, 1] * y / z + K[1, 2]
    return np.array([u, v], dtype=np.float64)


# =========================
# Object creation
# =========================

def create_marker(position, yaw_deg=0.0):
    """
    A thin square box as a simulated ArUco marker.
    We use PyBullet ground truth for T_camera_marker in this simulation.
    """
    half = MARKER_SIZE_M / 2.0
    thickness = 0.002

    collision_shape = p.createCollisionShape(
        shapeType=p.GEOM_BOX,
        halfExtents=[half, half, thickness / 2.0],
    )
    visual_shape = p.createVisualShape(
        shapeType=p.GEOM_BOX,
        halfExtents=[half, half, thickness / 2.0],
        rgbaColor=[0.0, 0.0, 0.0, 1.0],
    )

    marker_id = p.createMultiBody(
        baseMass=0.0,
        baseCollisionShapeIndex=collision_shape,
        baseVisualShapeIndex=visual_shape,
        basePosition=position.tolist(),
        baseOrientation=yaw_to_quat(yaw_deg),
    )
    return marker_id


def create_cup(position):
    """
    Use a simple cylinder as the virtual cup.
    """
    collision_shape = p.createCollisionShape(
        shapeType=p.GEOM_CYLINDER,
        radius=CUP_RADIUS_M,
        height=CUP_HEIGHT_M,
    )
    visual_shape = p.createVisualShape(
        shapeType=p.GEOM_CYLINDER,
        radius=CUP_RADIUS_M,
        length=CUP_HEIGHT_M,
        rgbaColor=[0.9, 0.9, 0.9, 1.0],
    )

    cup_id = p.createMultiBody(
        baseMass=0.1,
        baseCollisionShapeIndex=collision_shape,
        baseVisualShapeIndex=visual_shape,
        basePosition=position.tolist(),
        baseOrientation=p.getQuaternionFromEuler([0.0, 0.0, 0.0]),
    )
    return cup_id


def draw_base_axes(length=0.25):
    origin = [0, 0, 0]
    p.addUserDebugLine(origin, [length, 0, 0], [1, 0, 0], lineWidth=4)
    p.addUserDebugLine(origin, [0, length, 0], [0, 1, 0], lineWidth=4)
    p.addUserDebugLine(origin, [0, 0, length], [0, 0, 1], lineWidth=4)
    p.addUserDebugText("+X", [length, 0, 0], textColorRGB=[1, 0, 0])
    p.addUserDebugText("+Y", [0, length, 0], textColorRGB=[0, 1, 0])
    p.addUserDebugText("+Z", [0, 0, length], textColorRGB=[0, 0, 1])


# =========================
# RGB-D / point cloud utilities
# =========================

def render_rgbd(view_matrix, projection_matrix, width=WIDTH, height=HEIGHT):
    img = p.getCameraImage(
        width=width,
        height=height,
        viewMatrix=view_matrix,
        projectionMatrix=projection_matrix,
        renderer=p.ER_BULLET_HARDWARE_OPENGL,
    )

    rgba = np.reshape(img[2], (height, width, 4)).astype(np.uint8)
    rgb = rgba[:, :, :3]

    depth_buffer = np.reshape(img[3], (height, width))
    depth_m = depth_buffer_to_meters(depth_buffer)

    seg = np.reshape(img[4], (height, width))

    return rgb, depth_m, seg


def create_point_cloud_from_rgbd(rgb, depth_m, K, depth_trunc=DEPTH_TRUNC_M):
    """
    Create Open3D point cloud in OpenCV camera frame.
    This point cloud is NOT flipped and should be used for algorithms.
    """
    mask = np.isfinite(depth_m) & (depth_m > 0) & (depth_m < depth_trunc)

    v_coords, u_coords = np.where(mask)
    z = depth_m[v_coords, u_coords]
    x = (u_coords - K[0, 2]) * z / K[0, 0]
    y = (v_coords - K[1, 2]) * z / K[1, 1]

    points = np.stack([x, y, z], axis=1)
    colors = rgb[v_coords, u_coords].astype(np.float64) / 255.0

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    return pcd


def make_visualization_copy(pcd_camera):
    """
    Visualization-friendly copy for Open3D viewer.
    Do not use this for coordinate calculation.
    """
    pcd_vis = o3d.geometry.PointCloud(pcd_camera)
    pcd_vis.transform(
        np.array(
            [
                [1, 0, 0, 0],
                [0, -1, 0, 0],
                [0, 0, -1, 0],
                [0, 0, 0, 1],
            ],
            dtype=np.float64,
        )
    )
    return pcd_vis


def get_cup_pixel_from_segmentation(seg, cup_id):
    """
    Use PyBullet segmentation mask to find a pixel on the cup.
    This simulates a very simple object detector.
    """
    mask = seg == cup_id
    if not np.any(mask):
        return None, None

    vs, us = np.where(mask)
    u = int(np.round(np.median(us)))
    v = int(np.round(np.median(vs)))
    return (u, v), mask


def median_depth_around(depth_m, u, v, window=9):
    h, w = depth_m.shape
    half = window // 2
    u0 = max(0, u - half)
    u1 = min(w, u + half + 1)
    v0 = max(0, v - half)
    v1 = min(h, v + half + 1)

    patch = depth_m[v0:v1, u0:u1]
    valid = patch[np.isfinite(patch) & (patch > 0) & (patch < FAR)]
    if valid.size == 0:
        return None
    return float(np.median(valid))


def create_T_camera_object_from_pixel(K, depth_m, u, v):
    """
    Single-pixel version: this usually gives a point on the visible cup surface,
    not the true cup center. Keep it for debugging only.
    """
    depth = median_depth_around(depth_m, u, v, window=11)
    if depth is None:
        return None, None, None

    point_camera = deproject_pixel_to_camera(K, u, v, depth)
    T_camera_object = np.eye(4, dtype=np.float64)
    T_camera_object[:3, 3] = point_camera
    return T_camera_object, point_camera, depth


def create_T_camera_object_from_mask_aabb(K, depth_m, mask):
    """
    Better simulation version:
    Use the full visible cup segmentation mask, deproject all visible cup pixels,
    then estimate the object position as the center of the 3D axis-aligned bounding box.

    This is still an approximation because RGB-D only sees the visible surface,
    but it is much better than a single clicked pixel.
    """
    if mask is None or not np.any(mask):
        return None, None, None

    vs, us = np.where(mask)
    zs = depth_m[vs, us]
    valid = np.isfinite(zs) & (zs > 0) & (zs < FAR)

    if np.count_nonzero(valid) < 20:
        return None, None, None

    us = us[valid]
    vs = vs[valid]
    zs = zs[valid]

    xs = (us.astype(np.float64) - K[0, 2]) * zs / K[0, 0]
    ys = (vs.astype(np.float64) - K[1, 2]) * zs / K[1, 1]
    points = np.stack([xs, ys, zs], axis=1)

    # Remove a few extreme points caused by depth edges / segmentation edges.
    lower = np.percentile(points, 5, axis=0)
    upper = np.percentile(points, 95, axis=0)
    inlier_mask = np.all((points >= lower) & (points <= upper), axis=1)
    inlier_points = points[inlier_mask]

    if inlier_points.shape[0] < 20:
        inlier_points = points

    min_bound = np.min(inlier_points, axis=0)
    max_bound = np.max(inlier_points, axis=0)
    center_camera = (min_bound + max_bound) / 2.0

    T_camera_object = np.eye(4, dtype=np.float64)
    T_camera_object[:3, 3] = center_camera

    return T_camera_object, center_camera, inlier_points


# =========================
# Save utilities
# =========================

def save_outputs(
    rgb,
    depth_m,
    seg,
    pcd_camera,
    pcd_vis,
    T_base_marker,
    T_camera_marker,
    T_camera_object,
    T_base_object_est,
    T_base_object_gt,
    cup_pixel,
    cup_point_camera,
    position_error,
    K,
):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = SAVE_ROOT / f"sim_dynamic_calib_{timestamp}"
    save_dir.mkdir(parents=True, exist_ok=True)

    rgb_bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(save_dir / "sim_rgb.png"), rgb_bgr)

    depth_mm = np.clip(depth_m * 1000.0, 0, 65535).astype(np.uint16)
    cv2.imwrite(str(save_dir / "sim_depth_mm.png"), depth_mm)

    depth_colormap = cv2.applyColorMap(
        cv2.convertScaleAbs(depth_mm, alpha=0.03), cv2.COLORMAP_JET
    )
    cv2.imwrite(str(save_dir / "sim_depth_colormap.png"), depth_colormap)

    o3d.io.write_point_cloud(str(save_dir / "sim_scene_camera_frame.ply"), pcd_camera)
    o3d.io.write_point_cloud(str(save_dir / "sim_scene_visualization_flipped.ply"), pcd_vis)

    np.save(str(save_dir / "T_base_marker.npy"), T_base_marker)
    np.save(str(save_dir / "T_camera_marker.npy"), T_camera_marker)
    np.save(str(save_dir / "T_camera_object.npy"), T_camera_object)
    np.save(str(save_dir / "T_base_object_est.npy"), T_base_object_est)
    np.save(str(save_dir / "T_base_object_gt.npy"), T_base_object_gt)
    np.save(str(save_dir / "camera_matrix.npy"), K)

    metadata = {
        "description": "PyBullet dynamic calibration step2 result",
        "camera_convention": "OpenCV/RealSense style: x right, y down, z forward",
        "world_frame": "PyBullet world is treated as Kinova base frame",
        "marker_size_m": MARKER_SIZE_M,
        "marker_position_base_m": MARKER_POS_BASE.tolist(),
        "marker_yaw_deg": MARKER_YAW_DEG,
        "cup_position_base_gt_m": T_base_object_gt[:3, 3].tolist(),
        "cup_position_base_est_m": T_base_object_est[:3, 3].tolist(),
        "position_error_m": float(position_error),
        "cup_pixel_uv": None if cup_pixel is None else [int(cup_pixel[0]), int(cup_pixel[1])],
        "cup_point_camera_m": None if cup_point_camera is None else cup_point_camera.tolist(),
        "camera_eye": CAMERA_EYE,
        "camera_target": CAMERA_TARGET,
        "camera_up": CAMERA_UP,
        "width": WIDTH,
        "height": HEIGHT,
        "fov_deg": FOV_DEG,
        "near": NEAR,
        "far": FAR,
    }

    with open(save_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print(f"\n✅ Saved simulation outputs to: {save_dir}")
    print("   - sim_scene_camera_frame.ply: algorithm/camera-frame point cloud")
    print("   - sim_scene_visualization_flipped.ply: Open3D-friendly view")
    print("   - T_base_object_est.npy: dynamic-calibration result")
    print("   - T_base_object_gt.npy: PyBullet ground truth")


# =========================
# Main
# =========================

def main():
    SAVE_ROOT.mkdir(parents=True, exist_ok=True)

    physics_client = p.connect(p.GUI)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -9.81)

    p.loadURDF("plane.urdf")

    print(f"Loading robot: {ROBOT_URDF_PATH}")
    try:
        robot_id = p.loadURDF(
            ROBOT_URDF_PATH,
            basePosition=[0, 0, 0],
            baseOrientation=p.getQuaternionFromEuler([0, 0, 0]),
            useFixedBase=True,
        )
        print("Robot loaded.")
    except Exception as e:
        robot_id = None
        print("⚠️ Robot URDF failed to load. Continuing with marker/cup/camera only.")
        print("Error:", e)

    draw_base_axes()

    marker_id = create_marker(MARKER_POS_BASE, MARKER_YAW_DEG)
    cup_id = create_cup(CUP_POS_BASE)

    print("Marker ID:", marker_id)
    print("Cup ID:", cup_id)

    # Let simulation settle
    for _ in range(120):
        p.stepSimulation()
        time.sleep(1.0 / 240.0)

    view_matrix, projection_matrix, T_base_camera, T_camera_base, K = get_virtual_camera()

    # Render RGB-D
    rgb, depth_m, seg = render_rgbd(view_matrix, projection_matrix)

    # Ground-truth transforms in base/world frame
    T_base_marker = get_body_T_base(marker_id)
    T_base_object_gt = get_body_T_base(cup_id)

    # Simulated ArUco output:
    # In the real system, OpenCV gives T_camera_marker.
    # In simulation, we get the exact equivalent from PyBullet ground truth.
    T_camera_marker = T_camera_base @ T_base_marker

    # Simulated object detection:
    # Use segmentation to find a pixel on the cup, then use depth + intrinsics to recover T_camera_object.
    cup_pixel, cup_mask = get_cup_pixel_from_segmentation(seg, cup_id)
    if cup_pixel is None:
        print("❌ Cup not visible in the virtual camera. Adjust CAMERA_EYE / CAMERA_TARGET.")
        input("Press Enter to exit...")
        p.disconnect()
        return

    u, v = cup_pixel
    # Single-pixel estimate: useful for debugging, but it is usually a visible surface point.
    T_camera_object_pixel, cup_point_camera_pixel, cup_depth = create_T_camera_object_from_pixel(K, depth_m, u, v)

    if T_camera_object_pixel is None:
        print("❌ Could not get valid depth at cup pixel.")
        input("Press Enter to exit...")
        p.disconnect()
        return

    # Better estimate: use the full cup segmentation mask and estimate a 3D bounding-box center.
    T_camera_object, cup_point_camera, cup_mask_points = create_T_camera_object_from_mask_aabb(K, depth_m, cup_mask)

    if T_camera_object is None:
        print("⚠️ Mask-based estimate failed. Falling back to single-pixel estimate.")
        T_camera_object = T_camera_object_pixel
        cup_point_camera = cup_point_camera_pixel

    # Dynamic calibration formula
    T_base_object_est = transform_camera_object_to_base(
        T_base_marker,
        T_camera_marker,
        T_camera_object,
    )

    position_error = np.linalg.norm(
        T_base_object_est[:3, 3] - T_base_object_gt[:3, 3]
    )

    print("\n========== Simulation Dynamic Calibration Step 2 ==========")
    print("PyBullet world frame is treated as robot base frame.")
    print("\nT_base_marker:")
    print(T_base_marker)
    print("\nT_camera_marker simulated from ground truth:")
    print(T_camera_marker)
    print("\nSelected cup pixel (u, v):", cup_pixel)
    print(f"Depth at cup pixel: {cup_depth:.4f} m")
    print("Cup point in camera frame [x right, y down, z forward] m:")
    print(cup_point_camera)
    print("\nT_camera_object from virtual RGB-D:")
    print(T_camera_object)
    print("\nGround-truth cup position in base frame:")
    print(T_base_object_gt[:3, 3])
    print("\nEstimated cup position in base frame:")
    print(T_base_object_est[:3, 3])
    print(f"\nPosition error: {position_error:.6f} m")

    if position_error < 0.01:
        print("✅ Success: dynamic calibration chain works in simulation.")
    else:
        print("⚠️ Error is larger than 1 cm. This may happen because the selected pixel is on the visible cup surface, not the exact cup center.")
        print("   It is still useful, but you can improve it later with segmentation center / point cloud clustering.")

    # Visual debug image
    debug_rgb = rgb.copy()
    cv2.circle(debug_rgb, (u, v), 6, (255, 0, 0), -1)  # RGB red dot
    cv2.putText(
        debug_rgb,
        f"cup pixel ({u},{v})",
        (u + 8, max(20, v - 8)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 0, 0),
        2,
    )

    pcd_camera = create_point_cloud_from_rgbd(rgb, depth_m, K)
    pcd_vis = make_visualization_copy(pcd_camera)

    save_outputs(
        debug_rgb,
        depth_m,
        seg,
        pcd_camera,
        pcd_vis,
        T_base_marker,
        T_camera_marker,
        T_camera_object,
        T_base_object_est,
        T_base_object_gt,
        cup_pixel,
        cup_point_camera,
        position_error,
        K,
    )

    print("\nKeeping PyBullet GUI open. Press Ctrl+C in terminal to exit.")
    try:
        while True:
            p.stepSimulation()
            time.sleep(1.0 / 240.0)
    except KeyboardInterrupt:
        pass
    finally:
        p.disconnect()
        print("Simulation closed.")


if __name__ == "__main__":
    main()

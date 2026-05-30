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


# ============================================================
# User settings
# ============================================================

ROBOT_URDF_PATH = "kortex_description/robots/gen3_lite.urdf"
SAVE_ROOT = Path("sim_pick_place_outputs")

WIDTH = 640
HEIGHT = 480
FOV_DEG = 60.0
NEAR = 0.01
FAR = 2.0
DEPTH_TRUNC_M = 3.0

# PyBullet world frame = Kinova base frame
MARKER_SIZE_M = 0.096  # 你的真实 ArUco marker 黑色外边框边长：9.6 cm
MARKER_POS_BASE = np.array([0.30, -0.15, 0.002], dtype=np.float64)
MARKER_YAW_DEG = 0.0

CUP_RADIUS_M = 0.035
CUP_HEIGHT_M = 0.10
CUP_POS_BASE = np.array([0.45, 0.10, CUP_HEIGHT_M / 2.0], dtype=np.float64)

# 放置目标区域：你可以改成桌面上的其他位置
PLACE_POS_BASE = np.array([0.32, 0.28, CUP_HEIGHT_M / 2.0], dtype=np.float64)

# 虚拟相机位置
CAMERA_EYE = [0.50, -0.55, 0.45]
CAMERA_TARGET = [0.35, 0.00, 0.03]
CAMERA_UP = [0, 0, 1]

# ============================================================
# Motion parameters
# ============================================================

# 这里是 pipeline 验证版：目标是先跑通 perception -> motion -> attach -> place。
# cup_pos_est 是杯子中心，不是杯口顶部。
# 如果杯高 0.10 m，杯子中心 z≈0.05 m，杯口高度 z≈0.10 m。
# 所以 GRASP_HEIGHT_ABOVE_CUP=0.07 时，末端目标 z≈0.12 m，接近杯口上方一点。
PRE_GRASP_HEIGHT_ABOVE_CUP = 0.22
GRASP_HEIGHT_ABOVE_CUP = 0.07
PRE_PLACE_HEIGHT_ABOVE_TABLE = 0.22
PLACE_HEIGHT_ABOVE_TABLE = 0.07
RETREAT_HEIGHT_ABOVE_TABLE = 0.25

# 更慢、更软，避免 PyBullet position control 冲过头
MOVE_STEPS = 1200
SETTLE_STEPS = 160
POSITION_GAIN = 0.02
MAX_FORCE = 25
MAX_VELOCITY = 0.25

# 为了避免“流程验证阶段”机器人还没 attach 就把杯子撞倒，默认关闭 robot-cup 碰撞。
# 这不是物理真实抓取，只是先验证系统链路。
# 后面做真实接触抓取时，可以改成 False。
DISABLE_ROBOT_CUP_COLLISION_FOR_PIPELINE = True

# 你前面已经验证成功的末端 link index
END_EFFECTOR_LINK_INDEX_OVERRIDE = 7

# gripper 可选参数：如果脚本识别到 gripper/finger joints，会尝试开合；识别不到也没关系
GRIPPER_OPEN_VALUE = 0.04
GRIPPER_CLOSE_VALUE = 0.00
GRIPPER_FORCE = 30
GRIPPER_MAX_VELOCITY = 0.2


# ============================================================
# Transform utilities
# ============================================================

def yaw_to_quat(yaw_deg):
    return p.getQuaternionFromEuler([0.0, 0.0, math.radians(yaw_deg)])


def pose_to_T(pos, quat):
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
    return T_base_marker @ invert_T(T_camera_marker) @ T_camera_object


# ============================================================
# Camera utilities
# ============================================================

def get_camera_intrinsics(width, height, fov_deg):
    fov_rad = math.radians(fov_deg)
    fy = height / (2.0 * math.tan(fov_rad / 2.0))
    fx = fy
    cx = width / 2.0
    cy = height / 2.0
    return np.array(
        [[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )


def get_virtual_camera(width=WIDTH, height=HEIGHT, fov_deg=FOV_DEG, near=NEAR, far=FAR):
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

    # PyBullet/OpenGL view matrix: base/world -> OpenGL camera
    T_gl_base = np.array(view_matrix, dtype=np.float64).reshape(4, 4, order="F")

    # OpenGL camera: x right, y up, z backward
    # OpenCV camera: x right, y down, z forward
    T_cv_gl = np.diag([1.0, -1.0, -1.0, 1.0])
    T_camera_base = T_cv_gl @ T_gl_base
    T_base_camera = invert_T(T_camera_base)

    K = get_camera_intrinsics(width, height, fov_deg)
    return view_matrix, projection_matrix, T_base_camera, T_camera_base, K


def depth_buffer_to_meters(depth_buffer, near=NEAR, far=FAR):
    depth_buffer = np.asarray(depth_buffer, dtype=np.float64)
    return far * near / (far - (far - near) * depth_buffer)


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


# ============================================================
# Object creation
# ============================================================

def create_marker(position, yaw_deg=0.0):
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


def draw_target_sphere(position, color=(1, 0, 0, 1), radius=0.02, label=None):
    visual = p.createVisualShape(
        shapeType=p.GEOM_SPHERE,
        radius=radius,
        rgbaColor=color,
    )
    body = p.createMultiBody(
        baseMass=0,
        baseCollisionShapeIndex=-1,
        baseVisualShapeIndex=visual,
        basePosition=position.tolist() if isinstance(position, np.ndarray) else position,
    )
    if label is not None:
        p.addUserDebugText(label, position, textColorRGB=list(color[:3]))
    return body


# ============================================================
# RGB-D object estimation
# ============================================================

def get_cup_mask_from_segmentation(seg, cup_id):
    mask = seg == cup_id
    if not np.any(mask):
        return None
    return mask


def create_T_camera_object_from_mask_aabb(K, depth_m, mask):
    if mask is None or not np.any(mask):
        return None, None

    vs, us = np.where(mask)
    zs = depth_m[vs, us]
    valid = np.isfinite(zs) & (zs > 0) & (zs < FAR)

    if np.count_nonzero(valid) < 20:
        return None, None

    us = us[valid]
    vs = vs[valid]
    zs = zs[valid]

    xs = (us.astype(np.float64) - K[0, 2]) * zs / K[0, 0]
    ys = (vs.astype(np.float64) - K[1, 2]) * zs / K[1, 1]
    points = np.stack([xs, ys, zs], axis=1)

    lower = np.percentile(points, 5, axis=0)
    upper = np.percentile(points, 95, axis=0)
    inliers = np.all((points >= lower) & (points <= upper), axis=1)
    points_in = points[inliers] if np.count_nonzero(inliers) >= 20 else points

    center_camera = (np.min(points_in, axis=0) + np.max(points_in, axis=0)) / 2.0

    T_camera_object = np.eye(4, dtype=np.float64)
    T_camera_object[:3, 3] = center_camera
    return T_camera_object, center_camera


def create_point_cloud_from_rgbd(rgb, depth_m, K, depth_trunc=DEPTH_TRUNC_M):
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
    pcd_vis = o3d.geometry.PointCloud(pcd_camera)
    pcd_vis.transform(
        np.array(
            [[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]],
            dtype=np.float64,
        )
    )
    return pcd_vis


# ============================================================
# Robot utilities
# ============================================================

def print_joint_and_link_info(robot_id):
    print("\n========== PyBullet Joint / Link Info ==========")
    for i in range(p.getNumJoints(robot_id)):
        info = p.getJointInfo(robot_id, i)
        joint_name = info[1].decode("utf-8")
        joint_type = info[2]
        link_name = info[12].decode("utf-8")
        lower = info[8]
        upper = info[9]
        print(
            f"index={i:2d} | joint={joint_name:35s} | type={joint_type} | "
            f"link={link_name:35s} | limits=({lower:.3f}, {upper:.3f})"
        )


def get_movable_joint_indices(robot_id):
    movable = []
    for i in range(p.getNumJoints(robot_id)):
        joint_type = p.getJointInfo(robot_id, i)[2]
        if joint_type in [p.JOINT_REVOLUTE, p.JOINT_PRISMATIC]:
            movable.append(i)
    return movable


def get_arm_joint_indices(robot_id, movable_joints):
    """
    Prefer only the first 6 revolute arm joints for IK control.
    This avoids accidentally controlling finger joints with arm IK.
    """
    arm_joints = []
    for j in movable_joints:
        info = p.getJointInfo(robot_id, j)
        joint_name = info[1].decode("utf-8").lower()
        joint_type = info[2]
        if joint_type == p.JOINT_REVOLUTE and not any(k in joint_name for k in ["finger", "gripper"]):
            arm_joints.append(j)
        if len(arm_joints) == 6:
            break
    return arm_joints if len(arm_joints) >= 6 else movable_joints[:6]


def detect_gripper_joint_indices(robot_id):
    gripper_joints = []
    for i in range(p.getNumJoints(robot_id)):
        info = p.getJointInfo(robot_id, i)
        joint_name = info[1].decode("utf-8").lower()
        link_name = info[12].decode("utf-8").lower()
        joint_type = info[2]
        if joint_type in [p.JOINT_REVOLUTE, p.JOINT_PRISMATIC] and any(
            k in joint_name + " " + link_name for k in ["finger", "gripper"]
        ):
            gripper_joints.append(i)
    return gripper_joints


def set_gripper(robot_id, gripper_joints, value, steps=120):
    if len(gripper_joints) == 0:
        print("No gripper joints detected. Skipping visual gripper command.")
        return

    print(f"Setting gripper joints {gripper_joints} to {value}")
    for _ in range(steps):
        for j in gripper_joints:
            p.setJointMotorControl2(
                robot_id,
                j,
                p.POSITION_CONTROL,
                targetPosition=value,
                force=GRIPPER_FORCE,
                positionGain=0.2,
                maxVelocity=GRIPPER_MAX_VELOCITY,
            )
        p.stepSimulation()
        time.sleep(1.0 / 240.0)


def detect_end_effector_link_index(robot_id):
    if END_EFFECTOR_LINK_INDEX_OVERRIDE is not None:
        return END_EFFECTOR_LINK_INDEX_OVERRIDE
    return p.getNumJoints(robot_id) - 1


def get_joint_limits(robot_id, joints):
    lower_limits = []
    upper_limits = []
    joint_ranges = []
    rest_poses = []

    for j in joints:
        info = p.getJointInfo(robot_id, j)
        lower = info[8]
        upper = info[9]

        if lower > upper:
            lower = -math.pi
            upper = math.pi

        lower_limits.append(lower)
        upper_limits.append(upper)
        joint_ranges.append(upper - lower if upper > lower else 2 * math.pi)
        rest_poses.append(p.getJointState(robot_id, j)[0])

    return lower_limits, upper_limits, joint_ranges, rest_poses


def smoothstep(t):
    """0->1 smooth interpolation with zero velocity at start and end."""
    return 3.0 * t * t - 2.0 * t * t * t


def step_simulation(steps):
    for _ in range(steps):
        p.stepSimulation()
        time.sleep(1.0 / 240.0)


def move_robot_to_joint_positions(robot_id, joints, target_joint_positions, steps=MOVE_STEPS):
    """
    Smooth joint-space motion.

    The previous version sent the final joint target directly at every step.
    PyBullet motors could accelerate too aggressively and overshoot.
    This version interpolates from current joint positions to target joint positions.
    """
    n = min(len(joints), len(target_joint_positions))

    start_positions = np.array(
        [p.getJointState(robot_id, joints[idx])[0] for idx in range(n)],
        dtype=np.float64,
    )
    target_positions = np.array(target_joint_positions[:n], dtype=np.float64)

    for step in range(steps):
        t = step / max(1, steps - 1)
        s = smoothstep(t)
        current_targets = (1.0 - s) * start_positions + s * target_positions

        for idx in range(n):
            joint_index = joints[idx]
            p.setJointMotorControl2(
                bodyUniqueId=robot_id,
                jointIndex=joint_index,
                controlMode=p.POSITION_CONTROL,
                targetPosition=float(current_targets[idx]),
                force=MAX_FORCE,
                positionGain=POSITION_GAIN,
                maxVelocity=MAX_VELOCITY,
            )
        p.stepSimulation()
        time.sleep(1.0 / 240.0)

    # Hold final position and let the robot settle.
    for _ in range(SETTLE_STEPS):
        for idx in range(n):
            joint_index = joints[idx]
            p.setJointMotorControl2(
                bodyUniqueId=robot_id,
                jointIndex=joint_index,
                controlMode=p.POSITION_CONTROL,
                targetPosition=float(target_positions[idx]),
                force=MAX_FORCE,
                positionGain=POSITION_GAIN,
                maxVelocity=MAX_VELOCITY,
            )
        p.stepSimulation()
        time.sleep(1.0 / 240.0)


def move_end_effector_to_position(robot_id, ee_link_index, arm_joints, target_pos, target_quat=None, steps=MOVE_STEPS):
    lower_limits, upper_limits, joint_ranges, rest_poses = get_joint_limits(robot_id, arm_joints)

    kwargs = dict(
        bodyUniqueId=robot_id,
        endEffectorLinkIndex=ee_link_index,
        targetPosition=target_pos,
        lowerLimits=lower_limits,
        upperLimits=upper_limits,
        jointRanges=joint_ranges,
        restPoses=rest_poses,
        maxNumIterations=200,
        residualThreshold=1e-4,
    )
    if target_quat is not None:
        kwargs["targetOrientation"] = target_quat

    ik_solution = p.calculateInverseKinematics(**kwargs)
    move_robot_to_joint_positions(robot_id, arm_joints, ik_solution, steps=steps)
    return ik_solution


def get_link_pose(robot_id, link_index):
    state = p.getLinkState(robot_id, link_index, computeForwardKinematics=True)
    pos = np.array(state[4], dtype=np.float64)
    quat = np.array(state[5], dtype=np.float64)
    return pos, quat


def get_body_pose(body_id):
    pos, quat = p.getBasePositionAndOrientation(body_id)
    return np.array(pos, dtype=np.float64), np.array(quat, dtype=np.float64)


def disable_robot_object_collisions(robot_id, object_id):
    """
    Disable collisions between all robot links and the object.
    This is useful for pipeline-level testing where grasp is simulated by fixed constraint.
    """
    for link_idx in range(-1, p.getNumJoints(robot_id)):
        p.setCollisionFilterPair(robot_id, object_id, link_idx, -1, enableCollision=0)
    print("Robot-cup collisions disabled for pipeline validation.")


def attach_object_to_ee(robot_id, ee_link_index, object_id):
    """
    Create a fixed constraint that preserves the current relative transform
    between the end-effector link and the object.
    """
    ee_pos, ee_quat = get_link_pose(robot_id, ee_link_index)
    obj_pos, obj_quat = get_body_pose(object_id)

    inv_ee_pos, inv_ee_quat = p.invertTransform(ee_pos.tolist(), ee_quat.tolist())
    parent_frame_pos, parent_frame_quat = p.multiplyTransforms(
        inv_ee_pos,
        inv_ee_quat,
        obj_pos.tolist(),
        obj_quat.tolist(),
    )

    constraint_id = p.createConstraint(
        parentBodyUniqueId=robot_id,
        parentLinkIndex=ee_link_index,
        childBodyUniqueId=object_id,
        childLinkIndex=-1,
        jointType=p.JOINT_FIXED,
        jointAxis=[0, 0, 0],
        parentFramePosition=parent_frame_pos,
        childFramePosition=[0, 0, 0],
        parentFrameOrientation=parent_frame_quat,
        childFrameOrientation=[0, 0, 0, 1],
    )
    print(f"Created fixed grasp constraint: {constraint_id}")
    return constraint_id


def detach_object(constraint_id):
    if constraint_id is not None:
        p.removeConstraint(constraint_id)
        print(f"Removed grasp constraint: {constraint_id}")


# ============================================================
# Save utilities
# ============================================================

def save_debug_outputs(rgb, depth_m, pcd_camera, pcd_vis, matrices, debug_info):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = SAVE_ROOT / f"sim_pick_place_{timestamp}"
    save_dir.mkdir(parents=True, exist_ok=True)

    cv2.imwrite(str(save_dir / "sim_rgb.png"), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    depth_mm = np.clip(depth_m * 1000.0, 0, 65535).astype(np.uint16)
    cv2.imwrite(str(save_dir / "sim_depth_mm.png"), depth_mm)

    o3d.io.write_point_cloud(str(save_dir / "sim_scene_camera_frame.ply"), pcd_camera)
    o3d.io.write_point_cloud(str(save_dir / "sim_scene_visualization_flipped.ply"), pcd_vis)

    for name, mat in matrices.items():
        np.save(str(save_dir / f"{name}.npy"), mat)

    with open(save_dir / "debug_info.json", "w", encoding="utf-8") as f:
        json.dump(debug_info, f, indent=2)

    print(f"\n✅ Saved debug outputs to: {save_dir}")


# ============================================================
# Main
# ============================================================

def main():
    SAVE_ROOT.mkdir(parents=True, exist_ok=True)

    p.connect(p.GUI)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -9.81)

    p.loadURDF("plane.urdf")

    print(f"Loading robot: {ROBOT_URDF_PATH}")
    robot_id = p.loadURDF(
        ROBOT_URDF_PATH,
        basePosition=[0, 0, 0],
        baseOrientation=p.getQuaternionFromEuler([0, 0, 0]),
        useFixedBase=True,
    )
    print("Robot loaded.")

    draw_base_axes()
    print_joint_and_link_info(robot_id)

    movable_joints = get_movable_joint_indices(robot_id)
    arm_joints = get_arm_joint_indices(robot_id, movable_joints)
    gripper_joints = detect_gripper_joint_indices(robot_id)
    ee_link_index = detect_end_effector_link_index(robot_id)

    print("\nMovable joints:", movable_joints)
    print("Arm joints used for IK:", arm_joints)
    print("Detected gripper joints:", gripper_joints)
    print("Selected end effector link index:", ee_link_index)

    marker_id = create_marker(MARKER_POS_BASE, MARKER_YAW_DEG)
    cup_id = create_cup(CUP_POS_BASE)

    if DISABLE_ROBOT_CUP_COLLISION_FOR_PIPELINE:
        disable_robot_object_collisions(robot_id, cup_id)

    step_simulation(120)

    # ========================================================
    # Perception simulation: dynamic calibration result
    # ========================================================
    view_matrix, projection_matrix, T_base_camera, T_camera_base, K = get_virtual_camera()
    rgb, depth_m, seg = render_rgbd(view_matrix, projection_matrix)

    T_base_marker = get_body_T_base(marker_id)
    T_base_object_gt_before = get_body_T_base(cup_id)
    T_camera_marker = T_camera_base @ T_base_marker

    cup_mask = get_cup_mask_from_segmentation(seg, cup_id)
    T_camera_object, cup_center_camera = create_T_camera_object_from_mask_aabb(K, depth_m, cup_mask)

    if T_camera_object is None:
        print("❌ Could not estimate T_camera_object from virtual RGB-D.")
        p.disconnect()
        return

    T_base_object_est = transform_camera_object_to_base(
        T_base_marker,
        T_camera_marker,
        T_camera_object,
    )

    cup_pos_est = T_base_object_est[:3, 3]
    cup_pos_gt = T_base_object_gt_before[:3, 3]
    perception_error = np.linalg.norm(cup_pos_est - cup_pos_gt)

    print("\n========== Dynamic Calibration Perception Result ==========")
    print("Estimated cup position in base frame:", cup_pos_est)
    print("Ground truth cup position in base frame:", cup_pos_gt)
    print(f"Perception position error: {perception_error:.6f} m")

    # Debug spheres
    draw_target_sphere(cup_pos_gt, color=(0, 1, 0, 1), radius=0.018, label="GT cup")
    draw_target_sphere(cup_pos_est, color=(1, 0, 0, 1), radius=0.018, label="estimated cup")
    draw_target_sphere(PLACE_POS_BASE, color=(0, 1, 1, 1), radius=0.018, label="place target")

    # ========================================================
    # Motion targets
    # ========================================================
    pre_grasp_pos = np.array(
        [cup_pos_est[0], cup_pos_est[1], cup_pos_est[2] + PRE_GRASP_HEIGHT_ABOVE_CUP],
        dtype=np.float64,
    )
    grasp_pos = np.array(
        [cup_pos_est[0], cup_pos_est[1], cup_pos_est[2] + GRASP_HEIGHT_ABOVE_CUP],
        dtype=np.float64,
    )
    lift_pos = pre_grasp_pos.copy()

    pre_place_pos = np.array(
        [PLACE_POS_BASE[0], PLACE_POS_BASE[1], PLACE_POS_BASE[2] + PRE_PLACE_HEIGHT_ABOVE_TABLE],
        dtype=np.float64,
    )
    place_pos = np.array(
        [PLACE_POS_BASE[0], PLACE_POS_BASE[1], PLACE_POS_BASE[2] + PLACE_HEIGHT_ABOVE_TABLE],
        dtype=np.float64,
    )
    retreat_pos = np.array(
        [PLACE_POS_BASE[0], PLACE_POS_BASE[1], PLACE_POS_BASE[2] + RETREAT_HEIGHT_ABOVE_TABLE],
        dtype=np.float64,
    )

    draw_target_sphere(pre_grasp_pos, color=(0, 0, 1, 1), radius=0.015, label="pre-grasp")
    draw_target_sphere(grasp_pos, color=(1, 1, 0, 1), radius=0.015, label="grasp")
    draw_target_sphere(pre_place_pos, color=(1, 0, 1, 1), radius=0.015, label="pre-place")
    draw_target_sphere(place_pos, color=(0, 1, 1, 1), radius=0.015, label="place")

    print("\n========== Pick-Place Targets ==========")
    print("pre_grasp_pos:", pre_grasp_pos)
    print("grasp_pos:", grasp_pos)
    print("lift_pos:", lift_pos)
    print("pre_place_pos:", pre_place_pos)
    print("place_pos:", place_pos)
    print("retreat_pos:", retreat_pos)
    print("\n说明：当前版本会用 fixed constraint 模拟抓住杯子。")
    print("如果 DISABLE_ROBOT_CUP_COLLISION_FOR_PIPELINE=True，机器人不会物理撞倒杯子。")

    input("\nPress Enter to start simulated pick-place...")

    # ========================================================
    # Simulated pick-place pipeline
    # ========================================================
    grasp_constraint = None

    print("\n[1] Open gripper")
    set_gripper(robot_id, gripper_joints, GRIPPER_OPEN_VALUE, steps=120)

    print("\n[2] Move to pre-grasp")
    move_end_effector_to_position(robot_id, ee_link_index, arm_joints, pre_grasp_pos.tolist())

    print("\n[3] Descend to grasp")
    move_end_effector_to_position(robot_id, ee_link_index, arm_joints, grasp_pos.tolist(), steps=1500)

    print("\n[4] Close gripper and attach cup")
    set_gripper(robot_id, gripper_joints, GRIPPER_CLOSE_VALUE, steps=120)
    grasp_constraint = attach_object_to_ee(robot_id, ee_link_index, cup_id)

    print("\n[5] Lift cup")
    move_end_effector_to_position(robot_id, ee_link_index, arm_joints, lift_pos.tolist())

    print("\n[6] Move to pre-place")
    move_end_effector_to_position(robot_id, ee_link_index, arm_joints, pre_place_pos.tolist())

    print("\n[7] Descend to place")
    move_end_effector_to_position(robot_id, ee_link_index, arm_joints, place_pos.tolist(), steps=1500)

    print("\n[8] Detach cup and open gripper")
    detach_object(grasp_constraint)
    grasp_constraint = None
    set_gripper(robot_id, gripper_joints, GRIPPER_OPEN_VALUE, steps=120)

    # Let cup settle on table / plane
    step_simulation(180)

    print("\n[9] Retreat")
    move_end_effector_to_position(robot_id, ee_link_index, arm_joints, retreat_pos.tolist())

    # ========================================================
    # Report result
    # ========================================================
    T_base_object_gt_after = get_body_T_base(cup_id)
    final_cup_pos = T_base_object_gt_after[:3, 3]
    place_error_xy = np.linalg.norm(final_cup_pos[:2] - PLACE_POS_BASE[:2])
    place_error_xyz = np.linalg.norm(final_cup_pos - PLACE_POS_BASE)

    print("\n========== Pick-Place Result ==========")
    print("Final cup position:", final_cup_pos)
    print("Target place position:", PLACE_POS_BASE)
    print(f"Place error XY : {place_error_xy:.6f} m")
    print(f"Place error XYZ: {place_error_xyz:.6f} m")

    pcd_camera = create_point_cloud_from_rgbd(rgb, depth_m, K)
    pcd_vis = make_visualization_copy(pcd_camera)

    save_debug_outputs(
        rgb=rgb,
        depth_m=depth_m,
        pcd_camera=pcd_camera,
        pcd_vis=pcd_vis,
        matrices={
            "T_base_marker": T_base_marker,
            "T_camera_marker": T_camera_marker,
            "T_camera_object": T_camera_object,
            "T_base_object_est": T_base_object_est,
            "T_base_object_gt_before": T_base_object_gt_before,
            "T_base_object_gt_after": T_base_object_gt_after,
        },
        debug_info={
            "perception_error_m": float(perception_error),
            "pre_grasp_pos": pre_grasp_pos.tolist(),
            "grasp_pos": grasp_pos.tolist(),
            "lift_pos": lift_pos.tolist(),
            "pre_place_pos": pre_place_pos.tolist(),
            "place_pos": place_pos.tolist(),
            "retreat_pos": retreat_pos.tolist(),
            "final_cup_pos": final_cup_pos.tolist(),
            "target_place_pos": PLACE_POS_BASE.tolist(),
            "place_error_xy_m": float(place_error_xy),
            "place_error_xyz_m": float(place_error_xyz),
            "ee_link_index": int(ee_link_index),
            "arm_joints": [int(j) for j in arm_joints],
            "gripper_joints": [int(j) for j in gripper_joints],
            "disable_robot_cup_collision_for_pipeline": DISABLE_ROBOT_CUP_COLLISION_FOR_PIPELINE,
            "note": "Cup is attached to EE using a fixed PyBullet constraint after closing gripper. This is an intentional simplification for pipeline testing.",
        },
    )

    print("\nDone. PyBullet GUI will stay open. Press Ctrl+C in terminal to exit.")
    try:
        while True:
            p.stepSimulation()
            time.sleep(1.0 / 240.0)
    except KeyboardInterrupt:
        pass
    finally:
        if grasp_constraint is not None:
            detach_object(grasp_constraint)
        p.disconnect()
        print("Simulation closed.")


if __name__ == "__main__":
    main()

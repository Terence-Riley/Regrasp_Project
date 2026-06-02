import pybullet as p
import pybullet_data
import time
import math
import numpy as np

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False


def pose_to_T(pos, quat):
    """
    把 PyBullet 的 position + quaternion 转成 4x4 齐次变换矩阵 T
    T_A_B 表示：B 坐标系在 A 坐标系下的位姿
    """
    T = np.eye(4)
    R = np.array(p.getMatrixFromQuaternion(quat)).reshape(3, 3)
    T[:3, :3] = R
    T[:3, 3] = np.array(pos)
    return T


def get_body_T(body_id):
    """
    读取某个物体在 PyBullet world/base 坐标系下的位姿
    """
    pos, quat = p.getBasePositionAndOrientation(body_id)
    return pose_to_T(pos, quat)


def invert_T(T):
    return np.linalg.inv(T)


def create_cup(position):
    """
    用一个简单圆柱代替杯子。
    这不是完美杯子，但足够用来测试坐标变换和抓取流程。
    """
    radius = 0.035
    height = 0.10

    collision_shape = p.createCollisionShape(
        shapeType=p.GEOM_CYLINDER,
        radius=radius,
        height=height
    )

    visual_shape = p.createVisualShape(
        shapeType=p.GEOM_CYLINDER,
        radius=radius,
        length=height,
        rgbaColor=[0.9, 0.9, 0.9, 1.0]
    )

    cup_id = p.createMultiBody(
        baseMass=0.1,
        baseCollisionShapeIndex=collision_shape,
        baseVisualShapeIndex=visual_shape,
        basePosition=position
    )

    return cup_id


def create_marker(position):
    """
    用一个很薄的黑色方块代替 ArUco marker。
    第一版先不用 OpenCV 识别它，只用它作为固定参考物。
    """
    marker_size = 0.096  # 9.6 cm，和真实 ArUco 黑色外边框边长保持一致

    collision_shape = p.createCollisionShape(
        shapeType=p.GEOM_BOX,
        halfExtents=[marker_size / 2, marker_size / 2, 0.001]
    )

    visual_shape = p.createVisualShape(
        shapeType=p.GEOM_BOX,
        halfExtents=[marker_size / 2, marker_size / 2, 0.001],
        rgbaColor=[0.0, 0.0, 0.0, 1.0]
    )

    marker_id = p.createMultiBody(
        baseMass=0.0,
        baseCollisionShapeIndex=collision_shape,
        baseVisualShapeIndex=visual_shape,
        basePosition=position,
        baseOrientation=p.getQuaternionFromEuler([0, 0, 0])
    )

    return marker_id


def get_virtual_camera(width=640, height=480, fov=60, near=0.01, far=2.0):
    """
    架设一个虚拟相机，并返回：
    - view_matrix
    - projection_matrix
    - T_base_camera
    - T_camera_base

    这里假设 PyBullet world frame = robot base frame
    """
    camera_eye = [0.45, -0.55, 0.45]
    camera_target = [0.35, 0.00, 0.02]
    camera_up = [0, 0, 1]

    view_matrix = p.computeViewMatrix(
        cameraEyePosition=camera_eye,
        cameraTargetPosition=camera_target,
        cameraUpVector=camera_up
    )

    projection_matrix = p.computeProjectionMatrixFOV(
        fov=fov,
        aspect=width / height,
        nearVal=near,
        farVal=far
    )

    # PyBullet/OpenGL 的 view matrix 是 world -> OpenGL camera
    T_gl_base = np.array(view_matrix).reshape(4, 4, order="F")

    # OpenGL camera 坐标：x右，y上，z朝后
    # OpenCV camera 坐标：x右，y下，z朝前
    T_cv_gl = np.diag([1, -1, -1, 1])

    # base/world -> OpenCV camera
    T_camera_base = T_cv_gl @ T_gl_base

    # OpenCV camera -> base/world
    T_base_camera = invert_T(T_camera_base)

    return view_matrix, projection_matrix, T_base_camera, T_camera_base


def capture_virtual_image(view_matrix, projection_matrix, width=640, height=480):
    """
    用虚拟相机拍一张 RGB 图，保存为 sim_camera_rgb.png
    """
    img = p.getCameraImage(
        width=width,
        height=height,
        viewMatrix=view_matrix,
        projectionMatrix=projection_matrix,
        renderer=p.ER_BULLET_HARDWARE_OPENGL
    )

    rgba = np.reshape(img[2], (height, width, 4))
    rgb = rgba[:, :, :3]

    if HAS_CV2:
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        cv2.imwrite("sim_camera_rgb.png", bgr)
        print("已保存虚拟相机图像：sim_camera_rgb.png")
    else:
        print("未安装 cv2，跳过图像保存。")

    return rgb


def main():
    # 1. 启动 PyBullet
    physics_client = p.connect(p.GUI)

    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -9.81)

    # 2. 加载地面
    plane_id = p.loadURDF("plane.urdf")

    # 3. 加载机械臂
    robot_urdf_path = "kortex_description/robots/gen3_lite.urdf"
    print(f"正在加载机械臂: {robot_urdf_path}")

    robot_id = p.loadURDF(
        robot_urdf_path,
        basePosition=[0, 0, 0],
        useFixedBase=True
    )

    print("机械臂加载成功。")

    # 4. 加载 marker 和杯子
    # 这里 PyBullet world 就当成 robot base
    marker_id = create_marker(position=[0.30, -0.15, 0.002])
    cup_id = create_cup(position=[0.45, 0.10, 0.05])

    print("marker 和 cup 加载成功。")

    # 5. 让仿真稳定几步
    for _ in range(120):
        p.stepSimulation()
        time.sleep(1.0 / 240.0)

    # 6. 架设虚拟相机并拍图
    view_matrix, projection_matrix, T_base_camera, T_camera_base = get_virtual_camera()
    capture_virtual_image(view_matrix, projection_matrix)

    # 7. 读取 marker 和 cup 的真实位姿
    T_base_marker = get_body_T(marker_id)
    T_base_object_gt = get_body_T(cup_id)

    # 8. 模拟相机“看到” marker 和 object
    # 注意：这一步以后会被 OpenCV / 深度点云替代
    T_camera_marker = T_camera_base @ T_base_marker
    T_camera_object = T_camera_base @ T_base_object_gt

    # 9. 使用动态标定公式恢复 cup 在 base 下的位姿
    T_base_object_est = (
        T_base_marker
        @ invert_T(T_camera_marker)
        @ T_camera_object
    )

    print("\n========== 动态标定公式验证 ==========")

    print("真实 cup 位置 T_base_object_gt:")
    print(T_base_object_gt[:3, 3])

    print("通过动态标定公式估计出的 cup 位置 T_base_object_est:")
    print(T_base_object_est[:3, 3])

    error = np.linalg.norm(
        T_base_object_gt[:3, 3] - T_base_object_est[:3, 3]
    )

    print(f"位置误差: {error:.10f} m")

    if error < 1e-6:
        print("验证成功：你的动态标定矩阵链条是对的。")
    else:
        print("误差较大：需要检查坐标系定义。")

    print("\n保持仿真运行，按 Ctrl+C 退出。")

    try:
        while True:
            p.stepSimulation()
            time.sleep(1.0 / 240.0)
    except KeyboardInterrupt:
        p.disconnect()
        print("仿真已关闭。")


if __name__ == "__main__":
    main()

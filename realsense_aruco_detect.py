import cv2
import numpy as np
import pyrealsense2 as rs


MARKER_LENGTH = 0.096  # 9.6 cm，单位 m


def rvec_tvec_to_T(rvec, tvec):
    R, _ = cv2.Rodrigues(rvec)

    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = tvec.reshape(3)

    return T


def get_camera_matrix_from_intrinsics(intr):
    camera_matrix = np.array([
        [intr.fx, 0, intr.ppx],
        [0, intr.fy, intr.ppy],
        [0, 0, 1]
    ], dtype=np.float64)

    dist_coeffs = np.array(intr.coeffs, dtype=np.float64)

    return camera_matrix, dist_coeffs


def main():
    # 1. 启动 RealSense
    pipeline = rs.pipeline()
    config = rs.config()

    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

    profile = pipeline.start(config)

    # 2. 获取 color 相机内参
    color_stream = profile.get_stream(rs.stream.color)
    color_intr = color_stream.as_video_stream_profile().get_intrinsics()

    camera_matrix, dist_coeffs = get_camera_matrix_from_intrinsics(color_intr)

    print("========== RealSense Color Intrinsics ==========")
    print("fx:", color_intr.fx)
    print("fy:", color_intr.fy)
    print("cx/ppx:", color_intr.ppx)
    print("cy/ppy:", color_intr.ppy)
    print("dist coeffs:", color_intr.coeffs)
    print("camera_matrix:\n", camera_matrix)
    print("dist_coeffs:", dist_coeffs)

    # 3. ArUco detector
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
    parameters = cv2.aruco.DetectorParameters()

    try:
        detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)
        use_new_api = True
    except AttributeError:
        detector = None
        use_new_api = False

    # 4. 定义 marker 四个角点在 marker 自己坐标系下的位置
    # OpenCV 官方教程里 pose estimation 需要 marker side length，
    # 并用 marker 四个角点和图像角点通过 solvePnP 求 rvec/tvec。
    half = MARKER_LENGTH / 2.0

    obj_points = np.array([
        [-half,  half, 0],
        [ half,  half, 0],
        [ half, -half, 0],
        [-half, -half, 0]
    ], dtype=np.float32)

    print("\n开始检测 ArUco。按 q 退出。")

    try:
        while True:
            frames = pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()

            if not color_frame:
                continue

            color_image = np.asanyarray(color_frame.get_data())
            gray = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)

            if use_new_api:
                corners, ids, rejected = detector.detectMarkers(gray)
            else:
                corners, ids, rejected = cv2.aruco.detectMarkers(
                    gray,
                    aruco_dict,
                    parameters=parameters
                )

            vis = color_image.copy()

            if ids is not None:
                cv2.aruco.drawDetectedMarkers(vis, corners, ids)

                for i, marker_id in enumerate(ids.flatten()):
                    image_points = corners[i].reshape(-1, 2).astype(np.float32)

                    success, rvec, tvec = cv2.solvePnP(
                        obj_points,
                        image_points,
                        camera_matrix,
                        dist_coeffs
                    )

                    if not success:
                        continue

                    T_camera_marker = rvec_tvec_to_T(rvec, tvec)

                    cv2.drawFrameAxes(
                        vis,
                        camera_matrix,
                        dist_coeffs,
                        rvec,
                        tvec,
                        MARKER_LENGTH * 0.5
                    )

                    print("\nDetected marker ID:", marker_id)
                    print("tvec, marker position in camera frame [m]:")
                    print(tvec.reshape(3))
                    print("T_camera_marker:")
                    print(T_camera_marker)

            cv2.imshow("RealSense ArUco Detection", vis)

            key = cv2.waitKey(1)
            if key == ord("q"):
                break

    finally:
        pipeline.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
import json
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import open3d as o3d
import pyrealsense2 as rs


# =========================
# User settings
# =========================

# 你的 marker 实测黑色外边框边长：9.6 cm = 0.096 m
MARKER_LENGTH = 0.096

# 你生成的是 DICT_6X6_250, ID 23
TARGET_MARKER_ID = 23

WIDTH = 640
HEIGHT = 480
FPS = 15
DEPTH_TRUNC_M = 3.0
SAVE_ROOT = Path("captures")


# =========================
# Matrix utilities
# =========================

def rvec_tvec_to_T(rvec, tvec):
    """
    OpenCV rvec/tvec -> 4x4 homogeneous transform.
    Output: T_camera_marker, meaning marker frame expressed in camera frame.
    """
    R, _ = cv2.Rodrigues(rvec)
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    T[:3, 3] = np.asarray(tvec).reshape(3)
    return T


def get_camera_matrix_from_intrinsics(intr):
    """RealSense intrinsics -> OpenCV camera matrix and distortion coeffs."""
    camera_matrix = np.array(
        [
            [intr.fx, 0.0, intr.ppx],
            [0.0, intr.fy, intr.ppy],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    dist_coeffs = np.array(intr.coeffs, dtype=np.float64)
    return camera_matrix, dist_coeffs


# =========================
# ArUco utilities
# =========================

def create_aruco_detector():
    """Compatible with both newer and older OpenCV ArUco APIs."""
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)

    if hasattr(cv2.aruco, "ArucoDetector"):
        parameters = cv2.aruco.DetectorParameters()
        detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)
        return detector, aruco_dict, parameters, True

    parameters = cv2.aruco.DetectorParameters_create()
    return None, aruco_dict, parameters, False


def detect_markers(gray, detector, aruco_dict, parameters, use_new_api):
    if use_new_api:
        corners, ids, rejected = detector.detectMarkers(gray)
    else:
        corners, ids, rejected = cv2.aruco.detectMarkers(
            gray, aruco_dict, parameters=parameters
        )
    return corners, ids, rejected


def estimate_single_marker_pose(corners_one, camera_matrix, dist_coeffs):
    """
    Estimate one marker pose.
    Preferred: cv2.aruco.estimatePoseSingleMarkers.
    Fallback: cv2.solvePnP with explicit corner coordinates.
    """
    if hasattr(cv2.aruco, "estimatePoseSingleMarkers"):
        rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(
            [corners_one], MARKER_LENGTH, camera_matrix, dist_coeffs
        )
        return rvecs[0][0], tvecs[0][0]

    # Fallback for unusual OpenCV builds
    half = MARKER_LENGTH / 2.0
    obj_points = np.array(
        [
            [-half, half, 0.0],
            [half, half, 0.0],
            [half, -half, 0.0],
            [-half, -half, 0.0],
        ],
        dtype=np.float32,
    )
    img_points = corners_one.reshape(-1, 2).astype(np.float32)
    success, rvec, tvec = cv2.solvePnP(
        obj_points, img_points, camera_matrix, dist_coeffs
    )
    if not success:
        return None, None
    return rvec.reshape(3), tvec.reshape(3)


def choose_target_marker(corners, ids):
    """
    Pick TARGET_MARKER_ID if visible.
    Return: index in corners/ids, or None.
    """
    if ids is None:
        return None

    ids_flat = ids.flatten()
    matches = np.where(ids_flat == TARGET_MARKER_ID)[0]
    if len(matches) > 0:
        return int(matches[0])

    # If target ID is not visible but other markers exist, do not use them.
    return None


# =========================
# Depth / point cloud utilities
# =========================

def get_median_depth_m(depth_image_z16, u, v, depth_scale_m, window=7):
    """
    Median depth around pixel (u, v), in meters.
    This is more stable than using a single depth pixel.
    """
    h, w = depth_image_z16.shape
    half = window // 2

    u0 = max(0, u - half)
    u1 = min(w, u + half + 1)
    v0 = max(0, v - half)
    v1 = min(h, v + half + 1)

    patch = depth_image_z16[v0:v1, u0:u1].astype(np.float64)
    valid = patch[patch > 0]

    if valid.size == 0:
        return None

    return float(np.median(valid) * depth_scale_m)


def create_point_cloud_camera_frame(color_bgr, depth_z16, intr, depth_scale_m):
    """
    Create Open3D point cloud in camera frame.

    Important:
    - This point cloud is NOT flipped.
    - Use this one for algorithms / coordinate calculations.
    - It matches the normal camera convention: x right, y down, z forward.
    """
    color_rgb = cv2.cvtColor(color_bgr, cv2.COLOR_BGR2RGB)

    o3d_color = o3d.geometry.Image(color_rgb)
    o3d_depth = o3d.geometry.Image(depth_z16)

    o3d_intrinsics = o3d.camera.PinholeCameraIntrinsic(
        intr.width,
        intr.height,
        intr.fx,
        intr.fy,
        intr.ppx,
        intr.ppy,
    )

    # Open3D expects depth_in_meters = raw_depth / depth_scale
    # RealSense raw_depth * depth_scale_m = depth_in_meters
    open3d_depth_scale = 1.0 / depth_scale_m

    rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
        o3d_color,
        o3d_depth,
        depth_scale=open3d_depth_scale,
        depth_trunc=DEPTH_TRUNC_M,
        convert_rgb_to_intensity=False,
    )

    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, o3d_intrinsics)
    pcd.remove_non_finite_points()
    return pcd


def make_visualization_copy(pcd_camera):
    """
    Open3D visualization-friendly copy.
    Do NOT use this for coordinate calculation.
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


# =========================
# Save utilities
# =========================

def save_capture_package(
    color_bgr,
    depth_z16,
    depth_colormap,
    pcd_camera,
    pcd_vis,
    T_camera_marker,
    tvec,
    marker_center_uv,
    marker_center_depth_m,
    camera_matrix,
    dist_coeffs,
    intr,
    depth_scale_m,
):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = SAVE_ROOT / f"capture_{timestamp}"
    save_dir.mkdir(parents=True, exist_ok=True)

    cv2.imwrite(str(save_dir / "color_bgr.png"), color_bgr)
    cv2.imwrite(str(save_dir / "depth_raw_z16.png"), depth_z16)
    cv2.imwrite(str(save_dir / "depth_colormap.png"), depth_colormap)

    o3d.io.write_point_cloud(str(save_dir / "scene_camera_frame.ply"), pcd_camera)
    o3d.io.write_point_cloud(str(save_dir / "scene_visualization_flipped.ply"), pcd_vis)

    np.save(str(save_dir / "T_camera_marker.npy"), T_camera_marker)
    np.save(str(save_dir / "camera_matrix.npy"), camera_matrix)
    np.save(str(save_dir / "dist_coeffs.npy"), dist_coeffs)

    metadata = {
        "timestamp": timestamp,
        "marker_dictionary": "DICT_6X6_250",
        "marker_id": TARGET_MARKER_ID,
        "marker_length_m": MARKER_LENGTH,
        "tvec_camera_marker_m": np.asarray(tvec).reshape(3).tolist(),
        "marker_center_uv": None
        if marker_center_uv is None
        else [float(marker_center_uv[0]), float(marker_center_uv[1])],
        "marker_center_depth_m": marker_center_depth_m,
        "depth_scale_m_per_unit": depth_scale_m,
        "color_intrinsics": {
            "width": intr.width,
            "height": intr.height,
            "fx": intr.fx,
            "fy": intr.fy,
            "ppx": intr.ppx,
            "ppy": intr.ppy,
            "model": str(intr.model),
            "coeffs": list(intr.coeffs),
        },
        "files": {
            "color": "color_bgr.png",
            "depth_raw": "depth_raw_z16.png",
            "depth_colormap": "depth_colormap.png",
            "pcd_for_algorithm": "scene_camera_frame.ply",
            "pcd_for_visualization": "scene_visualization_flipped.ply",
            "T_camera_marker": "T_camera_marker.npy",
            "camera_matrix": "camera_matrix.npy",
            "dist_coeffs": "dist_coeffs.npy",
        },
    }

    with open(save_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print(f"\n✅ Saved capture package to: {save_dir}")
    print("   - scene_camera_frame.ply: 用于算法和坐标计算，不要翻转")
    print("   - scene_visualization_flipped.ply: 只用于 Open3D 查看")
    print("   - T_camera_marker.npy: marker 在 camera 坐标系下的位姿")


# =========================
# Main program
# =========================

def main():
    SAVE_ROOT.mkdir(parents=True, exist_ok=True)

    pipeline = rs.pipeline()
    config = rs.config()

    config.enable_stream(rs.stream.depth, WIDTH, HEIGHT, rs.format.z16, FPS)
    config.enable_stream(rs.stream.color, WIDTH, HEIGHT, rs.format.bgr8, FPS)

    print("Starting RealSense...")
    profile = pipeline.start(config)

    depth_sensor = profile.get_device().first_depth_sensor()
    depth_scale_m = float(depth_sensor.get_depth_scale())
    print(f"RealSense depth scale: {depth_scale_m} m/unit")

    # Align depth to color. This is critical.
    align = rs.align(rs.stream.color)

    color_stream = profile.get_stream(rs.stream.color)
    color_intr = color_stream.as_video_stream_profile().get_intrinsics()
    camera_matrix, dist_coeffs = get_camera_matrix_from_intrinsics(color_intr)

    print("\n========== Color Camera Intrinsics ==========")
    print("fx:", color_intr.fx)
    print("fy:", color_intr.fy)
    print("cx/ppx:", color_intr.ppx)
    print("cy/ppy:", color_intr.ppy)
    print("dist coeffs:", color_intr.coeffs)
    print("camera_matrix:\n", camera_matrix)

    detector, aruco_dict, parameters, use_new_api = create_aruco_detector()

    cv2.namedWindow("RealSense ArUco + Aligned Depth", cv2.WINDOW_AUTOSIZE)

    print("\n操作指南：")
    print(" - 确保 marker ID = 23，并且 MARKER_LENGTH 设置为你的实测边长")
    print(" - 按 s：保存当前 color/depth/pointcloud/T_camera_marker")
    print(" - 按 q 或 Esc：退出")

    latest = {
        "color_bgr": None,
        "depth_z16": None,
        "depth_colormap": None,
        "T_camera_marker": None,
        "tvec": None,
        "marker_center_uv": None,
        "marker_center_depth_m": None,
    }

    try:
        # Let auto exposure stabilize
        for _ in range(20):
            pipeline.wait_for_frames()

        while True:
            frames = pipeline.wait_for_frames()
            aligned_frames = align.process(frames)

            aligned_depth_frame = aligned_frames.get_depth_frame()
            color_frame = aligned_frames.get_color_frame()

            if not aligned_depth_frame or not color_frame:
                continue

            depth_z16 = np.asanyarray(aligned_depth_frame.get_data())
            color_bgr = np.asanyarray(color_frame.get_data())

            gray = cv2.cvtColor(color_bgr, cv2.COLOR_BGR2GRAY)
            corners, ids, rejected = detect_markers(
                gray, detector, aruco_dict, parameters, use_new_api
            )

            vis = color_bgr.copy()
            T_camera_marker = None
            tvec = None
            marker_center_uv = None
            marker_center_depth_m = None

            if ids is not None:
                cv2.aruco.drawDetectedMarkers(vis, corners, ids)

                target_idx = choose_target_marker(corners, ids)

                if target_idx is not None:
                    rvec, tvec = estimate_single_marker_pose(
                        corners[target_idx], camera_matrix, dist_coeffs
                    )

                    if rvec is not None and tvec is not None:
                        T_camera_marker = rvec_tvec_to_T(rvec, tvec)

                        cv2.drawFrameAxes(
                            vis,
                            camera_matrix,
                            dist_coeffs,
                            rvec,
                            tvec,
                            MARKER_LENGTH * 0.5,
                        )

                        marker_center_uv = corners[target_idx].reshape(-1, 2).mean(axis=0)
                        u = int(round(marker_center_uv[0]))
                        v = int(round(marker_center_uv[1]))

                        marker_center_depth_m = get_median_depth_m(
                            depth_z16, u, v, depth_scale_m, window=9
                        )

                        x, y, z = np.asarray(tvec).reshape(3)
                        cv2.putText(
                            vis,
                            f"ID {TARGET_MARKER_ID}  t=[{x:.3f},{y:.3f},{z:.3f}] m",
                            (20, 30),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.65,
                            (0, 255, 0),
                            2,
                        )

                        if marker_center_depth_m is not None:
                            cv2.putText(
                                vis,
                                f"depth@center={marker_center_depth_m:.3f} m",
                                (20, 60),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.65,
                                (0, 255, 0),
                                2,
                            )

            depth_colormap = cv2.applyColorMap(
                cv2.convertScaleAbs(depth_z16, alpha=0.03), cv2.COLORMAP_JET
            )
            display = np.hstack((vis, depth_colormap))

            latest.update(
                {
                    "color_bgr": color_bgr,
                    "depth_z16": depth_z16,
                    "depth_colormap": depth_colormap,
                    "T_camera_marker": T_camera_marker,
                    "tvec": tvec,
                    "marker_center_uv": marker_center_uv,
                    "marker_center_depth_m": marker_center_depth_m,
                }
            )

            cv2.imshow("RealSense ArUco + Aligned Depth", display)

            key = cv2.waitKey(1) & 0xFF

            if key == ord("q") or key == 27:
                break

            if key == ord("s"):
                if latest["T_camera_marker"] is None:
                    print("\n⚠️ 没有检测到目标 ArUco marker，无法保存 T_camera_marker。")
                    print("   请确认 marker ID、光照、距离、是否在画面内。")
                    continue

                print("\nGenerating point cloud...")
                pcd_camera = create_point_cloud_camera_frame(
                    latest["color_bgr"], latest["depth_z16"], color_intr, depth_scale_m
                )
                pcd_vis = make_visualization_copy(pcd_camera)

                save_capture_package(
                    color_bgr=latest["color_bgr"],
                    depth_z16=latest["depth_z16"],
                    depth_colormap=latest["depth_colormap"],
                    pcd_camera=pcd_camera,
                    pcd_vis=pcd_vis,
                    T_camera_marker=latest["T_camera_marker"],
                    tvec=latest["tvec"],
                    marker_center_uv=latest["marker_center_uv"],
                    marker_center_depth_m=latest["marker_center_depth_m"],
                    camera_matrix=camera_matrix,
                    dist_coeffs=dist_coeffs,
                    intr=color_intr,
                    depth_scale_m=depth_scale_m,
                )

                if latest["marker_center_depth_m"] is not None:
                    tvec_z = float(np.asarray(latest["tvec"]).reshape(3)[2])
                    depth_z = float(latest["marker_center_depth_m"])
                    print("\nDepth sanity check:")
                    print(f"   OpenCV ArUco tvec z     : {tvec_z:.4f} m")
                    print(f"   aligned depth at center : {depth_z:.4f} m")
                    print(f"   absolute difference     : {abs(tvec_z - depth_z):.4f} m")
                    print("   正常情况下，两者应大致接近；相机很斜时会有偏差。")

    finally:
        pipeline.stop()
        cv2.destroyAllWindows()
        print("Program ended.")


if __name__ == "__main__":
    main()

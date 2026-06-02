#!/usr/bin/env python3
"""Real perception smoke test.

中文说明：真实 RealSense/ArUco 感知 smoke test，用于验证动态标定公式和
物体点从相机坐标系转换到 Kinova base 坐标系。该脚本不移动机器人。

This script does not move the robot.

It estimates:
    T_base_object = T_base_marker @ inv(T_camera_marker) @ T_camera_object

Requirements:
- RealSense depth is aligned to color.
- T_camera_marker and T_camera_object are both in the color camera frame.
- MARKER_LENGTH is 0.096 m.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from perception.aruco_detector import (  # noqa: E402
    MARKER_LENGTH,
    TARGET_MARKER_ID,
    create_aruco_detector,
    detect_target_marker_pose,
)
from perception.object_locator import locate_clicked_object  # noqa: E402
from perception.realsense_camera import RealSenseCamera  # noqa: E402
from utils.transform_utils import transform_camera_object_to_base  # noqa: E402


WINDOW_NAME = "Real Perception Test - click cup center"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Estimate cup/object pose in Kinova base frame. No robot motion."
    )
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=15)
    parser.add_argument(
        "--calibration",
        type=Path,
        default=PROJECT_ROOT / "configs" / "real_calibration.yaml",
        help="YAML file containing T_base_marker.",
    )
    parser.add_argument(
        "--base-marker-npy",
        type=Path,
        default=None,
        help="Optional .npy file containing T_base_marker. Overrides --calibration.",
    )
    parser.add_argument(
        "--depth-window",
        type=int,
        default=9,
        help="Median depth window size around clicked object pixel.",
    )
    return parser.parse_args()


def load_T_base_marker(args) -> np.ndarray:
    if args.base_marker_npy is not None:
        T = np.load(args.base_marker_npy)
        return validate_T(T, f"--base-marker-npy {args.base_marker_npy}")

    if not args.calibration.exists():
        raise FileNotFoundError(
            f"Missing calibration file: {args.calibration}\n"
            "Create it from configs/real_calibration.example.yaml, or pass "
            "--base-marker-npy /path/to/T_base_marker.npy."
        )

    with open(args.calibration, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}

    if "T_base_marker" not in data:
        raise KeyError(f"{args.calibration} must contain T_base_marker.")

    T = np.asarray(data["T_base_marker"], dtype=np.float64)
    return validate_T(T, str(args.calibration))


def validate_T(T, source: str) -> np.ndarray:
    T = np.asarray(T, dtype=np.float64)
    if T.shape != (4, 4):
        raise ValueError(f"T_base_marker from {source} must be a 4x4 matrix.")
    if not np.allclose(T[3], np.array([0.0, 0.0, 0.0, 1.0]), atol=1e-9):
        raise ValueError(f"T_base_marker from {source} has an invalid last row.")
    return T


def draw_text(vis, lines, origin=(20, 28), line_height=25):
    x, y = origin
    for idx, line in enumerate(lines):
        cv2.putText(
            vis,
            line,
            (x, y + idx * line_height),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.62,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )


def print_result(T_base_marker, marker_detection, object_detection):
    T_base_object = transform_camera_object_to_base(
        T_base_marker=T_base_marker,
        T_camera_marker=marker_detection.T_camera_marker,
        T_camera_object=object_detection.T_camera_object,
    )

    print("\n========== Real Perception Result ==========")
    print(f"MARKER_LENGTH: {MARKER_LENGTH:.3f} m")
    print("Clicked object pixel (u, v):", object_detection.pixel_uv)
    print(f"Clicked aligned depth: {object_detection.depth_m:.4f} m")
    print("Object point in camera/color frame [m]:")
    print(object_detection.point_camera)
    print("\nT_camera_marker:")
    print(marker_detection.T_camera_marker)
    print("\nT_camera_object:")
    print(object_detection.T_camera_object)
    print("\nT_base_object:")
    print(T_base_object)
    print("\nObject position in Kinova base frame [m]:")
    print(T_base_object[:3, 3])
    print("===========================================")

    return T_base_object


def main():
    args = parse_args()
    T_base_marker = load_T_base_marker(args)

    print("========== Real Perception Test ==========")
    print("This script only estimates object pose. It does not move the robot.")
    print(f"Marker dictionary: DICT_6X6_250, target ID: {TARGET_MARKER_ID}")
    print(f"MARKER_LENGTH: {MARKER_LENGTH:.3f} m")
    print("\nLoaded T_base_marker:")
    print(T_base_marker)
    print("\nControls:")
    print(" - Left click cup/object center in the color image")
    print(" - Press p to print the latest result again")
    print(" - Press c to clear clicked object")
    print(" - Press q or Esc to quit")

    clicked_uv = {"value": None}
    latest_marker = {"value": None}
    latest_object = {"value": None}
    latest_T_base_object = {"value": None}
    printed_click = {"value": None}

    def on_mouse(event, x, y, flags, userdata):
        if event == cv2.EVENT_LBUTTONDOWN:
            clicked_uv["value"] = (int(x), int(y))
            printed_click["value"] = None
            print(f"\nClicked object pixel: u={x}, v={y}")

    detector_bundle = create_aruco_detector()
    camera = RealSenseCamera(width=args.width, height=args.height, fps=args.fps)

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback(WINDOW_NAME, on_mouse)

    try:
        print("\nStarting RealSense...")
        camera.start()
        print("RealSense started. Depth is aligned to color.")

        while True:
            frames = camera.get_aligned_frames()
            if frames is None:
                continue

            detection, corners, ids, rejected = detect_target_marker_pose(
                frames.color_bgr,
                frames.camera_matrix,
                frames.dist_coeffs,
                detector_bundle=detector_bundle,
            )
            latest_marker["value"] = detection

            vis = frames.color_bgr.copy()
            if ids is not None:
                cv2.aruco.drawDetectedMarkers(vis, corners, ids)

            status_lines = []
            if detection is not None:
                cv2.drawFrameAxes(
                    vis,
                    frames.camera_matrix,
                    frames.dist_coeffs,
                    detection.rvec,
                    detection.tvec,
                    MARKER_LENGTH * 0.5,
                )
                tx, ty, tz = detection.tvec.reshape(3)
                status_lines.append(f"marker ID {TARGET_MARKER_ID}: t=[{tx:.3f},{ty:.3f},{tz:.3f}] m")
            else:
                status_lines.append(f"marker ID {TARGET_MARKER_ID}: not detected")

            if clicked_uv["value"] is not None:
                u, v = clicked_uv["value"]
                cv2.circle(vis, (u, v), 5, (0, 0, 255), -1)
                object_detection = locate_clicked_object(
                    frames.camera_matrix,
                    frames.depth_z16,
                    frames.depth_scale_m,
                    clicked_uv["value"],
                    window=args.depth_window,
                )
                latest_object["value"] = object_detection

                if object_detection is None:
                    status_lines.append(f"object click ({u},{v}): invalid depth")
                else:
                    px, py, pz = object_detection.point_camera
                    status_lines.append(
                        f"object camera: [{px:.3f},{py:.3f},{pz:.3f}] m"
                    )

                    if detection is not None:
                        T_base_object = transform_camera_object_to_base(
                            T_base_marker,
                            detection.T_camera_marker,
                            object_detection.T_camera_object,
                        )
                        latest_T_base_object["value"] = T_base_object
                        bx, by, bz = T_base_object[:3, 3]
                        status_lines.append(f"object base: [{bx:.3f},{by:.3f},{bz:.3f}] m")

                        if printed_click["value"] != clicked_uv["value"]:
                            print_result(T_base_marker, detection, object_detection)
                            printed_click["value"] = clicked_uv["value"]
            else:
                latest_object["value"] = None
                status_lines.append("click cup/object center")

            draw_text(vis, status_lines)
            depth_colormap = cv2.applyColorMap(
                cv2.convertScaleAbs(frames.depth_z16, alpha=0.03),
                cv2.COLORMAP_JET,
            )
            display = np.hstack((vis, depth_colormap))
            cv2.imshow(WINDOW_NAME, display)

            key = cv2.waitKey(1) & 0xFF
            if key in [ord("q"), 27]:
                break
            if key == ord("c"):
                clicked_uv["value"] = None
                latest_object["value"] = None
                latest_T_base_object["value"] = None
                printed_click["value"] = None
                print("\nCleared clicked object.")
            if key == ord("p"):
                if latest_marker["value"] is None:
                    print("\nNo marker pose available yet.")
                elif latest_object["value"] is None:
                    print("\nNo object click/depth available yet.")
                else:
                    print_result(T_base_marker, latest_marker["value"], latest_object["value"])

    finally:
        camera.stop()
        cv2.destroyAllWindows()
        print("Real perception test ended.")


if __name__ == "__main__":
    main()

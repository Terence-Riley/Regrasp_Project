#!/usr/bin/env python3
"""Calibrate a table workspace polygon from four clicked RealSense pixels.

This script is perception-only and does not move the robot.

Click the four table corners in clockwise or counter-clockwise order. The script
uses aligned depth plus the current ArUco pose to save the corners in Kinova
base coordinates for later point-cloud filtering.
"""

from __future__ import annotations

import argparse
import sys
import time
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
from perception.realsense_camera import RealSenseCamera, get_median_depth_m  # noqa: E402
from utils.transform_utils import deproject_pixel_to_camera, invert_T  # noqa: E402


WINDOW_NAME = "Table Workspace Calibration"


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
    parser.add_argument(
        "--calibration",
        type=Path,
        default=PROJECT_ROOT / "configs" / "real_calibration.yaml",
        help="YAML file containing T_base_marker.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "configs" / "table_workspace.yaml",
        help="Output YAML workspace file.",
    )
    parser.add_argument(
        "--depth-window",
        type=int,
        default=11,
        help="Median depth window around each clicked corner pixel.",
    )
    parser.add_argument(
        "--z-below",
        type=float,
        default=0.03,
        help="Workspace z_min is table_z_mean minus this value.",
    )
    parser.add_argument(
        "--z-above",
        type=float,
        default=0.35,
        help="Workspace z_max is table_z_mean plus this value.",
    )
    return parser.parse_args()


def load_T_base_marker(path: Path) -> np.ndarray:
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    T_base_marker = np.asarray(data["T_base_marker"], dtype=np.float64)
    if T_base_marker.shape != (4, 4):
        raise ValueError("T_base_marker must be 4x4.")
    return T_base_marker


def sort_polygon_xy(points_xyz):
    points = np.asarray(points_xyz, dtype=np.float64).reshape(-1, 3)
    center = np.mean(points[:, :2], axis=0)
    angles = np.arctan2(points[:, 1] - center[1], points[:, 0] - center[0])
    order = np.argsort(angles)
    return points[order]


def draw_text(vis, lines):
    for idx, line in enumerate(lines):
        y = 28 + idx * 24
        cv2.putText(vis, line, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (0, 0, 0), 4)
        cv2.putText(vis, line, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (0, 255, 0), 2)


def save_workspace(path: Path, clicked_pixels, corners_base_xyz, args):
    sorted_corners = sort_polygon_xy(corners_base_xyz)
    table_z_mean = float(np.mean(sorted_corners[:, 2]))
    data = {
        "table_workspace": {
            "frame": "base",
            "polygon_base_xy": [[float(p[0]), float(p[1])] for p in sorted_corners],
            "corner_points_base_xyz": [[float(v) for v in p] for p in sorted_corners],
            "clicked_pixels_uv": [[int(u), int(v)] for u, v in clicked_pixels],
            "table_z_mean": table_z_mean,
            "z_min": table_z_mean - float(args.z_below),
            "z_max": table_z_mean + float(args.z_above),
            "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "notes": "Use polygon_base_xy plus z_min/z_max to crop base-frame point clouds.",
        }
    }

    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False, allow_unicode=False)
    print(f"\nSaved table workspace to: {path}")
    print("polygon_base_xy:")
    for p in data["table_workspace"]["polygon_base_xy"]:
        print(f"  [{p[0]:.4f}, {p[1]:.4f}]")
    print(f"z_min={data['table_workspace']['z_min']:.4f}, z_max={data['table_workspace']['z_max']:.4f}")


def main():
    args = parse_args()
    T_base_marker = load_T_base_marker(args.calibration)
    detector_bundle = create_aruco_detector()
    camera = RealSenseCamera(
        width=args.width,
        height=args.height,
        fps=args.fps,
        stream_order=args.camera_stream_order,
    )

    clicked_pixels = []
    corners_base_xyz = []
    latest = {"frames": None, "marker": None}

    def on_mouse(event, x, y, flags, userdata):
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        if len(corners_base_xyz) >= 4:
            print("Already have four corners. Press r to reset or Enter/s to save.")
            return

        frames = latest["frames"]
        marker_detection = latest["marker"]
        if frames is None or marker_detection is None:
            print("No valid frame/marker yet. Keep marker visible and try again.")
            return

        depth_m = get_median_depth_m(
            frames.depth_z16,
            x,
            y,
            frames.depth_scale_m,
            window=args.depth_window,
        )
        if depth_m is None:
            print(f"Clicked ({x}, {y}) has invalid depth. Try a nearby table pixel.")
            return

        point_camera = deproject_pixel_to_camera(frames.camera_matrix, x, y, depth_m)
        T_base_camera = T_base_marker @ invert_T(marker_detection.T_camera_marker)
        point_base = (T_base_camera @ np.r_[point_camera, 1.0])[:3]

        clicked_pixels.append((int(x), int(y)))
        corners_base_xyz.append(point_base)
        print(
            f"corner {len(corners_base_xyz)}/4: pixel=({x}, {y}), "
            f"depth={depth_m:.4f} m, base=[{point_base[0]:.4f}, {point_base[1]:.4f}, {point_base[2]:.4f}]"
        )

    print("========== Table Workspace Calibration ==========")
    print("This script only reads the RealSense. It does not move the robot.")
    print(f"Marker: DICT_6X6_250 ID {TARGET_MARKER_ID}, length {MARKER_LENGTH:.3f} m")
    print("Controls:")
    print("  left click: add table corner")
    print("  u: undo last corner")
    print("  r: reset all corners")
    print("  s or Enter: save after four corners")
    print("  q or Esc: quit")

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(WINDOW_NAME, on_mouse)

    try:
        camera.start(start_attempts=3, retry_delay_s=2.0)
        print("RealSense started. Keep ArUco marker visible while clicking corners.")

        while True:
            frames = camera.get_aligned_frames()
            if frames is None:
                continue

            marker_detection, corners, ids, rejected = detect_target_marker_pose(
                frames.color_bgr,
                frames.camera_matrix,
                frames.dist_coeffs,
                detector_bundle=detector_bundle,
            )
            latest["frames"] = frames
            latest["marker"] = marker_detection

            vis = frames.color_bgr.copy()
            if ids is not None:
                cv2.aruco.drawDetectedMarkers(vis, corners, ids)
            if marker_detection is not None:
                cv2.drawFrameAxes(
                    vis,
                    frames.camera_matrix,
                    frames.dist_coeffs,
                    marker_detection.rvec,
                    marker_detection.tvec,
                    MARKER_LENGTH * 0.5,
                )

            for idx, (u, v) in enumerate(clicked_pixels):
                cv2.circle(vis, (u, v), 6, (0, 0, 255), -1)
                cv2.putText(vis, str(idx + 1), (u + 8, v - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            if len(clicked_pixels) >= 2:
                cv2.polylines(vis, [np.asarray(clicked_pixels, dtype=np.int32)], False, (0, 255, 255), 2)
            if len(clicked_pixels) == 4:
                cv2.polylines(vis, [np.asarray(clicked_pixels, dtype=np.int32)], True, (0, 255, 255), 2)

            marker_status = "marker visible" if marker_detection is not None else "marker not detected"
            draw_text(
                vis,
                [
                    f"{marker_status}; corners {len(corners_base_xyz)}/4",
                    "Click four table corners. s/Enter save, u undo, r reset, q quit.",
                ],
            )
            cv2.imshow(WINDOW_NAME, vis)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord("u") and corners_base_xyz:
                removed_px = clicked_pixels.pop()
                removed_pt = corners_base_xyz.pop()
                print(f"Removed corner pixel={removed_px}, base={removed_pt}")
            if key == ord("r"):
                clicked_pixels.clear()
                corners_base_xyz.clear()
                print("Reset all corners.")
            if key in (ord("s"), 13):
                if len(corners_base_xyz) != 4:
                    print("Need exactly four corners before saving.")
                else:
                    save_workspace(args.output, clicked_pixels, corners_base_xyz, args)
                    break

    finally:
        camera.stop()
        cv2.destroyAllWindows()
        print("Table workspace calibration closed.")


if __name__ == "__main__":
    main()

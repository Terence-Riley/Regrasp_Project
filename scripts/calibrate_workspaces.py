#!/usr/bin/env python3
"""Calibrate pickup and place workspaces from clicked RealSense pixels.

中文说明：打开 RealSense 画面，用户点击杯子散落区和放置区的四个角。
脚本利用 ArUco 动态标定把点击点转换到 Kinova base 坐标系，并保存
`configs/real_workspaces.yaml`。该脚本只读相机，不移动机器人。

This script is perception-only and does not move the robot.

Click four corners for the cup pickup/scatter area, then four corners for the
cup placement area. Keep the ArUco marker visible while clicking.
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


WINDOW_NAME = "Workspace Calibration - pickup then place"
PHASES = ("pickup", "place")


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
        default=PROJECT_ROOT / "configs" / "real_workspaces.yaml",
        help="Output YAML workspace file.",
    )
    parser.add_argument("--depth-window", type=int, default=11)
    parser.add_argument("--pickup-z-below", type=float, default=0.03)
    parser.add_argument("--pickup-z-above", type=float, default=0.35)
    parser.add_argument("--place-z-below", type=float, default=0.03)
    parser.add_argument("--place-z-above", type=float, default=0.25)
    parser.add_argument(
        "--place-slot-spacing",
        type=float,
        default=0.09,
        help="Default spacing between generated placement slots in meters.",
    )
    parser.add_argument(
        "--place-slot-margin",
        type=float,
        default=0.045,
        help="Default margin from placement area edges in meters.",
    )
    return parser.parse_args()


def load_T_base_marker(path: Path) -> np.ndarray:
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    T_base_marker = np.asarray(data["T_base_marker"], dtype=np.float64)
    if T_base_marker.shape != (4, 4):
        raise ValueError("T_base_marker must be 4x4.")
    return T_base_marker


def sort_polygon_xyz(points_xyz):
    points = np.asarray(points_xyz, dtype=np.float64).reshape(-1, 3)
    center = np.mean(points[:, :2], axis=0)
    angles = np.arctan2(points[:, 1] - center[1], points[:, 0] - center[0])
    return points[np.argsort(angles)]


def workspace_entry(name, clicked_pixels, corners_base_xyz, z_below, z_above):
    sorted_corners = sort_polygon_xyz(corners_base_xyz)
    table_z_mean = float(np.mean(sorted_corners[:, 2]))
    return {
        "frame": "base",
        "polygon_base_xy": [[float(p[0]), float(p[1])] for p in sorted_corners],
        "corner_points_base_xyz": [[float(v) for v in p] for p in sorted_corners],
        "clicked_pixels_uv": [[int(u), int(v)] for u, v in clicked_pixels],
        "table_z_mean": table_z_mean,
        "z_min": table_z_mean - float(z_below),
        "z_max": table_z_mean + float(z_above),
        "notes": f"{name} workspace polygon in Kinova base frame.",
    }


def save_workspaces(path: Path, collected, args):
    data = {
        "workspaces": {
            "pickup": workspace_entry(
                "pickup",
                collected["pickup"]["pixels"],
                collected["pickup"]["points"],
                args.pickup_z_below,
                args.pickup_z_above,
            ),
            "place": workspace_entry(
                "place",
                collected["place"]["pixels"],
                collected["place"]["points"],
                args.place_z_below,
                args.place_z_above,
            ),
        },
        "placement": {
            "slot_spacing": float(args.place_slot_spacing),
            "slot_margin": float(args.place_slot_margin),
            "slot_line": "centerline_along_long_edge_perpendicular_to_short_edge",
        },
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False, allow_unicode=False)

    print(f"\nSaved workspaces to: {path}")
    for name in PHASES:
        ws = data["workspaces"][name]
        print(f"{name}: z=[{ws['z_min']:.4f}, {ws['z_max']:.4f}], table_z={ws['table_z_mean']:.4f}")
        for p in ws["polygon_base_xy"]:
            print(f"  [{p[0]:.4f}, {p[1]:.4f}]")


def draw_text(vis, lines):
    for idx, line in enumerate(lines):
        y = 28 + idx * 24
        cv2.putText(vis, line, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (0, 0, 0), 4)
        cv2.putText(vis, line, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (0, 255, 0), 2)


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

    collected = {name: {"pixels": [], "points": []} for name in PHASES}
    phase_idx = {"value": 0}
    latest = {"frames": None, "marker": None}

    def current_phase():
        return PHASES[phase_idx["value"]]

    def on_mouse(event, x, y, flags, userdata):
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        phase = current_phase()
        if len(collected[phase]["points"]) >= 4:
            print(f"{phase} already has four corners. Press n to continue.")
            return

        frames = latest["frames"]
        marker_detection = latest["marker"]
        if frames is None or marker_detection is None:
            print("No valid frame/marker yet. Keep marker visible and try again.")
            return

        depth_m = get_median_depth_m(frames.depth_z16, x, y, frames.depth_scale_m, window=args.depth_window)
        if depth_m is None:
            print(f"Clicked ({x}, {y}) has invalid depth. Try a nearby table pixel.")
            return

        point_camera = deproject_pixel_to_camera(frames.camera_matrix, x, y, depth_m)
        T_base_camera = T_base_marker @ invert_T(marker_detection.T_camera_marker)
        point_base = (T_base_camera @ np.r_[point_camera, 1.0])[:3]

        collected[phase]["pixels"].append((int(x), int(y)))
        collected[phase]["points"].append(point_base)
        print(
            f"{phase} corner {len(collected[phase]['points'])}/4: pixel=({x}, {y}), "
            f"depth={depth_m:.4f} m, base=[{point_base[0]:.4f}, {point_base[1]:.4f}, {point_base[2]:.4f}]"
        )

    print("========== Pickup / Place Workspace Calibration ==========")
    print("This script only reads the RealSense. It does not move the robot.")
    print(f"Marker: DICT_6X6_250 ID {TARGET_MARKER_ID}, length {MARKER_LENGTH:.3f} m")
    print("Controls:")
    print("  left click: add current workspace corner")
    print("  n: next workspace after four corners")
    print("  u: undo last corner in current workspace")
    print("  r: reset current workspace")
    print("  s or Enter: save after both workspaces have four corners")
    print("  q or Esc: quit")

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(WINDOW_NAME, on_mouse)

    try:
        camera.start(start_attempts=3, retry_delay_s=2.0)
        print("RealSense started. Click PICKUP area first, then PLACE area.")

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

            colors = {"pickup": (0, 0, 255), "place": (255, 0, 0)}
            for name in PHASES:
                pixels = collected[name]["pixels"]
                color = colors[name]
                for idx, (u, v) in enumerate(pixels):
                    cv2.circle(vis, (u, v), 6, color, -1)
                    cv2.putText(vis, f"{name[0]}{idx + 1}", (u + 8, v - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
                if len(pixels) >= 2:
                    cv2.polylines(vis, [np.asarray(pixels, dtype=np.int32)], len(pixels) == 4, color, 2)

            phase = current_phase()
            marker_status = "marker visible" if marker_detection is not None else "marker not detected"
            draw_text(
                vis,
                [
                    f"{marker_status}; current={phase}; corners {len(collected[phase]['points'])}/4",
                    "Click pickup 4 corners, n, click place 4 corners, s save.",
                ],
            )
            cv2.imshow(WINDOW_NAME, vis)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord("u"):
                phase = current_phase()
                if collected[phase]["points"]:
                    px = collected[phase]["pixels"].pop()
                    pt = collected[phase]["points"].pop()
                    print(f"Removed {phase} corner pixel={px}, base={pt}")
            if key == ord("r"):
                phase = current_phase()
                collected[phase]["pixels"].clear()
                collected[phase]["points"].clear()
                print(f"Reset {phase} corners.")
            if key == ord("n"):
                phase = current_phase()
                if len(collected[phase]["points"]) != 4:
                    print(f"Need four {phase} corners before continuing.")
                elif phase_idx["value"] < len(PHASES) - 1:
                    phase_idx["value"] += 1
                    print(f"Now click {current_phase()} area corners.")
            if key in (ord("s"), 13):
                if any(len(collected[name]["points"]) != 4 for name in PHASES):
                    print("Need four pickup corners and four place corners before saving.")
                else:
                    save_workspaces(args.output, collected, args)
                    break

    finally:
        camera.stop()
        cv2.destroyAllWindows()
        print("Workspace calibration closed.")


if __name__ == "__main__":
    main()

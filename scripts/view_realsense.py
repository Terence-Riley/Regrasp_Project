#!/usr/bin/env python3
"""Display RealSense color and aligned depth frames.

This script does not move the robot. It is intended for a quick visual check of
what the D435i currently sees.
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from perception.realsense_camera import RealSenseCamera, get_median_depth_m  # noqa: E402


WINDOW_NAME = "RealSense Viewer - color | aligned depth"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Show RealSense color and aligned depth images. No robot motion."
    )
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=15)
    parser.add_argument(
        "--depth-max",
        type=float,
        default=2.0,
        help="Depth value in meters mapped to the far end of the color map.",
    )
    parser.add_argument(
        "--depth-window",
        type=int,
        default=9,
        help="Median window size for the displayed center depth.",
    )
    parser.add_argument(
        "--save-dir",
        type=Path,
        default=PROJECT_ROOT / "debug_realsense_viewer",
        help="Directory used when pressing s to save frames.",
    )
    parser.add_argument(
        "--stream-order",
        choices=["depth_first", "color_first", "both"],
        default="depth_first",
        help="Stream enable order. Try color_first if Windows start fails.",
    )
    return parser.parse_args()


def make_depth_colormap(depth_z16: np.ndarray, depth_scale_m: float, depth_max_m: float):
    depth_m = depth_z16.astype(np.float32) * float(depth_scale_m)
    valid = depth_m > 0.0

    depth_norm = np.zeros(depth_m.shape, dtype=np.uint8)
    if depth_max_m > 0:
        clipped = np.clip(depth_m, 0.0, depth_max_m)
        depth_norm = np.uint8((clipped / depth_max_m) * 255.0)

    depth_color = cv2.applyColorMap(depth_norm, cv2.COLORMAP_JET)
    depth_color[~valid] = (0, 0, 0)
    return depth_color, depth_m


def draw_overlay(color_bgr, depth_color, center_depth_m, fps_text):
    h, w = color_bgr.shape[:2]
    center = (w // 2, h // 2)

    color_vis = color_bgr.copy()
    depth_vis = depth_color.copy()

    for vis in (color_vis, depth_vis):
        cv2.drawMarker(
            vis,
            center,
            (0, 255, 255),
            markerType=cv2.MARKER_CROSS,
            markerSize=18,
            thickness=2,
        )

    depth_text = "center depth: invalid"
    if center_depth_m is not None:
        depth_text = f"center depth: {center_depth_m:.3f} m"

    lines = [
        depth_text,
        fps_text,
        "q/Esc: quit   s: save color+depth",
    ]
    for idx, line in enumerate(lines):
        y = 26 + idx * 24
        cv2.putText(color_vis, line, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (0, 0, 0), 4)
        cv2.putText(color_vis, line, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (0, 255, 0), 2)

    cv2.putText(depth_vis, "aligned depth", (12, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4)
    cv2.putText(depth_vis, "aligned depth", (12, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    return np.hstack([color_vis, depth_vis])


def save_frames(save_dir: Path, color_bgr, depth_z16, depth_color):
    save_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    color_path = save_dir / f"realsense_color_{stamp}.png"
    depth_raw_path = save_dir / f"realsense_depth_z16_{stamp}.png"
    depth_vis_path = save_dir / f"realsense_depth_vis_{stamp}.png"

    cv2.imwrite(str(color_path), color_bgr)
    cv2.imwrite(str(depth_raw_path), depth_z16)
    cv2.imwrite(str(depth_vis_path), depth_color)

    print("\nSaved:")
    print(f"  {color_path}")
    print(f"  {depth_raw_path}")
    print(f"  {depth_vis_path}")


def main():
    args = parse_args()

    print("========== RealSense Viewer ==========")
    print("This script only displays camera frames. It does not move the robot.")
    print(f"Resolution: {args.width}x{args.height}@{args.fps}")
    print("Controls: q/Esc quit, s save current color/depth frames")

    camera = RealSenseCamera(
        width=args.width,
        height=args.height,
        fps=args.fps,
        stream_order=args.stream_order,
    )

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)

    tick_meter = cv2.TickMeter()
    fps_text = "fps: --"

    try:
        camera.start(warmup_frames=20, start_attempts=2, hardware_reset_on_fail=False)
        print(f"RealSense started. Depth scale: {camera.depth_scale_m:.6f} m/unit")

        while True:
            tick_meter.reset()
            tick_meter.start()

            frames = camera.get_aligned_frames()
            if frames is None:
                continue

            depth_color, _depth_m = make_depth_colormap(
                frames.depth_z16,
                frames.depth_scale_m,
                args.depth_max,
            )
            center_depth_m = get_median_depth_m(
                frames.depth_z16,
                frames.depth_z16.shape[1] // 2,
                frames.depth_z16.shape[0] // 2,
                frames.depth_scale_m,
                window=args.depth_window,
            )

            tick_meter.stop()
            elapsed = tick_meter.getTimeSec()
            if elapsed > 0:
                fps_text = f"fps: {1.0 / elapsed:.1f}"

            display = draw_overlay(frames.color_bgr, depth_color, center_depth_m, fps_text)
            cv2.imshow(WINDOW_NAME, display)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord("s"):
                save_frames(args.save_dir, frames.color_bgr, frames.depth_z16, depth_color)

    except KeyboardInterrupt:
        print("\nInterrupted by user.")
    finally:
        camera.stop()
        cv2.destroyAllWindows()
        print("RealSense viewer closed.")


if __name__ == "__main__":
    main()

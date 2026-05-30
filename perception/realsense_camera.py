"""RealSense D435i utilities.

All depth frames returned by this module are aligned to the color stream.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


WIDTH = 640
HEIGHT = 480
FPS = 15


@dataclass(frozen=True)
class CameraIntrinsics:
    width: int
    height: int
    fx: float
    fy: float
    ppx: float
    ppy: float
    coeffs: tuple[float, ...]
    model: str


@dataclass(frozen=True)
class AlignedFrames:
    color_bgr: np.ndarray
    depth_z16: np.ndarray
    camera_matrix: np.ndarray
    dist_coeffs: np.ndarray
    intrinsics: CameraIntrinsics
    depth_scale_m: float


def _require_rs():
    import pyrealsense2 as rs

    return rs


def intrinsics_to_camera_matrix(intr):
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


def freeze_intrinsics(intr) -> CameraIntrinsics:
    return CameraIntrinsics(
        width=int(intr.width),
        height=int(intr.height),
        fx=float(intr.fx),
        fy=float(intr.fy),
        ppx=float(intr.ppx),
        ppy=float(intr.ppy),
        coeffs=tuple(float(c) for c in intr.coeffs),
        model=str(intr.model),
    )


def get_median_depth_m(depth_image_z16, u, v, depth_scale_m, window=7):
    h, w = depth_image_z16.shape
    half = window // 2
    u0 = max(0, int(u) - half)
    u1 = min(w, int(u) + half + 1)
    v0 = max(0, int(v) - half)
    v1 = min(h, int(v) + half + 1)

    patch = depth_image_z16[v0:v1, u0:u1].astype(np.float64)
    valid = patch[patch > 0]
    if valid.size == 0:
        return None
    return float(np.median(valid) * depth_scale_m)


class RealSenseCamera:
    """Small wrapper for color-aligned RGB-D capture."""

    def __init__(self, width=WIDTH, height=HEIGHT, fps=FPS):
        rs = _require_rs()
        self._rs = rs
        self.width = width
        self.height = height
        self.fps = fps
        self.pipeline = None
        self.config = None
        self.align = rs.align(rs.stream.color)
        self.profile = None
        self.depth_scale_m = None
        self.camera_matrix = None
        self.dist_coeffs = None
        self.intrinsics = None

    def _make_pipeline_and_config(self, fps):
        pipeline = self._rs.pipeline()
        config = self._rs.config()
        config.enable_stream(self._rs.stream.depth, self.width, self.height, self._rs.format.z16, fps)
        config.enable_stream(self._rs.stream.color, self.width, self.height, self._rs.format.bgr8, fps)
        return pipeline, config

    def start(self, warmup_frames=20):
        fps_candidates = []
        for fps in [self.fps, 15, 30, 6]:
            if fps not in fps_candidates:
                fps_candidates.append(fps)

        last_error = None
        for fps in fps_candidates:
            self.pipeline, self.config = self._make_pipeline_and_config(fps)
            try:
                print(f"Starting RealSense {self.width}x{self.height}@{fps}...")
                self.profile = self.pipeline.start(self.config)
                self.fps = fps
                break
            except RuntimeError as exc:
                last_error = exc
                self.profile = None
                try:
                    self.pipeline.stop()
                except RuntimeError:
                    pass
                print(f"RealSense start failed at {fps} FPS: {exc}")

        if self.profile is None:
            raise RuntimeError(f"Could not start RealSense after trying FPS {fps_candidates}") from last_error

        depth_sensor = self.profile.get_device().first_depth_sensor()
        self.depth_scale_m = float(depth_sensor.get_depth_scale())

        color_stream = self.profile.get_stream(self._rs.stream.color)
        color_intr = color_stream.as_video_stream_profile().get_intrinsics()
        self.camera_matrix, self.dist_coeffs = intrinsics_to_camera_matrix(color_intr)
        self.intrinsics = freeze_intrinsics(color_intr)

        for _ in range(warmup_frames):
            self.pipeline.wait_for_frames()
        return self

    def stop(self):
        if self.profile is not None:
            self.pipeline.stop()
            self.profile = None

    def get_aligned_frames(self) -> AlignedFrames | None:
        frames = self.pipeline.wait_for_frames()
        aligned_frames = self.align.process(frames)
        aligned_depth_frame = aligned_frames.get_depth_frame()
        color_frame = aligned_frames.get_color_frame()

        if not aligned_depth_frame or not color_frame:
            return None

        return AlignedFrames(
            color_bgr=np.asanyarray(color_frame.get_data()),
            depth_z16=np.asanyarray(aligned_depth_frame.get_data()),
            camera_matrix=self.camera_matrix,
            dist_coeffs=self.dist_coeffs,
            intrinsics=self.intrinsics,
            depth_scale_m=self.depth_scale_m,
        )

"""Object localization helpers for clicked RGB-D targets."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from utils.transform_utils import deproject_pixel_to_camera, point_to_T_camera_object


@dataclass(frozen=True)
class ClickObjectDetection:
    pixel_uv: tuple[int, int]
    depth_m: float
    point_camera: np.ndarray
    T_camera_object: np.ndarray


def locate_clicked_object(
    K: np.ndarray,
    depth_z16: np.ndarray,
    depth_scale_m: float,
    pixel_uv: tuple[int, int],
    window: int = 9,
) -> ClickObjectDetection | None:
    """Use aligned depth around a clicked pixel to estimate T_camera_object."""
    from perception.realsense_camera import get_median_depth_m

    u, v = pixel_uv
    depth_m = get_median_depth_m(depth_z16, u, v, depth_scale_m, window=window)
    if depth_m is None:
        return None

    point_camera = deproject_pixel_to_camera(K, u, v, depth_m)
    T_camera_object = point_to_T_camera_object(point_camera)

    return ClickObjectDetection(
        pixel_uv=(int(u), int(v)),
        depth_m=float(depth_m),
        point_camera=point_camera,
        T_camera_object=T_camera_object,
    )

"""Homogeneous transform helpers.

Naming convention used throughout the project:
T_A_B means the pose of frame B expressed in frame A.
"""

from __future__ import annotations

import numpy as np


def make_T(rotation: np.ndarray | None = None, translation: np.ndarray | None = None) -> np.ndarray:
    """Create a 4x4 transform from a 3x3 rotation and 3D translation."""
    T = np.eye(4, dtype=np.float64)
    if rotation is not None:
        T[:3, :3] = np.asarray(rotation, dtype=np.float64).reshape(3, 3)
    if translation is not None:
        T[:3, 3] = np.asarray(translation, dtype=np.float64).reshape(3)
    return T


def quat_xyzw_to_R(quat_xyzw: np.ndarray) -> np.ndarray:
    """Convert an [x, y, z, w] quaternion to a 3x3 rotation matrix."""
    x, y, z, w = np.asarray(quat_xyzw, dtype=np.float64).reshape(4)
    norm = np.linalg.norm([x, y, z, w])
    if norm == 0.0:
        raise ValueError("Quaternion norm is zero.")
    x, y, z, w = x / norm, y / norm, z / norm, w / norm

    return np.array(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def pose_to_T(pos: np.ndarray, quat_xyzw: np.ndarray) -> np.ndarray:
    """Convert position and [x, y, z, w] quaternion to a 4x4 transform."""
    return make_T(rotation=quat_xyzw_to_R(quat_xyzw), translation=pos)


def invert_T(T_A_B: np.ndarray) -> np.ndarray:
    """Return T_B_A from T_A_B."""
    T_A_B = np.asarray(T_A_B, dtype=np.float64).reshape(4, 4)
    R_A_B = T_A_B[:3, :3]
    t_A_B = T_A_B[:3, 3]

    T_B_A = np.eye(4, dtype=np.float64)
    T_B_A[:3, :3] = R_A_B.T
    T_B_A[:3, 3] = -R_A_B.T @ t_A_B
    return T_B_A


def rvec_tvec_to_T(rvec: np.ndarray, tvec: np.ndarray) -> np.ndarray:
    """Convert OpenCV Rodrigues rvec/tvec to a 4x4 transform."""
    import cv2

    R, _ = cv2.Rodrigues(np.asarray(rvec, dtype=np.float64).reshape(3))
    return make_T(R, np.asarray(tvec, dtype=np.float64).reshape(3))


def transform_camera_object_to_base(
    T_base_marker: np.ndarray,
    T_camera_marker: np.ndarray,
    T_camera_object: np.ndarray,
) -> np.ndarray:
    """Dynamic calibration formula for object pose in the Kinova base frame."""
    return (
        np.asarray(T_base_marker, dtype=np.float64).reshape(4, 4)
        @ invert_T(T_camera_marker)
        @ np.asarray(T_camera_object, dtype=np.float64).reshape(4, 4)
    )


def deproject_pixel_to_camera(K: np.ndarray, u: float, v: float, depth_m: float) -> np.ndarray:
    """Pixel plus aligned depth to a 3D point in OpenCV/RealSense color camera frame."""
    K = np.asarray(K, dtype=np.float64).reshape(3, 3)
    z = float(depth_m)
    x = (float(u) - K[0, 2]) * z / K[0, 0]
    y = (float(v) - K[1, 2]) * z / K[1, 1]
    return np.array([x, y, z], dtype=np.float64)


def point_to_T_camera_object(point_camera: np.ndarray) -> np.ndarray:
    """Create T_camera_object with identity orientation from a camera-frame point."""
    return make_T(translation=np.asarray(point_camera, dtype=np.float64).reshape(3))

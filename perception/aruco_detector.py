"""OpenCV ArUco marker detection helpers."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from utils.transform_utils import rvec_tvec_to_T


MARKER_LENGTH = 0.096
TARGET_MARKER_ID = 23
ARUCO_DICT_NAME = "DICT_6X6_250"
ARUCO_DICT = cv2.aruco.DICT_6X6_250


@dataclass(frozen=True)
class ArucoDetection:
    marker_id: int
    corners: np.ndarray
    rvec: np.ndarray
    tvec: np.ndarray
    T_camera_marker: np.ndarray
    center_uv: np.ndarray


def create_aruco_detector():
    """Create an ArUco detector compatible with new and old OpenCV APIs."""
    aruco_dict = cv2.aruco.getPredefinedDictionary(ARUCO_DICT)

    if hasattr(cv2.aruco, "ArucoDetector"):
        parameters = cv2.aruco.DetectorParameters()
        detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)
        return detector, aruco_dict, parameters, True

    parameters = cv2.aruco.DetectorParameters_create()
    return None, aruco_dict, parameters, False


def detect_markers(gray, detector, aruco_dict, parameters, use_new_api):
    if use_new_api:
        return detector.detectMarkers(gray)
    return cv2.aruco.detectMarkers(gray, aruco_dict, parameters=parameters)


def choose_target_marker(corners, ids, target_marker_id: int = TARGET_MARKER_ID):
    """Return the index of the target marker, or None if it is not visible."""
    if ids is None:
        return None
    ids_flat = ids.flatten()
    matches = np.where(ids_flat == target_marker_id)[0]
    return None if len(matches) == 0 else int(matches[0])


def estimate_single_marker_pose(
    corners_one,
    camera_matrix,
    dist_coeffs,
    marker_length: float = MARKER_LENGTH,
):
    """Estimate marker pose as T_camera_marker."""
    if hasattr(cv2.aruco, "estimatePoseSingleMarkers"):
        rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(
            [corners_one], marker_length, camera_matrix, dist_coeffs
        )
        rvec = rvecs[0][0]
        tvec = tvecs[0][0]
        return rvec, tvec, rvec_tvec_to_T(rvec, tvec)

    half = marker_length / 2.0
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
    success, rvec, tvec = cv2.solvePnP(obj_points, img_points, camera_matrix, dist_coeffs)
    if not success:
        return None, None, None
    rvec = rvec.reshape(3)
    tvec = tvec.reshape(3)
    return rvec, tvec, rvec_tvec_to_T(rvec, tvec)


def detect_target_marker_pose(
    color_bgr,
    camera_matrix,
    dist_coeffs,
    detector_bundle=None,
    target_marker_id: int = TARGET_MARKER_ID,
    marker_length: float = MARKER_LENGTH,
):
    """Detect the configured marker in a BGR image and return an ArucoDetection."""
    if detector_bundle is None:
        detector_bundle = create_aruco_detector()
    detector, aruco_dict, parameters, use_new_api = detector_bundle

    gray = cv2.cvtColor(color_bgr, cv2.COLOR_BGR2GRAY)
    corners, ids, rejected = detect_markers(gray, detector, aruco_dict, parameters, use_new_api)
    target_idx = choose_target_marker(corners, ids, target_marker_id)
    if target_idx is None:
        return None, corners, ids, rejected

    rvec, tvec, T_camera_marker = estimate_single_marker_pose(
        corners[target_idx], camera_matrix, dist_coeffs, marker_length
    )
    if T_camera_marker is None:
        return None, corners, ids, rejected

    center_uv = corners[target_idx].reshape(-1, 2).mean(axis=0)
    detection = ArucoDetection(
        marker_id=target_marker_id,
        corners=corners[target_idx],
        rvec=rvec,
        tvec=tvec,
        T_camera_marker=T_camera_marker,
        center_uv=center_uv,
    )
    return detection, corners, ids, rejected


def draw_detection(vis_bgr, detection: ArucoDetection, camera_matrix, dist_coeffs):
    cv2.drawFrameAxes(
        vis_bgr,
        camera_matrix,
        dist_coeffs,
        detection.rvec,
        detection.tvec,
        MARKER_LENGTH * 0.5,
    )
    return vis_bgr

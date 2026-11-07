"""Ray-plane geometric projection from downward drone cameras to the ground plane (z = 0).

Projects 2D image pixel coordinates (u, v) to 3D world ground coordinates (x, y, 0)
using camera intrinsics K and 6-DoF camera pose, and computes the 4-corner camera
ground footprint polygon.

Inputs:
    Pixel coordinates (u, v), 3x3 intrinsics matrix K, camera position (x, y, z),
    and camera orientation quaternion (w, x, y, z).
Outputs:
    Ground coordinates (x, y) inside the venue bounds, or 4-vertex footprint polygon.
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import numpy as np

import config


def quat_wxyz_to_rotation_matrix(quat_wxyz: Sequence[float]) -> np.ndarray:
    """Convert a quaternion (w, x, y, z) to a 3x3 rotation matrix."""
    w, x, y, z = (
        float(quat_wxyz[0]),
        float(quat_wxyz[1]),
        float(quat_wxyz[2]),
        float(quat_wxyz[3]),
    )
    norm = np.hypot(np.hypot(w, x), np.hypot(y, z))
    if norm < 1e-9:
        return np.eye(3, dtype=np.float64)
    w, x, y, z = w / norm, x / norm, y / norm, z / norm
    return np.array(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - w * z), 2.0 * (x * z + w * y)],
            [2.0 * (x * y + w * z), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - w * x)],
            [2.0 * (x * z - w * y), 2.0 * (y * z + w * x), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def _camera_ray_in_world(
    u: float,
    v: float,
    intrinsics_3x3: Sequence[Sequence[float]],
    camera_orientation_wxyz: Sequence[float],
) -> np.ndarray:
    """Compute the 3D ray direction in world coordinates for pixel (u, v)."""
    fx = float(intrinsics_3x3[0][0])
    fy = float(intrinsics_3x3[1][1])
    cx = float(intrinsics_3x3[0][2])
    cy = float(intrinsics_3x3[1][2])

    x_cam = (float(u) - cx) / fx
    y_cam = (float(v) - cy) / fy

    rot = quat_wxyz_to_rotation_matrix(camera_orientation_wxyz)
    # Check whether `rot` already points the camera's +Z axis downward (z < 0)
    # or represents attitude relative to a level nadir mount.
    optical_axis_world = rot @ np.array([0.0, 0.0, 1.0], dtype=np.float64)
    if optical_axis_world[2] < -1e-6:
        ray_cam = np.array([x_cam, y_cam, 1.0], dtype=np.float64)
        return rot @ ray_cam
    # Nadir mount convention: +X_cam -> +X_world, +Y_cam -> +Y_world, +Z_cam -> -Z_world
    ray_nadir = np.array([x_cam, y_cam, -1.0], dtype=np.float64)
    return rot @ ray_nadir


def project_pixel_to_ground(
    u: float,
    v: float,
    camera_position_xyz: Sequence[float],
    camera_orientation_wxyz: Sequence[float],
    intrinsics_3x3: Sequence[Sequence[float]],
    venue_width: float = config.VENUE_WIDTH,
    venue_height: float = config.VENUE_HEIGHT,
    clip_to_venue: bool = False,
) -> Optional[Tuple[float, float]]:
    """Intersect the camera ray through pixel (u, v) with the ground plane z = 0.

    Returns None if the ray does not intersect z = 0 in front of the camera or if the
    intersection falls outside the venue (when `clip_to_venue` is False).
    """
    cam_x, cam_y, cam_z = (
        float(camera_position_xyz[0]),
        float(camera_position_xyz[1]),
        float(camera_position_xyz[2]),
    )
    if cam_z <= 0.0:
        return None

    ray_d = _camera_ray_in_world(u, v, intrinsics_3x3, camera_orientation_wxyz)
    dz = float(ray_d[2])
    if dz >= -1e-6:
        return None

    t = -cam_z / dz
    gx = cam_x + t * float(ray_d[0])
    gy = cam_y + t * float(ray_d[1])

    if clip_to_venue:
        gx = float(np.clip(gx, 0.0, venue_width))
        gy = float(np.clip(gy, 0.0, venue_height))
        return (gx, gy)

    if 0.0 <= gx <= venue_width and 0.0 <= gy <= venue_height:
        return (gx, gy)
    return None


def world_point_to_pixel(
    world_xy: Tuple[float, float],
    camera_position_xyz: Sequence[float],
    camera_orientation_wxyz: Sequence[float],
    intrinsics_3x3: Sequence[Sequence[float]],
    width: int = config.CAMERA_WIDTH,
    height: int = config.CAMERA_HEIGHT,
) -> Optional[Tuple[float, float]]:
    """Project a ground point (x, y, z=0) into image pixel coordinates (u, v)."""
    cam_x, cam_y, cam_z = (
        float(camera_position_xyz[0]),
        float(camera_position_xyz[1]),
        float(camera_position_xyz[2]),
    )
    if cam_z <= 0.1:
        return None

    rel_world = np.array(
        [float(world_xy[0]) - cam_x, float(world_xy[1]) - cam_y, -cam_z],
        dtype=np.float64,
    )
    rot = quat_wxyz_to_rotation_matrix(camera_orientation_wxyz)
    optical_axis_world = rot @ np.array([0.0, 0.0, 1.0], dtype=np.float64)
    if optical_axis_world[2] < -1e-6:
        p_cam = rot.T @ rel_world
        z_depth = float(p_cam[2])
        x_n = float(p_cam[0]) / max(z_depth, 1e-6)
        y_n = float(p_cam[1]) / max(z_depth, 1e-6)
    else:
        p_nadir = rot.T @ rel_world
        z_depth = -float(p_nadir[2])
        x_n = float(p_nadir[0]) / max(z_depth, 1e-6)
        y_n = float(p_nadir[1]) / max(z_depth, 1e-6)

    if z_depth <= 0.1:
        return None

    fx = float(intrinsics_3x3[0][0])
    fy = float(intrinsics_3x3[1][1])
    cx = float(intrinsics_3x3[0][2])
    cy = float(intrinsics_3x3[1][2])

    u = cx + fx * x_n
    v = cy + fy * y_n
    if 0.0 <= u < width and 0.0 <= v < height:
        return (u, v)
    return None


def compute_footprint_polygon(
    camera_position_xyz: Sequence[float],
    camera_orientation_wxyz: Sequence[float],
    intrinsics_3x3: Sequence[Sequence[float]],
    width: int = config.CAMERA_WIDTH,
    height: int = config.CAMERA_HEIGHT,
    venue_width: float = config.VENUE_WIDTH,
    venue_height: float = config.VENUE_HEIGHT,
) -> List[Tuple[float, float]]:
    """Project the 4 image corners to the ground plane (z = 0), clipped to the venue."""
    corners_uv = [
        (0.0, 0.0),
        (float(width), 0.0),
        (float(width), float(height)),
        (0.0, float(height)),
    ]
    footprint: List[Tuple[float, float]] = []
    for u, v in corners_uv:
        pt = project_pixel_to_ground(
            u=u,
            v=v,
            camera_position_xyz=camera_position_xyz,
            camera_orientation_wxyz=camera_orientation_wxyz,
            intrinsics_3x3=intrinsics_3x3,
            venue_width=venue_width,
            venue_height=venue_height,
            clip_to_venue=True,
        )
        if pt is not None:
            footprint.append(pt)
    return footprint

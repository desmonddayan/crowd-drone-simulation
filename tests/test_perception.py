"""Unit tests for perception/projector.py and perception/detector.py."""

from __future__ import annotations

import cv2
import numpy as np

import config
from interfaces.contract import FrameMessage
from perception.detector import CrowdPerceptionPipeline
from perception.projector import (
    compute_footprint_polygon,
    project_pixel_to_ground,
    world_point_to_pixel,
)


def test_ground_projection_center_and_round_trip() -> None:
    intrinsics = config.get_camera_intrinsics(width=320, height=240, fov_deg=60.0)
    cam_pos = (40.0, 30.0, 20.0)
    cam_quat = (1.0, 0.0, 0.0, 0.0)

    # Principal point (cx=160, cy=120) must project directly below the drone at (40.0, 30.0)
    pt_center = project_pixel_to_ground(160.0, 120.0, cam_pos, cam_quat, intrinsics)
    assert pt_center is not None
    assert np.allclose(pt_center, (40.0, 30.0), atol=1e-4)

    # Known offset ground point round-trips through world_point_to_pixel -> project_pixel_to_ground
    expected_xy = (44.5, 27.2)
    uv = world_point_to_pixel(expected_xy, cam_pos, cam_quat, intrinsics, 320, 240)
    assert uv is not None
    projected_xy = project_pixel_to_ground(uv[0], uv[1], cam_pos, cam_quat, intrinsics)
    assert projected_xy is not None
    assert np.allclose(projected_xy, expected_xy, atol=1e-3)


def test_outside_venue_points_are_discarded() -> None:
    intrinsics = config.get_camera_intrinsics(width=320, height=240, fov_deg=60.0)
    # Drone near West edge (x=2.0); left edge of image (u=0) projects to x < 0
    cam_pos = (2.0, 30.0, 20.0)
    cam_quat = (1.0, 0.0, 0.0, 0.0)
    pt_out = project_pixel_to_ground(0.0, 120.0, cam_pos, cam_quat, intrinsics, clip_to_venue=False)
    assert pt_out is None

    # Footprint polygon clips to venue [0, VENUE_WIDTH] x [0, VENUE_HEIGHT]
    fp = compute_footprint_polygon(cam_pos, cam_quat, intrinsics, 320, 240)
    assert len(fp) == 4
    for x, y in fp:
        assert 0.0 <= x <= config.VENUE_WIDTH
        assert 0.0 <= y <= config.VENUE_HEIGHT


def test_blob_perception_pipeline_end_to_end() -> None:
    pipeline = CrowdPerceptionPipeline(use_yolo=False)
    intrinsics = config.get_camera_intrinsics(width=320, height=240, fov_deg=60.0)
    cam_pos = (40.0, 30.0, 20.0)
    cam_quat = (1.0, 0.0, 0.0, 0.0)

    # Render synthetic image with 3 people at known world coordinates
    true_world_pts = [(36.0, 28.0), (40.0, 30.0), (43.5, 32.5)]
    rgb = np.full((240, 320, 3), 18, dtype=np.uint8)
    for wx, wy in true_world_pts:
        uv = world_point_to_pixel((wx, wy), cam_pos, cam_quat, intrinsics, 320, 240)
        assert uv is not None
        px, py = int(round(uv[0])), int(round(uv[1]))
        cv2.circle(rgb, (px, py), 3, (255, 200, 60), -1)

    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    ok, buf = cv2.imencode(".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
    assert ok

    msg = FrameMessage(
        drone_id=0,
        timestamp=1.0,
        jpeg_bytes=buf.tobytes(),
        camera_position_xyz=cam_pos,
        camera_orientation_wxyz=cam_quat,
        intrinsics_3x3=intrinsics,
        width=320,
        height=240,
    )
    res = pipeline.process_frame(msg)
    assert len(res.detections) == 3
    assert len(res.ground_points_xy) == 3
    assert res.confidence > 0.5

    # Each true world point should have a detected ground point within 0.6m
    for tx, ty in true_world_pts:
        min_err = min(np.hypot(gx - tx, gy - ty) for gx, gy in res.ground_points_xy)
        assert min_err < 0.65

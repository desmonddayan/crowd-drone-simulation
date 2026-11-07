"""Mock crowd-and-drone simulator (no Isaac Sim required).

Emulates the Isaac Sim process using the shared 2D social-force crowd model
(`sim/crowd_dynamics.py`) and renders synthetic downward RGB camera frames per drone
(dark concert venue background, barriers, coloured spotlight cones, and pedestrians
as bright blobs). Supports round-robin camera capture when N=25 drones are active,
and can operate either over ZeroMQ or in-process.

Inputs:
    `DroneCommand` messages from the fleet coordinator.
Outputs:
    `FrameMessage` (JPEG bytes + pose + intrinsics) and `DroneState` telemetry.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

import config
from interfaces.contract import DroneCommand, DroneState, FrameMessage
from perception.projector import world_point_to_pixel
from sim.crowd_dynamics import CrowdAndDroneWorld


class MockCrowdSimulator:
    """Synthetic downward-camera & crowd simulator matching the Isaac Sim contract."""

    def __init__(
        self,
        n_drones: int = config.N_DRONES,
        n_people: int = config.N_PEOPLE,
        seed: int = 42,
        surge_start_seconds: float = config.SURGE_START_SECONDS,
        camera_width: int = config.CAMERA_WIDTH,
        camera_height: int = config.CAMERA_HEIGHT,
        max_cameras_per_step: int = config.MAX_CAMERAS_PER_STEP,
    ) -> None:
        self.world = CrowdAndDroneWorld(
            n_drones=n_drones,
            n_people=n_people,
            seed=seed,
            surge_start_seconds=surge_start_seconds,
        )
        self.camera_width = int(camera_width)
        self.camera_height = int(camera_height)
        self.intrinsics = config.get_camera_intrinsics(
            width=self.camera_width,
            height=self.camera_height,
            fov_deg=config.CAMERA_FOV_DEG,
        )
        self.max_cameras_per_step = int(max_cameras_per_step)
        self._rr_Cursor: int = 0

    @property
    def sim_time(self) -> float:
        return self.world.sim_time

    def apply_commands(self, commands: List[DroneCommand]) -> None:
        """Apply a batch of DroneCommand objects from the agent process."""
        for cmd in commands:
            self.world.apply_command(cmd)

    def step(self, dt: float = 0.1, capture_all_cameras: bool = False) -> Tuple[List[FrameMessage], List[DroneState]]:
        """Advance physics by `dt` and render downward camera frames (round-robin or all)."""
        self.world.step(dt=dt)
        states = self.world.get_drone_states()

        n = self.world.n_drones
        if capture_all_cameras or n <= self.max_cameras_per_step:
            cam_ids = list(range(n))
        else:
            cam_ids = [
                (self._rr_Cursor + k) % n for k in range(self.max_cameras_per_step)
            ]
            self._rr_Cursor = (self._rr_Cursor + self.max_cameras_per_step) % n

        frames = [self.render_drone_frame(d_id) for d_id in cam_ids]
        return frames, states

    def render_drone_frame(self, drone_id: int) -> FrameMessage:
        """Render the downward RGB camera view for `drone_id` and encode as JPEG."""
        cam_pos = tuple(float(v) for v in self.world.drone_pos[drone_id])
        cam_quat = (1.0, 0.0, 0.0, 0.0)
        w, h = self.camera_width, self.camera_height

        # 1. Night-time concert ground background (dark slate)
        rgb = np.full((h, w, 3), 16, dtype=np.uint8)

        # 2. Draw drone spotlight cone on the ground (soft additive tint so people remain distinct)
        cmd = self.world.drone_commands[drone_id]
        intensity = self.world.get_modulated_light_intensity(drone_id)
        if intensity > 0.05:
            aim_uv = world_point_to_pixel(
                cmd.light_aim_xy, cam_pos, cam_quat, self.intrinsics, w, h
            )
            center_uv = (
                (int(round(0.5 * (w * 0.5 + aim_uv[0]))), int(round(0.5 * (h * 0.5 + aim_uv[1]))))
                if aim_uv is not None
                else (w // 2, h // 2)
            )
            cone_rad_px = int(
                round(
                    0.38
                    * w
                    * math.tan(math.radians(config.LIGHT_CONE_HALF_ANGLE_DEG))
                    / math.tan(math.radians(config.CAMERA_FOV_DEG * 0.5))
                )
            )
            overlay = np.zeros_like(rgb)
            # Keep blue/green/red spotlight tint subtle (< 70) so bright warm person cores stand out
            col_rgb = (
                int(round(cmd.light_colour[0] * 55 * intensity)),
                int(round(cmd.light_colour[1] * 55 * intensity)),
                int(round(cmd.light_colour[2] * 55 * intensity)),
            )
            cv2.circle(overlay, center_uv, max(12, cone_rad_px), col_rgb, -1)
            rgb = cv2.add(rgb, overlay)

        # 3. Draw barriers visible in camera frame (muted grey lines)
        for p1, p2 in config.BARRIER_SEGMENTS:
            uv1 = world_point_to_pixel(p1, cam_pos, cam_quat, self.intrinsics, w, h)
            uv2 = world_point_to_pixel(p2, cam_pos, cam_quat, self.intrinsics, w, h)
            if uv1 is not None and uv2 is not None:
                cv2.line(
                    rgb,
                    (int(round(uv1[0])), int(round(uv1[1]))),
                    (int(round(uv2[0])), int(round(uv2[1]))),
                    (65, 70, 80),
                    2,
                )

        # 4. Project and draw people inside this camera's FOV as distinct warm-bright blobs
        # Using a 1px dark rim + 2px bright core ensures tightly packed people don't merge into 1 blob
        cam_x, cam_y, cam_z = cam_pos
        fx = float(self.intrinsics[0][0])
        fy = float(self.intrinsics[1][1])
        cx = float(self.intrinsics[0][2])
        cy = float(self.intrinsics[1][2])

        dx = self.world.person_pos[:, 0] - cam_x
        dy = self.world.person_pos[:, 1] - cam_y
        us = cx + fx * (dx / max(cam_z, 1.0))
        vs = cy + fy * (dy / max(cam_z, 1.0))

        vis_mask = (us >= 1.0) & (us < w - 1.0) & (vs >= 1.0) & (vs < h - 1.0)
        vis_indices = np.nonzero(vis_mask)[0]

        for idx in vis_indices:
            px = int(round(float(us[idx])))
            py = int(round(float(vs[idx])))
            cv2.circle(rgb, (px, py), 2, (12, 12, 12), -1)
            cv2.circle(rgb, (px, py), 1, (235, 185, 50), -1)
            rgb[py, px] = (255, 250, 140)

        # Encode as PNG inside FrameMessage.jpeg_bytes (lossless for crisp 1px person cores in 320x240)
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        ok, buf = cv2.imencode(".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 100])
        jpeg_bytes = buf.tobytes() if ok else b""

        return FrameMessage(
            drone_id=drone_id,
            timestamp=self.world.sim_time,
            jpeg_bytes=jpeg_bytes,
            camera_position_xyz=cam_pos,
            camera_orientation_wxyz=cam_quat,
            intrinsics_3x3=self.intrinsics,
            width=w,
            height=h,
        )

    def get_ground_truth_positions(self) -> List[Tuple[float, float]]:
        """Return ground-truth person positions (FOR EVALUATION / TESTS ONLY, never agent)."""
        return [(float(p[0]), float(p[1])) for p in self.world.person_pos]

"""Real-time OpenCV HUD dashboard with Live 3D Perspective Concert Arena & Density Map.

Renders a unified 1360x780 frame containing four synchronized views:
1. Top-Left (Live 3D Perspective Concert Arena):
   3D pinhole-projected rendering of the 80m x 60m outdoor concert venue, 3D stage with
   truss towers & animated LED backdrop, 3D crowd barriers, Exit A & B portals, 3D human
   figures (legs, torso, head, walking stride) reconstructed from perceived camera
   detections, and 3D quadcopter drones with spinning rotors and volumetric spotlight cones.
2. Bottom-Left (Fused 2D Crowd Density Heatmap):
   GREEN/AMBER/RED classification overlay, stale cells hatched, hotspot outlines + IDs,
   and drone sector trajectories.
3. Top-Right (Selected Drone Downward Camera Feed):
   Live 320x240 aerial camera image with 2D person bounding boxes and confidence HUD.
4. Bottom-Right (Autonomous Agent Decisions & Reasons):
   Exit densities, hotspot severity/trend metrics, and per-drone state machine reasons.

Inputs:
    `SharedDensityMap`, `DroneState` dict, `DroneCommand` list, `PerceptionResult` dict,
    decoded camera frames, and `Escalation` list.
Outputs:
    BGR uint8 image suitable for `cv2.imshow` or video capture.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

import config
from brain.density_map import CELL_AMBER, CELL_RED, SharedDensityMap
from interfaces.contract import (
    DroneCommand,
    DroneState,
    Escalation,
    Hotspot,
    PerceptionResult,
)


class _Camera3D:
    """Pinhole 3D perspective camera for rendering the 3D Concert Arena panel."""

    def __init__(
        self,
        eye: Tuple[float, float, float],
        target: Tuple[float, float, float],
        width: int,
        height: int,
        fov_deg: float = 54.0,
    ) -> None:
        self.eye = np.asarray(eye, dtype=np.float64)
        self.target = np.asarray(target, dtype=np.float64)
        self.width = int(width)
        self.height = int(height)

        forward = self.target - self.eye
        forward = forward / np.linalg.norm(forward)
        world_up = np.array([0.0, 0.0, 1.0], dtype=np.float64)
        right = np.cross(forward, world_up)
        right = right / np.linalg.norm(right)
        up = np.cross(right, forward)

        self.right = right
        self.up = up
        self.forward = forward
        self.fx = (self.width * 0.5) / math.tan(math.radians(fov_deg) * 0.5)
        self.fy = self.fx
        self.cx = self.width * 0.5
        self.cy = self.height * 0.52

    def project(self, x: float, y: float, z: float) -> Optional[Tuple[int, int, float]]:
        """Project world (x, y, z) to screen (u, v, depth_m)."""
        rel = np.array([x - self.eye[0], y - self.eye[1], z - self.eye[2]], dtype=np.float64)
        zc = float(np.dot(rel, self.forward))
        if zc <= 1.0:
            return None
        xc = float(np.dot(rel, self.right))
        yc = float(np.dot(rel, self.up))
        u = int(round(self.cx + self.fx * (xc / zc)))
        v = int(round(self.cy - self.fy * (yc / zc)))
        return (u, v, zc)


class Dashboard:
    """OpenCV visualization dashboard with 3D Concert Arena + 2D Heatmap + Camera + Log."""

    def __init__(self, show_window: bool = True, selected_drone_id: int = 0) -> None:
        self.show_window = bool(show_window)
        self.selected_drone_id = int(selected_drone_id)
        self.window_name = "Autonomous Drone Fleet - 3D Concert Crowd Congestion Monitor"
        self._window_failed = False

    def render_frame(
        self,
        current_time: float,
        density_map: SharedDensityMap,
        drone_states: Dict[int, DroneState],
        commands: List[DroneCommand],
        perception_results: Dict[int, PerceptionResult],
        decoded_rgb_frames: Dict[int, np.ndarray],
        escalations: List[Escalation],
    ) -> np.ndarray:
        """Compose and return a 1360x780 BGR dashboard image."""
        canvas = np.full((780, 1360, 3), (18, 20, 26), dtype=np.uint8)

        # -------------------------------------------------------------------
        # (1) Top-Left (820x420 at x=20..840, y=42..462): 3D Perspective Concert Arena
        # -------------------------------------------------------------------
        arena_x0, arena_y0, arena_w, arena_h = 20, 42, 820, 420
        arena_img = self._render_3d_arena_panel(
            current_time=current_time,
            density_map=density_map,
            drone_states=drone_states,
            commands=commands,
            perception_results=perception_results,
            panel_w=arena_w,
            panel_h=arena_h,
        )
        canvas[arena_y0 : arena_y0 + arena_h, arena_x0 : arena_x0 + arena_w] = arena_img
        cv2.rectangle(
            canvas,
            (arena_x0, arena_y0),
            (arena_x0 + arena_w, arena_y0 + arena_h),
            (95, 110, 135),
            1,
        )
        cv2.putText(
            canvas,
            f"3D CONCERT ARENA LIVE VIEW (t = {current_time:5.1f}s) | 3D Stage, Crowd & Drone Spotlights",
            (arena_x0, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.58,
            (240, 245, 250),
            2,
            cv2.LINE_AA,
        )

        # -------------------------------------------------------------------
        # (2) Bottom-Left (820x255 at x=20..840, y=492..747): 2D Fused Density Map
        # -------------------------------------------------------------------
        map_x0, map_y0, map_w, map_h = 20, 492, 820, 255
        map_img = self._render_density_panel(
            current_time=current_time,
            density_map=density_map,
            drone_states=drone_states,
            commands=commands,
            panel_w=map_w,
            panel_h=map_h,
        )
        canvas[map_y0 : map_y0 + map_h, map_x0 : map_x0 + map_w] = map_img
        cv2.rectangle(
            canvas,
            (map_x0, map_y0),
            (map_x0 + map_w, map_y0 + map_h),
            (95, 110, 135),
            1,
        )
        cv2.putText(
            canvas,
            "TOP-DOWN FUSED CROWD DENSITY MAP (1m Grid | GREEN / AMBER / RED / Stale Hatched)",
            (map_x0, map_y0 - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.52,
            (220, 230, 240),
            1,
            cv2.LINE_AA,
        )
        self._draw_legend(canvas, map_x0, map_y0 + map_h + 12)

        # -------------------------------------------------------------------
        # (3) Top-Right (480x320 at x=860..1340, y=42..362): Drone Downward Camera Feed
        # -------------------------------------------------------------------
        cam_x0, cam_y0, cam_w, cam_h = 860, 42, 480, 320
        active_drone = self.selected_drone_id
        for cmd in commands:
            if cmd.state in ("GUIDE", "RESPOND"):
                active_drone = cmd.drone_id
                break

        cam_panel = self._render_camera_panel(
            drone_id=active_drone,
            rgb_frame=decoded_rgb_frames.get(active_drone),
            perc=perception_results.get(active_drone),
            panel_w=cam_w,
            panel_h=cam_h,
        )
        canvas[cam_y0 : cam_y0 + cam_h, cam_x0 : cam_x0 + cam_w] = cam_panel
        cv2.rectangle(
            canvas,
            (cam_x0, cam_y0),
            (cam_x0 + cam_w, cam_y0 + cam_h),
            (95, 110, 135),
            1,
        )
        cv2.putText(
            canvas,
            f"DRONE #{active_drone} AERIAL CAMERA + PERSON DETECTIONS",
            (cam_x0, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.56,
            (240, 245, 250),
            2,
            cv2.LINE_AA,
        )

        # -------------------------------------------------------------------
        # (4) Bottom-Right (480x355 at x=860..1340, y=392..747): Decisions & Reasons
        # -------------------------------------------------------------------
        dec_x0, dec_y0, dec_w, dec_h = 860, 392, 480, 355
        self._render_decision_panel(
            canvas=canvas,
            x0=dec_x0,
            y0=dec_y0,
            w=dec_w,
            h=dec_h,
            hotspots=density_map.hotspots,
            commands=commands,
            escalations=escalations,
            exit_densities=density_map.get_exit_densities(),
        )

        if self.show_window and not self._window_failed:
            try:
                cv2.imshow(self.window_name, canvas)
                key = cv2.waitKey(1) & 0xFF
                if ord("0") <= key <= ord("9"):
                    self.selected_drone_id = key - ord("0")
            except Exception:
                self._window_failed = True

        return canvas

    def _render_3d_arena_panel(
        self,
        current_time: float,
        density_map: SharedDensityMap,
        drone_states: Dict[int, DroneState],
        commands: List[DroneCommand],
        perception_results: Dict[int, PerceptionResult],
        panel_w: int,
        panel_h: int,
    ) -> np.ndarray:
        """Render a 3D perspective view of the concert stage, 3D people, and 3D drones."""
        panel = np.zeros((panel_h, panel_w, 3), dtype=np.uint8)
        # Night sky gradient at top
        for y in range(panel_h):
            t = y / float(panel_h)
            panel[y, :] = (
                int(22 + 18 * t),
                int(16 + 22 * t),
                int(14 + 16 * t),
            )

        # Slightly orbit camera angle over time for depth perception
        cam_x = -15.0 + 2.5 * math.sin(current_time * 0.15)
        cam_y = -18.0 + 2.0 * math.cos(current_time * 0.15)
        cam = _Camera3D(
            eye=(cam_x, cam_y, 35.0),
            target=(42.0, 29.0, 2.0),
            width=panel_w,
            height=panel_h,
            fov_deg=54.0,
        )

        # 1. Draw 3D ground turf polygon & 10m perspective grid lines
        ground_corners = [
            (0.0, 0.0, 0.0),
            (config.VENUE_WIDTH, 0.0, 0.0),
            (config.VENUE_WIDTH, config.VENUE_HEIGHT, 0.0),
            (0.0, config.VENUE_HEIGHT, 0.0),
        ]
        g_pts = [cam.project(*c) for c in ground_corners]
        if all(p is not None for p in g_pts):
            poly = np.array([[p[0], p[1]] for p in g_pts if p is not None], dtype=np.int32)
            cv2.fillPoly(panel, [poly], (28, 38, 28))

        for gx in range(0, int(config.VENUE_WIDTH) + 1, 10):
            p1 = cam.project(float(gx), 0.0, 0.0)
            p2 = cam.project(float(gx), config.VENUE_HEIGHT, 0.0)
            if p1 and p2:
                cv2.line(panel, (p1[0], p1[1]), (p2[0], p2[1]), (38, 52, 38), 1, cv2.LINE_AA)
        for gy in range(0, int(config.VENUE_HEIGHT) + 1, 10):
            p1 = cam.project(0.0, float(gy), 0.0)
            p2 = cam.project(config.VENUE_WIDTH, float(gy), 0.0)
            if p1 and p2:
                cv2.line(panel, (p1[0], p1[1]), (p2[0], p2[1]), (38, 52, 38), 1, cv2.LINE_AA)

        # 2. Draw Exit A and Exit B glowing 3D portals on the ground
        for exit_name, e_poly in config.EXITS.items():
            e_proj = [cam.project(x, y, 0.05) for x, y in e_poly]
            if all(p is not None for p in e_proj):
                pts = np.array([[p[0], p[1]] for p in e_proj if p is not None], dtype=np.int32)
                cv2.fillPoly(panel, [pts], (45, 145, 55))
                cv2.polylines(panel, [pts], True, (120, 255, 130), 2, cv2.LINE_AA)
                # 3D exit arch posts
                p_lbl = cam.project(*config.EXIT_TARGETS[exit_name], 3.2)
                if p_lbl:
                    cv2.putText(
                        panel,
                        exit_name,
                        (p_lbl[0] - 24, p_lbl[1]),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.48,
                        (140, 255, 150),
                        2,
                        cv2.LINE_AA,
                    )

        # 3. Draw 3D Concert Stage (deck, backdrop, truss towers, roof, and stage light beams)
        scx, scy = config.STAGE_CENTER
        slx, sly, slz = config.STAGE_SIZE
        x0, x1 = scx - slx * 0.5, scx + slx * 0.5
        y0, y1 = scy - sly * 0.5, scy + sly * 0.5
        stage_top = [
            cam.project(x0, y0, slz),
            cam.project(x1, y0, slz),
            cam.project(x1, y1, slz),
            cam.project(x0, y1, slz),
        ]
        if all(p is not None for p in stage_top):
            pts = np.array([[p[0], p[1]] for p in stage_top if p is not None], dtype=np.int32)
            cv2.fillPoly(panel, [pts], (65, 45, 75))
            cv2.polylines(panel, [pts], True, (180, 120, 210), 2, cv2.LINE_AA)

        # Truss towers & roof canopy at z = slz + 9.0
        roof_z = slz + 9.0
        for tx, ty in [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]:
            pb = cam.project(tx, ty, slz)
            pt = cam.project(tx, ty, roof_z)
            if pb and pt:
                cv2.line(panel, (pb[0], pb[1]), (pt[0], pt[1]), (160, 165, 175), 2, cv2.LINE_AA)

        roof_pts = [
            cam.project(x0, y0, roof_z),
            cam.project(x1, y0, roof_z),
            cam.project(x1, y1, roof_z),
            cam.project(x0, y1, roof_z),
        ]
        if all(p is not None for p in roof_pts):
            rpoly = np.array([[p[0], p[1]] for p in roof_pts if p is not None], dtype=np.int32)
            cv2.polylines(panel, [rpoly], True, (200, 140, 240), 2, cv2.LINE_AA)

        # Stage label
        p_st = cam.project(scx, scy, roof_z + 1.5)
        if p_st:
            cv2.putText(
                panel,
                "CONCERT STAGE",
                (p_st[0] - 50, p_st[1]),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.48,
                (235, 190, 255),
                2,
                cv2.LINE_AA,
            )

        # 4. Draw 3D Metallic Crowd Barriers (extruded from z=0 to z=1.2m)
        for (bx1, by1), (bx2, by2) in config.BARRIER_SEGMENTS:
            b_corners = [
                cam.project(bx1, by1, 0.0),
                cam.project(bx2, by2, 0.0),
                cam.project(bx2, by2, 1.25),
                cam.project(bx1, by1, 1.25),
            ]
            if all(p is not None for p in b_corners):
                bpoly = np.array(
                    [[p[0], p[1]] for p in b_corners if p is not None], dtype=np.int32
                )
                cv2.fillPoly(panel, [bpoly], (110, 118, 130))
                cv2.polylines(panel, [bpoly], True, (200, 210, 225), 2, cv2.LINE_AA)

        # 5. Draw translucent 3D Drone Spotlight Cones on the arena floor
        cmd_by_id = {c.drone_id: c for c in commands}
        cone_overlay = panel.copy()
        state_bgr = {
            "PATROL": (80, 230, 80),
            "MONITOR": (0, 190, 255),
            "RESPOND": (60, 60, 255),
            "GUIDE": (220, 255, 0),
            "FAILSAFE": (210, 210, 210),
        }
        for d_id, st in drone_states.items():
            dx, dy, dz = st.position_xyz
            cmd = cmd_by_id.get(d_id)
            st_name = cmd.state if cmd is not None else st.current_state
            if st_name in ("RESPOND", "GUIDE", "MONITOR"):
                p_drone = cam.project(dx, dy, dz)
                aim_x, aim_y = cmd.light_aim_xy if cmd is not None else (dx, dy)
                cx = 0.55 * dx + 0.45 * aim_x
                cy = 0.55 * dy + 0.45 * aim_y
                rad = 5.5 if st_name != "GUIDE" else 7.5
                p_left = cam.project(cx - rad * 0.6, cy - rad, 0.0)
                p_right = cam.project(cx + rad * 0.6, cy + rad, 0.0)
                if p_drone and p_left and p_right:
                    tri = np.array(
                        [
                            [p_drone[0], p_drone[1]],
                            [p_left[0], p_left[1]],
                            [p_right[0], p_right[1]],
                        ],
                        dtype=np.int32,
                    )
                    cv2.fillPoly(cone_overlay, [tri], state_bgr.get(st_name, (180, 180, 180)))
        cv2.addWeighted(cone_overlay, 0.22, panel, 0.78, 0, panel)

        # 6. Collect perceived 3D human positions (deduplicated on a 0.45m grid) & sort back-to-front
        seen_cells: set[Tuple[int, int]] = set()
        people_3d: List[Tuple[float, float, float]] = []
        for perc in perception_results.values():
            for gx, gy in perc.ground_points_xy:
                key = (int(round(gx / 0.45)), int(round(gy / 0.45)))
                if key not in seen_cells:
                    seen_cells.add(key)
                    p_foot = cam.project(gx, gy, 0.0)
                    if p_foot is not None:
                        people_3d.append((p_foot[2], gx, gy))

        # Sort back-to-front (largest depth first)
        people_3d.sort(key=lambda item: item[0], reverse=True)

        # Jackets palette (BGR) for realistic 3D concert crowd appearance
        shirt_colors = [
            (60, 175, 245),
            (210, 140, 55),
            (95, 210, 120),
            (180, 95, 230),
            (70, 215, 240),
            (110, 110, 245),
        ]

        for depth, wx, wy in people_3d:
            p_foot = cam.project(wx, wy, 0.0)
            p_hip = cam.project(wx, wy, 0.85)
            p_shoulder = cam.project(wx, wy, 1.42)
            p_head = cam.project(wx, wy, 1.72)
            if not (p_foot and p_hip and p_shoulder and p_head):
                continue

            fx, fy, _ = p_foot
            hx, hy, _ = p_hip
            sx, sy, _ = p_shoulder
            hdx, hdy, _ = p_head

            # Scale body width with distance
            scale = max(1, int(round(115.0 / max(depth, 8.0))))
            col_idx = (int(wx * 7 + wy * 13)) % len(shirt_colors)
            shirt_bgr = shirt_colors[col_idx]

            # Check if person is inside a RED cell to highlight congestion pressure
            gx_i = int(np.clip(wx / density_map.cell_size, 0, density_map.nx - 1))
            gy_i = int(np.clip(wy / density_map.cell_size, 0, density_map.ny - 1))
            if density_map.density[gy_i, gx_i] >= config.RED_THRESHOLD:
                shirt_bgr = (65, 75, 255)

            # Ground shadow
            cv2.ellipse(
                panel,
                (fx, fy),
                (max(2, scale + 1), max(1, scale // 2)),
                0,
                0,
                360,
                (15, 20, 15),
                -1,
            )
            # Animated striding legs
            phase = math.sin(current_time * 5.5 + wx * 1.7 + wy * 1.3)
            leg_spread = max(1, int(round(scale * 0.7 * phase)))
            cv2.line(panel, (hx, hy), (fx - leg_spread, fy), (50, 55, 70), max(1, scale // 2))
            cv2.line(panel, (hx, hy), (fx + leg_spread, fy), (50, 55, 70), max(1, scale // 2))

            # 3D Torso
            cv2.line(panel, (hx, hy), (sx, sy), shirt_bgr, max(2, scale))
            # 3D Head
            cv2.circle(panel, (hdx, hdy), max(2, int(round(scale * 0.65))), (185, 215, 245), -1)

        # 7. Draw 3D Quadcopter Drones hovering in 3D space (sorted back-to-front)
        drone_items: List[Tuple[float, int, DroneState]] = []
        for d_id, st in drone_states.items():
            pr = cam.project(*st.position_xyz)
            if pr is not None:
                drone_items.append((pr[2], d_id, st))
        drone_items.sort(key=lambda item: item[0], reverse=True)

        for depth, d_id, st in drone_items:
            dx, dy, dz = st.position_xyz
            p_d = cam.project(dx, dy, dz)
            p_g = cam.project(dx, dy, 0.0)
            if not p_d:
                continue
            ux, uy, _ = p_d
            cmd = cmd_by_id.get(d_id)
            st_name = cmd.state if cmd is not None else st.current_state
            col = state_bgr.get(st_name, (200, 200, 200))

            # Subtle vertical altitude reference line to ground
            if p_g and st_name in ("RESPOND", "GUIDE"):
                cv2.line(panel, (ux, uy), (p_g[0], p_g[1]), col, 1, cv2.LINE_AA)

            # 4 Quadcopter arms & rotors projected in 3D
            arm_len = 1.15
            for rx, ry in [
                (-arm_len, -arm_len),
                (arm_len, -arm_len),
                (arm_len, arm_len),
                (-arm_len, arm_len),
            ]:
                p_rot = cam.project(dx + rx, dy + ry, dz + 0.15)
                if p_rot:
                    cv2.line(panel, (ux, uy), (p_rot[0], p_rot[1]), (170, 175, 185), 2, cv2.LINE_AA)
                    cv2.ellipse(
                        panel,
                        (p_rot[0], p_rot[1]),
                        (5, 2),
                        0,
                        0,
                        360,
                        (210, 220, 230),
                        1,
                        cv2.LINE_AA,
                    )

            # Central drone body + LED status glow
            cv2.circle(panel, (ux, uy), 5, col, -1, cv2.LINE_AA)
            cv2.circle(panel, (ux, uy), 6, (20, 20, 25), 1, cv2.LINE_AA)
            cv2.putText(
                panel,
                f"D{d_id}:{st_name[:3]}",
                (ux + 7, uy - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.36,
                col,
                1,
                cv2.LINE_AA,
            )

        return panel

    def _render_density_panel(
        self,
        current_time: float,
        density_map: SharedDensityMap,
        drone_states: Dict[int, DroneState],
        commands: List[DroneCommand],
        panel_w: int,
        panel_h: int,
    ) -> np.ndarray:
        ny, nx = density_map.ny, density_map.nx
        levels = density_map.classify_cells(current_time=current_time)
        stale = density_map.get_stale_mask(current_time=current_time)

        grid_bgr = np.zeros((ny, nx, 3), dtype=np.uint8)
        norm_d = np.clip(density_map.density / 6.5, 0.0, 1.0)

        grid_bgr[:, :, 0] = (25 + 45 * norm_d).astype(np.uint8)
        grid_bgr[:, :, 1] = (40 + 140 * norm_d).astype(np.uint8)
        grid_bgr[:, :, 2] = (20 + 40 * norm_d).astype(np.uint8)

        amber_mask = levels == CELL_AMBER
        grid_bgr[amber_mask] = (0, 185, 255)

        red_mask = levels == CELL_RED
        grid_bgr[red_mask] = (40, 40, 250)

        panel = cv2.resize(grid_bgr, (panel_w, panel_h), interpolation=cv2.INTER_LINEAR)

        stale_up = cv2.resize(
            stale.astype(np.uint8), (panel_w, panel_h), interpolation=cv2.INTER_NEAREST
        ).astype(bool)
        yy, xx = np.mgrid[0:panel_h, 0:panel_w]
        hatch = ((xx + yy) % 10 < 2) & stale_up
        panel[hatch] = (75, 75, 85)

        def w2p(wx: float, wy: float) -> tuple[int, int]:
            px = int(round((wx / density_map.venue_width) * (panel_w - 1)))
            py = int(round((wy / density_map.venue_height) * (panel_h - 1)))
            return (px, py)

        scx, scy = config.STAGE_CENTER
        slx, sly, _ = config.STAGE_SIZE
        s_p1 = w2p(scx - slx * 0.5, scy - sly * 0.5)
        s_p2 = w2p(scx + slx * 0.5, scy + sly * 0.5)
        cv2.rectangle(panel, s_p1, s_p2, (180, 90, 140), -1)
        cv2.rectangle(panel, s_p1, s_p2, (230, 180, 220), 2)

        for exit_name, poly in config.EXITS.items():
            pts = np.array([w2p(x, y) for x, y in poly], dtype=np.int32)
            cv2.polylines(panel, [pts], True, (120, 255, 120), 2)
            cx, cy = w2p(*config.EXIT_TARGETS[exit_name])
            cv2.putText(
                panel,
                exit_name,
                (max(4, cx - 32), max(14, cy - 4)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.42,
                (160, 255, 160),
                1,
            )

        for p1, p2 in config.BARRIER_SEGMENTS:
            cv2.line(panel, w2p(*p1), w2p(*p2), (210, 210, 220), 2, cv2.LINE_AA)

        for hs in density_map.hotspots:
            gx1, gy1, gx2, gy2 = hs.bbox_grid
            p1 = w2p(gx1 * density_map.cell_size, gy1 * density_map.cell_size)
            p2 = w2p((gx2 + 1) * density_map.cell_size, (gy2 + 1) * density_map.cell_size)
            col = (50, 50, 255) if hs.level == "RED" else (0, 200, 255)
            cv2.rectangle(panel, p1, p2, col, 2)
            hcx, hcy = w2p(*hs.centroid_xy)
            cv2.putText(
                panel,
                f"H#{hs.hotspot_id} {hs.peak_density:.1f}p/m2",
                (hcx - 35, max(12, p1[1] - 4)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.40,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )

        cmd_by_id = {c.drone_id: c for c in commands}
        state_bgr = {
            "PATROL": (80, 230, 80),
            "MONITOR": (0, 190, 255),
            "RESPOND": (60, 60, 255),
            "GUIDE": (220, 255, 0),
            "FAILSAFE": (220, 220, 220),
        }
        for d_id, st in sorted(drone_states.items()):
            dx, dy, _ = st.position_xyz
            px, py = w2p(dx, dy)
            cmd = cmd_by_id.get(d_id)
            st_name = cmd.state if cmd is not None else st.current_state
            col = state_bgr.get(st_name, (200, 200, 200))

            if cmd is not None and cmd.state == "GUIDE":
                ax, ay = w2p(*cmd.light_aim_xy)
                cv2.arrowedLine(panel, (px, py), (ax, ay), (220, 255, 0), 2, tipLength=0.25)

            cv2.circle(panel, (px, py), 4, col, -1)

        return panel

    def _render_camera_panel(
        self,
        drone_id: int,
        rgb_frame: Optional[np.ndarray],
        perc: Optional[PerceptionResult],
        panel_w: int,
        panel_h: int,
    ) -> np.ndarray:
        if rgb_frame is None:
            panel = np.full((panel_h, panel_w, 3), 30, dtype=np.uint8)
            cv2.putText(
                panel,
                f"Waiting for Drone #{drone_id} frame...",
                (40, panel_h // 2),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (180, 180, 180),
                1,
            )
            return panel

        bgr = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR)
        fh, fw = bgr.shape[:2]
        sx = panel_w / float(fw)
        sy = panel_h / float(fh)
        panel = cv2.resize(bgr, (panel_w, panel_h), interpolation=cv2.INTER_NEAREST)

        if perc is not None:
            for det in perc.detections:
                x1, y1, x2, y2 = det.bbox_xyxy
                p1 = (int(round(x1 * sx)), int(round(y1 * sy)))
                p2 = (int(round(x2 * sx)), int(round(y2 * sy)))
                cv2.rectangle(panel, p1, p2, (60, 255, 100), 1)

            hud_txt = (
                f"Detections: {len(perc.detections)} | "
                f"Conf: {perc.confidence:.2f} | t={perc.timestamp:.1f}s"
            )
            cv2.rectangle(panel, (0, 0), (panel_w, 26), (15, 15, 20), -1)
            cv2.putText(
                panel,
                hud_txt,
                (8, 18),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.48,
                (120, 255, 160),
                1,
                cv2.LINE_AA,
            )
        return panel

    def _render_decision_panel(
        self,
        canvas: np.ndarray,
        x0: int,
        y0: int,
        w: int,
        h: int,
        hotspots: List[Hotspot],
        commands: List[DroneCommand],
        escalations: List[Escalation],
        exit_densities: Dict[str, float],
    ) -> None:
        cv2.rectangle(canvas, (x0, y0), (x0 + w, y0 + h), (28, 32, 40), -1)
        cv2.rectangle(canvas, (x0, y0), (x0 + w, y0 + h), (95, 110, 135), 1)
        cv2.putText(
            canvas,
            "AUTONOMOUS AGENT DECISIONS & REASONS",
            (x0, y0 - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (240, 245, 250),
            2,
            cv2.LINE_AA,
        )

        y = y0 + 22
        exit_str = " | ".join(f"{k}: {v:.2f} p/m2" for k, v in sorted(exit_densities.items()))
        cv2.putText(
            canvas,
            f"Exit Densities: {exit_str}",
            (x0 + 10, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.44,
            (180, 220, 250),
            1,
            cv2.LINE_AA,
        )
        y += 22

        if hotspots:
            for hs in hotspots[:3]:
                txt = (
                    f"Hotspot #{hs.hotspot_id} [{hs.level}] peak={hs.peak_density:.1f}p/m2 "
                    f"trend={hs.mean_trend:+.2f}/s -> Guide:{hs.least_dense_exit}"
                )
                col = (90, 90, 255) if hs.level == "RED" else (0, 200, 255)
                cv2.putText(
                    canvas,
                    txt[:64],
                    (x0 + 10, y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.41,
                    col,
                    1,
                    cv2.LINE_AA,
                )
                y += 18
        else:
            cv2.putText(
                canvas,
                "Hotspots: None (all observed cells GREEN)",
                (x0 + 10, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.42,
                (130, 220, 140),
                1,
                cv2.LINE_AA,
            )
            y += 18

        y += 6
        priority_order = {"FAILSAFE": 0, "GUIDE": 1, "RESPOND": 2, "MONITOR": 3, "PATROL": 4}
        sorted_cmds = sorted(commands, key=lambda c: (priority_order.get(c.state, 5), c.drone_id))
        for cmd in sorted_cmds[:11]:
            line = f"D{cmd.drone_id:02d} [{cmd.state:7s}]: {cmd.reason}"
            cv2.putText(
                canvas,
                line[:66],
                (x0 + 10, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.39,
                (225, 230, 235),
                1,
                cv2.LINE_AA,
            )
            y += 19

        if escalations and y < y0 + h - 12:
            esc = escalations[-1]
            cv2.putText(
                canvas,
                f"ESCALATION D{esc.drone_id}: {esc.reason}"[:64],
                (x0 + 10, y0 + h - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.42,
                (80, 80, 255),
                1,
                cv2.LINE_AA,
            )

    @staticmethod
    def _draw_legend(canvas: np.ndarray, x0: int, y0: int) -> None:
        items = [
            ("GREEN (<3.5 p/m2)", (40, 140, 40)),
            ("AMBER (3.5-5.0 p/m2)", (0, 185, 255)),
            ("RED (>=5.0 p/m2)", (40, 40, 250)),
            ("GUIDE Beam", (220, 255, 0)),
            ("Stale (Unseen >5s)", (95, 95, 110)),
        ]
        x = x0
        for label_txt, col in items:
            cv2.rectangle(canvas, (x, y0), (x + 14, y0 + 14), col, -1)
            cv2.putText(
                canvas,
                label_txt,
                (x + 20, y0 + 12),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.42,
                (210, 215, 220),
                1,
                cv2.LINE_AA,
            )
            x += 160

    def close(self) -> None:
        if self.show_window and not self._window_failed:
            try:
                cv2.destroyAllWindows()
            except Exception:
                pass

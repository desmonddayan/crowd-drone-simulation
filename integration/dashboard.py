"""Real-time OpenCV HUD dashboard for screen-recording and live telemetry inspection.

Renders a single unified 1280x720 frame containing:
(a) Left panel: Fused crowd density heatmap with GREEN/AMBER/RED classification overlay,
    stale cells hatched, stage/barriers/exits, hotspot bounding outlines + IDs, and
    drone positions, light cones, and state labels.
(b) Top-right panel: Selected drone's downward camera feed with live 2D person
    detection bounding boxes and confidence telemetry.
(c) Bottom-right panel: Live decision log showing per-drone state transitions,
    mathematical hotspot metrics, and human-readable reasons.

Inputs:
    `SharedDensityMap`, `DroneState` dict, `DroneCommand` list, `PerceptionResult` dict,
    decoded camera frames, and `Escalation` list.
Outputs:
    BGR uint8 image suitable for `cv2.imshow` or video capture (`cv2.VideoWriter`).
"""

from __future__ import annotations

from typing import Dict, List, Optional

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


class Dashboard:
    """OpenCV visualization dashboard for the autonomous drone fleet."""

    def __init__(self, show_window: bool = True, selected_drone_id: int = 0) -> None:
        self.show_window = bool(show_window)
        self.selected_drone_id = int(selected_drone_id)
        self.window_name = "Autonomous Drone Fleet - Crowd Congestion Monitor"
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
        """Compose and return a 1280x720 BGR dashboard image."""
        canvas = np.full((720, 1280, 3), (22, 24, 30), dtype=np.uint8)

        # -------------------------------------------------------------------
        # (a) Left panel (760x660 at x=20..780, y=40..700): Fused Density Map
        # -------------------------------------------------------------------
        map_w, map_h = 760, 570  # Preserves 80:60 (4:3) aspect ratio
        map_x0, map_y0 = 20, 55
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
            (90, 100, 120),
            1,
        )

        # Title bar & legend
        cv2.putText(
            canvas,
            f"FUSED CROWD DENSITY MAP (t = {current_time:5.1f}s) | Camera-Only Perception",
            (map_x0, 36),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.62,
            (235, 240, 245),
            2,
            cv2.LINE_AA,
        )
        self._draw_legend(canvas, map_x0, map_y0 + map_h + 18)

        # -------------------------------------------------------------------
        # (b) Top-right panel (460x310 at x=800..1260, y=55..365): Drone Camera Feed
        # -------------------------------------------------------------------
        cam_x0, cam_y0, cam_w, cam_h = 800, 55, 460, 310
        # Automatically focus on a drone in GUIDE or RESPOND state if available
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
            (90, 100, 120),
            1,
        )
        cv2.putText(
            canvas,
            f"DRONE #{active_drone} DOWNWARD CAMERA + DETECTIONS",
            (cam_x0, 36),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.58,
            (235, 240, 245),
            2,
            cv2.LINE_AA,
        )

        # -------------------------------------------------------------------
        # (c) Bottom-right panel (460x305 at x=800..1260, y=395..700): Decisions & Reasons
        # -------------------------------------------------------------------
        dec_x0, dec_y0, dec_w, dec_h = 800, 395, 460, 305
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

        # Base BGR image at grid resolution (ny, nx)
        grid_bgr = np.zeros((ny, nx, 3), dtype=np.uint8)
        norm_d = np.clip(density_map.density / 6.5, 0.0, 1.0)

        # GREEN cells: soft dark-emerald proportional to density
        grid_bgr[:, :, 0] = (25 + 45 * norm_d).astype(np.uint8)
        grid_bgr[:, :, 1] = (40 + 140 * norm_d).astype(np.uint8)
        grid_bgr[:, :, 2] = (20 + 40 * norm_d).astype(np.uint8)

        # AMBER cells: bright amber-orange
        amber_mask = levels == CELL_AMBER
        grid_bgr[amber_mask] = (0, 185, 255)

        # RED cells: intense crimson-red
        red_mask = levels == CELL_RED
        grid_bgr[red_mask] = (40, 40, 250)

        # Upscale to panel size
        panel = cv2.resize(grid_bgr, (panel_w, panel_h), interpolation=cv2.INTER_LINEAR)

        # Hatch stale cells with diagonal grey lines
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

        # Draw stage
        scx, scy = config.STAGE_CENTER
        slx, sly, _ = config.STAGE_SIZE
        s_p1 = w2p(scx - slx * 0.5, scy - sly * 0.5)
        s_p2 = w2p(scx + slx * 0.5, scy + sly * 0.5)
        cv2.rectangle(panel, s_p1, s_p2, (180, 90, 140), -1)
        cv2.rectangle(panel, s_p1, s_p2, (230, 180, 220), 2)
        cv2.putText(
            panel,
            "STAGE",
            (s_p1[0] + 6, (s_p1[1] + s_p2[1]) // 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (255, 255, 255),
            1,
        )

        # Draw exits
        for exit_name, poly in config.EXITS.items():
            pts = np.array([w2p(x, y) for x, y in poly], dtype=np.int32)
            cv2.polylines(panel, [pts], True, (120, 255, 120), 2)
            cx, cy = w2p(*config.EXIT_TARGETS[exit_name])
            cv2.putText(
                panel,
                exit_name,
                (max(4, cx - 32), max(16, cy - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.50,
                (160, 255, 160),
                2,
            )

        # Draw barriers
        for p1, p2 in config.BARRIER_SEGMENTS:
            cv2.line(panel, w2p(*p1), w2p(*p2), (210, 210, 220), 3, cv2.LINE_AA)

        # Draw hotspot outlines and IDs
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
                (hcx - 35, max(15, p1[1] - 6)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )

        # Draw drones and light cones
        cmd_by_id = {c.drone_id: c for c in commands}
        state_bgr = {
            "PATROL": (80, 230, 80),
            "MONITOR": (0, 190, 255),
            "RESPOND": (60, 60, 255),
            "GUIDE": (220, 255, 0),
            "FAILSAFE": (220, 220, 220),
        }
        for d_id, st in sorted(drone_states.items()):
            dx, dy, dz = st.position_xyz
            px, py = w2p(dx, dy)
            cmd = cmd_by_id.get(d_id)
            st_name = cmd.state if cmd is not None else st.current_state
            col = state_bgr.get(st_name, (200, 200, 200))

            # Draw guidance vector arrow when in GUIDE state
            if cmd is not None and cmd.state == "GUIDE":
                ax, ay = w2p(*cmd.light_aim_xy)
                cv2.arrowedLine(panel, (px, py), (ax, ay), (220, 255, 0), 2, tipLength=0.25)

            cv2.circle(panel, (px, py), 6, col, -1)
            cv2.circle(panel, (px, py), 7, (15, 15, 15), 1)
            cv2.putText(
                panel,
                f"D{d_id}:{st_name[:3]}",
                (px + 8, py + 4),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.38,
                col,
                1,
                cv2.LINE_AA,
            )

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
        cv2.rectangle(canvas, (x0, y0), (x0 + w, y0 + h), (30, 34, 42), -1)
        cv2.rectangle(canvas, (x0, y0), (x0 + w, y0 + h), (90, 100, 120), 1)
        cv2.putText(
            canvas,
            "AUTONOMOUS AGENT DECISIONS & REASONS",
            (x0, y0 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.58,
            (235, 240, 245),
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
            for hs in hotspots[:2]:
                txt = (
                    f"Hotspot #{hs.hotspot_id} [{hs.level}] peak={hs.peak_density:.1f}p/m2 "
                    f"trend={hs.mean_trend:+.2f}/s -> Guide:{hs.least_dense_exit}"
                )
                col = (90, 90, 255) if hs.level == "RED" else (0, 200, 255)
                cv2.putText(
                    canvas,
                    txt[:62],
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
        # Prioritise showing drones in GUIDE / RESPOND / MONITOR / FAILSAFE first
        priority_order = {"FAILSAFE": 0, "GUIDE": 1, "RESPOND": 2, "MONITOR": 3, "PATROL": 4}
        sorted_cmds = sorted(commands, key=lambda c: (priority_order.get(c.state, 5), c.drone_id))
        for cmd in sorted_cmds[:9]:
            line = f"D{cmd.drone_id:02d} [{cmd.state:7s}]: {cmd.reason}"
            cv2.putText(
                canvas,
                line[:64],
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
                f"ESCALATION D{esc.drone_id}: {esc.reason}"[:62],
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
            x += 152

    def close(self) -> None:
        if self.show_window and not self._window_failed:
            try:
                cv2.destroyAllWindows()
            except Exception:
                pass

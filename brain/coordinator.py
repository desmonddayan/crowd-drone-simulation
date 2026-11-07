"""Fleet coordinator and safety layer for multi-drone crowd congestion management.

Runs every coordinator tick (default 5 Hz):
1. Builds a drones x hotspot_slots cost matrix:
       cost = distance / max_speed - w * severity
   and solves optimal assignment via `scipy.optimize.linear_sum_assignment`.
   Assigns at most K drones per hotspot based on hotspot area.
2. Unassigned drones:
   - If any grid cells are stale, dispatches the nearest free drone to the stalest region
     (coverage memory).
   - If a neighbouring drone has left its sector to respond to a hotspot, raises patrol
     altitude to `WIDE_COVERAGE_ALTITUDE` to widen camera footprint coverage.
   - Otherwise returns to home sector center at `PATROL_ALTITUDE`.
3. Safety layer overrides all outputs:
   - Enforces minimum horizontal/vertical separation between all drone targets.
   - Enforces venue geofence boundaries.
   - Ensures no drone hovers below `INSPECTION_ALTITUDE` over a RED cell.
   - Preserves FAILSAFE commands and escalations.

Inputs:
    SharedDensityMap, latest DroneState dict, perception timestamps/confidences,
    and optional strategic overrides.
Outputs:
    List of safety-verified DroneCommand objects and list of Escalation alerts.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.optimize import linear_sum_assignment

import config
from brain.density_map import CELL_RED, SharedDensityMap
from brain.state_machine import DroneAgent
from interfaces.contract import DroneCommand, DroneState, Escalation, Hotspot


class FleetCoordinator:
    """Centralized assignment & safety coordinator for N autonomous drone agents."""

    def __init__(
        self,
        n_drones: int = config.N_DRONES,
        venue_width: float = config.VENUE_WIDTH,
        venue_height: float = config.VENUE_HEIGHT,
        max_drones_per_hotspot: int = config.MAX_DRONES_PER_HOTSPOT,
        severity_weight: float = config.SEVERITY_WEIGHT,
        min_sep_horizontal: float = config.MIN_SEP_HORIZONTAL,
        min_sep_vertical: float = config.MIN_SEP_VERTICAL,
    ) -> None:
        self.n_drones = int(n_drones)
        self.venue_width = float(venue_width)
        self.venue_height = float(venue_height)
        self.max_drones_per_hotspot = int(max_drones_per_hotspot)
        self.severity_weight = float(severity_weight)
        self.min_sep_horizontal = float(min_sep_horizontal)
        self.min_sep_vertical = float(min_sep_vertical)

        self.sector_centers = config.compute_sector_centers(
            n_drones=self.n_drones,
            venue_width=self.venue_width,
            venue_height=self.venue_height,
            altitude=config.PATROL_ALTITUDE,
        )
        self.agents: Dict[int, DroneAgent] = {
            i: DroneAgent(drone_id=i, home_xyz=self.sector_centers[i])
            for i in range(self.n_drones)
        }

        # Optional strategic overrides set by brain/strategist.py
        self.priority_multipliers: Dict[int, float] = {}
        self.forced_guidance_exits: Dict[int, str] = {}

    def record_perception(self, drone_id: int, timestamp: float, confidence: float) -> None:
        """Update perception health metrics for a specific drone."""
        if drone_id in self.agents:
            self.agents[drone_id].record_perception(timestamp, confidence)

    def _drones_needed_for_hotspot(self, hotspot: Hotspot) -> int:
        if hotspot.level == "AMBER":
            return 1
        k = int(math.ceil(hotspot.area_m2 / config.HOTSPOT_AREA_PER_DRONE))
        return max(1, min(self.max_drones_per_hotspot, k))

    def assign_hotspots(
        self,
        drone_positions: Dict[int, Tuple[float, float, float]],
        hotspots: List[Hotspot],
    ) -> Dict[int, Tuple[Hotspot, int]]:
        """Solve drone-to-hotspot assignment via Hungarian algorithm (`linear_sum_assignment`).

        Returns:
            Mapping `drone_id -> (assigned_hotspot, slot_index_within_hotspot)`.
        """
        if not hotspots or not drone_positions:
            return {}

        drone_ids = sorted(drone_positions.keys())
        # Expand hotspots into slots (at most K slots per hotspot based on area)
        slots: List[Tuple[Hotspot, int]] = []
        for hs in hotspots:
            k = self._drones_needed_for_hotspot(hs)
            for slot_idx in range(k):
                slots.append((hs, slot_idx))

        cost_matrix = np.zeros((len(drone_ids), len(slots)), dtype=np.float64)
        for r, d_id in enumerate(drone_ids):
            dx, dy, _ = drone_positions[d_id]
            for c, (hs, slot_idx) in enumerate(slots):
                hx, hy = hs.centroid_xy
                dist = math.hypot(dx - hx, dy - hy)
                mult = self.priority_multipliers.get(hs.hotspot_id, 1.0)
                # Slightly discount secondary/tertiary slots so at least one drone covers each hotspot first
                slot_discount = 1.0 / (1.0 + 0.35 * slot_idx)
                eff_severity = hs.severity * mult * slot_discount
                cost_matrix[r, c] = (
                    dist / config.MAX_DRONE_SPEED - self.severity_weight * eff_severity
                )

        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        assignments: Dict[int, Tuple[Hotspot, int]] = {}
        for r, c in zip(row_ind, col_ind):
            d_id = drone_ids[int(r)]
            hs, slot_idx = slots[int(c)]
            assignments[d_id] = (hs, slot_idx)
        return assignments

    def _apply_safety_layer(
        self,
        commands: List[DroneCommand],
        density_map: Optional[SharedDensityMap],
        current_time: float,
    ) -> List[DroneCommand]:
        """Enforce geofence, minimum altitude over RED cells, and minimum inter-drone separation."""
        if not commands:
            return commands

        red_cells = (
            (density_map.classify_cells(current_time) == CELL_RED)
            if density_map is not None
            else None
        )

        # Extract mutable target coordinates
        targets = np.array([list(cmd.target_xyz) for cmd in commands], dtype=np.float64)
        n = len(commands)

        # 1. Geofence and minimum altitude over RED cells
        margin = 2.0
        for i, cmd in enumerate(commands):
            targets[i, 0] = float(np.clip(targets[i, 0], margin, self.venue_width - margin))
            targets[i, 1] = float(np.clip(targets[i, 1], margin, self.venue_height - margin))
            min_z = (
                config.SAFE_ALTITUDE
                if cmd.state == "FAILSAFE"
                else config.INSPECTION_ALTITUDE
            )
            if red_cells is not None and density_map is not None:
                gx = int(np.clip(targets[i, 0] / density_map.cell_size, 0, density_map.nx - 1))
                gy = int(np.clip(targets[i, 1] / density_map.cell_size, 0, density_map.ny - 1))
                if red_cells[gy, gx]:
                    min_z = max(min_z, config.INSPECTION_ALTITUDE)
            targets[i, 2] = float(np.clip(targets[i, 2], min_z, config.SAFE_ALTITUDE))

        # 2. Enforce minimum horizontal separation iteratively
        for _ in range(12):
            moved = False
            for i in range(n):
                for j in range(i + 1, n):
                    dz = abs(targets[i, 2] - targets[j, 2])
                    dx = targets[j, 0] - targets[i, 0]
                    dy = targets[j, 1] - targets[i, 1]
                    dist_xy = math.hypot(dx, dy)
                    if dist_xy < self.min_sep_horizontal and dz < self.min_sep_vertical:
                        moved = True
                        if dist_xy < 1e-3:
                            angle = (i + 1) * 1.1 + j * 0.7
                            ux, uy = math.cos(angle), math.sin(angle)
                        else:
                            ux, uy = dx / dist_xy, dy / dist_xy
                        push = 0.55 * (self.min_sep_horizontal - dist_xy) + 0.05
                        targets[i, 0] = np.clip(
                            targets[i, 0] - ux * push, margin, self.venue_width - margin
                        )
                        targets[i, 1] = np.clip(
                            targets[i, 1] - uy * push, margin, self.venue_height - margin
                        )
                        targets[j, 0] = np.clip(
                            targets[j, 0] + ux * push, margin, self.venue_width - margin
                        )
                        targets[j, 1] = np.clip(
                            targets[j, 1] + uy * push, margin, self.venue_height - margin
                        )
            if not moved:
                break

        # Write sanitized targets and capped pulse frequencies back to commands
        safe_commands: List[DroneCommand] = []
        for i, cmd in enumerate(commands):
            safe_commands.append(
                DroneCommand(
                    drone_id=cmd.drone_id,
                    state=cmd.state,
                    target_xyz=(
                        round(float(targets[i, 0]), 3),
                        round(float(targets[i, 1]), 3),
                        round(float(targets[i, 2]), 3),
                    ),
                    light_colour=cmd.light_colour,
                    light_intensity=float(np.clip(cmd.light_intensity, 0.0, 1.0)),
                    light_pulse_hz=float(np.clip(cmd.light_pulse_hz, 0.0, config.MAX_PULSE_HZ)),
                    light_aim_xy=cmd.light_aim_xy,
                    guide_to_exit=cmd.guide_to_exit,
                    reason=cmd.reason,
                    timestamp=cmd.timestamp,
                )
            )
        return safe_commands

    def step(
        self,
        current_time: float,
        density_map: SharedDensityMap,
        drone_states: Optional[Dict[int, DroneState]] = None,
    ) -> Tuple[List[DroneCommand], List[Escalation]]:
        """Run one fleet coordination tick and return safety-checked commands + escalations."""
        drone_states = drone_states or {}
        hotspots = density_map.extract_hotspots(current_time=current_time)

        # Current positions of all drones (fall back to home sector if state not yet received)
        positions: Dict[int, Tuple[float, float, float]] = {}
        for d_id, agent in self.agents.items():
            if d_id in drone_states:
                positions[d_id] = drone_states[d_id].position_xyz
            else:
                positions[d_id] = agent.home_xyz

        # Filter out drones in FAILSAFE before assigning hotspots
        healthy_positions: Dict[int, Tuple[float, float, float]] = {}
        for d_id, agent in self.agents.items():
            st = drone_states.get(d_id)
            bat_ok = (st is None) or (st.battery >= agent.low_battery)
            frame_ok = (agent.last_frame_time is None) or (
                (current_time - agent.last_frame_time) <= agent.frame_timeout_seconds
            )
            conf_ok = agent.last_confidence >= agent.min_confidence
            if bat_ok and frame_ok and conf_ok:
                healthy_positions[d_id] = positions[d_id]

        assignments = self.assign_hotspots(healthy_positions, hotspots)

        # Identify sectors whose home drone has been pulled away to a hotspot
        vacated_homes: List[Tuple[float, float, float]] = []
        for d_id, (hs, _) in assignments.items():
            hx, hy, _ = self.agents[d_id].home_xyz
            if math.hypot(hs.centroid_xy[0] - hx, hs.centroid_xy[1] - hy) > 12.0:
                vacated_homes.append(self.agents[d_id].home_xyz)

        commands: List[DroneCommand] = []
        escalations: List[Escalation] = []

        for d_id in sorted(self.agents.keys()):
            agent = self.agents[d_id]
            d_state = drone_states.get(d_id)

            if d_id in assignments:
                hs, slot_idx = assignments[d_id]
                forced_exit = self.forced_guidance_exits.get(hs.hotspot_id)
                cmd, esc = agent.step(
                    current_time=current_time,
                    drone_state=d_state,
                    assigned_hotspot=hs,
                    forced_guide_exit=forced_exit,
                )
                # Spread multiple drones assigned to the same hotspot around its perimeter
                if slot_idx > 0 and cmd.state in ("RESPOND", "MONITOR"):
                    angle = 2.0 * math.pi * (slot_idx / max(2, self.max_drones_per_hotspot))
                    offset_r = self.min_sep_horizontal * 1.15
                    tx = cmd.target_xyz[0] + math.cos(angle) * offset_r
                    ty = cmd.target_xyz[1] + math.sin(angle) * offset_r
                    cmd.target_xyz = (tx, ty, cmd.target_xyz[2])
            else:
                # Unassigned drone: check if any region is stale, or widen coverage if neighbour left
                stale_xy = density_map.find_stalest_region_center(
                    current_time=current_time,
                    reference_xy=(agent.home_xyz[0], agent.home_xyz[1]),
                )
                neighbour_left = any(
                    math.hypot(vx - agent.home_xyz[0], vy - agent.home_xyz[1]) < 32.0
                    for vx, vy, _ in vacated_homes
                )
                alt = (
                    config.WIDE_COVERAGE_ALTITUDE
                    if neighbour_left
                    else config.PATROL_ALTITUDE
                )
                if stale_xy is not None:
                    fallback_xyz = (stale_xy[0], stale_xy[1], alt)
                    fallback_reason = (
                        f"Coverage memory: inspecting stale cells at ({stale_xy[0]:.0f}, {stale_xy[1]:.0f})"
                    )
                elif neighbour_left:
                    fallback_xyz = (agent.home_xyz[0], agent.home_xyz[1], alt)
                    fallback_reason = "Neighbouring drone deployed -> widening sector coverage"
                else:
                    fallback_xyz = agent.home_xyz
                    fallback_reason = "Sector clear -> PATROL"

                cmd, esc = agent.step(
                    current_time=current_time,
                    drone_state=d_state,
                    assigned_hotspot=None,
                    fallback_target_xyz=fallback_xyz,
                    fallback_reason=fallback_reason,
                )

            commands.append(cmd)
            if esc is not None:
                escalations.append(esc)

        safe_commands = self._apply_safety_layer(commands, density_map, current_time)
        return safe_commands, escalations

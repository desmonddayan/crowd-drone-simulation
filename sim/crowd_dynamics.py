"""Shared crowd social-force model and kinematic drone fleet physics.

Used by both `integration/mock_sim.py` (2D headless/OpenCV simulator) and
`sim/isaac_env.py` (NVIDIA Isaac Sim 5.1 3D stage) so crowd dynamics, surge
bottlenecks, and guidance responses behave identically in both modes.

Simulation Assumptions (explicitly documented):
1. Kinematic Drones: Drones are modelled as holonomic kinematic agents with velocity
   smoothing, maximum speed clamping, and pairwise separation repulsion rather than
   full 6-DoF aerodynamic rotor physics.
2. Crowd Guidance Response: Pedestrians inside (or within direct visual range of) a
   drone's spotlight cone while the drone is in `GUIDE` state switch their destination
   goal to `command.guide_to_exit` as a Poisson process with rate `p` per second
   (`config.GUIDANCE_SWITCH_PROB_PER_SEC`).

Inputs:
    Applied `DroneCommand` dicts/objects and physics timestep `dt`.
Outputs:
    Updated pedestrian positions/velocities/goals and `DroneState` telemetry.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import numpy as np

import config
from interfaces.contract import DroneCommand, DroneState


class CrowdAndDroneWorld:
    """Deterministic 2D social-force crowd + 3D kinematic drone fleet simulation core."""

    def __init__(
        self,
        n_drones: int = config.N_DRONES,
        n_people: int = config.N_PEOPLE,
        seed: int = 42,
        surge_start_seconds: float = config.SURGE_START_SECONDS,
    ) -> None:
        self.rng = np.random.default_rng(seed)
        self.n_drones = int(n_drones)
        self.n_people = int(n_people)
        self.surge_start_seconds = float(surge_start_seconds)
        self.sim_time: float = 0.0
        self.surge_triggered: bool = False
        self.guided_switches_count: int = 0

        # Initialize drone positions at sector centers
        self.sector_centers = config.compute_sector_centers(self.n_drones)
        self.drone_pos = np.array(self.sector_centers, dtype=np.float64)
        self.drone_vel = np.zeros((self.n_drones, 3), dtype=np.float64)
        self.drone_battery = np.ones(self.n_drones, dtype=np.float64)
        self.drone_commands: Dict[int, DroneCommand] = {}
        for i in range(self.n_drones):
            hx, hy, hz = self.sector_centers[i]
            preset = config.LIGHT_PRESETS["PATROL"]
            self.drone_commands[i] = DroneCommand(
                drone_id=i,
                state="PATROL",
                target_xyz=(hx, hy, hz),
                light_colour=preset["colour"],  # type: ignore[arg-type]
                light_intensity=float(preset["intensity"]),  # type: ignore[arg-type]
                light_pulse_hz=0.0,
                light_aim_xy=(hx, hy),
                guide_to_exit=None,
                reason="Initial PATROL",
                timestamp=0.0,
            )

        # Initialize pedestrians in front of stage and mid-venue
        self.person_pos = np.zeros((self.n_people, 2), dtype=np.float64)
        self.person_vel = np.zeros((self.n_people, 2), dtype=np.float64)
        self.person_goals: List[str] = ["watch_stage"] * self.n_people
        self.person_preferred_targets = np.zeros((self.n_people, 2), dtype=np.float64)
        self._init_people()

    def _init_people(self) -> None:
        """Spawn people watching the stage and mingling across the venue."""
        for i in range(self.n_people):
            if i < int(0.80 * self.n_people):
                # Clustered in the main arena / stage-east zone so surge reaches Exit_A funnel quickly
                x = self.rng.uniform(36.0, 58.0)
                y = self.rng.uniform(21.0, 39.0)
            else:
                x = self.rng.uniform(16.0, 55.0)
                y = self.rng.uniform(14.0, 46.0)
            self.person_pos[i] = [x, y]
            tx = self.rng.uniform(34.0, 54.0)
            ty = self.rng.uniform(22.0, 38.0)
            self.person_preferred_targets[i] = [tx, ty]

    def trigger_surge(self) -> None:
        """Trigger the scripted SURGE event: a large group heads to Exit_A at once."""
        if self.surge_triggered:
            return
        self.surge_triggered = True
        n_surge = int(round(self.n_people * config.SURGE_FRACTION))
        dists = np.hypot(self.person_pos[:, 0] - 64.0, self.person_pos[:, 1] - 30.0)
        order = np.argsort(dists)
        for idx in order[:n_surge]:
            self.person_goals[int(idx)] = "Exit_A"
            # Concentrate surge targets at the narrow choke point of the Exit_A funnel
            self.person_preferred_targets[int(idx)] = [
                self.rng.uniform(66.0, 68.5),
                self.rng.uniform(28.8, 31.2),
            ]

    def apply_command(self, cmd: DroneCommand) -> None:
        """Store the latest DroneCommand for a drone to be applied each physics step."""
        if 0 <= cmd.drone_id < self.n_drones:
            self.drone_commands[cmd.drone_id] = cmd

    def get_modulated_light_intensity(self, drone_id: int) -> float:
        """Compute instantaneous spotlight intensity with smooth sine modulation (<= 2 Hz).

        Photosensitivity note: Frequency is clamped to <= 2.0 Hz (config.MAX_PULSE_HZ)
        using a smooth sine wave (never a sharp square strobe) to remain well below
        photosensitive epilepsy thresholds.
        """
        cmd = self.drone_commands[drone_id]
        base = float(np.clip(cmd.light_intensity, 0.0, 1.0))
        freq = float(np.clip(cmd.light_pulse_hz, 0.0, config.MAX_PULSE_HZ))
        if freq <= 1e-3:
            return base
        mod = 0.60 + 0.40 * math.sin(2.0 * math.pi * freq * self.sim_time)
        return float(np.clip(base * mod, 0.05, 1.0))

    def get_drone_states(self) -> List[DroneState]:
        """Return current `DroneState` telemetry for all N drones."""
        states: List[DroneState] = []
        for i in range(self.n_drones):
            cmd = self.drone_commands[i]
            pos = self.drone_pos[i]
            vel = self.drone_vel[i]
            states.append(
                DroneState(
                    drone_id=i,
                    timestamp=self.sim_time,
                    position_xyz=(float(pos[0]), float(pos[1]), float(pos[2])),
                    velocity_xyz=(float(vel[0]), float(vel[1]), float(vel[2])),
                    battery=float(self.drone_battery[i]),
                    current_state=cmd.state,
                    light_colour=cmd.light_colour,
                    light_intensity=self.get_modulated_light_intensity(i),
                    light_pulse_hz=min(cmd.light_pulse_hz, config.MAX_PULSE_HZ),
                    light_aim_xy=cmd.light_aim_xy,
                )
            )
        return states

    def step(self, dt: float = 0.1) -> None:
        """Advance drone kinematics and crowd social-force dynamics by `dt` seconds."""
        dt = float(dt)
        self.sim_time += dt

        if not self.surge_triggered and self.sim_time >= self.surge_start_seconds:
            self.trigger_surge()

        self._step_drones(dt)
        self._step_crowd_guidance(dt)
        self._step_crowd_social_forces(dt)

    def _step_drones(self, dt: float) -> None:
        """Move drones kinematically toward target_xyz with max speed & pairwise repulsion."""
        desired_vel = np.zeros_like(self.drone_vel)
        for i in range(self.n_drones):
            cmd = self.drone_commands[i]
            target = np.array(cmd.target_xyz, dtype=np.float64)
            diff = target - self.drone_pos[i]
            dist = float(np.linalg.norm(diff))
            if dist > 1e-4:
                speed = min(config.MAX_DRONE_SPEED, dist * config.DRONE_ACCEL_GAIN)
                desired_vel[i] = (diff / dist) * speed

        # Pairwise separation repulsion between drones
        for i in range(self.n_drones):
            for j in range(i + 1, self.n_drones):
                diff_xy = self.drone_pos[i, :2] - self.drone_pos[j, :2]
                dist_xy = float(np.hypot(diff_xy[0], diff_xy[1]))
                dz = abs(float(self.drone_pos[i, 2] - self.drone_pos[j, 2]))
                if dist_xy < config.MIN_SEP_HORIZONTAL and dz < config.MIN_SEP_VERTICAL:
                    if dist_xy < 1e-3:
                        unit = np.array([1.0, 0.0])
                    else:
                        unit = diff_xy / dist_xy
                    rep = config.REPULSION_GAIN * (config.MIN_SEP_HORIZONTAL - dist_xy) * unit
                    desired_vel[i, :2] += rep
                    desired_vel[j, :2] -= rep

        # Smooth velocity update and speed cap
        for i in range(self.n_drones):
            v = 0.4 * self.drone_vel[i] + 0.6 * desired_vel[i]
            spd = float(np.linalg.norm(v))
            if spd > config.MAX_DRONE_SPEED:
                v = (v / spd) * config.MAX_DRONE_SPEED
            self.drone_vel[i] = v
            self.drone_pos[i] += v * dt
            self.drone_pos[i, 0] = np.clip(self.drone_pos[i, 0], 1.0, config.VENUE_WIDTH - 1.0)
            self.drone_pos[i, 1] = np.clip(self.drone_pos[i, 1], 1.0, config.VENUE_HEIGHT - 1.0)
            self.drone_pos[i, 2] = np.clip(
                self.drone_pos[i, 2], config.INSPECTION_ALTITUDE, config.SAFE_ALTITUDE
            )
            self.drone_battery[i] = max(0.05, self.drone_battery[i] - 0.0002 * dt)

    def _step_crowd_guidance(self, dt: float) -> None:
        """Switch goals of pedestrians inside a GUIDE drone's spotlight cone with prob p/s."""
        if self.n_people == 0:
            return
        switch_prob = 1.0 - math.exp(-config.GUIDANCE_SWITCH_PROB_PER_SEC * dt)
        half_angle_rad = math.radians(config.LIGHT_CONE_HALF_ANGLE_DEG)

        for d_id in range(self.n_drones):
            cmd = self.drone_commands[d_id]
            if cmd.state != "GUIDE" or not cmd.guide_to_exit:
                continue
            target_exit = cmd.guide_to_exit
            if target_exit not in config.EXIT_TARGETS:
                continue

            dx, dy, dz = self.drone_pos[d_id]
            aim_x, aim_y = cmd.light_aim_xy
            # Spotlight cone center on ground is between nadir (dx, dy) and aim_xy
            cone_cx = 0.55 * dx + 0.45 * aim_x
            cone_cy = 0.55 * dy + 0.45 * aim_y
            # Effective ground radius of the spotlight cone + visual awareness halo
            cone_radius = max(10.0, dz * math.tan(half_angle_rad) * 1.45)

            dists_to_cone = np.hypot(
                self.person_pos[:, 0] - cone_cx, self.person_pos[:, 1] - cone_cy
            )
            dists_to_nadir = np.hypot(
                self.person_pos[:, 0] - dx, self.person_pos[:, 1] - dy
            )
            in_cone = (dists_to_cone <= cone_radius) | (dists_to_nadir <= cone_radius)

            indices = np.nonzero(in_cone)[0]
            for idx in indices:
                p_idx = int(idx)
                if self.person_goals[p_idx] != target_exit:
                    if self.rng.random() < switch_prob:
                        self.person_goals[p_idx] = target_exit
                        ex, ey = config.EXIT_TARGETS[target_exit]
                        self.person_preferred_targets[p_idx] = [
                            ex + self.rng.uniform(-3.0, 3.0),
                            ey + self.rng.uniform(-1.5, 1.5),
                        ]
                        self.guided_switches_count += 1

    def _step_crowd_social_forces(self, dt: float) -> None:
        """Update pedestrian positions using goal attraction + neighbour/barrier repulsion."""
        if self.n_people == 0:
            return

        # 1. Goal attraction vector
        targets = self.person_preferred_targets.copy()
        for i, goal in enumerate(self.person_goals):
            if goal == "Exit_A":
                # If west of the funnel neck, aim into the funnel neck (67.5, 30.0)
                if self.person_pos[i, 0] < 65.0:
                    targets[i] = [67.5, 30.0]
            elif goal in config.EXIT_TARGETS:
                targets[i] = list(config.EXIT_TARGETS[goal])

        diff_goal = targets - self.person_pos
        dist_goal = np.hypot(diff_goal[:, 0], diff_goal[:, 1])[:, None]
        dir_goal = np.where(dist_goal > 0.2, diff_goal / np.maximum(dist_goal, 1e-4), 0.0)

        # People in the Exit_A funnel bottleneck experience a slow exit gate (narrow turnstile)
        in_exit_a_neck = (
            (self.person_pos[:, 0] > 64.0)
            & (self.person_pos[:, 0] < 74.0)
            & (np.abs(self.person_pos[:, 1] - 30.0) < 4.5)
        )
        desired_speed = np.full((self.n_people, 1), config.MAX_PERSON_SPEED, dtype=np.float64)
        for i in range(self.n_people):
            if in_exit_a_neck[i] and self.person_goals[i] == "Exit_A":
                # Bottleneck slows forward progress at the choke point so density builds realistically
                desired_speed[i, 0] = 0.25
            elif self.person_goals[i] == "Exit_B":
                # Guided attendees disperse briskly toward Exit_B
                desired_speed[i, 0] = config.MAX_PERSON_SPEED

        force = dir_goal * desired_speed * 2.4

        # 2. Vectorized pairwise short-range repulsion (allows ~0.24m packing in a crush bottleneck)
        dx = self.person_pos[:, 0][:, None] - self.person_pos[:, 0][None, :]
        dy = self.person_pos[:, 1][:, None] - self.person_pos[:, 1][None, :]
        dist = np.hypot(dx, dy)
        np.fill_diagonal(dist, 1e9)
        min_personal_dist = 0.24
        close_mask = dist < min_personal_dist
        if np.any(close_mask):
            mag = np.where(close_mask, (min_personal_dist - dist) * 2.5, 0.0)
            ux = np.where(close_mask, dx / np.maximum(dist, 1e-3), 0.0)
            uy = np.where(close_mask, dy / np.maximum(dist, 1e-3), 0.0)
            force[:, 0] += np.sum(mag * ux, axis=1)
            force[:, 1] += np.sum(mag * uy, axis=1)

        # 3. Barrier segment repulsion
        for (x1, y1), (x2, y2) in config.BARRIER_SEGMENTS:
            seg_dx, seg_dy = x2 - x1, y2 - y1
            seg_len_sq = max(seg_dx * seg_dx + seg_dy * seg_dy, 1e-6)
            t_proj = np.clip(
                ((self.person_pos[:, 0] - x1) * seg_dx + (self.person_pos[:, 1] - y1) * seg_dy)
                / seg_len_sq,
                0.0,
                1.0,
            )
            closest_x = x1 + t_proj * seg_dx
            closest_y = y1 + t_proj * seg_dy
            bdx = self.person_pos[:, 0] - closest_x
            bdy = self.person_pos[:, 1] - closest_y
            bdist = np.hypot(bdx, bdy)
            bmask = bdist < 0.85
            if np.any(bmask):
                bmag = np.where(bmask, (0.85 - bdist) * 4.5, 0.0)
                force[:, 0] += bmag * (bdx / np.maximum(bdist, 1e-3))
                force[:, 1] += bmag * (bdy / np.maximum(bdist, 1e-3))

        # 4. Update velocity and position with max speed cap
        vel = 0.3 * self.person_vel + 0.7 * force
        speed = np.hypot(vel[:, 0], vel[:, 1])
        max_spd = config.MAX_PERSON_SPEED
        over = speed > max_spd
        if np.any(over):
            vel[over, 0] = vel[over, 0] / speed[over] * max_spd
            vel[over, 1] = vel[over, 1] / speed[over] * max_spd

        self.person_vel = vel
        self.person_pos += self.person_vel * dt
        self.person_pos[:, 0] = np.clip(self.person_pos[:, 0], 11.2, config.VENUE_WIDTH - 1.0)
        self.person_pos[:, 1] = np.clip(self.person_pos[:, 1], 1.5, config.VENUE_HEIGHT - 1.0)

    def count_people_in_exit_a_funnel(self) -> int:
        """Helper metric for tests/evaluation: count people currently inside Exit_A funnel."""
        in_funnel = (
            (self.person_pos[:, 0] >= 54.0)
            & (self.person_pos[:, 0] <= 76.0)
            & (self.person_pos[:, 1] >= 22.0)
            & (self.person_pos[:, 1] <= 38.0)
        )
        return int(np.sum(in_funnel))

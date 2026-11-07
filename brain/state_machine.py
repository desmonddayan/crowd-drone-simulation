"""Per-drone autonomous state machine with hysteresis and failsafe handling.

States:
    - PATROL: Hover at home sector (or stale coverage target) at patrol/wide altitude,
      steady low-intensity green light.
    - MONITOR: AMBER hotspot in/near sector or assigned AMBER hotspot; slow-pulsing amber light.
    - RESPOND: Assigned to a RED hotspot; fly above hotspot centroid at inspection altitude
      (never below inspection altitude over RED cells), pulsing red light.
    - GUIDE: Assigned RED hotspot has remained RED & rising (or high severity) for > GUIDE_DELAY;
      position offset from hotspot toward least-dense exit, aim spotlight toward that exit,
      pulse distinct cyan-green guidance light.
    - FAILSAFE: Triggered immediately when frame age > FRAME_TIMEOUT, confidence < MIN_CONFIDENCE,
      or battery < LOW_BATTERY; climb to safe altitude, move to safe zone, emit Escalation.

Inputs:
    Drone telemetry, latest perception timestamp & confidence, assigned/local Hotspot,
    and strategic overrides.
Outputs:
    Next state, DroneCommand, and optional Escalation.
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import config
from interfaces.contract import DroneCommand, DroneState, Escalation, Hotspot


class DroneAgent:
    """Autonomous state machine for a single drone."""

    def __init__(
        self,
        drone_id: int,
        home_xyz: Tuple[float, float, float],
        min_dwell_seconds: float = config.MIN_DWELL_SECONDS,
        guide_delay_seconds: float = config.GUIDE_DELAY_SECONDS,
        frame_timeout_seconds: float = config.FRAME_TIMEOUT_SECONDS,
        min_confidence: float = config.MIN_CONFIDENCE_THRESHOLD,
        low_battery: float = config.LOW_BATTERY_THRESHOLD,
    ) -> None:
        self.drone_id = int(drone_id)
        self.home_xyz = (float(home_xyz[0]), float(home_xyz[1]), float(home_xyz[2]))
        self.min_dwell_seconds = float(min_dwell_seconds)
        self.guide_delay_seconds = float(guide_delay_seconds)
        self.frame_timeout_seconds = float(frame_timeout_seconds)
        self.min_confidence = float(min_confidence)
        self.low_battery = float(low_battery)

        self.state: str = "PATROL"
        self.state_entered_at: float = -1e9
        self.last_frame_time: Optional[float] = None
        self.last_confidence: float = 1.0
        self.red_rising_since: Optional[float] = None
        self.last_hotspot_peak: float = 0.0

    def record_perception(self, timestamp: float, confidence: float) -> None:
        """Record fresh perception metadata from this drone's camera."""
        self.last_frame_time = float(timestamp)
        self.last_confidence = float(confidence)

    def _can_transition(self, current_time: float, target_state: str) -> bool:
        """Enforce minimum dwell time hysteresis except when escalating to FAILSAFE or higher urgency."""
        if target_state == self.state:
            return False
        if target_state == "FAILSAFE":
            return True
        urgency = {"PATROL": 0, "MONITOR": 1, "RESPOND": 2, "GUIDE": 3, "FAILSAFE": 4}
        elapsed = current_time - self.state_entered_at
        # Allow immediate escalation from PATROL -> MONITOR -> RESPOND if dwell >= 0.5*min_dwell
        # but enforce full min_dwell_seconds before de-escalating to prevent flicker.
        if urgency.get(target_state, 0) < urgency.get(self.state, 0):
            return elapsed >= self.min_dwell_seconds
        return elapsed >= (0.5 * self.min_dwell_seconds) or self.state_entered_at < 0.0

    def step(
        self,
        current_time: float,
        drone_state: Optional[DroneState],
        assigned_hotspot: Optional[Hotspot],
        fallback_target_xyz: Optional[Tuple[float, float, float]] = None,
        fallback_reason: str = "Sector clear -> PATROL",
        forced_guide_exit: Optional[str] = None,
    ) -> Tuple[DroneCommand, Optional[Escalation]]:
        """Evaluate state transitions and produce a DroneCommand (and Escalation if FAILSAFE)."""
        battery = drone_state.battery if drone_state is not None else 1.0

        # 1. Check FAILSAFE conditions (safety override)
        failsafe_reason: Optional[str] = None
        if battery < self.low_battery:
            failsafe_reason = f"Battery {battery:.0%} < {self.low_battery:.0%} -> FAILSAFE"
        elif (
            self.last_frame_time is not None
            and (current_time - self.last_frame_time) > self.frame_timeout_seconds
        ):
            age = current_time - self.last_frame_time
            failsafe_reason = (
                f"Frame timeout {age:.1f}s > {self.frame_timeout_seconds:.1f}s -> FAILSAFE"
            )
        elif self.last_confidence < self.min_confidence:
            failsafe_reason = (
                f"Low confidence {self.last_confidence:.2f} < {self.min_confidence:.2f} -> FAILSAFE"
            )

        if failsafe_reason is not None:
            if self.state != "FAILSAFE":
                self.state = "FAILSAFE"
                self.state_entered_at = current_time
            self.red_rising_since = None
            preset = config.LIGHT_PRESETS["FAILSAFE"]
            safe_xyz = (
                config.SAFE_ZONE_XYZ[0] + (self.drone_id % 5) * config.MIN_SEP_HORIZONTAL,
                config.SAFE_ZONE_XYZ[1] + (self.drone_id // 5) * config.MIN_SEP_HORIZONTAL,
                config.SAFE_ALTITUDE,
            )
            cmd = DroneCommand(
                drone_id=self.drone_id,
                state="FAILSAFE",
                target_xyz=safe_xyz,
                light_colour=preset["colour"],  # type: ignore[arg-type]
                light_intensity=float(preset["intensity"]),  # type: ignore[arg-type]
                light_pulse_hz=0.0,
                light_aim_xy=(safe_xyz[0], safe_xyz[1]),
                guide_to_exit=None,
                reason=failsafe_reason,
                timestamp=current_time,
            )
            esc = Escalation(
                drone_id=self.drone_id,
                timestamp=current_time,
                zone=f"Sector_{self.drone_id}",
                reason=failsafe_reason,
                severity="HIGH",
            )
            return cmd, esc

        # Recover from FAILSAFE once telemetry is healthy and dwell time has passed
        if self.state == "FAILSAFE" and self._can_transition(current_time, "PATROL"):
            self.state = "PATROL"
            self.state_entered_at = current_time

        # 2. Determine desired operational state from hotspot + hysteresis thresholds
        desired_state = "PATROL"
        if assigned_hotspot is not None:
            peak = assigned_hotspot.peak_density
            trend = assigned_hotspot.mean_trend
            self.last_hotspot_peak = peak

            is_red = assigned_hotspot.level == "RED" or peak >= config.RED_THRESHOLD
            # Hysteresis: if already in RESPOND or GUIDE, stay in RED tier until below RED_EXIT_THRESHOLD
            if self.state in ("RESPOND", "GUIDE") and peak >= config.RED_EXIT_THRESHOLD:
                is_red = True

            if is_red:
                # Track how long the hotspot has been RED and rising (or sustained severe RED)
                is_rising_or_severe = (
                    trend >= -0.05 or peak >= (config.RED_THRESHOLD + 0.5)
                )
                if is_rising_or_severe:
                    if self.red_rising_since is None:
                        self.red_rising_since = current_time
                else:
                    self.red_rising_since = None

                red_duration = (
                    (current_time - self.red_rising_since)
                    if self.red_rising_since is not None
                    else 0.0
                )
                if (
                    forced_guide_exit is not None
                    or red_duration >= self.guide_delay_seconds
                    or (self.state == "GUIDE" and peak >= config.RED_EXIT_THRESHOLD)
                ):
                    desired_state = "GUIDE"
                else:
                    desired_state = "RESPOND"
            else:
                self.red_rising_since = None
                is_amber = assigned_hotspot.level == "AMBER" or peak >= config.AMBER_THRESHOLD
                if self.state == "MONITOR" and peak >= config.AMBER_EXIT_THRESHOLD:
                    is_amber = True
                if is_amber:
                    desired_state = "MONITOR"
        else:
            self.red_rising_since = None
            # Hysteresis on de-escalation when hotspot briefly disappears
            if self.state != "PATROL" and not self._can_transition(current_time, "PATROL"):
                desired_state = self.state

        if desired_state != self.state and self._can_transition(current_time, desired_state):
            self.state = desired_state
            self.state_entered_at = current_time

        # 3. Build target position, spotlight parameters, and human-readable reason
        preset = config.LIGHT_PRESETS[self.state]
        colour: Tuple[float, float, float] = preset["colour"]  # type: ignore[assignment]
        intensity = float(preset["intensity"])  # type: ignore[arg-type]
        pulse_hz = min(float(preset["pulse_hz"]), config.MAX_PULSE_HZ)  # type: ignore[arg-type]

        if self.state == "GUIDE" and assigned_hotspot is not None:
            hx, hy = assigned_hotspot.centroid_xy
            guide_exit = forced_guide_exit or assigned_hotspot.least_dense_exit
            ex, ey = config.EXIT_TARGETS.get(guide_exit, config.EXIT_TARGETS["Exit_B"])
            dx, dy = ex - hx, ey - hy
            dist = max(math.hypot(dx, dy), 1e-3)
            ux, uy = dx / dist, dy / dist
            # Stay offset 3.5 m from hotspot centroid toward the target exit
            tx = hx + ux * 3.5
            ty = hy + uy * 3.5
            tz = config.INSPECTION_ALTITUDE
            # Aim spotlight 8 m ahead along the guidance vector toward the exit
            aim_xy = (hx + ux * 9.0, hy + uy * 9.0)
            reason = (
                f"Hotspot #{assigned_hotspot.hotspot_id} {assigned_hotspot.peak_density:.1f} p/m² "
                f"(trend {assigned_hotspot.mean_trend:+.2f}/s) -> GUIDE to {guide_exit}"
            )
            return (
                DroneCommand(
                    drone_id=self.drone_id,
                    state="GUIDE",
                    target_xyz=(tx, ty, tz),
                    light_colour=colour,
                    light_intensity=intensity,
                    light_pulse_hz=pulse_hz,
                    light_aim_xy=aim_xy,
                    guide_to_exit=guide_exit,
                    reason=reason,
                    timestamp=current_time,
                ),
                None,
            )

        if self.state == "RESPOND" and assigned_hotspot is not None:
            hx, hy = assigned_hotspot.centroid_xy
            tx, ty, tz = hx, hy, config.INSPECTION_ALTITUDE
            reason = (
                f"{assigned_hotspot.nearest_exit} hotspot #{assigned_hotspot.hotspot_id} "
                f"{assigned_hotspot.peak_density:.1f} p/m² "
                f"rising {assigned_hotspot.mean_trend:+.2f}/s -> RESPOND"
            )
            return (
                DroneCommand(
                    drone_id=self.drone_id,
                    state="RESPOND",
                    target_xyz=(tx, ty, tz),
                    light_colour=colour,
                    light_intensity=intensity,
                    light_pulse_hz=pulse_hz,
                    light_aim_xy=(hx, hy),
                    guide_to_exit=None,
                    reason=reason,
                    timestamp=current_time,
                ),
                None,
            )

        if self.state == "MONITOR" and assigned_hotspot is not None:
            hx, hy = assigned_hotspot.centroid_xy
            # Hover halfway between sector home and AMBER hotspot at patrol altitude
            tx = 0.5 * self.home_xyz[0] + 0.5 * hx
            ty = 0.5 * self.home_xyz[1] + 0.5 * hy
            tz = config.PATROL_ALTITUDE
            reason = (
                f"Hotspot #{assigned_hotspot.hotspot_id} AMBER "
                f"({assigned_hotspot.peak_density:.1f} p/m²) -> MONITOR"
            )
            return (
                DroneCommand(
                    drone_id=self.drone_id,
                    state="MONITOR",
                    target_xyz=(tx, ty, tz),
                    light_colour=colour,
                    light_intensity=intensity,
                    light_pulse_hz=pulse_hz,
                    light_aim_xy=(hx, hy),
                    guide_to_exit=None,
                    reason=reason,
                    timestamp=current_time,
                ),
                None,
            )

        # Default PATROL (or holding previous state during dwell de-escalation)
        target_xyz = fallback_target_xyz if fallback_target_xyz is not None else self.home_xyz
        return (
            DroneCommand(
                drone_id=self.drone_id,
                state=self.state,
                target_xyz=target_xyz,
                light_colour=colour,
                light_intensity=intensity,
                light_pulse_hz=pulse_hz,
                light_aim_xy=(target_xyz[0], target_xyz[1]),
                guide_to_exit=None,
                reason=fallback_reason,
                timestamp=current_time,
            ),
            None,
        )

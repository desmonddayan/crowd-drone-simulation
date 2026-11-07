"""Unit tests for brain/state_machine.py and brain/coordinator.py."""

from __future__ import annotations

import math

import config
from brain.coordinator import FleetCoordinator
from brain.density_map import SharedDensityMap
from brain.state_machine import DroneAgent
from interfaces.contract import DroneState, Hotspot, PerceptionResult


def test_coordinator_two_hotspots_three_drones_and_separation() -> None:
    coord = FleetCoordinator(n_drones=3)
    dmap = SharedDensityMap()

    # Seed the density map so cells aren't all stale
    full_fp = [
        (0.0, 0.0),
        (config.VENUE_WIDTH, 0.0),
        (config.VENUE_WIDTH, config.VENUE_HEIGHT),
        (0.0, config.VENUE_HEIGHT),
    ]
    # Hotspot 1 near (20, 20) (compact 3x3m RED), Hotspot 2 near (65, 30) (compact 3x3m RED)
    pts = []
    for dx in (-1.0, 0.0, 1.0):
        for dy in (-1.0, 0.0, 1.0):
            pts.extend([(20.5 + dx, 20.5 + dy)] * 8)
            pts.extend([(65.5 + dx, 30.5 + dy)] * 8)
    for t in (1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0):
        dmap.update(
            PerceptionResult(
                drone_id=0,
                timestamp=t,
                ground_points_xy=pts,
                footprint_polygon_xy=full_fp,
                confidence=0.95,
            )
        )

    states = {
        0: DroneState(drone_id=0, timestamp=4.0, position_xyz=(18.0, 18.0, 20.0), velocity_xyz=(0.0, 0.0, 0.0)),
        1: DroneState(drone_id=1, timestamp=4.0, position_xyz=(62.0, 28.0, 20.0), velocity_xyz=(0.0, 0.0, 0.0)),
        2: DroneState(drone_id=2, timestamp=4.0, position_xyz=(40.0, 45.0, 20.0), velocity_xyz=(0.0, 0.0, 0.0)),
    }
    for i in range(3):
        coord.record_perception(i, timestamp=4.0, confidence=0.95)

    commands, escalations = coord.step(current_time=4.0, density_map=dmap, drone_states=states)
    assert len(commands) == 3
    assert len(escalations) == 0

    # Drone 0 (near 20,20) and Drone 1 (near 65,30) should respond to their nearest hotspots
    cmd_by_id = {c.drone_id: c for c in commands}
    assert cmd_by_id[0].state == "RESPOND"
    assert math.hypot(cmd_by_id[0].target_xyz[0] - 20.5, cmd_by_id[0].target_xyz[1] - 20.5) < 6.0
    assert cmd_by_id[1].state == "RESPOND"
    assert math.hypot(cmd_by_id[1].target_xyz[0] - 65.5, cmd_by_id[1].target_xyz[1] - 30.5) < 6.0

    # Verify minimum separation across all pairs of output targets
    for i in range(len(commands)):
        for j in range(i + 1, len(commands)):
            p1 = commands[i].target_xyz
            p2 = commands[j].target_xyz
            d_xy = math.hypot(p1[0] - p2[0], p1[1] - p2[1])
            d_z = abs(p1[2] - p2[2])
            assert d_xy >= config.MIN_SEP_HORIZONTAL - 1e-2 or d_z >= config.MIN_SEP_VERTICAL


def test_safety_layer_separates_colocated_drones() -> None:
    coord = FleetCoordinator(n_drones=3)
    dmap = SharedDensityMap()
    # One large RED hotspot so all 3 drones want the same centroid
    full_fp = [
        (0.0, 0.0),
        (config.VENUE_WIDTH, 0.0),
        (config.VENUE_WIDTH, config.VENUE_HEIGHT),
        (0.0, config.VENUE_HEIGHT),
    ]
    pts = []
    for x in range(45, 53):
        for y in range(25, 33):
            pts.extend([(x + 0.5, y + 0.5)] * 8)

    for t in (1.0, 1.5, 2.0, 2.5):
        dmap.update(
            PerceptionResult(
                drone_id=0,
                timestamp=t,
                ground_points_xy=pts,
                footprint_polygon_xy=full_fp,
                confidence=0.95,
            )
        )

    for i in range(3):
        coord.record_perception(i, timestamp=2.5, confidence=0.95)

    commands, _ = coord.step(current_time=2.5, density_map=dmap)
    for i in range(len(commands)):
        for j in range(i + 1, len(commands)):
            p1 = commands[i].target_xyz
            p2 = commands[j].target_xyz
            d_xy = math.hypot(p1[0] - p2[0], p1[1] - p2[1])
            assert d_xy >= config.MIN_SEP_HORIZONTAL - 1e-2


def test_state_machine_hysteresis_guide_and_failsafe() -> None:
    agent = DroneAgent(
        drone_id=0,
        home_xyz=(20.0, 20.0, 20.0),
        min_dwell_seconds=1.5,
        guide_delay_seconds=2.5,
        frame_timeout_seconds=3.0,
        min_confidence=0.25,
    )
    d_state = DroneState(
        drone_id=0,
        timestamp=0.0,
        position_xyz=(20.0, 20.0, 20.0),
        velocity_xyz=(0.0, 0.0, 0.0),
    )
    agent.record_perception(timestamp=0.0, confidence=0.9)

    # 1. Start in PATROL
    cmd, esc = agent.step(current_time=0.0, drone_state=d_state, assigned_hotspot=None)
    assert cmd.state == "PATROL"
    assert esc is None

    # 2. AMBER hotspot -> MONITOR
    amber_hs = Hotspot(
        hotspot_id=1,
        level="AMBER",
        centroid_xy=(25.0, 25.0),
        area_m2=12.0,
        peak_density=3.9,
        mean_trend=0.05,
        severity=49.0,
        nearest_exit="Exit_A",
        least_dense_exit="Exit_B",
    )
    agent.record_perception(timestamp=1.0, confidence=0.9)
    cmd, _ = agent.step(current_time=1.0, drone_state=d_state, assigned_hotspot=amber_hs)
    assert cmd.state == "MONITOR"

    # 3. Hysteresis: immediately removing the hotspot at t=1.2 (< 1.5s dwell) must NOT flicker back to PATROL
    agent.record_perception(timestamp=1.2, confidence=0.9)
    cmd, _ = agent.step(current_time=1.2, drone_state=d_state, assigned_hotspot=None)
    assert cmd.state == "MONITOR"

    # 4. RED rising hotspot -> RESPOND first, then GUIDE after guide_delay_seconds (2.5s)
    red_hs = Hotspot(
        hotspot_id=1,
        level="RED",
        centroid_xy=(65.0, 30.0),
        area_m2=25.0,
        peak_density=5.8,
        mean_trend=0.35,
        severity=195.0,
        nearest_exit="Exit_A",
        least_dense_exit="Exit_B",
    )
    agent.record_perception(timestamp=2.0, confidence=0.9)
    cmd, _ = agent.step(current_time=2.0, drone_state=d_state, assigned_hotspot=red_hs)
    assert cmd.state == "RESPOND"

    # After 2.6s of sustained rising RED (t=4.6), agent transitions to GUIDE toward Exit_B
    agent.record_perception(timestamp=4.6, confidence=0.9)
    cmd, _ = agent.step(current_time=4.6, drone_state=d_state, assigned_hotspot=red_hs)
    assert cmd.state == "GUIDE"
    assert cmd.guide_to_exit == "Exit_B"

    # 5. Low confidence triggers immediate FAILSAFE + Escalation
    agent.record_perception(timestamp=5.0, confidence=0.10)
    cmd, esc = agent.step(current_time=5.0, drone_state=d_state, assigned_hotspot=red_hs)
    assert cmd.state == "FAILSAFE"
    assert esc is not None

    # 6. Frame timeout (> 3.0s since last frame) also triggers FAILSAFE
    agent.record_perception(timestamp=6.0, confidence=0.95)
    cmd, esc = agent.step(current_time=9.5, drone_state=d_state, assigned_hotspot=None)
    assert cmd.state == "FAILSAFE"
    assert esc is not None
    assert "timeout" in esc.reason.lower()

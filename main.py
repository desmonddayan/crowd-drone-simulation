"""Main entry point for the autonomous drone fleet crowd-congestion monitoring system.

Supports two execution modes:
- `--mock`: Runs the 2D social-force crowd & synthetic downward camera simulator
  (`integration/mock_sim.py`) end-to-end with the camera-only perception pipeline,
  shared density map, per-drone state machines, fleet coordinator, OpenCV dashboard,
  and JSONL decision logger.
- `--live`: Connects via ZeroMQ (`sim/zmq_bridge.py`) to a running NVIDIA Isaac Sim 5.1
  instance (`C:\\isaacsim\\python.bat -m sim.isaac_env`).

Hard Rule Enforcement:
    All agent decisions use ONLY `PerceptionResult` derived from decoded JPEG camera
    frames (`FrameMessage`). Ground-truth positions are never read by the agent.

Inputs:
    CLI flags (`--mock`, `--live`, `--drones`, `--steps`, `--headless`, `--yolo`, `--strategist`).
Outputs:
    Live OpenCV dashboard, JSONL decision logs in `runs/`, and console summary.
"""

from __future__ import annotations

import argparse
import time
from typing import Dict, List, Set

import numpy as np

import config
from brain.coordinator import FleetCoordinator
from brain.density_map import SharedDensityMap
from brain.strategist import LocalStrategist
from integration.dashboard import Dashboard
from integration.logger import DecisionLogger
from integration.mock_sim import MockCrowdSimulator
from interfaces.contract import DroneCommand, DroneState, Escalation, PerceptionResult
from perception.detector import CrowdPerceptionPipeline
from sim.zmq_bridge import AgentZmqEndpoint


def parse_args(argv: List[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Autonomous Drone Fleet Crowd-Congestion Monitoring System"
    )
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument(
        "--mock",
        action="store_true",
        default=True,
        help="Run with built-in 2D mock simulator (default)",
    )
    mode_group.add_argument(
        "--live",
        action="store_true",
        help="Connect to live NVIDIA Isaac Sim process over ZeroMQ",
    )
    parser.add_argument(
        "--drones",
        type=int,
        default=config.N_DRONES,
        help="Number of autonomous drones in the fleet (e.g. 4 or 25)",
    )
    parser.add_argument(
        "--people",
        type=int,
        default=config.N_PEOPLE,
        help="Number of simulated concert attendees",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=280,
        help="Number of simulation steps to run (dt=0.1s per step; default 280 = 28s)",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Disable interactive OpenCV window display",
    )
    parser.add_argument(
        "--yolo",
        action="store_true",
        help="Enable local YOLO nano person detector (falls back to blob detector if unavailable)",
    )
    parser.add_argument(
        "--strategist",
        action="store_true",
        help="Enable optional background Ollama (Gemma 4) strategic layer",
    )
    return parser.parse_args(argv)


def run_mock_simulation(args: argparse.Namespace) -> Dict[str, object]:
    """Run the full perceive -> decide -> act loop in mock mode and return summary metrics."""
    n_drones = int(args.drones)
    sim = MockCrowdSimulator(
        n_drones=n_drones,
        n_people=int(args.people),
        surge_start_seconds=3.5,
        max_cameras_per_step=max(8, n_drones // 2),
    )
    perception = CrowdPerceptionPipeline(use_yolo=bool(args.yolo))
    density_map = SharedDensityMap()
    coordinator = FleetCoordinator(n_drones=n_drones)
    dashboard = Dashboard(show_window=not bool(args.headless))
    logger = DecisionLogger()

    strategist = LocalStrategist() if args.strategist else None
    if strategist is not None:
        strategist.start_background(coordinator)

    observed_states: Set[str] = set()
    latest_perception: Dict[int, PerceptionResult] = {}
    commands: List[DroneCommand] = []
    escalations: List[Escalation] = []

    peak_exit_a_density: float = 0.0
    post_guide_exit_a_density: float = 0.0
    first_guide_time: float | None = None

    print(
        f"[START] Mock Simulation | Drones={n_drones} | People={args.people} | Steps={args.steps}"
    )

    try:
        for step_idx in range(int(args.steps)):
            # 1. Step simulation physics & capture downward camera frames
            # Capture all cameras every step so 25-drone perception updates smoothly
            frames, state_list = sim.step(dt=0.1, capture_all_cameras=True)
            drone_states: Dict[int, DroneState] = {s.drone_id: s for s in state_list}

            # 2. PERCEIVE: Decode camera JPEGs, detect people, project to ground, update density map
            for frame_msg in frames:
                perc_res = perception.process_frame(frame_msg)
                latest_perception[frame_msg.drone_id] = perc_res
                density_map.update(perc_res)
                coordinator.record_perception(
                    drone_id=perc_res.drone_id,
                    timestamp=perc_res.timestamp,
                    confidence=perc_res.confidence,
                )

            # 3. DECIDE: Run fleet coordinator and per-drone state machines (at 5 Hz -> every 2 steps)
            if step_idx % 2 == 0 or not commands:
                commands, escalations = coordinator.step(
                    current_time=sim.sim_time,
                    density_map=density_map,
                    drone_states=drone_states,
                )
                for cmd in commands:
                    observed_states.add(cmd.state)
                    if cmd.state == "GUIDE" and first_guide_time is None:
                        first_guide_time = sim.sim_time

                logger.log_tick(
                    sim_time=sim.sim_time,
                    commands=commands,
                    hotspots=density_map.hotspots,
                    escalations=escalations,
                )

                if strategist is not None:
                    strategist.update_context(sim.sim_time, density_map.hotspots, drone_states)

            # 4. ACT: Send DroneCommands back to the simulator
            sim.apply_commands(commands)

            # Track Exit A approach density over time (from camera-derived density map)
            exit_dens = density_map.get_exit_densities()
            exit_a_d = float(np.max(density_map.density[:, 52:76]))
            if exit_a_d > peak_exit_a_density:
                peak_exit_a_density = exit_a_d
            post_guide_exit_a_density = exit_a_d

            # Render OpenCV HUD frame
            dashboard.render_frame(
                current_time=sim.sim_time,
                density_map=density_map,
                drone_states=drone_states,
                commands=commands,
                perception_results=latest_perception,
                decoded_rgb_frames=perception.last_decoded_rgb,
                escalations=escalations,
            )

            if step_idx % 40 == 0 or step_idx == int(args.steps) - 1:
                state_counts: Dict[str, int] = {}
                for c in commands:
                    state_counts[c.state] = state_counts.get(c.state, 0) + 1
                print(
                    f"[t={sim.sim_time:5.1f}s] States={state_counts} | "
                    f"Hotspots={len(density_map.hotspots)} | "
                    f"Exit_A peak={exit_a_d:.2f} p/m² | "
                    f"Guided redirects={sim.world.guided_switches_count}"
                )
    finally:
        if strategist is not None:
            strategist.stop()
        logger.close()
        dashboard.close()

    summary = {
        "observed_states": sorted(observed_states),
        "peak_exit_a_density": round(peak_exit_a_density, 2),
        "final_exit_a_density": round(post_guide_exit_a_density, 2),
        "first_guide_time": first_guide_time,
        "guided_switches": sim.world.guided_switches_count,
        "log_file": str(logger.file_path),
    }
    print(
        f"[DONE] Observed states: {summary['observed_states']} | "
        f"Exit_A peak density: {summary['peak_exit_a_density']:.2f} p/m² -> "
        f"final: {summary['final_exit_a_density']:.2f} p/m² | "
        f"Log: {summary['log_file']}"
    )
    return summary


def run_live_agent(args: argparse.Namespace) -> None:
    """Run the autonomous agent connected to Isaac Sim over ZeroMQ."""
    n_drones = int(args.drones)
    zmq_agent = AgentZmqEndpoint()
    perception = CrowdPerceptionPipeline(use_yolo=bool(args.yolo))
    density_map = SharedDensityMap()
    coordinator = FleetCoordinator(n_drones=n_drones)
    dashboard = Dashboard(show_window=not bool(args.headless))
    logger = DecisionLogger()

    strategist = LocalStrategist() if args.strategist else None
    if strategist is not None:
        strategist.start_background(coordinator)

    drone_states: Dict[int, DroneState] = {}
    latest_perception: Dict[int, PerceptionResult] = {}
    commands: List[DroneCommand] = []
    escalations: List[Escalation] = []

    print(
        f"[LIVE] Agent listening on ZeroMQ ports Frame={config.ZMQ_FRAME_PORT}, "
        f"State={config.ZMQ_STATE_PORT}, Cmd={config.ZMQ_CMD_PORT}"
    )
    sim_time = 0.0
    step_idx = 0
    tick_period = 1.0 / config.COORDINATOR_HZ
    last_coord_wall = 0.0

    try:
        while args.steps <= 0 or step_idx < args.steps:
            new_states = zmq_agent.poll_states()
            if new_states:
                drone_states.update(new_states)
                sim_time = max(s.timestamp for s in new_states.values())

            frames = zmq_agent.poll_frames(max_frames=n_drones)
            for frame_msg in frames:
                sim_time = max(sim_time, frame_msg.timestamp)
                perc_res = perception.process_frame(frame_msg)
                latest_perception[frame_msg.drone_id] = perc_res
                density_map.update(perc_res)
                coordinator.record_perception(
                    drone_id=perc_res.drone_id,
                    timestamp=perc_res.timestamp,
                    confidence=perc_res.confidence,
                )

            now = time.perf_counter()
            if (now - last_coord_wall) >= tick_period:
                last_coord_wall = now
                commands, escalations = coordinator.step(
                    current_time=sim_time,
                    density_map=density_map,
                    drone_states=drone_states,
                )
                zmq_agent.send_commands(commands)
                logger.log_tick(sim_time, commands, density_map.hotspots, escalations)
                if strategist is not None:
                    strategist.update_context(sim_time, density_map.hotspots, drone_states)

                dashboard.render_frame(
                    current_time=sim_time,
                    density_map=density_map,
                    drone_states=drone_states,
                    commands=commands,
                    perception_results=latest_perception,
                    decoded_rgb_frames=perception.last_decoded_rgb,
                    escalations=escalations,
                )
                step_idx += 1
            else:
                time.sleep(0.01)
    except KeyboardInterrupt:
        print("[LIVE] Interrupted by user.")
    finally:
        if strategist is not None:
            strategist.stop()
        zmq_agent.close()
        logger.close()
        dashboard.close()


def main(argv: List[str] | None = None) -> None:
    args = parse_args(argv)
    if args.live:
        run_live_agent(args)
    else:
        run_mock_simulation(args)


if __name__ == "__main__":
    main()

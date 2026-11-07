"""Edge-device latency benchmark and perception accuracy evaluator.

Measures and prints:
1. Detection latency (ms/frame) at the configured camera resolution (320x240)
2. Shared density-map update latency (ms/update)
3. Fleet coordinator assignment + safety layer latency (ms/tick for 25 drones)
4. End-to-end latency from `FrameMessage` timestamp to `DroneCommand` output (ms)
5. Optional strategist latency (ms/call) with local Ollama fallback
6. Perception ground-projection accuracy (MAE in meters vs simulator ground truth)

Run:
    `python deliverables/benchmark_edge.py`

Inputs:
    Synthetic 25-drone frames and crowd ground truth from `MockCrowdSimulator`.
Outputs:
    Formatted latency budget table for the 1-page Technical Summary deliverable.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import List

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import config
from brain.coordinator import FleetCoordinator
from brain.density_map import SharedDensityMap
from brain.strategist import LocalStrategist
from integration.mock_sim import MockCrowdSimulator
from interfaces.contract import DroneState
from perception.detector import CrowdPerceptionPipeline


def run_benchmark(n_drones: int = 25, n_iters: int = 40) -> dict[str, float]:
    sim = MockCrowdSimulator(n_drones=n_drones, n_people=config.N_PEOPLE)
    sim.world.trigger_surge()
    for _ in range(30):
        sim.step(dt=0.1, capture_all_cameras=True)

    frames, state_list = sim.step(dt=0.1, capture_all_cameras=True)
    drone_states: dict[int, DroneState] = {s.drone_id: s for s in state_list}

    pipeline = CrowdPerceptionPipeline(use_yolo=False)
    density_map = SharedDensityMap()
    coordinator = FleetCoordinator(n_drones=n_drones)

    det_times_ms: List[float] = []
    dmap_times_ms: List[float] = []
    coord_times_ms: List[float] = []
    e2e_times_ms: List[float] = []

    for i in range(n_iters):
        frame_msg = frames[i % len(frames)]
        t_start = time.perf_counter()

        # 1. Perception (JPEG decode + detection + ray-plane projection)
        t0 = time.perf_counter()
        perc_res = pipeline.process_frame(frame_msg)
        t1 = time.perf_counter()
        det_times_ms.append((t1 - t0) * 1000.0)

        # 2. Density map fusion
        t2 = time.perf_counter()
        density_map.update(perc_res)
        t3 = time.perf_counter()
        dmap_times_ms.append((t3 - t2) * 1000.0)

        coordinator.record_perception(
            perc_res.drone_id, perc_res.timestamp, perc_res.confidence
        )

        # 3. Coordinator + state machine + safety layer
        t4 = time.perf_counter()
        _cmds, _escs = coordinator.step(
            current_time=sim.sim_time + i * 0.05,
            density_map=density_map,
            drone_states=drone_states,
        )
        t5 = time.perf_counter()
        coord_times_ms.append((t5 - t4) * 1000.0)
        e2e_times_ms.append((t5 - t_start) * 1000.0)

    # 4. Measure perception projection error against ground-truth positions (allowed here in evaluation)
    gt_pts = np.asarray(sim.get_ground_truth_positions(), dtype=np.float64)
    proj_errors: List[float] = []
    for f in frames:
        res = pipeline.process_frame(f)
        for gx, gy in res.ground_points_xy:
            dists = np.hypot(gt_pts[:, 0] - gx, gt_pts[:, 1] - gy)
            proj_errors.append(float(np.min(dists)))
    mean_proj_err_m = float(np.mean(proj_errors)) if proj_errors else 0.0

    # 5. Optional Strategist call timing
    strategist = LocalStrategist(timeout_seconds=0.5)
    summary = strategist.build_compact_summary(sim.sim_time, density_map.hotspots, drone_states)
    strategist.query_once(summary, coordinator)
    strat_ms = strategist.last_call_ms

    metrics = {
        "detection_ms_mean": float(np.mean(det_times_ms)),
        "detection_ms_p95": float(np.percentile(det_times_ms, 95)),
        "density_map_ms_mean": float(np.mean(dmap_times_ms)),
        "coordinator_ms_mean": float(np.mean(coord_times_ms)),
        "e2e_ms_mean": float(np.mean(e2e_times_ms)),
        "e2e_ms_p95": float(np.percentile(e2e_times_ms, 95)),
        "strategist_ms": float(strat_ms),
        "projection_mae_m": mean_proj_err_m,
    }

    print("=" * 68)
    print(f"EDGE-DEVICE COMPUTE & LATENCY BENCHMARK (N={n_drones} Drones, {config.CAMERA_WIDTH}x{config.CAMERA_HEIGHT})")
    print("=" * 68)
    print(
        f"1. Perception (decode + detect + project): {metrics['detection_ms_mean']:6.2f} ms/frame "
        f"(p95: {metrics['detection_ms_p95']:6.2f} ms)"
    )
    print(
        f"2. Shared Density Map EMA + Gaussian update: {metrics['density_map_ms_mean']:6.2f} ms/update"
    )
    print(
        f"3. Fleet Coordinator + Hungarian + Safety : {metrics['coordinator_ms_mean']:6.2f} ms/tick"
    )
    print(
        f"4. End-to-End Frame -> Command Latency    : {metrics['e2e_ms_mean']:6.2f} ms "
        f"(p95: {metrics['e2e_ms_p95']:6.2f} ms)"
    )
    print(
        f"5. Strategist (Ollama local tool call)    : {metrics['strategist_ms']:6.2f} ms/call (async background)"
    )
    print(
        f"6. Camera -> Ground 3D Projection MAE     : {metrics['projection_mae_m']:6.3f} m"
    )
    print("=" * 68)
    return metrics


if __name__ == "__main__":
    run_benchmark()

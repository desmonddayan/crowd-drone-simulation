"""End-to-end integration tests for mock simulation, logger, and strategist validation."""

from __future__ import annotations

from pathlib import Path

from brain.coordinator import FleetCoordinator
from brain.strategist import LocalStrategist
from interfaces.contract import DroneState, Hotspot
from main import parse_args, run_mock_simulation


def test_mock_end_to_end_25_drones_state_progression_and_guidance(tmp_path: Path) -> None:
    args = parse_args(["--mock", "--drones", "25", "--steps", "260", "--headless"])
    summary = run_mock_simulation(args)

    observed = set(summary["observed_states"])  # type: ignore[arg-type]
    # Must show PATROL -> MONITOR -> RESPOND -> GUIDE as surge builds and guidance activates
    assert "PATROL" in observed
    assert "MONITOR" in observed
    assert "RESPOND" in observed
    assert "GUIDE" in observed

    # Peak density at Exit A must exceed RED_THRESHOLD, and fall after guidance takes effect
    assert float(summary["peak_exit_a_density"]) >= 5.0  # type: ignore[arg-type]
    assert float(summary["final_exit_a_density"]) < float(summary["peak_exit_a_density"])  # type: ignore[arg-type]
    assert int(summary["guided_switches"]) > 0  # type: ignore[arg-type]
    assert Path(str(summary["log_file"])).exists()


def test_strategist_tool_validation_and_fallback() -> None:
    coord = FleetCoordinator(n_drones=4)
    strat = LocalStrategist(ollama_url="http://127.0.0.1:59999/api/chat", timeout_seconds=0.05)

    hs = Hotspot(
        hotspot_id=3,
        level="RED",
        centroid_xy=(65.0, 30.0),
        area_m2=20.0,
        peak_density=5.6,
        mean_trend=0.3,
        severity=145.0,
        nearest_exit="Exit_A",
        least_dense_exit="Exit_B",
    )
    states = {
        0: DroneState(
            drone_id=0,
            timestamp=5.0,
            position_xyz=(60.0, 30.0, 20.0),
            velocity_xyz=(0.0, 0.0, 0.0),
        )
    }
    summary = strat.build_compact_summary(5.0, [hs], states)

    # Valid tool calls succeed
    assert strat.validate_and_apply_tool_call(
        "set_priority", {"hotspot_id": 3, "priority_multiplier": 1.8}, coord, {3}, 5.0
    )
    assert coord.priority_multipliers[3] == 1.8

    assert strat.validate_and_apply_tool_call(
        "request_guidance", {"hotspot_id": 3, "exit": "Exit_B"}, coord, {3}, 5.0
    )
    assert coord.forced_guidance_exits[3] == "Exit_B"

    assert strat.validate_and_apply_tool_call(
        "escalate", {"zone": "Exit_A_Funnel", "reason": "Sustained crush risk"}, coord, {3}, 5.0
    )
    assert len(strat.pending_escalations) == 1

    # Invalid hotspot_id or invalid exit name is rejected
    assert not strat.validate_and_apply_tool_call(
        "request_guidance", {"hotspot_id": 999, "exit": "Exit_B"}, coord, {3}, 5.0
    )
    assert not strat.validate_and_apply_tool_call(
        "request_guidance", {"hotspot_id": 3, "exit": "NonExistentExit"}, coord, {3}, 5.0
    )

    # Unreachable Ollama server falls back silently with 0 applied calls
    assert strat.query_once(summary, coord) == 0

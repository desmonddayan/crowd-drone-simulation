"""JSON Lines (JSONL) decision logger for auditability and post-run analysis.

Appends every drone command decision and hotspot context to `runs/<timestamp>.jsonl`.

Inputs:
    Simulation timestamp, list of `DroneCommand`, list of `Hotspot`, and optional `Escalation` list.
Outputs:
    Appended JSONL records in `runs/<timestamp>.jsonl`.
"""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
from typing import List, Optional

from interfaces.contract import DroneCommand, Escalation, Hotspot


class DecisionLogger:
    """Structured JSONL logger recording every autonomous decision and its perceived reason."""

    def __init__(self, output_dir: str | Path = "runs", run_name: Optional[str] = None) -> None:
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)
        if run_name is None:
            stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
            run_name = f"run_{stamp}.jsonl"
        self.file_path = out_path / run_name
        self._handle = open(self.file_path, "a", encoding="utf-8")

    def log_tick(
        self,
        sim_time: float,
        commands: List[DroneCommand],
        hotspots: List[Hotspot],
        escalations: Optional[List[Escalation]] = None,
    ) -> None:
        """Write one JSONL entry per drone command for the current coordinator tick."""
        hs_summary = [h.to_dict() for h in hotspots]
        esc_by_drone = {e.drone_id: e.to_dict() for e in (escalations or [])}
        for cmd in commands:
            record = {
                "time": round(float(sim_time), 3),
                "drone_id": cmd.drone_id,
                "state": cmd.state,
                "hotspot_inputs": hs_summary,
                "command": cmd.to_dict(),
                "reason": cmd.reason,
                "escalation": esc_by_drone.get(cmd.drone_id),
            }
            self._handle.write(json.dumps(record) + "\n")
        self._handle.flush()

    def close(self) -> None:
        if self._handle and not self._handle.closed:
            self._handle.close()

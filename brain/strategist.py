"""Optional local LLM strategic layer (Gemma 4 via Ollama with function calling).

Runs asynchronously in a background thread (or on demand for benchmarking), taking a
compact JSON summary of active hotspots and drone states and invoking strictly validated
tool calls:
    - `set_priority(hotspot_id, priority_multiplier)`
    - `request_guidance(hotspot_id, exit_name)`
    - `escalate(zone, reason)`

Applies a strict HTTP timeout and falls back silently to the mathematical FleetCoordinator
whenever Ollama is offline, times out, or returns invalid arguments.

Inputs:
    Compact JSON summary of `Hotspot` list and `DroneState` dict.
Outputs:
    Validated tool-call overrides applied to `FleetCoordinator` and optional `Escalation` objects.
"""

from __future__ import annotations

import json
import threading
import time
from typing import Any, Dict, List, Optional

import requests

import config
from brain.coordinator import FleetCoordinator
from interfaces.contract import DroneState, Escalation, Hotspot


STRATEGIST_TOOLS: List[Dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "set_priority",
            "description": "Adjust assignment priority multiplier for a tracked hotspot.",
            "parameters": {
                "type": "object",
                "properties": {
                    "hotspot_id": {"type": "integer"},
                    "priority_multiplier": {"type": "number"},
                },
                "required": ["hotspot_id", "priority_multiplier"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "request_guidance",
            "description": "Direct drones at a hotspot to guide the crowd toward a specific exit.",
            "parameters": {
                "type": "object",
                "properties": {
                    "hotspot_id": {"type": "integer"},
                    "exit": {"type": "string"},
                },
                "required": ["hotspot_id", "exit"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "escalate",
            "description": "Raise a human-operator safety escalation for a venue zone.",
            "parameters": {
                "type": "object",
                "properties": {
                    "zone": {"type": "string"},
                    "reason": {"type": "string"},
                },
                "required": ["zone", "reason"],
            },
        },
    },
]


class LocalStrategist:
    """Asynchronous local Ollama (Gemma 4) strategic advisor with validated tool calls."""

    def __init__(
        self,
        model: str = "gemma4:latest",
        ollama_url: str = "http://127.0.0.1:11434/api/chat",
        timeout_seconds: float = 1.5,
        interval_seconds: float = 4.0,
    ) -> None:
        self.model = model
        self.ollama_url = ollama_url
        self.timeout_seconds = float(timeout_seconds)
        self.interval_seconds = float(interval_seconds)
        self.last_call_ms: float = 0.0
        self.pending_escalations: List[Escalation] = []
        self._lock = threading.Lock()
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._latest_summary: Optional[Dict[str, Any]] = None
        self._coordinator: Optional[FleetCoordinator] = None

    @staticmethod
    def build_compact_summary(
        sim_time: float,
        hotspots: List[Hotspot],
        drone_states: Dict[int, DroneState],
    ) -> Dict[str, Any]:
        return {
            "sim_time": round(float(sim_time), 2),
            "exits": list(config.EXITS.keys()),
            "hotspots": [
                {
                    "id": h.hotspot_id,
                    "level": h.level,
                    "peak_density": round(h.peak_density, 2),
                    "trend": round(h.mean_trend, 2),
                    "nearest_exit": h.nearest_exit,
                    "least_dense_exit": h.least_dense_exit,
                }
                for h in hotspots
            ],
            "drone_state_counts": {
                st: sum(1 for d in drone_states.values() if d.current_state == st)
                for st in ("PATROL", "MONITOR", "RESPOND", "GUIDE", "FAILSAFE")
            },
        }

    def validate_and_apply_tool_call(
        self,
        tool_name: str,
        args: Dict[str, Any],
        coordinator: FleetCoordinator,
        valid_hotspot_ids: set[int],
        sim_time: float = 0.0,
    ) -> bool:
        """Strictly validate a tool call from the model and apply it if valid."""
        if tool_name == "set_priority":
            hid = args.get("hotspot_id")
            mult = args.get("priority_multiplier")
            if isinstance(hid, int) and hid in valid_hotspot_ids and isinstance(mult, (int, float)):
                clamped = float(max(0.5, min(3.0, float(mult))))
                with self._lock:
                    coordinator.priority_multipliers[hid] = clamped
                return True
            return False

        if tool_name == "request_guidance":
            hid = args.get("hotspot_id")
            exit_name = args.get("exit")
            if (
                isinstance(hid, int)
                and hid in valid_hotspot_ids
                and isinstance(exit_name, str)
                and exit_name in config.EXITS
            ):
                with self._lock:
                    coordinator.forced_guidance_exits[hid] = exit_name
                return True
            return False

        if tool_name == "escalate":
            zone = args.get("zone")
            reason = args.get("reason")
            if isinstance(zone, str) and zone.strip() and isinstance(reason, str) and reason.strip():
                with self._lock:
                    self.pending_escalations.append(
                        Escalation(
                            drone_id=-1,
                            timestamp=sim_time,
                            zone=zone.strip()[:32],
                            reason=f"Strategist: {reason.strip()[:80]}",
                            severity="MEDIUM",
                        )
                    )
                return True
            return False

        return False

    def query_once(
        self,
        summary: Dict[str, Any],
        coordinator: FleetCoordinator,
    ) -> int:
        """Execute a single synchronous tool-calling query to Ollama with silent fallback."""
        t0 = time.perf_counter()
        applied = 0
        valid_ids = {int(h["id"]) for h in summary.get("hotspots", [])}
        sim_time = float(summary.get("sim_time", 0.0))

        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are the strategic safety supervisor for an autonomous drone fleet "
                        "monitoring concert crowd density. Call tools only when a RED or rising "
                        "hotspot requires priority boost, guidance redirection, or escalation."
                    ),
                },
                {"role": "user", "content": json.dumps(summary)},
            ],
            "tools": STRATEGIST_TOOLS,
            "stream": False,
        }
        try:
            resp = requests.post(self.ollama_url, json=payload, timeout=self.timeout_seconds)
            if resp.status_code == 200:
                body = resp.json()
                tool_calls = body.get("message", {}).get("tool_calls", []) or []
                for tc in tool_calls:
                    fn = tc.get("function", {})
                    name = str(fn.get("name", ""))
                    raw_args = fn.get("arguments", {})
                    if isinstance(raw_args, str):
                        raw_args = json.loads(raw_args)
                    if isinstance(raw_args, dict):
                        if self.validate_and_apply_tool_call(
                            name, raw_args, coordinator, valid_ids, sim_time
                        ):
                            applied += 1
        except Exception:
            # Silent fallback to the mathematical FleetCoordinator
            pass
        finally:
            self.last_call_ms = (time.perf_counter() - t0) * 1000.0
        return applied

    def start_background(self, coordinator: FleetCoordinator) -> None:
        if self._running:
            return
        self._coordinator = coordinator
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def update_context(
        self,
        sim_time: float,
        hotspots: List[Hotspot],
        drone_states: Dict[int, DroneState],
    ) -> None:
        with self._lock:
            self._latest_summary = self.build_compact_summary(sim_time, hotspots, drone_states)

    def _loop(self) -> None:
        while self._running:
            summary = None
            with self._lock:
                if self._latest_summary is not None:
                    summary = dict(self._latest_summary)
            if summary is not None and self._coordinator is not None:
                self.query_once(summary, self._coordinator)
            time.sleep(self.interval_seconds)

    def stop(self) -> None:
        self._running = False

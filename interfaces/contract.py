"""Shared data schemas and serialization contract for inter-process communication.

All messages exchanged between the simulator (Isaac Sim or Mock Sim), the perception
pipeline, and the autonomous drone agent fleet use these dataclasses.

Inputs:
    Raw field values or dictionaries deserialized from JSON / ZeroMQ frames.
Outputs:
    Strongly typed dataclass instances and JSON-serializable dictionaries via
    `to_dict()` / `from_dict()`.
"""

from __future__ import annotations

import base64
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

CONTRACT_VERSION: str = "1.0.0"


@dataclass
class DroneState:
    """Kinematic and operational telemetry reported by the simulator for one drone."""

    drone_id: int
    timestamp: float
    position_xyz: Tuple[float, float, float]
    velocity_xyz: Tuple[float, float, float]
    battery: float = 1.0
    current_state: str = "PATROL"
    light_colour: Tuple[float, float, float] = (0.1, 0.9, 0.2)
    light_intensity: float = 0.35
    light_pulse_hz: float = 0.0
    light_aim_xy: Tuple[float, float] = (0.0, 0.0)
    contract_version: str = CONTRACT_VERSION

    def to_dict(self) -> Dict[str, Any]:
        return {
            "drone_id": int(self.drone_id),
            "timestamp": float(self.timestamp),
            "position_xyz": [float(v) for v in self.position_xyz],
            "velocity_xyz": [float(v) for v in self.velocity_xyz],
            "battery": float(self.battery),
            "current_state": str(self.current_state),
            "light_colour": [float(v) for v in self.light_colour],
            "light_intensity": float(self.light_intensity),
            "light_pulse_hz": float(self.light_pulse_hz),
            "light_aim_xy": [float(v) for v in self.light_aim_xy],
            "contract_version": str(self.contract_version),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DroneState":
        pos = data["position_xyz"]
        vel = data["velocity_xyz"]
        col = data.get("light_colour", (0.1, 0.9, 0.2))
        aim = data.get("light_aim_xy", (pos[0], pos[1]))
        return cls(
            drone_id=int(data["drone_id"]),
            timestamp=float(data["timestamp"]),
            position_xyz=(float(pos[0]), float(pos[1]), float(pos[2])),
            velocity_xyz=(float(vel[0]), float(vel[1]), float(vel[2])),
            battery=float(data.get("battery", 1.0)),
            current_state=str(data.get("current_state", "PATROL")),
            light_colour=(float(col[0]), float(col[1]), float(col[2])),
            light_intensity=float(data.get("light_intensity", 0.35)),
            light_pulse_hz=float(data.get("light_pulse_hz", 0.0)),
            light_aim_xy=(float(aim[0]), float(aim[1])),
            contract_version=str(data.get("contract_version", CONTRACT_VERSION)),
        )


@dataclass
class FrameMessage:
    """Downward camera frame and pose published by the simulator for one drone."""

    drone_id: int
    timestamp: float
    jpeg_bytes: bytes
    camera_position_xyz: Tuple[float, float, float]
    camera_orientation_wxyz: Tuple[float, float, float, float]
    intrinsics_3x3: List[List[float]]
    width: int = 320
    height: int = 240
    contract_version: str = CONTRACT_VERSION

    def to_dict(self) -> Dict[str, Any]:
        return {
            "drone_id": int(self.drone_id),
            "timestamp": float(self.timestamp),
            "jpeg_b64": base64.b64encode(self.jpeg_bytes).decode("ascii"),
            "camera_position_xyz": [float(v) for v in self.camera_position_xyz],
            "camera_orientation_wxyz": [float(v) for v in self.camera_orientation_wxyz],
            "intrinsics_3x3": [[float(x) for x in row] for row in self.intrinsics_3x3],
            "width": int(self.width),
            "height": int(self.height),
            "contract_version": str(self.contract_version),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "FrameMessage":
        pos = data["camera_position_xyz"]
        quat = data["camera_orientation_wxyz"]
        raw_jpeg = data.get("jpeg_bytes")
        if raw_jpeg is None:
            raw_jpeg = base64.b64decode(data.get("jpeg_b64", ""))
        return cls(
            drone_id=int(data["drone_id"]),
            timestamp=float(data["timestamp"]),
            jpeg_bytes=bytes(raw_jpeg),
            camera_position_xyz=(float(pos[0]), float(pos[1]), float(pos[2])),
            camera_orientation_wxyz=(
                float(quat[0]),
                float(quat[1]),
                float(quat[2]),
                float(quat[3]),
            ),
            intrinsics_3x3=[[float(x) for x in row] for row in data["intrinsics_3x3"]],
            width=int(data.get("width", 320)),
            height=int(data.get("height", 240)),
            contract_version=str(data.get("contract_version", CONTRACT_VERSION)),
        )


@dataclass
class Detection:
    """2D person bounding box in image pixel coordinates."""

    bbox_xyxy: Tuple[float, float, float, float]
    score: float
    label: str = "person"

    @property
    def bottom_center_uv(self) -> Tuple[float, float]:
        """Bottom-centre pixel (u, v) used for ray-plane ground projection."""
        x1, y1, x2, y2 = self.bbox_xyxy
        return ((x1 + x2) * 0.5, y2)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "bbox_xyxy": [float(v) for v in self.bbox_xyxy],
            "score": float(self.score),
            "label": str(self.label),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Detection":
        b = data["bbox_xyxy"]
        return cls(
            bbox_xyxy=(float(b[0]), float(b[1]), float(b[2]), float(b[3])),
            score=float(data["score"]),
            label=str(data.get("label", "person")),
        )


@dataclass
class PerceptionResult:
    """Output of the perception module for one camera frame."""

    drone_id: int
    timestamp: float
    detections: List[Detection] = field(default_factory=list)
    ground_points_xy: List[Tuple[float, float]] = field(default_factory=list)
    footprint_polygon_xy: List[Tuple[float, float]] = field(default_factory=list)
    confidence: float = 1.0
    contract_version: str = CONTRACT_VERSION

    def to_dict(self) -> Dict[str, Any]:
        return {
            "drone_id": int(self.drone_id),
            "timestamp": float(self.timestamp),
            "detections": [d.to_dict() for d in self.detections],
            "ground_points_xy": [[float(x), float(y)] for x, y in self.ground_points_xy],
            "footprint_polygon_xy": [[float(x), float(y)] for x, y in self.footprint_polygon_xy],
            "confidence": float(self.confidence),
            "contract_version": str(self.contract_version),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PerceptionResult":
        return cls(
            drone_id=int(data["drone_id"]),
            timestamp=float(data["timestamp"]),
            detections=[Detection.from_dict(d) for d in data.get("detections", [])],
            ground_points_xy=[(float(p[0]), float(p[1])) for p in data.get("ground_points_xy", [])],
            footprint_polygon_xy=[
                (float(p[0]), float(p[1])) for p in data.get("footprint_polygon_xy", [])
            ],
            confidence=float(data.get("confidence", 1.0)),
            contract_version=str(data.get("contract_version", CONTRACT_VERSION)),
        )


@dataclass
class Hotspot:
    """Extracted congestion cluster from the shared density map."""

    hotspot_id: int
    level: str  # "AMBER" or "RED"
    centroid_xy: Tuple[float, float]
    area_m2: float
    peak_density: float
    mean_trend: float
    severity: float
    nearest_exit: str
    least_dense_exit: str
    bbox_grid: Tuple[int, int, int, int] = (0, 0, 0, 0)  # xmin, ymin, xmax, ymax in grid indices

    def to_dict(self) -> Dict[str, Any]:
        return {
            "hotspot_id": int(self.hotspot_id),
            "level": str(self.level),
            "centroid_xy": [float(self.centroid_xy[0]), float(self.centroid_xy[1])],
            "area_m2": float(self.area_m2),
            "peak_density": float(self.peak_density),
            "mean_trend": float(self.mean_trend),
            "severity": float(self.severity),
            "nearest_exit": str(self.nearest_exit),
            "least_dense_exit": str(self.least_dense_exit),
            "bbox_grid": [int(v) for v in self.bbox_grid],
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Hotspot":
        c = data["centroid_xy"]
        bb = data.get("bbox_grid", [0, 0, 0, 0])
        return cls(
            hotspot_id=int(data["hotspot_id"]),
            level=str(data["level"]),
            centroid_xy=(float(c[0]), float(c[1])),
            area_m2=float(data["area_m2"]),
            peak_density=float(data["peak_density"]),
            mean_trend=float(data["mean_trend"]),
            severity=float(data["severity"]),
            nearest_exit=str(data["nearest_exit"]),
            least_dense_exit=str(data["least_dense_exit"]),
            bbox_grid=(int(bb[0]), int(bb[1]), int(bb[2]), int(bb[3])),
        )


@dataclass
class DroneCommand:
    """Control command sent from the autonomous fleet agent to the simulator."""

    drone_id: int
    state: str  # PATROL, MONITOR, RESPOND, GUIDE, FAILSAFE
    target_xyz: Tuple[float, float, float]
    light_colour: Tuple[float, float, float]
    light_intensity: float
    light_pulse_hz: float
    light_aim_xy: Tuple[float, float]
    guide_to_exit: Optional[str]
    reason: str
    timestamp: float = 0.0
    contract_version: str = CONTRACT_VERSION

    def to_dict(self) -> Dict[str, Any]:
        return {
            "drone_id": int(self.drone_id),
            "state": str(self.state),
            "target_xyz": [float(v) for v in self.target_xyz],
            "light_colour": [float(v) for v in self.light_colour],
            "light_intensity": float(self.light_intensity),
            "light_pulse_hz": float(self.light_pulse_hz),
            "light_aim_xy": [float(v) for v in self.light_aim_xy],
            "guide_to_exit": self.guide_to_exit,
            "reason": str(self.reason),
            "timestamp": float(self.timestamp),
            "contract_version": str(self.contract_version),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DroneCommand":
        t = data["target_xyz"]
        col = data["light_colour"]
        aim = data["light_aim_xy"]
        return cls(
            drone_id=int(data["drone_id"]),
            state=str(data["state"]),
            target_xyz=(float(t[0]), float(t[1]), float(t[2])),
            light_colour=(float(col[0]), float(col[1]), float(col[2])),
            light_intensity=float(data["light_intensity"]),
            light_pulse_hz=float(data["light_pulse_hz"]),
            light_aim_xy=(float(aim[0]), float(aim[1])),
            guide_to_exit=data.get("guide_to_exit"),
            reason=str(data["reason"]),
            timestamp=float(data.get("timestamp", 0.0)),
            contract_version=str(data.get("contract_version", CONTRACT_VERSION)),
        )


@dataclass
class Escalation:
    """Alert emitted when a drone enters FAILSAFE or an unmitigated hazard persists."""

    drone_id: int
    timestamp: float
    zone: str
    reason: str
    severity: str = "HIGH"
    contract_version: str = CONTRACT_VERSION

    def to_dict(self) -> Dict[str, Any]:
        return {
            "drone_id": int(self.drone_id),
            "timestamp": float(self.timestamp),
            "zone": str(self.zone),
            "reason": str(self.reason),
            "severity": str(self.severity),
            "contract_version": str(self.contract_version),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Escalation":
        return cls(
            drone_id=int(data["drone_id"]),
            timestamp=float(data["timestamp"]),
            zone=str(data["zone"]),
            reason=str(data["reason"]),
            severity=str(data.get("severity", "HIGH")),
            contract_version=str(data.get("contract_version", CONTRACT_VERSION)),
        )

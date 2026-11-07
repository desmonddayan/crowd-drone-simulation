"""Shared configuration for the autonomous drone crowd-congestion monitoring system.

Both the Isaac Sim process (bundled Python 3.11) and the agent/perception process
(standard venv) import this module so all spatial, physical, sensor, and ZeroMQ
parameters stay synchronized.

Inputs:
    Environment variable overrides (optional) or module-level defaults.
Outputs:
    Typed configuration constants and helper functions for sector layout and intrinsics.
"""

from __future__ import annotations

import math
from typing import Dict, List, Tuple

# ---------------------------------------------------------------------------
# Venue geometry (meters, world frame: X in [0, VENUE_WIDTH], Y in [0, VENUE_HEIGHT], Z up)
# ---------------------------------------------------------------------------
VENUE_WIDTH: float = 80.0
VENUE_HEIGHT: float = 60.0
GRID_CELL_SIZE: float = 1.0  # c = 1.0 m per cell

# Stage located at the West end (x near 0..10, y centered around 30)
STAGE_CENTER: Tuple[float, float] = (6.0, 30.0)
STAGE_SIZE: Tuple[float, float, float] = (8.0, 24.0, 1.5)  # length_x, width_y, height_z
STAGE_WATCH_ZONE: Tuple[float, float, float, float] = (11.0, 14.0, 32.0, 46.0)  # xmin, ymin, xmax, ymax

# Exits represented by name -> polygon vertices [(x, y), ...] and target point
EXITS: Dict[str, List[Tuple[float, float]]] = {
    "Exit_A": [
        (72.0, 24.0),
        (80.0, 24.0),
        (80.0, 36.0),
        (72.0, 36.0),
    ],
    "Exit_B": [
        (34.0, 54.0),
        (46.0, 54.0),
        (46.0, 60.0),
        (34.0, 60.0),
    ],
}

EXIT_TARGETS: Dict[str, Tuple[float, float]] = {
    "Exit_A": (78.0, 30.0),
    "Exit_B": (40.0, 58.5),
}

# Barriers defined as line segments: ((x1, y1), (x2, y2))
# Forms a front-of-stage pen and a narrowing funnel leading into Exit A
BARRIER_SEGMENTS: List[Tuple[Tuple[float, float], Tuple[float, float]]] = [
    # Front-of-stage barrier
    ((10.5, 14.0), (10.5, 46.0)),
    # Funnel walls narrowing toward Exit A (creates dense bottleneck during surge)
    ((52.0, 16.0), (74.0, 26.5)),
    ((52.0, 44.0), (74.0, 33.5)),
]

# ---------------------------------------------------------------------------
# Drone fleet & kinematics
# ---------------------------------------------------------------------------
N_DRONES: int = 4  # Default 4 for dev; up to 25 via CLI (--drones 25)
PATROL_ALTITUDE: float = 20.0
INSPECTION_ALTITUDE: float = 12.0
WIDE_COVERAGE_ALTITUDE: float = 24.0
SAFE_ALTITUDE: float = 26.0
SAFE_ZONE_XYZ: Tuple[float, float, float] = (4.0, 4.0, 26.0)

MAX_DRONE_SPEED: float = 8.0  # m/s
DRONE_ACCEL_GAIN: float = 3.5  # proportional approach gain toward target_xyz
MIN_SEP_HORIZONTAL: float = 4.0  # m
MIN_SEP_VERTICAL: float = 2.0  # m
REPULSION_GAIN: float = 6.0

# ---------------------------------------------------------------------------
# Camera & perception settings
# ---------------------------------------------------------------------------
CAMERA_WIDTH: int = 320
CAMERA_HEIGHT: int = 240
CAMERA_FOV_DEG: float = 60.0  # Horizontal field of view in degrees
CAMERA_CAPTURE_HZ: float = 10.0
MAX_CAMERAS_PER_STEP: int = 7  # Round-robin subset when running 25 drones

# Confidence thresholds
MIN_CONFIDENCE_THRESHOLD: float = 0.25
OPTIMAL_ALTITUDE_FOR_CONF: float = 14.0

# ---------------------------------------------------------------------------
# Shared density map & hotspot thresholds
# Note: Real-world crowd-safety thresholds (e.g. UK Green Guide / Fruin LoS)
# typically flag >3.5-4.0 people/m^2 as critical and >=5.0 people/m^2 as crush
# risk, and MUST be validated with crowd-safety engineers before deployment.
# ---------------------------------------------------------------------------
DENSITY_ALPHA: float = 0.35  # EMA weight for visible cells
TREND_ALPHA: float = 0.35  # EMA weight for dD/dt
GAUSSIAN_SIGMA: float = 0.75  # cells (preserves localized bottleneck peak)
STALE_SECONDS: float = 5.0  # seconds before unseen cells are flagged stale

AMBER_THRESHOLD: float = 3.5  # people / m^2 (enter AMBER)
AMBER_EXIT_THRESHOLD: float = 2.8  # hysteresis exit threshold for AMBER
RED_THRESHOLD: float = 5.0  # people / m^2 (enter RED)
RED_EXIT_THRESHOLD: float = 4.2  # hysteresis exit threshold for RED
RISING_TREND_THRESHOLD: float = 0.20  # (people/m^2)/s escalates AMBER -> RED
HOTSPOT_MATCH_MAX_DIST: float = 10.0  # m for stable ID tracking across ticks

# ---------------------------------------------------------------------------
# Agent state machine & fleet coordinator
# ---------------------------------------------------------------------------
COORDINATOR_HZ: float = 5.0
MIN_DWELL_SECONDS: float = 1.5  # hysteresis minimum time in a state
GUIDE_DELAY_SECONDS: float = 2.5  # RED & rising duration before switching to GUIDE
FRAME_TIMEOUT_SECONDS: float = 3.0  # no frame -> FAILSAFE
LOW_BATTERY_THRESHOLD: float = 0.15  # < 15% battery -> FAILSAFE

MAX_DRONES_PER_HOTSPOT: int = 3
HOTSPOT_AREA_PER_DRONE: float = 18.0  # m^2 per assigned drone
SEVERITY_WEIGHT: float = 2.0  # w in cost = dist / max_speed - w * severity

# ---------------------------------------------------------------------------
# Spotlight & visual signalling settings
# Photosensitivity safety constraint: pulse rates are strictly capped at <= 2.0 Hz
# (well below the 3 Hz photosensitive epilepsy hazard threshold).
# ---------------------------------------------------------------------------
MAX_PULSE_HZ: float = 2.0
LIGHT_CONE_HALF_ANGLE_DEG: float = 28.0

LIGHT_PRESETS: Dict[str, Dict[str, object]] = {
    "PATROL": {
        "colour": (0.1, 0.9, 0.2),  # steady soft green
        "intensity": 0.35,
        "pulse_hz": 0.0,
    },
    "MONITOR": {
        "colour": (1.0, 0.72, 0.0),  # warm amber
        "intensity": 0.75,
        "pulse_hz": 0.8,
    },
    "RESPOND": {
        "colour": (1.0, 0.12, 0.12),  # high-visibility red
        "intensity": 1.0,
        "pulse_hz": 1.6,
    },
    "GUIDE": {
        "colour": (0.0, 1.0, 0.85),  # distinct cyan-green guidance beam
        "intensity": 1.0,
        "pulse_hz": 1.2,
    },
    "FAILSAFE": {
        "colour": (0.9, 0.9, 0.95),  # steady white or off
        "intensity": 0.2,
        "pulse_hz": 0.0,
    },
}

# ---------------------------------------------------------------------------
# Crowd simulation parameters (for sim/ and mock_sim.py)
# ---------------------------------------------------------------------------
N_PEOPLE: int = 150
MAX_PERSON_SPEED: float = 1.3  # m/s
SURGE_START_SECONDS: float = 6.0  # time when surge toward Exit_A triggers
SURGE_FRACTION: float = 0.82  # fraction of crowd that joins the Exit_A surge
GUIDANCE_SWITCH_PROB_PER_SEC: float = 0.55  # p/s inside a GUIDE cone

# ---------------------------------------------------------------------------
# ZeroMQ endpoints
# ---------------------------------------------------------------------------
ZMQ_HOST: str = "127.0.0.1"
ZMQ_FRAME_PORT: int = 5555  # PUB/SUB: FrameMessage
ZMQ_STATE_PORT: int = 5556  # PUB/SUB: DroneState
ZMQ_CMD_PORT: int = 5557  # PUSH/PULL or PUB/SUB: DroneCommand
ZMQ_GT_PORT: int = 5558  # PUB/SUB: GroundTruth (evaluation/tests ONLY, never subscribed by agent)


def get_camera_intrinsics(
    width: int = CAMERA_WIDTH,
    height: int = CAMERA_HEIGHT,
    fov_deg: float = CAMERA_FOV_DEG,
) -> List[List[float]]:
    """Compute a 3x3 pinhole camera intrinsics matrix K for the given resolution and horizontal FOV."""
    fx = (width / 2.0) / math.tan(math.radians(fov_deg) / 2.0)
    fy = fx
    cx = width / 2.0
    cy = height / 2.0
    return [
        [fx, 0.0, cx],
        [0.0, fy, cy],
        [0.0, 0.0, 1.0],
    ]


def compute_sector_centers(
    n_drones: int = N_DRONES,
    venue_width: float = VENUE_WIDTH,
    venue_height: float = VENUE_HEIGHT,
    altitude: float = PATROL_ALTITUDE,
) -> List[Tuple[float, float, float]]:
    """Divide the venue into a near-square grid of sectors and return (x, y, z) centers."""
    if n_drones <= 0:
        return []
    cols = max(1, int(math.ceil(math.sqrt(n_drones * (venue_width / venue_height)))))
    rows = max(1, int(math.ceil(n_drones / cols)))
    # Keep a neat grid (e.g. 2x2 for 4, 5x5 for 25)
    sq = int(round(math.sqrt(n_drones)))
    if sq * sq == n_drones:
        cols, rows = sq, sq

    centers: List[Tuple[float, float, float]] = []
    dx = venue_width / cols
    dy = venue_height / rows
    for idx in range(n_drones):
        r = idx // cols
        c = idx % cols
        x = (c + 0.5) * dx
        y = (r + 0.5) * dy
        centers.append((round(x, 3), round(y, 3), float(altitude)))
    return centers

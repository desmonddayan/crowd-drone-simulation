"""Unit tests for config.py and interfaces/contract.py serialization."""

from __future__ import annotations

import json

import config
from interfaces.contract import (
    CONTRACT_VERSION,
    Detection,
    DroneCommand,
    DroneState,
    Escalation,
    FrameMessage,
    Hotspot,
    PerceptionResult,
)


def test_config_geometry_and_sectors() -> None:
    centers_4 = config.compute_sector_centers(4)
    assert len(centers_4) == 4
    for x, y, z in centers_4:
        assert 0.0 < x < config.VENUE_WIDTH
        assert 0.0 < y < config.VENUE_HEIGHT
        assert z == config.PATROL_ALTITUDE

    centers_25 = config.compute_sector_centers(25)
    assert len(centers_25) == 25

    k = config.get_camera_intrinsics()
    assert len(k) == 3 and len(k[0]) == 3
    assert k[0][2] == config.CAMERA_WIDTH / 2.0
    assert k[1][2] == config.CAMERA_HEIGHT / 2.0


def test_drone_state_round_trip() -> None:
    state = DroneState(
        drone_id=3,
        timestamp=12.5,
        position_xyz=(20.0, 15.0, 20.0),
        velocity_xyz=(1.0, -0.5, 0.0),
        battery=0.88,
        current_state="MONITOR",
        light_colour=(1.0, 0.7, 0.0),
        light_intensity=0.75,
        light_pulse_hz=0.8,
        light_aim_xy=(22.0, 16.0),
    )
    payload = json.dumps(state.to_dict())
    restored = DroneState.from_dict(json.loads(payload))
    assert restored == state
    assert restored.contract_version == CONTRACT_VERSION


def test_frame_message_round_trip() -> None:
    msg = FrameMessage(
        drone_id=1,
        timestamp=4.2,
        jpeg_bytes=b"\xff\xd8\xff\xe0mockjpegbytes\xff\xd9",
        camera_position_xyz=(40.0, 30.0, 20.0),
        camera_orientation_wxyz=(1.0, 0.0, 0.0, 0.0),
        intrinsics_3x3=config.get_camera_intrinsics(),
        width=320,
        height=240,
    )
    payload = json.dumps(msg.to_dict())
    restored = FrameMessage.from_dict(json.loads(payload))
    assert restored == msg
    assert restored.jpeg_bytes == msg.jpeg_bytes


def test_perception_result_and_detection_round_trip() -> None:
    det = Detection(bbox_xyxy=(10.0, 20.0, 30.0, 50.0), score=0.91)
    assert det.bottom_center_uv == (20.0, 50.0)

    res = PerceptionResult(
        drone_id=2,
        timestamp=5.0,
        detections=[det],
        ground_points_xy=[(41.2, 29.8), (42.0, 31.0)],
        footprint_polygon_xy=[(30.0, 22.0), (50.0, 22.0), (50.0, 38.0), (30.0, 38.0)],
        confidence=0.84,
    )
    payload = json.dumps(res.to_dict())
    restored = PerceptionResult.from_dict(json.loads(payload))
    assert restored == res


def test_hotspot_command_escalation_round_trip() -> None:
    hs = Hotspot(
        hotspot_id=7,
        level="RED",
        centroid_xy=(65.0, 30.0),
        area_m2=24.0,
        peak_density=5.8,
        mean_trend=0.35,
        severity=24.0 * 5.8 * 1.35,
        nearest_exit="Exit_A",
        least_dense_exit="Exit_B",
        bbox_grid=(60, 26, 68, 34),
    )
    assert Hotspot.from_dict(json.loads(json.dumps(hs.to_dict()))) == hs

    cmd = DroneCommand(
        drone_id=0,
        state="GUIDE",
        target_xyz=(62.0, 32.0, 12.0),
        light_colour=(0.0, 1.0, 0.85),
        light_intensity=1.0,
        light_pulse_hz=1.2,
        light_aim_xy=(40.0, 58.5),
        guide_to_exit="Exit_B",
        reason="Exit_A funnel 5.8 p/m² rising +0.35/s -> GUIDE to Exit_B",
        timestamp=9.1,
    )
    assert DroneCommand.from_dict(json.loads(json.dumps(cmd.to_dict()))) == cmd

    esc = Escalation(
        drone_id=4,
        timestamp=11.0,
        zone="Sector_4",
        reason="Frame timeout > 3.0s -> FAILSAFE",
        severity="HIGH",
    )
    assert Escalation.from_dict(json.loads(json.dumps(esc.to_dict()))) == esc

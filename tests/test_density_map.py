"""Unit tests for brain/density_map.py: density calculation, staleness, and hotspot tracking."""

from __future__ import annotations

import numpy as np

import config
from brain.density_map import CELL_AMBER, CELL_GREEN, CELL_RED, SharedDensityMap
from interfaces.contract import PerceptionResult


def test_density_map_synthetic_points_and_unseen_staleness() -> None:
    dmap = SharedDensityMap(
        venue_width=20.0,
        venue_height=20.0,
        cell_size=1.0,
        alpha=1.0,  # Immediate update to verify exact people/m^2
        sigma=0.0,  # No spatial blur so single-cell density is exact
        stale_seconds=5.0,
    )

    # Footprint covers x in [2, 10], y in [2, 10]
    footprint = [(2.0, 2.0), (10.0, 2.0), (10.0, 10.0), (2.0, 10.0)]
    # Place 5 people in cell (gx=5, gy=5) -> x=5.5, y=5.5
    pts = [(5.5, 5.5)] * 5

    res = PerceptionResult(
        drone_id=0,
        timestamp=1.0,
        ground_points_xy=pts,
        footprint_polygon_xy=footprint,
        confidence=0.9,
    )
    dmap.update(res)

    assert np.isclose(dmap.density[5, 5], 5.0)
    # Visible empty cell inside footprint updated to 0 and marked seen at t=1.0
    assert dmap.last_seen[4, 4] == 1.0
    assert dmap.density[4, 4] == 0.0

    # Unseen cell outside footprint stays untouched and stale
    assert not np.isfinite(dmap.last_seen[15, 15])
    stale_at_2s = dmap.get_stale_mask(current_time=2.0)
    assert not stale_at_2s[5, 5]
    assert stale_at_2s[15, 15]

    # After STALE_SECONDS (t=7.0 > 1.0 + 5.0), cell (5, 5) also becomes stale
    stale_at_7s = dmap.get_stale_mask(current_time=7.0)
    assert stale_at_7s[5, 5]


def test_unseen_cells_preserve_previous_density() -> None:
    dmap = SharedDensityMap(
        venue_width=20.0,
        venue_height=20.0,
        cell_size=1.0,
        alpha=0.5,
        sigma=0.0,
    )
    # Drone 0 observes region A around (4.5, 4.5)
    fp_a = [(1.0, 1.0), (8.0, 1.0), (8.0, 8.0), (1.0, 8.0)]
    dmap.update(
        PerceptionResult(
            drone_id=0,
            timestamp=1.0,
            ground_points_xy=[(4.5, 4.5)] * 6,
            footprint_polygon_xy=fp_a,
            confidence=0.95,
        )
    )
    val_a = dmap.density[4, 4]
    assert np.isclose(val_a, 3.0)

    # Drone 1 observes disjoint region B around (14.5, 14.5); region A must not be overwritten
    fp_b = [(12.0, 12.0), (18.0, 12.0), (18.0, 18.0), (12.0, 18.0)]
    dmap.update(
        PerceptionResult(
            drone_id=1,
            timestamp=2.0,
            ground_points_xy=[(14.5, 14.5)] * 4,
            footprint_polygon_xy=fp_b,
            confidence=0.95,
        )
    )
    assert np.isclose(dmap.density[4, 4], val_a)
    assert np.isclose(dmap.density[14, 14], 2.0)


def test_hotspot_extraction_and_stable_id_tracking() -> None:
    dmap = SharedDensityMap(
        venue_width=config.VENUE_WIDTH,
        venue_height=config.VENUE_HEIGHT,
        cell_size=1.0,
        alpha=0.5,
        sigma=0.8,
    )
    full_fp = [
        (0.0, 0.0),
        (config.VENUE_WIDTH, 0.0),
        (config.VENUE_WIDTH, config.VENUE_HEIGHT),
        (0.0, config.VENUE_HEIGHT),
    ]

    # Create a dense cluster near Exit_A funnel (x=66, y=30)
    cluster_pts = []
    for dx in (-0.4, 0.0, 0.4):
        for dy in (-0.4, 0.0, 0.4):
            cluster_pts.extend([(66.0 + dx, 30.0 + dy)] * 8)

    for step, t in enumerate([1.0, 1.5, 2.0, 2.5]):
        # Shift cluster slightly by +0.3m per step to test ID stability
        shifted = [(x + 0.3 * step, y) for x, y in cluster_pts]
        dmap.update(
            PerceptionResult(
                drone_id=0,
                timestamp=t,
                ground_points_xy=shifted,
                footprint_polygon_xy=full_fp,
                confidence=0.9,
            )
        )
        hotspots = dmap.extract_hotspots(current_time=t)

    red_hotspots = [h for h in hotspots if h.level == "RED"]
    assert len(red_hotspots) >= 1
    first_id = red_hotspots[0].hotspot_id

    # Move slightly again and confirm ID stays stable
    shifted_next = [(x + 1.2, y) for x, y in cluster_pts]
    dmap.update(
        PerceptionResult(
            drone_id=0,
            timestamp=3.0,
            ground_points_xy=shifted_next,
            footprint_polygon_xy=full_fp,
            confidence=0.9,
        )
    )
    hotspots_next = dmap.extract_hotspots(current_time=3.0)
    red_next = [h for h in hotspots_next if h.level == "RED"]
    assert len(red_next) >= 1
    assert red_next[0].hotspot_id == first_id
    assert red_next[0].nearest_exit == "Exit_A"
    # Since congestion is at Exit_A, least_dense_exit must be Exit_B
    assert red_next[0].least_dense_exit == "Exit_B"


def test_rising_trend_escalates_amber_to_red() -> None:
    dmap = SharedDensityMap(
        venue_width=20.0,
        venue_height=20.0,
        cell_size=1.0,
        alpha=1.0,
        trend_alpha=1.0,
        sigma=0.0,
    )
    fp = [(0.0, 0.0), (20.0, 0.0), (20.0, 20.0), (0.0, 20.0)]

    # t=1.0: 3.0 people/m^2 (GREEN)
    dmap.update(
        PerceptionResult(
            drone_id=0,
            timestamp=1.0,
            ground_points_xy=[(10.5, 10.5)] * 3,
            footprint_polygon_xy=fp,
        )
    )
    assert dmap.classify_cells(1.0)[10, 10] == CELL_GREEN

    # t=2.0: 4.0 people/m^2 (AMBER density, but trend = +1.0/s > RISING_TREND_THRESHOLD -> escalated to RED)
    dmap.update(
        PerceptionResult(
            drone_id=0,
            timestamp=2.0,
            ground_points_xy=[(10.5, 10.5)] * 4,
            footprint_polygon_xy=fp,
        )
    )
    assert dmap.classify_cells(2.0)[10, 10] == CELL_RED

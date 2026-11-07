"""Shared crowd density map and congestion hotspot extractor.

Maintains a 2D world grid of crowd density D (people/m^2), density rate-of-change
dD/dt, last observation timestamp, and observation confidence. Updates ONLY cells
within each drone camera's ground footprint polygon and flags unobserved cells as stale.

Note on thresholds:
    Default AMBER (3.5 people/m^2) and RED (5.0 people/m^2) thresholds are informed by
    crowd-safety guidance (e.g. UK Green Guide / Fruin Level of Service E-F) and MUST be
    validated by qualified crowd-safety engineers for any real-world venue.

Inputs:
    PerceptionResult instances containing projected ground points and camera footprint.
Outputs:
    Updated density/trend/staleness grids, classified cell levels (GREEN/AMBER/RED),
    and tracked Hotspot instances with stable IDs.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from scipy.ndimage import gaussian_filter, label

import config
from interfaces.contract import Hotspot, PerceptionResult

CELL_GREEN: int = 0
CELL_AMBER: int = 1
CELL_RED: int = 2


class SharedDensityMap:
    """World-grid crowd density estimator and hotspot tracker."""

    def __init__(
        self,
        venue_width: float = config.VENUE_WIDTH,
        venue_height: float = config.VENUE_HEIGHT,
        cell_size: float = config.GRID_CELL_SIZE,
        alpha: float = config.DENSITY_ALPHA,
        trend_alpha: float = config.TREND_ALPHA,
        sigma: float = config.GAUSSIAN_SIGMA,
        stale_seconds: float = config.STALE_SECONDS,
        amber_threshold: float = config.AMBER_THRESHOLD,
        red_threshold: float = config.RED_THRESHOLD,
        rising_threshold: float = config.RISING_TREND_THRESHOLD,
    ) -> None:
        self.venue_width = float(venue_width)
        self.venue_height = float(venue_height)
        self.cell_size = float(cell_size)
        self.nx = int(round(self.venue_width / self.cell_size))
        self.ny = int(round(self.venue_height / self.cell_size))

        self.alpha = float(alpha)
        self.trend_alpha = float(trend_alpha)
        self.sigma = float(sigma)
        self.stale_seconds = float(stale_seconds)
        self.amber_threshold = float(amber_threshold)
        self.red_threshold = float(red_threshold)
        self.rising_threshold = float(rising_threshold)

        # Persistent grid arrays indexed as [gy, gx] where gy in [0, ny), gx in [0, nx)
        self.density = np.zeros((self.ny, self.nx), dtype=np.float64)
        self.trend = np.zeros((self.ny, self.nx), dtype=np.float64)
        self.last_seen = np.full((self.ny, self.nx), -np.inf, dtype=np.float64)
        self.confidence = np.zeros((self.ny, self.nx), dtype=np.float64)

        # Hotspot tracking state: id -> last centroid (x, y)
        self._next_hotspot_id: int = 1
        self._tracked_centroids: Dict[int, Tuple[float, float]] = {}
        self.hotspots: List[Hotspot] = []

        # Precompute exit approach masks (within 14 m of each exit polygon)
        self._exit_masks: Dict[str, np.ndarray] = self._build_exit_masks()

    def _build_exit_masks(self) -> Dict[str, np.ndarray]:
        masks: Dict[str, np.ndarray] = {}
        ys, xs = np.mgrid[0 : self.ny, 0 : self.nx]
        wx = (xs + 0.5) * self.cell_size
        wy = (ys + 0.5) * self.cell_size
        for exit_name, poly in config.EXITS.items():
            pts = np.asarray(poly, dtype=np.float64)
            cx, cy = float(np.mean(pts[:, 0])), float(np.mean(pts[:, 1]))
            dist = np.hypot(wx - cx, wy - cy)
            masks[exit_name] = dist <= 16.0
        return masks

    def footprint_to_mask(self, footprint_polygon_xy: List[Tuple[float, float]]) -> np.ndarray:
        """Rasterize a ground footprint polygon (in meters) into a boolean (ny, nx) grid mask."""
        mask = np.zeros((self.ny, self.nx), dtype=np.uint8)
        if len(footprint_polygon_xy) < 3:
            return mask.astype(bool)
        pts_grid = np.array(
            [
                [
                    int(round(x / self.cell_size - 0.5)),
                    int(round(y / self.cell_size - 0.5)),
                ]
                for x, y in footprint_polygon_xy
            ],
            dtype=np.int32,
        )
        cv2.fillPoly(mask, [pts_grid], 1)
        return mask.astype(bool)

    def compute_frame_density(self, ground_points_xy: List[Tuple[float, float]]) -> np.ndarray:
        """Compute Gaussian-smoothed instantaneous density (people/m^2) from ground points."""
        if not ground_points_xy:
            counts = np.zeros((self.ny, self.nx), dtype=np.float64)
        else:
            pts = np.asarray(ground_points_xy, dtype=np.float64)
            xs = pts[:, 0]
            ys = pts[:, 1]
            counts, _, _ = np.histogram2d(
                ys,
                xs,
                bins=[self.ny, self.nx],
                range=[[0.0, self.venue_height], [0.0, self.venue_width]],
            )
        cell_area = self.cell_size * self.cell_size
        raw_density = counts / cell_area
        if self.sigma > 0.0:
            return gaussian_filter(raw_density, sigma=self.sigma, mode="nearest")
        return raw_density

    def update(
        self,
        result: PerceptionResult,
        visible_mask: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Fuse a single PerceptionResult into the shared density map.

        Updates ONLY visible cells using exponential moving average:
            D = (1 - alpha) * D + alpha * frame_density
        and updates dD/dt via exponential smoothing of the per-update difference / dt.
        """
        if visible_mask is None:
            vis = self.footprint_to_mask(result.footprint_polygon_xy)
        else:
            vis = np.asarray(visible_mask, dtype=bool)

        if not np.any(vis):
            return self.density

        frame_density = self.compute_frame_density(result.ground_points_xy)
        t = float(result.timestamp)

        # Separate cells already updated at this exact timestamp `t` from cells advancing in time
        same_t_mask = vis & (np.abs(self.last_seen - t) < 1e-6)
        advance_mask = vis & (~same_t_mask)

        if np.any(advance_mask):
            prev_d = self.density[advance_mask].copy()
            prev_seen = self.last_seen[advance_mask].copy()
            was_seen = np.isfinite(prev_seen)

            new_d = (1.0 - self.alpha) * prev_d + self.alpha * frame_density[advance_mask]
            self.density[advance_mask] = new_d

            if np.any(was_seen):
                dt = np.maximum(t - prev_seen[was_seen], 0.05)
                inst_trend = (new_d[was_seen] - prev_d[was_seen]) / dt
                prev_trend = self.trend[advance_mask]
                prev_trend[was_seen] = (
                    1.0 - self.trend_alpha
                ) * prev_trend[was_seen] + self.trend_alpha * inst_trend
                self.trend[advance_mask] = prev_trend

            self.last_seen[advance_mask] = t
            self.confidence[advance_mask] = float(np.clip(result.confidence, 0.0, 1.0))

        if np.any(same_t_mask):
            cand_d = (1.0 - self.alpha) * self.density[same_t_mask] + self.alpha * frame_density[
                same_t_mask
            ]
            self.density[same_t_mask] = np.maximum(self.density[same_t_mask], cand_d)
            self.confidence[same_t_mask] = np.maximum(
                self.confidence[same_t_mask], float(np.clip(result.confidence, 0.0, 1.0))
            )

        return self.density

    def get_stale_mask(self, current_time: float) -> np.ndarray:
        """Return boolean (ny, nx) mask of cells not observed within STALE_SECONDS."""
        return (~np.isfinite(self.last_seen)) | (
            (float(current_time) - self.last_seen) > self.stale_seconds
        )

    def classify_cells(self, current_time: Optional[float] = None) -> np.ndarray:
        """Classify grid cells into CELL_GREEN (0), CELL_AMBER (1), CELL_RED (2).

        Escalates AMBER cells to RED when trend > rising_threshold.
        """
        levels = np.full((self.ny, self.nx), CELL_GREEN, dtype=np.int32)
        amber_mask = (self.density >= self.amber_threshold) & (self.density < self.red_threshold)
        red_mask = (self.density >= self.red_threshold) | (
            amber_mask & (self.trend > self.rising_threshold)
        )
        if current_time is not None:
            fresh = ~self.get_stale_mask(current_time)
            amber_mask = amber_mask & fresh
            red_mask = red_mask & fresh

        levels[amber_mask] = CELL_AMBER
        levels[red_mask] = CELL_RED
        return levels

    def get_exit_densities(self) -> Dict[str, float]:
        """Compute mean crowd density in the approach zone of each configured exit."""
        out: Dict[str, float] = {}
        for exit_name, mask in self._exit_masks.items():
            if np.any(mask):
                out[exit_name] = float(np.mean(self.density[mask]))
            else:
                out[exit_name] = 0.0
        return out

    def _nearest_and_least_dense_exit(self, centroid_xy: Tuple[float, float]) -> Tuple[str, str]:
        cx, cy = centroid_xy
        nearest_name = min(
            config.EXIT_TARGETS.keys(),
            key=lambda name: math.hypot(
                cx - config.EXIT_TARGETS[name][0], cy - config.EXIT_TARGETS[name][1]
            ),
        )
        exit_densities = self.get_exit_densities()
        least_dense_name = min(
            exit_densities.keys(),
            key=lambda name: (exit_densities[name], name),
        )
        return nearest_name, least_dense_name

    def extract_hotspots(self, current_time: Optional[float] = None) -> List[Hotspot]:
        """Extract RED and AMBER connected-component hotspots and assign stable IDs."""
        levels = self.classify_cells(current_time=current_time)
        cell_area = self.cell_size * self.cell_size
        raw_hotspots: List[Dict[str, object]] = []

        # Connected-component labelling on RED cells first, then on AMBER cells
        for target_level, level_name in ((CELL_RED, "RED"), (CELL_AMBER, "AMBER")):
            mask = levels == target_level
            if not np.any(mask):
                continue
            labeled, num_features = label(mask)
            for comp_idx in range(1, num_features + 1):
                comp_mask = labeled == comp_idx
                num_cells = int(np.sum(comp_mask))
                if num_cells == 0:
                    continue
                ys, xs = np.nonzero(comp_mask)
                weights = np.maximum(self.density[comp_mask], 1e-3)
                cx = float(np.average((xs + 0.5) * self.cell_size, weights=weights))
                cy = float(np.average((ys + 0.5) * self.cell_size, weights=weights))
                area_m2 = float(num_cells * cell_area)
                peak = float(np.max(self.density[comp_mask]))
                mean_trend = float(np.mean(self.trend[comp_mask]))
                severity = float(area_m2 * peak * (1.0 + max(mean_trend, 0.0)))
                nearest_exit, least_dense_exit = self._nearest_and_least_dense_exit((cx, cy))
                bbox = (int(np.min(xs)), int(np.min(ys)), int(np.max(xs)), int(np.max(ys)))
                raw_hotspots.append(
                    {
                        "level": level_name,
                        "centroid_xy": (cx, cy),
                        "area_m2": area_m2,
                        "peak_density": peak,
                        "mean_trend": mean_trend,
                        "severity": severity,
                        "nearest_exit": nearest_exit,
                        "least_dense_exit": least_dense_exit,
                        "bbox_grid": bbox,
                    }
                )

        # Match raw hotspots to existing IDs by nearest centroid
        used_old_ids: set[int] = set()
        new_tracked: Dict[int, Tuple[float, float]] = {}
        matched_hotspots: List[Hotspot] = []

        # Sort by severity descending so dominant hotspots claim their previous IDs first
        raw_hotspots.sort(key=lambda h: float(h["severity"]), reverse=True)

        for item in raw_hotspots:
            cx, cy = item["centroid_xy"]  # type: ignore[misc]
            best_id: Optional[int] = None
            best_dist = config.HOTSPOT_MATCH_MAX_DIST
            for old_id, (ox, oy) in self._tracked_centroids.items():
                if old_id in used_old_ids:
                    continue
                d = math.hypot(cx - ox, cy - oy)
                if d <= best_dist:
                    best_dist = d
                    best_id = old_id

            if best_id is None:
                best_id = self._next_hotspot_id
                self._next_hotspot_id += 1
            else:
                used_old_ids.add(best_id)

            new_tracked[best_id] = (cx, cy)
            matched_hotspots.append(
                Hotspot(
                    hotspot_id=best_id,
                    level=str(item["level"]),
                    centroid_xy=(cx, cy),
                    area_m2=float(item["area_m2"]),
                    peak_density=float(item["peak_density"]),
                    mean_trend=float(item["mean_trend"]),
                    severity=float(item["severity"]),
                    nearest_exit=str(item["nearest_exit"]),
                    least_dense_exit=str(item["least_dense_exit"]),
                    bbox_grid=item["bbox_grid"],  # type: ignore[arg-type]
                )
            )

        self._tracked_centroids = new_tracked
        self.hotspots = matched_hotspots
        return matched_hotspots

    def find_stalest_region_center(
        self,
        current_time: float,
        reference_xy: Tuple[float, float],
    ) -> Optional[Tuple[float, float]]:
        """Return world (x, y) center of a stale region for coverage memory patrol."""
        stale = self.get_stale_mask(current_time)
        if not np.any(stale):
            return None
        # Compute how long each stale cell has gone unseen (never-seen cells get a large age)
        ages = np.where(np.isfinite(self.last_seen), current_time - self.last_seen, 1e6)
        stale_ys, stale_xs = np.nonzero(stale)
        max_age = float(np.max(ages[stale]))
        top_mask = stale & (ages >= max_age * 0.95)
        tys, txs = np.nonzero(top_mask)
        wx = (txs + 0.5) * self.cell_size
        wy = (tys + 0.5) * self.cell_size
        rx, ry = reference_xy
        dists = np.hypot(wx - rx, wy - ry)
        idx = int(np.argmin(dists))
        return (float(wx[idx]), float(wy[idx]))

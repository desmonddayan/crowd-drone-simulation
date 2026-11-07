"""Camera-based crowd perception package: person detection and ray-plane ground projection."""

from perception.detector import CrowdPerceptionPipeline, PersonDetector
from perception.projector import (
    compute_footprint_polygon,
    project_pixel_to_ground,
    world_point_to_pixel,
)

__all__ = [
    "CrowdPerceptionPipeline",
    "PersonDetector",
    "compute_footprint_polygon",
    "project_pixel_to_ground",
    "world_point_to_pixel",
]

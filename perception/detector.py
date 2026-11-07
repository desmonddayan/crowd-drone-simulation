"""Person detection and full perception pipeline for downward drone camera frames.

Supports:
1. Primary detector: Local YOLO nano model (`ultralytics`, e.g. `yolov8n.pt`) on the
   decoded RGB image when enabled and installed.
2. Fallback detector: High-speed blob / local-maxima detector in colour/brightness space
   for mock and capsule rendering modes. Automatically splits merged crowd blobs when
   local maxima or contour areas indicate multiple adjacent people.

Inputs:
    `FrameMessage` containing JPEG-encoded RGB image, camera pose, and intrinsics.
Outputs:
    `PerceptionResult` with 2D detections, projected ground coordinates (z = 0),
    ground footprint polygon, and observation confidence.
"""

from __future__ import annotations

import math
from typing import List, Optional, Tuple

import cv2
import numpy as np
from scipy.ndimage import maximum_filter

import config
from interfaces.contract import Detection, FrameMessage, PerceptionResult
from perception.projector import compute_footprint_polygon, project_pixel_to_ground


class PersonDetector:
    """Detects people in downward aerial RGB frames using YOLO (primary) or blob fallback."""

    def __init__(
        self,
        use_yolo: bool = False,
        yolo_model_name: str = "yolov8n.pt",
        score_threshold: float = 0.25,
    ) -> None:
        self.use_yolo = bool(use_yolo)
        self.score_threshold = float(score_threshold)
        self._yolo_model = None

        if self.use_yolo:
            try:
                from ultralytics import YOLO  # type: ignore[import-not-found]

                self._yolo_model = YOLO(yolo_model_name)
            except Exception:
                # Fall back cleanly to blob detector if ultralytics/weights are unavailable
                self._yolo_model = None
                self.use_yolo = False

    def detect_rgb(
        self,
        rgb_image: np.ndarray,
        camera_altitude: float = config.PATROL_ALTITUDE,
    ) -> Tuple[List[Detection], float]:
        """Run detection on an HxWx3 RGB uint8 image and return (detections, confidence)."""
        if rgb_image is None or rgb_image.size == 0:
            return [], 0.0

        if self.use_yolo and self._yolo_model is not None:
            detections = self._detect_yolo(rgb_image)
            # If YOLO finds nothing on synthetic/capsule frames, automatically run blob fallback
            if not detections:
                detections = self._detect_blobs(rgb_image, camera_altitude)
        else:
            detections = self._detect_blobs(rgb_image, camera_altitude)

        confidence = self._compute_confidence(rgb_image, detections, camera_altitude)
        return detections, confidence

    def _detect_yolo(self, rgb_image: np.ndarray) -> List[Detection]:
        results = self._yolo_model.predict(rgb_image, verbose=False, conf=self.score_threshold)
        out: List[Detection] = []
        for res in results:
            if res.boxes is None:
                continue
            boxes = res.boxes.xyxy.cpu().numpy()
            scores = res.boxes.conf.cpu().numpy()
            classes = res.boxes.cls.cpu().numpy()
            for box, sc, cls_id in zip(boxes, scores, classes):
                # COCO class 0 is "person"
                if int(cls_id) == 0 and float(sc) >= self.score_threshold:
                    out.append(
                        Detection(
                            bbox_xyxy=(
                                float(box[0]),
                                float(box[1]),
                                float(box[2]),
                                float(box[3]),
                            ),
                            score=float(sc),
                            label="person",
                        )
                    )
        return out

    def _detect_blobs(
        self,
        rgb_image: np.ndarray,
        camera_altitude: float,
    ) -> List[Detection]:
        """Detect person blobs in synthetic or capsule top-down imagery.

        Uses HSV / warm-bright person signature filtering + local peak extraction so both
        isolated pedestrians and tightly packed surge crowds are accurately resolved.
        """
        # People markers in mock_sim and capsule mode are rendered with a bright warm
        # orange-gold / high-value core: R > 170, G > 110, R > B + 45 (or very bright > 215).
        r = rgb_image[:, :, 0].astype(np.int16)
        g = rgb_image[:, :, 1].astype(np.int16)
        b = rgb_image[:, :, 2].astype(np.int16)

        warm_score = np.clip((r - b) + (g - b) // 2, 0, 255).astype(np.uint8)
        bright_mask = ((r > 165) & (g > 105) & ((r - b) > 35)) | (
            (r > 220) & (g > 220) & (b > 220)
        )

        if not np.any(bright_mask):
            return []

        # Expected person radius in pixels at current altitude
        alt = max(float(camera_altitude), 5.0)
        px_per_m = (rgb_image.shape[1] / 2.0) / (
            alt * math.tan(math.radians(config.CAMERA_FOV_DEG) / 2.0)
        )
        person_radius_px = max(2.0, 0.24 * px_per_m)
        box_half = max(2.5, person_radius_px * 1.1)

        # Smooth the masked intensity slightly and find local maxima (3x3 window resolves closely packed crowds)
        masked_signal = np.where(bright_mask, np.maximum(r, warm_score), 0).astype(np.float32)
        blurred = cv2.GaussianBlur(masked_signal, (3, 3), 0.45)
        local_max = maximum_filter(blurred, size=3)
        peaks_mask = (blurred == local_max) & (blurred >= 110.0) & bright_mask

        ys, xs = np.nonzero(peaks_mask)
        detections: List[Detection] = []
        h, w = rgb_image.shape[:2]

        for py, px in zip(ys, xs):
            val = float(blurred[py, px])
            score = float(np.clip(val / 245.0, 0.45, 0.99))
            x1 = max(0.0, float(px) - box_half)
            y1 = max(0.0, float(py) - box_half)
            x2 = min(float(w - 1), float(px) + box_half)
            y2 = min(float(h - 1), float(py) + box_half)
            detections.append(
                Detection(
                    bbox_xyxy=(x1, y1, x2, y2),
                    score=score,
                    label="person",
                )
            )
        return detections

    def _compute_confidence(
        self,
        rgb_image: np.ndarray,
        detections: List[Detection],
        camera_altitude: float,
    ) -> float:
        """Compute observation confidence penalized by extreme altitude or severe darkness."""
        mean_det_score = (
            float(np.mean([d.score for d in detections])) if detections else 0.85
        )
        # Altitude penalty when flying higher than optimal inspection/patrol range
        alt = max(float(camera_altitude), 1.0)
        alt_factor = float(np.clip(1.0 - max(0.0, alt - 20.0) * 0.025, 0.5, 1.0))

        # Check that image is not completely black / corrupted
        max_luma = float(np.max(rgb_image))
        luma_factor = 0.15 if max_luma < 8.0 else 1.0

        return float(np.clip(mean_det_score * alt_factor * luma_factor, 0.0, 1.0))


class CrowdPerceptionPipeline:
    """End-to-end perception pipeline converting `FrameMessage` -> `PerceptionResult`."""

    def __init__(
        self,
        use_yolo: bool = False,
        venue_width: float = config.VENUE_WIDTH,
        venue_height: float = config.VENUE_HEIGHT,
    ) -> None:
        self.detector = PersonDetector(use_yolo=use_yolo)
        self.venue_width = float(venue_width)
        self.venue_height = float(venue_height)
        self.last_decoded_rgb: dict[int, np.ndarray] = {}

    @staticmethod
    def decode_jpeg(jpeg_bytes: bytes) -> Optional[np.ndarray]:
        """Decode JPEG bytes into an HxWx3 RGB uint8 numpy array."""
        if not jpeg_bytes:
            return None
        arr = np.frombuffer(jpeg_bytes, dtype=np.uint8)
        bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if bgr is None:
            return None
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    def process_frame(self, frame_msg: FrameMessage) -> PerceptionResult:
        """Decode a FrameMessage, detect people, project to z=0, and compute footprint."""
        rgb = self.decode_jpeg(frame_msg.jpeg_bytes)
        if rgb is None:
            return PerceptionResult(
                drone_id=frame_msg.drone_id,
                timestamp=frame_msg.timestamp,
                detections=[],
                ground_points_xy=[],
                footprint_polygon_xy=[],
                confidence=0.0,
            )

        self.last_decoded_rgb[frame_msg.drone_id] = rgb
        altitude = float(frame_msg.camera_position_xyz[2])
        detections, confidence = self.detector.detect_rgb(rgb, camera_altitude=altitude)

        ground_points: List[Tuple[float, float]] = []
        for det in detections:
            u, v = det.bottom_center_uv
            pt = project_pixel_to_ground(
                u=u,
                v=v,
                camera_position_xyz=frame_msg.camera_position_xyz,
                camera_orientation_wxyz=frame_msg.camera_orientation_wxyz,
                intrinsics_3x3=frame_msg.intrinsics_3x3,
                venue_width=self.venue_width,
                venue_height=self.venue_height,
                clip_to_venue=False,
            )
            if pt is not None:
                ground_points.append(pt)

        footprint = compute_footprint_polygon(
            camera_position_xyz=frame_msg.camera_position_xyz,
            camera_orientation_wxyz=frame_msg.camera_orientation_wxyz,
            intrinsics_3x3=frame_msg.intrinsics_3x3,
            width=frame_msg.width,
            height=frame_msg.height,
            venue_width=self.venue_width,
            venue_height=self.venue_height,
        )

        return PerceptionResult(
            drone_id=frame_msg.drone_id,
            timestamp=frame_msg.timestamp,
            detections=detections,
            ground_points_xy=ground_points,
            footprint_polygon_xy=footprint,
            confidence=confidence,
        )

"""Rilevamento cialde caffe con YOLOv8n + fallback Hough Circles.

Pipeline:
1. YOLOv8n (TensorRT FP16 su Jetson) - detection primaria
2. Se YOLO fallisce: Hough Circles (cialde sono dischi ~44mm)
"""

from typing import List, Tuple
from dataclasses import dataclass

import numpy as np
import cv2

from rclpy.node import Node

from src.utils.config_loader import ConfigLoader


@dataclass
class PodDetection:
    """Una cialda rilevata nell'immagine."""
    center_px: Tuple[int, int]   # Centro (u, v) in pixel
    radius_px: float             # Raggio in pixel
    confidence: float            # Confidenza [0, 1]
    bbox: Tuple[int, int, int, int]  # (x1, y1, x2, y2)
    method: str                  # "yolo" o "hough"


class PodDetector:
    """Rileva cialde caffe nell'immagine RGB."""

    def __init__(self, node: Node, config: ConfigLoader):
        self._node = node
        self._config = config
        self._logger = node.get_logger()

        # YOLO
        self._yolo_model = None
        self._yolo_conf = config.get('perception.yolo.confidence_threshold', 0.6)
        self._yolo_nms = config.get('perception.yolo.nms_threshold', 0.45)
        self._yolo_size = config.get('perception.yolo.input_size', 640)
        self._use_tensorrt = config.get('perception.yolo.use_tensorrt', True)

        # Hough fallback
        self._hough_enabled = config.get('perception.hough_fallback.enabled', True)
        self._hough_min_r = config.get('perception.hough_fallback.min_radius_px', 15)
        self._hough_max_r = config.get('perception.hough_fallback.max_radius_px', 50)
        self._hough_dp = config.get('perception.hough_fallback.dp', 1.2)
        self._hough_min_dist = config.get('perception.hough_fallback.min_dist_px', 40)
        self._hough_param1 = config.get('perception.hough_fallback.param1', 50)
        self._hough_param2 = config.get('perception.hough_fallback.param2', 30)

        self._init_yolo()

    def _init_yolo(self):
        """Inizializza il modello YOLO."""
        model_path = self._config.get('perception.yolo.model_path')
        try:
            from ultralytics import YOLO
            self._yolo_model = YOLO(model_path)

            if self._use_tensorrt:
                try:
                    self._yolo_model = YOLO(model_path.replace('.pt', '.engine'))
                    self._logger.info("YOLO TensorRT engine caricato")
                except Exception:
                    self._logger.warn(
                        "TensorRT engine non trovato, uso modello PyTorch"
                    )

            self._logger.info(f"Modello YOLO caricato: {model_path}")
        except Exception as e:
            self._logger.warn(f"YOLO non disponibile: {e}. Solo Hough fallback.")
            self._yolo_model = None

    def detect(self, rgb_image: np.ndarray,
               roi: Tuple[int, int, int, int] = None) -> List[PodDetection]:
        """Rileva cialde nell'immagine.

        Args:
            rgb_image: immagine BGR
            roi: regione di interesse opzionale (x1, y1, x2, y2)

        Returns:
            Lista di PodDetection
        """
        if roi is not None:
            x1, y1, x2, y2 = roi
            crop = rgb_image[y1:y2, x1:x2]
        else:
            crop = rgb_image
            x1, y1 = 0, 0

        # Prova YOLO prima
        detections = self._detect_yolo(crop, x1, y1)

        # Fallback a Hough se YOLO non trova nulla
        if not detections and self._hough_enabled:
            self._logger.info("YOLO: nessuna cialda. Provo Hough fallback...")
            detections = self._detect_hough(crop, x1, y1)

        return detections

    def _detect_yolo(self, image: np.ndarray,
                      offset_x: int = 0, offset_y: int = 0) -> List[PodDetection]:
        """Detection con YOLOv8n."""
        if self._yolo_model is None:
            return []

        try:
            results = self._yolo_model(
                image,
                conf=self._yolo_conf,
                iou=self._yolo_nms,
                imgsz=self._yolo_size,
                verbose=False,
            )

            detections = []
            for result in results:
                for box in result.boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                    conf = float(box.conf[0])

                    cx = (x1 + x2) // 2 + offset_x
                    cy = (y1 + y2) // 2 + offset_y
                    radius = max(x2 - x1, y2 - y1) / 2

                    detections.append(PodDetection(
                        center_px=(cx, cy),
                        radius_px=radius,
                        confidence=conf,
                        bbox=(x1 + offset_x, y1 + offset_y,
                              x2 + offset_x, y2 + offset_y),
                        method="yolo",
                    ))

            return detections

        except Exception as e:
            self._logger.warn(f"Errore YOLO: {e}")
            return []

    def _detect_hough(self, image: np.ndarray,
                       offset_x: int = 0, offset_y: int = 0) -> List[PodDetection]:
        """Fallback detection con Hough Circles."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (9, 9), 2)

        circles = cv2.HoughCircles(
            gray,
            cv2.HOUGH_GRADIENT,
            dp=self._hough_dp,
            minDist=self._hough_min_dist,
            param1=self._hough_param1,
            param2=self._hough_param2,
            minRadius=self._hough_min_r,
            maxRadius=self._hough_max_r,
        )

        detections = []
        if circles is not None:
            circles = np.uint16(np.around(circles[0]))
            for cx, cy, r in circles:
                cx_abs = int(cx) + offset_x
                cy_abs = int(cy) + offset_y
                r = int(r)

                detections.append(PodDetection(
                    center_px=(cx_abs, cy_abs),
                    radius_px=float(r),
                    confidence=0.5,  # Hough non fornisce confidenza
                    bbox=(cx_abs - r, cy_abs - r, cx_abs + r, cy_abs + r),
                    method="hough",
                ))

        return detections

    def draw_detections(self, image: np.ndarray,
                         detections: List[PodDetection]) -> np.ndarray:
        """Disegna le detection sull'immagine per debug."""
        vis = image.copy()
        for det in detections:
            color = (0, 255, 0) if det.method == "yolo" else (255, 165, 0)
            cx, cy = det.center_px
            r = int(det.radius_px)

            cv2.circle(vis, (cx, cy), r, color, 2)
            cv2.circle(vis, (cx, cy), 3, color, -1)
            label = f"{det.method} {det.confidence:.2f}"
            cv2.putText(vis, label, (cx - r, cy - r - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        return vis

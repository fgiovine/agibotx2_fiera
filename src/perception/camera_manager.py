"""Gestione camera RGB-D: sottoscrizione, sincronizzazione frame."""

import threading
from typing import Optional, Tuple

import numpy as np
import cv2

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge

from src.utils.ros_helpers import sensor_qos
from src.utils.config_loader import ConfigLoader


class CameraManager:
    """Sottoscrive alla camera RGB-D e fornisce frame sincronizzati."""

    def __init__(self, node: Node, config: ConfigLoader):
        self._node = node
        self._config = config
        self._logger = node.get_logger()
        self._bridge = CvBridge()
        self._lock = threading.Lock()

        # Frame correnti
        self._rgb_image: Optional[np.ndarray] = None
        self._depth_image: Optional[np.ndarray] = None
        self._camera_info: Optional[CameraInfo] = None
        self._frame_stamp = None

        # Parametri intrinseci camera
        self._fx: float = 0.0
        self._fy: float = 0.0
        self._cx: float = 0.0
        self._cy: float = 0.0

        # Sottoscrizioni
        rgb_topic = config.get('perception.camera_topic_rgb')
        depth_topic = config.get('perception.camera_topic_depth')
        info_topic = config.get('perception.camera_info_topic')

        self._rgb_sub = node.create_subscription(
            Image, rgb_topic, self._rgb_callback, sensor_qos()
        )
        self._depth_sub = node.create_subscription(
            Image, depth_topic, self._depth_callback, sensor_qos()
        )
        self._info_sub = node.create_subscription(
            CameraInfo, info_topic, self._info_callback, sensor_qos()
        )

    def _rgb_callback(self, msg: Image):
        with self._lock:
            self._rgb_image = self._bridge.imgmsg_to_cv2(msg, "bgr8")
            self._frame_stamp = msg.header.stamp

    def _depth_callback(self, msg: Image):
        with self._lock:
            self._depth_image = self._bridge.imgmsg_to_cv2(
                msg, desired_encoding="passthrough"
            )

    def _info_callback(self, msg: CameraInfo):
        if self._camera_info is None:
            self._camera_info = msg
            self._fx = msg.k[0]
            self._fy = msg.k[4]
            self._cx = msg.k[2]
            self._cy = msg.k[5]
            self._logger.info(
                f"Camera intrinsics: fx={self._fx:.1f} fy={self._fy:.1f} "
                f"cx={self._cx:.1f} cy={self._cy:.1f}"
            )

    def get_frame(self) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """Ritorna l'ultima coppia (rgb, depth) disponibile.

        Returns:
            Tuple (rgb_bgr, depth_meters) o None se non disponibile
        """
        with self._lock:
            if self._rgb_image is None or self._depth_image is None:
                return None
            rgb = self._rgb_image.copy()
            depth = self._depth_image.copy()

        # Converti depth in metri se necessario (uint16 -> float)
        if depth.dtype == np.uint16:
            depth = depth.astype(np.float32) / 1000.0

        return rgb, depth

    def get_rgb(self) -> Optional[np.ndarray]:
        """Ritorna solo l'immagine RGB."""
        with self._lock:
            if self._rgb_image is None:
                return None
            return self._rgb_image.copy()

    def get_depth_at(self, u: int, v: int) -> float:
        """Ritorna la profondita in metri a un pixel specifico.

        Returns:
            Profondita in metri, o 0.0 se non disponibile
        """
        with self._lock:
            if self._depth_image is None:
                return 0.0
            h, w = self._depth_image.shape[:2]
            if 0 <= u < w and 0 <= v < h:
                d = self._depth_image[v, u]
                if self._depth_image.dtype == np.uint16:
                    return float(d) / 1000.0
                return float(d)
            return 0.0

    @property
    def intrinsics(self) -> Tuple[float, float, float, float]:
        """Ritorna (fx, fy, cx, cy)."""
        return self._fx, self._fy, self._cx, self._cy

    @property
    def has_data(self) -> bool:
        with self._lock:
            return self._rgb_image is not None and self._depth_image is not None

    @property
    def has_intrinsics(self) -> bool:
        return self._camera_info is not None

    def destroy(self):
        self._node.destroy_subscription(self._rgb_sub)
        self._node.destroy_subscription(self._depth_sub)
        self._node.destroy_subscription(self._info_sub)

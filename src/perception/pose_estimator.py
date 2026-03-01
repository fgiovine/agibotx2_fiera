"""Stima posizione 3D da detection 2D + profondita.

Pipeline: pixel (u,v) + depth -> punto 3D camera frame -> robot base frame
via catena cinematica testa.
"""

from typing import Optional, List, Tuple

import numpy as np

from rclpy.node import Node

from src.utils.transforms import pixel_to_3d, transform_point
from src.utils.config_loader import ConfigLoader
from src.perception.pod_detector import PodDetection


class PoseEstimator:
    """Stima posizioni 3D degli oggetti rilevati."""

    def __init__(self, node: Node, config: ConfigLoader):
        self._node = node
        self._config = config
        self._logger = node.get_logger()

        # Trasformazione camera -> base robot (da calibrazione hand-eye)
        self._T_camera_to_base: Optional[np.ndarray] = None

    def set_camera_to_base_transform(self, T: np.ndarray):
        """Imposta la trasformazione camera -> base robot.

        Questa viene dalla calibrazione hand-eye o dalla catena cinematica
        (head joints + offset camera montata sulla testa).
        """
        self._T_camera_to_base = T

    def update_transform_from_joints(self, head_yaw: float, head_pitch: float,
                                       T_base_to_head: np.ndarray,
                                       T_head_to_camera: np.ndarray):
        """Aggiorna la trasformazione basata sulle posizioni correnti della testa.

        Args:
            head_yaw: angolo yaw testa (rad)
            head_pitch: angolo pitch testa (rad)
            T_base_to_head: trasformazione base -> montaggio testa
            T_head_to_camera: trasformazione montaggio testa -> camera
        """
        # Rotazione testa
        cy, sy = np.cos(head_yaw), np.sin(head_yaw)
        cp, sp = np.cos(head_pitch), np.sin(head_pitch)

        R_yaw = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
        R_pitch = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])

        R_head = R_yaw @ R_pitch
        T_head_rot = np.eye(4)
        T_head_rot[:3, :3] = R_head

        T_base_to_camera = T_base_to_head @ T_head_rot @ T_head_to_camera
        self._T_camera_to_base = T_base_to_camera  # Gia nel frame base

    def estimate_3d_position(self, detection: PodDetection,
                              depth_image: np.ndarray,
                              fx: float, fy: float,
                              cx: float, cy: float) -> Optional[np.ndarray]:
        """Stima posizione 3D di una detection.

        Args:
            detection: PodDetection con centro pixel
            depth_image: immagine depth in metri
            fx, fy, cx, cy: parametri intrinseci camera

        Returns:
            Posizione 3D nel frame base robot [x, y, z], o None
        """
        u, v = detection.center_px
        h, w = depth_image.shape[:2]

        if not (0 <= u < w and 0 <= v < h):
            return None

        # Media depth in un'area 5x5 attorno al centro per robustezza
        r = 2
        u_min, u_max = max(0, u - r), min(w, u + r + 1)
        v_min, v_max = max(0, v - r), min(h, v + r + 1)
        patch = depth_image[v_min:v_max, u_min:u_max]

        valid = patch[patch > 0.1]
        if len(valid) == 0:
            self._logger.warn(f"Nessun depth valido a pixel ({u}, {v})")
            return None

        depth = float(np.median(valid))

        # Pixel -> 3D camera frame
        point_cam = pixel_to_3d(u, v, depth, fx, fy, cx, cy)

        # Camera frame -> base frame
        if self._T_camera_to_base is not None:
            point_base = transform_point(point_cam, self._T_camera_to_base)
        else:
            self._logger.warn("Trasformazione camera->base non impostata")
            point_base = point_cam

        return point_base

    def estimate_multiple(self, detections: List[PodDetection],
                           depth_image: np.ndarray,
                           fx: float, fy: float,
                           cx: float, cy: float) -> List[Tuple[PodDetection, np.ndarray]]:
        """Stima posizioni 3D per multiple detection.

        Returns:
            Lista di (detection, posizione_3d) per detection con depth valido
        """
        results = []
        for det in detections:
            pos = self.estimate_3d_position(det, depth_image, fx, fy, cx, cy)
            if pos is not None:
                results.append((det, pos))
            else:
                self._logger.debug(
                    f"Skip detection a ({det.center_px}): depth non valido"
                )

        return results

    def select_nearest_pod(self, pods_3d: List[Tuple[PodDetection, np.ndarray]],
                            robot_position: np.ndarray) -> Optional[Tuple[PodDetection, np.ndarray]]:
        """Seleziona la cialda piu vicina al robot.

        Args:
            pods_3d: lista di (detection, posizione_3d)
            robot_position: posizione corrente del robot [x, y, z]

        Returns:
            (detection, posizione_3d) della cialda piu vicina, o None
        """
        if not pods_3d:
            return None

        best = None
        best_dist = float('inf')

        for det, pos in pods_3d:
            dist = np.linalg.norm(pos[:2] - robot_position[:2])
            if dist < best_dist:
                best_dist = dist
                best = (det, pos)

        return best

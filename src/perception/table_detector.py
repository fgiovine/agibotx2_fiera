"""Rilevamento tavoli tramite RANSAC su point cloud.

Trova piani orizzontali nell'intervallo di altezza previsto,
li identifica come tavoli SX/CENTRO/DX in base alla posizione relativa.
"""

from typing import Optional, List, Tuple
from dataclasses import dataclass

import numpy as np

try:
    import open3d as o3d
except ImportError:
    o3d = None

from rclpy.node import Node

from src.utils.config_loader import ConfigLoader


@dataclass
class TableInfo:
    """Informazioni su un tavolo rilevato."""
    name: str              # "pod_table", "box_table", "full_table"
    center: np.ndarray     # Posizione 3D del centro [x, y, z]
    normal: np.ndarray     # Normale al piano
    corners: np.ndarray    # 4 angoli del piano
    height: float          # Altezza dal suolo
    size: Tuple[float, float]  # Dimensioni stimate (w, d)


class TableDetector:
    """Rileva e identifica i 3 tavoli usando segmentazione RANSAC."""

    def __init__(self, node: Node, config: ConfigLoader):
        self._node = node
        self._config = config
        self._logger = node.get_logger()

        # Parametri RANSAC
        self._dist_thresh = config.get('perception.ransac.distance_threshold_m', 0.01)
        self._num_iters = config.get('perception.ransac.num_iterations', 1000)
        self._min_points = config.get('perception.ransac.min_points', 500)

        # Altezza attesa tavoli
        self._h_min, self._h_max = config.get(
            'tables.height_range_m', [0.6, 0.9]
        )

        # Tavoli rilevati
        self._tables: dict = {}  # name -> TableInfo
        self._last_detection_time: float = 0.0

    def detect_from_pointcloud(self, points: np.ndarray) -> List[TableInfo]:
        """Rileva tavoli da una point cloud.

        Args:
            points: array Nx3 di punti 3D nel frame base robot

        Returns:
            Lista di TableInfo rilevati
        """
        if o3d is None:
            self._logger.error("Open3D non disponibile")
            return []

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)

        # Filtra punti nell'intervallo di altezza dei tavoli
        pts = np.asarray(pcd.points)
        mask = (pts[:, 2] >= self._h_min) & (pts[:, 2] <= self._h_max)
        filtered = pts[mask]

        if len(filtered) < self._min_points:
            self._logger.warn(
                f"Troppo pochi punti nell'intervallo altezza: {len(filtered)}"
            )
            return []

        filtered_pcd = o3d.geometry.PointCloud()
        filtered_pcd.points = o3d.utility.Vector3dVector(filtered)

        tables = []
        remaining_pcd = filtered_pcd

        # Cerca fino a 3 piani
        for _ in range(3):
            if len(remaining_pcd.points) < self._min_points:
                break

            plane_model, inliers = remaining_pcd.segment_plane(
                distance_threshold=self._dist_thresh,
                ransac_n=3,
                num_iterations=self._num_iters,
            )

            if len(inliers) < self._min_points:
                break

            # Verifica che sia orizzontale (normale quasi verticale)
            normal = np.array(plane_model[:3])
            if abs(normal[2]) < 0.8:  # Non abbastanza orizzontale
                remaining_pcd = remaining_pcd.select_by_index(inliers, invert=True)
                continue

            inlier_cloud = remaining_pcd.select_by_index(inliers)
            inlier_points = np.asarray(inlier_cloud.points)

            center = np.mean(inlier_points, axis=0)
            height = center[2]

            # Stima dimensioni con bounding box
            min_pt = np.min(inlier_points, axis=0)
            max_pt = np.max(inlier_points, axis=0)
            size_x = max_pt[0] - min_pt[0]
            size_y = max_pt[1] - min_pt[1]

            corners = np.array([
                [min_pt[0], min_pt[1], height],
                [max_pt[0], min_pt[1], height],
                [max_pt[0], max_pt[1], height],
                [min_pt[0], max_pt[1], height],
            ])

            tables.append(TableInfo(
                name="",
                center=center,
                normal=normal,
                corners=corners,
                height=height,
                size=(size_x, size_y),
            ))

            remaining_pcd = remaining_pcd.select_by_index(inliers, invert=True)

        # Assegna nomi in base alla posizione X (sinistra -> destra)
        tables.sort(key=lambda t: t.center[1])  # Ordina per Y (laterale)

        names = ["pod_table", "box_table", "full_table"]
        for i, table in enumerate(tables):
            if i < len(names):
                table.name = names[i]

        self._tables = {t.name: t for t in tables}

        self._logger.info(
            f"Rilevati {len(tables)} tavoli: "
            + ", ".join(f"{t.name} ({t.center[0]:.2f}, {t.center[1]:.2f}, {t.center[2]:.2f})"
                        for t in tables)
        )

        return tables

    def detect_from_depth(self, depth_image: np.ndarray,
                           fx: float, fy: float, cx: float, cy: float,
                           T_camera_to_base: np.ndarray) -> List[TableInfo]:
        """Rileva tavoli da un'immagine depth.

        Converte l'immagine depth in point cloud e chiama detect_from_pointcloud.
        """
        h, w = depth_image.shape[:2]
        u_coords, v_coords = np.meshgrid(np.arange(w), np.arange(h))

        # Maschera punti validi
        valid = depth_image > 0.1  # Ignora punti troppo vicini
        valid &= depth_image < 3.0  # Ignora punti troppo lontani

        z = depth_image[valid]
        u = u_coords[valid]
        v = v_coords[valid]

        x = (u - cx) * z / fx
        y = (v - cy) * z / fy

        points_camera = np.stack([x, y, z], axis=1)

        # Trasforma nel frame base
        R = T_camera_to_base[:3, :3]
        t = T_camera_to_base[:3, 3]
        points_base = (R @ points_camera.T).T + t

        return self.detect_from_pointcloud(points_base)

    def get_table(self, name: str) -> Optional[TableInfo]:
        """Ritorna info di un tavolo specifico."""
        return self._tables.get(name)

    def check_tables_moved(self, new_tables: List[TableInfo]) -> bool:
        """Verifica se i tavoli si sono spostati rispetto all'ultima rilevazione.

        Args:
            new_tables: nuova rilevazione

        Returns:
            True se almeno un tavolo si e spostato oltre la soglia
        """
        threshold = self._config.get('tables.table_moved_threshold_m', 0.15)

        for new_t in new_tables:
            if new_t.name in self._tables:
                old_center = self._tables[new_t.name].center
                dist = np.linalg.norm(new_t.center - old_center)
                if dist > threshold:
                    self._logger.warn(
                        f"Tavolo {new_t.name} spostato di {dist:.3f}m!"
                    )
                    return True
        return False

    @property
    def tables(self) -> dict:
        return self._tables

    @property
    def all_tables_detected(self) -> bool:
        return len(self._tables) >= 3

"""Tracking posizione robot: dead-reckoning + correzione visiva.

Il SDK Agibot X2 non fornisce odometria built-in, quindi usiamo:
- Dead-reckoning basato su comandi di velocita
- Correzione visiva quando i tavoli sono visibili
"""

import time
from typing import Optional

import numpy as np

from rclpy.node import Node

from src.utils.config_loader import ConfigLoader
from src.perception.table_detector import TableInfo


class PositionTracker:
    """Traccia la posizione del robot nel mondo."""

    def __init__(self, node: Node, config: ConfigLoader):
        self._node = node
        self._config = config
        self._logger = node.get_logger()

        # Posizione e orientamento stimati
        self._position = np.array([0.0, 0.0, 0.0])  # x, y, z
        self._yaw: float = 0.0  # Orientamento nel piano
        self._last_update: float = time.time()

        # Tavoli di riferimento per correzione
        self._reference_tables: dict = {}

    def reset(self, position: np.ndarray = None, yaw: float = 0.0):
        """Reset posizione."""
        self._position = position if position is not None else np.zeros(3)
        self._yaw = yaw
        self._last_update = time.time()

    def update_from_velocity(self, vx: float, vy: float, omega: float):
        """Aggiorna posizione con dead-reckoning.

        Args:
            vx: velocita x (m/s) nel frame robot
            vy: velocita y (m/s) nel frame robot
            omega: velocita angolare (rad/s)
        """
        now = time.time()
        dt = now - self._last_update
        self._last_update = now

        if dt > 0.5:
            # Troppo tempo dall'ultimo update, skip
            return

        # Rotazione nel frame mondo
        cos_yaw = np.cos(self._yaw)
        sin_yaw = np.sin(self._yaw)

        # Velocita nel frame mondo
        world_vx = vx * cos_yaw - vy * sin_yaw
        world_vy = vx * sin_yaw + vy * cos_yaw

        self._position[0] += world_vx * dt
        self._position[1] += world_vy * dt
        self._yaw += omega * dt

        # Normalizza yaw
        while self._yaw > np.pi:
            self._yaw -= 2 * np.pi
        while self._yaw < -np.pi:
            self._yaw += 2 * np.pi

    def update_from_table_observation(self, table: TableInfo,
                                       observed_distance: float,
                                       observed_angle: float):
        """Corregge la posizione usando l'osservazione di un tavolo noto.

        Args:
            table: tavolo rilevato con posizione nota
            observed_distance: distanza misurata al tavolo
            observed_angle: angolo al tavolo nel frame camera
        """
        if table.name not in self._reference_tables:
            return

        ref_pos = self._reference_tables[table.name]

        # Stima posizione robot basata sull'osservazione
        angle_world = self._yaw + observed_angle
        estimated_x = ref_pos[0] - observed_distance * np.cos(angle_world)
        estimated_y = ref_pos[1] - observed_distance * np.sin(angle_world)

        # Filtro complementare: 70% dead-reckoning, 30% visione
        alpha = 0.3
        self._position[0] = (1 - alpha) * self._position[0] + alpha * estimated_x
        self._position[1] = (1 - alpha) * self._position[1] + alpha * estimated_y

    def set_reference_tables(self, tables: dict):
        """Imposta le posizioni di riferimento dei tavoli.

        Args:
            tables: dict {name: TableInfo}
        """
        self._reference_tables = {
            name: table.center for name, table in tables.items()
        }
        self._logger.info(
            f"Tavoli di riferimento impostati: {list(tables.keys())}"
        )

    @property
    def position(self) -> np.ndarray:
        return self._position.copy()

    @property
    def yaw(self) -> float:
        return self._yaw

    @property
    def pose_2d(self) -> tuple:
        """Ritorna (x, y, yaw)."""
        return self._position[0], self._position[1], self._yaw

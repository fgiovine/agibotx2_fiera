"""Tracciamento scatola: conta cialde piazzate, posizione scatola."""

from typing import Optional, Tuple

import numpy as np

from rclpy.node import Node

from src.utils.config_loader import ConfigLoader
from src.utils.transforms import compute_box_slot_position


class BoxTracker:
    """Tiene traccia dello stato della scatola e delle cialde piazzate."""

    def __init__(self, node: Node, config: ConfigLoader):
        self._node = node
        self._config = config
        self._logger = node.get_logger()

        # Configurazione griglia
        self._rows = config.get('pods.box_grid_rows', 4)
        self._cols = config.get('pods.box_grid_cols', 5)
        self._max_pods = config.get('pods.max_per_box', 20)
        self._spacing = config.get('pods.box_grid_spacing_mm', 50.0) / 1000.0

        # Stato corrente
        self._pod_count: int = 0
        self._grid: np.ndarray = np.zeros((self._rows, self._cols), dtype=bool)
        self._box_position: Optional[np.ndarray] = None
        self._box_detected: bool = False

    def reset(self):
        """Reset per nuova scatola."""
        self._pod_count = 0
        self._grid = np.zeros((self._rows, self._cols), dtype=bool)
        self._logger.info("Box tracker resettato per nuova scatola")

    def set_box_position(self, position: np.ndarray):
        """Imposta la posizione della scatola (angolo in basso a sinistra)."""
        self._box_position = position
        self._box_detected = True

    def add_pod(self) -> bool:
        """Registra una cialda piazzata.

        Returns:
            True se la scatola non e ancora piena
        """
        row, col = self._get_next_slot()
        if row is None:
            return False

        self._grid[row, col] = True
        self._pod_count += 1
        self._logger.info(
            f"Cialda piazzata in slot ({row}, {col}). "
            f"Totale: {self._pod_count}/{self._max_pods}"
        )
        return True

    def _get_next_slot(self) -> Tuple[Optional[int], Optional[int]]:
        """Trova il prossimo slot libero nella griglia.

        Riempie riga per riga, da sinistra a destra.
        """
        for row in range(self._rows):
            for col in range(self._cols):
                if not self._grid[row, col]:
                    return row, col
        return None, None

    def get_next_place_position(self) -> Optional[np.ndarray]:
        """Ritorna la posizione 3D del prossimo slot libero.

        Returns:
            Posizione 3D o None se scatola piena
        """
        if self._box_position is None:
            self._logger.warn("Posizione scatola non impostata")
            return None

        row, col = self._get_next_slot()
        if row is None:
            return None

        return compute_box_slot_position(
            row, col, self._box_position, self._spacing
        )

    @property
    def pod_count(self) -> int:
        return self._pod_count

    @property
    def is_full(self) -> bool:
        return self._pod_count >= self._max_pods

    @property
    def box_detected(self) -> bool:
        return self._box_detected

    @property
    def box_position(self) -> Optional[np.ndarray]:
        return self._box_position

    @property
    def remaining_slots(self) -> int:
        return self._max_pods - self._pod_count

    @property
    def fill_percentage(self) -> float:
        if self._max_pods == 0:
            return 100.0
        return (self._pod_count / self._max_pods) * 100.0

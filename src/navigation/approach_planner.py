"""Visual servoing per approccio ai tavoli.

Pipeline di approccio:
1. Fase grossolana: cammina verso tavolo (0.3 m/s) fino a 1m
2. Fase fine: rallenta a 0.15 m/s, visual servoing su bordo tavolo
3. Allineamento: correzioni angolari
4. Stop: a 0.45m dal bordo (raggiungibile con braccio)
"""

from typing import Optional, Tuple
from enum import Enum

import numpy as np

from rclpy.node import Node

from src.utils.config_loader import ConfigLoader
from src.utils.transforms import distance_2d, angle_to_target
from src.navigation.locomotion_controller import LocomotionController
from src.perception.table_detector import TableInfo


class ApproachPhase(Enum):
    COARSE = "coarse"       # Fase grossolana
    FINE = "fine"           # Approccio fine
    ALIGNMENT = "alignment"  # Allineamento angolare
    DONE = "done"           # Arrivato


class ApproachPlanner:
    """Pianifica e esegue l'approccio ai tavoli con visual servoing."""

    def __init__(self, node: Node, config: ConfigLoader,
                 locomotion: LocomotionController):
        self._node = node
        self._config = config
        self._locomotion = locomotion
        self._logger = node.get_logger()

        # Parametri
        self._approach_dist = config.get('tables.approach_distance_m', 0.45)
        self._coarse_dist = config.get('tables.coarse_approach_distance_m', 1.0)
        self._gain_lin = config.get('navigation.visual_servoing.gain_linear', 0.5)
        self._gain_ang = config.get('navigation.visual_servoing.gain_angular', 0.8)
        self._conv_lin = config.get(
            'navigation.visual_servoing.convergence_threshold_m', 0.02
        )
        self._conv_ang = config.get(
            'navigation.visual_servoing.convergence_threshold_rad', 0.05
        )

        # Stato
        self._phase = ApproachPhase.COARSE
        self._target_table: Optional[TableInfo] = None

    def start_approach(self, table: TableInfo):
        """Inizia l'approccio verso un tavolo."""
        self._target_table = table
        self._phase = ApproachPhase.COARSE
        self._logger.info(f"Inizio approccio a tavolo: {table.name}")

    def update(self, robot_position: np.ndarray,
               robot_yaw: float,
               table_position: Optional[np.ndarray] = None) -> ApproachPhase:
        """Aggiorna il controllo di approccio.

        Args:
            robot_position: posizione corrente robot [x, y, z]
            robot_yaw: orientamento corrente (rad)
            table_position: posizione aggiornata tavolo (da visione)

        Returns:
            Fase corrente dell'approccio
        """
        if self._target_table is None:
            return ApproachPhase.DONE

        # Usa posizione aggiornata se disponibile
        target = (
            table_position if table_position is not None
            else self._target_table.center
        )

        dist = distance_2d(robot_position, target)
        angle_err = angle_to_target(robot_position, robot_yaw, target)

        if self._phase == ApproachPhase.COARSE:
            self._update_coarse(dist, angle_err)
        elif self._phase == ApproachPhase.FINE:
            self._update_fine(dist, angle_err)
        elif self._phase == ApproachPhase.ALIGNMENT:
            self._update_alignment(angle_err)

        return self._phase

    def _update_coarse(self, dist: float, angle_err: float):
        """Fase grossolana: cammina verso il tavolo."""
        if dist <= self._coarse_dist:
            self._phase = ApproachPhase.FINE
            self._logger.info("Approccio: fase fine")
            return

        # Correggi angolo mentre cammini
        angular = np.clip(self._gain_ang * angle_err,
                          -0.3, 0.3)
        linear = self._config.get('navigation.max_speed_ms', 0.3)

        # Rallenta se l'angolo e grande
        if abs(angle_err) > 0.3:
            linear *= 0.5

        self._locomotion.send_velocity(linear_x=linear, angular_z=angular)

    def _update_fine(self, dist: float, angle_err: float):
        """Fase fine: visual servoing lento."""
        if dist <= self._approach_dist + self._conv_lin:
            self._locomotion.stop()
            self._phase = ApproachPhase.ALIGNMENT
            self._logger.info("Approccio: allineamento")
            return

        # Velocita proporzionale alla distanza
        speed_factor = min(1.0, (dist - self._approach_dist) / 0.5)
        linear = self._config.get('navigation.fine_approach_speed_ms', 0.15) * speed_factor

        angular = np.clip(self._gain_ang * angle_err, -0.2, 0.2)

        self._locomotion.send_velocity(linear_x=linear, angular_z=angular)

    def _update_alignment(self, angle_err: float):
        """Fase allineamento: solo rotazione per centrare."""
        if abs(angle_err) < self._conv_ang:
            self._locomotion.stop()
            self._phase = ApproachPhase.DONE
            self._logger.info("Approccio completato!")
            return

        angular = np.clip(self._gain_ang * angle_err, -0.15, 0.15)
        self._locomotion.send_velocity(angular_z=angular)

    @property
    def phase(self) -> ApproachPhase:
        return self._phase

    @property
    def is_done(self) -> bool:
        return self._phase == ApproachPhase.DONE

    def abort(self):
        """Interrompe l'approccio."""
        self._locomotion.stop()
        self._phase = ApproachPhase.DONE
        self._target_table = None

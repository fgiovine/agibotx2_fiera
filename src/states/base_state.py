"""Classe base per tutti gli stati della FSM.

Ogni stato ha accesso a tutti i moduli del robot tramite il context.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional, Any

from rclpy.node import Node


@dataclass
class StateResult:
    """Risultato dell'esecuzione di uno stato."""
    next_state: str             # Nome dello stato successivo
    data: dict = field(default_factory=dict)  # Dati da passare al prossimo stato
    error: Optional[str] = None  # Messaggio di errore se fallito


class BaseState(ABC):
    """Classe base per tutti gli stati."""

    NAME: str = "BASE"

    def __init__(self, context: 'DemoContext'):
        """
        Args:
            context: contesto condiviso con tutti i moduli del robot
        """
        self._context = context
        self._logger = context.node.get_logger()

    # --- Shortcut per accesso ai moduli ---

    @property
    def node(self) -> Node:
        return self._context.node

    @property
    def config(self):
        return self._context.config

    @property
    def camera(self):
        return self._context.camera

    @property
    def table_detector(self):
        return self._context.table_detector

    @property
    def pod_detector(self):
        return self._context.pod_detector

    @property
    def pose_estimator(self):
        return self._context.pose_estimator

    @property
    def box_tracker(self):
        return self._context.box_tracker

    @property
    def ik_solver(self):
        return self._context.ik_solver

    @property
    def arm_controller(self):
        return self._context.arm_controller

    @property
    def gripper(self):
        return self._context.gripper

    @property
    def trajectory_planner(self):
        return self._context.trajectory_planner

    @property
    def locomotion(self):
        return self._context.locomotion

    @property
    def approach_planner(self):
        return self._context.approach_planner

    @property
    def position_tracker(self):
        return self._context.position_tracker

    @property
    def commentary(self):
        return self._context.commentary

    @property
    def mode_manager(self):
        return self._context.mode_manager

    @property
    def safety(self):
        return self._context.safety

    @property
    def voice_interaction(self):
        return self._context.voice_interaction

    @property
    def arm_side(self) -> str:
        return self._context.config.get('manipulation.arm.default_side', 'left')

    @abstractmethod
    def enter(self):
        """Chiamato quando si entra nello stato."""
        pass

    @abstractmethod
    def execute(self) -> StateResult:
        """Esegue la logica dello stato.

        Returns:
            StateResult con il prossimo stato e dati opzionali
        """
        pass

    @abstractmethod
    def exit(self):
        """Chiamato quando si esce dallo stato."""
        pass

    def check_tables_moved(self) -> bool:
        """Controlla se i tavoli si sono spostati (evento globale)."""
        rgb = self.camera.get_rgb()
        if rgb is None:
            return False

        frame = self.camera.get_frame()
        if frame is None:
            return False

        rgb, depth = frame
        fx, fy, cx, cy = self.camera.intrinsics
        if fx == 0:
            return False

        # Periodicamente verifica posizione tavoli
        # (implementazione semplificata - nella versione completa
        # userebbe la point cloud)
        return False

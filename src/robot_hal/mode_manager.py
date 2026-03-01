"""Gestione modalita operative del robot Agibot X2.

Modalita disponibili dal SDK:
- STAND: robot in piedi, stabile
- LOCOMOTION: camminata attiva
- MANIPULATION: braccio controllabile
- STAND_BY: standby basso consumo
"""

import rclpy
from rclpy.node import Node

from agt_sdk_msgs.srv import ChangeMode
from agt_sdk_msgs.msg import RobotMode

from src.utils.ros_helpers import call_service_sync


class RobotMode:
    STAND = 0
    LOCOMOTION = 1
    MANIPULATION = 2
    STAND_BY = 3


MODE_NAMES = {
    RobotMode.STAND: "STAND",
    RobotMode.LOCOMOTION: "LOCOMOTION",
    RobotMode.MANIPULATION: "MANIPULATION",
    RobotMode.STAND_BY: "STAND_BY",
}


class ModeManager:
    """Gestisce il cambio modalita del robot."""

    SERVICE_NAME = "/robot/change_mode"

    def __init__(self, node: Node):
        self._node = node
        self._current_mode: int = RobotMode.STAND
        self._client = node.create_client(ChangeMode, self.SERVICE_NAME)
        self._logger = node.get_logger()

    @property
    def current_mode(self) -> int:
        return self._current_mode

    @property
    def current_mode_name(self) -> str:
        return MODE_NAMES.get(self._current_mode, "UNKNOWN")

    def switch_mode(self, target_mode: int, timeout_sec: float = 10.0) -> bool:
        """Cambia modalita robot.

        Args:
            target_mode: modalita target (RobotMode.*)
            timeout_sec: timeout

        Returns:
            True se il cambio e riuscito
        """
        if target_mode == self._current_mode:
            self._logger.info(f"Gia in modalita {MODE_NAMES.get(target_mode)}")
            return True

        self._logger.info(
            f"Cambio modalita: {self.current_mode_name} -> {MODE_NAMES.get(target_mode)}"
        )

        request = ChangeMode.Request()
        request.target_mode = target_mode

        response = call_service_sync(
            self._node, self._client, request, timeout_sec=timeout_sec
        )

        if response is not None and response.success:
            self._current_mode = target_mode
            self._logger.info(f"Modalita cambiata a {MODE_NAMES.get(target_mode)}")
            return True

        self._logger.error(
            f"Cambio modalita fallito verso {MODE_NAMES.get(target_mode)}"
        )
        return False

    def ensure_manipulation(self) -> bool:
        """Assicura che il robot sia in modalita MANIPULATION."""
        return self.switch_mode(RobotMode.MANIPULATION)

    def ensure_locomotion(self) -> bool:
        """Assicura che il robot sia in modalita LOCOMOTION."""
        return self.switch_mode(RobotMode.LOCOMOTION)

    def ensure_stand(self) -> bool:
        """Assicura che il robot sia in modalita STAND."""
        return self.switch_mode(RobotMode.STAND)

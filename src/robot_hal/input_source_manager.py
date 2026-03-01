"""Registrazione e gestione sorgente input per il robot.

Il SDK Agibot X2 richiede la registrazione di una sorgente di input
prima di poter inviare comandi. Solo una sorgente alla volta puo
controllare il robot.
"""

import rclpy
from rclpy.node import Node

from agt_sdk_msgs.srv import RegisterInputSource, UnregisterInputSource
from agt_sdk_msgs.msg import InputSourceStatus

from src.utils.ros_helpers import call_service_sync


class InputSourceManager:
    """Gestisce la registrazione come sorgente di input del robot."""

    REGISTER_SERVICE = "/robot/register_input_source"
    UNREGISTER_SERVICE = "/robot/unregister_input_source"
    STATUS_TOPIC = "/robot/input_source_status"

    def __init__(self, node: Node, source_name: str = "demo_fiera"):
        self._node = node
        self._source_name = source_name
        self._source_id: int = -1
        self._is_registered = False
        self._logger = node.get_logger()

        self._register_client = node.create_client(
            RegisterInputSource, self.REGISTER_SERVICE
        )
        self._unregister_client = node.create_client(
            UnregisterInputSource, self.UNREGISTER_SERVICE
        )

        # Sottoscrivi allo stato
        self._status_sub = node.create_subscription(
            InputSourceStatus,
            self.STATUS_TOPIC,
            self._status_callback,
            10,
        )

    def _status_callback(self, msg: InputSourceStatus):
        """Monitora lo stato della sorgente input."""
        if msg.source_id == self._source_id:
            self._is_registered = msg.is_active

    @property
    def is_registered(self) -> bool:
        return self._is_registered

    @property
    def source_id(self) -> int:
        return self._source_id

    def register(self, timeout_sec: float = 10.0) -> bool:
        """Registra questa applicazione come sorgente di input.

        Returns:
            True se registrazione riuscita
        """
        self._logger.info(f"Registrazione input source: {self._source_name}")

        request = RegisterInputSource.Request()
        request.source_name = self._source_name
        request.priority = 10  # Alta priorita per demo

        response = call_service_sync(
            self._node, self._register_client, request, timeout_sec=timeout_sec
        )

        if response is not None and response.success:
            self._source_id = response.source_id
            self._is_registered = True
            self._logger.info(
                f"Input source registrato con ID: {self._source_id}"
            )
            return True

        self._logger.error("Registrazione input source fallita")
        return False

    def unregister(self) -> bool:
        """Rimuove la registrazione."""
        if not self._is_registered:
            return True

        request = UnregisterInputSource.Request()
        request.source_id = self._source_id

        response = call_service_sync(
            self._node, self._unregister_client, request
        )

        if response is not None and response.success:
            self._is_registered = False
            self._logger.info("Input source rimosso")
            return True

        return False

    def destroy(self):
        """Cleanup: rimuove registrazione e sottoscrizioni."""
        self.unregister()
        self._node.destroy_subscription(self._status_sub)

"""Controllo gripper OmniPicker.

Posizione: 0.0 (chiuso) - 1.0 (aperto)
Stato: 0=moving, 1=open, 2=stalled (presa OK), 3=closed
"""

import time
from typing import Optional

import rclpy
from rclpy.node import Node

from agt_sdk_msgs.msg import GripperCommand, GripperState

from src.utils.config_loader import ConfigLoader
from src.utils.ros_helpers import reliable_qos, sensor_qos


class GripperController:
    """Controlla il gripper OmniPicker."""

    LEFT_CMD_TOPIC = "/left_gripper/command"
    RIGHT_CMD_TOPIC = "/right_gripper/command"
    LEFT_STATE_TOPIC = "/left_gripper/state"
    RIGHT_STATE_TOPIC = "/right_gripper/state"

    # Stati gripper
    MOVING = 0
    OPEN = 1
    STALLED = 2   # Presa riuscita
    CLOSED = 3

    def __init__(self, node: Node, config: ConfigLoader):
        self._node = node
        self._config = config
        self._logger = node.get_logger()

        # Parametri
        self._open_pos = config.get('manipulation.gripper.open_position', 0.7)
        self._close_pos = config.get('manipulation.gripper.close_position', 0.0)
        self._grasp_effort = config.get('manipulation.gripper.grasp_effort', 0.4)
        self._box_effort = config.get('manipulation.gripper.box_grasp_effort', 0.8)
        self._stall_state = config.get('manipulation.gripper.stall_state', 2)

        # Stato
        self._left_state: Optional[int] = None
        self._right_state: Optional[int] = None
        self._left_position: float = 0.0
        self._right_position: float = 0.0

        # Publisher
        self._left_pub = node.create_publisher(
            GripperCommand, self.LEFT_CMD_TOPIC, reliable_qos()
        )
        self._right_pub = node.create_publisher(
            GripperCommand, self.RIGHT_CMD_TOPIC, reliable_qos()
        )

        # Subscriber
        self._left_state_sub = node.create_subscription(
            GripperState, self.LEFT_STATE_TOPIC,
            self._left_state_callback, sensor_qos()
        )
        self._right_state_sub = node.create_subscription(
            GripperState, self.RIGHT_STATE_TOPIC,
            self._right_state_callback, sensor_qos()
        )

    def _left_state_callback(self, msg: GripperState):
        self._left_state = msg.state
        self._left_position = msg.position

    def _right_state_callback(self, msg: GripperState):
        self._right_state = msg.state
        self._right_position = msg.position

    def open(self, side: str = "left", position: float = None) -> bool:
        """Apri il gripper.

        Args:
            side: "left" o "right"
            position: posizione apertura (default da config)

        Returns:
            True se comando inviato
        """
        pos = position if position is not None else self._open_pos
        return self._send_command(side, pos, effort=0.3)

    def close_for_pod(self, side: str = "left") -> bool:
        """Chiudi il gripper per afferrare una cialda (effort gentile)."""
        return self._send_command(side, self._close_pos, self._grasp_effort)

    def close_for_box(self, side: str = "left") -> bool:
        """Chiudi il gripper per afferrare una scatola (effort forte)."""
        return self._send_command(side, self._close_pos, self._box_effort)

    def _send_command(self, side: str, position: float,
                       effort: float) -> bool:
        """Invia comando al gripper."""
        msg = GripperCommand()
        msg.position = float(np.clip(position, 0.0, 1.0))
        msg.effort = float(np.clip(effort, 0.0, 1.0))

        pub = self._left_pub if side == "left" else self._right_pub
        pub.publish(msg)

        self._logger.debug(
            f"Gripper {side}: pos={position:.2f}, effort={effort:.2f}"
        )
        return True

    def wait_for_grasp(self, side: str = "left",
                        timeout_s: float = 3.0) -> bool:
        """Attende che il gripper completi la presa.

        Returns:
            True se presa riuscita (stato STALLED)
        """
        start = time.time()
        while (time.time() - start) < timeout_s:
            state = self._left_state if side == "left" else self._right_state
            if state == self.STALLED:
                self._logger.info(f"Presa {side} riuscita!")
                return True
            if state == self.CLOSED:
                self._logger.warn(f"Gripper {side} chiuso senza oggetto")
                return False
            rclpy.spin_once(self._node, timeout_sec=0.05)

        self._logger.warn(f"Timeout attesa presa {side}")
        return False

    def has_object(self, side: str = "left") -> bool:
        """Verifica se il gripper tiene un oggetto."""
        state = self._left_state if side == "left" else self._right_state
        return state == self.STALLED

    def get_position(self, side: str = "left") -> float:
        """Ritorna la posizione corrente del gripper."""
        return self._left_position if side == "left" else self._right_position

    def destroy(self):
        self._node.destroy_subscription(self._left_state_sub)
        self._node.destroy_subscription(self._right_state_sub)


# Import necessario per np.clip nel _send_command
import numpy as np

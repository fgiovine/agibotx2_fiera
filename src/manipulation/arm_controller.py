"""Controllo braccio: pubblica JointCommandArray.

Invia comandi di posizione ai giunti del braccio via topic ROS2.
"""

import time
from typing import List, Optional

import numpy as np

import rclpy
from rclpy.node import Node

from agt_sdk_msgs.msg import JointCommandArray, JointCommand

from src.utils.config_loader import ConfigLoader
from src.utils.ros_helpers import reliable_qos
from src.robot_hal.safety_monitor import SafetyMonitor


class ArmController:
    """Controlla i giunti del braccio pubblicando comandi."""

    LEFT_CMD_TOPIC = "/left_arm/joint_command"
    RIGHT_CMD_TOPIC = "/right_arm/joint_command"
    LEFT_STATE_TOPIC = "/left_arm/joint_state"
    RIGHT_STATE_TOPIC = "/right_arm/joint_state"

    def __init__(self, node: Node, config: ConfigLoader,
                 safety: SafetyMonitor):
        self._node = node
        self._config = config
        self._safety = safety
        self._logger = node.get_logger()

        # Publisher comandi
        self._left_pub = node.create_publisher(
            JointCommandArray, self.LEFT_CMD_TOPIC, reliable_qos()
        )
        self._right_pub = node.create_publisher(
            JointCommandArray, self.RIGHT_CMD_TOPIC, reliable_qos()
        )

        # Stato corrente giunti
        self._left_positions: Optional[np.ndarray] = None
        self._right_positions: Optional[np.ndarray] = None

        # Sottoscrizione stato
        from agt_sdk_msgs.msg import JointStateArray
        self._left_state_sub = node.create_subscription(
            JointStateArray, self.LEFT_STATE_TOPIC,
            self._left_state_callback, 10
        )
        self._right_state_sub = node.create_subscription(
            JointStateArray, self.RIGHT_STATE_TOPIC,
            self._right_state_callback, 10
        )

    def _left_state_callback(self, msg):
        self._left_positions = np.array([j.position for j in msg.joints])

    def _right_state_callback(self, msg):
        self._right_positions = np.array([j.position for j in msg.joints])

    def get_current_positions(self, side: str = "left") -> Optional[np.ndarray]:
        """Ritorna le posizioni correnti dei giunti."""
        if side == "left":
            return self._left_positions.copy() if self._left_positions is not None else None
        return self._right_positions.copy() if self._right_positions is not None else None

    def send_positions(self, positions: np.ndarray,
                        side: str = "left") -> bool:
        """Invia comandi di posizione al braccio.

        Args:
            positions: 7 posizioni in radianti
            side: "left" o "right"

        Returns:
            True se il comando e stato inviato
        """
        # Controllo sicurezza
        group = f"{side}_arm"
        if not self._safety.check_joint_positions(list(positions), group):
            self._logger.error(f"Posizioni fuori limiti per {side} arm!")
            return False

        msg = JointCommandArray()
        for i, pos in enumerate(positions):
            cmd = JointCommand()
            cmd.position = float(pos)
            cmd.mode = 1  # Modalita posizione
            msg.joints.append(cmd)

        pub = self._left_pub if side == "left" else self._right_pub
        pub.publish(msg)
        return True

    def move_to_positions(self, target: np.ndarray,
                           side: str = "left",
                           duration_s: float = 2.0,
                           rate_hz: float = 50.0) -> bool:
        """Muovi il braccio interpolando linearmente verso il target.

        Args:
            target: posizioni target (7 valori)
            side: "left" o "right"
            duration_s: durata del movimento
            rate_hz: frequenza di pubblicazione

        Returns:
            True se il movimento e completato
        """
        current = self.get_current_positions(side)
        if current is None:
            self._logger.error(f"Posizioni correnti {side} arm non disponibili")
            return False

        steps = int(duration_s * rate_hz)
        dt = 1.0 / rate_hz

        for i in range(steps + 1):
            if not self._safety.is_safe:
                self._logger.error("Safety stop durante il movimento!")
                return False

            alpha = i / steps
            # Interpolazione con smooth step
            alpha = alpha * alpha * (3 - 2 * alpha)

            q = current + alpha * (target - current)
            if not self.send_positions(q, side):
                return False

            time.sleep(dt)

        return True

    def execute_trajectory(self, trajectory: list,
                            side: str = "left",
                            rate_hz: float = 50.0) -> bool:
        """Esegue una traiettoria pre-pianificata.

        Args:
            trajectory: lista di dict con 'positions', 'time_from_start'
            side: braccio
            rate_hz: frequenza comandi

        Returns:
            True se completata
        """
        group = f"{side}_arm"
        if not self._safety.check_trajectory(trajectory, group):
            self._logger.error("Traiettoria non sicura, esecuzione annullata")
            return False

        dt = 1.0 / rate_hz
        start_time = time.time()

        for waypoint in trajectory:
            target_time = waypoint.get('time_from_start', 0.0)

            # Attendi il tempo giusto
            while (time.time() - start_time) < target_time:
                if not self._safety.is_safe:
                    self._logger.error("Safety stop durante traiettoria!")
                    return False
                time.sleep(dt)

            positions = np.array(waypoint['positions'])
            if not self.send_positions(positions, side):
                return False

        return True

    def is_at_target(self, target: np.ndarray, side: str = "left",
                      tolerance: float = 0.02) -> bool:
        """Verifica se il braccio ha raggiunto il target."""
        current = self.get_current_positions(side)
        if current is None:
            return False
        return float(np.max(np.abs(current - target))) < tolerance

    def destroy(self):
        self._node.destroy_subscription(self._left_state_sub)
        self._node.destroy_subscription(self._right_state_sub)

"""Controllo locomozione: velocita + gestione input source.

Invia comandi di velocita lineare e angolare al robot per la camminata.
"""

import time
from typing import Optional

import numpy as np

import rclpy
from rclpy.node import Node

from geometry_msgs.msg import Twist

from src.utils.config_loader import ConfigLoader
from src.utils.ros_helpers import reliable_qos
from src.robot_hal.mode_manager import ModeManager, RobotMode


class LocomotionController:
    """Controlla la locomozione del robot."""

    VEL_CMD_TOPIC = "/cmd_vel"

    def __init__(self, node: Node, config: ConfigLoader,
                 mode_manager: ModeManager):
        self._node = node
        self._config = config
        self._mode_manager = mode_manager
        self._logger = node.get_logger()

        # Limiti velocita
        self._max_linear = config.get('navigation.max_speed_ms', 0.30)
        self._fine_linear = config.get('navigation.fine_approach_speed_ms', 0.15)
        self._max_angular = config.get('navigation.angular_speed_rads', 0.3)

        # Publisher
        self._vel_pub = node.create_publisher(
            Twist, self.VEL_CMD_TOPIC, reliable_qos()
        )

        # Stato
        self._is_walking = False

    def send_velocity(self, linear_x: float, linear_y: float = 0.0,
                       angular_z: float = 0.0) -> bool:
        """Invia comando di velocita.

        Args:
            linear_x: velocita avanti/indietro (m/s)
            linear_y: velocita laterale (m/s)
            angular_z: velocita rotazione (rad/s)

        Returns:
            True se il comando e stato inviato
        """
        # Assicura modalita locomotion
        if self._mode_manager.current_mode != RobotMode.LOCOMOTION:
            if not self._mode_manager.ensure_locomotion():
                return False

        # Limita velocita
        linear_x = np.clip(linear_x, -self._max_linear, self._max_linear)
        linear_y = np.clip(linear_y, -self._max_linear, self._max_linear)
        angular_z = np.clip(angular_z, -self._max_angular, self._max_angular)

        msg = Twist()
        msg.linear.x = float(linear_x)
        msg.linear.y = float(linear_y)
        msg.angular.z = float(angular_z)

        self._vel_pub.publish(msg)
        self._is_walking = (linear_x != 0 or linear_y != 0 or angular_z != 0)
        return True

    def walk_forward(self, speed: float = None) -> bool:
        """Cammina in avanti."""
        if speed is None:
            speed = self._max_linear
        return self.send_velocity(linear_x=speed)

    def walk_fine(self, speed: float = None) -> bool:
        """Cammina in avanti a velocita ridotta (approccio fine)."""
        if speed is None:
            speed = self._fine_linear
        return self.send_velocity(linear_x=speed)

    def rotate(self, angular_speed: float) -> bool:
        """Ruota sul posto."""
        return self.send_velocity(linear_x=0.0, angular_z=angular_speed)

    def stop(self) -> bool:
        """Ferma il robot."""
        result = self.send_velocity(0.0, 0.0, 0.0)
        self._is_walking = False
        return result

    def walk_to_distance(self, distance_m: float,
                          speed: float = None,
                          timeout_s: float = 30.0) -> bool:
        """Cammina per una distanza approssimativa (dead-reckoning).

        Args:
            distance_m: distanza da percorrere (positiva=avanti)
            speed: velocita (default max)
            timeout_s: timeout sicurezza

        Returns:
            True se completato
        """
        if speed is None:
            speed = self._max_linear

        if distance_m < 0:
            speed = -abs(speed)
            distance_m = abs(distance_m)
        else:
            speed = abs(speed)

        duration = distance_m / abs(speed)
        start = time.time()

        self.send_velocity(linear_x=speed)

        while (time.time() - start) < min(duration, timeout_s):
            rclpy.spin_once(self._node, timeout_sec=0.05)

        self.stop()
        return (time.time() - start) < timeout_s

    def rotate_angle(self, angle_rad: float,
                      speed: float = None,
                      timeout_s: float = 20.0) -> bool:
        """Ruota di un angolo approssimativo.

        Args:
            angle_rad: angolo (positivo=antiorario)
            speed: velocita angolare
            timeout_s: timeout

        Returns:
            True se completato
        """
        if speed is None:
            speed = self._max_angular

        direction = 1.0 if angle_rad > 0 else -1.0
        angular_speed = direction * abs(speed)
        duration = abs(angle_rad) / abs(speed)

        start = time.time()
        self.send_velocity(angular_z=angular_speed)

        while (time.time() - start) < min(duration, timeout_s):
            rclpy.spin_once(self._node, timeout_sec=0.05)

        self.stop()
        return True

    @property
    def is_walking(self) -> bool:
        return self._is_walking

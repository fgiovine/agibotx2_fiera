"""Monitoraggio sicurezza: temperature, limiti giunti, watchdog.

Implementa un watchdog che se non riceve heartbeat per 5 secondi
attiva l'emergency stop. Controlla temperature motori e limiti giunti.
"""

import time
import math
from typing import Optional

import rclpy
from rclpy.node import Node
from std_msgs.msg import Bool

from src.utils.config_loader import ConfigLoader, JointLimitsLoader


class SafetyMonitor:
    """Monitora la sicurezza del robot durante la demo."""

    def __init__(self, node: Node, config: ConfigLoader):
        self._node = node
        self._config = config
        self._logger = node.get_logger()

        # Parametri
        self._temp_limit = config.get('safety.motor_temp_limit_c', 60.0)
        self._watchdog_timeout = config.get('safety.watchdog_timeout_s', 5.0)
        self._watchdog_period = config.get('safety.watchdog_period_s', 1.0)
        self._max_joint_vel = config.get('safety.max_joint_velocity_rads', 2.0)
        self._joint_margin = math.radians(
            config.get('manipulation.arm.joint_margin_deg', 5.0)
        )

        # Limiti giunti
        self._joint_limits = JointLimitsLoader()

        # Stato
        self._last_heartbeat = time.time()
        self._is_safe = True
        self._estop_active = False
        self._motor_temps: dict = {}

        # Publisher emergency stop
        self._estop_pub = node.create_publisher(
            Bool,
            config.get('robot.emergency_stop_topic', '/emergency_stop'),
            10,
        )

        # Timer watchdog
        self._watchdog_timer = node.create_timer(
            self._watchdog_period, self._watchdog_callback
        )

    def heartbeat(self):
        """Aggiorna il timestamp dell'ultimo heartbeat."""
        self._last_heartbeat = time.time()

    def _watchdog_callback(self):
        """Controlla se il heartbeat e ancora valido."""
        elapsed = time.time() - self._last_heartbeat
        if elapsed > self._watchdog_timeout:
            self._logger.error(
                f"Watchdog timeout! Nessun heartbeat per {elapsed:.1f}s"
            )
            self.trigger_estop("Watchdog timeout")

    def update_motor_temps(self, temps: dict):
        """Aggiorna le temperature dei motori.

        Args:
            temps: dizionario {nome_giunto: temperatura_celsius}
        """
        self._motor_temps = temps
        for name, temp in temps.items():
            if temp > self._temp_limit:
                self._logger.error(
                    f"Temperatura motore {name} troppo alta: {temp:.1f}C "
                    f"(limite: {self._temp_limit}C)"
                )
                self.trigger_estop(f"Sovratemperatura motore {name}")

    def check_joint_positions(self, positions: list, group: str = "left_arm") -> bool:
        """Verifica che le posizioni dei giunti siano nei limiti.

        Args:
            positions: posizioni in radianti
            group: gruppo giunti (left_arm, right_arm)

        Returns:
            True se tutte le posizioni sono nei limiti sicuri
        """
        safe_limits = self._joint_limits.get_safe_limits(group)
        for i, (pos, (lo, hi)) in enumerate(zip(positions, safe_limits)):
            lo_rad = math.radians(lo)
            hi_rad = math.radians(hi)
            if pos < lo_rad or pos > hi_rad:
                self._logger.warn(
                    f"Giunto {group}[{i}] fuori limiti: {math.degrees(pos):.1f}deg "
                    f"(limiti: [{lo:.1f}, {hi:.1f}]deg)"
                )
                return False
        return True

    def check_joint_velocities(self, velocities: list) -> bool:
        """Verifica che le velocita dei giunti siano nei limiti."""
        for i, vel in enumerate(velocities):
            if abs(vel) > self._max_joint_vel:
                self._logger.warn(
                    f"Velocita giunto [{i}] troppo alta: {vel:.2f} rad/s "
                    f"(limite: {self._max_joint_vel:.2f})"
                )
                return False
        return True

    def check_trajectory(self, trajectory: list, group: str = "left_arm") -> bool:
        """Verifica una traiettoria completa prima dell'esecuzione.

        Args:
            trajectory: lista di waypoint, ciascuno con 'positions' e 'velocities'
            group: gruppo giunti

        Returns:
            True se la traiettoria e sicura
        """
        if not self._config.get('safety.trajectory_check_enabled', True):
            return True

        for i, waypoint in enumerate(trajectory):
            if not self.check_joint_positions(waypoint['positions'], group):
                self._logger.error(f"Traiettoria non sicura al waypoint {i}")
                return False
            if 'velocities' in waypoint:
                if not self.check_joint_velocities(waypoint['velocities']):
                    self._logger.error(
                        f"Velocita traiettoria non sicura al waypoint {i}"
                    )
                    return False
        return True

    def trigger_estop(self, reason: str):
        """Attiva l'emergency stop."""
        if not self._estop_active:
            self._estop_active = True
            self._is_safe = False
            self._logger.error(f"EMERGENCY STOP: {reason}")

            msg = Bool()
            msg.data = True
            self._estop_pub.publish(msg)

    def reset_estop(self):
        """Reset dell'emergency stop (richiede verifica manuale)."""
        self._estop_active = False
        self._is_safe = True
        self._last_heartbeat = time.time()
        self._logger.info("Emergency stop resettato")

        msg = Bool()
        msg.data = False
        self._estop_pub.publish(msg)

    @property
    def is_safe(self) -> bool:
        return self._is_safe

    @property
    def estop_active(self) -> bool:
        return self._estop_active

    def destroy(self):
        """Cleanup."""
        self._watchdog_timer.cancel()

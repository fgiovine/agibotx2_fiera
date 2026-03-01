"""Pianificazione traiettorie smooth con Ruckig.

Genera traiettorie con limiti su velocita, accelerazione e jerk
per movimenti fluidi del braccio.
"""

from typing import List, Optional

import numpy as np

from rclpy.node import Node

from src.utils.config_loader import ConfigLoader

try:
    from ruckig import InputParameter, Ruckig, Trajectory as RuckigTrajectory
except ImportError:
    Ruckig = None


class TrajectoryPlanner:
    """Pianifica traiettorie smooth con Ruckig."""

    def __init__(self, node: Node, config: ConfigLoader):
        self._node = node
        self._config = config
        self._logger = node.get_logger()

        # Limiti
        self._max_vel = config.get('manipulation.trajectory.max_velocity', 1.0)
        self._max_acc = config.get('manipulation.trajectory.max_acceleration', 2.0)
        self._max_jerk = config.get('manipulation.trajectory.max_jerk', 10.0)

        if Ruckig is None:
            self._logger.warn(
                "Ruckig non disponibile, uso interpolazione lineare"
            )

    def plan(self, start: np.ndarray, goal: np.ndarray,
             start_vel: np.ndarray = None,
             rate_hz: float = 50.0) -> List[dict]:
        """Pianifica una traiettoria da start a goal.

        Args:
            start: configurazione iniziale (7 valori)
            goal: configurazione finale (7 valori)
            start_vel: velocita iniziali (default zero)
            rate_hz: frequenza di campionamento

        Returns:
            Lista di waypoint con 'positions', 'velocities', 'time_from_start'
        """
        n_joints = len(start)

        if start_vel is None:
            start_vel = np.zeros(n_joints)

        if Ruckig is not None:
            return self._plan_ruckig(start, goal, start_vel, n_joints, rate_hz)
        return self._plan_linear(start, goal, n_joints, rate_hz)

    def _plan_ruckig(self, start: np.ndarray, goal: np.ndarray,
                      start_vel: np.ndarray, n_joints: int,
                      rate_hz: float) -> List[dict]:
        """Pianifica con Ruckig (smooth, con limiti jerk)."""
        otg = Ruckig(n_joints, 1.0 / rate_hz)
        inp = InputParameter(n_joints)

        inp.current_position = start.tolist()
        inp.current_velocity = start_vel.tolist()
        inp.current_acceleration = [0.0] * n_joints

        inp.target_position = goal.tolist()
        inp.target_velocity = [0.0] * n_joints
        inp.target_acceleration = [0.0] * n_joints

        inp.max_velocity = [self._max_vel] * n_joints
        inp.max_acceleration = [self._max_acc] * n_joints
        inp.max_jerk = [self._max_jerk] * n_joints

        trajectory = RuckigTrajectory(n_joints)
        result = otg.calculate(inp, trajectory)

        if result < 0:
            self._logger.error(f"Ruckig planning fallito (errore {result})")
            return self._plan_linear(start, goal, n_joints, rate_hz)

        duration = trajectory.duration
        dt = 1.0 / rate_hz
        waypoints = []

        t = 0.0
        while t <= duration:
            new_pos = [0.0] * n_joints
            new_vel = [0.0] * n_joints
            new_acc = [0.0] * n_joints
            trajectory.at_time(t, new_pos, new_vel, new_acc)

            waypoints.append({
                'positions': list(new_pos),
                'velocities': list(new_vel),
                'accelerations': list(new_acc),
                'time_from_start': t,
            })
            t += dt

        # Aggiungi waypoint finale
        if waypoints and waypoints[-1]['time_from_start'] < duration:
            waypoints.append({
                'positions': goal.tolist(),
                'velocities': [0.0] * n_joints,
                'accelerations': [0.0] * n_joints,
                'time_from_start': duration,
            })

        self._logger.info(
            f"Traiettoria Ruckig: {len(waypoints)} waypoints, "
            f"durata {duration:.2f}s"
        )
        return waypoints

    def _plan_linear(self, start: np.ndarray, goal: np.ndarray,
                      n_joints: int, rate_hz: float) -> List[dict]:
        """Fallback: interpolazione lineare con smooth step."""
        # Stima durata basata sulla distanza massima dei giunti
        max_dist = np.max(np.abs(goal - start))
        duration = max(0.5, max_dist / self._max_vel)

        dt = 1.0 / rate_hz
        steps = int(duration * rate_hz)
        waypoints = []

        for i in range(steps + 1):
            t = i * dt
            alpha = t / duration if duration > 0 else 1.0
            alpha = min(alpha, 1.0)
            # Smooth step (Hermite)
            alpha_smooth = alpha * alpha * (3 - 2 * alpha)

            pos = start + alpha_smooth * (goal - start)
            vel = np.zeros(n_joints)
            if 0 < alpha < 1:
                vel_alpha = 6 * alpha * (1 - alpha) / duration
                vel = vel_alpha * (goal - start)

            waypoints.append({
                'positions': pos.tolist(),
                'velocities': vel.tolist(),
                'time_from_start': t,
            })

        self._logger.info(
            f"Traiettoria lineare: {len(waypoints)} waypoints, "
            f"durata {duration:.2f}s"
        )
        return waypoints

    def plan_via_waypoints(self, waypoints_positions: List[np.ndarray],
                            rate_hz: float = 50.0) -> List[dict]:
        """Pianifica una traiettoria passando per punti intermedi.

        Args:
            waypoints_positions: lista di configurazioni da attraversare
            rate_hz: frequenza

        Returns:
            Traiettoria completa
        """
        full_trajectory = []
        time_offset = 0.0

        for i in range(len(waypoints_positions) - 1):
            segment = self.plan(
                waypoints_positions[i],
                waypoints_positions[i + 1],
                rate_hz=rate_hz
            )

            for wp in segment:
                wp['time_from_start'] += time_offset
                full_trajectory.append(wp)

            if segment:
                time_offset = segment[-1]['time_from_start']

        return full_trajectory

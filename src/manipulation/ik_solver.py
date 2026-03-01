"""Inverse Kinematics con Pinocchio per braccio 7-DOF.

Usa il modello URDF del robot per risolvere l'IK.
"""

import os
from typing import Optional, Tuple

import numpy as np

from rclpy.node import Node

from src.utils.config_loader import ConfigLoader, JointLimitsLoader

try:
    import pinocchio as pin
except ImportError:
    pin = None


class IKSolver:
    """Risolve la cinematica inversa usando Pinocchio."""

    def __init__(self, node: Node, config: ConfigLoader):
        self._node = node
        self._config = config
        self._logger = node.get_logger()

        self._max_iters = config.get('manipulation.arm.ik_max_iterations', 100)
        self._tolerance = config.get('manipulation.arm.ik_tolerance', 0.001)

        # Limiti giunti
        self._joint_limits = JointLimitsLoader()

        # Modello Pinocchio
        self._model = None
        self._data = None
        self._left_arm_ids: list = []
        self._right_arm_ids: list = []
        self._left_ee_frame_id: int = 0
        self._right_ee_frame_id: int = 0

        self._init_model()

    def _init_model(self):
        """Carica il modello URDF in Pinocchio."""
        if pin is None:
            self._logger.error("Pinocchio non disponibile! IK disabilitato.")
            return

        urdf_path = os.path.join(
            os.path.dirname(__file__), '..', '..', 'config', 'urdf', 'agibot_x2.urdf'
        )

        if not os.path.exists(urdf_path):
            self._logger.warn(
                f"URDF non trovato a {urdf_path}. IK usera modalita placeholder."
            )
            return

        try:
            self._model = pin.buildModelFromUrdf(urdf_path)
            self._data = self._model.createData()

            # Identifica gli indici dei giunti per ogni braccio
            for i, name in enumerate(self._model.names):
                if 'left' in name.lower():
                    self._left_arm_ids.append(i)
                elif 'right' in name.lower():
                    self._right_arm_ids.append(i)

            # Frame ID end-effector
            for i in range(self._model.nframes):
                frame_name = self._model.frames[i].name
                if 'left' in frame_name.lower() and 'hand' in frame_name.lower():
                    self._left_ee_frame_id = i
                elif 'right' in frame_name.lower() and 'hand' in frame_name.lower():
                    self._right_ee_frame_id = i

            self._logger.info(
                f"Modello Pinocchio caricato: {self._model.nq} DOF, "
                f"{self._model.nframes} frames"
            )
        except Exception as e:
            self._logger.error(f"Errore caricamento URDF: {e}")

    def solve(self, target_position: np.ndarray,
              target_rotation: np.ndarray,
              q_init: np.ndarray = None,
              side: str = "left") -> Optional[np.ndarray]:
        """Risolve l'IK per una posizione e orientamento target.

        Args:
            target_position: posizione target [x, y, z] nel frame base
            target_rotation: matrice di rotazione 3x3 target
            q_init: configurazione iniziale (seed)
            side: "left" o "right"

        Returns:
            Configurazione giunti soluzione (7 valori) o None se fallito
        """
        if self._model is None:
            self._logger.warn("Modello non caricato, uso IK analitico semplificato")
            return self._solve_placeholder(target_position, side)

        ee_frame_id = (
            self._left_ee_frame_id if side == "left"
            else self._right_ee_frame_id
        )

        # Target come SE3
        target_se3 = pin.SE3(target_rotation, target_position)

        # Configurazione iniziale
        if q_init is None:
            q_init = pin.neutral(self._model)

        q = q_init.copy()
        eps = self._tolerance
        dt = 1e-1
        damp = 1e-12

        for i in range(self._max_iters):
            pin.forwardKinematics(self._model, self._data, q)
            pin.updateFramePlacements(self._model, self._data)

            current_se3 = self._data.oMf[ee_frame_id]
            err = pin.log6(current_se3.actInv(target_se3)).vector

            if np.linalg.norm(err) < eps:
                self._logger.debug(f"IK convergito in {i + 1} iterazioni")
                return self._extract_arm_joints(q, side)

            # Jacobiano
            J = pin.computeFrameJacobian(
                self._model, self._data, q, ee_frame_id,
                pin.ReferenceFrame.LOCAL_WORLD_ALIGNED
            )

            # Damped least squares (DLS)
            JtJ = J.T @ J + damp * np.eye(self._model.nv)
            dq = np.linalg.solve(JtJ, J.T @ err)

            q = pin.integrate(self._model, q, dq * dt)

            # Applica limiti giunti
            q = np.clip(q, self._model.lowerPositionLimit,
                        self._model.upperPositionLimit)

        self._logger.warn(f"IK non convergito dopo {self._max_iters} iterazioni")
        return None

    def _extract_arm_joints(self, q_full: np.ndarray, side: str) -> np.ndarray:
        """Estrae i 7 giunti del braccio dalla configurazione completa."""
        ids = self._left_arm_ids if side == "left" else self._right_arm_ids
        if ids:
            return q_full[ids]
        # Fallback: assume primi 7 giunti per braccio sinistro
        return q_full[:7]

    def _solve_placeholder(self, target_position: np.ndarray,
                            side: str) -> Optional[np.ndarray]:
        """IK placeholder quando Pinocchio/URDF non disponibile.

        Ritorna una configurazione ragionevole basata sulla posizione target.
        """
        q = np.zeros(7)
        x, y, z = target_position

        # Stime grossolane basate sulla geometria
        q[0] = np.arctan2(-z + 0.8, np.sqrt(x**2 + y**2))  # shoulder pitch
        q[1] = np.arctan2(y, x) if side == "left" else -np.arctan2(y, x)  # shoulder roll
        q[3] = -np.clip(np.sqrt(x**2 + y**2 + (z-0.8)**2) / 0.558 * 1.2, 0, 2.5)  # elbow

        return q

    def forward_kinematics(self, q: np.ndarray,
                            side: str = "left") -> Tuple[np.ndarray, np.ndarray]:
        """Cinematica diretta: da giunti a posizione end-effector.

        Returns:
            (posizione [x,y,z], rotazione 3x3)
        """
        if self._model is None:
            return np.zeros(3), np.eye(3)

        q_full = pin.neutral(self._model)
        ids = self._left_arm_ids if side == "left" else self._right_arm_ids
        for i, joint_id in enumerate(ids):
            if i < len(q):
                q_full[joint_id] = q[i]

        pin.forwardKinematics(self._model, self._data, q_full)
        pin.updateFramePlacements(self._model, self._data)

        ee_id = (
            self._left_ee_frame_id if side == "left"
            else self._right_ee_frame_id
        )
        oMf = self._data.oMf[ee_id]

        return oMf.translation.copy(), oMf.rotation.copy()

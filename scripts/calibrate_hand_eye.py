#!/usr/bin/env python3
"""Calibrazione hand-eye camera-braccio.

Procedura:
1. Muove il braccio in N posizioni diverse
2. Per ogni posizione, rileva un pattern di calibrazione (checkerboard)
3. Calcola la trasformazione camera -> end-effector usando OpenCV

Uso:
    python scripts/calibrate_hand_eye.py --pattern-size 7x5 --square-size 0.03
"""

import argparse
import json
import os
import time
from typing import List, Tuple

import numpy as np
import cv2

import rclpy
from rclpy.node import Node

from src.utils.config_loader import ConfigLoader
from src.perception.camera_manager import CameraManager
from src.manipulation.arm_controller import ArmController
from src.manipulation.predefined_poses import get_arm_pose
from src.robot_hal.mode_manager import ModeManager
from src.robot_hal.safety_monitor import SafetyMonitor


class HandEyeCalibrator(Node):
    """Nodo per calibrazione hand-eye."""

    def __init__(self, pattern_size: Tuple[int, int], square_size: float):
        super().__init__('hand_eye_calibrator')

        self._config = ConfigLoader()
        self._pattern_size = pattern_size
        self._square_size = square_size

        self._camera = CameraManager(self, self._config)
        self._safety = SafetyMonitor(self, self._config)
        self._mode = ModeManager(self)
        self._arm = ArmController(self, self._config, self._safety)

        # Dati calibrazione
        self._R_gripper2base: List[np.ndarray] = []
        self._t_gripper2base: List[np.ndarray] = []
        self._R_target2cam: List[np.ndarray] = []
        self._t_target2cam: List[np.ndarray] = []

    def collect_sample(self, joint_positions: np.ndarray, side: str = "left") -> bool:
        """Raccoglie un campione di calibrazione.

        Returns:
            True se il campione e valido
        """
        # Muovi braccio
        self._arm.move_to_positions(joint_positions, side, duration_s=3.0)
        time.sleep(1.0)  # Stabilizzazione

        # Cattura immagine
        rgb = self._camera.get_rgb()
        if rgb is None:
            self.get_logger().warn("Frame non disponibile")
            return False

        # Rileva pattern
        gray = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCorners(
            gray, self._pattern_size, None
        )

        if not ret:
            self.get_logger().warn("Pattern non rilevato in questa posizione")
            return False

        # Raffina corners
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)

        # Punti 3D del pattern
        objp = np.zeros((self._pattern_size[0] * self._pattern_size[1], 3), np.float32)
        objp[:, :2] = np.mgrid[
            0:self._pattern_size[0],
            0:self._pattern_size[1]
        ].T.reshape(-1, 2) * self._square_size

        # Calcola pose del pattern nella camera
        fx, fy, cx, cy = self._camera.intrinsics
        camera_matrix = np.array([
            [fx, 0, cx],
            [0, fy, cy],
            [0, 0, 1]
        ])
        dist_coeffs = np.zeros(5)

        ret, rvec, tvec = cv2.solvePnP(
            objp, corners, camera_matrix, dist_coeffs
        )

        if not ret:
            return False

        R_target2cam, _ = cv2.Rodrigues(rvec)
        t_target2cam = tvec.flatten()

        self._R_target2cam.append(R_target2cam)
        self._t_target2cam.append(t_target2cam)

        # FK del braccio -> trasformazione end-effector nel frame base
        # (Per ora uso placeholder, nella pratica viene dalla FK reale)
        from src.manipulation.ik_solver import IKSolver
        ik = IKSolver(self, self._config)
        ee_pos, ee_rot = ik.forward_kinematics(joint_positions, side)

        self._R_gripper2base.append(ee_rot)
        self._t_gripper2base.append(ee_pos)

        self.get_logger().info(
            f"Campione {len(self._R_target2cam)} raccolto"
        )
        return True

    def calibrate(self) -> Tuple[np.ndarray, np.ndarray]:
        """Esegue la calibrazione hand-eye.

        Returns:
            (R_cam2gripper, t_cam2gripper) trasformazione camera -> end-effector
        """
        if len(self._R_target2cam) < 3:
            self.get_logger().error("Servono almeno 3 campioni!")
            return np.eye(3), np.zeros(3)

        R_cam2gripper, t_cam2gripper = cv2.calibrateHandEye(
            self._R_gripper2base,
            self._t_gripper2base,
            self._R_target2cam,
            self._t_target2cam,
            method=cv2.CALIB_HAND_EYE_TSAI,
        )

        self.get_logger().info("Calibrazione completata!")
        self.get_logger().info(f"R_cam2gripper:\n{R_cam2gripper}")
        self.get_logger().info(f"t_cam2gripper: {t_cam2gripper.flatten()}")

        return R_cam2gripper, t_cam2gripper

    def save_calibration(self, R: np.ndarray, t: np.ndarray,
                          output_path: str):
        """Salva la calibrazione su file."""
        data = {
            'R_cam2gripper': R.tolist(),
            't_cam2gripper': t.flatten().tolist(),
        }
        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2)
        self.get_logger().info(f"Calibrazione salvata in: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Calibrazione hand-eye')
    parser.add_argument('--pattern-size', type=str, default='7x5',
                        help='Dimensione pattern checkerboard (colxrig)')
    parser.add_argument('--square-size', type=float, default=0.03,
                        help='Dimensione quadrato in metri')
    parser.add_argument('--num-samples', type=int, default=15,
                        help='Numero di campioni da raccogliere')
    parser.add_argument('--output', type=str,
                        default='config/hand_eye_calibration.json',
                        help='File di output')
    parser.add_argument('--side', type=str, default='left',
                        choices=['left', 'right'])

    args = parser.parse_args()

    pattern = tuple(int(x) for x in args.pattern_size.split('x'))

    rclpy.init()
    calibrator = HandEyeCalibrator(pattern, args.square_size)

    print(f"\nCalibrazione hand-eye con {args.num_samples} campioni")
    print(f"Pattern: {pattern}, quadrato: {args.square_size}m")
    print("\nPositionare il pattern checkerboard visibile alla camera.")
    print("Il braccio si muovera in posizioni diverse automaticamente.\n")

    # Genera posizioni di calibrazione
    # (nella pratica, usare posizioni che coprono bene lo spazio di lavoro)
    base_pose = get_arm_pose("pre_grasp", args.side)
    offsets = np.random.uniform(-0.3, 0.3, (args.num_samples, 7))
    offsets *= np.array([0.3, 0.2, 0.2, 0.3, 0.2, 0.2, 0.1])

    collected = 0
    for i in range(args.num_samples):
        pose = base_pose + offsets[i]
        print(f"\nCampione {i + 1}/{args.num_samples}...")

        if calibrator.collect_sample(pose, args.side):
            collected += 1

    print(f"\nRaccolti {collected} campioni validi")

    if collected >= 3:
        R, t = calibrator.calibrate()
        calibrator.save_calibration(R, t, args.output)

    calibrator.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()

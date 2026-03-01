#!/usr/bin/env python3
"""Test standalone per il modulo di manipolazione.

Testa IK solver, arm controller, gripper, traiettorie.

Uso:
    python scripts/test_manipulation.py [--test ik|arm|gripper|trajectory|all]
"""

import argparse
import time
import sys

import numpy as np

import rclpy
from rclpy.node import Node

from src.utils.config_loader import ConfigLoader
from src.manipulation.ik_solver import IKSolver
from src.manipulation.arm_controller import ArmController
from src.manipulation.gripper_controller import GripperController
from src.manipulation.trajectory_planner import TrajectoryPlanner
from src.manipulation.predefined_poses import get_arm_pose, LEFT_ARM_POSES
from src.robot_hal.mode_manager import ModeManager
from src.robot_hal.safety_monitor import SafetyMonitor


class ManipulationTester(Node):
    """Nodo per test manipolazione."""

    def __init__(self):
        super().__init__('manipulation_tester')
        self._config = ConfigLoader()
        self._safety = SafetyMonitor(self, self._config)
        self._mode = ModeManager(self)
        self._ik = IKSolver(self, self._config)
        self._arm = ArmController(self, self._config, self._safety)
        self._gripper = GripperController(self, self._config)
        self._traj = TrajectoryPlanner(self, self._config)

    def test_ik(self, side: str = "left") -> bool:
        """Testa l'IK solver."""
        print("\n=== Test IK Solver ===")

        # Target: posizione a 0.5m davanti, 0.5m altezza
        target_pos = np.array([0.4, 0.2, 0.6])
        target_rot = np.eye(3)
        target_rot[1, 1] = -1
        target_rot[2, 2] = -1  # Approccio dall'alto

        print(f"Target posizione: {target_pos}")
        q = self._ik.solve(target_pos, target_rot, side=side)

        if q is not None:
            print(f"OK: Soluzione IK trovata")
            print(f"  Giunti (rad): {q.round(3)}")
            print(f"  Giunti (deg): {np.degrees(q).round(1)}")

            # Verifica con FK
            pos_fk, rot_fk = self._ik.forward_kinematics(q, side)
            error = np.linalg.norm(pos_fk - target_pos)
            print(f"  Errore FK: {error * 1000:.2f} mm")
            return error < 0.01  # 10mm
        else:
            print("FAIL: IK non ha trovato soluzione")
            return False

    def test_arm(self, side: str = "left") -> bool:
        """Testa il controller del braccio."""
        print("\n=== Test Arm Controller ===")

        # Attendi stato giunti
        for i in range(30):
            rclpy.spin_once(self, timeout_sec=0.1)

        positions = self._arm.get_current_positions(side)
        if positions is None:
            print("FAIL: Posizioni correnti non disponibili")
            return False

        print(f"Posizioni correnti (deg): {np.degrees(positions).round(1)}")

        # Muovi a home
        print("Muovo a posizione home...")
        home = get_arm_pose("home", side)
        ok = self._arm.move_to_positions(home, side, duration_s=3.0)

        if ok:
            print("OK: Movimento a home completato")
        else:
            print("FAIL: Movimento fallito")

        return ok

    def test_gripper(self, side: str = "left") -> bool:
        """Testa il gripper."""
        print("\n=== Test Gripper ===")

        print("Apertura gripper...")
        self._gripper.open(side)
        time.sleep(1.0)

        print("Chiusura gripper (effort basso)...")
        self._gripper.close_for_pod(side)
        time.sleep(1.0)

        print("Apertura gripper...")
        self._gripper.open(side)
        time.sleep(0.5)

        print("Chiusura gripper (effort alto)...")
        self._gripper.close_for_box(side)
        time.sleep(1.0)

        print("Apertura finale...")
        self._gripper.open(side)

        print("OK: Sequenza gripper completata")
        return True

    def test_trajectory(self, side: str = "left") -> bool:
        """Testa il planner di traiettorie."""
        print("\n=== Test Trajectory Planner ===")

        start = get_arm_pose("home", side)
        goal = get_arm_pose("pre_grasp", side)

        print(f"Pianificazione traiettoria home -> pre_grasp...")
        trajectory = self._traj.plan(start, goal)

        print(f"OK: {len(trajectory)} waypoints generati")
        if trajectory:
            duration = trajectory[-1]['time_from_start']
            print(f"  Durata: {duration:.2f}s")
            print(f"  Primo waypoint: {np.round(trajectory[0]['positions'], 3)}")
            print(f"  Ultimo waypoint: {np.round(trajectory[-1]['positions'], 3)}")

        # Verifica sicurezza traiettoria
        safe = self._safety.check_trajectory(trajectory, f"{side}_arm")
        print(f"  Sicurezza: {'OK' if safe else 'FAIL'}")

        return len(trajectory) > 0 and safe


def main():
    parser = argparse.ArgumentParser(description='Test manipolazione')
    parser.add_argument('--test', type=str, default='all',
                        choices=['ik', 'arm', 'gripper', 'trajectory', 'all'])
    parser.add_argument('--side', type=str, default='left',
                        choices=['left', 'right'])

    args = parser.parse_args()

    rclpy.init()
    tester = ManipulationTester()

    results = {}

    if args.test in ('ik', 'all'):
        results['ik'] = tester.test_ik(args.side)

    if args.test in ('trajectory', 'all'):
        results['trajectory'] = tester.test_trajectory(args.side)

    if args.test in ('arm', 'all'):
        results['arm'] = tester.test_arm(args.side)

    if args.test in ('gripper', 'all'):
        results['gripper'] = tester.test_gripper(args.side)

    print("\n=== Risultati ===")
    all_ok = True
    for name, result in results.items():
        status = "PASS" if result else "FAIL"
        print(f"  {name}: {status}")
        if not result:
            all_ok = False

    tester.destroy_node()
    rclpy.shutdown()

    sys.exit(0 if all_ok else 1)


if __name__ == '__main__':
    main()

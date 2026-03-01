#!/usr/bin/env python3
"""Test standalone per il modulo di navigazione.

Testa locomozione, approccio tavoli, position tracking.

Uso:
    python scripts/test_navigation.py [--test walk|approach|tracking|all]
"""

import argparse
import time
import sys

import numpy as np

import rclpy
from rclpy.node import Node

from src.utils.config_loader import ConfigLoader
from src.navigation.locomotion_controller import LocomotionController
from src.navigation.approach_planner import ApproachPlanner, ApproachPhase
from src.navigation.position_tracker import PositionTracker
from src.perception.table_detector import TableDetector, TableInfo
from src.robot_hal.mode_manager import ModeManager


class NavigationTester(Node):
    """Nodo per test navigazione."""

    def __init__(self):
        super().__init__('navigation_tester')
        self._config = ConfigLoader()
        self._mode = ModeManager(self)
        self._locomotion = LocomotionController(self, self._config, self._mode)
        self._approach = ApproachPlanner(self, self._config, self._locomotion)
        self._tracker = PositionTracker(self, self._config)

    def test_walk(self) -> bool:
        """Testa la camminata base."""
        print("\n=== Test Camminata ===")

        print("Passo a modalita LOCOMOTION...")
        if not self._mode.ensure_locomotion():
            print("FAIL: Impossibile passare a LOCOMOTION")
            return False

        print("Cammino in avanti per 2 secondi a 0.2 m/s...")
        self._locomotion.send_velocity(linear_x=0.2)
        time.sleep(2.0)

        print("Stop...")
        self._locomotion.stop()
        time.sleep(0.5)

        print("Rotazione 90 gradi...")
        self._locomotion.rotate_angle(np.pi / 2)

        print("OK: Test camminata completato")
        return True

    def test_approach(self) -> bool:
        """Testa l'approccio a un tavolo simulato."""
        print("\n=== Test Approccio Tavolo ===")

        # Crea tavolo simulato
        table = TableInfo(
            name="test_table",
            center=np.array([1.5, 0.0, 0.75]),
            normal=np.array([0.0, 0.0, 1.0]),
            corners=np.zeros((4, 3)),
            height=0.75,
            size=(0.7, 0.7),
        )

        print(f"Tavolo target: ({table.center[0]:.2f}, {table.center[1]:.2f})")

        self._approach.start_approach(table)

        # Simula aggiornamenti
        for i in range(100):
            robot_pos = self._tracker.position
            robot_yaw = self._tracker.yaw

            phase = self._approach.update(robot_pos, robot_yaw)
            print(f"  Step {i}: phase={phase.value}, pos=({robot_pos[0]:.2f}, {robot_pos[1]:.2f})")

            if phase == ApproachPhase.DONE:
                print("OK: Approccio completato!")
                return True

            # Simula dead-reckoning
            self._tracker.update_from_velocity(0.2, 0.0, 0.0)
            time.sleep(0.1)

        print("FAIL: Approccio non completato in tempo")
        return False

    def test_tracking(self) -> bool:
        """Testa il position tracker."""
        print("\n=== Test Position Tracker ===")

        self._tracker.reset()
        print(f"Posizione iniziale: {self._tracker.position}")

        # Simula camminata
        for i in range(10):
            self._tracker.update_from_velocity(0.3, 0.0, 0.0)
            time.sleep(0.1)

        pos = self._tracker.position
        print(f"Dopo 1s a 0.3m/s: ({pos[0]:.3f}, {pos[1]:.3f})")
        expected_x = 0.3  # circa 0.3m
        error = abs(pos[0] - expected_x)
        print(f"Errore atteso: {error:.3f}m")

        ok = error < 0.1  # Tolleranza 10cm
        print(f"{'OK' if ok else 'FAIL'}: Dead-reckoning {'corretto' if ok else 'impreciso'}")
        return ok


def main():
    parser = argparse.ArgumentParser(description='Test navigazione')
    parser.add_argument('--test', type=str, default='all',
                        choices=['walk', 'approach', 'tracking', 'all'])

    args = parser.parse_args()

    rclpy.init()
    tester = NavigationTester()

    results = {}

    if args.test in ('tracking', 'all'):
        results['tracking'] = tester.test_tracking()

    if args.test in ('walk', 'all'):
        results['walk'] = tester.test_walk()

    if args.test in ('approach', 'all'):
        results['approach'] = tester.test_approach()

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

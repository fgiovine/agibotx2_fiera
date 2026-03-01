#!/usr/bin/env python3
"""Test ciclo completo della demo.

Testa l'intero ciclo: SCAN -> IDLE -> DETECT -> PICK -> PLACE -> ...

Uso:
    python scripts/test_full_cycle.py [--single-pod | --full-box]
"""

import argparse
import time
import sys

import rclpy
from rclpy.node import Node

from src.utils.config_loader import ConfigLoader
from src.state_machine import StateMachine


class FullCycleTester(Node):
    """Testa il ciclo completo della demo."""

    def __init__(self):
        super().__init__('full_cycle_tester')
        self._config = ConfigLoader()
        self._fsm = StateMachine(self, self._config)

    def test_single_pod(self) -> bool:
        """Testa il ciclo per una singola cialda (senza camminata)."""
        print("\n=== Test Singola Cialda ===")
        print("Ciclo: DETECT -> PICK -> PLACE")
        print("(Posiziona una cialda sul tavolo e una scatola aperta)\n")

        if not self._fsm.initialize():
            print("FAIL: Inizializzazione fallita")
            return False

        # Parti da SCAN_TABLES
        self._fsm.start("SCAN_TABLES")

        max_steps = 200
        step = 0
        target_states = {"IDLE"}  # Stato finale atteso dopo un ciclo

        while step < max_steps and self._fsm.is_running:
            self._fsm.step()
            rclpy.spin_once(self, timeout_sec=0.05)

            state = self._fsm.current_state_name
            print(f"  Step {step}: {state}")

            # Se torniamo a IDLE dopo aver fatto PLACE, siamo OK
            if state == "IDLE" and step > 10:
                print("\nOK: Ciclo singola cialda completato!")
                self._fsm.stop()
                return True

            if state == "ERROR":
                error = self._fsm.context.state_data.get("error", "?")
                print(f"\nERRORE: {error}")

            step += 1
            time.sleep(0.1)

        print(f"\nFAIL: Ciclo non completato in {max_steps} passi")
        self._fsm.stop()
        return False

    def test_full_box(self) -> bool:
        """Testa il ciclo completo fino a riempire una scatola."""
        print("\n=== Test Scatola Completa ===")
        print("(Posiziona cialde continuamente sul tavolo)\n")

        if not self._fsm.initialize():
            print("FAIL: Inizializzazione fallita")
            return False

        self._fsm.start("SCAN_TABLES")

        max_steps = 2000
        step = 0

        while step < max_steps and self._fsm.is_running:
            self._fsm.step()
            rclpy.spin_once(self, timeout_sec=0.05)

            state = self._fsm.current_state_name
            pods = self._fsm.context.box_tracker.pod_count

            if step % 10 == 0:
                print(f"  Step {step}: {state} (cialde: {pods}/20)")

            # Se arriviamo a PLACE_EMPTY_BOX, la scatola e stata cambiata
            if state == "IDLE" and pods == 0 and step > 100:
                print("\nOK: Ciclo scatola completa terminato!")
                self._fsm.stop()
                return True

            step += 1
            time.sleep(0.1)

        self._fsm.stop()
        return False


def main():
    parser = argparse.ArgumentParser(description='Test ciclo completo')
    parser.add_argument('--single-pod', action='store_true',
                        help='Testa ciclo singola cialda')
    parser.add_argument('--full-box', action='store_true',
                        help='Testa ciclo scatola completa')

    args = parser.parse_args()

    rclpy.init()
    tester = FullCycleTester()

    if args.full_box:
        result = tester.test_full_box()
    else:
        result = tester.test_single_pod()

    tester.destroy_node()
    rclpy.shutdown()

    sys.exit(0 if result else 1)


if __name__ == '__main__':
    main()

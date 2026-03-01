#!/usr/bin/env python3
"""Registra pose del braccio per la demo.

Uso interattivo: muovi il braccio manualmente (gravity comp o teach mode),
poi premi ENTER per salvare la posizione corrente.

Uso:
    python scripts/record_poses.py --side left --output config/recorded_poses.json
"""

import argparse
import json
import os
import time

import numpy as np

import rclpy
from rclpy.node import Node

from src.utils.config_loader import ConfigLoader
from src.manipulation.arm_controller import ArmController
from src.robot_hal.mode_manager import ModeManager
from src.robot_hal.safety_monitor import SafetyMonitor


class PoseRecorder(Node):
    """Registra pose del braccio in modo interattivo."""

    def __init__(self):
        super().__init__('pose_recorder')
        self._config = ConfigLoader()
        self._safety = SafetyMonitor(self, self._config)
        self._mode = ModeManager(self)
        self._arm = ArmController(self, self._config, self._safety)
        self._recorded_poses: dict = {}

    def record_pose(self, name: str, side: str = "left") -> bool:
        """Registra la posizione corrente del braccio.

        Args:
            name: nome della posa
            side: braccio

        Returns:
            True se registrata
        """
        rclpy.spin_once(self, timeout_sec=0.5)

        positions = self._arm.get_current_positions(side)
        if positions is None:
            self.get_logger().error("Posizioni non disponibili")
            return False

        self._recorded_poses[name] = {
            'positions': positions.tolist(),
            'side': side,
            'positions_deg': np.degrees(positions).tolist(),
        }

        self.get_logger().info(
            f"Posa '{name}' registrata: "
            f"{np.degrees(positions).round(1).tolist()} deg"
        )
        return True

    def save(self, output_path: str):
        """Salva le pose su file JSON."""
        with open(output_path, 'w') as f:
            json.dump(self._recorded_poses, f, indent=2)
        self.get_logger().info(f"Pose salvate in: {output_path}")

    def load(self, input_path: str):
        """Carica pose da file JSON."""
        if os.path.exists(input_path):
            with open(input_path, 'r') as f:
                self._recorded_poses = json.load(f)
            self.get_logger().info(
                f"Caricate {len(self._recorded_poses)} pose da {input_path}"
            )

    def replay_pose(self, name: str, side: str = "left",
                     duration_s: float = 3.0) -> bool:
        """Riproduci una posa registrata."""
        if name not in self._recorded_poses:
            self.get_logger().error(f"Posa '{name}' non trovata")
            return False

        positions = np.array(self._recorded_poses[name]['positions'])
        self.get_logger().info(f"Replay posa '{name}'...")
        return self._arm.move_to_positions(positions, side, duration_s)

    def generate_python_code(self) -> str:
        """Genera codice Python per predefined_poses.py."""
        lines = ["# Pose registrate - generato da record_poses.py\n"]
        lines.append("import numpy as np\n\n")

        for name, data in self._recorded_poses.items():
            positions = data['positions']
            side = data.get('side', 'left')
            lines.append(
                f"# {name} ({side})\n"
                f'"{name}": np.array({positions}),\n\n'
            )

        return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description='Registra pose braccio')
    parser.add_argument('--side', type=str, default='left',
                        choices=['left', 'right'])
    parser.add_argument('--output', type=str,
                        default='config/recorded_poses.json')
    parser.add_argument('--load', type=str, default=None,
                        help='Carica pose esistenti')
    parser.add_argument('--replay', type=str, default=None,
                        help='Riproduci una posa salvata')

    args = parser.parse_args()

    rclpy.init()
    recorder = PoseRecorder()

    if args.load:
        recorder.load(args.load)

    if args.replay:
        recorder.load(args.output)
        recorder.replay_pose(args.replay, args.side)
        recorder.destroy_node()
        rclpy.shutdown()
        return

    # Modalita interattiva
    pose_names = [
        "home", "pre_grasp", "pre_place_box", "box_grasp",
        "box_carry", "lid_place", "walk_safe", "present"
    ]

    print("\n=== Registrazione Pose Braccio ===")
    print(f"Braccio: {args.side}")
    print(f"Output: {args.output}")
    print("\nPose da registrare:")
    for i, name in enumerate(pose_names):
        print(f"  {i + 1}. {name}")
    print("\nIstruzioni:")
    print("- Muovi il braccio nella posizione desiderata")
    print("- Premi ENTER per registrare")
    print("- Digita 'skip' per saltare")
    print("- Digita 'quit' per salvare e uscire\n")

    for name in pose_names:
        response = input(f"Registrare '{name}'? [ENTER/skip/quit]: ").strip()

        if response.lower() == 'quit':
            break
        if response.lower() == 'skip':
            continue

        recorder.record_pose(name, args.side)

    recorder.save(args.output)
    print("\n" + recorder.generate_python_code())

    recorder.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
